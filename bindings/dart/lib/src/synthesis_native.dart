// The only synthesis code that borrows native pointers. Blocking operations
// execute on the adapter isolate; cancellation is a short direct ABI call.
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'package:ffi/ffi.dart';
import 'bindings.g.dart' as native;

class NativeFailure implements Exception {
  final String code;
  final String message;
  const NativeFailure(this.code, this.message);
  @override
  String toString() => 'MLSR $code: $message';
}

String _decode(native.MlsrString value) =>
    value.len == 0 ? '' : utf8.decode(value.data.asTypedList(value.len));

int _check(int status, Pointer<Uint64> error) {
  if (status >= 0) return status;
  try {
    return using((arena) {
      final view = arena<native.MlsrErrorInfo>();
      if (error.value != 0 &&
          native.mlsr_error_read(
                error.value,
                view,
                sizeOf<native.MlsrErrorInfo>(),
              ) ==
              0) {
        throw NativeFailure(_decode(view.ref.code), _decode(view.ref.message));
      }
      throw const NativeFailure(
        'native_error',
        'Native call failed without a diagnostic',
      );
    });
  } finally {
    native.mlsr_error_release(error.value);
  }
}

native.MlsrString _string(Arena arena, String text) {
  final bytes = utf8.encode(text);
  final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
  data.asTypedList(bytes.length).setAll(0, bytes);
  final value = arena<native.MlsrString>();
  value.ref
    ..data = data
    ..len = bytes.length;
  return value.ref;
}

void checkSynthesisApi() {
  if (native.mlsr_abi_version() != 1 || native.mlsr_features() & 1 == 0) {
    throw const NativeFailure(
      'unsupported_abi',
      'Native synthesis API is unavailable',
    );
  }
}

void cancelNative(int job) => using((arena) {
  final error = arena<Uint64>();
  _check(native.mlsr_job_cancel(job, error), error);
});

// Also usable from a fresh cleanup isolate if the original adapter exits.
void releaseNative(String kind, int handle) {
  final status = kind == 'engine'
      ? native.mlsr_engine_release(handle)
      : native.mlsr_job_release(handle);
  if (status != 0) {
    throw NativeFailure('release_failed', '$kind release failed');
  }
}

Object? nativeOperation(String operation, Map<String, Object?> args) {
  return using((arena) {
    final error = arena<Uint64>();
    final output = arena<Uint64>();
    switch (operation) {
      case 'open':
        checkSynthesisApi();
        final options = arena<native.MlsrEngineOptions>();
        native.mlsr_engine_options_default(options);
        options.ref
          ..compute_policy = args['policy'] as int
          ..reuse_output_buffers = args['reuse'] as int
          ..execution_variant = args['variant'] as int;
        _check(
          native.mlsr_engine_open(
            _string(arena, args['manifest'] as String),
            _string(arena, args['bundle'] as String),
            options,
            output,
            error,
          ),
          error,
        );
        return output.value;
      case 'start':
        final options = arena<native.MlsrJobOptions>();
        native.mlsr_job_options_default(options);
        options.ref
          ..steps = args['steps'] as int
          ..speed = args['speed'] as double
          ..seed = args['seed'] as int
          ..queue_frames = args['queue'] as int
          ..chunk_frames = args['chunk'] as int
          ..max_segment_frames = args['segment'] as int;
        _check(
          native.mlsr_job_start(
            args['engine'] as int,
            _string(arena, args['text'] as String),
            _string(arena, args['language'] as String),
            _string(arena, args['voice'] as String),
            options,
            output,
            error,
          ),
          error,
        );
        return [output.value, native.mlsr_job_id(output.value)];
      case 'poll':
        final status = _check(
          native.mlsr_job_poll(
            args['job'] as int,
            args['timeout'] as int,
            output,
            error,
          ),
          error,
        );
        if (status != 0) return {'status': status};
        final handle = output.value;
        try {
          final info = arena<native.MlsrEventInfo>();
          _check(
            native.mlsr_event_read(
              handle,
              info,
              sizeOf<native.MlsrEventInfo>(),
              error,
            ),
            error,
          );
          final value = info.ref;
          if (value.kind < 1 || value.kind > 5 || value.sample_count > 65536) {
            throw const NativeFailure(
              'invalid_event',
              'Unexpected native event layout',
            );
          }
          return <String, Object?>{
            'status': 0,
            'kind': value.kind,
            'job': value.job_id,
            'segment': value.segment_id,
            'textStart': value.text_start,
            'textEnd': value.text_end,
            'startFrame': value.start_frame,
            'frames': value.frames,
            'predictedFrames': value.predicted_frames,
            'rate': value.sample_rate,
            'channels': value.channels,
            'segments': value.completed_segments,
            'samples': value.sample_count == 0
                ? null
                : TransferableTypedData.fromList([
                    value.samples.asTypedList(value.sample_count),
                  ]),
            'errorCode': _decode(value.error.code),
            'errorMessage': _decode(value.error.message),
          };
        } finally {
          native.mlsr_event_release(handle);
        }
      case 'wait':
        return _check(
              native.mlsr_job_wait(
                args['job'] as int,
                args['timeout'] as int,
                error,
              ),
              error,
            ) ==
            0;
      case 'releaseJob':
        releaseNative('job', args['job'] as int);
        return null;
      case 'releaseEngine':
        releaseNative('engine', args['engine'] as int);
        return null;
      default:
        throw NativeFailure('invalid_operation', operation);
    }
  });
}

void adapterMain(SendPort responses) {
  final commands = ReceivePort();
  responses.send(commands.sendPort);
  commands.listen((dynamic message) {
    final values = message as List;
    final id = values[0] as int;
    try {
      final result = nativeOperation(
        values[1] as String,
        Map<String, Object?>.from(values[2] as Map),
      );
      responses.send([id, result, null]);
    } on NativeFailure catch (error) {
      responses.send([
        id,
        null,
        [error.code, error.message],
      ]);
    } catch (error) {
      responses.send([
        id,
        null,
        ['adapter_error', error.toString()],
      ]);
    }
  });
}
