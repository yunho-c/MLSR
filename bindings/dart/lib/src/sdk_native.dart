part of 'synthesis_native.dart';

void checkDeveloperApi() {
  checkSynthesisApi();
  if (native.mlsr_features() & 8 == 0) {
    throw const NativeFailure(
      'unsupported_abi',
      'Developer SDK schema v1 is unavailable',
    );
  }
}

// The closure captures only sendable arguments, never a facade/finalizer/port.
// Long operations use their own worker, so they cannot starve state observation.
Future<Object?> sdkWorkerOperation(
  String operation,
  Map<String, Object?> args,
) => Isolate.run(() => sdkNativeOperation(operation, args));

Map<String, Object?> _sdkJson(int handle) {
  try {
    return using((arena) {
      final error = arena<Uint64>(), view = arena<native.MlsrString>();
      _check(native.mlsr_json_read(handle, view, error), error);
      final envelope = Map<String, Object?>.from(
        jsonDecode(_decode(view.ref)) as Map,
      );
      if (envelope['schema_version'] != 1 || envelope['type'] is! String) {
        throw const NativeFailure(
          'unsupported_schema',
          'Unexpected native metadata envelope',
        );
      }
      return Map<String, Object?>.from(envelope['data'] as Map);
    });
  } finally {
    native.mlsr_sdk_release(handle);
  }
}

int _sdkCheck(int status, Pointer<Uint64> error) {
  if (status >= 0) return status;
  try {
    return using((arena) {
      final view = arena<native.MlsrErrorInfo>();
      if (error.value == 0 ||
          native.mlsr_error_read(
                error.value,
                view,
                sizeOf<native.MlsrErrorInfo>(),
              ) !=
              0) {
        throw const NativeFailure(
          'native_error',
          'Native call failed without a diagnostic',
        );
      }
      final code = _decode(view.ref.code), message = _decode(view.ref.message);
      final output = arena<Uint64>(), nested = arena<Uint64>();
      Map<String, Object?>? failure;
      try {
        if (native.mlsr_error_details(error.value, output, nested) == 0) {
          final data = _sdkJson(output.value);
          if (data['failure'] != null) {
            failure = Map<String, Object?>.from(data['failure'] as Map);
          }
        }
      } finally {
        native.mlsr_error_release(nested.value);
      }
      throw NativeFailure(
        code,
        message,
        category: failure?['category'] as String?,
        stage: failure?['stage'] as String?,
        backend: failure?['backend'] as String?,
        requestId: failure?['request_id'] as int?,
      );
    });
  } finally {
    native.mlsr_error_release(error.value);
  }
}

Pointer<native.MlsrRequest> _sdkRequest(
  Arena arena,
  Map<String, Object?> args,
) {
  final r = arena<native.MlsrRequest>();
  native.mlsr_request_default(r);
  r.ref
    ..text = _string(arena, args['text'] as String)
    ..locale = _string(arena, args['locale'] as String)
    ..voice = _string(arena, args['voice'] as String)
    ..locale_fallback = args['fallback'] as int
    ..speaking_rate = args['speakingRate'] as double
    ..supertonic_steps = args['steps'] as int
    ..seed = args['seed'] as int;
  return r;
}

Pointer<native.MlsrSpeechOptions> _sdkSpeechOptions(
  Arena arena,
  Map<String, Object?> args,
) {
  final s = arena<native.MlsrSpeechOptions>();
  native.mlsr_speech_options_default(s);
  s.ref
    ..mode = args['mode'] as int
    ..playback_rate = args['playbackRate'] as double
    ..volume = args['volume'] as double
    ..paused = (args['paused'] as bool) ? 1 : 0
    ..retain_audio = (args['retainAudio'] as bool) ? 1 : 0
    ..lookahead_frames = args['lookaheadFrames'] as int;
  return s;
}

Map<String, Object?> _sdkReadPcm(int handle) {
  try {
    return using((arena) {
      final view = arena<native.MlsrPcmView>(),
          error = arena<Uint64>(),
          metadata = arena<Uint64>();
      _sdkCheck(
        native.mlsr_audio_read(
          handle,
          view,
          sizeOf<native.MlsrPcmView>(),
          error,
        ),
        error,
      );
      final value = view.ref;
      if (value.sample_count > 1024 * 1024 * 1024 ~/ 4 ||
          value.frames * value.channels != value.sample_count) {
        throw const NativeFailure(
          'invalid_audio',
          'Unexpected native PCM layout',
        );
      }
      _sdkCheck(native.mlsr_sdk_snapshot(handle, metadata, error), error);
      return {
        ..._sdkJson(metadata.value),
        'samples': TransferableTypedData.fromList([
          value.samples.asTypedList(value.sample_count),
        ]),
      };
    });
  } finally {
    native.mlsr_sdk_release(handle);
  }
}

Object? sdkNativeOperation(String operation, Map<String, Object?> args) {
  checkDeveloperApi();
  return using((arena) {
    final error = arena<Uint64>(), output = arena<Uint64>();
    final h = args['handle'] as int? ?? 0;
    switch (operation) {
      case 'sdkOpen':
      case 'sdkFromModel':
        final engine = arena<native.MlsrEngineOptions>();
        native.mlsr_engine_options_default(engine);
        engine.ref
          ..compute_policy = args['policy'] as int
          ..reuse_output_buffers = args['reuse'] as int
          ..execution_variant = args['variant'] as int;
        final s = arena<native.MlsrTtsOptions>();
        native.mlsr_tts_options_default(s);
        s.ref
          ..output_mode = args['outputMode'] as int
          ..max_requests = args['maxRequests'] as int
          ..max_text_bytes = args['maxTextBytes'] as int
          ..queue_frames = args['queueFrames'] as int
          ..chunk_frames = args['chunkFrames'] as int
          ..max_segment_frames = args['maxSegmentFrames'] as int
          ..max_retained_stores = args['maxRetainedStores'] as int
          ..max_retained_bytes = args['maxRetainedBytes'] as int;
        _sdkCheck(
          operation == 'sdkFromModel'
              ? native.mlsr_tts_from_model(
                  args['model'] as int,
                  engine,
                  s,
                  output,
                  error,
                )
              : native.mlsr_tts_open(
                  _string(arena, args['manifest'] as String),
                  _string(arena, args['bundle'] as String),
                  engine,
                  s,
                  output,
                  error,
                ),
          error,
        );
        return output.value;
      case 'sdkVoices':
        _sdkCheck(native.mlsr_voice_catalog(h, output, error), error);
        return _sdkJson(output.value);
      case 'sdkResolve':
        _sdkCheck(
          native.mlsr_voice_resolve(
            h,
            _string(arena, args['locale'] as String),
            _string(arena, args['voice'] as String),
            args['fallback'] as int,
            output,
            error,
          ),
          error,
        );
        return _sdkJson(output.value);
      case 'sdkController':
        _sdkCheck(
          native.mlsr_tts_controller(
            h,
            (args['independent'] as bool) ? 1 : 0,
            output,
            error,
          ),
          error,
        );
        return output.value;
      case 'sdkSynthesize':
        _sdkCheck(
          native.mlsr_tts_synthesize(
            h,
            _sdkRequest(arena, args),
            output,
            error,
          ),
          error,
        );
        return [output.value, native.mlsr_sdk_id(output.value)];
      case 'sdkSpeak':
        _sdkCheck(
          native.mlsr_speak(
            h,
            _sdkRequest(arena, args),
            _sdkSpeechOptions(arena, args),
            output,
            error,
          ),
          error,
        );
        return [output.value, native.mlsr_sdk_id(output.value)];
      case 'sdkSnapshot':
      case 'sdkRequest':
      case 'sdkPlan':
        final status = _sdkCheck(switch (operation) {
          'sdkSnapshot' => native.mlsr_sdk_snapshot(h, output, error),
          'sdkRequest' => native.mlsr_sdk_request(h, output, error),
          _ => native.mlsr_text_plan(h, output, error),
        }, error);
        return status == 0 ? _sdkJson(output.value) : null;
      case 'sdkWait':
      case 'sdkStatePoll':
        final status = _sdkCheck(
          operation == 'sdkWait'
              ? native.mlsr_sdk_wait(h, args['timeout'] as int, output, error)
              : native.mlsr_state_poll(
                  h,
                  args['timeout'] as int,
                  output,
                  error,
                ),
          error,
        );
        return {
          'status': status,
          'data': status == 0 ? _sdkJson(output.value) : null,
        };
      case 'sdkStates':
        _sdkCheck(native.mlsr_sdk_states(h, output, error), error);
        return output.value;
      case 'sdkAudio':
        _sdkCheck(native.mlsr_synthesis_audio(h, output, error), error);
        return output.value;
      case 'sdkAudioPoll':
        final completion = arena<Uint64>();
        final status = _sdkCheck(
          native.mlsr_audio_stream_poll(
            h,
            args['timeout'] as int,
            output,
            completion,
            error,
          ),
          error,
        );
        return {
          'status': status,
          'audio': output.value == 0 ? null : _sdkReadPcm(output.value),
          'completion': completion.value == 0
              ? null
              : _sdkJson(completion.value),
        };
      case 'sdkCollect':
        _sdkCheck(
          native.mlsr_tts_collect(
            h,
            args['job'] as int,
            args['maxBytes'] as int,
            output,
            error,
          ),
          error,
        );
        return _sdkReadPcm(output.value);
      case 'sdkWriteJob':
      case 'sdkWriteSpeech':
        final path = _string(arena, args['path'] as String);
        _sdkCheck(
          operation == 'sdkWriteJob'
              ? native.mlsr_tts_write_job(
                  h,
                  args['source'] as int,
                  path,
                  args['maxFrames'] as int,
                  output,
                  error,
                )
              : native.mlsr_tts_write_speech(
                  h,
                  args['source'] as int,
                  path,
                  args['maxFrames'] as int,
                  output,
                  error,
                ),
          error,
        );
        return _sdkJson(output.value);
      case 'sdkWritePcm':
        final samples = (args['samples'] as TransferableTypedData)
            .materialize()
            .asFloat32List();
        if (samples.length > 1024 * 1024 * 1024 ~/ 4) {
          throw const NativeFailure('render_limit', 'PCM exceeds 1 GiB');
        }
        final channels = args['channels'] as int;
        if (channels < 1 || samples.length % channels != 0) {
          throw const NativeFailure('invalid_audio', 'Incomplete PCM frames');
        }
        final data = arena<Float>(samples.isEmpty ? 1 : samples.length);
        data.asTypedList(samples.length).setAll(0, samples);
        final view = arena<native.MlsrPcmView>();
        view.ref
          ..sample_rate = args['sampleRate'] as int
          ..channels = channels
          ..frames = samples.length ~/ channels
          ..samples = data
          ..sample_count = samples.length;
        _sdkCheck(
          native.mlsr_pcm_write_wav(
            view,
            _string(arena, args['path'] as String),
            output,
            error,
          ),
          error,
        );
        return _sdkJson(output.value);
      case 'sdkReplay':
        _sdkCheck(
          native.mlsr_speech_replay(
            h,
            args['frame'] as int,
            _sdkSpeechOptions(arena, args),
            output,
            error,
          ),
          error,
        );
        return [output.value, native.mlsr_sdk_id(output.value)];
      case 'sdkNavigate':
        final speech = arena<Uint64>();
        _sdkCheck(
          native.mlsr_speech_navigate(
            h,
            args['target'] as int,
            speech,
            output,
            error,
          ),
          error,
        );
        try {
          return {
            ..._sdkJson(output.value),
            'handle': speech.value,
            'id': native.mlsr_sdk_id(speech.value),
          };
        } catch (_) {
          native.mlsr_sdk_release(speech.value);
          rethrow;
        }
      case 'sdkCommand':
        _sdkCheck(
          native.mlsr_speech_command(
            h,
            args['command'] as int,
            args['value'] as double,
            error,
          ),
          error,
        );
        return null;
      case 'sdkSeek':
        _sdkCheck(
          native.mlsr_speech_seek(h, args['frame'] as int, error),
          error,
        );
        return null;
      case 'sdkStart':
        _sdkCheck(native.mlsr_synthesis_start(h, error), error);
        return null;
      case 'sdkCancelSynthesis':
        _sdkCheck(
          native.mlsr_synthesis_cancel(h, args['reason'] as int? ?? 0, error),
          error,
        );
        return null;
      case 'sdkRelease':
      case 'sdkForget':
        final status = operation == 'sdkForget'
            ? native.mlsr_sdk_forget(h)
            : native.mlsr_sdk_release(h);
        if (status != 0) {
          throw const NativeFailure(
            'release_failed',
            'SDK handle release failed',
          );
        }
        return null;
      case 'sdkDispose':
        final kind = args['kind'] as String;
        Map<String, Object?>? completion, snapshot;
        try {
          if (kind == 'speech' || kind == 'synthesis') {
            _sdkCheck(
              kind == 'speech'
                  ? native.mlsr_speech_command(h, 0, 0, error)
                  : native.mlsr_synthesis_cancel(h, 0, error),
              error,
            );
            while (true) {
              if (_sdkCheck(
                    native.mlsr_sdk_wait(h, 50, output, error),
                    error,
                  ) ==
                  0) {
                completion = _sdkJson(output.value);
                break;
              }
            }
            _sdkCheck(native.mlsr_sdk_snapshot(h, output, error), error);
            snapshot = _sdkJson(output.value);
          }
          return {'completion': completion, 'snapshot': snapshot};
        } finally {
          if (native.mlsr_sdk_release(h) != 0) {
            throw const NativeFailure(
              'release_failed',
              'SDK handle release failed',
            );
          }
        }
      default:
        throw NativeFailure('invalid_operation', operation);
    }
  });
}
