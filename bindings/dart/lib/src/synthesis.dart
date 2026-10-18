import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'synthesis_native.dart';

part 'models.dart';

/// Native failure with a stable category. Terminal inference failures are also
/// represented by SynthesisEvent.error so the stream retains one final outcome.
typedef SynthesisException = NativeFailure;

/// Permitted compute devices, not a placement/residency guarantee.
enum ComputePolicy { cpuOnly, cpuAndNeuralEngine }

/// Per-job inference and memory limits. Queue/chunk/segment sizes are PCM frames.
class SynthesisOptions {
  final int steps, seed, queueFrames, chunkFrames, maxSegmentFrames;
  final double speed;
  const SynthesisOptions({
    this.steps = 8,
    this.speed = 1.05,
    this.seed = 0,
    this.queueFrames = 44100,
    this.chunkFrames = 4096,
    this.maxSegmentFrames = 1572864,
  });
  Map<String, Object?> _values() {
    for (final value in [steps, queueFrames, chunkFrames, maxSegmentFrames]) {
      if (value < 0 || value > 0xffffffff) {
        throw ArgumentError('Options must fit uint32');
      }
    }
    if (seed < 0) {
      throw ArgumentError('Seed must be a nonnegative Dart integer');
    }
    return {
      'steps': steps,
      'speed': speed,
      'seed': seed,
      'queue': queueFrames,
      'chunk': chunkFrames,
      'segment': maxSegmentFrames,
    };
  }
}

enum EventKind { predicted, audio, completed, cancelled, failed }

/// An owned event. textStart/textEnd index the original string's UTF-8 bytes,
/// not Dart UTF-16 code units. Samples remain valid after disposal.
class SynthesisEvent {
  final EventKind kind;
  final int jobId,
      segmentId,
      textStart,
      textEnd,
      startFrame,
      frames,
      predictedFrames;
  final int sampleRate, channels, completedSegments;
  final Float32List samples;
  final SynthesisException? error;
  SynthesisEvent._(Map<String, Object?> data)
    : kind = EventKind.values[(data['kind'] as int) - 1],
      jobId = data['job'] as int,
      segmentId = data['segment'] as int,
      textStart = data['textStart'] as int,
      textEnd = data['textEnd'] as int,
      startFrame = data['startFrame'] as int,
      frames = data['frames'] as int,
      predictedFrames = data['predictedFrames'] as int,
      sampleRate = data['rate'] as int,
      channels = data['channels'] as int,
      completedSegments = data['segments'] as int,
      samples =
          (data['samples'] as TransferableTypedData?)
              ?.materialize()
              .asFloat32List() ??
          Float32List(0),
      error = data['kind'] == 5
          ? SynthesisException(
              data['errorCode'] as String,
              data['errorMessage'] as String,
            )
          : null;
  bool get terminal => kind.index >= EventKind.completed.index;
}

/// Persistent engine with a dedicated adapter isolate for blocking FFI calls.
/// Always await dispose(); no native model or job is owned by a Dart pointer.
class Engine {
  final _Adapter _adapter;
  final int _handle;
  final Set<Job> _jobs = {};
  bool _disposed = false;
  int _starting = 0;
  Completer<void>? _startsIdle;
  Future<void>? _disposal;
  Engine._(this._adapter, this._handle);

  /// Reuse verified model data and retain its lease independently. The caller
  /// may dispose the model after this future completes, including on failure.
  static Future<Engine> fromModel(
    ModelHandle model, {
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    int executionVariant = 0,
  }) => model._use((handle) async {
    checkModelApi();
    if (executionVariant < 0 || executionVariant > 0xffffffff) {
      throw ArgumentError('Invalid variant');
    }
    final adapter = await _Adapter.spawn();
    try {
      final engine =
          await adapter.call('openModel', {
                'model': handle,
                'policy': computePolicy.index,
                'reuse': reuseOutputBuffers ? 1 : 0,
                'variant': executionVariant,
              })
              as int;
      return Engine._(adapter, engine);
    } catch (_) {
      adapter.close();
      rethrow;
    }
  });

  static Future<Engine> load({
    required String manifest,
    required String bundle,
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    int executionVariant = 0,
  }) async {
    checkSynthesisApi(); // Keeps native assets loaded in the calling isolate too.
    if (executionVariant < 0 || executionVariant > 0xffffffff) {
      throw ArgumentError('Invalid variant');
    }
    final adapter = await _Adapter.spawn();
    try {
      final handle =
          await adapter.call('open', {
                'manifest': manifest,
                'bundle': bundle,
                'policy': computePolicy.index,
                'reuse': reuseOutputBuffers ? 1 : 0,
                'variant': executionVariant,
              })
              as int;
      return Engine._(adapter, handle);
    } catch (_) {
      adapter.close();
      rethrow;
    }
  }

  /// Copy input and start one job. A competing submission fails with engine_busy.
  Future<Job> start(
    String text, {
    String language = 'en',
    String voice = 'M1',
    SynthesisOptions options = const SynthesisOptions(),
  }) async {
    if (_disposed) {
      throw const SynthesisException('engine_closed', 'Engine is disposed');
    }
    _starting++;
    try {
      final values =
          await _adapter.call('start', {
                'engine': _handle,
                'text': text,
                'language': language,
                'voice': voice,
                ...options._values(),
              })
              as List;
      final handle = values[0] as int;
      if (_disposed) {
        await _release('job', handle);
        throw const SynthesisException(
          'engine_closed',
          'Engine closed during submission',
        );
      }
      final job = Job._(this, handle, values[1] as int);
      _jobs.add(job);
      return job;
    } finally {
      if (--_starting == 0) {
        _startsIdle?.complete();
        _startsIdle = null;
      }
    }
  }

  Future<void> _release(String kind, int handle) async {
    if (_adapter.closed) {
      await Isolate.run(() => releaseNative(kind, handle));
    } else {
      await _adapter.call(kind == 'job' ? 'releaseJob' : 'releaseEngine', {
        kind: handle,
      });
    }
  }

  /// Stop all owned jobs and join native workers without blocking the UI isolate.
  Future<void> dispose() => _disposal ??= _dispose();
  Future<void> _dispose() async {
    _disposed = true;
    if (_starting > 0) await (_startsIdle ??= Completer<void>()).future;
    Object? failure;
    StackTrace? trace;
    try {
      for (final job in _jobs.toList()) {
        try {
          await job.dispose();
        } catch (error, stack) {
          failure ??= error;
          trace ??= stack;
        }
      }
      await _release('engine', _handle);
    } finally {
      _adapter.close();
    }
    if (failure != null) Error.throwWithStackTrace(failure, trace!);
  }
}

/// One single-subscription stream. Polling occurs only when the stream resumes;
/// pause/await-for applies backpressure across Dart and Rust. A listener's async
/// callback must pause its subscription if it needs to await slow consumption.
class Job {
  final Engine _engine;
  int _handle;
  final int id;
  bool _cancelled = false;
  Future<void>? _disposal;
  late final Stream<SynthesisEvent> events = _events();
  Job._(this._engine, this._handle, this.id);

  Stream<SynthesisEvent> _events() async* {
    try {
      while (_handle != 0) {
        final data = Map<String, Object?>.from(
          await _engine._adapter.call('poll', {'job': _handle, 'timeout': 100})
              as Map,
        );
        if (_handle == 0) break;
        if (data['status'] == 1) continue;
        if (data['status'] == 2) break;
        final event = SynthesisEvent._(data);
        if (event.jobId != id) {
          throw const SynthesisException('stale_event', 'Job ID mismatch');
        }
        if (!_cancelled || event.terminal) yield event;
        if (event.terminal) break;
      }
    } finally {
      await dispose();
    }
  }

  /// Immediate flag/queue cancellation; does not wait on the adapter isolate.
  void cancel() {
    _cancelled = true;
    if (_handle != 0) cancelNative(_handle);
  }

  /// Wait for quiescence off the UI isolate, without draining audio.
  Future<bool> wait({int timeoutMs = 100}) async {
    if (timeoutMs < 0 || timeoutMs > 0xffffffff) {
      throw ArgumentError('Invalid timeout');
    }
    if (_handle == 0) return true;
    return await _engine._adapter.call('wait', {
          'job': _handle,
          'timeout': timeoutMs,
        })
        as bool;
  }

  /// Cancel and release. Stream cancellation also calls this method.
  Future<void> dispose() => _disposal ??= _dispose();
  Future<void> _dispose() async {
    final handle = _handle;
    _handle = 0;
    _cancelled = true;
    try {
      if (handle != 0) {
        try {
          cancelNative(handle);
        } finally {
          await _engine._release('job', handle);
        }
      }
    } finally {
      _engine._jobs.remove(this);
    }
  }
}

class _Adapter {
  final _responses = ReceivePort(),
      _errors = ReceivePort(),
      _exits = ReceivePort();
  final _ready = Completer<SendPort>();
  final Map<int, Completer<Object?>> _pending = {};
  late final SendPort _commands;
  Isolate? _isolate;
  int _next = 0;
  bool closed = false;
  _Adapter() {
    _responses.listen((dynamic message) {
      if (message is SendPort) {
        if (!_ready.isCompleted) _ready.complete(message);
        return;
      }
      final values = message as List;
      final completer = _pending.remove(values[0]);
      if (completer == null) return;
      final error = values[2] as List?;
      if (error == null) {
        completer.complete(values[1]);
      } else {
        completer.completeError(
          SynthesisException(error[0] as String, error[1] as String),
        );
      }
    });
    _errors.listen(
      (dynamic error) =>
          _fail(SynthesisException('worker_failed', error.toString())),
    );
    _exits.listen((_) {
      if (!closed) {
        _fail(
          const SynthesisException('worker_failed', 'Adapter isolate exited'),
        );
      }
    });
  }
  static Future<_Adapter> spawn() async {
    final worker = _Adapter();
    try {
      worker._isolate = await Isolate.spawn(
        adapterMain,
        worker._responses.sendPort,
        onError: worker._errors.sendPort,
        onExit: worker._exits.sendPort,
        debugName: 'mlsr-ffi',
      );
      worker._commands = await worker._ready.future;
      return worker;
    } catch (_) {
      worker.close();
      rethrow;
    }
  }

  Future<Object?> call(String operation, Map<String, Object?> args) {
    if (closed) {
      return Future.error(
        const SynthesisException('worker_closed', 'Adapter is closed'),
      );
    }
    final id = ++_next;
    final completer = Completer<Object?>();
    _pending[id] = completer;
    _commands.send([id, operation, args]);
    return completer.future;
  }

  void _fail(SynthesisException error) {
    if (!_ready.isCompleted) _ready.completeError(error);
    for (final completer in _pending.values) {
      completer.completeError(error);
    }
    _pending.clear();
    close();
  }

  void close() {
    if (closed) return;
    closed = true;
    for (final completer in _pending.values) {
      completer.completeError(
        const SynthesisException('worker_closed', 'Adapter is closed'),
      );
    }
    _pending.clear();
    _isolate?.kill(priority: Isolate.immediate);
    _responses.close();
    _errors.close();
    _exits.close();
  }
}
