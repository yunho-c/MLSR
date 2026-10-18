part of 'synthesis.dart';

/// Catalog-only discovery, without model loading or an audio device.
Future<List<Voice>> getVoices() async =>
    _voicesFrom(await sdkWorkerOperation('sdkVoices', {}));
List<Voice> _voicesFrom(Object? data) =>
    List<Voice>.unmodifiable((_sdkMap(data)['voices'] as List).map(Voice._));
Future<ResolvedVoice> resolveVoice(
  String locale, {
  String voice = '',
  VoiceFallback fallback = VoiceFallback.strict,
}) async => ResolvedVoice._(
  _sdkMap(
    await sdkWorkerOperation('sdkResolve', {
      'locale': locale,
      'voice': voice,
      'fallback': fallback.index,
    }),
  ),
);

/// Verified asset readiness can be inspected before inference is loaded.
extension ModelVoiceCatalog on ModelHandle {
  Future<List<Voice>> getVoices() => _use(
    (handle) async =>
        _voicesFrom(await sdkWorkerOperation('sdkVoices', {'handle': handle})),
  );
}

class _SdkService {
  final _Adapter adapter;
  int _references = 0;
  _SdkService(this.adapter);
  void retain() => _references++;
  void release() {
    if (--_references == 0) adapter.close();
  }

  Future<Object?> call(String operation, Map<String, Object?> args) =>
      adapter.closed
      ? sdkWorkerOperation(operation, args)
      : adapter.call(operation, args);
}

// A finalizer token must not point back to the public object. The service keeps
// its adapter alive until the final child/observer cleanup has completed.
class _SdkLease {
  final int handle;
  final String kind;
  final _SdkService service;
  bool closing = false;
  int _users = 0;
  Completer<void>? _idle;
  Future<Map<String, Object?>>? _disposal;
  _SdkLease(this.handle, this.kind, this.service) {
    service.retain();
  }
  int get value {
    if (closing) {
      throw const SynthesisException('handle_closed', 'SDK handle is disposed');
    }
    return handle;
  }

  Future<T> use<T>(Future<T> Function(int) action) async {
    final handle = value;
    _users++;
    try {
      return await action(handle);
    } finally {
      if (--_users == 0) {
        _idle?.complete();
        _idle = null;
      }
    }
  }

  Future<Object?> call(
    String operation, [
    Map<String, Object?> args = const {},
  ]) => use((handle) => service.call(operation, {'handle': handle, ...args}));
  Object? sync(String operation, [Map<String, Object?> args = const {}]) =>
      sdkNativeOperation(operation, {'handle': value, ...args});
  Future<void> _waitUsers() async {
    if (_users != 0) await (_idle ??= Completer<void>()).future;
  }

  Future<Map<String, Object?>> dispose({bool forget = false}) {
    if (_disposal != null) return _disposal!;
    closing = true;
    return _disposal = _dispose(forget);
  }

  Future<Map<String, Object?>> _dispose(bool forget) async {
    try {
      if (forget) {
        // Finish already-issued metadata calls before forgetting their registry
        // ID. Explicit dispose instead cancels first, so a blocked render wakes.
        await _waitUsers();
        await sdkWorkerOperation('sdkForget', {'handle': handle});
        return {};
      }
      return _sdkMap(
        await sdkWorkerOperation('sdkDispose', {
          'handle': handle,
          'kind': kind,
        }),
      );
    } finally {
      await _waitUsers();
      service.release();
    }
  }
}

final _sdkFinalizer = Finalizer<_SdkLease>((lease) {
  unawaited(
    lease
        .dispose(forget: lease.kind == 'speech' || lease.kind == 'synthesis')
        .then<void>((_) {}, onError: (Object _, StackTrace _) {}),
  );
});

class _SdkObject {
  final _SdkLease _lease;
  _SdkObject(int handle, String kind, _SdkService service)
    : _lease = _SdkLease(handle, kind, service) {
    _sdkFinalizer.attach(this, _lease, detach: this);
  }
  bool get isDisposed => _lease.closing;
  Future<void> dispose() async {
    _sdkFinalizer.detach(this);
    await _lease.dispose();
  }
}

/// Persistent shared facade. Native speech stays in Rust; render/export and
/// cleanup run off the UI isolate. Children remain valid after owner disposal
/// to observe their final outcome and release their own handles.
class Tts extends _SdkObject {
  SpeechController? _defaultController;
  Tts._(int handle, _SdkService service) : super(handle, 'tts', service);
  static Future<Tts> _open(String operation, Map<String, Object?> args) async {
    checkDeveloperApi();
    final service = _SdkService(await _Adapter.spawn());
    try {
      final handle = await service.call(operation, args) as int;
      return Tts._(handle, service);
    } catch (_) {
      service.adapter.close();
      rethrow;
    }
  }

  static Future<Tts> open({
    required String manifest,
    required String bundle,
    TtsOptions options = const TtsOptions(),
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    int executionVariant = 0,
  }) => _open('sdkOpen', {
    'manifest': manifest,
    'bundle': bundle,
    'policy': computePolicy.index,
    'reuse': reuseOutputBuffers ? 1 : 0,
    'variant': _sdkUInt(executionVariant),
    ...options._values,
  });
  static Future<Tts> fromModel(
    ModelHandle model, {
    TtsOptions options = const TtsOptions(),
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    int executionVariant = 0,
  }) => model._use(
    (handle) => _open('sdkFromModel', {
      'model': handle,
      'policy': computePolicy.index,
      'reuse': reuseOutputBuffers ? 1 : 0,
      'variant': _sdkUInt(executionVariant),
      ...options._values,
    }),
  );
  Future<List<Voice>> getVoices() async =>
      _voicesFrom(await _lease.call('sdkVoices'));
  Future<ResolvedVoice> resolveVoice(
    String locale, {
    String voice = '',
    VoiceFallback fallback = VoiceFallback.strict,
  }) async => ResolvedVoice._(
    _sdkMap(
      await _lease.call('sdkResolve', {
        'locale': locale,
        'voice': voice,
        'fallback': fallback.index,
      }),
    ),
  );
  Future<TtsState> snapshot() async =>
      TtsState._(_sdkMap(await _lease.call('sdkSnapshot')));
  SpeechController get controller => _defaultController ??= _controller(false);
  SpeechController createController() => _controller(true);
  SpeechController _controller(bool independent) => SpeechController._(
    this,
    _lease.sync('sdkController', {'independent': independent}) as int,
  );

  /// Immediate admission. Default replace is scoped to the default controller.
  Speech speak(
    String text, {
    RequestOptions request = const RequestOptions(),
    SpeechOptions options = const SpeechOptions(),
    SpeakMode mode = SpeakMode.replace,
  }) => _speak(this, _lease, text, request, options, mode);

  /// Lazy until audio is listened to or start is called. Observing never starts it.
  SynthesisJob synthesize(
    String text, {
    RequestOptions options = const RequestOptions(),
  }) {
    final values = _lease.sync('sdkSynthesize', options._values(text)) as List;
    return SynthesisJob._(this, values[0] as int, values[1] as int);
  }

  Future<AudioClip> collect(
    SynthesisJob job, {
    int maxBytes = 32 * 1024 * 1024,
  }) async {
    _sdkUInt(maxBytes, 64);
    final data = await _lease.use(
      (handle) => job._lease.use(
        (source) => sdkWorkerOperation('sdkCollect', {
          'handle': handle,
          'job': source,
          'maxBytes': maxBytes,
        }),
      ),
    );
    return AudioClip._(_sdkMap(data));
  }

  Future<AudioClip> render(
    String text, {
    RequestOptions options = const RequestOptions(),
    int maxBytes = 32 * 1024 * 1024,
  }) async {
    final job = synthesize(text, options: options);
    try {
      return await collect(job, maxBytes: maxBytes);
    } finally {
      await job.dispose();
    }
  }

  Future<WavFile> _write(
    _SdkObject source,
    String operation,
    String path,
    int maxFrames,
  ) async {
    _sdkUInt(maxFrames, 64);
    final data = await _lease.use(
      (handle) => source._lease.use(
        (source) => sdkWorkerOperation(operation, {
          'handle': handle,
          'source': source,
          'path': path,
          'maxFrames': maxFrames,
        }),
      ),
    );
    return WavFile._(_sdkMap(data));
  }

  Future<WavFile> writeJob(
    SynthesisJob job,
    String path, {
    int maxFrames = 30 * 60 * 44100,
  }) => _write(job, 'sdkWriteJob', path, maxFrames);
  Future<WavFile> renderToFile(
    String text,
    String path, {
    RequestOptions options = const RequestOptions(),
    int maxFrames = 30 * 60 * 44100,
  }) async {
    final job = synthesize(text, options: options);
    try {
      return await writeJob(job, path, maxFrames: maxFrames);
    } finally {
      await job.dispose();
    }
  }

  /// Export retained completed PCM without another inference pass.
  Future<WavFile> writeSpeech(
    Speech speech,
    String path, {
    int maxFrames = 30 * 60 * 44100,
  }) => _write(speech, 'sdkWriteSpeech', path, maxFrames);
  @override
  Future<void> dispose() async {
    try {
      await super.dispose();
    } finally {
      await _defaultController?.dispose();
    }
  }
}

class SpeechController extends _SdkObject {
  final Tts _owner;

  /// Facade whose inference lane and retained-storage budget this controller uses.
  Tts get tts => _owner;
  SpeechController._(this._owner, int handle)
    : super(handle, 'controller', _owner._lease.service);
  Future<OutputCapabilities> capabilities() async =>
      OutputCapabilities._(_sdkMap(await _lease.call('sdkSnapshot')));
  Speech speak(
    String text, {
    RequestOptions request = const RequestOptions(),
    SpeechOptions options = const SpeechOptions(),
    SpeakMode mode = SpeakMode.replace,
  }) => _speak(this, _lease, text, request, options, mode);
}

Speech _speak(
  Object owner,
  _SdkLease lease,
  String text,
  RequestOptions request,
  SpeechOptions options,
  SpeakMode mode,
) {
  final values =
      lease.sync('sdkSpeak', {
            ...request._values(text),
            ...options._values(mode),
          })
          as List;
  return Speech._(owner, lease.service, values[0] as int, values[1] as int);
}

class _SdkRequest extends _SdkObject {
  final Object _owner;
  final int id;
  Future<Completion>? _done;
  _SdkRequest(
    this._owner,
    int handle,
    this.id,
    String kind,
    _SdkService service,
  ) : super(handle, kind, service);
  Future<RequestMetadata> request() async =>
      RequestMetadata._(_sdkMap(await _lease.call('sdkRequest')));
  Future<TextPlan?> plan() async {
    final data = await _lease.call('sdkPlan');
    return data == null ? null : TextPlan._(_sdkMap(data));
  }

  /// Independent of audio consumption. Failed outcomes become typed errors.
  Future<Completion> get done => _done ??= _wait();
  Future<Completion> _wait() async {
    while (true) {
      if (_lease.closing) {
        final result = await _lease.dispose();
        if (result['completion'] == null) {
          throw const SynthesisException(
            'handle_closed',
            'Request was forgotten',
          );
        }
        return Completion._(_sdkMap(result['completion'])).result();
      }
      try {
        final result = _sdkMap(await _lease.call('sdkWait', {'timeout': 50}));
        if (result['data'] != null) {
          return Completion._(_sdkMap(result['data'])).result();
        }
      } catch (_) {
        if (!_lease.closing) rethrow;
      }
    }
  }

  Future<Map<String, Object?>> _snapshot() async {
    if (_lease.closing) {
      final value = (await _lease.dispose())['snapshot'];
      if (value != null) return _sdkMap(value);
    }
    return _sdkMap(await _lease.call('sdkSnapshot'));
  }

  Stream<Map<String, Object?>> _states() async* {
    if (_lease.closing) {
      yield await _snapshot();
      return;
    }
    _SdkObject? observer;
    try {
      final handle = await _lease.call('sdkStates') as int;
      observer = _SdkObject(handle, 'watch', _lease.service);
      while (true) {
        final result = _sdkMap(
          await observer._lease.call('sdkStatePoll', {'timeout': 50}),
        );
        if (result['status'] == 2) break;
        if (result['status'] == 1) continue;
        final data = _sdkMap(result['data']);
        yield data;
        if (data['completion'] != null) break;
      }
    } finally {
      await observer?.dispose();
    }
  }
}

class SynthesisJob extends _SdkRequest {
  SynthesisJob._(Tts owner, int handle, int id)
    : super(owner, handle, id, 'synthesis', owner._lease.service);
  void start() => _lease.sync('sdkStart');
  void cancel() => _lease.sync('sdkCancelSynthesis');
  Future<SynthesisState> snapshot() async =>
      SynthesisState._(await _snapshot());

  /// A fresh single-subscription, coalesced stream on each access. A paused
  /// listener stops polling; it cannot accumulate an unbounded state queue.
  Stream<SynthesisState> get states => _states().map(SynthesisState._);

  /// One consumer, claimed on listen. await-for/paused subscriptions backpressure
  /// Rust. Cancelling early uses consumerDropped and joins off the UI isolate.
  Stream<AudioChunk> get audio => _audio();
  Stream<AudioChunk> _audio() async* {
    _SdkObject? consumer;
    try {
      final handle = await _lease.call('sdkAudio') as int;
      consumer = _SdkObject(handle, 'stream', _lease.service);
      while (true) {
        final result = _sdkMap(
          await consumer._lease.call('sdkAudioPoll', {'timeout': 50}),
        );
        if (result['status'] == 2) break;
        if (result['status'] == 1) continue;
        if (result['completion'] != null) {
          Completion._(_sdkMap(result['completion'])).result();
          break;
        }
        yield AudioChunk._(_sdkMap(result['audio']));
      }
    } finally {
      await consumer?.dispose();
    }
  }
}

enum NavigationKind { boundary, seeked, started }

class Navigation {
  final NavigationKind kind;
  final int? frame;
  final Speech? speech;
  Navigation._(this.kind, this.frame, this.speech);
}

class Speech extends _SdkRequest {
  Speech._(Object owner, _SdkService service, int handle, int id)
    : super(owner, handle, id, 'speech', service);
  Future<SpeechState> snapshot() async => SpeechState._(await _snapshot());
  Stream<SpeechState> get states => _states().map(SpeechState._);
  void _command(int command, [double value = 0]) =>
      _lease.sync('sdkCommand', {'command': command, 'value': value});
  void cancel() => _command(0);
  void pause() => _command(1);
  void resume() => _command(2);
  void setVolume(double volume) => _command(3, volume);
  void setPlaybackRate(double rate) => _command(4, rate);
  void seek(int frame) =>
      _lease.sync('sdkSeek', {'frame': _sdkUInt(frame, 64)});
  Speech replayFrom(
    int frame, {
    SpeechOptions options = const SpeechOptions(),
    SpeakMode mode = SpeakMode.replace,
  }) {
    final values =
        _lease.sync('sdkReplay', {
              'frame': _sdkUInt(frame, 64),
              ...options._values(mode),
            })
            as List;
    return Speech._(_owner, _lease.service, values[0] as int, values[1] as int);
  }

  Navigation _navigate(int target) {
    final data = _sdkMap(_lease.sync('sdkNavigate', {'target': target}));
    return Navigation._(
      _sdkEnum(NavigationKind.values, data['kind']),
      data['frame'] as int?,
      data['handle'] == 0
          ? null
          : Speech._(
              _owner,
              _lease.service,
              data['handle'] as int,
              data['id'] as int,
            ),
    );
  }

  Navigation previousSentence() => _navigate(-1);
  Navigation nextSentence() => _navigate(-2);
  Navigation navigateToSentence(int sentenceId) =>
      _navigate(_sdkUInt(sentenceId));
}
