part of 'synthesis.dart';

Map<String, Object?> _sdkMap(Object? value) =>
    Map<String, Object?>.from(value as Map);
int _sdkUInt(int value, [int bits = 32]) {
  if (value < 0 || (bits == 32 && value > 0xffffffff)) {
    throw ArgumentError('Value must fit uint$bits');
  }
  return value;
}

T _sdkEnum<T extends Enum>(List<T> values, Object? wire) {
  final name = (wire as String).replaceAllMapped(
    RegExp(r'_([a-z])'),
    (m) => m[1]!.toUpperCase(),
  );
  return values.firstWhere(
    (v) => v.name == name,
    orElse: () => throw NativeFailure('invalid_metadata', 'Unknown enum $wire'),
  );
}

enum VoiceReadiness { catalogOnly, assetsVerified, engineReady }

enum VoiceFallback { strict, sameLanguage }

enum OutputMode { platformDefault, headless, hostManaged }

enum SpeakMode { replace, enqueue }

enum SynthesisPhase {
  created,
  queued,
  preparing,
  synthesizing,
  generated,
  terminal,
}

enum PlaybackPhase {
  queued,
  buffering,
  playing,
  draining,
  paused,
  ended,
  stopped,
  failed,
}

enum OutcomeKind { completed, cancelled, failed }

enum CancelReason { requested, replaced, navigation, closed, consumerDropped }

enum FailureStage {
  planning,
  synthesis,
  rendering,
  writing,
  playback,
  retention,
}

/// Model-specific inference controls. Public speakingRate is separate.
class SupertonicOptions {
  final int steps, seed;
  const SupertonicOptions({this.steps = 8, this.seed = 0});
}

/// Immutable input options. Normal speakingRate=1.0 maps to raw model speed 1.05.
class RequestOptions {
  final String locale, voice;
  final VoiceFallback fallback;
  final double speakingRate;
  final SupertonicOptions backend;
  const RequestOptions({
    this.locale = 'en',
    this.voice = '',
    this.fallback = VoiceFallback.strict,
    this.speakingRate = 1.0,
    this.backend = const SupertonicOptions(),
  });
  Map<String, Object?> _values(String text) => {
    'text': text,
    'locale': locale,
    'voice': voice,
    'fallback': fallback.index,
    'speakingRate': speakingRate,
    'steps': _sdkUInt(backend.steps),
    'seed': _sdkUInt(backend.seed, 64),
  };
}

class TtsOptions {
  final OutputMode output;
  final int maxRequests,
      maxTextBytes,
      queueFrames,
      chunkFrames,
      maxSegmentFrames,
      maxRetainedStores,
      maxRetainedBytes;
  const TtsOptions({
    this.output = OutputMode.platformDefault,
    this.maxRequests = 16,
    this.maxTextBytes = 256 * 1024,
    this.queueFrames = 44100,
    this.chunkFrames = 4096,
    this.maxSegmentFrames = 512 * 3072,
    this.maxRetainedStores = 128,
    this.maxRetainedBytes = 128 * 1024 * 1024,
  });
  Map<String, Object?> get _values => {
    'outputMode': output.index,
    'maxRequests': _sdkUInt(maxRequests),
    'maxTextBytes': _sdkUInt(maxTextBytes),
    'queueFrames': _sdkUInt(queueFrames),
    'chunkFrames': _sdkUInt(chunkFrames),
    'maxSegmentFrames': _sdkUInt(maxSegmentFrames),
    'maxRetainedStores': _sdkUInt(maxRetainedStores),
    'maxRetainedBytes': _sdkUInt(maxRetainedBytes, 64),
  };
}

class SpeechOptions {
  final double playbackRate, volume;
  final bool paused, retainAudio;

  /// Generate unbuffered seek targets and evict unused cached segments as needed.
  /// Requires retainAudio. Full recording export is available only while complete.
  final bool onDemandSeeking;

  /// Retained speech starts segments within this source-frame horizon. Each
  /// admitted segment is retained in full, even past it. Output limits are separate.
  final int lookaheadFrames;
  const SpeechOptions({
    this.playbackRate = 1,
    this.volume = 1,
    this.paused = false,
    this.retainAudio = true,
    this.onDemandSeeking = false,
    this.lookaheadFrames = 88200,
  });
  Map<String, Object?> _values(SpeakMode mode) {
    if (onDemandSeeking && !retainAudio) {
      throw ArgumentError('On-demand seeking requires retainAudio');
    }
    return {
      'mode': mode.index,
      'playbackRate': playbackRate,
      'volume': volume,
      'paused': paused,
      'retainAudio': retainAudio,
      'onDemandSeeking': onDemandSeeking,
      'lookaheadFrames': _sdkUInt(lookaheadFrames),
    };
  }
}

/// Half-open range. The containing field names frames, UTF-8 or UTF-16.
class IndexRange {
  final int start, end;
  const IndexRange(this.start, this.end);
  factory IndexRange._(Object? value) {
    final data = _sdkMap(value);
    return IndexRange(data['start'] as int, data['end'] as int);
  }
  bool contains(int value) => value >= start && value < end;
}

/// Original source mapping. start/end use Dart UTF-16 code units directly.
/// Segment alignment does not claim word-level timestamps.
class SpeechTextRange {
  final int segmentId, sentenceId;
  final IndexRange utf8, utf16;
  SpeechTextRange._(Map<String, Object?> data)
    : segmentId = data['segment_id'] as int,
      sentenceId = data['sentence_id'] as int,
      utf8 = IndexRange._(data['utf8']),
      utf16 = IndexRange._(data['utf16']);
  int get start => utf16.start;
  int get end => utf16.end;
  String text(String original) => original.substring(start, end);
}

class AudioFormat {
  final int sampleRate, channels;
  const AudioFormat({required this.sampleRate, required this.channels});
  factory AudioFormat._(Object? value) {
    final data = _sdkMap(value);
    return AudioFormat(
      sampleRate: data['sample_rate'] as int,
      channels: data['channels'] as int,
    );
  }
  Duration duration(int frames) => Duration(
    microseconds: (frames * Duration.microsecondsPerSecond / sampleRate)
        .round(),
  );
}

class WavFile {
  final String path;
  final int frames;
  final AudioFormat format;
  WavFile._(Map<String, Object?> data)
    : path = data['path'] as String,
      frames = data['frames'] as int,
      format = AudioFormat._(data['format']);
  Duration get duration => format.duration(frames);
}

/// Dart-owned PCM. It remains valid after disposal of every native handle.
class AudioClip {
  final Float32List samples;
  final AudioFormat format;
  AudioClip(this.samples, this.format) {
    if (format.channels < 1 || samples.length % format.channels != 0) {
      throw ArgumentError('PCM must contain complete frames');
    }
  }
  factory AudioClip._(Map<String, Object?> data) => AudioClip(
    (data['samples'] as TransferableTypedData).materialize().asFloat32List(),
    AudioFormat._(data['format']),
  );
  int get frames => samples.length ~/ format.channels;
  Duration get duration => format.duration(frames);

  /// Shared Rust serializer, off the UI isolate. Existing paths are preserved.
  /// This copies PCM to the worker/native buffer; renderToFile avoids collection.
  Future<WavFile> writeWav(String path) async => WavFile._(
    _sdkMap(
      await sdkWorkerOperation('sdkWritePcm', {
        'path': path,
        'samples': TransferableTypedData.fromList([samples]),
        'sampleRate': format.sampleRate,
        'channels': format.channels,
      }),
    ),
  );
}

class AudioChunk {
  final int requestId, startFrame;
  final AudioClip audio;
  final SpeechTextRange textRange;
  AudioChunk._(Map<String, Object?> data)
    : requestId = data['request_id'] as int,
      startFrame = data['start_frame'] as int,
      audio = AudioClip._(data),
      textRange = SpeechTextRange._(data);
}

class SynthesisFailure {
  final int requestId;
  final String category, message;
  final String? backend;
  final FailureStage stage;
  SynthesisFailure._(Object? value) : this._map(_sdkMap(value));
  SynthesisFailure._map(Map<String, Object?> data)
    : requestId = data['request_id'] as int,
      category = data['category'] as String,
      message = data['message'] as String,
      backend = data['backend'] as String?,
      stage = _sdkEnum(FailureStage.values, data['stage']);
  SynthesisException get exception => SynthesisException(
    category,
    message,
    category: category,
    stage: stage.name,
    requestId: requestId,
    backend: backend,
  );
}

class Completion {
  final int requestId, frames;
  final OutcomeKind kind;
  final CancelReason? reason;
  final SynthesisFailure? failure;
  Completion._(Map<String, Object?> data)
    : this._parts(data, _sdkMap(data['outcome']));
  Completion._parts(Map<String, Object?> data, Map<String, Object?> outcome)
    : requestId = data['request_id'] as int,
      frames = data['frames'] as int,
      kind = _sdkEnum(OutcomeKind.values, outcome['kind']),
      reason = outcome['reason'] == null
          ? null
          : _sdkEnum(CancelReason.values, outcome['reason']),
      failure = outcome['error'] == null
          ? null
          : SynthesisFailure._(outcome['error']);
  Completion result() {
    if (failure != null) throw failure!.exception;
    return this;
  }
}

class VoiceCapabilities {
  final bool pcm, durationPrediction, segmentAlignment, wordAlignment;
  VoiceCapabilities._(Object? value) : this._map(_sdkMap(value));
  VoiceCapabilities._map(Map<String, Object?> data)
    : pcm = data['pcm'] as bool,
      durationPrediction = data['duration_prediction'] as bool,
      segmentAlignment = data['segment_alignment'] as bool,
      wordAlignment = data['word_alignment'] as bool;
}

class Voice {
  final String id,
      modelVoiceId,
      displayName,
      model,
      bundleId,
      revision,
      manifestSha256,
      backend;
  final List<String> languages, qualifiedLanguages;
  final bool requiresNetwork;
  final VoiceReadiness readiness;
  final VoiceCapabilities capabilities;
  Voice._(Object? value) : this._map(_sdkMap(value));
  Voice._map(Map<String, Object?> data)
    : id = data['id'] as String,
      modelVoiceId = data['model_voice_id'] as String,
      displayName = data['display_name'] as String,
      model = data['model'] as String,
      bundleId = data['bundle_id'] as String,
      revision = data['revision'] as String,
      manifestSha256 = data['manifest_sha256'] as String,
      backend = data['backend'] as String,
      languages = List<String>.unmodifiable(data['languages'] as List),
      qualifiedLanguages = List<String>.unmodifiable(
        data['qualified_languages'] as List,
      ),
      requiresNetwork = data['requires_network'] as bool,
      readiness = _sdkEnum(VoiceReadiness.values, data['readiness']),
      capabilities = VoiceCapabilities._(data['capabilities']);
}

class ResolvedVoice {
  final Voice voice;
  final String requestedLocale, language;
  final bool usedFallback;
  ResolvedVoice._(Map<String, Object?> data)
    : voice = Voice._(data['voice']),
      requestedLocale = data['requested_locale'] as String,
      language = data['language'] as String,
      usedFallback = data['used_fallback'] as bool;
}

class RequestMetadata {
  final int requestId, startSegment;
  final String text;
  final ResolvedVoice voice;
  final double speakingRate, effectiveSpeed;
  final SupertonicOptions backend;
  RequestMetadata._(Map<String, Object?> data)
    : requestId = data['request_id'] as int,
      startSegment = data['start_segment'] as int,
      text = data['text'] as String,
      voice = ResolvedVoice._(_sdkMap(data['voice'])),
      speakingRate = (data['speaking_rate'] as num).toDouble(),
      effectiveSpeed = (_sdkMap(data['effective_synthesis'])['speed'] as num)
          .toDouble(),
      backend = SupertonicOptions(
        steps: _sdkMap(data['backend'])['steps'] as int,
        seed: _sdkMap(data['backend'])['seed'] as int,
      );
}

class OutputCapabilities {
  final int maxChunkFrames;
  final bool pauseResume,
      volume,
      rebase,
      hostManagedPolicy,
      routeEvents,
      interruptionEvents;
  final List<double>? rateRange;
  OutputCapabilities._(Map<String, Object?> data)
    : maxChunkFrames = data['max_chunk_frames'] as int,
      pauseResume = data['pause_resume'] as bool,
      volume = data['volume'] as bool,
      rebase = data['rebase'] as bool,
      hostManagedPolicy = data['host_managed_policy'] as bool,
      routeEvents = data['route_events'] as bool,
      interruptionEvents = data['interruption_events'] as bool,
      rateRange = data['rate_range'] == null
          ? null
          : List<double>.unmodifiable(
              (data['rate_range'] as List).map((v) => (v as num).toDouble()),
            );
}

class RetentionStats {
  final int usedBytes, peakBytes, liveStores, peakStores;
  RetentionStats._(Object? value) : this._map(_sdkMap(value));
  RetentionStats._map(Map<String, Object?> data)
    : usedBytes = data['used_bytes'] as int,
      peakBytes = data['peak_bytes'] as int,
      liveStores = data['live_stores'] as int,
      peakStores = data['peak_stores'] as int;
}

class TtsState {
  final OutputCapabilities? outputCapabilities;
  final RetentionStats retention;
  TtsState._(Map<String, Object?> data)
    : outputCapabilities = data['output_capabilities'] == null
          ? null
          : OutputCapabilities._(_sdkMap(data['output_capabilities'])),
      retention = RetentionStats._(data['retention']);
}

class SynthesisState {
  final int revision,
      requestId,
      generatedFrames,
      consumedFrames,
      queuedFrames,
      peakQueuedFrames;
  final SynthesisPhase phase;
  final bool cancellationRequested, planReady;
  final AudioFormat format;
  final int? estimatedFrames, durationFrames;
  final Completion? completion;
  SynthesisState._(Map<String, Object?> data)
    : revision = data['revision'] as int,
      requestId = data['request_id'] as int,
      generatedFrames = data['generated_frames'] as int,
      consumedFrames = data['consumed_frames'] as int,
      queuedFrames = data['queued_frames'] as int,
      peakQueuedFrames = data['peak_queued_frames'] as int,
      phase = _sdkEnum(SynthesisPhase.values, data['phase']),
      cancellationRequested = data['cancellation_requested'] as bool,
      planReady = data['plan_ready'] as bool,
      format = AudioFormat._(data['format']),
      estimatedFrames = data['estimated_frames'] as int?,
      durationFrames = data['duration_frames'] as int?,
      completion = data['completion'] == null
          ? null
          : Completion._(_sdkMap(data['completion']));
}

class SpeechState {
  final int revision, requestId, positionFrame, bufferedUntilFrame;
  final SynthesisPhase synthesis;
  final PlaybackPhase playback;
  final AudioFormat format;
  final List<IndexRange> seekableRanges;
  final List<IndexRange> bufferedRanges;
  final bool seeking, recordingComplete;
  final int? seekTargetFrame;
  final int? estimatedFrames, durationFrames;
  final double playbackRate, volume;
  final SpeechTextRange? activeTextRange;
  final double? outputLatencySeconds;
  final CancelReason? cancellation;
  final Completion? completion;
  SpeechState._(Map<String, Object?> data)
    : revision = data['revision'] as int,
      requestId = data['request_id'] as int,
      positionFrame = data['position_frame'] as int,
      bufferedUntilFrame = data['buffered_until_frame'] as int,
      synthesis = _sdkEnum(SynthesisPhase.values, data['synthesis']),
      playback = _sdkEnum(PlaybackPhase.values, data['playback']),
      format = AudioFormat._(data['format']),
      seekableRanges = List<IndexRange>.unmodifiable(
        (data['seekable_ranges'] as List).map(IndexRange._),
      ),
      bufferedRanges = List<IndexRange>.unmodifiable(
        ((data['buffered_ranges'] ?? data['seekable_ranges']) as List).map(
          IndexRange._,
        ),
      ),
      seeking = data['seeking'] as bool? ?? false,
      seekTargetFrame = data['seek_target_frame'] as int?,
      recordingComplete = data['recording_complete'] as bool? ?? false,
      estimatedFrames = data['estimated_frames'] as int?,
      durationFrames = data['duration_frames'] as int?,
      playbackRate = (data['playback_rate'] as num).toDouble(),
      volume = (data['volume'] as num).toDouble(),
      activeTextRange = data['active_text_range'] == null
          ? null
          : SpeechTextRange._(_sdkMap(data['active_text_range'])),
      outputLatencySeconds = (data['output_latency_seconds'] as num?)
          ?.toDouble(),
      cancellation = data['cancellation'] == null
          ? null
          : _sdkEnum(CancelReason.values, data['cancellation']),
      completion = data['completion'] == null
          ? null
          : Completion._(_sdkMap(data['completion']));
  Duration get position => format.duration(positionFrame);
  Duration get bufferedUntil => format.duration(bufferedUntilFrame);
  Duration? get duration =>
      durationFrames == null ? null : format.duration(durationFrames!);
  Duration? get estimatedDuration =>
      estimatedFrames == null ? null : format.duration(estimatedFrames!);
}

class TextSegment {
  final int id, sentenceId;
  final IndexRange utf8, utf16;
  final String original, normalized;
  TextSegment._(Object? value) : this._map(_sdkMap(value));
  TextSegment._map(Map<String, Object?> data)
    : id = data['id'] as int,
      sentenceId = data['sentence_id'] as int,
      utf8 = IndexRange._(data['utf8']),
      utf16 = IndexRange._(data['utf16']),
      original = data['original'] as String,
      normalized = data['normalized'] as String;
}

class TextSentence {
  final int id;
  final IndexRange utf8, utf16, segments;
  TextSentence._(Object? value) : this._map(_sdkMap(value));
  TextSentence._map(Map<String, Object?> data)
    : id = data['id'] as int,
      utf8 = IndexRange._(data['utf8']),
      utf16 = IndexRange._(data['utf16']),
      segments = IndexRange._(data['segments']);
}

class TextPlan {
  final String original, language, manifestSha256;
  final List<TextSegment> segments;
  final List<TextSentence> sentences;
  TextPlan._(Map<String, Object?> data)
    : original = data['original'] as String,
      language = data['language'] as String,
      manifestSha256 = data['manifest_sha256'] as String,
      segments = List<TextSegment>.unmodifiable(
        (data['segments'] as List).map(TextSegment._),
      ),
      sentences = List<TextSentence>.unmodifiable(
        (data['sentences'] as List).map(TextSentence._),
      );
}
