import 'package:mlsr_flutter/mlsr_flutter.dart';

/// Presentation of one SDK snapshot. Total length, available audio and the
/// source cursor remain separate, including while generation is backpressured.
class SpeechTimeline {
  final SpeechState? state;
  const SpeechTimeline(this.state);

  int? get totalFrames => state?.durationFrames ?? state?.estimatedFrames;
  bool get hasTotal => totalFrames != null && totalFrames! > 0;
  bool get estimated =>
      state?.durationFrames == null && state?.estimatedFrames != null;
  int get position => state?.positionFrame ?? 0;
  int get displayPosition => state?.seekTargetFrame ?? position;
  int get bufferedUntil => state?.bufferedUntilFrame ?? 0;
  int get sampleRate => state?.format.sampleRate ?? 44100;
  List<IndexRange> get ranges => state?.seekableRanges ?? const [];
  List<IndexRange> get bufferedRanges => state?.bufferedRanges ?? const [];
  bool isBuffered(int frame) =>
      bufferedRanges.any((range) => frame >= range.start && frame < range.end);
  int get seekableUntil =>
      ranges.fold(0, (end, range) => range.end > end ? range.end : end);

  double? get progress =>
      hasTotal ? (position / totalFrames!).clamp(0, 1) : null;
  double? get bufferedProgress =>
      hasTotal ? (bufferedUntil / totalFrames!).clamp(0, 1) : null;
  bool canSeek(int frame) => ranges.any(
    (range) =>
        range.end > range.start && frame >= range.start && frame <= range.end,
  );

  String time(int frames) {
    final seconds = frames ~/ sampleRate;
    return '${seconds ~/ 60}:${(seconds % 60).toString().padLeft(2, '0')}';
  }

  String get positionLabel => time(position);
  String get totalLabel => totalFrames == null
      ? '—'
      : '${estimated ? '~' : ''}${time(totalFrames!)}';
  String get durationLabel => totalFrames == null
      ? 'Estimating duration…'
      : '${estimated ? 'Estimated duration' : 'Duration'}: ${time(totalFrames!)}';
}
