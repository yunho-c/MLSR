import 'package:flutter/material.dart';
import 'speech_controller.dart';
import 'buffered_track.dart';

/// Controls send commands to the public SDK. The slider's drag preview is local
/// UI state; its committed source-frame position is supplied by the SDK.
class ReaderControls extends StatefulWidget {
  final DemoController controller;
  const ReaderControls({super.key, required this.controller});
  @override
  State<ReaderControls> createState() => _ReaderControlsState();
}

class _ReaderControlsState extends State<ReaderControls> {
  double? _seekPreview;
  int? _dragRequest;
  int? _unavailableRequest;
  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final state = controller.state;
    final timeline = controller.timeline;
    final active = controller.speaking;
    final maximum = timeline.hasTotal ? timeline.totalFrames!.toDouble() : 1.0;
    final sameRequest = _dragRequest == state?.requestId;
    final position =
        (timeline.hasTotal
                ? ((sameRequest ? _seekPreview : null) ??
                      timeline.displayPosition.toDouble())
                : 0.0)
            .clamp(0.0, maximum);
    final canSeek =
        timeline.hasTotal &&
        timeline.ranges.isNotEmpty &&
        (active || controller.canReplay);
    final original = controller.original;
    final highlight = state?.activeTextRange;
    final spans = highlight == null
        ? [TextSpan(text: original)]
        : [
            TextSpan(text: original.substring(0, highlight.start)),
            TextSpan(
              text: original.substring(highlight.start, highlight.end),
              style: const TextStyle(
                backgroundColor: Color(0xffdfefb3),
                color: Color(0xff243e33),
                fontWeight: FontWeight.w700,
              ),
            ),
            TextSpan(text: original.substring(highlight.end)),
          ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 12,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            IconButton(
              key: const Key('previous-sentence'),
              tooltip: 'Previous sentence',
              onPressed: active ? () => controller.navigate(next: false) : null,
              icon: const Icon(Icons.skip_previous_rounded),
            ),
            FilledButton.tonalIcon(
              key: const Key('pause'),
              onPressed: active || controller.canReplay
                  ? controller.togglePause
                  : null,
              icon: Icon(
                controller.paused || controller.canReplay
                    ? Icons.play_arrow_rounded
                    : Icons.pause_rounded,
              ),
              label: Text(
                controller.canReplay
                    ? 'Replay'
                    : controller.paused
                    ? 'Resume'
                    : 'Pause',
              ),
            ),
            IconButton(
              key: const Key('next-sentence'),
              tooltip: 'Next sentence',
              onPressed: active ? () => controller.navigate(next: true) : null,
              icon: const Icon(Icons.skip_next_rounded),
            ),
            const Text('Speed'),
            DropdownButton<double>(
              key: const Key('rate'),
              value: controller.playbackRate,
              items: [.5, .75, 1.0, 1.25, 1.5, 2.0]
                  .map((v) => DropdownMenuItem(value: v, child: Text('$v×')))
                  .toList(),
              onChanged: (value) => controller.setRate(value!),
            ),
            const Icon(Icons.volume_up_outlined, size: 20),
            SizedBox(
              width: 135,
              child: Slider(
                key: const Key('volume'),
                value: controller.volume,
                label: '${(controller.volume * 100).round()}%',
                onChanged: controller.setVolume,
              ),
            ),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            trackShape: BufferedTrack(
              timeline.bufferedRanges,
              timeline.totalFrames ?? 0,
            ),
          ),
          child: Slider(
            key: const Key('seek'),
            value: position,
            max: maximum,
            secondaryTrackValue: timeline.hasTotal
                ? timeline.bufferedUntil.toDouble().clamp(0.0, maximum)
                : null,
            label: timeline.canSeek(position.round())
                ? '${timeline.time(position.round())}${timeline.isBuffered(position.round()) ? '' : ' · Prepare on seek'}'
                : '${timeline.time(position.round())} · Not ready',
            semanticFormatterCallback: (value) =>
                '${timeline.time(value.round())} of ${timeline.totalLabel}'
                '${timeline.isBuffered(value.round())
                    ? ''
                    : timeline.canSeek(value.round())
                    ? ', will prepare audio'
                    : ', audio not ready'}',
            onChangeStart: canSeek
                ? (value) => setState(() {
                    _dragRequest = state?.requestId;
                    _seekPreview = value;
                    _unavailableRequest = null;
                  })
                : null,
            onChanged: canSeek
                ? (value) {
                    if (_dragRequest != controller.state?.requestId) return;
                    setState(() => _seekPreview = value);
                  }
                : null,
            onChangeEnd: (value) {
              // Recheck the current document and navigable timeline at release.
              // Ignore stale drags after replacement or navigation.
              if (_dragRequest != null &&
                  _dragRequest == controller.state?.requestId) {
                if (controller.timeline.canSeek(value.round())) {
                  controller.seek(value.round());
                } else {
                  _unavailableRequest = _dragRequest;
                }
              }
              setState(() {
                _seekPreview = null;
                _dragRequest = null;
              });
            },
          ),
        ),
        Text(
          _unavailableRequest != null && _unavailableRequest == state?.requestId
              ? 'Audio at this position is not ready yet.'
              : '${timeline.durationLabel}${state?.seeking ?? false ? ' · Preparing selection…' : ''}',
          key: const Key('timeline-detail'),
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 14),
        const Text(
          'READING · ORIGINAL TEXT',
          style: TextStyle(
            fontSize: 10,
            letterSpacing: 1.2,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 8),
        ConstrainedBox(
          constraints: const BoxConstraints(maxHeight: 160),
          child: SingleChildScrollView(
            child: SelectableText.rich(
              TextSpan(children: spans),
              key: const Key('reading-text'),
              style: const TextStyle(fontSize: 15, height: 1.6),
            ),
          ),
        ),
      ],
    );
  }
}
