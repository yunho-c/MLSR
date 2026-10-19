import 'package:flutter/material.dart';
import 'speech_controller.dart';

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
  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final state = controller.state;
    final active = controller.speaking;
    final ranges = state?.seekableRanges ?? [];
    final maximum = ranges.isEmpty ? 0 : ranges.last.end;
    final sameRequest = _dragRequest == controller.speech?.id;
    final position =
        ((sameRequest ? _seekPreview : null) ??
                controller.playedFrames.toDouble())
            .clamp(0.0, maximum.toDouble());
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
              onPressed: active || controller.canExport
                  ? controller.togglePause
                  : null,
              icon: Icon(
                controller.paused || controller.canExport
                    ? Icons.play_arrow_rounded
                    : Icons.pause_rounded,
              ),
              label: Text(
                controller.canExport
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
            const Text('Playback'),
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
        Slider(
          key: const Key('seek'),
          value: position,
          max: maximum > 0 ? maximum.toDouble() : 1,
          label: '${(position / 44100).toStringAsFixed(1)} s',
          onChanged: maximum > 0 && (active || controller.canExport)
              ? (value) => setState(() {
                  _dragRequest = controller.speech?.id;
                  _seekPreview = value;
                })
              : null,
          onChangeEnd: (value) {
            if (_dragRequest == controller.speech?.id) {
              controller.seek(value.round());
            }
            setState(() {
              _seekPreview = null;
              _dragRequest = null;
            });
          },
        ),
        Text(
          state?.durationFrames != null
              ? 'Duration: ${(state!.durationFrames! / 44100).toStringAsFixed(1)} s'
              : state?.estimatedFrames != null
              ? 'Estimated duration: ${(state!.estimatedFrames! / 44100).toStringAsFixed(1)} s'
              : 'Duration becomes available as the text is prepared.',
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
              textDirection: controller.originalIsRtl
                  ? TextDirection.rtl
                  : TextDirection.ltr,
              style: const TextStyle(fontSize: 15, height: 1.6),
            ),
          ),
        ),
      ],
    );
  }
}
