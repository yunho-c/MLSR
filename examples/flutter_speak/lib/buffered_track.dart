import 'package:flutter/material.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';

/// Paint actual cached intervals without filling the ungenerated gaps between
/// them. Flutter's built-in secondary track represents only one prefix.
class BufferedTrack extends RoundedRectSliderTrackShape {
  final List<IndexRange> ranges;
  final int total;
  const BufferedTrack(this.ranges, this.total);

  @override
  void paint(
    PaintingContext context,
    Offset offset, {
    required RenderBox parentBox,
    required SliderThemeData sliderTheme,
    required Animation<double> enableAnimation,
    required TextDirection textDirection,
    required Offset thumbCenter,
    Offset? secondaryOffset,
    bool isDiscrete = false,
    bool isEnabled = false,
    double additionalActiveTrackHeight = 2,
  }) {
    super.paint(
      context,
      offset,
      parentBox: parentBox,
      sliderTheme: sliderTheme,
      enableAnimation: enableAnimation,
      textDirection: textDirection,
      thumbCenter: thumbCenter,
      isDiscrete: isDiscrete,
      isEnabled: isEnabled,
      additionalActiveTrackHeight: additionalActiveTrackHeight,
    );
    if (total <= 0 || !isEnabled) return;
    final rect = getPreferredRect(
      parentBox: parentBox,
      offset: offset,
      sliderTheme: sliderTheme,
      isEnabled: isEnabled,
      isDiscrete: isDiscrete,
    );
    final paint = Paint()
      ..color =
          sliderTheme.secondaryActiveTrackColor ?? const Color(0xffaab3aa);
    final rtl = textDirection == TextDirection.rtl;
    double x(int frame) {
      final fraction = (frame / total).clamp(0.0, 1.0);
      return rect.left + (rtl ? 1 - fraction : fraction) * rect.width;
    }

    context.canvas.save();
    context.canvas.clipRRect(
      RRect.fromRectAndRadius(rect, Radius.circular(rect.height / 2)),
    );
    // Preserve the active position indicator, including during drag previews.
    context.canvas.clipRect(
      Rect.fromLTRB(
        rtl ? rect.left : thumbCenter.dx,
        rect.top,
        rtl ? thumbCenter.dx : rect.right,
        rect.bottom,
      ),
    );
    for (final range in ranges) {
      context.canvas.drawRect(
        Rect.fromLTRB(
          rtl ? x(range.end) : x(range.start),
          rect.top,
          rtl ? x(range.start) : x(range.end),
          rect.bottom,
        ),
        paint,
      );
    }
    context.canvas.restore();
  }
}
