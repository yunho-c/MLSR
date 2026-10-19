import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_speak/main.dart';
import 'package:mlsr_speak/speech_controller.dart';
import 'package:mlsr_speak/speech_timeline.dart';

class _Range implements IndexRange {
  @override
  final int start, end;
  _Range(this.start, this.end);
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Snapshot implements SpeechState {
  @override
  final int requestId, positionFrame, bufferedUntilFrame;
  @override
  final int? estimatedFrames, durationFrames;
  @override
  final AudioFormat format;
  @override
  final List<IndexRange> seekableRanges;
  @override
  final PlaybackPhase playback;
  _Snapshot({
    this.requestId = 1,
    this.positionFrame = 8 * 44100,
    this.bufferedUntilFrame = 10 * 44100,
    this.estimatedFrames = 60 * 44100,
    this.durationFrames,
    this.format = const AudioFormat(sampleRate: 44100, channels: 1),
    this.playback = PlaybackPhase.playing,
    List<IndexRange>? ranges,
  }) : seekableRanges = ranges ?? [_Range(0, bufferedUntilFrame)];
  @override
  SpeechTextRange? get activeTextRange => null;
  @override
  Completion? get completion => null;
  @override
  SynthesisPhase get synthesis => SynthesisPhase.synthesizing;
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Controller extends DemoController {
  SpeechState current;
  final seeks = <int>[];
  _Controller(this.current) {
    ready = true;
  }
  @override
  SpeechState get state => current;
  @override
  bool get speaking => true;
  @override
  bool get canExport => false;
  @override
  String get original => 'A long reading with audio still being prepared.';
  @override
  String get status => 'Reading aloud';
  @override
  void seek(int frame) => seeks.add(frame);
  void update(SpeechState value) {
    current = value;
    notifyListeners();
  }
}

Future<_Controller> _showPlayer(WidgetTester tester, SpeechState state) async {
  tester.view.physicalSize = const Size(1280, 1100);
  tester.view.devicePixelRatio = 1;
  final controller = _Controller(state);
  addTearDown(() async {
    await tester.pumpWidget(const SizedBox());
    controller.dispose();
    tester.view.resetPhysicalSize();
    tester.view.resetDevicePixelRatio();
  });
  await tester.pumpWidget(SpeakApp(controller: controller));
  return controller;
}

Slider seekSlider(WidgetTester tester) =>
    tester.widget<Slider>(find.byKey(const Key('seek')));
LinearProgressIndicator playedProgress(WidgetTester tester) =>
    tester.widget<LinearProgressIndicator>(
      find.byKey(const Key('playback-progress')),
    );

void main() {
  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    for (final family in ['Manrope', 'Fraunces']) {
      await (FontLoader(
        family,
      )..addFont(rootBundle.load('assets/fonts/$family.ttf'))).load();
    }
  });
  testWidgets(
    'buffer appends never rescale either control or the displayed total',
    (tester) async {
      final controller = await _showPlayer(tester, _Snapshot());
      var slider = seekSlider(tester);
      expect(slider.max, 60 * 44100);
      expect(slider.value / slider.max, closeTo(8 / 60, 1e-9));
      expect(slider.secondaryTrackValue, 10 * 44100);
      expect(playedProgress(tester).value, closeTo(8 / 60, 1e-9));
      expect(find.text('~1:00'), findsOneWidget);
      expect(find.text('Played 0:08 / ~1:00'), findsOneWidget);
      final before = slider.value / slider.max;
      controller.update(
        _Snapshot(positionFrame: 354564, bufferedUntilFrame: 445096),
      );
      await tester.pump();
      slider = seekSlider(tester);
      expect(slider.max, 60 * 44100);
      expect(slider.value / slider.max, greaterThan(before));
      expect(playedProgress(tester).value, greaterThan(before));
      expect(slider.secondaryTrackValue, 445096);
      expect(find.text('~1:00'), findsOneWidget);
    },
  );

  testWidgets(
    'unknown duration stays indeterminate even when audio is available',
    (tester) async {
      await _showPlayer(tester, _Snapshot(estimatedFrames: null));
      expect(playedProgress(tester).value, isNull);
      expect(seekSlider(tester).onChanged, isNull);
      expect(find.text('—'), findsOneWidget);
      expect(find.textContaining('Estimating duration'), findsOneWidget);
    },
  );

  testWidgets(
    'final duration overrides the estimate without following retention or rate',
    (tester) async {
      final controller = await _showPlayer(
        tester,
        _Snapshot(durationFrames: 58 * 44100),
      );
      expect(seekSlider(tester).max, 58 * 44100);
      expect(playedProgress(tester).value, closeTo(8 / 58, 1e-9));
      expect(find.text('0:58'), findsOneWidget);
      controller.setRate(2.0);
      await tester.pump();
      expect(seekSlider(tester).max, 58 * 44100);
      expect(playedProgress(tester).value, closeTo(8 / 58, 1e-9));
    },
  );

  testWidgets(
    'retained endpoint is seekable; unavailable positions leave playback alone',
    (tester) async {
      final controller = await _showPlayer(
        tester,
        _Snapshot(playback: PlaybackPhase.paused),
      );
      var slider = seekSlider(tester);
      slider.onChangeStart!(slider.value);
      slider.onChanged!(25 * 44100);
      await tester.pump();
      slider = seekSlider(tester);
      expect(slider.label, contains('Not ready'));
      slider.onChangeEnd!(25 * 44100);
      await tester.pump();
      expect(controller.seeks, isEmpty);
      expect(seekSlider(tester).value, 8 * 44100);
      expect(
        find.text('Audio at this position is not ready yet.'),
        findsOneWidget,
      );
      slider = seekSlider(tester);
      slider.onChangeStart!(slider.value);
      slider.onChanged!(10 * 44100);
      await tester.pump();
      seekSlider(tester).onChangeEnd!(10 * 44100);
      await tester.pump();
      expect(controller.seeks, [10 * 44100]);
      expect(controller.paused, isTrue);
    },
  );

  testWidgets('replacement during a drag cannot seek the new request', (
    tester,
  ) async {
    final controller = await _showPlayer(tester, _Snapshot());
    seekSlider(tester).onChangeStart!(8 * 44100);
    seekSlider(tester).onChanged!(5 * 44100);
    await tester.pump();
    controller.update(_Snapshot(requestId: 2, positionFrame: 0));
    await tester.pump();
    expect(seekSlider(tester).value, 0);
    // Further movement from the same held pointer must not re-arm the drag
    // against the replacement request.
    seekSlider(tester).onChanged!(6 * 44100);
    await tester.pump();
    expect(seekSlider(tester).value, 0);
    seekSlider(tester).onChangeEnd!(5 * 44100);
    await tester.pump();
    expect(controller.seeks, isEmpty);
  });

  test(
    'source sample rate and exact retained intervals govern presentation',
    () {
      final timeline = SpeechTimeline(
        _Snapshot(
          format: const AudioFormat(sampleRate: 22050, channels: 1),
          positionFrame: 8 * 22050,
          estimatedFrames: 60 * 22050,
          ranges: [_Range(0, 100), _Range(200, 300)],
        ),
      );
      expect(timeline.positionLabel, '0:08');
      expect(timeline.totalLabel, '~1:00');
      expect(timeline.canSeek(100), isTrue);
      expect(timeline.canSeek(150), isFalse);
      expect(timeline.canSeek(301), isFalse);
    },
  );
}
