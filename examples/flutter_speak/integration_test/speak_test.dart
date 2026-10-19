import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
// Native registry counts are diagnostic test-only evidence.
// ignore: implementation_imports
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:mlsr_speak/main.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_speak/speech_controller.dart';

int wavFrames(Uint8List bytes) {
  final view = ByteData.sublistView(bytes);
  for (var offset = 12; offset + 8 <= bytes.length;) {
    final length = view.getUint32(offset + 4, Endian.little);
    if (String.fromCharCodes(bytes.sublist(offset, offset + 4)) == 'data') {
      expect(offset + 8 + length, lessThanOrEqualTo(bytes.length));
      return length ~/ 4;
    }
    offset += 8 + length + (length & 1);
  }
  throw StateError('WAV has no data chunk');
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;
  testWidgets('public SDK speak, reader and export keep the UI responsive', (
    tester,
  ) async {
    const output = String.fromEnvironment('MLSR_TEST_OUTPUT');
    expect(output, isNotEmpty);
    final directory = Directory(output);
    expect(await directory.exists(), isFalse);
    await directory.create(recursive: true);
    final speech = DemoController(muted: true);
    final handles = native.mlsr_live_handles();
    await tester.pumpWidget(SpeakApp(controller: speech));
    await tester.pumpAndSettle();
    final clock = Stopwatch()..start();
    var lastTick = clock.elapsedMicroseconds, ticks = 0, maxGap = 0;
    final heartbeat = Timer.periodic(const Duration(milliseconds: 16), (_) {
      final now = clock.elapsedMicroseconds;
      if (now - lastTick > maxGap) maxGap = now - lastTick;
      lastTick = now;
      ticks++;
    });
    final report = <String, Object?>{
      'passed': false,
      'muted': true,
      'timing_boundary':
          'UI observation at 20 ms pump intervals; not acoustic or native flush timing',
    };
    Future<void> until(bool Function() done) async {
      final deadline = DateTime.now().add(const Duration(seconds: 90));
      while (!done()) {
        if (DateTime.now().isAfter(deadline)) {
          throw StateError('Timed out: ${speech.status} ${speech.error}');
        }
        await tester.pump(const Duration(milliseconds: 20));
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
      await tester.pump();
    }

    Future<void> screenshot(String name) async {
      await tester.pump();
      final surface = tester.renderObject<RenderRepaintBoundary>(
        find.byKey(const Key('surface')),
      );
      final image = await surface.toImage(pixelRatio: 1);
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
      await File('$output/$name.png').writeAsBytes(bytes!.buffer.asUint8List());
      image.dispose();
    }

    Future<void> tap(String key) async {
      final finder = find.byKey(Key(key));
      await tester.ensureVisible(finder);
      await tester.tap(finder);
      await tester.pump();
    }

    try {
      final loading = speech.load(
        delivery: const bool.fromEnvironment('MLSR_TEST_MATERIALIZE')
            ? ModelDelivery.installedCopy
            : ModelDelivery.automatic,
      );
      await until(() => !speech.loading);
      await loading;
      expect(speech.ready, isTrue, reason: speech.error);
      expect(ticks, greaterThan(5));
      await screenshot('ready');
      await tester.enterText(
        find.byKey(const Key('script')),
        List.filled(12, english).join('\n'),
      );
      final onset = Stopwatch()..start();
      await tap('speak');
      await until(() => speech.generatedFrames >= 8192 || speech.error != null);
      expect(speech.error, isNull);
      report['first_retained_pcm_observed_ms'] =
          onset.elapsedMicroseconds / 1000;
      // A long request has a whole-document scale before playback has advanced,
      // while segment admission and retained bytes obey their separate budgets.
      final earlyTimeline = speech.state!;
      expect(earlyTimeline.estimatedFrames, isNotNull);
      expect(earlyTimeline.durationFrames, isNull);
      expect(
        earlyTimeline.estimatedFrames!,
        greaterThan(earlyTimeline.bufferedUntilFrame),
      );
      final seek = tester.widget<Slider>(find.byKey(const Key('seek')));
      expect(seek.max, earlyTimeline.estimatedFrames!.toDouble());
      final progress = tester.widget<LinearProgressIndicator>(
        find.byKey(const Key('playback-progress')),
      );
      expect(
        progress.value,
        earlyTimeline.positionFrame / earlyTimeline.estimatedFrames!,
      );
      report['early_duration_estimate_frames'] = earlyTimeline.estimatedFrames;
      report['position_at_early_estimate_frames'] = earlyTimeline.positionFrame;
      report['buffer_at_early_estimate_frames'] =
          earlyTimeline.bufferedUntilFrame;
      await until(() => speech.playedFrames > 0 || speech.error != null);
      report['first_position_observed_ms'] = onset.elapsedMicroseconds / 1000;
      final retained = (await speech.engineState()).retention;
      expect(retained.usedBytes, lessThanOrEqualTo(128 * 1024 * 1024));
      report['retained_peak_bytes'] = retained.peakBytes;
      await screenshot('playing');
      final stopping = Stopwatch()..start();
      await tap('stop');
      await until(() => !speech.speaking);
      report['cancel_to_terminal_observed_ms'] =
          stopping.elapsedMicroseconds / 1000;
      expect(speech.state!.completion!.reason, CancelReason.requested);
      expect(speech.canExport, isFalse);
      expect(speech.error, isNull);

      // An admitted segment drains to retention even while the native output
      // is paused. Seeking past the old two-second boundary must work.
      speech.speak(
        'This sentence has enough words to take well over two seconds to read.\n'
            'The second sentence follows later.',
        'en',
        'M1',
        paused: true,
      );
      await until(
        () =>
            (speech.generatedFrames > 88200 && speech.paused) ||
            speech.error != null,
      );
      expect(speech.error, isNull);
      await tester.pump(const Duration(milliseconds: 300));
      final retainedSegment = speech.generatedFrames;
      expect(speech.playedFrames, 0);
      expect(retainedSegment, lessThan(speech.state!.estimatedFrames!));
      await tester.pump(const Duration(milliseconds: 150));
      expect(speech.generatedFrames, retainedSegment);
      expect(
        tester
            .widget<Slider>(find.byKey(const Key('seek')))
            .secondaryTrackValue,
        retainedSegment.toDouble(),
      );
      const beyondLookahead = 88200 + 4096;
      speech.seek(beyondLookahead);
      await until(
        () => speech.playedFrames == beyondLookahead || speech.error != null,
      );
      expect(speech.error, isNull);
      expect(speech.paused, isTrue);
      report['paused_retained_segment_frames'] = retainedSegment;
      report['segment_start_lookahead_frames'] = 88200;
      report['seek_beyond_lookahead_frame'] = speech.playedFrames;
      await screenshot('paused-retention');
      speech.stop();
      await until(() => !speech.speaking);

      // Native admission and replace behavior with rapid calls from the UI layer.
      final replaced = speech.speak(
        'Replace this request.',
        'en',
        'M1',
        paused: true,
      )!;
      final current = speech.speak(
        'Hello from MLSR.',
        'en',
        'M1',
        paused: true,
      )!;
      expect((await replaced.done).reason, CancelReason.replaced);
      final queued = speech.speak(
        'A queued sentence.',
        'en',
        'F1',
        mode: SpeakMode.enqueue,
      )!;
      queued.cancel();
      expect((await queued.done).reason, CancelReason.requested);
      await until(
        () =>
            (speech.generatedFrames > 8192 && speech.paused) ||
            speech.error != null,
      );
      expect(speech.error, isNull);
      final pausedPosition = speech.playedFrames;
      await tester.pump(const Duration(milliseconds: 150));
      expect(speech.playedFrames, pausedPosition);
      speech.setRate(1.5);
      speech.setVolume(.4);
      speech.seek(4096);
      await until(() => speech.playedFrames == 4096);
      expect(speech.state!.playbackRate, 1.5);
      await tap('pause');
      await until(() => !speech.generating || speech.error != null);
      report['generation_finished_while_playing'] = speech.speaking;
      await until(() => !speech.speaking);
      expect((await current.done).kind, OutcomeKind.completed);
      expect(speech.error, isNull);
      expect(speech.playedFrames, speech.generatedFrames);
      final exportPath = '$output/hello.wav';
      await speech.exportTo(exportPath);
      final wav = await File(exportPath).readAsBytes();
      expect(wavFrames(wav), speech.generatedFrames);
      report['wav_frames'] = speech.generatedFrames;
      await expectLater(
        speech.exportTo(exportPath),
        throwsA(isA<SynthesisException>()),
      );
      expect(await File(exportPath).readAsBytes(), wav);
      await screenshot('completed');
      // Completed seek is a new retained replay identity, not a rewritten outcome.
      final oldId = speech.speech!.id;
      speech.seek(4096);
      expect(speech.speech!.id, isNot(oldId));
      await until(() => speech.paused && speech.playedFrames == 4096);
      speech.togglePause();
      await until(() => !speech.speaking);

      final unsupported = speech.speak('안녕하세요 👩🏽‍💻.', 'ko', 'F1')!;
      await expectLater(
        unsupported.done,
        throwsA(
          isA<SynthesisException>().having(
            (error) => error.code,
            'code',
            'unsupported_character',
          ),
        ),
      );
      await until(() => speech.error != null);
      expect(speech.canExport, isFalse);
      report['unsupported_character_rejected'] = true;
      final koreanRun = speech.speak(
        '😀 안녕하세요.\nCafe\u0301 and hello.',
        'ko',
        'F1',
        paused: true,
      )!;
      await until(
        () =>
            (speech.generatedFrames > 8192 && speech.paused) ||
            speech.error != null,
      );
      expect(speech.error, isNull);
      final plan = (await koreanRun.plan())!;
      expect(plan.original, speech.original);
      for (final segment in plan.segments) {
        expect(
          plan.original.substring(segment.utf16.start, segment.utf16.end),
          segment.original,
        );
      }
      await screenshot('reader');
      speech.navigate(next: true);
      await until(
        () =>
            speech.state?.activeTextRange?.sentenceId == 1 ||
            speech.error != null,
      );
      expect(speech.error, isNull);
      speech.navigate(next: false);
      await until(
        () =>
            speech.state?.activeTextRange?.sentenceId == 0 ||
            speech.error != null,
      );
      speech.togglePause();
      await until(() => !speech.speaking);
      expect(speech.error, isNull);
      expect(speech.generatedFrames, greaterThan(44100));
      await speech.exportTo('$output/korean.wav');
      await speech.close();
      expect(native.mlsr_live_handles(), handles);
      report['handles_before'] = handles;
      report['handles_after'] = native.mlsr_live_handles();
      report['heartbeat_ticks'] = ticks;
      report['heartbeat_max_gap_ms'] = maxGap / 1000;
      expect(
        maxGap / 1000,
        lessThan(500),
        reason: 'UI isolate stalled for half a second',
      );
      report['passed'] = true;
    } catch (failure) {
      report['failure'] = failure.toString();
      report['failure_generated_frames'] = speech.generatedFrames;
      report['failure_played_frames'] = speech.playedFrames;
      rethrow;
    } finally {
      heartbeat.cancel();
      await speech.close();
      await File(
        '$output/report.json',
      ).writeAsString(const JsonEncoder.withIndent('  ').convert(report));
      await tester.pumpWidget(const SizedBox());
      speech.dispose();
    }
  });
}
