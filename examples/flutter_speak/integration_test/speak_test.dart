import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
// The diagnostic counts native registry entries; this is intentionally a test.
// ignore: implementation_imports
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:mlsr_speak/main.dart';
import 'package:mlsr_speak/speech_controller.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;
  testWidgets('real voices, responsive UI, bounded playback, stop/reuse and WAV', (
    tester,
  ) async {
    const manifest = String.fromEnvironment('MLSR_MANIFEST');
    const bundle = String.fromEnvironment('MLSR_BUNDLE');
    const output = String.fromEnvironment('MLSR_TEST_OUTPUT');
    expect(manifest, isNotEmpty);
    expect(bundle, isNotEmpty);
    expect(output, isNotEmpty);
    final directory = Directory(output);
    expect(await directory.exists(), isFalse);
    await directory.create(recursive: true);
    final speech = SpeechController(muted: true);
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
    final report = <String, Object?>{'passed': false, 'muted': true};
    Future<void> until(bool Function() done) async {
      final deadline = DateTime.now().add(const Duration(seconds: 90));
      while (!done()) {
        if (DateTime.now().isAfter(deadline)) {
          throw StateError('Timed out: ${speech.status} ${speech.error}');
        }
        await tester.pump(const Duration(milliseconds: 20));
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
      // Publish the final controller state before the next UI action. In
      // particular, stop makes the editor writable on the next Flutter frame.
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

    try {
      final loading = speech.load(manifest, bundle);
      await until(() => !speech.loading);
      await loading;
      expect(speech.ready, isTrue, reason: speech.error);
      expect(ticks, greaterThan(5));
      await screenshot('ready');
      await tester.enterText(
        find.byKey(const Key('script')),
        List.filled(12, english).join('\n'),
      );
      await tester.tap(find.byKey(const Key('speak')));
      await until(() => speech.generatedFrames >= 8192 || speech.error != null);
      expect(speech.error, isNull);
      final playing = await speech.player.state();
      expect(playing['queuedFrames'] as int, inInclusiveRange(1, 22050));
      expect(playing['peakQueuedFrames'] as int, lessThanOrEqualTo(22050));
      expect(playing['pendingFrames'] as int, lessThanOrEqualTo(4096));
      report['playing'] = playing;
      await screenshot('playing');
      final stopping = Stopwatch()..start();
      await tester.tap(find.byKey(const Key('stop')));
      // This query is queued directly after stop on the native audio queue;
      // no inference-worker wait or controller completion is needed to flush.
      final stopped = await speech.player.state();
      report['stop_to_playback_flush_ms'] = stopping.elapsedMicroseconds / 1000;
      expect(stopped['queuedFrames'], 0);
      expect(stopped['active'], isFalse);
      await until(() => !speech.speaking);
      report['stop_to_controller_idle_ms'] =
          stopping.elapsedMicroseconds / 1000;
      report['stopped'] = stopped;
      expect(speech.error, isNull);
      expect(
        speech.wav,
        isNull,
        reason: 'Stopped partial audio must not be exportable',
      );
      // Reuse the same engine and exercise real, muted AVAudioEngine playback to
      // its completion callback, separately from the synthesis terminal event.
      await tester.pumpAndSettle();
      expect(
        tester.widget<TextField>(find.byKey(const Key('script'))).readOnly,
        isFalse,
      );
      // A desktop button takes focus. showKeyboard caches the EditableText
      // identity, so tap the editor again just as a user would before typing.
      await tester.tap(find.byKey(const Key('script')));
      await tester.pump();
      await tester.enterText(
        find.byKey(const Key('script')),
        'Hello from MLSR.',
      );
      expect(
        tester
            .widget<TextField>(find.byKey(const Key('script')))
            .controller!
            .text,
        'Hello from MLSR.',
      );
      await tester.tap(find.byKey(const Key('speak')));
      await until(() => speech.wav != null || speech.error != null);
      expect(speech.error, isNull);
      report['generation_finished_while_playing'] = speech.speaking;
      await until(() => !speech.speaking);
      expect(speech.playedFrames, speech.generatedFrames);
      final exportPath = '$output/hello.wav';
      await speech.exportTo(exportPath);
      final wav = await File(exportPath).readAsBytes();
      expect(wav.length, 56 + speech.generatedFrames * 4);
      report['wav_frames'] = speech.generatedFrames;
      await expectLater(
        speech.exportTo(exportPath),
        throwsA(isA<FileSystemException>()),
      );
      expect(await File(exportPath).readAsBytes(), wav);
      await screenshot('completed');
      final koreanRun = speech.speak('안녕하세요. 반갑습니다.', 'ko', 'F1');
      await until(() => speech.speaking);
      await until(() => !speech.speaking);
      await koreanRun;
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
      report['failure_player'] = await speech.player.state();
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
