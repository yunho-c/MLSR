import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_app_models/app_models.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'public controller owns queued cancellation and retained replay',
    (tester) async {
      await tester.pumpWidget(const SizedBox());
      final tts = await FlutterMlsr.openTts(model: AppModels.supertonic3);
      Speech? first, queued, replay;
      try {
        first = tts.speak(
          'Hello from MLSR. Another sentence follows.',
          options: const SpeechOptions(paused: true, volume: 0),
        );
        queued = tts.speak(
          'This queued request is cancelled.',
          mode: SpeakMode.enqueue,
          options: const SpeechOptions(volume: 0),
        );
        queued.cancel();
        expect((await queued.done).reason, CancelReason.requested);
        final deadline = DateTime.now().add(const Duration(seconds: 60));
        while ((await first.snapshot()).bufferedUntilFrame < 8192) {
          if (DateTime.now().isAfter(deadline)) {
            throw StateError('No retained audio');
          }
          await Future<void>.delayed(const Duration(milliseconds: 20));
        }
        first.seek(4096);
        first.setPlaybackRate(1.5);
        first.resume();
        expect((await first.done).kind, OutcomeKind.completed);
        final completed = await first.snapshot();
        expect(completed.positionFrame, completed.durationFrames);
        replay = first.replayFrom(
          4096,
          options: const SpeechOptions(volume: 0, playbackRate: 2),
        );
        expect(replay.id, isNot(first.id));
        expect((await replay.done).kind, OutcomeKind.completed);
        expect((await first.done).kind, OutcomeKind.completed);
      } finally {
        await tts.dispose();
        await replay?.dispose();
        await queued?.dispose();
        await first?.dispose();
      }
    },
  );
}
