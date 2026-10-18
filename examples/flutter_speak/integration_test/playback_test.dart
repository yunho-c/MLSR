import 'dart:typed_data';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mlsr_speak/pcm_player.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'native playback drains, flushes a blocked producer and restarts',
    (tester) async {
      await tester.pumpWidget(const SizedBox());
      final player = PcmPlayer();
      final pcm = Float32List(4096);
      try {
        for (final epoch in [1, 3]) {
          await player.begin(epoch, muted: true);
          await Future<void>.delayed(const Duration(milliseconds: 100));
          for (var i = 0; i < 10; i++) {
            await player
                .enqueue(epoch, pcm)
                .timeout(const Duration(seconds: 3));
          }
          await player.finish(epoch).timeout(const Duration(seconds: 3));
          final state = await player.state();
          expect(state['queuedFrames'], 0);
          expect(state['playedFrames'], 40960);
          expect(state['active'], false);
        }
        await player.begin(4, muted: true);
        for (var i = 0; i < 5; i++) {
          await player.enqueue(4, pcm);
        }
        final blocked = expectLater(player.enqueue(4, pcm), throwsException);
        await player.stop(4);
        await blocked;
        final state = await player.state();
        expect(state['queuedFrames'], 0);
        expect(state['pendingFrames'], 0);
        expect(state['active'], false);
        await player.begin(6, muted: true);
        await player.enqueue(6, pcm);
        await player.finish(6).timeout(const Duration(seconds: 3));
        expect((await player.state())['playedFrames'], 4096);
      } catch (failure) {
        // Keep real device state in the test log if the host audio route stalls.
        debugPrint('Playback failure: $failure; ${await player.state()}');
        rethrow;
      } finally {
        final state = await player.state();
        await player.stop(state['epoch'] as int);
      }
    },
  );
}
