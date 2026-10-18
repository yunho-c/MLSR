import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:mlsr/mlsr.dart';
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:test/test.dart';

void main() {
  test('owned native errors and isolate cleanup without models', () async {
    final before = native.mlsr_live_handles();
    await expectLater(
      Engine.load(
        manifest: '/missing/mlsr-manifest.yaml',
        bundle: '/missing/mlsr-bundle',
      ),
      throwsA(
        isA<SynthesisException>().having((e) => e.code, 'code', 'io_error'),
      ),
    );
    expect(native.mlsr_live_handles(), before);
  });
  test(
    'real stream, paused consumer cancellation, reuse and owned audio',
    () async {
      final before = native.mlsr_live_handles();
      final engine = await Engine.load(
        manifest: Platform.environment['MLSR_TEST_MANIFEST']!,
        bundle: Platform.environment['MLSR_TEST_BUNDLE']!,
      );
      final output = Platform.environment['MLSR_TEST_OUTPUT'];
      if (output != null) {
        expect(Directory(output).existsSync(), isFalse);
        await Directory(output).create(recursive: true);
      }
      Float32List? retained;
      try {
        final stalled = await engine.start(
          'Hello from MLSR. Another sentence.',
          options: const SynthesisOptions(queueFrames: 1, chunkFrames: 1),
        );
        final iterator = StreamIterator(stalled.events);
        expect(await iterator.moveNext(), isTrue);
        expect(iterator.current.kind, EventKind.predicted);
        await expectLater(
          engine.start('Busy.'),
          throwsA(
            isA<SynthesisException>().having(
              (e) => e.code,
              'code',
              'engine_busy',
            ),
          ),
        );
        stalled.cancel();
        expect(await stalled.wait(timeoutMs: 30000), isTrue);
        expect(await iterator.moveNext(), isTrue);
        expect(iterator.current.kind, EventKind.cancelled);
        expect(await iterator.moveNext(), isFalse);
        await iterator.cancel();
        final cases = [('en', 'M1'), ('en', 'F1'), ('ko', 'M1'), ('ko', 'F1')];
        for (var index = 0; index < cases.length; index++) {
          final (language, voice) = cases[index];
          final text = language == 'en' ? 'Hello from MLSR.' : '안녕하세요. 반갑습니다.';
          final job = await engine.start(
            text,
            language: language,
            voice: voice,
            options: SynthesisOptions(seed: 1729 + index),
          );
          var frames = 0, terminals = 0;
          final pcm = BytesBuilder(copy: false);
          await for (final event in job.events) {
            expect(event.jobId, job.id);
            if (event.kind == EventKind.audio) {
              expect(event.startFrame, frames);
              expect(event.frames, lessThanOrEqualTo(4096));
              expect(event.sampleRate, 44100);
              expect(event.channels, 1);
              frames += event.frames;
              retained = event.samples;
              pcm.add(event.samples.buffer.asUint8List());
            }
            if (event.terminal) {
              terminals++;
              expect(event.kind, EventKind.completed);
              expect(event.frames, frames);
            }
          }
          expect(frames, greaterThan(44100));
          expect(terminals, 1);
          if (output != null) {
            await File(
              '$output/${language}_$voice.f32',
            ).writeAsBytes(pcm.takeBytes());
          }
        }
      } finally {
        await engine.dispose();
      }
      expect(retained, isNotEmpty);
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_BUNDLE'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );
}
