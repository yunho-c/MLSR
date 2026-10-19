import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:mlsr/mlsr.dart';
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:test/test.dart';
import 'model_fixture.dart';

ComputePolicy get _testPolicy =>
    Platform.environment['MLSR_TEST_POLICY'] == 'qnn_htp'
    ? ComputePolicy.qnnHtp
    : ComputePolicy.cpuAndNeuralEngine;

void main() {
  test(
    'real headless SDK retains store lease during model disposal',
    () async {
      final before = native.mlsr_live_handles();
      final root = await Directory.systemTemp.createTemp('mlsr-sdk-store-');
      final store = ModelStore(root.path);
      final descriptor = ModelDescriptor(
        catalogYaml: await File(
          Platform.environment['MLSR_TEST_DELIVERY_CATALOG']!,
        ).readAsString(),
        id:
            Platform.environment['MLSR_TEST_DELIVERY_ID'] ??
            'supertonic3-coreml-int4-v1',
        platform:
            Platform.environment['MLSR_TEST_DELIVERY_PLATFORM'] ??
            'macos-arm64',
      );
      final transaction = await store.begin(descriptor);
      Tts? tts;
      try {
        final source = Directory(
          Platform.environment['MLSR_TEST_DELIVERY_BUNDLE']!,
        );
        await for (final entry in source.list(recursive: true)) {
          if (entry is File) {
            final relative = entry.path.substring(source.path.length + 1);
            final target = File('${transaction.directory}/$relative');
            await target.parent.create(recursive: true);
            await entry.copy(target.path);
          }
        }
        final model = await transaction.commit();
        final opening = Tts.fromModel(
          model,
          computePolicy: _testPolicy,
          options: const TtsOptions(output: OutputMode.headless),
        );
        final disposing = model.dispose();
        tts = await opening;
        await disposing;
        final voices = await tts.getVoices();
        expect(voices.first.readiness, VoiceReadiness.engineReady);
        if (_testPolicy == ComputePolicy.qnnHtp) {
          expect(voices.first.backend, 'qnn');
          expect(voices.first.qualifiedLanguages, isEmpty);
        }
        await expectLater(
          store.remove(descriptor),
          throwsA(
            isA<SynthesisException>().having(
              (e) => e.code,
              'code',
              'model_in_use',
            ),
          ),
        );
        final clip = await tts.render('Hello from MLSR.');
        expect(clip.frames, greaterThan(44100));
        await tts.dispose();
        await store.remove(descriptor);
        expect(clip.frames, greaterThan(44100));
      } finally {
        await tts?.dispose();
        await transaction.dispose();
        await root.delete(recursive: true);
      }
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_DELIVERY_CATALOG'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );
  test(
    'catalog, verified readiness, explicit resolution and owned WAV',
    () async {
      final before = native.mlsr_live_handles();
      final voices = await getVoices();
      expect(voices.map((v) => v.id), ['supertonic3:M1', 'supertonic3:F1']);
      expect(voices.first.readiness, VoiceReadiness.catalogOnly);
      expect(voices.first.capabilities.wordAlignment, isFalse);
      await expectLater(
        resolveVoice('en-US'),
        throwsA(isA<SynthesisException>()),
      );
      final resolved = await resolveVoice(
        'en-US',
        voice: 'supertonic3:F1',
        fallback: VoiceFallback.sameLanguage,
      );
      expect(resolved.language, 'en');
      expect(resolved.usedFallback, isTrue);
      final fixture = await ModelFixture.create();
      try {
        final model = await ModelHandle.openDirectory(
          model: fixture.model,
          directory: fixture.source.path,
        );
        try {
          expect(
            (await model.getVoices()).first.readiness,
            VoiceReadiness.assetsVerified,
          );
        } finally {
          await model.dispose();
        }
      } finally {
        await fixture.dispose();
      }
      final root = await Directory.systemTemp.createTemp('mlsr-sdk-wav-');
      try {
        final clip = AudioClip(
          Float32List.fromList([0, -0.5, 0.75]),
          const AudioFormat(sampleRate: 44100, channels: 1),
        );
        final file = await clip.writeWav('${root.path}/clip.wav');
        expect(file.frames, 3);
        final bytes = await File(file.path).readAsBytes();
        await expectLater(
          clip.writeWav(file.path),
          throwsA(isA<SynthesisException>()),
        );
        expect(await File(file.path).readAsBytes(), bytes);
        expect(await root.list().length, 1);
      } finally {
        await root.delete(recursive: true);
      }
      await expectLater(
        Tts.open(
          manifest: '/missing',
          bundle: '/missing',
          options: const TtsOptions(maxRequests: 0),
        ),
        throwsA(isA<SynthesisException>()),
      );
      expect(native.mlsr_live_handles(), before);
    },
  );

  test(
    'real lazy jobs, coalesced states, owned PCM and shared file parity',
    () async {
      final before = native.mlsr_live_handles();
      final tts = await Tts.open(
        manifest: Platform.environment['MLSR_TEST_MANIFEST']!,
        bundle: Platform.environment['MLSR_TEST_BUNDLE']!,
        computePolicy: _testPolicy,
        options: const TtsOptions(output: OutputMode.headless),
      );
      const text = 'Hello from MLSR.\nA second sentence.';
      const options = RequestOptions(backend: SupertonicOptions(seed: 1729));
      try {
        expect(
          (await tts.getVoices()).first.readiness,
          VoiceReadiness.engineReady,
        );
        expect((await tts.snapshot()).outputCapabilities, isNull);
        final job = tts.synthesize(text, options: options);
        try {
          expect((await job.snapshot()).phase, SynthesisPhase.created);
          final request = await job.request();
          expect(request.text, text);
          expect(request.effectiveSpeed, closeTo(1.05, 1e-6));
          expect(request.requestId, job.id);
          expect(request.backend.seed, 1729);
          final future = job.done;
          await expectLater(
            future.timeout(const Duration(milliseconds: 40)),
            throwsA(isA<TimeoutException>()),
          );
          expect(await job.plan(), isNull);
          final states = StreamIterator(job.states);
          expect(await states.moveNext(), isTrue);
          expect(states.current.phase, SynthesisPhase.created);
          final pcm = BytesBuilder(copy: false);
          var frames = 0;
          final audio = StreamIterator(job.audio);
          try {
            expect(await audio.moveNext(), isTrue);
            await expectLater(
              job.audio.first,
              throwsA(isA<SynthesisException>()),
            );
            do {
              final chunk = audio.current;
              expect(chunk.startFrame, frames);
              expect(chunk.requestId, job.id);
              expect(chunk.textRange.text(text), isNotEmpty);
              frames += chunk.audio.frames;
              pcm.add(chunk.audio.samples.buffer.asUint8List());
            } while (await audio.moveNext());
          } finally {
            await audio.cancel();
          }
          final done = await future;
          expect(done.kind, OutcomeKind.completed);
          expect(done.frames, frames);
          expect(await states.moveNext(), isTrue);
          expect(states.current.completion?.kind, OutcomeKind.completed);
          expect(await states.moveNext(), isFalse);
          await states.cancel();
          final clip = await tts.render(text, options: options);
          expect(clip.samples.buffer.asUint8List(), pcm.takeBytes());
          final root = await Directory.systemTemp.createTemp(
            'mlsr-sdk-real-wav-',
          );
          try {
            final a = await tts.renderToFile(
              text,
              '${root.path}/render.wav',
              options: options,
            );
            final b = await clip.writeWav('${root.path}/clip.wav');
            expect(
              await File(a.path).readAsBytes(),
              await File(b.path).readAsBytes(),
            );
          } finally {
            await root.delete(recursive: true);
          }
          await expectLater(
            tts.render(text, options: options, maxBytes: 4),
            throwsA(
              isA<SynthesisException>()
                  .having((e) => e.stage, 'stage', 'rendering')
                  .having((e) => e.requestId, 'request', isNotNull),
            ),
          );
        } finally {
          await job.dispose();
        }
        expect(() => tts.speak(text), throwsA(isA<SynthesisException>()));
      } finally {
        await tts.dispose();
      }
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_BUNDLE'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );

  test(
    'real UTF-16 plans, consumer drop and dispose with pending done',
    () async {
      final before = native.mlsr_live_handles();
      final tts = await Tts.open(
        manifest: Platform.environment['MLSR_TEST_MANIFEST']!,
        bundle: Platform.environment['MLSR_TEST_BUNDLE']!,
        computePolicy: _testPolicy,
        options: const TtsOptions(output: OutputMode.headless),
      );
      try {
        const text = '😀 안녕하세요.\nCafe\u0301 and hello.';
        final job = tts.synthesize(
          text,
          options: const RequestOptions(locale: 'ko'),
        );
        try {
          final iterator = StreamIterator(job.audio);
          expect(await iterator.moveNext(), isTrue);
          final plan = (await job.plan())!;
          expect(plan.original, text);
          for (final segment in plan.segments) {
            expect(
              text.substring(segment.utf16.start, segment.utf16.end),
              segment.original,
            );
            expect(
              utf8.decode(
                utf8.encode(text).sublist(segment.utf8.start, segment.utf8.end),
              ),
              segment.original,
            );
          }
          final done = job.done;
          await iterator.cancel();
          expect((await done).reason, CancelReason.consumerDropped);
        } finally {
          await job.dispose();
        }
        final lazy = tts.synthesize('Never start this.');
        final waiting = lazy.done;
        await lazy.dispose();
        expect((await waiting).reason, CancelReason.requested);
        expect(
          (await lazy.snapshot()).completion?.reason,
          CancelReason.requested,
        );
        final closing = tts.synthesize('Owner closes before work starts.');
        final closingDone = closing.done;
        await tts.dispose();
        expect((await closingDone).reason, CancelReason.closed);
        await closing.dispose();
      } finally {
        await tts.dispose();
      }
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_BUNDLE'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );

  test(
    'real native reader, retained export and completed replay',
    () async {
      final before = native.mlsr_live_handles();
      final tts = await Tts.open(
        manifest: Platform.environment['MLSR_TEST_MANIFEST']!,
        bundle: Platform.environment['MLSR_TEST_BUNDLE']!,
        computePolicy: _testPolicy,
      );
      const paused = SpeechOptions(paused: true, volume: 0);
      try {
        expect((await tts.snapshot()).outputCapabilities!.pauseResume, isTrue);
        final speech = tts.controller.speak(
          'Hello from MLSR.',
          options: paused,
          request: const RequestOptions(backend: SupertonicOptions(seed: 1729)),
        );
        try {
          final state = await speech.states
              .firstWhere((s) => s.durationFrames != null)
              .timeout(const Duration(seconds: 10));
          expect(state.playback, PlaybackPhase.paused);
          expect(state.positionFrame, 0);
          expect(state.completion, isNull);
          final queued = tts.speak(
            'Queued.',
            options: paused,
            mode: SpeakMode.enqueue,
          );
          queued.cancel();
          expect((await queued.done).reason, CancelReason.requested);
          await queued.dispose();
          speech.setVolume(0);
          if ((await tts.snapshot()).outputCapabilities!.rateRange == null) {
            expect(
              () => speech.setPlaybackRate(2),
              throwsA(isA<SynthesisException>()),
            );
            speech.setPlaybackRate(1);
          } else {
            speech.setPlaybackRate(2);
          }
          speech.seek(0);
          speech.resume();
          final done = await speech.done.timeout(const Duration(seconds: 10));
          expect(done.kind, OutcomeKind.completed);
          expect(
            (await speech.snapshot()).activeTextRange!.text('Hello from MLSR.'),
            'Hello from MLSR.',
          );
          final root = await Directory.systemTemp.createTemp(
            'mlsr-sdk-recording-',
          );
          try {
            expect(
              (await tts.writeSpeech(
                speech,
                '${root.path}/recording.wav',
              )).frames,
              done.frames,
            );
          } finally {
            await root.delete(recursive: true);
          }
          final replay = speech.replayFrom(done.frames ~/ 2, options: paused);
          try {
            expect(replay.id, isNot(speech.id));
            final ready = await replay.states
                .firstWhere((s) => s.playback == PlaybackPhase.paused)
                .timeout(const Duration(seconds: 5));
            expect(ready.positionFrame, done.frames ~/ 2);
            replay.cancel();
            expect((await replay.done).reason, CancelReason.requested);
            expect((await speech.done).kind, OutcomeKind.completed);
          } finally {
            await replay.dispose();
          }
        } finally {
          await speech.dispose();
        }
        expect((await tts.snapshot()).retention.usedBytes, 0);
      } finally {
        await tts.dispose();
      }
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_NATIVE_OUTPUT'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );
  test(
    'real SDK cross-binding export corpus',
    () async {
      final root = Directory(Platform.environment['MLSR_SDK_OUTPUT']!);
      expect(await root.exists(), isFalse);
      await root.create();
      final tts = await Tts.open(
        manifest: Platform.environment['MLSR_TEST_MANIFEST']!,
        bundle: Platform.environment['MLSR_TEST_BUNDLE']!,
        computePolicy: _testPolicy,
        options: const TtsOptions(output: OutputMode.headless),
      );
      try {
        final cases = [('en', 'M1'), ('en', 'F1'), ('ko', 'M1'), ('ko', 'F1')];
        for (var i = 0; i < cases.length; i++) {
          final (language, voice) = cases[i];
          final file = await tts.renderToFile(
            language == 'en' ? 'Hello from MLSR.' : '안녕하세요. 반갑습니다.',
            '${root.path}/${language}_$voice.wav',
            options: RequestOptions(
              locale: language,
              voice: 'supertonic3:$voice',
              backend: SupertonicOptions(seed: 1729 + i),
            ),
          );
          expect(file.frames, greaterThan(44100));
        }
      } finally {
        await tts.dispose();
      }
    },
    skip: Platform.environment['MLSR_SDK_OUTPUT'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );
}
