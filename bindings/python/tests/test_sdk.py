import array
import asyncio
import gc
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from mlsr import (
    AudioClip,
    AudioFormat,
    Request,
    SpeechOptions,
    SupertonicOptions,
    Tts,
    TtsError,
    TtsOptions,
    resolve_voice,
    voices,
)
from mlsr import _sdk as ffi
from mlsr.sdk import _offload


class MetadataAndExport(unittest.TestCase):
    def test_catalog_resolution_and_no_clobber(self):
        lib = ffi.library()
        before = lib.mlsr_live_handles()
        catalog = voices()
        self.assertEqual([v.id for v in catalog], ["supertonic3:M1", "supertonic3:F1"])
        self.assertEqual(catalog[0].readiness, "catalog_only")
        self.assertFalse(catalog[0].capabilities.word_alignment)
        with self.assertRaises(TtsError):
            resolve_voice("en-US")
        resolved = resolve_voice("en-US", "supertonic3:F1", fallback="same_language")
        self.assertEqual(resolved.language, "en")
        self.assertTrue(resolved.used_fallback)
        clip = AudioClip(array.array("f", [0.0, -0.5, 0.75]), AudioFormat(44100, 1))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "clip.wav"
            file = clip.write_wav(path)
            self.assertEqual(file.frames, 3)
            self.assertEqual(file.format, clip.format)
            content = path.read_bytes()
            with self.assertRaises(TtsError):
                clip.write_wav(path)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(Path(root).iterdir()), [path])
        self.assertEqual(lib.mlsr_live_handles(), before)

    def test_options_fail_before_loading(self):
        with self.assertRaises(ValueError):
            Tts("/missing", "/missing", options=TtsOptions(max_retained_bytes=-1))
        with self.assertRaises(TtsError):
            Tts("/missing", "/missing", options=TtsOptions(max_requests=0))


class AsyncOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_joins_and_disposes_a_late_result(self):
        started, release = threading.Event(), threading.Event()
        disposed = []

        def work():
            started.set()
            release.wait(3)
            return "late owner"

        task = asyncio.create_task(
            _offload(work, on_cancel=release.set, discard=disposed.append)
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(disposed, ["late owner"])


@unittest.skipUnless(
    os.environ.get("MLSR_TEST_BUNDLE"), "real model assets not requested"
)
class RealDeveloperAPI(unittest.TestCase):
    def setUp(self):
        self.lib = ffi.library()
        gc.collect()
        self.before = self.lib.mlsr_live_handles()
        self.tts = Tts(
            os.environ["MLSR_TEST_MANIFEST"],
            os.environ["MLSR_TEST_BUNDLE"],
            compute_policy=os.environ.get("MLSR_TEST_POLICY", "cpu_and_neural_engine"),
            options=TtsOptions(output_mode="headless"),
        )
        self.request = Request(
            "Hello from MLSR.\nA second sentence.", backend=SupertonicOptions(seed=1729)
        )

    def tearDown(self):
        self.tts.close()
        gc.collect()
        self.assertEqual(self.lib.mlsr_live_handles(), self.before)

    def test_lazy_single_consumer_states_pcm_and_file_parity(self):
        self.assertEqual(self.tts.voices[0].readiness, "engine_ready")
        backend = "qnn" if os.environ.get("MLSR_TEST_POLICY") == "qnn_htp" else "coreml"
        self.assertEqual(self.tts.voices[0].backend, backend)
        if backend == "qnn":
            self.assertFalse(self.tts.voices[0].qualified_languages)
            with self.assertRaises(TtsError) as invalid:
                self.tts.synthesize(
                    Request("Hello.", backend=SupertonicOptions(steps=7))
                )
            self.assertEqual(invalid.exception.code, "invalid_options")
        self.assertIsNone(self.tts.output_capabilities)
        with self.tts.synthesize(self.request) as job:
            self.assertEqual(job.snapshot.phase, "created")
            self.assertEqual(job.request.text, self.request.text)
            self.assertEqual(job.request.request_id, job.id)
            self.assertEqual(job.request.speaking_rate, 1.0)
            self.assertAlmostEqual(job.request.effective_synthesis.speed, 1.05)
            self.assertEqual(job.request.effective_synthesis.seed, 1729)
            self.assertIsNone(job.wait(0.01))
            self.assertIsNone(job.plan)
            with job.states() as states:
                self.assertEqual(next(states).phase, "created")
                with job.audio() as stream:
                    with self.assertRaises(TtsError):
                        job.audio()
                    samples = array.array("f")
                    for chunk in stream:
                        self.assertEqual(chunk.request_id, job.id)
                        self.assertEqual(chunk.start_frame, len(samples))
                        self.assertEqual(
                            chunk.text_range.text(self.request.text),
                            job.plan.segments[chunk.text_range.segment_id].original,
                        )
                        samples.extend(chunk.audio.samples)
                    self.assertEqual(stream.completion.outcome.kind, "completed")
                done = job.wait(1)
                self.assertEqual(done.frames, len(samples))
                self.assertEqual(next(states).completion, done)
                with self.assertRaises(StopIteration):
                    next(states)
        clip = self.tts.render(self.request)
        self.assertEqual(clip.samples.tobytes(), samples.tobytes())
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "render.wav", Path(root) / "clip.wav"
            self.tts.render_to_file(self.request, a)
            clip.write_wav(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
        with self.assertRaises(TtsError) as failure:
            self.tts.render(self.request, max_bytes=4)
        self.assertEqual(failure.exception.stage, "rendering")
        self.assertEqual(failure.exception.backend, backend)
        self.assertIsNotNone(failure.exception.request_id)
        with self.assertRaises(TtsError):
            self.tts.speak("headless output is unavailable")

    @unittest.skipUnless(
        os.environ.get("MLSR_SDK_OUTPUT"), "cross-binding corpus not requested"
    )
    def test_sdk_export_corpus(self):
        root = Path(os.environ["MLSR_SDK_OUTPUT"])
        root.mkdir()
        for index, (language, voice) in enumerate(
            [("en", "M1"), ("en", "F1"), ("ko", "M1"), ("ko", "F1")]
        ):
            request = Request(
                "Hello from MLSR." if language == "en" else "안녕하세요. 반갑습니다.",
                locale=language,
                voice="supertonic3:" + voice,
                backend=SupertonicOptions(seed=1729 + index),
            )
            result = self.tts.render_to_file(request, root / f"{language}_{voice}.wav")
            self.assertGreater(result.frames, 44100)

    def test_abandoned_consumer_cancel_and_async_render(self):
        with self.tts.synthesize(self.request) as job:
            with job.audio() as stream:
                self.assertGreater(next(stream).audio.frames, 0)
            self.assertEqual(job.wait(1).outcome.reason, "consumer_dropped")

        async def exercise():
            async with self.tts.synthesize(self.request) as job:
                async with job.audio() as audio:
                    count = 0
                    async for chunk in audio:
                        count += chunk.audio.frames
                self.assertEqual((await job.wait_async()).frames, count)
            clip = await self.tts.arender(self.request)
            self.assertGreater(clip.frames, 0)
            async with self.tts.synthesize(self.request) as lazy:
                waiting = asyncio.create_task(lazy.wait_async())
                await asyncio.sleep(0.01)
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
                self.assertEqual(lazy.snapshot.completion.outcome.reason, "requested")

        asyncio.run(exercise())


@unittest.skipUnless(
    os.environ.get("MLSR_TEST_NATIVE_OUTPUT"), "explicit native playback not requested"
)
class RealNativeSpeech(unittest.TestCase):
    def test_shared_reader_controls_retention_and_fire_and_forget(self):
        lib = ffi.library()
        before = lib.mlsr_live_handles()
        with Tts(
            os.environ["MLSR_TEST_MANIFEST"], os.environ["MLSR_TEST_BUNDLE"]
        ) as tts:
            self.assertTrue(tts.output_capabilities.pause_resume)
            paused = SpeechOptions(paused=True, volume=0.0)
            # Long enough for normal native playback, short enough to finish in 8s.
            request = Request("Hello from MLSR.", backend=SupertonicOptions(seed=1729))
            with tts.controller.speak(request, options=paused) as speech:
                self.assertEqual(speech.id, speech.snapshot.request_id)
                with speech.states() as states:
                    end = time.monotonic() + 10
                    while True:
                        state = states.poll(100)
                        if state and state.duration_frames is not None:
                            break
                        self.assertLess(time.monotonic(), end)
                self.assertIsNone(speech.wait(0))
                with tts.controller.speak(
                    request, options=paused, mode="enqueue"
                ) as queued:
                    queued.cancel()
                    self.assertEqual(queued.wait(1).outcome.reason, "requested")
                speech.set_volume(0)
                speech.set_playback_rate(2)
                speech.seek(0)
                speech.resume()
                done = speech.wait(8)
                self.assertIsNotNone(done)
                self.assertEqual(done.outcome.kind, "completed")
                self.assertEqual(
                    speech.snapshot.active_text_range.text(request.text), request.text
                )
                with tempfile.TemporaryDirectory() as root:
                    path = Path(root) / "recording.wav"
                    self.assertEqual(tts.write_speech(speech, path).frames, done.frames)
                with speech.replay_from(done.frames // 2, options=paused) as replay:
                    self.assertNotEqual(replay.id, speech.id)
                    self.assertEqual(replay.plan.original, request.text)
                    replay.cancel()
                    self.assertEqual(replay.wait(1).outcome.reason, "requested")
                self.assertEqual(speech.wait(0), done)
            self.assertEqual(tts.retention_stats.used_bytes, 0)
            # GC forgets the external handle; a default-controller replacement
            # must still find and cancel the fire-and-forget native request.
            transient = tts.speak(request, options=paused)
            transient_id = transient.id
            del transient
            gc.collect()
            with tts.speak(request, options=paused) as replacement:
                self.assertGreater(replacement.id, transient_id)
                replacement.cancel()
                replacement.wait(1)
        gc.collect()
        self.assertEqual(lib.mlsr_live_handles(), before)


class VerifiedModelReadiness(unittest.TestCase):
    def test_invalid_catalog_does_not_create_an_owner(self):
        from mlsr import ModelDescriptor, ModelHandle

        lib = ffi.library()
        before = lib.mlsr_live_handles()
        with self.assertRaises(TtsError):
            ModelHandle(
                "/missing", ModelDescriptor("not a catalog", "model", "macos-arm64")
            )
        self.assertEqual(lib.mlsr_live_handles(), before)

    @unittest.skipUnless(
        os.environ.get("MLSR_TEST_CATALOG"), "verified model catalog not requested"
    )
    def test_verified_assets_then_independent_loaded_engine(self):
        from mlsr import ModelDescriptor, ModelHandle

        before = ffi.library().mlsr_live_handles()
        descriptor = ModelDescriptor(
            Path(os.environ["MLSR_TEST_CATALOG"]).read_text(),
            os.environ.get("MLSR_TEST_DELIVERY_ID", "supertonic3-coreml-int4-v1"),
            os.environ.get("MLSR_TEST_DELIVERY_PLATFORM", "macos-arm64"),
        )
        model = ModelHandle(
            os.environ.get("MLSR_TEST_VERIFIED_BUNDLE", os.environ["MLSR_TEST_BUNDLE"]),
            descriptor,
        )
        tts = None
        try:
            self.assertEqual(model.voices[0].readiness, "assets_verified")
            self.assertEqual(model.resolve_voice().voice.readiness, "assets_verified")
            tts = Tts.from_model(
                model,
                compute_policy=os.environ.get(
                    "MLSR_TEST_POLICY", "cpu_and_neural_engine"
                ),
                options=TtsOptions(output_mode="headless"),
            )
            model.close()
            self.assertEqual(tts.voices[0].readiness, "engine_ready")
            with self.assertRaises(TtsError):
                _ = model.voices
            with tts.synthesize("Model ownership has moved.") as job:
                job.cancel()
                self.assertEqual(job.wait().outcome.reason, "requested")
        finally:
            if tts:
                tts.close()
            model.close()
        self.assertEqual(ffi.library().mlsr_live_handles(), before)
