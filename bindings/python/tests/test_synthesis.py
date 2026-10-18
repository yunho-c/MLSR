import os
from pathlib import Path
import threading
import unittest
from mlsr import Engine, SynthesisError
from mlsr import _synthesis as ffi


class SynthesisTests(unittest.TestCase):
    def test_options_and_owned_error_without_models(self):
        lib = ffi.library()
        before = lib.mlsr_live_handles()
        with self.assertRaises(SynthesisError) as error:
            Engine("/missing/mlsr-manifest.yaml", "/missing/mlsr-bundle")
        self.assertEqual(error.exception.code, "io_error")
        self.assertEqual(lib.mlsr_live_handles(), before)

    @unittest.skipUnless(
        os.environ.get("MLSR_TEST_BUNDLE"), "real model assets not requested"
    )
    def test_real_stream_cancel_reuse_and_owned_audio(self):
        lib = ffi.library()
        before = lib.mlsr_live_handles()
        retained = None
        output = os.environ.get("MLSR_TEST_OUTPUT")
        if output:
            Path(output).mkdir(parents=True, exist_ok=False)
        with Engine(
            os.environ["MLSR_TEST_MANIFEST"],
            os.environ["MLSR_TEST_BUNDLE"],
            compute_policy=os.environ.get("MLSR_TEST_POLICY", "cpu_and_neural_engine"),
        ) as engine:
            with engine.start(
                "Hello from MLSR. Another sentence.", queue_frames=1, chunk_frames=1
            ) as job:
                self.assertEqual(next(job).kind, "predicted")
                with self.assertRaises(SynthesisError) as failure:
                    engine.start("Busy.")
                self.assertEqual(failure.exception.code, "engine_busy")
                thread = threading.Thread(target=job.cancel)
                thread.start()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertTrue(job.wait(30000))
                self.assertEqual(next(job).kind, "cancelled")
                with self.assertRaises(StopIteration):
                    next(job)
            for index, (language, voice) in enumerate(
                [("en", "M1"), ("en", "F1"), ("ko", "M1"), ("ko", "F1")]
            ):
                text = (
                    "Hello from MLSR."
                    if language == "en"
                    else "안녕하세요. 반갑습니다."
                )
                frames = 0
                pcm = bytearray()
                terminals = 0
                with engine.start(
                    text, language=language, voice=voice, seed=1729 + index
                ) as job:
                    for event in job:
                        self.assertEqual(event.job_id, job.id)
                        if event.kind == "audio":
                            self.assertEqual(event.start_frame, frames)
                            frames += event.frames
                            self.assertEqual(event.sample_rate, 44100)
                            self.assertEqual(event.channels, 1)
                            self.assertLessEqual(event.frames, 4096)
                            self.assertTrue(
                                text.encode()[slice(*event.text_span)].decode()
                            )
                            pcm.extend(event.samples.tobytes())
                            retained = event.samples
                        if event.terminal:
                            terminals += 1
                            self.assertEqual(event.kind, "completed")
                            self.assertEqual(event.frames, frames)
                self.assertGreater(frames, 44100)
                self.assertEqual(terminals, 1)
                if output:
                    (Path(output) / f"{language}_{voice}.f32").write_bytes(pcm)
            with engine.start("😀") as job:
                failed = next(job)
                self.assertEqual(failed.kind, "failed")
                self.assertEqual(failed.error.code, "empty_text")
        self.assertTrue(retained)
        self.assertEqual(lib.mlsr_live_handles(), before)


if __name__ == "__main__":
    unittest.main()
