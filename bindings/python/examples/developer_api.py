"""Public SDK export or native speech, with explicit bounded output and cleanup."""

import argparse
from pathlib import Path

from mlsr import Request, SpeechOptions, SupertonicOptions, Tts, TtsOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="Hello from MLSR.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default="supertonic3:M1")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--muted", action="store_true")
    parser.add_argument("--cancel-first", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("cpu_only", "cpu_and_neural_engine", "qnn_htp"),
        default="cpu_and_neural_engine",
    )
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    request = Request(
        args.text,
        locale=args.language,
        voice=args.voice,
        backend=SupertonicOptions(seed=args.seed),
    )
    with Tts(
        args.manifest,
        args.bundle,
        compute_policy=args.policy,
        options=TtsOptions(output_mode="default" if args.speak else "headless"),
    ) as tts:
        if args.cancel_first:
            with tts.synthesize(request) as cancelled:
                cancelled.start()
                cancelled.cancel()
                assert cancelled.wait().outcome.kind == "cancelled"
        if args.speak:
            with tts.speak(
                request, options=SpeechOptions(volume=0 if args.muted else 1)
            ) as speech:
                speech.wait()
                result = tts.write_speech(
                    speech, args.output, max_frames=args.max_seconds * 44100
                )
        else:
            # A separate lazy job remains independently cancellable while the
            # shared writer consumes bounded chunks. No full PCM array is made.
            with tts.synthesize(request) as job:
                result = tts.write_job(
                    job, args.output, max_frames=args.max_seconds * 44100
                )
                assert job.wait().outcome.kind == "completed"
        print(f"Wrote {result.frames} frames to {result.path}")


if __name__ == "__main__":
    main()
