"""Installed-package example: cancel/reuse, then stream a new Float32 WAV."""

import argparse
import struct
import sys
from pathlib import Path
from mlsr import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="Hello from MLSR.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--cancel-first", action="store_true")
    args = parser.parse_args()
    with args.output.open("xb") as wav:
        try:
            wav.write(bytes(56))
            frames = 0
            with Engine(args.manifest, args.bundle) as engine:
                if args.cancel_first:
                    with engine.start(
                        args.text,
                        language=args.language,
                        voice=args.voice,
                        queue_frames=1,
                        chunk_frames=1,
                    ) as job:
                        next(job)
                        job.cancel()
                        assert job.wait(30000)
                        assert next(job).kind == "cancelled"
                with engine.start(
                    args.text, language=args.language, voice=args.voice, seed=1729
                ) as job:
                    for event in job:
                        if event.kind == "audio":
                            frames += event.frames
                            if frames * 4 > 0xFFFFFFFF - 48:
                                raise ValueError("Classic WAV size limit exceeded")
                            if sys.byteorder != "little":
                                event.samples.byteswap()
                            wav.write(event.samples.tobytes())
                        if event.terminal and event.kind != "completed":
                            raise event.error or RuntimeError("Synthesis cancelled")
            wav.seek(0)
            wav.write(
                struct.pack(
                    "<4sI4s4sIHHIIHH4sII4sI",
                    b"RIFF",
                    48 + frames * 4,
                    b"WAVE",
                    b"fmt ",
                    16,
                    3,
                    1,
                    44100,
                    176400,
                    4,
                    32,
                    b"fact",
                    4,
                    frames,
                    b"data",
                    frames * 4,
                )
            )
        except BaseException:
            wav.close()
            args.output.unlink(missing_ok=True)
            raise
    print(f"Wrote {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
