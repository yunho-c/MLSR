"""Independent WAV receipt validation, without using the Rust WAV decoder."""

from pathlib import Path
import struct
import tempfile
import unittest

from tools.supertonic3.cli_smoke import wav_pcm


class CliWaveTests(unittest.TestCase):
    def test_float_pcm_bits_and_invalid_containers(self):
        pcm = struct.pack("<fff", -0.0, 1.5, -1.25)
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 176400, 4, 32)
        body = (
            b"WAVEfmt "
            + struct.pack("<I", len(fmt))
            + fmt
            + b"data"
            + struct.pack("<I", len(pcm))
            + pcm
        )
        wave = b"RIFF" + struct.pack("<I", len(body)) + body
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "audio.wav"
            file.write_bytes(wave)
            self.assertEqual(wav_pcm(file), pcm)
            invalid = [
                wave[:-1],
                b"not a wave",
                wave[:20] + struct.pack("<H", 1) + wave[22:],
                wave[:-4] + struct.pack("<f", float("nan")),
            ]
            for data in invalid:
                file.write_bytes(data)
                with self.assertRaises(ValueError):
                    wav_pcm(file)
