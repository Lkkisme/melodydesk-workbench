"""Locked unit tests for subpkgs/stem_mir."""
import tempfile
import unittest
from pathlib import Path
import sys
import wave
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SUBPKG_ROOT = THIS_DIR.parent
MELODY_ROOT = SUBPKG_ROOT.parent.parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from subpkgs.stem_mir import (
    resolve_device,
    separate_stems,
    track_beats_and_downbeats,
    recognize_accompaniment_chords,
    extract_instrumental_melody,
)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 22050) -> None:
    peak = float(np.max(np.abs(samples))) if len(samples) else 1.0
    if peak > 1e-6:
        samples = samples / peak * 0.85
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class StemMirTests(unittest.TestCase):
    def test_resolve_device(self):
        dev = resolve_device(prefer_cuda=True)
        self.assertIn(dev, ("cuda", "cpu"))
        self.assertEqual(resolve_device(prefer_cuda=False), "cpu")

    def test_track_beats_and_downbeats_on_click_track(self):
        sr = 22050
        bpm = 120.0
        spb = 60.0 / bpm
        duration = 8.0
        y = np.zeros(int(sr * duration), dtype=np.float32)
        for beat_idx in range(16):
            t0 = int(beat_idx * spb * sr)
            length = int(0.03 * sr)
            freq = 1200.0 if (beat_idx % 4 == 0) else 660.0
            t = np.arange(length) / sr
            y[t0 : t0 + length] += np.sin(2 * np.pi * freq * t) * np.exp(-t * 80.0)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clicks.wav"
            _write_wav(wav_path, y, sr=sr)
            res = track_beats_and_downbeats(wav_path, prefer_cuda=True)
            self.assertIn("bpm", res)
            self.assertGreaterEqual(res["bpm"], 60.0)
            self.assertLessEqual(res["bpm"], 180.0)
            self.assertGreaterEqual(res["beat_origin"], 0.0)
            self.assertIn(res["meter"], ("4/4", "3/4"))
            self.assertIsInstance(res["beats"], list)
            self.assertIsInstance(res["downbeats"], list)

    def test_recognize_accompaniment_chords_and_split_bar_and_slash(self):
        sr = 22050
        bpm = 120.0
        spb = 60.0 / bpm
        bar_sec = 4 * spb  # 2.0s per bar
        y = np.zeros(int(sr * bar_sec * 4), dtype=np.float32)

        def synth_chord(midis, start_s, end_s):
            s0 = int(start_s * sr)
            s1 = int(end_s * sr)
            t = np.arange(s1 - s0) / sr
            sig = np.zeros_like(t)
            for idx, m in enumerate(midis):
                f = 440.0 * (2.0 ** ((m - 69.0) / 12.0))
                amp = 1.35 if idx == 0 else 0.85
                sig += amp * np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
            y[s0:s1] = sig

        # Bar 0: C major (C2, C4, E4, G4)
        synth_chord([36, 60, 64, 67], 0.0, 2.0)
        # Bar 1: G/B slash chord (B2=47 bass, G3=55, D4=62, G4=67)
        synth_chord([47, 55, 62, 67], 2.0, 4.0)
        # Bar 2: Am (A2=45, A3=57, C4=60, E4=64)
        synth_chord([45, 57, 60, 64], 4.0, 6.0)
        # Bar 3: Split-bar F (0..1s) -> G (1..2s)
        synth_chord([41, 53, 65, 69], 6.0, 7.0)
        synth_chord([43, 55, 59, 67], 7.0, 8.0)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "prog.wav"
            _write_wav(wav_path, y, sr=sr)

            res_full = recognize_accompaniment_chords(
                wav_path, bpm=120.0, beat_origin=0.0, meter="4/4", bar_count=4, tonic="C", split_bar=False
            )
            self.assertEqual(len(res_full["chords"]), 4)
            self.assertTrue(res_full["chords"][0].startswith("C"))
            self.assertTrue(res_full["chords"][1].startswith("G"))
            self.assertIn("/B", res_full["chords"][1])
            self.assertTrue(res_full["chords"][2].startswith("Am"))

            res_split = recognize_accompaniment_chords(
                wav_path, bpm=120.0, beat_origin=0.0, meter="4/4", bar_count=4, tonic="C", split_bar=True
            )
            self.assertEqual(len(res_split["split_chords"]), 4)
            # Bar 3 has two distinct half-bar chords (F -> G)
            bar3_segs = res_split["split_chords"][3]
            self.assertEqual(len(bar3_segs), 2)
            self.assertTrue(bar3_segs[0]["chord"].startswith("F"))
            self.assertTrue(bar3_segs[1]["chord"].startswith("G"))

    def test_extract_instrumental_melody_for_intro_bar(self):
        sr = 22050
        bpm = 120.0
        spb = 60.0 / bpm
        y = np.zeros(int(sr * 2.0), dtype=np.float32)
        # 4 quarter notes in Bar 0: E5 (76), D5 (74), C5 (72), G4 (67)
        for beat_i, midi_p in enumerate([76, 74, 72, 67]):
            s0 = int(beat_i * spb * sr)
            s1 = int((beat_i + 0.85) * spb * sr)
            t = np.arange(s1 - s0) / sr
            f = 440.0 * (2.0 ** ((midi_p - 69.0) / 12.0))
            y[s0:s1] = np.sin(2 * np.pi * f * t) * np.exp(-t * 2.0)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "intro.wav"
            _write_wav(wav_path, y, sr=sr)
            mel = extract_instrumental_melody(
                wav_path, bpm=120.0, beat_origin=0.0, meter="4/4", silent_bars=[0]
            )
            self.assertIn(0, mel)
            self.assertGreaterEqual(len(mel[0]), 3)
            pitches = [n["pitch"] for n in mel[0]]
            self.assertTrue(any(abs(p - 76) <= 1 for p in pitches))
            self.assertTrue(any(abs(p - 72) <= 1 for p in pitches))


if __name__ == "__main__":
    unittest.main()
