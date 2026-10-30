"""Locked unit tests for subpkgs/stem_mir (100% Pure Vocal-Driven Pipeline)."""
import inspect
import tempfile
import types
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
    synthesize_melody_conditioned_harmony,
)
import subpkgs.stem_mir.chord_recognizer as cr_mod


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
        vocal_onsets = []
        for beat_idx in range(16):
            t0 = int(beat_idx * spb * sr)
            vocal_onsets.append(beat_idx * spb)
            length = int(0.03 * sr)
            freq = 1200.0 if (beat_idx % 4 == 0) else 660.0
            t = np.arange(length) / sr
            y[t0 : t0 + length] += np.sin(2 * np.pi * freq * t) * np.exp(-t * 80.0)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clicks.wav"
            _write_wav(wav_path, y, sr=sr)
            res = track_beats_and_downbeats(wav_path, prefer_cuda=True, vocal_onsets_sec=vocal_onsets)
            self.assertIn("bpm", res)
            self.assertGreaterEqual(res["bpm"], 60.0)
            self.assertLessEqual(res["bpm"], 180.0)
            self.assertGreaterEqual(res["beat_origin"], 0.0)
            self.assertIn(res["meter"], ("4/4", "3/4"))
            self.assertIsInstance(res["beats"], list)
            self.assertIsInstance(res["downbeats"], list)

    def test_pure_vocal_driven_harmony_and_dynamics_major_and_minor(self):
        # Major key (A major): Bar 0 sings A4 + C#5 -> A; Bar 1 sings B4 + G#4 -> E
        vocal_notes_maj = [
            types.SimpleNamespace(pitch=69, start=0.0, duration=2.0),
            types.SimpleNamespace(pitch=73, start=2.0, duration=2.0),
            types.SimpleNamespace(pitch=71, start=4.0, duration=2.0),
            types.SimpleNamespace(pitch=68, start=6.0, duration=2.0),
        ]
        res_maj = synthesize_melody_conditioned_harmony(
            notes=vocal_notes_maj,
            bpm=84.0,
            beat_origin=0.0,
            meter="4/4",
            tonic="A",
            clip_duration=9.0,
            minor=False,
        )
        self.assertEqual(len(res_maj["chords"]), 4)
        self.assertTrue(all(c != "N.C." for c in res_maj["chords"]))
        self.assertEqual(res_maj["chords"][0], "A")
        self.assertEqual(res_maj["chords"][1], "E")
        self.assertEqual(len(res_maj["bar_dynamics"]), 4)

        # Dorian / Natural Minor key (E minor, tonic="G", minor=True):
        # Bar 0 sings E4 + G4 + B4 -> Em; Bar 1 sings D4 + F#4 + A4 -> D (bVII, not B major V!)
        vocal_notes_min = [
            types.SimpleNamespace(pitch=64, start=0.0, duration=1.5),
            types.SimpleNamespace(pitch=67, start=1.5, duration=1.5),
            types.SimpleNamespace(pitch=62, start=3.0, duration=1.5),
            types.SimpleNamespace(pitch=69, start=4.5, duration=1.5),
        ]
        res_min = synthesize_melody_conditioned_harmony(
            notes=vocal_notes_min,
            bpm=90.0,
            beat_origin=0.0,
            meter="3/4",
            tonic="G",
            clip_duration=4.0,
            minor=True,
        )
        self.assertEqual(res_min["chords"][0], "Em")
        self.assertEqual(res_min["chords"][1], "D")

    def test_zero_accompaniment_pollution_in_chord_recognizer(self):
        """Ensure chord_recognizer contains zero accompaniment audio / CQT parameters or functions."""
        sig = inspect.signature(synthesize_melody_conditioned_harmony)
        self.assertNotIn("accompaniment_path", sig.parameters)
        src = inspect.getsource(cr_mod)
        self.assertNotIn("recognize_accompaniment_chords", src)
        self.assertNotIn("extract_instrumental_melody", src)
        self.assertNotIn("librosa.cqt", src)


if __name__ == "__main__":
    unittest.main()
