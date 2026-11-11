"""Unit tests for subpkgs/stem_mir (Plan A Hybrid Accompaniment-First + Pure-Vocal Fallback Pipeline)."""
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
    recognize_accompaniment_chords,
    resolve_device,
    separate_stems,
    synthesize_melody_conditioned_harmony,
    track_beats_and_downbeats,
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
            self.assertIn(res["meter"], ("4/4", "3/4", "6/8"))
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

        # Dorian / Natural Minor key (E minor, tonic="E", minor=True):
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
            tonic="E",
            clip_duration=4.0,
            minor=True,
        )
        self.assertEqual(res_min["chords"][0], "Em")
        self.assertEqual(res_min["chords"][1], "D")

    def test_hybrid_accompaniment_chord_recognition_and_acappella_fallback(self):
        """Accompaniment chord recognizer extracts true C -> Am -> F -> G from audio and returns available=False on silence."""
        sr = 22050
        bar_sec = 2.0  # 120 BPM in 4/4
        chord_midis = [
            [48, 60, 64, 67],  # C:  C3, C4, E4, G4
            [45, 57, 60, 64],  # Am: A2, A3, C4, E4
            [41, 53, 57, 60],  # F:  F2, F3, A3, C4
            [43, 55, 59, 62],  # G:  G2, G3, B3, D4
        ]
        t_bar = np.arange(int(sr * bar_sec), dtype=np.float64) / sr
        bars_audio = []
        for midis in chord_midis:
            sig = np.zeros_like(t_bar)
            for m in midis:
                f0 = 440.0 * (2.0 ** ((m - 69) / 12.0))
                sig += 0.35 * np.sin(2 * np.pi * f0 * t_bar) + 0.14 * np.sin(4 * np.pi * f0 * t_bar)
            bars_audio.append(sig.astype(np.float32))
        full_acc = np.concatenate(bars_audio)

        with tempfile.TemporaryDirectory() as tmp:
            acc_path = Path(tmp) / "acc.wav"
            silent_path = Path(tmp) / "silent.wav"
            _write_wav(acc_path, full_acc, sr=sr)
            pcm_zero = np.zeros(int(sr * 4.0), dtype="<i2")
            with wave.open(str(silent_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm_zero.tobytes())

            res = recognize_accompaniment_chords(
                accompaniment_path=acc_path,
                bpm=120.0,
                beat_origin=0.0,
                meter="4/4",
                clip_duration=8.0,
                bar_start_seconds=[0.0, 2.0, 4.0, 6.0, 8.0],
            )
            self.assertTrue(res["available"])
            self.assertEqual(res["tonic"], "C")
            self.assertFalse(res["minor"])
            self.assertEqual([c.split("/")[0] for c in res["chords"]], ["C", "Am", "F", "G"])
            self.assertEqual(len(res["bar_dynamics"]), 4)

            res_silent = recognize_accompaniment_chords(
                accompaniment_path=silent_path,
                bpm=120.0,
                beat_origin=0.0,
                meter="4/4",
                clip_duration=4.0,
            )
            self.assertFalse(res_silent["available"])


if __name__ == "__main__":
    unittest.main()
