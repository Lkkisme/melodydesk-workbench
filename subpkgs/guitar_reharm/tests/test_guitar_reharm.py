"""Locked unit tests for subpkgs/guitar_reharm."""
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

from music import Note, Project, parse_jianpu
from subpkgs.guitar_reharm import (
    parse_extended_chord,
    generate_voicings,
    optimize_voice_leading,
    REHARMONIZE_PRESETS,
    roman_numeral,
    check_melody_clash,
    reharmonize_progression,
    suggest_bar_substitutions,
    arrange_advanced,
    synthesize_realistic_guitar,
    mix_real_vocal_and_guitar,
    evaluate_vocal_guitar_mix,
)


class GuitarReharmTests(unittest.TestCase):
    def setUp(self):
        # Bar 0 is silent intro (notes start at beat 4.0 = Bar 1)
        notes = [
            Note(start=4.0, duration=1.0, pitch=64, raw_start=2.0, raw_duration=0.5),
            Note(start=5.0, duration=1.0, pitch=67, raw_start=2.5, raw_duration=0.5),
            Note(start=6.0, duration=2.0, pitch=72, raw_start=3.0, raw_duration=1.0),
            Note(start=8.0, duration=2.0, pitch=69, raw_start=4.0, raw_duration=1.0),
            Note(start=10.0, duration=2.0, pitch=67, raw_start=5.0, raw_duration=1.0),
            Note(start=12.0, duration=2.0, pitch=65, raw_start=6.0, raw_duration=1.0),
            Note(start=14.0, duration=2.0, pitch=62, raw_start=7.0, raw_duration=1.0),
        ]
        self.project = Project(title="吉他改谱与仿真测试", tonic="C", meter="4/4", bpm=120.0, beat_origin=0.25, notes=notes)

    def test_extended_and_slash_chords_and_voice_leading(self):
        for sym, exp_roman in [
            ("Cmaj7", "Imaj7"),
            ("Am7/G", "vi7/5"),
            ("G/B", "V/7"),
            ("Fadd9", "IVadd9"),
            ("Fm6", "iv6"),
            ("D/F#", "II/#4"),
        ]:
            p = parse_extended_chord(sym)
            self.assertIsNotNone(p)
            self.assertEqual(roman_numeral(sym, "C"), exp_roman)
            vs = generate_voicings(sym, capo=0)
            self.assertGreaterEqual(len(vs), 1)
            self.assertEqual(len(vs[0]["frets"]), 6)

        vl = optimize_voice_leading(["Cmaj7", "G/B", "Am7", "Am7/G", "Fmaj7", "Fm6"], capo=0)
        self.assertEqual(len(vl), 6)

    def test_reharmonize_presets_and_split_bar_option(self):
        base = ["C", "Am", "F", "G"]
        for preset in ("original", "folk_easy", "neosoul_jazz", "emotional_pop", "walking_bass", "suspend_dreamy"):
            res = reharmonize_progression(base, self.project, preset=preset, tonic="C", split_bar=False)
            self.assertEqual(len(res), 4)
            for item in res:
                self.assertNotEqual(item["clash"]["status"], "clash")

        # Option A: Split-Bar Harmony (2 chords per bar)
        res_split = reharmonize_progression(base, self.project, preset="emotional_pop", tonic="C", split_bar=True)
        self.assertEqual(len(res_split), 4)
        self.assertTrue(any(len(bar.get("split_segments", [])) == 2 for bar in res_split))

    def test_advanced_articulations_and_intro_chord_melody(self):
        # Option B (X slap, P.M., H/P, Passing Bass) + Option C (Intro Chord-Melody in silent Bar 0)
        arr = arrange_advanced(
            self.project,
            chords=["Cmaj7", "Am7", "Fmaj7", "G7sus4"],
            style="指弹律动",
            capo=0,
            split_chords=[
                [{"beat": 0.0, "duration": 4.0, "chord": "Cmaj7"}],
                [{"beat": 0.0, "duration": 2.0, "chord": "Am7"}, {"beat": 2.0, "duration": 2.0, "chord": "Am7/G"}],
                [{"beat": 0.0, "duration": 2.0, "chord": "Fmaj7"}, {"beat": 2.0, "duration": 2.0, "chord": "Fm6"}],
                [{"beat": 0.0, "duration": 2.0, "chord": "G7sus4"}, {"beat": 2.0, "duration": 2.0, "chord": "G7"}],
            ],
            passing_bass=True,
            slap_x=True,
            palm_mute=True,
            embellish_hp=True,
            intro_mode="chord_melody",
        )
        techniques = {e.get("technique", "") for e in arr["events"]}
        self.assertIn("X", techniques)
        self.assertIn("PM", techniques)
        self.assertTrue(any(t in ("H", "P") for t in techniques))
        # Bar 0 (intro silent vocal bar) must contain Chord-Melody lead events (`CM`)
        bar0_techs = {e.get("technique", "") for e in arr["events"] if e["bar"] == 0}
        self.assertIn("CM", bar0_techs)

    def test_realistic_guitar_simulator_and_real_vocal_unplugged_mixer(self):
        arr = arrange_advanced(
            self.project,
            chords=["Cmaj7", "Am7", "Fmaj7", "G7"],
            style="扫弦伴奏",
            capo=0,
            slap_x=True,
            palm_mute=True,
            embellish_hp=True,
            intro_mode="chord_melody",
        )
        with tempfile.TemporaryDirectory() as tmp:
            guitar_wav = Path(tmp) / "guitar.wav"
            stereo = synthesize_realistic_guitar(self.project, arr, out_wav_path=guitar_wav, sr=44100)
            self.assertTrue(guitar_wav.exists())
            self.assertEqual(stereo.ndim, 2)
            self.assertEqual(stereo.shape[1], 2)
            self.assertGreater(float(np.max(np.abs(stereo))), 0.1)

            # Synthesize a realistic singing vocal track aligned to beat_origin=0.25s
            sr = 44100
            spb = 60.0 / self.project.bpm
            dur_s = 9.0
            vocal = np.zeros(int(sr * dur_s), dtype=np.float32)
            for n in self.project.notes:
                s0 = int((self.project.beat_origin + n.start * spb) * sr)
                s1 = int((self.project.beat_origin + (n.start + n.duration) * spb) * sr)
                t = np.arange(max(1, s1 - s0)) / sr
                f = 440.0 * (2.0 ** ((n.pitch - 69.0) / 12.0))
                phase = 2 * np.pi * np.cumsum(f * (1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * t))) / sr
                vocal[s0:s1] = (0.65 * np.sin(phase) + 0.25 * np.sin(2 * phase)) * np.exp(-t * 0.4)

            vocal_wav = Path(tmp) / "vocals.wav"
            pcm = (np.clip(vocal, -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(vocal_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())

            mix_wav = Path(tmp) / "vocal_guitar_mix.wav"
            report = mix_real_vocal_and_guitar(vocal_wav, self.project, arr, mix_wav, sr=44100)
            self.assertTrue(mix_wav.exists())
            self.assertGreater(mix_wav.stat().st_size, 50000)
            self.assertIn("spectral_consonance", report)
            self.assertGreater(report["spectral_consonance"], 0.25)
            self.assertIn("clash_frame_ratio", report)


if __name__ == "__main__":
    unittest.main()
