"""Unit tests for subpkgs/score_web."""
import inspect
import tempfile
import unittest
import wave
from pathlib import Path
import sys
import numpy as np
from fastapi.testclient import TestClient

THIS_DIR = Path(__file__).resolve().parent
SUBPKG_ROOT = THIS_DIR.parent
MELODY_ROOT = SUBPKG_ROOT.parent.parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from music import Note, Project, parse_jianpu
from subpkgs.score_web import to_alphatex, render_publication_score_html, app
import subpkgs.score_web.server as srv


class ScoreWebTests(unittest.TestCase):
    def setUp(self):
        notes, _ = parse_jianpu("1:1 3:1 5:2 | 6:1 5:1 3:2 | 4:1 6:1 1':2 | 5:1 2:1 1:2")
        self.project = Project(title="联谱渲染测试", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        self.dummy_arrangement = {
            "title": "联谱渲染测试",
            "tonic": "C",
            "meter": "4/4",
            "bpm": 120.0,
            "style": "指弹律动",
            "capo": 0,
            "bar_count": 4,
            "measure_beats": 4.0,
            "chords": ["Cmaj7", "Am7", "Fmaj7", "G7sus4"],
            "bars": [
                {
                    "bar": 0,
                    "chord": "Cmaj7",
                    "roman": "Imaj7",
                    "frets": [-1, 3, 2, 0, 0, 0],
                    "split_segments": [{"beat": 0.0, "duration": 4.0, "chord": "Cmaj7", "roman": "Imaj7", "frets": [-1, 3, 2, 0, 0, 0]}],
                },
                {
                    "bar": 1,
                    "chord": "Am7",
                    "roman": "vi7",
                    "frets": [-1, 0, 2, 0, 1, 0],
                    "split_segments": [
                        {"beat": 0.0, "duration": 2.0, "chord": "Am7", "roman": "vi7", "frets": [-1, 0, 2, 0, 1, 0]},
                        {"beat": 2.0, "duration": 2.0, "chord": "Am7/G", "roman": "vi7/5", "frets": [3, 0, 2, 0, 1, 0]},
                    ],
                },
                {
                    "bar": 2,
                    "chord": "Fmaj7",
                    "roman": "IVmaj7",
                    "frets": [1, 3, 3, 2, 1, 0],
                    "split_segments": [
                        {"beat": 0.0, "duration": 2.0, "chord": "Fmaj7", "roman": "IVmaj7", "frets": [1, 3, 3, 2, 1, 0]},
                        {"beat": 2.0, "duration": 2.0, "chord": "Fm6", "roman": "iv6", "frets": [1, 3, 3, 1, 3, 1]},
                    ],
                },
                {
                    "bar": 3,
                    "chord": "G7sus4",
                    "roman": "V7sus4",
                    "frets": [3, 2, 0, 0, 1, 1],
                    "split_segments": [{"beat": 0.0, "duration": 4.0, "chord": "G7sus4", "roman": "V7sus4", "frets": [3, 2, 0, 0, 1, 1]}],
                },
            ],
            "events": [
                {"bar": 0, "start": 0.0, "duration": 0.5, "string": 5, "fret": 3, "pitch": 48, "velocity": 88, "technique": "PM"},
                {"bar": 0, "start": 1.0, "duration": 0.25, "string": 6, "fret": 0, "pitch": 40, "velocity": 92, "technique": "X"},
                {"bar": 0, "start": 3.5, "duration": 0.5, "string": 2, "fret": 1, "pitch": 60, "velocity": 82, "technique": "H"},
                {"bar": 1, "start": 4.0, "duration": 0.5, "string": 5, "fret": 0, "pitch": 45, "velocity": 88, "technique": ""},
                {"bar": 2, "start": 8.0, "duration": 0.5, "string": 6, "fret": 1, "pitch": 41, "velocity": 88, "technique": ""},
                {"bar": 3, "start": 12.0, "duration": 0.5, "string": 6, "fret": 3, "pitch": 43, "velocity": 88, "technique": ""},
            ],
        }
        self.note_lyrics = ["今", "夜", "-", "微", "风", "-", "吹", "过", "-", "心", "海", "-"]

    def test_to_alphatex_exports_dual_tracks_and_techniques(self):
        tex = to_alphatex(self.project, self.dummy_arrangement, self.note_lyrics)
        self.assertIn("\\track \"主唱简谱 (Lead Vocal)\"", tex)
        self.assertIn("\\track \"吉他伴奏 (Acoustic Guitar)\"", tex)
        self.assertIn("\\staff {score numbered}", tex)
        self.assertIn("\\staff {tabs}", tex)
        self.assertTrue("x.6" in tex or "{x}" in tex or "{dead}" in tex)
        self.assertIn("{pm}", tex)
        self.assertIn("{h}", tex)

    def test_render_publication_score_html_includes_split_bar_and_lyrics_and_6_8_meter(self):
        html_doc = render_publication_score_html(self.project, self.dummy_arrangement, self.note_lyrics)
        self.assertIn("<svg", html_doc)
        self.assertIn("Cmaj7", html_doc)
        self.assertIn("Am7/G", html_doc)
        self.assertIn("Fm6", html_doc)
        self.assertIn("今", html_doc)

        # Verify compound 6/8 rendering produces valid SVG without cross-beat beam failures
        notes_68 = [Note(pitch=62, start=0.0, duration=1.5), Note(pitch=66, start=1.5, duration=1.5)]
        proj_68 = Project(title="成都6/8测试", tonic="D", meter="6/8", bpm=90.0, notes=notes_68)
        arr_68 = {
            "title": "成都6/8测试",
            "tonic": "D",
            "meter": "6/8",
            "bpm": 90.0,
            "style": "抒情琶音",
            "capo": 2,
            "bar_count": 1,
            "measure_beats": 3.0,
            "chords": ["D"],
            "bars": [
                {
                    "bar": 0,
                    "chord": "D",
                    "roman": "I",
                    "frets": [-1, 3, 2, 0, 1, 0],
                    "split_segments": [{"beat": 0.0, "duration": 3.0, "chord": "D", "roman": "I", "frets": [-1, 3, 2, 0, 1, 0]}],
                }
            ],
            "events": [
                {"bar": 0, "start": 0.5 * i, "duration": 0.5, "string": s, "fret": f, "pitch": p, "velocity": 80, "technique": ""}
                for i, (s, f, p) in enumerate([(5, 3, 50), (3, 0, 57), (2, 1, 62), (1, 0, 66), (2, 1, 62), (3, 0, 57)])
            ],
        }
        html_68 = render_publication_score_html(proj_68, arr_68, ["成", "都"])
        self.assertIn("<svg", html_68)
        self.assertIn("6/8", html_68)

    def test_fastapi_reharmonize_with_all_three_options_and_audio_stream(self):
        client = TestClient(app)
        r = client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        st = r.json()
        self.assertIn("alphatex", st)
        self.assertIn("publication_html", st)

        r2 = client.post(
            "/api/reharmonize",
            json={
                "preset": "neosoul_jazz",
                "style": "Neo-Soul 律动",
                "capo": 0,
                "passing_bass": True,
                "split_bar": True,
                "slap_x": True,
                "palm_mute": True,
                "embellish_hp": True,
                "intro_mode": "chord_melody",
            },
        )
        self.assertEqual(r2.status_code, 200)
        st2 = r2.json()
        self.assertEqual(st2["preset"], "neosoul_jazz")
        self.assertTrue(st2["split_bar"])
        self.assertTrue(st2["slap_x"])

        r_wav = client.get("/api/audio/guitar-only.wav")
        self.assertEqual(r_wav.status_code, 200)
        self.assertGreater(len(r_wav.content), 20000)

    def test_hybrid_accompaniment_first_with_acappella_fallback_pipeline(self):
        """Verify Plan A hybrid architecture: accompaniment-first beat/chord extraction + automatic pure-vocal fallback."""
        src = inspect.getsource(srv)
        self.assertIn("_has_audible_accompaniment", src)
        self.assertIn("recognize_accompaniment_chords", src)
        self.assertIn("synthesize_melody_conditioned_harmony", src)

        sr = 22050
        t = np.arange(sr, dtype=np.float64) / sr
        audible = (0.3 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype("<i2")
        silent = np.zeros(sr, dtype="<i2")
        with tempfile.TemporaryDirectory() as tmp:
            aud_path = Path(tmp) / "audible.wav"
            sil_path = Path(tmp) / "silent.wav"
            for p, arr in ((aud_path, audible), (sil_path, silent)):
                with wave.open(str(p), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(arr.tobytes())
            self.assertTrue(srv._has_audible_accompaniment(aud_path))
            self.assertFalse(srv._has_audible_accompaniment(sil_path))
            self.assertFalse(srv._has_audible_accompaniment(None))


if __name__ == "__main__":
    unittest.main()
