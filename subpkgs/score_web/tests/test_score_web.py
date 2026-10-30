"""Locked unit tests for subpkgs/score_web."""
import unittest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

THIS_DIR = Path(__file__).resolve().parent
SUBPKG_ROOT = THIS_DIR.parent
MELODY_ROOT = SUBPKG_ROOT.parent.parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from music import Note, Project, parse_jianpu
from subpkgs.score_web import to_alphatex, render_publication_score_html, app


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

    def test_render_publication_score_html_includes_split_bar_and_lyrics(self):
        html_doc = render_publication_score_html(self.project, self.dummy_arrangement, self.note_lyrics)
        self.assertIn("<svg", html_doc)
        self.assertIn("Cmaj7", html_doc)
        self.assertIn("Am7/G", html_doc)
        self.assertIn("Fm6", html_doc)
        self.assertIn("今", html_doc)

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

    def test_architectural_invariant_100_percent_vocal_driven_pipeline(self):
        """Hard architectural lock: server.py must track beats from stems['vocals'] and synthesize harmony with accompaniment_path=None."""
        import inspect
        import subpkgs.score_web.server as srv
        src = inspect.getsource(srv)
        self.assertIn("track_beats_and_downbeats(stems['vocals']", src)
        self.assertNotIn("track_beats_and_downbeats(audio_path", src)
        self.assertIn("accompaniment_path=None", src)
        self.assertNotIn("accompaniment_path=stems.get('no_vocals')", src)
        self.assertNotIn("accompaniment_path=stems['accompaniment']", src)


if __name__ == "__main__":
    unittest.main()
