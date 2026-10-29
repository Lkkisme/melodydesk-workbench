"""Locked unit tests for subpkgs/singing_lyrics."""
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
from subpkgs.singing_lyrics import (
    tokenize_lyrics,
    transcribe_word_timestamps,
    align_lyrics_to_notes,
    transcribe_vocal_notes,
)


class SingingLyricsTests(unittest.TestCase):
    def test_tokenize_mixed_cjk_and_english_lyrics(self):
        tokens = tokenize_lyrics("春风吹过 my heart，\n夜色温柔。")
        texts = [t["text"] for t in tokens]
        self.assertEqual(texts, ["春", "风", "吹", "过", "my", "heart", "夜", "色", "温", "柔"])
        # Clause break on 'heart', line break on '柔'
        self.assertTrue(tokens[5]["is_line_end"])
        self.assertTrue(tokens[-1]["is_line_end"])

    def test_align_lyrics_syllabic_and_melisma_and_phrase_gap(self):
        # 12 notes across 3 bars, 8 Chinese syllables -> 4 melisma '-' extensions
        notes, _ = parse_jianpu("1:1 3:1 5:2 | 6:1 5:1 3:2 | 4:1 6:1 1':2 | 5:1 2:1 1:2")
        project = Project(title="词曲对齐测试", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        res = align_lyrics_to_notes(project, raw_lyrics="今夜微风吹过心海")
        self.assertEqual(len(res["note_lyrics"]), len(project.notes))
        self.assertIn("-", res["note_lyrics"])
        # Every one of the 8 Chinese characters must appear in order
        chars = [c for c in res["note_lyrics"] if c and c != "-"]
        self.assertEqual("".join(chars), "今夜微风吹过心海")

    def test_align_lyrics_with_word_timestamp_anchors(self):
        notes, _ = parse_jianpu("1:1 2:1 3:1 5:1")
        project = Project(title="锚点对齐", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        # 120 BPM -> 0.5s per beat: notes at 0.0s, 0.5s, 1.0s, 1.5s
        word_ts = [
            {"word": "星", "start": 0.0, "end": 0.45},
            {"word": "光", "start": 0.5, "end": 1.45},  # spans 2 notes (beat 1 & beat 2 -> melisma '-')
            {"word": "落", "start": 1.5, "end": 1.95},
        ]
        res = align_lyrics_to_notes(project, raw_lyrics="星光落", word_timestamps=word_ts)
        self.assertEqual(res["note_lyrics"], ["星", "光", "-", "落"])

    def test_transcribe_vocal_notes_on_synthesized_singing_phrase(self):
        sr = 22050
        bpm = 120.0
        spb = 60.0 / bpm
        y = np.zeros(int(sr * 3.0), dtype=np.float32)
        for idx, midi_p in enumerate([60, 62, 64, 67]):
            s0 = int((0.2 + idx * spb) * sr)
            s1 = int((0.2 + (idx + 0.85) * spb) * sr)
            t = np.arange(s1 - s0) / sr
            f = 440.0 * (2.0 ** ((midi_p - 69.0) / 12.0))
            # Vibrato + harmonics to mimic singing voice
            f_mod = f * (1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
            phase = 2 * np.pi * np.cumsum(f_mod) / sr
            sig = 0.7 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
            y[s0:s1] = sig

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "vocal.wav"
            pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())

            out = transcribe_vocal_notes(wav_path, bpm=120.0, beat_origin=0.2, prefer_cuda=True)
            self.assertIn("notes", out)
            self.assertGreaterEqual(len(out["notes"]), 2)


if __name__ == "__main__":
    unittest.main()
