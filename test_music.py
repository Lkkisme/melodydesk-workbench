import copy
from pathlib import Path
import tempfile
import unittest
from music import Note, Project, parse_jianpu, quantize, numbered, midi_bytes, text_score


class MusicTests(unittest.TestCase):
    def test_manual_pitch_rhythm_and_rests(self):
        notes, total = parse_jianpu("1:1 #4':0.5 0:0.5 7,:2", tonic='D', bpm=120)
        self.assertEqual([n.pitch for n in notes], [62, 80, 61])
        self.assertEqual([n.start for n in notes], [0, 1, 2])
        self.assertEqual(total, 4)
        self.assertEqual(notes[-1].raw_start, 1)

    def test_invalid_manual(self):
        for text in ['1', '1:0', '1:-1', '9:1', '0:4', '#0:1', "1''''':1"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_jianpu(text)

    def test_numbered_octaves_and_accidentals(self):
        self.assertEqual(numbered(60), '1')
        self.assertEqual(numbered(72), "1'")
        self.assertEqual(numbered(59), '7,')
        self.assertEqual(numbered(66), '#4')
        self.assertEqual(numbered(62, 'D'), '1')

    def test_quantization_keeps_raw_timing(self):
        notes = [Note(60, 0, 1, .07, .43, .9), Note(62, 0, 1, .52, .43, .9)]
        result = quantize(notes, 120)
        self.assertEqual([n.start for n in result], [.25, 1])
        self.assertEqual(result[0].raw_start, .07)
        self.assertEqual(notes[0].start, 0)
        Project(notes=result).validate()

    def test_grid_collisions_do_not_overlap(self):
        notes = [Note(60, 0, 1, .01, .15, .4), Note(62, 0, 1, .03, .2, .9), Note(64, 0, 1, .2, .5, .8)]
        result = quantize(notes, 120)
        self.assertEqual(result[0].pitch, 62)
        Project(notes=result).validate()

    def test_non_finite_and_overlapping_rejected(self):
        for project in [Project(bpm=float('nan')), Project(notes=[Note(60, 0, 2), Note(62, 1, 1)]),
                        Project(notes=[Note(60, float('inf'), 1)]), Project(notes=[Note(60.5, 0, 1)])]:
            with self.assertRaises(ValueError):
                project.validate()

    def test_project_roundtrip(self):
        notes, _ = parse_jianpu('1:1 0:1 5:2')
        project = Project(title='测试旋律', notes=notes, clip_start=30, beat_origin=-.2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'project.json'
            project.save(path)
            self.assertEqual(project, Project.load(path))

    def test_midi_and_text(self):
        notes, _ = parse_jianpu('1:1 0:1 5:2')
        data = midi_bytes(Project(notes=notes, meter='6/8'))
        self.assertEqual(data[:4], b'MThd')
        self.assertEqual(data[-4:], b'\x00\xff\x2f\x00')
        self.assertIn(bytes([0xff, 0x58, 4, 6, 3, 24, 8]), data)
        self.assertIn(bytes([0x90, 60, 90]), data)
        self.assertIn('0:1', text_score(Project(notes=notes)))


if __name__ == '__main__':
    unittest.main()
