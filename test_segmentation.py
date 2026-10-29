import unittest
import numpy as np
from pitch_notes import segment_pitch
from music import Note, performance_timing, Project


class SegmentationTests(unittest.TestCase):
    def segment(self, midi, energy=None, probability=None):
        n = len(midi)
        return segment_pitch(midi, np.ones(n)*.9 if probability is None else probability,
                             np.ones(n)*.2 if energy is None else energy, .01, n*.01)

    def test_vibrato_does_not_become_many_notes(self):
        t = np.arange(200)*.01
        notes = self.segment(60 + .58*np.sin(2*np.pi*6*t))
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch, 60)
        self.assertAlmostEqual(notes[0].raw_duration, 2)

    def test_real_pitch_changes_survive(self):
        notes = self.segment(np.repeat([60, 62, 64, 62], 40))
        self.assertEqual([n.pitch for n in notes], [60, 62, 64, 62])

    def test_wide_vibrato_preserves_center(self):
        t=np.arange(200)*.01
        notes=self.segment(60+1.25*np.sin(2*np.pi*5*t))
        self.assertEqual([n.pitch for n in notes],[60])

    def test_short_articulated_steps_are_not_vibrato(self):
        notes=self.segment(np.repeat([60,62,60,62,60,62],15))
        self.assertEqual([n.pitch for n in notes],[60,62,60,62,60,62])

    def test_wide_vibrato_followed_by_new_note(self):
        t=np.arange(100)*.01
        notes=self.segment(np.r_[60+1.2*np.sin(2*np.pi*5*t),np.ones(100)*64])
        self.assertEqual([n.pitch for n in notes],[60,64])

    def test_wide_vibrato_rearticulation_is_retained(self):
        t=np.arange(200)*.01
        energy=np.ones(200)*.2
        energy[95:101]=np.linspace(.2,.01,6)
        energy[101:108]=np.linspace(.01,.2,7)
        notes=self.segment(60+1.2*np.sin(2*np.pi*5*t),energy)
        self.assertGreaterEqual(len(notes),2)

    def test_short_octave_glitch_is_bridged(self):
        values = np.ones(200)*60
        values[80:86] = 72
        self.assertEqual(len(self.segment(values)), 1)

    def test_actual_rest_remains(self):
        values = np.ones(200)*60
        values[90:110] = np.nan
        notes = self.segment(values)
        self.assertEqual(len(notes), 2)
        self.assertGreater(notes[1].raw_start - notes[0].raw_start - notes[0].raw_duration, .19)

    def test_repeated_note_with_rearticulation_survives(self):
        energy = np.ones(200)*.2
        energy[95:101] = np.linspace(.2,.02,6)
        energy[101:108] = np.linspace(.02,.2,7)
        self.assertEqual(len(self.segment(np.ones(200)*60, energy)), 2)

    def test_low_confidence_leakage_rejected(self):
        self.assertEqual(self.segment(np.ones(100)*39, probability=np.ones(100)*.01), [])

    def test_no_implicit_rhythm_quantization(self):
        notes = performance_timing([Note(60,0,1,.137,.431), Note(62,0,1,.613,.237)],120,-.05)
        self.assertAlmostEqual(notes[0].start,.374)
        self.assertAlmostEqual(notes[0].duration,.862)
        self.assertAlmostEqual(notes[1].raw_start,.613)
        Project(notes=notes).validate()


if __name__ == '__main__':
    unittest.main()
