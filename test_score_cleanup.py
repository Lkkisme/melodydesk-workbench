import copy
import unittest
from music import Note,Project
from score_cleanup import organize

class ScoreTests(unittest.TestCase):
    def notes(self,starts,length=.23):
        return [Note(60+i%5,s*2,length*2,s,length) for i,s in enumerate(starts)]

    def test_late_singing_snaps_to_common_rhythm_without_losing_source(self):
        notes=self.notes([.03,.28,.52,.79,1.02])
        before=copy.deepcopy(notes)
        out=organize(notes,120)
        self.assertEqual([n.start for n in out],[0,.5,1,1.5,2])
        self.assertEqual(notes,before)
        self.assertEqual([n.raw_start for n in out],[n.raw_start for n in notes])
        Project(notes=out).validate()

    def test_triplet_phrase_is_not_forced_to_binary(self):
        out=organize(self.notes([0,1/6,2/6,.5,4/6,5/6],.15),120)
        for n,s in zip(out,[0,1/3,2/3,1,4/3,5/3]):
            self.assertAlmostEqual(n.start,s)

    def test_long_rest_survives(self):
        out=organize(self.notes([0,.5,2,2.5],.22),120)
        self.assertGreater(out[2].start-out[1].start-out[1].duration,1)

    def test_dense_notes_not_silently_deleted(self):
        with self.assertRaises(ValueError):
            organize(self.notes([i*.005 for i in range(20)],.005),120)

    def test_offbeat_and_chromatic_pitches_survive(self):
        notes=self.notes([.25,.75,1.25],.4)
        out=organize(notes,120,mode='binary')
        self.assertEqual([n.start for n in out],[.5,1.5,2.5])
        self.assertEqual([n.pitch for n in out],[n.pitch for n in notes])

if __name__=='__main__':
    unittest.main()
