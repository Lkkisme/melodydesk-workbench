import unittest
import numpy as np
from music import Note
from vocal_activity import filter_notes, SINGING

class VocalGateTests(unittest.TestCase):
    def test_rejects_instrument_region_preserves_vocal_timing(self):
        notes=[Note(60,0,1,0.1,.5),Note(62,2,1,2.1,.5)]
        result=filter_notes(notes,np.arange(4),np.array([.02,.02,.7,.8]))
        self.assertEqual(result,[notes[1]])
        self.assertIs(result[0],notes[1])

    def test_spoken_voice_included_whistling_excluded(self):
        self.assertIn('Speech',SINGING)
        self.assertIn('Male singing',SINGING)
        self.assertNotIn('Whistling',SINGING)

    def test_no_events(self):
        self.assertEqual(filter_notes([],np.arange(2),np.zeros(2)),[])

if __name__=='__main__':
    unittest.main()
