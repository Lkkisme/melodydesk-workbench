import io
import tempfile
import unittest
from pathlib import Path
import mido
from music import Project,parse_jianpu
import guitar

class GuitarTests(unittest.TestCase):
    def setUp(self):
        notes,_=parse_jianpu('1:1 3:1 5:2 | 4:1 6:1 1\':2')
        self.project=Project(title='<test>',notes=notes)

    def test_every_voicing_has_only_chord_tones_including_capo(self):
        for root in range(12):
            for minor in (False,True):
                for capo in range(8):
                    frets=guitar.voicing(guitar.chord_name(root,minor),capo)
                    pitches={(guitar.TUNING[i]+f+capo)%12 for i,f in enumerate(frets) if f>=0}
                    self.assertEqual(pitches,{root,(root+(3 if minor else 4))%12,(root+7)%12})
                    self.assertGreaterEqual(sum(f>=0 for f in frets),4)
                    self.assertLessEqual(max(frets)-min(f for f in frets if f>0),3)

    def test_midi_has_separate_melody_and_guitar(self):
        a=guitar.arrange(self.project,['C','F'])
        midi=mido.MidiFile(file=io.BytesIO(guitar.midi(self.project,a)))
        self.assertEqual(len(midi.tracks),2)
        self.assertTrue(any(e.type=='program_change' and e.program==24 for e in midi.tracks[1]))
        self.assertEqual(sum(e.type=='note_on' for e in midi.tracks[0]),len(self.project.notes))

    def test_styles_have_different_patterns_and_tab_matches_audio(self):
        a=guitar.arrange(self.project,['C','F'])
        b=guitar.arrange(self.project,['C','F'],'流行扫弦',2)
        self.assertGreater(len(b['events']),len(a['events']))
        for e in b['events']:
            self.assertEqual(e['pitch'],guitar.TUNING[e['string']]+e['fret']+2)
            self.assertGreater(e['duration'],0)

    def test_silence_bar_and_wrong_chord_count(self):
        self.assertEqual(guitar.arrange(self.project,['N.C.','C'])['bars'][0]['slots'],[[]]*8)
        with self.assertRaises(ValueError):guitar.arrange(self.project,['C'])
        with self.assertRaises(ValueError):guitar.arrange(self.project,['C','H'])

    def test_export_is_local_and_title_escaped(self):
        with tempfile.TemporaryDirectory() as folder:
            path=guitar.export_bundle(self.project,['C','F'],folder)
            self.assertIn('&lt;test&gt;',path.read_text(encoding='utf-8'))
            self.assertTrue((Path(folder)/'guitar-only.wav').stat().st_size>44)
            self.assertTrue((Path(folder)/'accompaniment.mid').is_file())

    def test_suggestions_use_one_chord_per_bar(self):
        self.assertEqual(len(guitar.suggest(self.project,0)),2)
        self.assertEqual(guitar.suggest(self.project,0)[0],'C')

    def test_compound_meter_events_stay_inside_bars(self):
        self.project.meter='6/8'
        chords=guitar.suggest(self.project,0)
        a=guitar.arrange(self.project,chords,'流行扫弦')
        for e in a['events']:
            self.assertLessEqual(e['start']+e['duration'],(e['bar']+1)*3+1e-6)

    def test_capo_recommendation_reduces_barres(self):
        self.assertEqual(guitar.suggest_capo(['G#','C#','D#','Fm','A#m']),1)

    def test_svg_has_six_strings_per_bar(self):
        import re
        import xml.etree.ElementTree as ET
        a=guitar.arrange(self.project,['C','F'])
        for svg in re.findall(r'<svg.*?</svg>',guitar.score_html(a),re.S):
            root=ET.fromstring(svg)
            self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}line[@class='tab-string']")),6)

if __name__=='__main__':unittest.main()
