import unittest
import tempfile
from pathlib import Path
from music import Project,parse_jianpu
import guitar
from lyrics import parse_text,lyric_text
from score_layout import diagram

class LayoutTests(unittest.TestCase):
    def test_chord_diagram_and_lyrics_escaped(self):
        notes,_=parse_jianpu('1:1 3:1 5:2')
        p=Project(notes=notes,lyrics=[{'start':0,'end':2,'text':'<校对歌词>','reviewed':False}])
        a=guitar.arrange(p,['C'])
        page=guitar.score_html(a)
        self.assertIn('chord-diagram',page)
        self.assertIn('&lt;校对歌词&gt;',page)
        self.assertIn('待校对',page)
        self.assertIn('拍</text>',page)

    def test_high_position_and_open_strings(self):
        self.assertIn('7品',diagram([7,9,9,8,7,7]))
        self.assertIn('○',diagram([0,2,2,1,0,0]))
        self.assertIn('×',diagram([-1,3,2,0,1,0]))

    def test_lyrics_round_trip_and_invalid_times(self):
        lines=parse_text('0.2\t1.8\t校对示例',2)
        self.assertEqual(parse_text(lyric_text(lines),2),lines)
        with self.assertRaises(ValueError):parse_text('nan\t3\t错误',4)
        with self.assertRaises(ValueError):parse_text('3\t2\t错误',4)

    def test_long_project_round_trip(self):
        p=Project(clip_duration=264.432,lyrics=parse_text('120\t124\t测试',265))
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/'long.json';p.save(path)
            self.assertEqual(Project.load(path),p)
