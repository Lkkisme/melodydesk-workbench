"""Portable musical project model. Times in beats are quarter-note units."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math
import re
import struct

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEGREES = ['1', '#1', '2', '#2', '3', '4', '#4', '5', '#5', '6', '#6', '7']


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + '必须是有限数字')
    return value


@dataclass
class Note:
    pitch: int
    start: float
    duration: float
    raw_start: float = 0.0
    raw_duration: float = 0.0
    confidence: float = 1.0
    edited: bool = False

    def validate(self):
        if isinstance(self.pitch, bool) or int(self.pitch) != self.pitch or not 21 <= self.pitch <= 108:
            raise ValueError('音高必须是 21–108 的整数 MIDI 音高')
        self.pitch = int(self.pitch)
        for key in ('start', 'duration', 'raw_start', 'raw_duration', 'confidence'):
            setattr(self, key, finite(getattr(self, key), key))
        if self.start < 0 or self.duration <= 0 or self.raw_start < 0 or self.raw_duration < 0:
            raise ValueError('起始拍不能为负数，时值必须大于零')
        if not 0 <= self.confidence <= 1:
            raise ValueError('置信度超出范围')


@dataclass
class Project:
    title: str = '未命名旋律'
    source: str = ''
    clip_start: float = 0.0
    clip_duration: float = 30.0
    bpm: float = 120.0
    meter: str = '4/4'
    tonic: str = 'C'
    beat_origin: float = 0.0
    notes: list = field(default_factory=list)
    audio: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    engine: str = 'manual'
    schema_version: int = 1
    reference_notes: list = field(default_factory=list)
    lyrics: list = field(default_factory=list)

    def validate(self):
        self.bpm = finite(self.bpm, '速度')
        self.clip_start = finite(self.clip_start, '片段开始')
        self.clip_duration = finite(self.clip_duration, '片段长度')
        if len(self.lyrics)>20000:
            raise ValueError('歌词过长')
        for line in self.lyrics:
            if not isinstance(line.get('text'),str) or len(line['text'])>2000:
                raise ValueError('歌词文本无效')
            a=finite(line.get('start'),'歌词起点');b=finite(line.get('end'),'歌词终点')
            if a<0 or b<a or b>self.clip_duration+1:
                raise ValueError('歌词时间超出音频范围')
        self.beat_origin = finite(self.beat_origin, '拍点偏移')
        if not 30 <= self.bpm <= 300:
            raise ValueError('速度应在 30–300 BPM 之间')
        if self.meter not in ('2/4', '3/4', '4/4', '6/8') or self.tonic not in NAMES:
            raise ValueError('调号或拍号无效')
        if self.clip_start < 0 or not 0 < self.clip_duration <= 3600:
            raise ValueError('片段时间无效')
        if len(self.notes) > 20000:
            raise ValueError('音符数量过多')
        for note in self.notes:
            note.validate()
        if len(self.reference_notes) > 20000:
            raise ValueError('参考音符数量过多')
        for note in self.reference_notes:
            note.validate()
        self.notes.sort(key=lambda n: n.start)
        for left, right in zip(self.notes, self.notes[1:]):
            if left.start + left.duration > right.start + 1e-6:
                raise ValueError('主旋律音符重叠，请调整起始拍或时值')
        if self.notes and self.notes[-1].start + self.notes[-1].duration > 14400:
            raise ValueError('乐谱长度过大')

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        if Path(path).stat().st_size > 10_000_000:
            raise ValueError('项目文件过大')
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema_version') != 1:
            raise ValueError('不支持的项目版本')
        data['notes'] = [Note(**n) for n in data.get('notes', [])]
        data['reference_notes'] = [Note(**n) for n in data.get('reference_notes', [])]
        project = cls(**data)
        project.validate()
        return project


def pitch_name(pitch):
    return NAMES[pitch % 12] + str(pitch // 12 - 1)


def numbered(pitch, tonic='C'):
    relative = pitch - (60 + NAMES.index(tonic))
    octave, semitone = divmod(relative, 12)
    return DEGREES[semitone] + ("'" * octave if octave >= 0 else ',' * -octave)


def quantize(notes, bpm, origin=0.0, grid=0.25):
    """Keep raw timing immutable; snap to a monophonic sixteenth-note grid."""
    result = []
    seconds_per_beat = 60 / bpm
    for note in sorted(notes, key=lambda n: n.raw_start):
        start = max(0.0, round((note.raw_start - origin) / seconds_per_beat / grid) * grid)
        end = max(start + grid, round((note.raw_start + note.raw_duration - origin) / seconds_per_beat / grid) * grid)
        candidate = Note(note.pitch, start, end - start, note.raw_start, note.raw_duration, note.confidence, note.edited)
        if result and result[-1].start + result[-1].duration > start:
            if result[-1].start == start:
                if result[-1].confidence >= candidate.confidence:
                    continue
                result.pop()
            else:
                result[-1].duration = start - result[-1].start
        result.append(candidate)
    return result


def performance_timing(notes, bpm, origin=0.0):
    """Represent performance timing in beats without imposing a note grid."""
    result = []
    for note in notes:
        start = max(0, (note.raw_start - origin) * bpm / 60)
        end = max(start, (note.raw_start + note.raw_duration - origin) * bpm / 60)
        if end > start:
            result.append(Note(note.pitch, start, end - start, note.raw_start,
                               note.raw_duration, note.confidence, note.edited))
    return result


def parse_jianpu(text, tonic='C', bpm=120):
    """Tokens: #4':0.5, 1:1, 0:1; apostrophe/comma denote octaves."""
    scale = [0, 2, 4, 5, 7, 9, 11]
    beat = 0.0
    notes = []
    for token in text.replace('|', ' ').split():
        match = re.fullmatch(r'([#b]?)([0-7])([\x27,]*):([0-9]+(?:\.[0-9]+)?)', token)
        if not match:
            raise ValueError('无法识别：' + token + '。请使用 1:1、3:0.5、0:1 等格式')
        accidental, degree, octave, duration = match.groups()
        duration = finite(duration, '时值')
        if not 0 < duration <= 64:
            raise ValueError('每个音的时值须大于 0 且不超过 64 拍')
        if degree != '0':
            pitch = 60 + NAMES.index(tonic) + scale[int(degree) - 1]
            pitch += 12 * (octave.count("'") - octave.count(','))
            pitch += 1 if accidental == '#' else -1 if accidental == 'b' else 0
            note = Note(pitch, beat, duration, beat * 60 / bpm, duration * 60 / bpm, 1.0, True)
            note.validate()
            notes.append(note)
        elif accidental or octave:
            raise ValueError('休止符不能带升降号或八度记号')
        beat += duration
    if not notes:
        raise ValueError('请至少输入一个音符')
    return notes, beat


def vlq(value):
    data = [value & 127]
    value >>= 7
    while value:
        data.insert(0, (value & 127) | 128)
        value >>= 7
    return bytes(data)


def midi_bytes(project):
    project.validate()
    ticks = 480
    tempo = round(60_000_000 / project.bpm)
    numerator, denominator = map(int, project.meter.split('/'))
    events = [(0, 0, b'\xff\x51\x03' + tempo.to_bytes(3, 'big')),
              (0, 0, b'\xff\x58\x04' + bytes([numerator, int(math.log2(denominator)), 24, 8])),
              (0, 0, b'\xc0\x00')]
    for note in project.notes:
        start = round(note.start * ticks)
        end = max(start + 1, round((note.start + note.duration) * ticks))
        events += [(start, 2, bytes([0x90, note.pitch, 90])), (end, 1, bytes([0x80, note.pitch, 0]))]
    track = bytearray()
    previous = 0
    for tick, _, event in sorted(events, key=lambda e: (e[0], e[1])):
        track.extend(vlq(tick - previous))
        track.extend(event)
        previous = tick
    track.extend(b'\x00\xff\x2f\x00')
    return b'MThd' + struct.pack('>IHHH', 6, 0, 1, ticks) + b'MTrk' + struct.pack('>I', len(track)) + track


def text_score(project):
    project.validate()
    lines = [project.title, '1=' + project.tonic + '  ' + project.meter + '  BPM=' + str(round(project.bpm, 2)),
             '数字后冒号表示四分音符拍数；\x27 表示高八度，, 表示低八度；0 为休止。',
             '下表为事件简谱（不是排版完成的出版谱）；小节从确认后的第 0 拍起算。', '']
    beat = 0.0
    measure = int(project.meter.split('/')[0]) * 4 / int(project.meter.split('/')[1])
    for note in project.notes:
        if note.start > beat + 1e-6:
            lines.append(f'拍 {beat:g}    0:{note.start - beat:g}')
        lines.append(f'小节 {int(note.start // measure) + 1:02}  拍 {note.start:g}    {numbered(note.pitch, project.tonic)}:{note.duration:g}')
        beat = note.start + note.duration
    return '\n'.join(lines)
