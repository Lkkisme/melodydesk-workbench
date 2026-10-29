"""Multi-Track alphaTex Exporter (Track 1: Lead Vocal Numbered Jianpu + Syllabic Lyrics; Track 2: 6-Line Guitar Tab + Split-Bar Chords + X/PM/HP Articulations)."""
from __future__ import annotations
import math
from music import pitch_name
from ._score_timing import build_score_timeline, duration_parts, TICKS


def _quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\r', ' ').replace('\n', ' ') + '"'


def _lyric(value):
    # These visible Unicode counterparts cannot become lyric-parser operators.
    return str(value).replace('[', '［').replace(']', '］').replace('+', '＋').replace('_', '＿').replace(' ', '＋').replace('\n', '＋') or '_'


def _beat(notes, part, chord=None, text=None):
    body = notes[0] if len(notes) == 1 else '(' + ' '.join(notes) + ')'
    effects = []
    if part['dots']:
        effects.append('d' if part['dots'] == 1 else 'dd')
    if part['tuplet']:
        effects.append('tu 3')
    if chord is not None:
        effects.append('ch ' + _quote(chord))
    if text is not None:
        effects.append('txt ' + _quote(text))
    suffix = ' {' + ' '.join(effects) + '}' if effects else ''
    return ':' + str(part['denominator']) + ' ' + body + suffix


def _guitar_bars(arrangement, timeline):
    measure = timeline['measure_ticks']
    events = []
    for event in arrangement['events']:
        start = round(event['start'] * TICKS)
        end = round((event['start'] + event['duration']) * TICKS)
        if end <= start or not 1 <= event['string'] <= 6:
            raise ValueError('Invalid guitar event on notation grid')
        events.append((start, end, event))
    # A later attack on a string cuts off its previous sustain.
    events.sort(key=lambda row: row[0])
    normalized = []
    for i, (start, end, event) in enumerate(events):
        following = [a for a, _, e in events[i+1:] if e['string'] == event['string'] and a > start]
        normalized.append((start, min([end] + following), event))
    bars = []
    for index in range(len(timeline['bars'])):
        left, right = index * measure, (index + 1) * measure
        chords = {}
        if index < len(arrangement['bars']):
            bar = arrangement['bars'][index]
            segments = bar['split_segments'] if 'split_segments' in bar else [dict(beat=0, chord=bar['chord'])]
            chords = {left + round(s['beat'] * TICKS): s['chord'] for s in segments}
        boundaries = {left, right, *chords}
        for start, end, _ in normalized:
            if start < right and end > left:
                boundaries.update((max(left, start), min(right, end)))
        boundaries = sorted(boundaries)
        beats = []
        for a, b in zip(boundaries, boundaries[1:]):
            active = [(start, event) for start, end, event in normalized if start <= a < end]
            for component, part in enumerate(duration_parts(b-a)):
                notes = []
                for start, event in active:
                    technique = event.get('technique', '')
                    note = ('x' if technique == 'X' else str(event['fret'])) + '.' + str(event['string'])
                    effects = []
                    if technique == 'PM':
                        effects.append('pm')
                    elif technique in ('H', 'P'):
                        effects.append('h')
                    if (start < a or component) and technique != 'X':
                        effects.append('t')
                    if effects:
                        note += ' {' + ' '.join(effects) + '}'
                    notes.append(note)
                text = 'CM' if component == 0 and any(e.get('technique') == 'CM' for _, e in active) else None
                beat = _beat(notes or ['r'], part, chords[a] if a in chords and component == 0 else None, text)
                beats.append(beat)
        bars.append(' '.join(beats))
    return bars


def to_alphatex(
    project,
    arrangement: dict,
    note_lyrics: list[str] | None = None,
) -> str:
    """Compile Project + Guitar Arrangement + aligned lyrics into multi-track alphaTex markup."""
    if not math.isfinite(project.bpm) or project.bpm <= 0:
        raise ValueError('Tempo must be finite and positive')
    if note_lyrics and len(note_lyrics) != len(project.notes):
        raise ValueError('Lyrics must contain one entry per source note')
    timeline = build_score_timeline(project, arrangement)
    vocal, lyrics = [], []
    for spans in timeline['bars']:
        beats = []
        for span in spans:
            for i, part in enumerate(duration_parts(span['duration_ticks'])):
                continuation = span['continuation'] or i > 0
                note = 'r' if span['pitch'] is None else pitch_name(span['pitch']).lower()
                if span['pitch'] is not None:
                    if continuation:
                        note += ' {t}'
                    lyrics.append('_' if continuation or not note_lyrics else _lyric(note_lyrics[span['note_index']]))
                beats.append(_beat([note], part))
        vocal.append(' '.join(beats))
    n, d = project.meter.split('/')
    header = [r'\title ' + _quote(project.title), r'\tempo ' + str(project.bpm), '.',
              r'\track "主唱简谱 (Lead Vocal)" "Vocal"', r'\staff {score numbered}',
              r'\instrument 73', r'\tuning piano']
    if note_lyrics:
        header.append(r'\lyrics ' + _quote(' '.join(lyrics) + ' '))
    header += [f'\\ts {n} {d}', r'\ks ' + project.tonic, ' |\n'.join(vocal),
               r'\track "吉他伴奏 (Acoustic Guitar)" "Gtr."', r'\staff {tabs}',
               r'\instrument 24', r'\tuning E4 B3 G3 D3 A2 E2',
               r'\capo ' + str(arrangement['capo']), f'\\ts {n} {d}',
               ' |\n'.join(_guitar_bars(arrangement, timeline))]
    return '\n'.join(header) + '\n'
