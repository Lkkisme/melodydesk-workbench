"""Functional Roman Numeral Analysis, Pure Interval-Algebra Clash/Tension Detector, 6 Re-harmonization Presets, and Split-Bar Dual-Chord Re-harmonizer."""
from __future__ import annotations

import math
from .chord_voicing import parse_extended_chord, _note_pc, EXTENDED_INTERVALS, generate_voicings, optimize_voice_leading

_NAMES = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
_SCALE = (0,2,4,5,7,9,11)
_MAJOR = {'','7','maj7','add9','6','9','maj9'}
_MINOR = {'m','m7','madd9','m6','m9'}
REHARMONIZE_PRESETS = {name: {'name': label, 'description': description} for name, label, description in (
    ('original', '原版和弦 (Original)', 'Preserve harmony and resolve melody clashes'),
    ('folk_easy', '民谣简化 (Folk Easy)', 'Simplify extended major and minor harmony'),
    ('neosoul_jazz', 'Neo-Soul 爵士 (Maj7/m9)', 'Extend harmony with sevenths and ninths'),
    ('emotional_pop', '流行走心 (Emotional Pop)', 'Add tonic sevenths and melodic color'),
    ('walking_bass', '顺阶低音 (Walking Bass)', 'Use third and minor-seventh inversions'),
    ('suspend_dreamy', '空灵挂留 (Sus2/Sus4)', 'Use suspended harmony'),
)}


def _spell_from(base_name: str, semitones: int, letter_steps: int) -> str:
    letters = 'CDEFGAB'
    base_idx = letters.index(base_name[0].upper())
    target_idx = (base_idx + letter_steps) % 7
    target_letter = letters[target_idx]
    target_pc = (_note_pc(base_name) + semitones) % 12
    natural_pc = _note_pc(target_letter)
    diff = ((target_pc - natural_pc + 6) % 12) - 6
    if diff == 1:
        return target_letter + '#'
    if diff == -1:
        return target_letter + 'b'
    return _NAMES[target_pc]


def _measure_beats(project):
    numerator, denominator = map(int, project.meter.split('/'))
    if numerator <= 0 or denominator <= 0:
        raise ValueError('Invalid meter')
    return numerator * 4.0 / denominator


def _overlapping_notes(notes, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('Invalid interval boundaries')
    return [n for n in notes if n.start < end and n.start+n.duration > start]


def _parse(symbol):
    parsed = parse_extended_chord(symbol)
    if parsed is None:
        raise ValueError(f'Invalid chord: {symbol!r}')
    return parsed


def _degree(name, tonic):
    degree = ('CDEFGAB'.index(name[0].upper())-'CDEFGAB'.index(tonic[0].upper())) % 7
    difference = ((_note_pc(name)-_note_pc(tonic)-_SCALE[degree]+6)%12)-6
    accidental = '#' * difference if difference >= 0 else 'b' * -difference
    return accidental, degree


def _status(report):
    return 'clash' if report['clashes'] else 'tension' if report['tensions'] else 'safe'


def _transform(parsed, tonic, preset):
    quality, bass = parsed['quality'], parsed['bass']
    displacement = (parsed['root']-_note_pc(tonic))%12
    if preset == 'folk_easy':
        quality = '' if quality in _MAJOR else 'm' if quality in _MINOR else 'sus4' if quality == '7sus4' else quality
    elif preset == 'neosoul_jazz':
        quality = {'': '9' if displacement == 7 else 'maj9', 'm':'m9','7':'9',
                   'maj7':'maj9','m7':'m9','dim':'dim7','sus4':'7sus4','add9':'maj9','madd9':'m9'}.get(quality,quality)
    elif preset == 'emotional_pop':
        quality = ('maj7' if displacement == 0 else 'add9') if quality == '' else 'm7' if quality == 'm' else quality
    elif preset == 'walking_bass':
        if quality in _MAJOR:
            bass = _spell_from(parsed['root_name'], 4, 2)
        elif quality in _MINOR:
            bass = _spell_from(parsed['root_name'], 10, 6)
            quality = 'm7' if quality == 'm' else quality
    elif preset == 'suspend_dreamy':
        quality = 'sus2' if quality in {'','m'} else '7sus4' if quality == '7' else quality
    return parsed['root_name']+quality+('/'+bass if bass else '')


def _resolve_harmony(symbol, previous, following, tonic, notes, start, end):
    report = check_melody_clash(symbol, notes, start, end)
    if report['status'] != 'clash' and generate_voicings(symbol):
        return symbol, report
    for candidate in suggest_bar_substitutions(symbol,previous,following,tonic,notes,start,end):
        if candidate['clash']['status'] != 'clash' and generate_voicings(candidate['chord']):
            return candidate['chord'], candidate['clash']
    pitches = [n.pitch for n in _overlapping_notes(notes,start,end)]
    raise ValueError(f'No non-clashing playable harmony for interval [{start}, {end}); melody pitches {pitches}')


def roman_numeral(symbol: str, tonic: str = "C") -> str:
    """Compute functional Roman numeral label (e.g. Imaj7, V/7, vi7/5, IVadd9, iv6, V7/vi) relative to tonic."""
    _note_pc(tonic)
    if isinstance(symbol, str) and symbol.strip() == 'N.C.':
        return 'N.C.'
    parsed = _parse(symbol)
    accidental, degree = _degree(parsed['root_name'],tonic)
    quality = parsed['quality']
    numeral = ('I','II','III','IV','V','VI','VII')[degree]
    if quality in _MINOR | {'m7b5','dim','dim7'}:
        numeral = numeral.lower()
        quality = quality[1:] if quality.startswith('m') else quality
    result = accidental+numeral+quality
    if parsed['bass']:
        accidental, degree = _degree(parsed['bass'],tonic)
        result += '/'+accidental+str(degree+1)
    return result


def check_melody_clash(
    chord_symbol: str,
    melody_notes: list,
    bar_start: float,
    bar_end: float,
) -> dict:
    """Pure music-theory interval algebra clash & color-tension detector (zero hand-tuned weights)."""
    notes = _overlapping_notes(melody_notes,bar_start,bar_end)
    parsed = None if (isinstance(chord_symbol, str) and chord_symbol.strip() == 'N.C.') else _parse(chord_symbol)
    tones = sorted(set(parsed['pcs']) | {parsed['bass_pc']}) if parsed else []
    report = dict(status='safe', clashes=[], tensions=[], chord_tones=tones, notes_checked=len(notes))
    if parsed is None:
        return report
    labels = {1:'b9',2:'9',3:'#9',5:'11',6:'#11',8:'b13',9:'13',10:'b7',11:'maj7'}
    for note in notes:
        pitch = note.pitch
        if pitch%12 in tones:
            continue
        interval = (pitch-parsed['root'])%12
        reason = None
        # Compare the nearest octave-equivalent harmonic representatives below the melody.
        representatives = [pc+12*((pitch-pc-1)//12) for pc in tones]
        if (parsed['root']+4)%12 in tones and interval == 5:
            reason = 'fourth_over_major_third'
        elif any(pitch > ct and (pitch-ct)%12 == 1 for ct in representatives):
            reason = 'minor_ninth'
        entry = dict(pitch=pitch, start=note.start, interval=interval, reason=reason or labels.get(interval,'color'))
        report['clashes' if reason else 'tensions'].append(entry)
    report['status'] = _status(report)
    return report


def suggest_bar_substitutions(
    current_chord: str,
    prev_chord: str | None,
    next_chord: str | None,
    tonic: str,
    melody_notes: list,
    bar_start: float,
    bar_end: float,
) -> list[dict]:
    """Generate ranked chord substitutions for a single bar with Roman numeral and clash/tension status."""
    current = _parse(current_chord)
    tonic_pc = _note_pc(tonic)
    neighbors = [_parse(s) for s in (prev_chord,next_chord) if s is not None and s.strip() != 'N.C.']
    scale = {(tonic_pc+x)%12 for x in _SCALE}
    roots = [(tonic_pc+x)%12 for x in _SCALE]
    roots += [x for x in range(12) if x not in scale]
    symbols = list(dict.fromkeys([current_chord]+[_NAMES[r]+q for r in roots for q in EXTENDED_INTERVALS]))
    ranked = []
    for index, symbol in enumerate(symbols):
        parsed = _parse(symbol)
        pcs = set(parsed['pcs'])
        report = check_melody_clash(symbol,melody_notes,bar_start,bar_end)
        reason_label = '同功能安全代用' if report['status'] == 'safe' else ('色彩张力代用' if report['status'] == 'tension' else '避用音冲突')
        key = ({'safe':0,'tension':1,'clash':2}[report['status']], len(pcs-scale),
               len(pcs ^ set(current['pcs'])), sum(len(pcs ^ set(n['pcs'])) for n in neighbors), index)
        ranked.append((key,dict(chord=symbol,roman=roman_numeral(symbol,tonic),clash=report,reason=reason_label)))
    return [record for _,record in sorted(ranked,key=lambda pair:pair[0])]


def reharmonize_progression(
    base_chords: list[str],
    project,
    preset: str = "original",
    tonic: str | None = None,
    split_bar: bool = False,
) -> list[dict]:
    """Transform base chord progression using one of 6 presets, optionally generating Split-Bar (2 chords/bar) progressions."""
    if preset not in REHARMONIZE_PRESETS:
        raise ValueError(f'Unknown preset {preset!r}; supported: {", ".join(REHARMONIZE_PRESETS)}')
    tonic = project.tonic if tonic is None else tonic
    tonic_pc = _note_pc(tonic)
    if not base_chords:
        return []
    measure = _measure_beats(project)
    bars, flattened = [], []
    for index, original in enumerate(base_chords):
        parsed = None if (isinstance(original, str) and original.strip() == 'N.C.') else _parse(original)
        proposal = (original if preset == 'original' else _transform(parsed,tonic,preset)) if parsed else 'N.C.'
        pair = None
        if split_bar and parsed:
            degree = (parsed['root']-tonic_pc)%12
            root = parsed['root_name']
            if parsed['quality'] in _MAJOR:
                if degree == 0:
                    pair = (root+'maj7',root+'/'+_spell_from(root, 4, 2))
                elif degree == 5:
                    pair = (root+'maj7',root+'m6')
                elif degree == 7:
                    pair = (root+'7sus4',root+'7')
            elif parsed['quality'] in _MINOR and degree == 9:
                pair = (root+'m7',root+'m7/'+_spell_from(tonic, 7, 4))
        proposals = pair if pair else (proposal,)
        duration = measure/len(proposals)
        segments, reports = [], []
        prev_chord = base_chords[index-1] if index else None
        next_chord = base_chords[index+1] if index+1 < len(base_chords) else None
        for j, symbol in enumerate(proposals):
            start = index*measure+j*duration
            selected, report = _resolve_harmony(symbol,prev_chord,next_chord,tonic,project.notes,start,start+duration)
            segment = dict(beat=j*duration,duration=duration,chord=selected,roman=roman_numeral(selected,tonic),frets=[],capo=0)
            segments.append(segment)
            flattened.append(segment)
            reports.append(report)
        aggregate = dict(status='safe',clashes=[],tensions=[],chord_tones=sorted({p for r in reports for p in r['chord_tones']}),
                         notes_checked=sum(r['notes_checked'] for r in reports))
        for report in reports:
            aggregate['clashes'].extend(report['clashes'])
            aggregate['tensions'].extend(report['tensions'])
        aggregate['status'] = _status(aggregate)
        cands = suggest_bar_substitutions(segments[0]['chord'], prev_chord, next_chord, tonic, project.notes, index*measure, (index+1)*measure)[:8] if parsed else []
        bars.append(dict(bar=index,original_chord=original,chord=segments[0]['chord'],roman=segments[0]['roman'],
                         frets=[],clash=aggregate,split_segments=segments,candidates=cands))
    for segment, voicing in zip(flattened,optimize_voice_leading([s['chord'] for s in flattened])):
        segment['frets'] = list(voicing['frets'])
    for bar in bars:
        bar['frets'] = list(bar['split_segments'][0]['frets'])
    return bars
