"""Functional Roman Numeral Analysis, 12-D TIS Consonance & Dyadic Roughness Clash Detector, 6 Re-harmonization Presets with Global Viterbi Bass-Line Decoder, and Split-Bar Cadential / Secondary-Dominant Solver."""
from __future__ import annotations

import cmath
from functools import lru_cache
import math
from .chord_voicing import parse_extended_chord, _note_pc, EXTENDED_INTERVALS, generate_voicings, optimize_voice_leading

_NAMES = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
_SCALE = (0,2,4,5,7,9,11)
_MAJOR = {'','7','maj7','add9','add11','add#11','add2','add4','6','6/9','9','maj9','11','maj11','13','maj13','7b9','7#9','7b5','7#5','7#11','maj7#11','9#11','maj9#11','13#11','maj13#11','7b13','7alt','aug7'}
_MINOR = {'m','m7','madd9','madd11','m6','m6/9','m9','m11','m13','mmaj7','mmaj9','mmaj11','mmaj13'}
REHARMONIZE_PRESETS = {name: {'name': label, 'description': description} for name, label, description in (
    ('original', '原版和弦 (Original)', '保持原曲基础三/七和弦与主旋律严密协和'),
    ('folk_easy', '民谣简化 (Folk Easy)', '简化复杂延伸音，优先开放把位易按指型'),
    ('neosoul_jazz', 'Neo-Soul 爵士 (Maj7/m9)', '升级为 Maj9 / m9 / 9 / 7sus4 爵士色彩和弦'),
    ('emotional_pop', '流行走心 (Emotional Pop)', '加入 Imaj7 / IVadd9 / vi7 华语抒情催泪色彩'),
    ('walking_bass', '顺阶低音 (Walking Bass)', '使用转位斜杠和弦 (C/E, G/B) 构成顺阶低音线'),
    ('suspend_dreamy', '空灵挂留 (Sus2/Sus4)', '使用 sus2 / 7sus4 挂留和弦营造开阔空间感'),
)}

W_TIS = (2.0, 11.0, 17.0, 16.0, 19.0, 7.0)
_TWO_PI = 2.0 * cmath.pi


def _tonal_interval_vector(pc_weights: list[float]) -> list[complex]:
    total = sum(pc_weights) or 1.0
    result = []
    for k in range(1, 7):
        T_k = W_TIS[k - 1] * sum(
            pc_weights[p] * cmath.exp(-1j * _TWO_PI * k * p / 12.0)
            for p in range(12)
        ) / total
        result.append(T_k)
    return result


def _tis_similarity(vec_a: list[complex], vec_b: list[complex]) -> float:
    dot = sum((a * b.conjugate()).real for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(abs(a) ** 2 for a in vec_a)) if vec_a else 0.0
    mag_b = math.sqrt(sum(abs(b) ** 2 for b in vec_b)) if vec_b else 0.0
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (mag_a * mag_b)))


def _dyadic_roughness(chord_tones: set[int], parsed: dict | None, melody_pc: int) -> float:
    if parsed is None:
        return 0.0
    roughness = 0.0
    for ct in chord_tones:
        interval = (melody_pc - ct) % 12
        if interval == 1 or interval == 11:
            roughness += 1.0
    if (parsed['root'] + 4) % 12 in chord_tones:
        interval_from_root = (melody_pc - parsed['root']) % 12
        if interval_from_root == 5 and (parsed['root'] + 5) % 12 not in chord_tones:
            roughness += 0.8
    return roughness


def _chord_tiv(parsed: dict | None) -> list[complex]:
    if parsed is None:
        return [0j] * 6
    weights = [0.0] * 12
    for pc in parsed['pcs']:
        weights[pc] += 1.0
    weights[parsed['bass_pc']] += 0.5
    return _tonal_interval_vector(weights)


def _spell_from(base_name: str, semitones: int, letter_steps: int) -> str:
    letters = 'CDEFGAB'
    base_idx = letters.index(base_name[0].upper())
    target_idx = (base_idx + letter_steps) % 7
    target_letter = letters[target_idx]
    target_pc = (_note_pc(base_name) + semitones) % 12
    natural_pc = _note_pc(target_letter)
    diff = ((target_pc - natural_pc + 6) % 12) - 6
    if diff > 0:
        return target_letter + ('#' * diff)
    if diff < 0:
        return target_letter + ('b' * (-diff))
    return target_letter


def _measure_beats(project):
    numerator, denominator = map(int, project.meter.split('/'))
    if numerator <= 0 or denominator <= 0:
        raise ValueError('Invalid meter')
    return numerator * 4.0 / denominator


def _overlapping_notes(notes, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('Invalid interval boundaries')
    return [n for n in notes if n.start < end and n.start+n.duration > start]


@lru_cache(maxsize=1024)
def _parse(symbol):
    parsed = parse_extended_chord(symbol)
    if parsed is None:
        raise ValueError(f'Invalid chord: {symbol!r}')
    return parsed


@lru_cache(maxsize=1024)
def _degree(name, tonic):
    degree = ('CDEFGAB'.index(name[0].upper())-'CDEFGAB'.index(tonic[0].upper())) % 7
    difference = ((_note_pc(name)-_note_pc(tonic)-_SCALE[degree]+6)%12)-6
    accidental = '#' * difference if difference >= 0 else 'b' * -difference
    return accidental, degree


def _transform(parsed, tonic, preset):
    candidates = _generate_preset_candidates(parsed, _note_pc(tonic), preset, 0, 1)
    if candidates:
        return candidates[0][0]
    return parsed['root_name'] + parsed['quality'] + ('/' + parsed['bass'] if parsed['bass'] else '')


def _generate_preset_candidates(parsed: dict, tonic_pc: int, preset: str, bar_idx: int, total_bars: int) -> list[tuple[str, float]]:
    root = parsed['root_name']
    quality = parsed['quality']
    bass = parsed['bass']
    displacement = (parsed['root'] - tonic_pc) % 12
    original_sym = root + quality + ('/' + bass if bass else '')

    if preset == 'original':
        return [(original_sym, 1.0)]

    candidates: list[tuple[str, float]] = []

    ivs = parsed['intervals']
    is_dim_q = (3 in ivs and 6 in ivs and 7 not in ivs)
    is_min_q = (3 in ivs and 4 not in ivs and not is_dim_q) or quality in _MINOR
    is_dom_q = (4 in ivs and 10 in ivs) or quality in {'7', '9', '11', '13'}
    is_maj_q = (4 in ivs and 3 not in ivs) or quality in _MAJOR

    if preset == 'folk_easy':
        if is_maj_q:
            candidates.append((root, 1.0))
            candidates.append((root + 'add9', 0.7))
        elif is_min_q:
            candidates.append((root + 'm', 1.0))
            candidates.append((root + 'm7', 0.75))
        elif 5 in ivs and 10 in ivs:
            candidates.append((root + 'sus4', 0.9))
            candidates.append((root, 0.6))
        else:
            candidates.append((original_sym, 0.8))
        candidates.append((root + 'sus4', 0.4))

    elif preset == 'neosoul_jazz':
        if is_dom_q:
            candidates.append((root + '9', 0.9))
            candidates.append((root + '7', 0.85))
            candidates.append((root + '7sus4', 0.75))
        elif is_maj_q:
            candidates.append((root + ('9' if displacement == 7 else 'maj7'), 0.95))
            candidates.append((root + '6', 0.8))
            candidates.append((root + 'add9', 0.7))
        elif is_min_q:
            candidates.append((root + 'm9', 0.95))
            candidates.append((root + 'm7', 0.85))
            candidates.append((root + 'm6', 0.7))
        elif quality == 'dim' or (is_dim_q and 9 not in ivs and 10 not in ivs):
            candidates.append((root + 'dim7', 0.85))
        elif 5 in ivs and 3 not in ivs and 4 not in ivs:
            candidates.append((root + '7sus4', 0.9))
        else:
            candidates.append((original_sym, 0.8))

    elif preset == 'emotional_pop':
        if is_dom_q:
            candidates.append((root + '7sus4', 0.85))
            candidates.append((root + '7', 0.8))
        elif is_maj_q:
            if displacement == 0:
                candidates.append((root + 'maj7', 0.95))
            else:
                candidates.append((root + 'add9', 0.9))
            candidates.append((root + 'sus2', 0.65))
            candidates.append((root, 0.6))
        elif is_min_q:
            candidates.append((root + 'm7', 0.9))
            candidates.append((root + 'm9', 0.75))
            candidates.append((root + 'm', 0.6))
        else:
            candidates.append((original_sym, 0.8))

    elif preset == 'walking_bass':
        third_semis, third_steps = (3, 2) if 3 in ivs else ((4, 2) if 4 in ivs else ((2, 1) if 2 in ivs else (5, 3)))
        fifth_semis = 6 if 6 in ivs else (8 if 8 in ivs else 7)
        _bass_3rd = _spell_from(root, third_semis, third_steps)
        _bass_5th = _spell_from(root, fifth_semis, 4)
        candidates.append((original_sym, 0.7))
        candidates.append((root + quality + '/' + _bass_3rd, 0.85))
        candidates.append((root + quality + '/' + _bass_5th, 0.75))
        if is_min_q or 10 in ivs:
            _bass_b7 = _spell_from(root, 10, 6)
            candidates.append((root + (quality if quality != 'm' else 'm7') + '/' + _bass_b7, 0.8))
        if is_maj_q and not is_dom_q:
            _bass_maj7 = _spell_from(root, 11, 6)
            candidates.append((root + 'maj7/' + _bass_maj7, 0.55))

    elif preset == 'suspend_dreamy':
        if is_dom_q:
            candidates.append((root + '7sus4', 0.9))
        elif is_maj_q or is_min_q:
            candidates.append((root + 'sus2', 0.95))
            candidates.append((root + 'add9', 0.75))
        else:
            candidates.append((original_sym, 0.8))
        candidates.append((root + 'sus4', 0.5))
        candidates.append((root + 'm7', 0.4))

    else:
        candidates.append((original_sym, 0.8))

    is_boundary_bar = (bar_idx == 0 or (total_bars > 1 and bar_idx == total_bars - 1))
    is_cadence_prep = (total_bars >= 4 and (bar_idx + 1) % 4 == 0)
    valid = []
    for sym, prior in candidates:
        p = parse_extended_chord(sym)
        if p is not None and generate_voicings(sym):
            adj_prior = float(prior)
            if is_boundary_bar and p['root'] == tonic_pc and p['bass_pc'] == tonic_pc:
                adj_prior += 0.08
            elif is_cadence_prep and ('sus' in p['quality'] or '7' in p['quality'] or p['bass'] is not None):
                adj_prior += 0.06
            valid.append((sym, adj_prior))
    if not valid:
        valid = [(original_sym, 0.5)]
    return valid


def _resolve_harmony(symbol, previous, following, tonic, notes, start, end):
    report = check_melody_clash(symbol, notes, start, end, tonic)
    if report['status'] != 'clash' and generate_voicings(symbol):
        return symbol, report
    parsed = _parse(symbol)
    root = parsed['root_name']
    is_min = (3 in parsed['intervals'] and 4 not in parsed['intervals']) or parsed['quality'] in _MINOR
    same_root_fallbacks = ([root + 'm7', root + 'm', root + 'madd9', root + 'sus2'] if is_min
                           else [root + 'add9', root, root + 'sus2', root + '6'])
    for alt in same_root_fallbacks:
        alt_rep = check_melody_clash(alt, notes, start, end, tonic)
        if alt_rep['status'] != 'clash' and generate_voicings(alt):
            return alt, alt_rep
    subs = suggest_bar_substitutions(symbol,previous,following,tonic,notes,start,end)
    for candidate in subs:
        if candidate['clash']['status'] != 'clash' and generate_voicings(candidate['chord']):
            return candidate['chord'], candidate['clash']
    for candidate in subs:
        if generate_voicings(candidate['chord']):
            return candidate['chord'], candidate['clash']
    return symbol, report


@lru_cache(maxsize=1024)
def roman_numeral(symbol: str, tonic: str = "C") -> str:
    """Compute functional Roman numeral label (e.g. Imaj7, V/7, vi7/5, IVadd9, iv6, II/#4) relative to tonic."""
    _note_pc(tonic)
    if isinstance(symbol, str) and symbol.strip() == 'N.C.':
        return 'N.C.'
    parsed = _parse(symbol)
    accidental, degree = _degree(parsed['root_name'],tonic)
    quality = parsed['quality']
    numeral = ('I','II','III','IV','V','VI','VII')[degree]
    if (3 in parsed['intervals'] and 4 not in parsed['intervals']) or quality in _MINOR | {'m7b5','dim','dim7'}:
        numeral = numeral.lower()
        if quality.startswith('min'):
            quality = quality[3:]
        elif quality.startswith('m') and not quality.startswith('maj'):
            quality = quality[1:]
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
    tonic: str = 'C',
) -> dict:
    """12-D TIS consonance + dyadic roughness clash detector with duration-weighted melody pitch-class distribution."""
    notes = _overlapping_notes(melody_notes, bar_start, bar_end)
    parsed = None if (isinstance(chord_symbol, str) and chord_symbol.strip() == 'N.C.') else _parse(chord_symbol)
    tones = set(parsed['pcs']) | {parsed['bass_pc']} if parsed else set()
    report = dict(status='safe', clashes=[], tensions=[], chord_tones=sorted(tones), notes_checked=len(notes),
                  clashing_pitches=[], reason='')
    if parsed is None or not notes:
        return report

    seg_len = max(1e-6, bar_end - bar_start)
    melody_weights = [0.0] * 12
    for n in notes:
        overlap = min(bar_end, n.start + n.duration) - max(bar_start, n.start)
        if overlap <= 0:
            continue
        beat_salience = 1.25 if abs(n.start - bar_start) < 0.13 else 1.0
        melody_weights[n.pitch % 12] += overlap * beat_salience

    tonic_pc = _note_pc(tonic)
    diatonic_pcs = {(tonic_pc + s) % 12 for s in _SCALE}
    chord_tiv = _chord_tiv(parsed)
    melody_tiv = _tonal_interval_vector(melody_weights)
    tis_sim = _tis_similarity(chord_tiv, melody_tiv)

    chord_tone_beats = sum(
        min(bar_end, n.start + n.duration) - max(bar_start, n.start)
        for n in notes if n.pitch % 12 in tones
    )

    clashing_pitches = []
    reasons = []
    for idx_n, n in enumerate(notes):
        pitch = n.pitch
        if pitch % 12 in tones:
            continue
        overlap_dur = min(bar_end, n.start + n.duration) - max(bar_start, n.start)
        if overlap_dur <= 0:
            continue

        prev_n = notes[idx_n - 1] if idx_n > 0 else None
        next_n = notes[idx_n + 1] if idx_n + 1 < len(notes) else None
        has_step_in = (
            prev_n is not None
            and abs((prev_n.start + prev_n.duration) - n.start) <= 0.08
            and 1 <= abs(prev_n.pitch - pitch) <= 2
            and (prev_n.pitch % 12) in tones
        )
        has_step_out = (
            next_n is not None
            and abs(next_n.start - (n.start + n.duration)) <= 0.08
            and 1 <= abs(next_n.pitch - pitch) <= 2
            and (next_n.pitch % 12) in tones
        )
        is_weak_passing = (
            overlap_dur <= 0.55
            and chord_tone_beats >= max(1.0, 2.0 * overlap_dur)
            and has_step_in
            and has_step_out
        )
        if is_weak_passing:
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=(pitch - parsed['root']) % 12, reason='passing'))
            continue

        roughness = _dyadic_roughness(tones, parsed, pitch % 12)
        interval = (pitch - parsed['root']) % 12
        reason = None
        if roughness >= 0.8:
            if (parsed['root'] + 4) % 12 in tones and interval == 5:
                reason = 'fourth_over_major_third'
            elif (parsed['root'] + 5) % 12 in tones and (parsed['root'] + 3) % 12 not in tones and interval == 4:
                reason = 'third_against_sus4'
            elif any((pitch - ct) % 12 == 1 for ct in tones):
                reason = 'minor_ninth'
            elif any((ct - pitch % 12) % 12 == 1 for ct in tones):
                reason = 'semitone_below_chord_tone'
            else:
                reason = 'dyadic_roughness'
        if reason:
            clashing_pitches.append(pitch)
            reasons.append(reason)
            report['clashes'].append(dict(pitch=pitch, start=n.start, interval=interval, reason=reason))
        else:
            labels = {1: 'b9', 2: '9', 3: '#9', 4: '3', 5: '11', 6: '#11', 7: '5', 8: 'b13', 9: '13', 10: 'b7', 11: 'maj7'}
            t_label = labels.get(interval, 'color')
            if (pitch % 12) not in diatonic_pcs:
                t_label = f'chromatic_{t_label}'
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=interval, reason=t_label))

    if clashing_pitches:
        report['status'] = 'clash'
        report['clashing_pitches'] = clashing_pitches
        report['reason'] = '; '.join(dict.fromkeys(reasons))
    elif report['tensions'] or tis_sim < 0.35:
        report['status'] = 'tension'
        report['reason'] = f'tis_similarity={tis_sim:.2f}' if tis_sim < 0.35 else 'color_tensions'
    else:
        report['status'] = 'safe'

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
    func_map = {0: 'T', 4: 'T', 9: 'T', 2: 'S', 5: 'S', 7: 'D', 11: 'D'}
    cur_func = func_map.get((current['root'] - tonic_pc) % 12)
    cur_pcs = set(current['pcs'])
    ranked = []
    for index, symbol in enumerate(symbols):
        parsed = _parse(symbol)
        pcs = set(parsed['pcs'])
        report = check_melody_clash(symbol,melody_notes,bar_start,bar_end,tonic)
        status = report['status']
        cand_func = func_map.get((parsed['root'] - tonic_pc) % 12)
        same_func = (parsed['root'] == current['root']) or (
            cur_func is not None and cand_func == cur_func and len(pcs & cur_pcs) >= 2
        )
        if status == 'clash':
            reason_label = '避用音冲突'
        elif status == 'tension':
            reason_label = '同功能张力代用' if same_func else '色彩张力代用'
        else:
            reason_label = '同功能安全代用' if same_func else '异功能协和代用'
        is_harsh_q = 1 if parsed['quality'] in {'aug', 'dim', 'dim7', 'm7b5'} else 0
        key = ({'safe':0,'tension':1,'clash':2}[status], 0 if same_func else 1, len(pcs-scale), is_harsh_q,
               len(pcs ^ cur_pcs), sum(len(pcs ^ set(n['pcs'])) for n in neighbors), index)
        ranked.append((key,dict(chord=symbol,roman=roman_numeral(symbol,tonic),clash=report,reason=reason_label)))
    return [record for _,record in sorted(ranked,key=lambda pair:pair[0])]


def _melody_tiv_for_bar(notes, bar_start, bar_end):
    weights = [0.0] * 12
    for n in notes:
        overlap = min(bar_end, n.start + n.duration) - max(bar_start, n.start)
        if overlap > 0:
            salience = 1.25 if abs(n.start - bar_start) < 0.13 else 1.0
            weights[n.pitch % 12] += overlap * salience
    return _tonal_interval_vector(weights), any(w > 0 for w in weights)


def _bass_pc_of(symbol):
    p = parse_extended_chord(symbol)
    return p['bass_pc'] if p else None


def _bass_transition_score(bass_a, bass_b, preset):
    if bass_a is None or bass_b is None:
        return 0.0
    delta = (bass_b - bass_a) % 12
    if delta in (5, 7):
        score = 0.6
    elif delta in (1, 2, 10, 11):
        score = 0.5 if preset == 'walking_bass' else 0.25
    elif delta == 0:
        score = -0.1
    elif delta == 6:
        score = -0.4
    else:
        score = 0.0
    if preset == 'walking_bass' and delta in (1, 2, 10, 11):
        score += 0.4
    return score


def _solve_split_bar_chord(chosen_sym, base_sym, next_base_sym, notes, bar_idx, measure, tonic):
    mid = bar_idx * measure + measure / 2.0
    bar_end = (bar_idx + 1) * measure
    chosen_parsed = parse_extended_chord(chosen_sym)
    if chosen_parsed is None:
        return None

    root = chosen_parsed['root_name']
    quality = chosen_parsed['quality']
    next_bass = _bass_pc_of(next_base_sym) if next_base_sym else chosen_parsed['root']

    candidates = []

    ivs = chosen_parsed['intervals']
    third_semis, third_steps = (3, 2) if 3 in ivs else ((4, 2) if 4 in ivs else ((2, 1) if 2 in ivs else (5, 3)))
    fifth_semis = 6 if 6 in ivs else (8 if 8 in ivs else 7)
    _3rd_name = _spell_from(root, third_semis, third_steps)
    _5th_name = _spell_from(root, fifth_semis, 4)

    inversions = [
        root + quality + '/' + _3rd_name,
        root + quality + '/' + _5th_name,
    ]
    if 10 in ivs or quality == 'm':
        inversions.append(root + (quality if quality != 'm' else 'm7') + '/' + _spell_from(root, 10, 6))
    elif 11 in ivs or quality == '':
        inversions.append(root + (quality if quality else 'maj7') + '/' + _spell_from(root, 11, 6))
    elif 9 in ivs and quality == 'dim7':
        inversions.append(root + 'dim7/' + _spell_from(root, 9, 6))

    for sym in inversions:
        if parse_extended_chord(sym) and generate_voicings(sym):
            candidates.append(sym)

    is_maj_split = (4 in ivs and 3 not in ivs) or quality in _MAJOR
    if is_maj_split:
        for q in ['sus4', 'sus2']:
            sym = root + q
            if parse_extended_chord(sym) and generate_voicings(sym):
                candidates.append(sym)
    if 5 in ivs and 10 in ivs and 3 not in ivs and 4 not in ivs:
        candidates.append(root + '7')
    if is_maj_split and 3 not in ivs and 4 in ivs:
        sym = root + 'm6'
        if parse_extended_chord(sym) and generate_voicings(sym):
            candidates.append(sym)

    if next_base_sym and next_base_sym.strip() != 'N.C.':
        next_parsed = parse_extended_chord(next_base_sym)
        if next_parsed:
            v_root_pc = (next_parsed['root'] + 7) % 12
            v_root_name = _NAMES[v_root_pc]
            v_3rd_name = _spell_from(v_root_name, 4, 2)
            for q in ['7', '', '/' + v_3rd_name]:
                sym = v_root_name + q
                if parse_extended_chord(sym) and generate_voicings(sym):
                    candidates.append(sym)

    if not candidates:
        return None

    best_sym = None
    best_score = -1e9
    for sym in dict.fromkeys(candidates):
        report = check_melody_clash(sym, notes, mid, bar_end, tonic)
        if report['status'] == 'clash':
            continue
        p = parse_extended_chord(sym)
        if p is None:
            continue
        chord_tiv = _chord_tiv(p)
        mel_tiv, has_mel = _melody_tiv_for_bar(notes, mid, bar_end)
        tis_score = _tis_similarity(chord_tiv, mel_tiv) if has_mel else 0.5
        bass_score = _bass_transition_score(p['bass_pc'], next_bass, 'walking_bass')
        score = tis_score * 2.0 + bass_score
        if score > best_score:
            best_score = score
            best_sym = sym

    return best_sym


def reharmonize_progression(
    base_chords: list[str],
    project,
    preset: str = "original",
    tonic: str | None = None,
    split_bar: bool = False,
) -> list[dict]:
    """Transform base chord progression using Viterbi-decoded TIS reharmonization with optional Split-Bar cadential solver."""
    if preset not in REHARMONIZE_PRESETS:
        raise ValueError(f'Unknown preset {preset!r}; supported: {", ".join(REHARMONIZE_PRESETS)}')
    tonic = project.tonic if tonic is None else tonic
    tonic_pc = _note_pc(tonic)
    if not base_chords:
        return []
    measure = _measure_beats(project)
    n_bars = len(base_chords)

    candidate_lists: list[list[tuple[str, float]]] = []
    melody_tivs: list[tuple[list[complex], bool]] = []
    clash_cache: dict[tuple[int, str], dict] = {}

    for idx, original in enumerate(base_chords):
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        mel_tiv, has_mel = _melody_tiv_for_bar(project.notes, bar_start, bar_end)
        melody_tivs.append((mel_tiv, has_mel))

        if isinstance(original, str) and original.strip() == 'N.C.':
            candidate_lists.append([('N.C.', 1.0)])
            continue
        parsed = _parse(original)

        cands = _generate_preset_candidates(parsed, tonic_pc, preset, idx, n_bars)
        if not cands:
            cands = [(original, 0.5)]
        candidate_lists.append(cands)

    for idx, cands in enumerate(candidate_lists):
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        for sym, _ in cands:
            if (idx, sym) not in clash_cache:
                clash_cache[(idx, sym)] = check_melody_clash(sym, project.notes, bar_start, bar_end, tonic)
        if preset != 'original' and all(clash_cache[(idx, s)].get('status') == 'clash' for s, _ in cands if s != 'N.C.'):
            base_sym = base_chords[idx]
            prev_c = base_chords[idx - 1] if idx > 0 else None
            next_c = base_chords[idx + 1] if idx + 1 < n_bars else None
            safe_sym, safe_rep = _resolve_harmony(base_sym, prev_c, next_c, tonic, project.notes, bar_start, bar_end)
            clash_cache[(idx, safe_sym)] = safe_rep
            if all(s != safe_sym for s, _ in cands):
                cands.append((safe_sym, 0.65))

    INF = float('inf')
    dp = [[INF] * len(candidate_lists[0])]
    back = [[0] * len(candidate_lists[0])]

    mel_tiv_0, has_mel_0 = melody_tivs[0]
    for j, (sym, prior) in enumerate(candidate_lists[0]):
        report = clash_cache.get((0, sym), {})
        is_clash = report.get('status') == 'clash'
        chord_tiv = _chord_tiv(parse_extended_chord(sym))
        tis_score = _tis_similarity(chord_tiv, mel_tiv_0) if has_mel_0 else 0.5
        emission = prior + tis_score * 1.5 + (-100.0 if is_clash else 0.0)
        dp[0][j] = -emission

    for b in range(1, n_bars):
        mel_tiv_b, has_mel_b = melody_tivs[b]
        n_cands = len(candidate_lists[b])
        dp.append([INF] * n_cands)
        back.append([0] * n_cands)
        for j, (sym_j, prior_j) in enumerate(candidate_lists[b]):
            report_j = clash_cache.get((b, sym_j), {})
            is_clash_j = report_j.get('status') == 'clash'
            chord_tiv_j = _chord_tiv(parse_extended_chord(sym_j))
            tis_score_j = _tis_similarity(chord_tiv_j, mel_tiv_b) if has_mel_b else 0.5
            emission_j = prior_j + tis_score_j * 1.5 + (-100.0 if is_clash_j else 0.0)

            bass_j = _bass_pc_of(sym_j)
            best_cost = INF
            best_prev = 0
            for k, (sym_k, _) in enumerate(candidate_lists[b - 1]):
                bass_k = _bass_pc_of(sym_k)
                trans = _bass_transition_score(bass_k, bass_j, preset)
                p_k = parse_extended_chord(sym_k)
                p_j = parse_extended_chord(sym_j)
                tis_trans = 0.0
                if p_k and p_j:
                    tis_trans = _tis_similarity(_chord_tiv(p_k), chord_tiv_j) * 0.3
                cost = dp[b - 1][k] - (trans + tis_trans) - emission_j
                if cost < best_cost:
                    best_cost = cost
                    best_prev = k
            dp[b][j] = best_cost
            back[b][j] = best_prev

    last_idx = min(range(len(candidate_lists[-1])), key=lambda j: dp[-1][j])
    chosen_indices = [0] * n_bars
    chosen_indices[-1] = last_idx
    for b in range(n_bars - 2, -1, -1):
        chosen_indices[b] = back[b + 1][chosen_indices[b + 1]]

    chosen_syms = []
    for b in range(n_bars):
        sym = candidate_lists[b][chosen_indices[b]][0]
        report = clash_cache.get((b, sym), {})
        if preset != 'original' and report.get('status') == 'clash':
            base_sym = base_chords[b]
            fallback_report = check_melody_clash(base_sym, project.notes, b * measure, (b + 1) * measure, tonic)
            if fallback_report['status'] != 'clash' and generate_voicings(base_sym):
                sym = base_sym
            else:
                prev_c = base_chords[b - 1] if b > 0 else None
                next_c = base_chords[b + 1] if b + 1 < n_bars else None
                safe_sym, _ = _resolve_harmony(base_sym, prev_c, next_c, tonic, project.notes, b * measure, (b + 1) * measure)
                sym = safe_sym
        chosen_syms.append(sym)

    bars = []
    flattened = []
    for idx in range(n_bars):
        original = base_chords[idx]
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        chosen = chosen_syms[idx]

        if split_bar and chosen.strip() != 'N.C.':
            next_base = base_chords[idx + 1] if idx + 1 < n_bars else None
            second_sym = _solve_split_bar_chord(chosen, original, next_base, project.notes, idx, measure, tonic)
            if second_sym and second_sym != chosen:
                half = measure / 2.0
                rep1 = check_melody_clash(chosen, project.notes, bar_start, bar_start + half, tonic)
                rep2 = check_melody_clash(second_sym, project.notes, bar_start + half, bar_end, tonic)
                if rep1['status'] != 'clash' and rep2['status'] != 'clash':
                    seg1 = dict(beat=0.0, duration=half, chord=chosen, roman=roman_numeral(chosen, tonic), frets=[], capo=0)
                    seg2 = dict(beat=half, duration=half, chord=second_sym, roman=roman_numeral(second_sym, tonic), frets=[], capo=0)
                    segments = [seg1, seg2]
                    reports = [rep1, rep2]
                    flattened.extend(segments)
                    aggregate = _build_aggregate(reports)
                    prev_c = base_chords[idx - 1] if idx > 0 else None
                    next_c = base_chords[idx + 1] if idx + 1 < n_bars else None
                    cands = suggest_bar_substitutions(chosen, prev_c, next_c, tonic, project.notes, bar_start, bar_end)[:8]
                    bars.append(dict(bar=idx, original_chord=original, chord=chosen, roman=roman_numeral(chosen, tonic),
                                     frets=[], clash=aggregate, split_segments=segments, candidates=cands))
                    continue

        report = clash_cache.get((idx, chosen))
        if report is None:
            report = check_melody_clash(chosen, project.notes, bar_start, bar_end, tonic)
        segment = dict(beat=0.0, duration=measure, chord=chosen, roman=roman_numeral(chosen, tonic), frets=[], capo=0)
        segments = [segment]
        flattened.append(segment)
        aggregate = _build_aggregate([report])
        prev_c = base_chords[idx - 1] if idx > 0 else None
        next_c = base_chords[idx + 1] if idx + 1 < n_bars else None
        parsed_orig = parse_extended_chord(original)
        cands = suggest_bar_substitutions(chosen, prev_c, next_c, tonic, project.notes, bar_start, bar_end)[:8] if parsed_orig else []
        bars.append(dict(bar=idx, original_chord=original, chord=chosen, roman=roman_numeral(chosen, tonic),
                         frets=[], clash=aggregate, split_segments=segments, candidates=cands))

    chord_seq = [s['chord'] for s in flattened]
    if chord_seq:
        voicings = optimize_voice_leading(chord_seq)
        for segment, voicing in zip(flattened, voicings):
            segment['frets'] = list(voicing['frets'])
    for bar in bars:
        bar['frets'] = list(bar['split_segments'][0]['frets'])

    return bars


def _build_aggregate(reports):
    aggregate = dict(status='safe', clashes=[], tensions=[],
                     chord_tones=sorted({p for r in reports for p in r.get('chord_tones', [])}),
                     notes_checked=sum(r.get('notes_checked', 0) for r in reports),
                     clashing_pitches=[], reason='')
    for r in reports:
        aggregate['clashes'].extend(r.get('clashes', []))
        aggregate['tensions'].extend(r.get('tensions', []))
        aggregate['clashing_pitches'].extend(r.get('clashing_pitches', []))
    if aggregate['clashes']:
        aggregate['status'] = 'clash'
        aggregate['reason'] = '; '.join(dict.fromkeys(c.get('reason', '') for c in aggregate['clashes']))
    elif aggregate['tensions'] or any(r.get('status') == 'tension' for r in reports):
        aggregate['status'] = 'tension'
        reasons = [r.get('reason', '') for r in reports if r.get('status') == 'tension' and r.get('reason')]
        aggregate['reason'] = '; '.join(dict.fromkeys(reasons)) if reasons else 'color_tensions'
    return aggregate
