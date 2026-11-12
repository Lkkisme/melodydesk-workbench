"""Functional Roman Numeral Analysis, 12-D TIS Consonance & Plomp-Levelt / Sethares Roughness + Avoid-Note Clash Detector, 6 Re-harmonization Presets with Viterbi Bass-Line & TIS Decoder, and Split-Bar Cadential / Secondary-Dominant Solver."""
from __future__ import annotations

import cmath
from functools import lru_cache
import math
from .chord_voicing import parse_extended_chord, _note_pc, EXTENDED_INTERVALS, generate_voicings, optimize_voice_leading

_SCALE = (0,2,4,5,7,9,11)
_MAJOR = {'','7','maj7','add9','add11','add#11','add2','add4','6','6/9','9','maj9','11','maj11','13','maj13','7b9','7#9','7b5','7#5','7#11','maj7#11','9#11','maj9#11','13#11','maj13#11','7b13','7alt','aug7'}
_MINOR = {'m','m7','madd9','madd11','m6','m6/9','m9','m11','m13','mmaj7','mmaj9','mmaj11','mmaj13'}
_TASTEFUL_SUB_QUALITIES = (
    '', 'm', '7', 'maj7', 'm7', 'sus2', 'sus4', '7sus4',
    'add9', 'madd9', '6', 'm6', '6/9', '9', 'maj9', 'm9', 'm7b5', 'dim7',
)
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


def _midi_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def _plomp_levelt_pair(p1: float, p2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    """Sethares (1993) parameterisation of the Plomp-Levelt (1965) sensory dissonance curve for two partials:
    d = min(a1, a2) * (exp(-3.5 s |p2-p1|) - exp(-5.75 s |p2-p1|)), s = 0.24 / (0.0207 min(p1, p2) + 18.96)."""
    s = 0.24 / (0.0207 * min(p1, p2) + 18.96)
    df = abs(p2 - p1)
    return min(a1, a2) * (math.exp(-3.5 * s * df) - math.exp(-5.75 * s * df))


@lru_cache(maxsize=8192)
def _plomp_levelt_roughness(f1: float, f2: float, n_partials: int = 5) -> float:
    """Multi-partial Plomp-Levelt / Sethares sensory dissonance between two harmonic complex tones.

    Each tone has partials h*f (h = 1..n_partials) with amplitudes a_h = 0.88^(h-1).  The dissonance is the
    sum over all cross-tone partial pairs of the Sethares pair curve (intra-tone partial pairs are identical
    for every interval, so they are omitted).
    """
    if f1 <= 0 or f2 <= 0 or n_partials <= 0:
        return 0.0
    total = 0.0
    for h1 in range(1, n_partials + 1):
        a1 = 0.88 ** (h1 - 1)
        for h2 in range(1, n_partials + 1):
            total += _plomp_levelt_pair(h1 * f1, h2 * f2, a1, 0.88 ** (h2 - 1))
    return total


@lru_cache(maxsize=1)
def _pl_reference():
    """Reference roughness scale: a minor second at C4/C#4 (the canonical rough dyad) maps to ~1.0."""
    return _plomp_levelt_roughness(_midi_hz(60), _midi_hz(61))


def _chord_voice_pitch(pc: int, melody_pitch: int) -> int:
    """Place chord pitch-class `pc` in the closest octave *below* the melody (typical accompaniment register),
    but never under the guitar's open low E (MIDI 40)."""
    below = melody_pitch - ((melody_pitch - pc) % 12)
    pitch = below if below < melody_pitch else below - 12
    while pitch < 40:
        pitch += 12
    return pitch


def _avoid_note_reason(chord_tones: set[int], parsed: dict, melody_pc: int) -> str | None:
    """Functional avoid-note rules from jazz / pop harmony pedagogy (chord-scale theory)."""
    root = parsed['root']
    interval = (melody_pc - root) % 12
    if (root + 4) % 12 in chord_tones and interval == 5 and (root + 5) % 12 not in chord_tones:
        return 'fourth_over_major_third'
    if (root + 5) % 12 in chord_tones and (root + 3) % 12 not in chord_tones \
            and (root + 4) % 12 not in chord_tones and interval == 4:
        return 'third_against_sus4'
    if any((melody_pc - ct) % 12 == 1 for ct in chord_tones):
        return 'minor_ninth'
    if (root + 11) % 12 in chord_tones and interval == 10:
        return 'minor_seventh_against_major_seventh'
    if (root + 4) % 12 in chord_tones and (root + 10) % 12 not in chord_tones and interval == 3:
        return 'minor_third_against_major_third'
    return None


def _dyadic_roughness(chord_tones: set[int], parsed: dict | None, melody_pc: int, melody_pitch: int | None = None) -> float:
    """Clash score of a melody note against a chord: functional avoid-note rules + acoustic roughness.

    The acoustic term is the multi-partial Plomp-Levelt / Sethares roughness of the melody note against each
    chord tone voiced in the octave below it, normalised so a C4-C#4 minor second ≈ 1.0.  Semitone above a
    chord tone (`b9`), third against `sus4`, minor seventh against `maj7`, or minor third against an
    un-altered major chord add a functional penalty of 1.0, and a perfect fourth over a major third adds 0.8.
    """
    if parsed is None:
        return 0.0
    if melody_pitch is not None and math.isfinite(float(melody_pitch)):
        mel_eval = int(round(float(melody_pitch)))
    else:
        mel_eval = 60 + (melody_pc % 12)
    if mel_eval < 57:
        mel_eval += 12 * ((57 - mel_eval + 11) // 12)
    f_mel = _midi_hz(mel_eval)
    ref = _pl_reference() or 1.0
    acoustic = 0.0
    functional = 0.0
    reason = _avoid_note_reason(chord_tones, parsed, melody_pc % 12)
    m3_pc = (parsed['root'] + 4) % 12
    for ct in chord_tones:
        if ct == melody_pc % 12:
            continue
        acoustic += _plomp_levelt_roughness(f_mel, _midi_hz(_chord_voice_pitch(ct, mel_eval))) / ref
        if (melody_pc - ct) % 12 == 1:
            if reason == 'fourth_over_major_third' and ct == m3_pc:
                continue
            functional += 1.0
    if reason == 'fourth_over_major_third':
        functional += 0.8
    elif reason in ('third_against_sus4', 'minor_seventh_against_major_seventh', 'minor_third_against_major_third'):
        functional += 1.0
    # Acoustic roughness contributes a graded share (a lone major second ≈ 0.3) without overriding the rules.
    return functional + 0.35 * acoustic


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
    if abs(diff) > 1:
        use_flat = ('b' in base_name) or (_note_pc(base_name) in (5, 10, 3, 8, 1))
        flat_names = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B')
        sharp_names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        return flat_names[target_pc] if use_flat else sharp_names[target_pc]
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
    semis = (_note_pc(name) - _note_pc(tonic)) % 12
    if semis in _SCALE:
        return '', _SCALE.index(semis)
    degree = ('CDEFGAB'.index(name[0].upper())-'CDEFGAB'.index(tonic[0].upper())) % 7
    difference = ((semis - _SCALE[degree] + 6) % 12) - 6
    if abs(difference) > 1:
        chrom_map = {1: ('b', 1), 3: ('b', 2), 6: ('#', 3), 8: ('b', 5), 10: ('b', 6)}
        return chrom_map[semis]
    accidental = '#' * difference if difference >= 0 else 'b' * -difference
    return accidental, degree


def _transform(parsed, tonic, preset):
    candidates = _generate_preset_candidates(parsed, _note_pc(tonic), preset, 0, 1)
    if candidates:
        return max(candidates, key=lambda item: item[1])[0]
    return parsed['root_name'] + parsed['quality'] + ('/' + parsed['bass'] if parsed['bass'] else '')


def _generate_preset_candidates(
    parsed: dict,
    tonic_pc: int,
    preset: str,
    bar_idx: int,
    total_bars: int,
    capo: int = 0,
) -> list[tuple[str, float]]:
    root = parsed['root_name']
    if '#' in root and tonic_pc in (5, 10, 3, 8, 1):
        root = {1: 'Db', 3: 'Eb', 6: 'Gb', 8: 'Ab', 10: 'Bb'}.get(parsed['root'], root)
    quality = parsed['quality']
    bass = parsed['bass']
    if bass and '#' in bass and tonic_pc in (5, 10, 3, 8, 1):
        bass = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}.get(bass, bass)
    displacement = (parsed['root'] - tonic_pc) % 12
    has_explicit_slash = bool(bass) and (parsed.get('bass_pc') != parsed.get('root'))
    original_sym = root + quality + ('/' + bass if bass else '')

    if preset == 'original':
        return [(original_sym, 1.0)]

    candidates: list[tuple[str, float]] = []

    ivs = parsed['intervals']
    r_pc = int(parsed['root'])
    scale_pcs = {(tonic_pc + s) % 12 for s in _SCALE}
    has_diatonic_maj7 = (11 in ivs) or (((r_pc + 11) % 12) in scale_pcs and displacement != 7)
    has_diatonic_b7 = (10 in ivs) or (((r_pc + 10) % 12) in scale_pcs)
    has_diatonic_9 = (2 in ivs) or (14 in ivs) or (((r_pc + 2) % 12) in scale_pcs)
    has_diatonic_6 = (9 in ivs) or (21 in ivs) or (((r_pc + 9) % 12) in scale_pcs)

    is_dim_q = (3 in ivs and 6 in ivs and 7 not in ivs)
    is_min_q = (3 in ivs and 4 not in ivs and not is_dim_q) or quality in _MINOR
    is_dom_q = (4 in ivs and 10 in ivs) or quality in {'7', '9', '11', '13'}
    is_maj_q = (4 in ivs and 3 not in ivs) or quality in _MAJOR
    is_borrowed_minor_iv = is_min_q and displacement == 5

    if preset == 'folk_easy':
        if is_maj_q:
            candidates.append((root, 1.0))
            if has_diatonic_9:
                candidates.append((root + 'add9', 0.7))
            elif has_diatonic_b7:
                candidates.append((root + '7', 0.68))
        elif is_borrowed_minor_iv:
            candidates.append((root + 'm', 1.0))
            candidates.append((root + 'm6', 0.78))
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
            if has_diatonic_9:
                candidates.append((root + '9', 0.9))
                candidates.append((root + '7', 0.85))
            else:
                candidates.append((root + '7', 0.92))
            candidates.append((root + '7sus4', 0.75))
        elif is_maj_q:
            if has_diatonic_maj7:
                candidates.append((root + 'maj7', 0.95))
                if has_diatonic_6:
                    candidates.append((root + '6', 0.8))
                if has_diatonic_9:
                    candidates.append((root + 'add9', 0.7))
                candidates.append((root, 0.65))
            else:
                if has_diatonic_b7 and has_diatonic_9:
                    candidates.append((root + '9', 0.95))
                    candidates.append((root + '7', 0.88))
                    candidates.append((root + 'add9', 0.72))
                elif has_diatonic_b7:
                    candidates.append((root + '7', 0.94))
                candidates.append((root, 0.82))
                candidates.append((root + '7sus4', 0.72))
        elif is_borrowed_minor_iv:
            # Borrowed minor iv in major key: m6 (diatonic 2nd) avoids m7/m9's b3/#9 clash against major tonic 3rd
            candidates.append((root + 'm6', 0.96))
            candidates.append((root + 'm', 0.88))
            candidates.append((root + 'madd9', 0.80))
        elif is_min_q and displacement == 4:
            # Phrygian iii in major key: m7/m stay diatonic (m9 adds #4 tritone; m6 adds b2 clash against tonic)
            candidates.append((root + 'm7', 0.95))
            candidates.append((root + 'm', 0.85))
        elif is_min_q and displacement == 9:
            # Aeolian vi in major key: m9/m7 stay diatonic (m6 adds #4 tritone)
            candidates.append((root + 'm9', 0.95))
            candidates.append((root + 'm7', 0.88))
            candidates.append((root + 'm', 0.78))
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
                candidates.append((root + 'add9', 0.88))
                candidates.append((root, 0.85))
                candidates.append((root + 'maj7', 0.82))
                candidates.append((root + 'sus2', 0.68))
            elif has_diatonic_9:
                candidates.append((root + 'add9', 0.9))
                candidates.append((root, 0.78))
                if has_diatonic_b7 and not has_diatonic_maj7:
                    candidates.append((root + '7', 0.72))
                candidates.append((root + 'sus2', 0.68))
            else:
                candidates.append((root, 0.9))
                if has_diatonic_b7:
                    candidates.append((root + '7', 0.86))
                elif has_diatonic_maj7:
                    candidates.append((root + 'maj7', 0.82))
        elif is_borrowed_minor_iv:
            candidates.append((root + 'm6', 0.92))
            candidates.append((root + 'm', 0.88))
            candidates.append((root + 'madd9', 0.82))
        elif is_min_q:
            candidates.append((root + 'm7', 0.9))
            candidates.append((root + 'm9', 0.75))
            candidates.append((root + 'm', 0.68))
        else:
            candidates.append((original_sym, 0.8))

    elif preset == 'walking_bass':
        candidates.append((original_sym, 0.94 if has_explicit_slash else 0.7))
        third_pair = (3, 2) if 3 in ivs else ((4, 2) if 4 in ivs else ((2, 1) if 2 in ivs else ((5, 3) if 5 in ivs else None)))
        if third_pair is not None:
            _bass_3rd = _spell_from(root, third_pair[0], third_pair[1])
            candidates.append((root + quality + '/' + _bass_3rd, 0.85))
        fifth_semis = 6 if 6 in ivs else (8 if 8 in ivs else (7 if 7 in ivs else None))
        if fifth_semis is not None:
            _bass_5th = _spell_from(root, fifth_semis, 4)
            candidates.append((root + quality + '/' + _bass_5th, 0.75))
        if 10 in ivs or quality == 'm' or (quality == '' and has_diatonic_b7 and not has_diatonic_maj7):
            _bass_b7 = _spell_from(root, 10, 6)
            _q_b7 = '7' if quality == '' else (quality if quality != 'm' else 'm7')
            candidates.append((root + _q_b7 + '/' + _bass_b7, 0.8))
        elif 11 in ivs or (quality == '' and has_diatonic_maj7):
            _bass_maj7 = _spell_from(root, 11, 6)
            candidates.append((root + (quality if quality else 'maj7') + '/' + _bass_maj7, 0.55))

    elif preset == 'suspend_dreamy':
        if is_dom_q:
            candidates.append((root + '7sus4', 0.9))
        elif is_maj_q and not has_diatonic_9:
            candidates.append((root + 'sus4', 0.92))
            if has_diatonic_b7:
                candidates.append((root + '7sus4', 0.85))
            candidates.append((root, 0.78))
        elif is_maj_q or is_min_q:
            candidates.append((root + 'sus2', 0.95))
            candidates.append((root + 'add9', 0.75))
        else:
            candidates.append((original_sym, 0.8))
        candidates.append((root + 'sus4', 0.5))
        if is_min_q:
            candidates.append((root + 'm7', 0.4))

    else:
        candidates.append((original_sym, 0.8))

    if has_explicit_slash and preset != 'walking_bass':
        slash_expanded: list[tuple[str, float]] = []
        root_fallbacks: list[tuple[str, float]] = []
        seen_syms: set[str] = set()
        for sym_c, pr_c in candidates:
            if '/' not in sym_c:
                sl_sym = sym_c + '/' + bass
                if sl_sym not in seen_syms:
                    slash_expanded.append((sl_sym, pr_c + 0.12))
                    seen_syms.add(sl_sym)
                if sym_c not in seen_syms:
                    root_fallbacks.append((sym_c, pr_c - 1.25))
                    seen_syms.add(sym_c)
            elif sym_c not in seen_syms:
                slash_expanded.append((sym_c, pr_c))
                seen_syms.add(sym_c)
        if original_sym not in seen_syms:
            slash_expanded.append((original_sym, 0.94))
        candidates = slash_expanded + root_fallbacks
    elif all(sym != original_sym for sym, _ in candidates):
        candidates.append((original_sym, 0.72))

    is_boundary_bar = (bar_idx == 0 or (total_bars > 1 and bar_idx == total_bars - 1))
    is_cadence_prep = (total_bars >= 4 and (bar_idx + 1) % 4 == 0)
    valid = []
    for sym, prior in candidates:
        p = parse_extended_chord(sym)
        if p is not None and generate_voicings(sym, capo):
            adj_prior = float(prior)
            if is_boundary_bar and p['root'] == tonic_pc and p['bass_pc'] == tonic_pc:
                adj_prior += 0.08
            elif is_cadence_prep and ('sus' in p['quality'] or '7' in p['quality'] or p['bass'] is not None):
                adj_prior += 0.06
            valid.append((sym, adj_prior))
    if not valid:
        fallback_root = (root + 'm') if is_min_q else root
        if parse_extended_chord(original_sym) is not None and generate_voicings(original_sym, capo):
            valid = [(original_sym, 0.5)]
        elif parse_extended_chord(fallback_root) is not None and generate_voicings(fallback_root, capo):
            valid = [(fallback_root, 0.5)]
        else:
            valid = [(original_sym, 0.5)]
    else:
        valid.sort(key=lambda item: item[1], reverse=True)
    return valid


def _resolve_harmony(symbol, previous, following, tonic, notes, start, end, capo: int = 0):
    report = check_melody_clash(symbol, notes, start, end, tonic)
    if report['status'] != 'clash' and generate_voicings(symbol, capo):
        return symbol, report
    parsed = _parse(symbol)
    root = parsed['root_name']
    bass = parsed['bass']
    has_slash = bool(bass) and (parsed.get('bass_pc') != parsed.get('root'))
    tonic_pc = _note_pc(tonic)
    scale = {(tonic_pc + x) % 12 for x in _SCALE}
    is_min = (3 in parsed['intervals'] and 4 not in parsed['intervals']) or parsed['quality'] in _MINOR
    base_fallbacks = (
        [root + 'm', root + 'm6', root + 'm7', root + 'madd9']
        if (is_min and ((parsed['root'] - tonic_pc) % 12) == 5)
        else ([root + 'm', root + 'm7', root + 'madd9', root + 'm6']
              if is_min
              else [root + 'add9', root, root + 'sus4', root + 'sus2', root + '6', root + 'maj7'])
    )
    same_root_fallbacks = ([fb + '/' + bass for fb in base_fallbacks] + base_fallbacks) if has_slash else base_fallbacks
    for alt in same_root_fallbacks:
        if parse_extended_chord(alt) is None:
            continue
        alt_rep = check_melody_clash(alt, notes, start, end, tonic)
        if alt_rep['status'] != 'clash' and generate_voicings(alt, capo):
            return alt, alt_rep
    subs = suggest_bar_substitutions(symbol, previous, following, tonic, notes, start, end)
    orig_pcs = set(parsed['pcs'])
    # Prefer in-key substitutions that share at least one chord tone with the original harmony before any chromatic root shift
    for candidate in subs:
        if candidate['clash']['status'] == 'clash' or not generate_voicings(candidate['chord'], capo):
            continue
        p_cand = parse_extended_chord(candidate['chord'])
        if p_cand and len(set(p_cand['pcs']) - scale) == 0 and len(set(p_cand['pcs']) & orig_pcs) >= 1:
            return candidate['chord'], candidate['clash']
    for candidate in subs:
        if candidate['clash']['status'] != 'clash' and generate_voicings(candidate['chord'], capo):
            return candidate['chord'], candidate['clash']
    for candidate in subs:
        if generate_voicings(candidate['chord'], capo):
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
    """Melody/chord clash detector: functional avoid-note rules + multi-partial Plomp-Levelt / Sethares sensory
    roughness per non-chord melody note, plus 12-D TIS similarity of the duration-weighted melody pitch-class
    distribution to the chord."""
    notes = _overlapping_notes(melody_notes, bar_start, bar_end)
    all_notes = [
        n for n in (melody_notes or [])
        if math.isfinite(getattr(n, 'start', math.nan))
        and math.isfinite(getattr(n, 'duration', math.nan))
        and math.isfinite(getattr(n, 'pitch', math.nan))
        and n.duration > 0
    ]
    all_notes.sort(key=lambda n: (n.start, n.duration, n.pitch))
    parsed = None if (isinstance(chord_symbol, str) and chord_symbol.strip() == 'N.C.') else _parse(chord_symbol)
    tones = set(parsed['pcs']) | {parsed['bass_pc']} if parsed else set()
    report = dict(status='safe', clashes=[], tensions=[], chord_tones=sorted(tones), notes_checked=len(notes),
                  clashing_pitches=[], reason='')
    if parsed is None or not notes:
        return report

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

        full_idx = next(
            (k for k, m in enumerate(all_notes) if m is n or (abs(m.start - n.start) <= 1e-9 and abs(m.duration - n.duration) <= 1e-9 and m.pitch == n.pitch)),
            None,
        )
        if full_idx is not None:
            prev_n = all_notes[full_idx - 1] if full_idx > 0 else None
            next_n = all_notes[full_idx + 1] if full_idx + 1 < len(all_notes) else None
        else:
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
        not_downbeat = abs(n.start - bar_start) >= 0.20
        avoid_tag = _avoid_note_reason(tones, parsed, pitch % 12)
        is_weak_passing = (
            (
                overlap_dur <= 1.05
                and not_downbeat
                and has_step_in
                and has_step_out
                and chord_tone_beats >= max(1.0, overlap_dur)
            )
            or (
                overlap_dur <= 0.55
                and not_downbeat
                and chord_tone_beats >= max(1.0, 2.0 * overlap_dur)
                and (
                    (has_step_in or has_step_out)
                    if avoid_tag == 'minor_ninth'
                    else (has_step_in or has_step_out or chord_tone_beats >= 1.5)
                )
            )
        )
        if is_weak_passing:
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=(pitch - parsed['root']) % 12, reason='passing'))
            continue

        roughness = _dyadic_roughness(tones, parsed, pitch % 12, pitch)
        interval = (pitch - parsed['root']) % 12
        is_minor_chord = (3 in parsed['intervals'] and 4 not in parsed['intervals'])
        if avoid_tag is None and is_minor_chord and interval in (2, 5):
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=interval,
                                           reason='9' if interval == 2 else '11', roughness=round(roughness, 3)))
            continue
        if avoid_tag == 'fourth_over_major_third' and overlap_dur <= 0.85 and chord_tone_beats >= max(0.75, overlap_dur):
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=interval, reason='passing_11',
                                           roughness=round(roughness, 3)))
            continue
        reason = None
        if roughness >= 0.8:
            reason = avoid_tag or 'sensory_roughness'
        if reason:
            clashing_pitches.append(pitch)
            reasons.append(reason)
            report['clashes'].append(dict(pitch=pitch, start=n.start, interval=interval, reason=reason,
                                          roughness=round(roughness, 3)))
        else:
            labels = {1: 'b9', 2: '9', 3: '#9', 4: '3', 5: '11', 6: '#11', 7: '5', 8: 'b13', 9: '13', 10: 'b7', 11: 'maj7'}
            t_label = labels.get(interval, 'color')
            if (pitch % 12) not in diatonic_pcs:
                t_label = f'chromatic_{t_label}'
            report['tensions'].append(dict(pitch=pitch, start=n.start, interval=interval, reason=t_label,
                                           roughness=round(roughness, 3)))

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
    degree_steps = ((0, 0), (2, 1), (4, 2), (5, 3), (7, 4), (9, 5), (11, 6),
                    (1, 1), (3, 2), (6, 3), (8, 5), (10, 6))
    step_by_semis = dict(degree_steps)
    root_names = [_spell_from(tonic, semis, steps) for semis, steps in degree_steps]
    raw_symbols = [current_chord] + [
        r_name + q for r_name in root_names for q in _TASTEFUL_SUB_QUALITIES if q in EXTENDED_INTERVALS
    ]
    symbols = []
    seen_chords = set()
    for sym in raw_symbols:
        p_sym = parse_extended_chord(sym)
        if p_sym is None:
            continue
        canon_key = (p_sym['root'], p_sym['quality'], p_sym['bass_pc'])
        if canon_key not in seen_chords:
            seen_chords.add(canon_key)
            if p_sym['bass'] is None:
                semis_off = (p_sym['root'] - tonic_pc) % 12
                sym = _spell_from(tonic, semis_off, step_by_semis[semis_off]) + p_sym['quality']
            symbols.append(sym)
    valid_bounds = math.isfinite(bar_start) and math.isfinite(bar_end) and bar_end > bar_start
    all_n = sorted([n for n in (melody_notes or []) if math.isfinite(getattr(n,'start',math.nan)) and math.isfinite(getattr(n,'duration',math.nan)) and math.isfinite(getattr(n,'pitch',math.nan)) and n.duration > 0], key=lambda n: (n.start, n.duration, n.pitch)) if valid_bounds else []
    ov = {i for i, n in enumerate(all_n) if n.start < bar_end and n.start + n.duration > bar_start}
    bar_notes = [all_n[i] for i in sorted(ov | {i - 1 for i in ov if i > 0} | {i + 1 for i in ov if i + 1 < len(all_n)})] if valid_bounds else melody_notes
    func_map = {0: 'T', 4: 'T', 9: 'T', 2: 'S', 5: 'S', 7: 'D', 11: 'D'}
    cur_func = func_map.get((current['root'] - tonic_pc) % 12)
    cur_pcs = set(current['pcs'])
    ranked = []
    for index, symbol in enumerate(symbols):
        parsed = _parse(symbol)
        pcs = set(parsed['pcs'])
        report = check_melody_clash(symbol,bar_notes,bar_start,bar_end,tonic)
        status = report['status']
        cand_func = func_map.get((parsed['root'] - tonic_pc) % 12)
        same_func = (parsed['root'] == current['root']) or (
            cur_func is not None and cand_func == cur_func and len(pcs & cur_pcs) >= 2
        )
        reason_label = ('避用音冲突' if status == 'clash' else
                        ('同功能张力代用' if same_func else '色彩张力代用') if status == 'tension'
                        else ('同功能安全代用' if same_func else '异功能协和代用'))
        is_harsh_q = (2 if parsed['quality'] in {'5', 'aug', 'dim', 'dim7', 'm7b5', '7alt', '7b9', '7#9', '7b5', '7#5', '7b13', 'aug7', 'mmaj7', 'mmaj9', 'mmaj11', 'mmaj13', '13#11', 'maj13#11'}
                      else 1 if ('#11' in parsed['quality'] or 'add11' in parsed['quality'] or 'maj11' in parsed['quality']) else 0)
        out_of_key = len(pcs - scale)
        diatonic_tier = 0 if out_of_key == 0 else (1 if parsed['root'] == current['root'] else 2)
        key = ({'safe':0,'tension':1,'clash':2}[status], diatonic_tier, 0 if same_func else 1, out_of_key, is_harsh_q,
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
        score = 0.9 if preset == 'walking_bass' else 0.25
    elif delta == 0:
        score = -0.1
    elif delta == 6:
        score = -0.4
    else:
        score = 0.0
    return score


def _solve_split_bar_chord(
    chosen_sym,
    next_sym,
    notes,
    bar_idx,
    measure,
    tonic,
    preset='walking_bass',
    capo: int = 0,
    split_beat: float | None = None,
):
    d1_local = (measure / 2.0) if split_beat is None else float(split_beat)
    mid = bar_idx * measure + d1_local
    bar_end = (bar_idx + 1) * measure
    chosen_parsed = parse_extended_chord(chosen_sym)
    if chosen_parsed is None:
        return None

    root = chosen_parsed['root_name']
    tonic_pc = _note_pc(tonic) if tonic else None
    if '#' in root and tonic_pc in (5, 10, 3, 8, 1):
        root = {1: 'Db', 3: 'Eb', 6: 'Gb', 8: 'Ab', 10: 'Bb'}.get(chosen_parsed['root'], root)
    quality = chosen_parsed['quality']
    next_bass = _bass_pc_of(next_sym) if next_sym else chosen_parsed['root']

    candidates = []

    ivs = chosen_parsed['intervals']
    inversions = []
    third_pair = (3, 2) if 3 in ivs else ((4, 2) if 4 in ivs else ((2, 1) if 2 in ivs else ((5, 3) if 5 in ivs else None)))
    if third_pair is not None:
        inversions.append(root + quality + '/' + _spell_from(root, third_pair[0], third_pair[1]))
    fifth_semis = 6 if 6 in ivs else (8 if 8 in ivs else (7 if 7 in ivs else None))
    if fifth_semis is not None:
        inversions.append(root + quality + '/' + _spell_from(root, fifth_semis, 4))
    if 10 in ivs or quality == 'm':
        inversions.append(root + (quality if quality != 'm' else 'm7') + '/' + _spell_from(root, 10, 6))
    elif 11 in ivs or quality == '':
        inversions.append(root + (quality if quality else 'maj7') + '/' + _spell_from(root, 11, 6))
    elif 9 in ivs and quality == 'dim7':
        inversions.append(root + 'dim7/' + _spell_from(root, 9, 6))

    for sym in inversions:
        if parse_extended_chord(sym) and generate_voicings(sym, capo):
            candidates.append(sym)

    is_maj_split = (4 in ivs and 3 not in ivs) or quality in _MAJOR
    if is_maj_split:
        for q in ['sus4', 'sus2']:
            sym = root + q
            if parse_extended_chord(sym) and generate_voicings(sym, capo):
                candidates.append(sym)
    if 5 in ivs and 10 in ivs and 3 not in ivs and 4 not in ivs:
        sym7 = root + '7'
        if parse_extended_chord(sym7) and generate_voicings(sym7, capo):
            candidates.append(sym7)
    if is_maj_split and 3 not in ivs and 4 in ivs:
        sym = root + 'm6'
        if parse_extended_chord(sym) and generate_voicings(sym, capo):
            candidates.append(sym)

    if next_sym and next_sym.strip() != 'N.C.':
        next_parsed = parse_extended_chord(next_sym)
        if next_parsed:
            v_root_name = _spell_from(next_parsed['root_name'], 7, 4)
            v_3rd_name = _spell_from(v_root_name, 4, 2)
            for q in ['7', '', '/' + v_3rd_name]:
                sym = v_root_name + q
                if parse_extended_chord(sym) and generate_voicings(sym, capo):
                    candidates.append(sym)

    if not candidates:
        return None

    chosen_cand_sym = None
    max_cand_score = -1e9
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
        bass_score = _bass_transition_score(p['bass_pc'], next_bass, preset)
        score = tis_score * 2.0 + bass_score
        if score > max_cand_score:
            max_cand_score = score
            chosen_cand_sym = sym

    return chosen_cand_sym


def reharmonize_progression(
    base_chords: list[str],
    project,
    preset: str = "original",
    tonic: str | None = None,
    split_bar: bool = False,
    capo: int = 0,
    detected_split_chords: dict | None = None,
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
    melody_pc_shares: list[list[float]] = []
    clash_cache: dict[tuple[int, str], dict] = {}

    for idx, original in enumerate(base_chords):
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        mel_tiv, has_mel = _melody_tiv_for_bar(project.notes, bar_start, bar_end)
        melody_tivs.append((mel_tiv, has_mel))
        pc_w = [0.0] * 12
        for n_item in project.notes:
            ov_dur = min(bar_end, n_item.start + n_item.duration) - max(bar_start, n_item.start)
            if ov_dur > 0:
                pc_w[n_item.pitch % 12] += ov_dur
        tot_w = sum(pc_w)
        melody_pc_shares.append([w / tot_w for w in pc_w] if tot_w > 1e-6 else [0.0] * 12)

        if isinstance(original, str) and original.strip() == 'N.C.':
            candidate_lists.append([('N.C.', 1.0)])
            continue
        parsed = _parse(original)

        cands = _generate_preset_candidates(parsed, tonic_pc, preset, idx, n_bars, capo=capo)
        if not cands:
            cands = [(original, 0.5)]
        candidate_lists.append(cands)

    for idx, cands in enumerate(candidate_lists):
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        for sym, _ in cands:
            if (idx, sym) not in clash_cache:
                clash_cache[(idx, sym)] = check_melody_clash(sym, project.notes, bar_start, bar_end, tonic)
        # Only rescue bars whose real chord candidates all clash; an all-N.C. bar (all([]) == True) is left alone.
        non_nc_syms = [s for s, _ in cands if s != 'N.C.']
        if preset != 'original' and non_nc_syms and all(clash_cache[(idx, s)].get('status') == 'clash' for s in non_nc_syms):
            base_sym = base_chords[idx]
            prev_c = base_chords[idx - 1] if idx > 0 else None
            next_c = base_chords[idx + 1] if idx + 1 < n_bars else None
            safe_sym, safe_rep = _resolve_harmony(base_sym, prev_c, next_c, tonic, project.notes, bar_start, bar_end, capo=capo)
            clash_cache[(idx, safe_sym)] = safe_rep
            if all(s != safe_sym for s, _ in cands):
                cands.append((safe_sym, 0.65))

    INF = float('inf')
    dp = [[INF] * len(candidate_lists[0])]
    back = [[0] * len(candidate_lists[0])]

    def _maj7_root_rub_penalty(bar_i: int, cand_parsed: dict | None) -> float:
        if preset == 'original' or cand_parsed is None or 'maj7' not in cand_parsed.get('quality', ''):
            return 0.0
        orig_p = parse_extended_chord(base_chords[bar_i])
        if orig_p is not None and 'maj7' in orig_p.get('quality', ''):
            return 0.0
        r_share = melody_pc_shares[bar_i][cand_parsed['root'] % 12]
        return -0.45 * r_share if r_share >= 0.35 else 0.0

    mel_tiv_0, has_mel_0 = melody_tivs[0]
    for j, (sym, prior) in enumerate(candidate_lists[0]):
        report = clash_cache.get((0, sym), {})
        is_clash = report.get('status') == 'clash'
        p_0 = parse_extended_chord(sym)
        chord_tiv = _chord_tiv(p_0)
        tis_score = _tis_similarity(chord_tiv, mel_tiv_0) if has_mel_0 else 0.5
        emission = prior + tis_score * 1.5 + _maj7_root_rub_penalty(0, p_0) + (-100.0 if is_clash else 0.0)
        dp[0][j] = -emission

    _COLOR_REPEAT_QUALS = {'maj7', 'maj9', 'sus2', 'sus4', 'm9', '9'}
    for b in range(1, n_bars):
        mel_tiv_b, has_mel_b = melody_tivs[b]
        n_cands = len(candidate_lists[b])
        dp.append([INF] * n_cands)
        back.append([0] * n_cands)
        for j, (sym_j, prior_j) in enumerate(candidate_lists[b]):
            report_j = clash_cache.get((b, sym_j), {})
            is_clash_j = report_j.get('status') == 'clash'
            p_j = parse_extended_chord(sym_j)
            chord_tiv_j = _chord_tiv(p_j)
            tis_score_j = _tis_similarity(chord_tiv_j, mel_tiv_b) if has_mel_b else 0.5
            emission_j = prior_j + tis_score_j * 1.5 + _maj7_root_rub_penalty(b, p_j) + (-100.0 if is_clash_j else 0.0)

            bass_j = _bass_pc_of(sym_j)
            min_path_cost = INF
            prev_cand_idx = 0
            for k, (sym_k, _) in enumerate(candidate_lists[b - 1]):
                bass_k = _bass_pc_of(sym_k)
                trans = _bass_transition_score(bass_k, bass_j, preset)
                p_k = parse_extended_chord(sym_k)
                tis_trans = 0.0
                if p_k and p_j:
                    tis_trans = _tis_similarity(_chord_tiv(p_k), chord_tiv_j) * 0.3
                    if (
                        preset != 'original'
                        and p_k['root'] == p_j['root']
                        and p_k['quality'] == p_j['quality']
                        and p_j['quality'] in _COLOR_REPEAT_QUALS
                    ):
                        trans -= 0.25
                cost = dp[b - 1][k] - (trans + tis_trans) - emission_j
                if cost < min_path_cost:
                    min_path_cost = cost
                    prev_cand_idx = k
            dp[b][j] = min_path_cost
            back[b][j] = prev_cand_idx

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
            if fallback_report['status'] != 'clash' and generate_voicings(base_sym, capo):
                sym = base_sym
            else:
                prev_c = chosen_syms[b - 1] if b > 0 else None
                next_c = candidate_lists[b + 1][chosen_indices[b + 1]][0] if b + 1 < n_bars else None
                safe_sym, _ = _resolve_harmony(base_sym, prev_c, next_c, tonic, project.notes, b * measure, (b + 1) * measure, capo=capo)
                sym = safe_sym
        chosen_syms.append(sym)

    meter_str = str(project.meter).strip()
    n_slots_m = int(round(measure * 2.0))
    if meter_str.endswith('/8') and n_slots_m >= 6 and n_slots_m % 3 == 0:
        n_pulses = n_slots_m // 3
        split_d1 = float(math.ceil(n_pulses / 2.0)) * 1.5 if n_pulses >= 2 else (measure / 2.0)
    elif measure >= 2.0:
        split_d1 = float(math.ceil(measure / 2.0))
    else:
        split_d1 = measure / 2.0
    det_map = detected_split_chords if isinstance(detected_split_chords, dict) else {}

    def _adapt_half_chord(raw_sym: str, seg_s: float, seg_e: float, bar_i: int) -> str | None:
        p_raw = parse_extended_chord(raw_sym)
        if p_raw is None:
            return None
        if preset == 'original':
            return raw_sym if generate_voicings(raw_sym, capo) else None
        mel_tiv_h, has_mel_h = _melody_tiv_for_bar(project.notes, seg_s, seg_e)
        best_sym_h = None
        best_sc_h = -1e9
        for cand_sym, prior_h in _generate_preset_candidates(p_raw, tonic_pc, preset, bar_i, n_bars, capo=capo):
            if not generate_voicings(cand_sym, capo):
                continue
            rep_h = check_melody_clash(cand_sym, project.notes, seg_s, seg_e, tonic)
            if rep_h['status'] == 'clash':
                continue
            p_c = parse_extended_chord(cand_sym)
            tis_h = _tis_similarity(_chord_tiv(p_c), mel_tiv_h) if has_mel_h else 0.5
            sc_h = prior_h + 1.5 * tis_h
            if sc_h > best_sc_h:
                best_sc_h = sc_h
                best_sym_h = cand_sym
        if best_sym_h is not None:
            return best_sym_h
        if generate_voicings(raw_sym, capo) and check_melody_clash(raw_sym, project.notes, seg_s, seg_e, tonic)['status'] != 'clash':
            return raw_sym
        return None

    bars = []
    flattened = []
    for idx in range(n_bars):
        original = base_chords[idx]
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        chosen = chosen_syms[idx]

        det_pair = det_map.get(idx) if idx in det_map else det_map.get(str(idx))
        if (
            isinstance(det_pair, (list, tuple))
            and len(det_pair) == 2
            and measure >= 2.0
            and chosen.strip() != 'N.C.'
        ):
            d1 = split_d1
            d2 = measure - d1
            c1_use = _adapt_half_chord(str(det_pair[0]).strip(), bar_start, bar_start + d1, idx)
            c2_use = _adapt_half_chord(str(det_pair[1]).strip(), bar_start + d1, bar_end, idx)
            if c1_use and c2_use and c1_use != c2_use:
                rep1 = check_melody_clash(c1_use, project.notes, bar_start, bar_start + d1, tonic)
                rep2 = check_melody_clash(c2_use, project.notes, bar_start + d1, bar_end, tonic)
                if preset == 'original' or (rep1['status'] != 'clash' and rep2['status'] != 'clash'):
                    seg1 = dict(beat=0.0, duration=d1, chord=c1_use, roman=roman_numeral(c1_use, tonic), frets=[], capo=capo)
                    seg2 = dict(beat=d1, duration=d2, chord=c2_use, roman=roman_numeral(c2_use, tonic), frets=[], capo=capo)
                    segments = [seg1, seg2]
                    flattened.extend(segments)
                    aggregate = _build_aggregate([rep1, rep2])
                    prev_c = chosen_syms[idx - 1] if idx > 0 else None
                    next_c = chosen_syms[idx + 1] if idx + 1 < n_bars else None
                    cands = suggest_bar_substitutions(c1_use, prev_c, next_c, tonic, project.notes, bar_start, bar_end)[:8]
                    bars.append(dict(bar=idx, original_chord=original, chord=c1_use, roman=roman_numeral(c1_use, tonic),
                                     frets=[], clash=aggregate, split_segments=segments, candidates=cands))
                    continue

        if split_bar and (n_bars < 2 or idx < n_bars - 1) and chosen.strip() != 'N.C.':
            next_sym = chosen_syms[idx + 1] if idx + 1 < n_bars else None
            d1 = split_d1
            d2 = measure - d1
            second_sym = _solve_split_bar_chord(
                chosen, next_sym, project.notes, idx, measure, tonic, preset=preset, capo=capo, split_beat=d1
            )
            if second_sym and second_sym != chosen:
                rep1 = check_melody_clash(chosen, project.notes, bar_start, bar_start + d1, tonic)
                rep2 = check_melody_clash(second_sym, project.notes, bar_start + d1, bar_end, tonic)
                if rep1['status'] != 'clash' and rep2['status'] != 'clash':
                    seg1 = dict(beat=0.0, duration=d1, chord=chosen, roman=roman_numeral(chosen, tonic), frets=[], capo=capo)
                    seg2 = dict(beat=d1, duration=d2, chord=second_sym, roman=roman_numeral(second_sym, tonic), frets=[], capo=capo)
                    segments = [seg1, seg2]
                    reports = [rep1, rep2]
                    flattened.extend(segments)
                    aggregate = _build_aggregate(reports)
                    prev_c = chosen_syms[idx - 1] if idx > 0 else None
                    next_c = chosen_syms[idx + 1] if idx + 1 < n_bars else None
                    cands = suggest_bar_substitutions(chosen, prev_c, next_c, tonic, project.notes, bar_start, bar_end)[:8]
                    bars.append(dict(bar=idx, original_chord=original, chord=chosen, roman=roman_numeral(chosen, tonic),
                                     frets=[], clash=aggregate, split_segments=segments, candidates=cands))
                    continue

        report = clash_cache.get((idx, chosen))
        if report is None:
            report = check_melody_clash(chosen, project.notes, bar_start, bar_end, tonic)
        segment = dict(beat=0.0, duration=measure, chord=chosen, roman=roman_numeral(chosen, tonic), frets=[], capo=capo)
        segments = [segment]
        flattened.append(segment)
        aggregate = _build_aggregate([report])
        prev_c = chosen_syms[idx - 1] if idx > 0 else None
        next_c = chosen_syms[idx + 1] if idx + 1 < n_bars else None
        parsed_orig = parse_extended_chord(original)
        cands = suggest_bar_substitutions(chosen, prev_c, next_c, tonic, project.notes, bar_start, bar_end)[:8] if parsed_orig else []
        bars.append(dict(bar=idx, original_chord=original, chord=chosen, roman=roman_numeral(chosen, tonic),
                         frets=[], clash=aggregate, split_segments=segments, candidates=cands))

    chord_seq = [s['chord'] for s in flattened]
    if chord_seq:
        voicings = optimize_voice_leading(chord_seq, capo=capo)
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


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _tri_peak(x: float, peak: float = 0.25, upper: float = 0.60) -> float:
    if x <= 0.0 or x >= upper:
        return 0.0
    if x <= peak:
        return x / max(1e-6, peak)
    return (upper - x) / max(1e-6, upper - peak)


_AS_MAJOR = (0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081)
_AS_MINOR = (0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.072, 0.043)


def _albrecht_shanahan_minor_confidence(notes: list) -> float:
    """Return Albrecht-Shanahan 12-PC minor-mode confidence in [0.0, 1.0] (0.0 when major wins or notes empty)."""
    if not notes:
        return 0.0
    chroma = [0.0] * 12
    for n in notes:
        conf = max(0.35, float(getattr(n, 'confidence', 1.0)))
        dur = float(getattr(n, 'duration', 1.0))
        chroma[int(n.pitch) % 12] += dur * conf
    last_pc = int(notes[-1].pitch) % 12
    first_pc = int(notes[0].pitch) % 12
    best_score, top_minor, top_corr = -2.0, False, 0.0
    for k in range(12):
        rotated = [chroma[(k + d) % 12] for d in range(12)]
        mean_r = sum(rotated) / 12.0
        var_r = sum((x - mean_r) ** 2 for x in rotated)
        for is_minor, profile in ((False, _AS_MAJOR), (True, _AS_MINOR)):
            mean_p = sum(profile) / 12.0
            var_p = sum((p - mean_p) ** 2 for p in profile)
            cov = sum((rotated[i] - mean_r) * (profile[i] - mean_p) for i in range(12))
            denom = math.sqrt(var_r * var_p) if var_r > 0 and var_p > 0 else 1e-12
            raw_corr = cov / denom
            score = raw_corr + (0.12 if last_pc == k else 0.0) + (0.06 if first_pc == k else 0.0)
            if score > best_score:
                best_score = score
                top_minor = is_minor
                top_corr = raw_corr
    return min(1.0, max(0.0, float(top_corr))) if top_minor else 0.0


def recommend_artistry(
    project,
    base_chords: list[str] | None,
    bar_dynamics: list[float] | None = None,
) -> dict:
    """Recommend one of the 4 UI artistry cards + groove style from harmony, meter, tempo, melody, and optional bar-dynamics features."""
    from .arrangement_grooves import ADVANCED_STYLES

    meter_str = str(project.meter).strip()
    m_num, m_den = map(int, meter_str.split('/'))
    m_num = max(1, m_num)
    m_den = max(1, m_den)
    measure = m_num * 4.0 / m_den
    is_compound = (m_den == 8 and m_num % 3 == 0)
    is_triple = (m_num == 3 and m_den == 4)
    f_tri = 1.0 if is_triple else 0.0
    f_cmp = 1.0 if is_compound else 0.0
    f_dup = 1.0 if (not is_triple and not is_compound) else 0.0

    bpm = max(30.0, float(getattr(project, 'bpm', 100.0) or 100.0))
    pulse_bpm = bpm * (2.0 / 3.0) if is_compound else bpm
    f_slow = _sigmoid((84.0 - pulse_bpm) / 8.0)
    f_drive = _sigmoid((pulse_bpm - 96.0) / 10.0)

    tonic_str = str(project.tonic).strip()
    t_clean = tonic_str
    for suf in ('minor', 'min', 'major', 'maj', 'm'):
        if len(t_clean) > len(suf) and t_clean.lower().endswith(suf):
            t_clean = t_clean[:-len(suf)].strip()
            break
    t_pc = _note_pc(t_clean)
    maj_scale = {(t_pc + s) % 12 for s in _SCALE}
    pent_scale = {(t_pc + s) % 12 for s in (0, 2, 4, 7, 9)}

    raw_chords = [str(c).strip() for c in (base_chords or []) if str(c).strip() and str(c).strip() != 'N.C.']
    parsed_list = []
    for sym in raw_chords:
        p = parse_extended_chord(sym)
        if p is not None:
            parsed_list.append((sym, p))

    n_ch = len(parsed_list)
    if n_ch > 0:
        ext_vals = []
        chr_cnt = 0
        c145_cnt = 0
        inv_cnt = 0
        modal_cnt = 0
        vi_w = 0.0
        one_w = 0.0
        for idx_c, (sym, p) in enumerate(parsed_list):
            deg = (p['root'] - t_pc) % 12
            qual = p['quality']
            pcs = set(p['pcs'])
            if qual == '7' and deg == 7:
                ext_vals.append(0.25)
            else:
                ext_vals.append(min(1.0, max(0.0, (len(pcs) - 3) / 2.0)))
            if any(pc not in maj_scale for pc in pcs):
                chr_cnt += 1
            if deg in (0, 5, 7) and qual in {'', '7', 'sus2', 'sus4', '7sus4', 'add9', 'maj7', '6'}:
                c145_cnt += 1
            if p['bass_pc'] != p['root']:
                inv_cnt += 1
            next_deg = (parsed_list[idx_c + 1][1]['root'] - t_pc) % 12 if idx_c + 1 < n_ch else -1
            if (deg == 10 and qual in _MAJOR) or (deg == 2 and qual in _MAJOR and next_deg != 7):
                modal_cnt += 1
            bar_w = 2.0 if (idx_c == 0 or idx_c == n_ch - 1 or (idx_c + 1) % 4 == 0) else 1.0
            if deg == 9 and qual in _MINOR:
                vi_w += bar_w
            elif deg == 0 and qual in _MAJOR:
                one_w += bar_w
        f_ext = sum(ext_vals) / n_ch
        f_chr = chr_cnt / n_ch
        f_c145 = c145_cnt / n_ch
        f_inv = inv_cnt / n_ch
        f_voc = min(1.0, max(0.0, (len({s for s, _ in parsed_list}) - 3.0) / 5.0))
        f_modal = modal_cnt / n_ch
        harm_min = vi_w / (vi_w + one_w) if (vi_w + one_w) > 0 else 0.0
    else:
        f_ext = f_chr = f_inv = f_voc = f_modal = harm_min = 0.0
        f_c145 = 0.5

    notes = list(getattr(project, 'notes', None) or [])
    has_notes = len(notes) > 0
    mel_min = 0.0
    f_syn = 0.0
    f_s16 = 0.0
    f_pent = 0.0
    f_long = 0.0
    if has_notes:
        mel_min = _albrecht_shanahan_minor_confidence(notes)

        def _grid_weight(pos_in_bar: float) -> float:
            slot16 = int(round(pos_in_bar * 4.0)) % max(1, int(round(measure * 4.0)))
            if slot16 == 0:
                return 1.0
            if is_compound:
                if slot16 % 6 == 0:
                    return 0.75
                if slot16 % 2 == 0:
                    return 0.35
                return 0.125
            half_s = int(round(measure * 2.0))
            if half_s > 0 and slot16 == half_s and int(round(measure)) % 2 == 0:
                return 0.75
            if slot16 % 4 == 0:
                return 0.50
            if slot16 % 2 == 0:
                return 0.25
            return 0.125

        syn_vals = []
        s16_cnt = 0
        pent_dur = 0.0
        long_dur = 0.0
        tot_dur = 0.0
        for n in notes:
            st = float(n.start)
            dur = max(1e-4, float(n.duration))
            tot_dur += dur
            if (int(n.pitch) % 12) in pent_scale:
                pent_dur += dur
            if dur >= 1.5 - 1e-6:
                long_dur += dur
            w_on = _grid_weight(st % measure)
            if w_on <= 0.125 + 1e-6:
                s16_cnt += 1
            k_step = math.ceil((st + 1e-6) * 4.0)
            end_step = math.floor((st + dur - 1e-6) * 4.0)
            w_held = w_on
            for gs in range(k_step, min(k_step + 32, end_step + 1)):
                w_held = max(w_held, _grid_weight((gs * 0.25) % measure))
            syn_vals.append(max(0.0, w_held - w_on) / 0.25)
        f_syn = min(1.0, sum(syn_vals) / len(syn_vals))
        f_s16 = s16_cnt / len(notes)
        f_pent = pent_dur / max(1e-6, tot_dur)
        f_long = long_dur / max(1e-6, tot_dur)

    f_min = (0.6 * harm_min + 0.4 * mel_min) if (n_ch > 0 and has_notes) else (harm_min if n_ch > 0 else mel_min)
    has_dyn = bar_dynamics is not None and len(bar_dynamics) >= 2
    if has_dyn:
        d_sorted = sorted(float(x) for x in bar_dynamics)
        p10 = d_sorted[max(0, int(0.10 * (len(d_sorted) - 1)))]
        p90 = d_sorted[min(len(d_sorted) - 1, int(round(0.90 * (len(d_sorted) - 1))))]
        f_dyn = min(1.0, max(0.0, (p90 - p10) / 0.20))
    else:
        f_dyn = 0.0

    feats = dict(
        tri=f_tri, cmp=f_cmp, dup=f_dup, slow=f_slow, drive=f_drive,
        ext=f_ext, chr=f_chr, c145=f_c145, inv=f_inv, voc=f_voc,
        min=f_min, modal=f_modal, syn=f_syn, s16=f_s16, pent=f_pent,
        long=f_long, dyn=f_dyn,
    )

    mel_gate = 1.0 if has_notes else 0.0
    dyn_gate = 1.0 if has_dyn else 0.0
    ch_gate = 1.0 if n_ch >= 8 else (n_ch / (n_ch + 4.0))

    folk_terms = [
        ('基础主属和弦清晰', 1.35 * f_c145 * ch_gate),
        ('三和弦自然质朴', 0.75 * max(0.0, 1.0 - 2.5 * f_ext) * ch_gate),
        ('自然音阶顺畅纯净', 0.80 * max(0.0, 1.0 - 3.5 * f_chr) * ch_gate),
        ('五声音阶旋律感强', 0.65 * f_pent * mel_gate),
        ('四拍律动明快', 0.40 * f_dup + 0.50 * f_drive),
        ('明亮大调色彩', 0.25 * (1.0 - f_min)),
        ('和声简洁易上手', (-0.60 * f_inv - 0.65 * f_voc - 1.20 * f_chr) * ch_gate),
        ('律动平稳少切分', -0.85 * f_syn * mel_gate - 0.55 * f_s16 * mel_gate),
    ]
    orig_terms = [
        ('三拍子古典叙事律动', 1.50 * f_tri),
        ('小调叙事气质鲜明', 0.95 * f_min),
        ('中古调式色彩独特', 1.00 * f_modal * ch_gate),
        ('正拍稳重少切分', 0.55 * (1.0 - f_syn) * mel_gate + 0.25 * (1.0 - f_s16) * mel_gate),
        ('原曲纯净三和弦质感', 0.45 * max(0.0, 1.0 - 2.0 * f_ext) * ch_gate),
        ('从容叙事速度', -0.50 * f_drive - 0.45 * f_chr * ch_gate),
    ]
    pop_terms = [
        ('舒缓抒情慢板', 1.05 * f_slow),
        ('流行叙事拍号', 0.80 * f_cmp + 0.45 * f_dup),
        ('长音抒情线条丰富', 1.10 * f_long * mel_gate),
        ('段落情绪起伏明显', 0.65 * f_dyn * dyn_gate),
        ('顺滑低音转位连接', (0.80 * f_inv + 0.25 * f_voc) * ch_gate),
        ('适度七九色彩点缀', 0.65 * _tri_peak(f_ext, 0.22, 0.55) * ch_gate),
        ('旋律平稳深情', -0.90 * max(0.0, f_syn - 0.42) * mel_gate - 0.60 * max(0.0, f_s16 - 0.22) * mel_gate - 0.60 * f_modal * ch_gate),
    ]
    jazz_terms = [
        ('七九延伸和弦丰富', 1.80 * f_ext * ch_gate),
        ('半音与借用和弦色彩浓', 2.00 * f_chr * ch_gate),
        ('R&B 切分律动突出', 1.25 * f_syn * mel_gate + 0.85 * f_s16 * mel_gate),
        ('和声语汇丰富灵动', 0.55 * f_voc * ch_gate + 0.45 * (1.0 - f_long) * f_syn * mel_gate),
        ('现代律动拍感', 0.35 * f_dup - 0.45 * f_long * f_slow * mel_gate),
        ('非传统民谣三拍框架', -1.10 * f_tri - 0.90 * max(0.0, f_c145 - 0.60) * ch_gate),
    ]

    term_map = {
        'folk_easy': folk_terms,
        'original': orig_terms,
        'emotional_pop': pop_terms,
        'neosoul_jazz': jazz_terms,
    }
    raw_scores = {k: sum(v for _, v in terms) for k, terms in term_map.items()}

    max_s = max(raw_scores.values())
    exps = {k: math.exp((v - max_s) / 0.5) for k, v in raw_scores.items()}
    sum_exp = sum(exps.values()) or 1.0
    probs = {k: exps[k] / sum_exp for k in raw_scores}
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    top_preset, top_prob = ranked[0]
    second_preset, second_prob = ranked[1]
    close_tie = (top_prob - second_prob) < 0.10

    if top_preset == 'folk_easy':
        rec_style = '扫弦伴奏' if f_drive >= 0.35 else '分解和弦'
    elif top_preset == 'original':
        rec_style = '指弹律动' if (f_dup > 0.5 and pulse_bpm >= 90.0) else '分解和弦'
    elif top_preset == 'emotional_pop':
        rec_style = '抒情琶音'
    else:
        rec_style = 'Neo-Soul 律动' if f_dup > 0.5 else '抒情琶音'
    if rec_style not in ADVANCED_STYLES:
        raise ValueError(f'Recommended style {rec_style!r} not in ADVANCED_STYLES')

    pos_reasons = [lbl for lbl, val in sorted(term_map[top_preset], key=lambda x: -x[1]) if val > 0.12][:2]
    reason_core = '、'.join(pos_reasons) if pos_reasons else '贴合本曲和声与旋律气质'
    if close_tie:
        second_label = REHARMONIZE_PRESETS[second_preset]['name'].split(' ')[0]
        reason_zh = f'{reason_core}（与「{second_label}」两者皆宜）'
    else:
        reason_zh = reason_core

    return {
        'preset': top_preset,
        'secondary_preset': second_preset if close_tie else None,
        'style': rec_style,
        'reason': reason_zh,
        'scores': {k: round(v, 4) for k, v in probs.items()},
        'features': {k: round(v, 3) for k, v in feats.items()},
    }
