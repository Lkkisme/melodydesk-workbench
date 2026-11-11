"""Extended & Slash Chord Interval Table, Voicing Feasibility Validator, Fretboard Geometry Heuristic, and Voice-Leading Optimizer."""
from __future__ import annotations

import math
import re
from functools import lru_cache

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
EXTENDED_INTERVALS = {
    "": (0,4,7), "m": (0,3,7), "7": (0,4,7,10),
    "maj7": (0,4,7,11), "m7": (0,3,7,10), "m7b5": (0,3,6,10),
    "dim": (0,3,6), "dim7": (0,3,6,9), "aug": (0,4,8),
    "sus2": (0,2,7), "sus4": (0,5,7), "sus": (0,5,7), "7sus4": (0,5,7,10), "7sus2": (0,2,7,10),
    "9sus4": (0,5,7,10,14), "13sus4": (0,5,7,10,14,21),
    "add9": (0,4,7,14), "madd9": (0,3,7,14), "add11": (0,4,7,17), "madd11": (0,3,7,17),
    "add#11": (0,4,7,18), "add2": (0,2,4,7), "add4": (0,4,5,7), "5": (0,7),
    "6": (0,4,7,9), "m6": (0,3,7,9), "6/9": (0,4,7,9,14), "m6/9": (0,3,7,9,14),
    "9": (0,4,7,10,14), "maj9": (0,4,7,11,14), "m9": (0,3,7,10,14),
    "11": (0,4,7,10,14,17), "maj11": (0,4,7,11,14,17), "m11": (0,3,7,10,14,17),
    "13": (0,4,7,10,14,21), "maj13": (0,4,7,11,14,21), "m13": (0,3,7,10,14,21),
    "7b9": (0,4,7,10,13), "7#9": (0,4,7,10,15), "7b5": (0,4,6,10), "7#5": (0,4,8,10),
    "7#11": (0,4,7,10,18), "maj7#11": (0,4,7,11,18), "9#11": (0,4,7,10,14,18), "maj9#11": (0,4,7,11,14,18),
    "13#11": (0,4,7,10,14,18,21), "maj13#11": (0,4,7,11,14,18,21),
    "7b13": (0,4,7,10,20), "7alt": (0,4,10,13,15,20),
    "aug7": (0,4,8,10), "mmaj7": (0,3,7,11), "mmaj9": (0,3,7,11,14),
    "mmaj11": (0,3,7,11,14,17), "mmaj13": (0,3,7,11,14,21),
}


def _note_pc(name):
    if not isinstance(name, str):
        raise ValueError(f'Invalid note: {name!r}')
    norm = name.replace('♯', '#').replace('♭', 'b').replace('𝄪', '##').replace('𝄫', 'bb')
    if not re.fullmatch(r'[A-Ga-g](?:#+|b+)?', norm):
        raise ValueError(f'Invalid note: {name!r}')
    acc = norm[1:]
    offset = len(acc) if acc.startswith('#') else (-len(acc) if acc.startswith('b') else 0)
    return ({'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[norm[0].upper()] + offset) % 12


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'Invalid {name}: {value!r}')
    return value


def _fingers(frets):
    count, barre = 0, False
    for fret in sorted(set(f for f in frets if f > 0)):
        previous, size = None, 0
        for index, value in enumerate(frets):
            if value != fret:
                continue
            if previous is None or any(0 <= x < fret for x in frets[previous+1:index]):
                count += 1
                size = 0
            size += 1
            barre |= size > 1
            previous = index
    return count, barre


def _required(parsed):
    required = set(parsed['pcs'])
    if len(required) >= 4 and 7 in parsed['intervals']:
        required.discard((parsed['root'] + 7) % 12)
    if len(required) >= 5 and 14 in parsed['intervals'] and any(x >= 17 for x in parsed['intervals']):
        required.discard((parsed['root'] + 2) % 12)
    bass_set = {parsed['bass_pc']}
    if len(required | bass_set) > 4:
        for drop_iv in (14, 13, 15, 17, 18, 20, 21, 9, 6, 8, 2, 5, 4, 3):
            if len(required | bass_set) <= 4:
                break
            if drop_iv not in parsed['intervals']:
                continue
            pc_drop = (parsed['root'] + drop_iv) % 12
            if pc_drop != parsed['root'] and pc_drop != parsed['bass_pc']:
                required.discard(pc_drop)
    return required


def _make_voicing(symbol, frets, capo, label):
    _integer(capo, 0, 12, 'capo')
    if len(frets) != 6:
        raise ValueError('Frets must contain six integers')
    for fret in frets:
        _integer(fret, -1, 24-capo, 'fret')
    parsed = parse_extended_chord(symbol)
    rest = isinstance(symbol, str) and symbol.strip() == 'N.C.'
    if parsed is None and not rest:
        raise ValueError(f'Invalid chord: {symbol!r}')
    pitches = [t+capo+f if f >= 0 else -1 for t,f in zip(STANDARD_TUNING, frets)]
    active = [p for p in pitches if p >= 0]
    positive = [f for f in frets if f > 0]
    span = max(positive)-min(positive) if positive else 0
    fingers, barre = _fingers(frets)
    if rest:
        if active:
            raise ValueError('N.C. must be muted')
    elif (len(active) < 3 or active[0] != min(active)
          or min(active) % 12 != parsed['bass_pc']
          or not {p%12 for p in active} <= set(parsed['pcs']) | {parsed['bass_pc']}
          or not _required(parsed) <= {p%12 for p in active}
          or span > 4 or fingers > 4):
        raise ValueError(f'Unplayable or harmonically invalid voicing for {symbol}: {frets}')
    return dict(frets=list(frets), pitches=pitches, base_fret=1 if 0 in frets else min(positive) if positive else 1,
                barre=barre, fingers=fingers, span=span, label=label)


_SHAPES = {
    'C':[-1,3,2,0,1,0], 'Cm':[-1,3,5,5,4,3], 'Cmaj7':[-1,3,2,0,0,0],
    'Cadd9':[-1,3,2,0,3,0], 'Csus2':[-1,3,0,0,1,3], 'Csus4':[-1,3,3,0,1,1],
    'C7':[-1,3,2,3,1,0], 'Cm7':[-1,3,5,3,4,3], 'Cmaj9':[-1,3,2,4,3,0],
    'D':[-1,-1,0,2,3,2], 'Dm':[-1,-1,0,2,3,1], 'D7':[-1,-1,0,2,1,2],
    'Dm7':[-1,-1,0,2,1,1], 'Dmaj7':[-1,-1,0,2,2,2], 'Dsus2':[-1,-1,0,2,3,0],
    'Dsus4':[-1,-1,0,2,3,3], 'Dadd9':[-1,5,4,2,3,0], 'D6':[-1,-1,0,2,0,2],
    'E':[0,2,2,1,0,0], 'Em':[0,2,2,0,0,0], 'E7':[0,2,0,1,0,0],
    'Em7':[0,2,2,0,3,0], 'Emaj7':[0,2,1,1,0,0], 'Esus4':[0,2,2,2,0,0],
    'Em9':[0,2,0,0,0,2], 'Em6':[0,2,2,0,2,0],
    'F':[1,3,3,2,1,1], 'Fmaj7':[-1,-1,3,2,1,0], 'Fm6':[1,-1,0,1,1,1],
    'Fadd9':[-1,-1,3,2,1,3], 'Fsus2':[-1,-1,3,0,1,1], 'Fm':[1,3,3,1,1,1],
    'G':[3,2,0,0,0,3], 'G7':[3,2,0,0,0,1], 'G7sus4':[3,3,0,0,1,1],
    'Gadd9':[3,0,0,2,0,3], 'Gsus2':[3,0,0,0,3,3], 'Gsus4':[3,3,0,0,1,3], 'Gmaj7':[3,2,0,0,0,2], 'G6':[3,2,0,0,0,0],
    'A':[-1,0,2,2,2,0], 'Am':[-1,0,2,2,1,0], 'Am7':[-1,0,2,0,1,0],
    'Amaj7':[-1,0,2,1,2,0], 'Asus2':[-1,0,2,2,0,0], 'Asus4':[-1,0,2,2,3,0], 'A6':[-1,0,2,2,2,2],
    'Amadd9':[-1,0,2,4,1,0], 'Am9':[-1,0,5,5,5,7],
    'B':[-1,2,4,4,4,2], 'Bm':[-1,2,4,4,3,2], 'Bm7':[-1,2,4,2,3,2],
    'B7':[-1,2,1,2,0,2],
    'C6':[-1,3,2,2,1,0], 'Cm6':[-1,3,1,2,1,3], 'C9':[-1,3,2,3,3,3],
    'C11':[-1,3,2,3,3,1], 'Cm11':[-1,3,1,3,3,1], 'Cmaj11':[-1,3,2,4,3,1],
    'C13':[-1,3,2,3,3,5], 'Cm13':[-1,3,1,3,3,5], 'Cmaj13':[-1,3,2,2,0,0],
    'C6/9':[-1,3,2,2,3,3], 'Cm6/9':[-1,3,1,2,3,3],
    'Cdim7':[-1,3,4,2,4,-1], 'Cm7b5':[-1,3,4,3,4,-1],
    'C/E':[0,3,2,0,1,0], 'G/B':[-1,2,0,0,0,3], 'Am7/G':[3,0,2,0,1,0],
    'D/F#':[2,-1,0,2,3,2], 'F/G':[3,-1,3,2,1,1], 'C/G':[3,3,2,0,1,0],
}


def _internal_mutes(frets):
    active_idx = [i for i, f in enumerate(frets) if f >= 0]
    if len(active_idx) < 2:
        return 0
    lo, hi = active_idx[0], active_idx[-1]
    return sum(1 for f in frets[lo:hi+1] if f < 0)


# --- Physical fretboard geometry (steel-string dreadnought reference) ---------------------------
SCALE_LENGTH_MM = 648.0          # nut-to-saddle scale length L
STRING_SPACING_NUT_MM = 7.2      # centre-to-centre string pitch at the nut (≈43 mm nut width)
STRING_SPACING_SADDLE_MM = 10.5  # centre-to-centre string pitch at the saddle
# Reference fingertip-to-fingertip reach thresholds (mm) per finger pair (1=index, 2=middle, 3=ring, 4=pinky).
FINGER_PAIR_COMFORT_MM = {
    (1, 2): 38.0, (2, 3): 32.0, (3, 4): 28.0, (1, 4): 85.0,
    (1, 3): 62.0, (2, 4): 52.0,
}
_SPAN_REFERENCE_MM = 32.0        # ≈ one first-position fret width; normalises hand-span stretch term


def _fret_position_mm(fret: float, capo: int = 0) -> float:
    """Distance from the nut (mm) of the fret wire for `fret` frets above the capo: x(f) = L(1 - 2^(-(f+capo)/12))."""
    return SCALE_LENGTH_MM * (1.0 - 2.0 ** (-(fret + capo) / 12.0))


def _fingertip_position_mm(fret: int, capo: int) -> float:
    """Fingertip contact point: just behind the fret wire (≈35% of the way back into the fret slot)."""
    if fret <= 0:
        return _fret_position_mm(0, capo)
    wire = _fret_position_mm(fret, capo)
    previous_wire = _fret_position_mm(fret - 1, capo)
    return wire - 0.35 * (wire - previous_wire)


def _string_spacing_mm(x_mm: float) -> float:
    ratio = min(1.0, max(0.0, x_mm / SCALE_LENGTH_MM))
    return STRING_SPACING_NUT_MM + (STRING_SPACING_SADDLE_MM - STRING_SPACING_NUT_MM) * ratio


def _pair_strain_mm(assignment, capo):
    """Heuristic squared excess fingertip distance (mm) over each finger pair's reference reach threshold.

    Distance combines the nearest contact point and the fingertip anchor (lowest-indexed string held by a
    barre) in Euclidean longitudinal/transverse millimetre coordinates.  A higher-numbered finger placed
    nearer the nut than a lower one is a crossed-finger posture and pays a fixed penalty.
    """
    strain = 0.0
    fingers = sorted(assignment)
    for a_idx, da in enumerate(fingers):
        sa_tip, fa_tip = min(assignment[da], key=lambda item: item[0])
        xa_tip = _fingertip_position_mm(fa_tip, capo)
        for db in fingers[a_idx + 1:]:
            comfort = FINGER_PAIR_COMFORT_MM[(da, db)]
            sb_tip, fb_tip = min(assignment[db], key=lambda item: item[0])
            xb_tip = _fingertip_position_mm(fb_tip, capo)
            anchor_d = math.hypot(
                xb_tip - xa_tip,
                abs(sb_tip - sa_tip) * _string_spacing_mm(0.5 * (xa_tip + xb_tip)),
            )
            min_d, backward = float('inf'), False
            for sa, fa in assignment[da]:
                xa = _fingertip_position_mm(fa, capo)
                for sb, fb in assignment[db]:
                    xb = _fingertip_position_mm(fb, capo)
                    longitudinal = xb - xa
                    transverse = abs(sb - sa) * _string_spacing_mm(0.5 * (xa + xb))
                    d = math.hypot(longitudinal, transverse)
                    if d < min_d:
                        min_d, backward = d, longitudinal < -1e-9
            eff_d = 0.5 * (min_d + anchor_d)
            strain += 0.6 * (max(0.0, eff_d - comfort) / 10.0) ** 2
            if backward:
                strain += 1.0
    return strain


_PARTIAL_BARRE_MM_PENALTY = 0.15  # per extra string held by one non-index finger (mini-barre)


def _barre_feasible(frets, strings, fret):
    """One finger flattened across `strings` at `fret` must not cover a sounding string at a lower fret."""
    lo, hi = min(strings), max(strings)
    return all(frets[k] < 0 or frets[k] >= fret for k in range(lo, hi + 1))


def _same_fret_groupings(notes):
    """Yield partitions of (fret, string)-sorted notes into consecutive same-fret groups."""
    if not notes:
        yield []
        return
    fret = notes[0][0]
    run = 1
    while run < len(notes) and notes[run][0] == fret:
        run += 1
    for size in range(1, run + 1):
        head = notes[:size]
        for tail in _same_fret_groupings(notes[size:]):
            yield [head] + tail


@lru_cache(maxsize=8192)
def _assign_fingers(frets: tuple, capo: int):
    """Order-preserving 4-finger assignment heuristic (1=index, 2=middle, 3=ring, 4=pinky).

    Fretted notes are sorted by (fret, string_idx) and receive strictly increasing finger numbers, so the
    index finger takes the lowest-fret group (as a barre when it holds several strings).  Notes sharing a
    fret may be held by one finger (full or partial barre) when no lower-fretted sounding string lies under
    it.  Among all order-preserving assignments - including which fingers are skipped, e.g. index + ring
    across a two-fret gap - the one with the smallest millimetre pair-distance penalty is chosen.
    Returns (assignment: finger -> [(string_idx, fret)], overflow: notes beyond four fingers, strain).
    """
    from itertools import combinations
    notes = sorted((f, i) for i, f in enumerate(frets) if f > 0)
    if not notes:
        return {}, 0, 0.0
    chosen, chosen_strain = None, float('inf')
    for groups in _same_fret_groupings(notes):
        if len(groups) > 4:
            continue
        if any(len(g) > 1 and not _barre_feasible(frets, [i for _, i in g], g[0][0]) for g in groups):
            continue
        # Index finger is pinned to the lowest-fret group; the remaining groups choose from middle/ring/pinky.
        for combo in combinations((2, 3, 4), len(groups) - 1):
            assignment = {1: [(i, f) for f, i in groups[0]]}
            for finger, group in zip(combo, groups[1:]):
                assignment[finger] = [(i, f) for f, i in group]
            strain = _pair_strain_mm(assignment, capo)
            strain += sum(_PARTIAL_BARRE_MM_PENALTY * (len(g) - 1) for g in groups[1:])
            if strain < chosen_strain:
                chosen, chosen_strain = assignment, strain
    if chosen is not None:
        return chosen, 0, chosen_strain
    # No four-finger solution: index takes the lowest fret, the next three notes take 2/3/4, the rest overflow.
    min_f = notes[0][0]
    assignment = {1: [(i, f) for f, i in notes if f == min_f]}
    rest = [(f, i) for f, i in notes if f != min_f]
    for finger, (fret, string_idx) in zip((2, 3, 4), rest):
        assignment[finger] = [(string_idx, fret)]
    return assignment, max(0, len(rest) - 3), _pair_strain_mm(assignment, capo)


def _fretboard_stretch_mm(frets, capo):
    """Millimetre fretboard-geometry stretch proxy for the fretting hand.

    Fret positions follow x(f) = L(1 - 2^(-(f+capo)/12)) with L = 648 mm.  Returns
    (hand_span_mm, span_energy, pair_strain, position_energy, overflow).
    """
    positive = [f for f in frets if f > 0]
    if not positive:
        return 0.0, 0.0, 0.0, 0.0, 0
    xs = [_fingertip_position_mm(f, capo) for f in positive]
    hand_span_mm = max(xs) - min(xs)
    span_energy = 0.15 * (hand_span_mm / _SPAN_REFERENCE_MM) ** 2
    # Arm reach / hand rotation away from the first position, in mm from the nut.
    # 0.0075/mm ≈ 1.15 at the 5th fret and ≈ 2.3 at the 12th (body joint), preferring open-position resonance.
    position_energy = 0.0075 * (sum(xs) / len(xs))
    _, overflow, pair_strain = _assign_fingers(tuple(frets), capo)
    return hand_span_mm, span_energy, pair_strain, position_energy, overflow


def _ergonomic_voicing_cost(frets, parsed, capo):
    """Total voicing preference cost combining millimetre fretboard-stretch penalties, harmonic completeness,
    bass-root fidelity, and internal-mute / richness / string-crossing penalties.  Lower is better."""
    sounding = [i for i, f in enumerate(frets) if f >= 0]
    if not sounding:
        return 100.0
    pitches = [STANDARD_TUNING[i] + capo + frets[i] for i in sounding]
    pcs = {p % 12 for p in pitches}
    # Harmonic: bass-root fidelity and chord-tone completeness.
    bass_cost = 0.0 if pitches[0] % 12 == parsed['bass_pc'] else 3.0
    req = _required(parsed)
    non_fifth = set(parsed['pcs']) - ({(parsed['root'] + 7) % 12} if len(parsed['pcs']) >= 4 and 7 in parsed['intervals'] else set())
    completeness_cost = len(req - pcs) * 2.5 + len(non_fifth - pcs) * 1.5
    # Texture: internal mutes (hard to damp cleanly), muted treble strings (muddy arpeggios), and thin voicings.
    interior = _internal_mutes(frets)
    mute_cost = interior * 2.0 if not (interior == 1 and parsed['bass']) else interior * 0.8
    if frets[4] < 0 and frets[5] < 0:
        top_mute_cost = 3.5
    elif frets[5] < 0 and sum(1 for f in frets[2:6] if f >= 0) < 3:
        top_mute_cost = 1.2
    else:
        top_mute_cost = 0.0
    richness_cost = max(0, 4 - len(sounding)) * 1.5
    # Fretboard geometry: millimetre span, pairwise finger-reach proxy, and neck-position penalty.
    _, span_energy, pair_strain, position_energy, overflow = _fretboard_stretch_mm(frets, capo)
    positive = [f for f in frets if f > 0]
    barre_cost = 0.0
    if positive:
        barre_strings = sum(1 for f in frets if f == min(positive))
        barre_cost = (barre_strings - 1) * 0.8 if barre_strings > 1 else 0.0
    fingers_cnt, is_barre = _fingers(frets)
    fingers_cost = max(0, fingers_cnt - 3) * 1.5 + (0.35 if is_barre else 0.0) + overflow * 4.0
    cross_cost = sum(0.35 for a, b in zip(pitches, pitches[1:]) if b < a)
    return (bass_cost + completeness_cost + mute_cost + top_mute_cost + richness_cost
            + span_energy + pair_strain + position_energy + barre_cost + fingers_cost + cross_cost)


@lru_cache(maxsize=512)
def _search(symbol, capo):
    parsed = parse_extended_chord(symbol)
    allowed = set(parsed['pcs']) | {parsed['bass_pc']}
    required = _required(parsed) | {parsed['bass_pc']}
    found = {}
    canonical = set()

    def accept(frets, is_canon=False):
        try:
            voicing = _make_voicing(symbol, frets, capo, symbol)
        except ValueError:
            return
        t = tuple(frets)
        found[t] = voicing
        if is_canon:
            canonical.add(t)

    for name, shape in _SHAPES.items():
        template = parse_extended_chord(name)
        if template['quality'] == parsed['quality'] and (template['bass'] is not None) == (parsed['bass'] is not None):
            shift = (parsed['root']-template['root']-capo) % 12
            shifted = [f+shift if f >= 0 else -1 for f in shape]
            accept(shifted, is_canon=(shift == 0))
    for window in range(1, 25-capo):
        options = []
        for tuning in STANDARD_TUNING:
            options.append([-1] + [f for f in [0, *range(window, min(window+4,24-capo)+1)]
                                     if (tuning+capo+f)%12 in allowed])
        suffix = [set() for _ in range(7)]
        for i in range(5,-1,-1):
            suffix[i] = suffix[i+1] | {(STANDARD_TUNING[i]+capo+f)%12 for f in options[i] if f >= 0}

        def visit(i, frets, pcs, active, last):
            if len(active)+6-i < 3 or not required-pcs <= suffix[i]:
                return
            if i == 6:
                accept(frets)
                return
            for fret in options[i]:
                pitch = STANDARD_TUNING[i]+capo+fret if fret >= 0 else -1
                if pitch >= 0 and ((active and pitch < active[0]) or (not active and pitch%12 != parsed['bass_pc'])):
                    continue
                visit(i+1, frets+[fret], pcs | ({pitch%12} if pitch >= 0 else set()),
                      active+([pitch] if pitch >= 0 else []), pitch if pitch >= 0 else last)
        visit(0, [], set(), [], -1)

    def key(item):
        frets, _ = item
        canonical_bonus = 0.0 if frets in canonical else 2.0
        return canonical_bonus + _ergonomic_voicing_cost(list(frets), parsed, capo)

    return tuple(frets for frets, _ in sorted(found.items(), key=key))


def parse_extended_chord(symbol: str) -> dict | None:
    """Parse any chord symbol (including Maj7, m7, 7, sus2, sus4, add9, m7b5, dim7, 6, m6, Maj9, m9, and Slash Chords like C/E, G/B, Am7/G, D/F#, F/G)."""
    if not isinstance(symbol, str):
        return None
    symbol = symbol.strip().replace('♯','#').replace('♭','b').replace('𝄪','##').replace('𝄫','bb')
    match = re.fullmatch(r'([A-Ga-g](?:#+|b+)?)(.*?)(?:/([A-Ga-g](?:#+|b+)?))?', symbol)
    if not match:
        return None
    root, quality, bass = match.groups()
    root = root[0].upper()+root[1:]
    bass = bass[0].upper()+bass[1:] if bass else None
    q_norm = quality.strip().replace('(', '').replace(')', '')
    for src_p, dst_p in (
        ('minMaj','mmaj'),('minmaj','mmaj'),('mMaj','mmaj'),('mM','mmaj'),('mmaj','mmaj'),
        ('Major','maj'),('major','maj'),('MAJ','maj'),('Maj','maj'),
        ('min','m'),
    ):
        if q_norm.startswith(src_p):
            q_norm = dst_p + q_norm[len(src_p):]
            break
    if q_norm.startswith('M') and len(q_norm) >= 2 and q_norm[1].isdigit():
        q_norm = 'maj' + q_norm[1:]
    quality = {
        'maj':'','Maj':'','M':'','Major':'','major':'',
        'Maj7':'maj7','M7':'maj7','MAJ7':'maj7','Major7':'maj7','major7':'maj7',
        'Maj9':'maj9','M9':'maj9','MAJ9':'maj9','Major9':'maj9','major9':'maj9',
        'Maj11':'maj11','M11':'maj11','MAJ11':'maj11','Major11':'maj11','major11':'maj11',
        'Maj13':'maj13','M13':'maj13','MAJ13':'maj13','Major13':'maj13','major13':'maj13',
        'min':'m','min7':'m7','min9':'m9','min11':'m11','min13':'m13',
        'mMaj7':'mmaj7','mM7':'mmaj7','minMaj7':'mmaj7','mMaj9':'mmaj9','mM9':'mmaj9','minMaj9':'mmaj9',
        'mMaj11':'mmaj11','mM11':'mmaj11','minMaj11':'mmaj11','mMaj13':'mmaj13','mM13':'mmaj13','minMaj13':'mmaj13',
        'sus':'sus4','aug7':'7#5','+':'aug','+7':'7#5','o':'dim','o7':'dim7','ø':'m7b5','ø7':'m7b5',
    }.get(q_norm, q_norm)
    if quality in EXTENDED_INTERVALS:
        intervals = EXTENDED_INTERVALS[quality]
    else:
        base_Prefixes = (
            ('mmaj13',(0,3,7,11,14,21)),('mmaj11',(0,3,7,11,14,17)),('mmaj9',(0,3,7,11,14)),('mmaj7',(0,3,7,11)),
            ('mMaj13',(0,3,7,11,14,21)),('mMaj11',(0,3,7,11,14,17)),('mMaj9',(0,3,7,11,14)),('mMaj7',(0,3,7,11)),
            ('maj13',(0,4,7,11,14,21)),('Maj13',(0,4,7,11,14,21)),('M13',(0,4,7,11,14,21)),
            ('maj11',(0,4,7,11,14,17)),('Maj11',(0,4,7,11,14,17)),('M11',(0,4,7,11,14,17)),
            ('maj9',(0,4,7,11,14)),('Maj9',(0,4,7,11,14)),('M9',(0,4,7,11,14)),
            ('maj7',(0,4,7,11)),('Maj7',(0,4,7,11)),('M7',(0,4,7,11)),
            ('m13',(0,3,7,10,14,21)),('min13',(0,3,7,10,14,21)),
            ('m11',(0,3,7,10,14,17)),('min11',(0,3,7,10,14,17)),
            ('m9',(0,3,7,10,14)),('min9',(0,3,7,10,14)),
            ('m7b5',(0,3,6,10)),('m7',(0,3,7,10)),('min7',(0,3,7,10)),
            ('m6/9',(0,3,7,9,14)),('m6',(0,3,7,9)),('min6',(0,3,7,9)),
            ('dim7',(0,3,6,9)),('dim',(0,3,6)),('aug7',(0,4,8,10)),('aug',(0,4,8)),
            ('13',(0,4,7,10,14,21)),('11',(0,4,7,10,14,17)),('9',(0,4,7,10,14)),('7',(0,4,7,10)),
            ('6/9',(0,4,7,9,14)),('6',(0,4,7,9)),('5',(0,7)),
            ('major',(0,4,7)),('Major',(0,4,7)),('maj',(0,4,7)),('Maj',(0,4,7)),('min',(0,3,7)),('m',(0,3,7)),('M',(0,4,7)),('',(0,4,7)),
        )
        base_ivs, rem, matched_pref = (0, 4, 7), quality, ''
        for pref, pivs in base_Prefixes:
            if quality.startswith(pref):
                base_ivs, rem, matched_pref = pivs, quality[len(pref):], pref
                break
        if not matched_pref and not rem:
            return None
        ivs_set = set(base_ivs)
        pos = 0
        mod_re = re.compile(r'(sus2|sus4|sus|alt|(?:no|omit)(?:2|3|4|5|6|7|9|11|13)|(?:add)?([#b+-]?)(2|4|5|6|7|9|11|13))')
        while pos < len(rem):
            m_mod = mod_re.match(rem, pos)
            if not m_mod:
                return None
            tok, acc, deg = m_mod.group(1), m_mod.group(2), m_mod.group(3)
            if tok == 'sus2':
                ivs_set.discard(3); ivs_set.discard(4); ivs_set.add(2)
            elif tok in ('sus4', 'sus'):
                ivs_set.discard(3); ivs_set.discard(4); ivs_set.add(5)
            elif tok == 'alt':
                ivs_set.discard(7); ivs_set.discard(14); ivs_set.discard(21); ivs_set.update((10, 13, 15, 20))
            elif tok.startswith(('no', 'omit')):
                om = tok[2:] if tok.startswith('no') else tok[4:]
                if om == '3': ivs_set.discard(3); ivs_set.discard(4)
                elif om == '5': ivs_set.discard(6); ivs_set.discard(7); ivs_set.discard(8)
                elif om in ('2', '9'): ivs_set.discard(1); ivs_set.discard(2); ivs_set.discard(13); ivs_set.discard(14); ivs_set.discard(15)
                elif om in ('4', '11'): ivs_set.discard(5); ivs_set.discard(16); ivs_set.discard(17); ivs_set.discard(18)
                elif om in ('6', '13'): ivs_set.discard(8); ivs_set.discard(9); ivs_set.discard(20); ivs_set.discard(21); ivs_set.discard(22)
                elif om == '7': ivs_set.discard(10); ivs_set.discard(11)
            else:
                if deg == '5' and not acc and not tok.startswith('add'):
                    return None
                is_add = tok.startswith('add')
                off = 1 if acc in ('#', '+') else (-1 if acc in ('b', '-') else 0)
                if deg == '5':
                    if not is_add: ivs_set.discard(6); ivs_set.discard(7); ivs_set.discard(8)
                    ivs_set.add(7 + off)
                elif deg in ('2', '9'):
                    if off != 0 and not is_add: ivs_set.discard(2); ivs_set.discard(14)
                    ivs_set.add((2 if deg == '2' else 14) + off)
                elif deg in ('4', '11'):
                    if off != 0 and not is_add: ivs_set.discard(5); ivs_set.discard(17)
                    ivs_set.add((5 if deg == '4' else 17) + off)
                elif deg in ('6', '13'):
                    if off != 0 and not is_add: ivs_set.discard(9); ivs_set.discard(21)
                    ivs_set.add((9 if deg == '6' else 21) + off)
                elif deg == '7':
                    if not is_add: ivs_set.discard(10); ivs_set.discard(11)
                    ivs_set.add(10 if acc in ('b', '-', '') else 11)
            pos = m_mod.end()
        intervals = tuple(sorted(ivs_set))
        norm_pcs_t = tuple(sorted({x % 12 for x in intervals}))
        for k_ext, v_ext in EXTENDED_INTERVALS.items():
            if k_ext in ('sus', 'aug7', 'add2', 'add4'):
                continue
            if v_ext == intervals or tuple(sorted({x % 12 for x in v_ext})) == norm_pcs_t:
                quality = k_ext
                intervals = v_ext
                break
        else:
            if quality.startswith(('b', '#')):
                quality = f'({quality})'
    pc = _note_pc(root)
    return dict(root=pc, root_name=root, quality=quality, bass=bass,
                bass_pc=_note_pc(bass) if bass else pc,
                pcs=sorted({(pc+x)%12 for x in intervals}), intervals=intervals)


def generate_voicings(symbol: str, capo: int = 0, max_candidates: int = 6) -> list[dict]:
    """Return playable 6-string fretboard voicings ranked by canonical-shape bonus and voicing preference cost."""
    _integer(capo,0,12,'capo')
    _integer(max_candidates,0,float('inf'),'max_candidates')
    if max_candidates == 0:
        return []
    if isinstance(symbol,str) and symbol.strip() == 'N.C.':
        return [_make_voicing(symbol,[-1]*6,capo,symbol)]
    parsed = parse_extended_chord(symbol)
    if parsed is None:
        return []
    normalized = parsed['root_name']+parsed['quality']+('/'+parsed['bass'] if parsed['bass'] else '')
    return [_make_voicing(symbol,f,capo,symbol) for f in _search(normalized,capo)[:max_candidates]]


def _transition_cost(a, b):
    """Transition cost between two 6-string voicings `a` and `b`.

    Fret values use `-1` for muted strings, `0` for open strings, and `> 0` for fretted notes;
    `pitches` uses `-1` for muted strings and MIDI note numbers (`>= 0`) for sounding strings.
    Combines fretted-hand centroid shift (`f > 0`), common-finger anchor bonus (`fa == fb > 0`),
    per-string fret movement on sounding strings (`fa >= 0 and fb >= 0`), and soprano pitch leap.
    """
    frets_a, frets_b = a['frets'], b['frets']
    positive_a = [f for f in frets_a if f > 0]
    positive_b = [f for f in frets_b if f > 0]
    centroid_a = sum(positive_a) / len(positive_a) if positive_a else 0.0
    centroid_b = sum(positive_b) / len(positive_b) if positive_b else 0.0
    centroid_cost = abs(centroid_a - centroid_b) * 1.15
    anchor_bonus = 0.0
    jump_cost = 0.0
    for fa, fb in zip(frets_a, frets_b):
        if fa > 0 and fa == fb:
            anchor_bonus -= 0.35
        elif fa >= 0 and fb >= 0:
            jump_cost += abs(fa - fb) * 0.12
    pitches_a = [p for p in a['pitches'] if p >= 0]
    pitches_b = [p for p in b['pitches'] if p >= 0]
    soprano_cost = 0.0
    if pitches_a and pitches_b:
        top_a, top_b = max(pitches_a), max(pitches_b)
        leap = abs(top_a - top_b)
        soprano_cost = 0.06 * leap + max(0, leap - 3) * 0.22
    return centroid_cost + anchor_bonus + jump_cost + soprano_cost


def _optimize_candidates(layers):
    if not layers:
        return []
    if any(not layer for layer in layers):
        raise ValueError('Empty candidate layer')
    # Intrinsic rank penalty: canonical #1 voicings win unless a lower-ranked shape saves a real fret jump.
    rank_penalty = 3.5
    costs, pointers = [idx*rank_penalty for idx in range(len(layers[0]))], []
    for previous, current in zip(layers,layers[1:]):
        choices = [min(range(len(previous)), key=lambda k: costs[k]+_transition_cost(previous[k],v)) for v in current]
        costs = [costs[k]+_transition_cost(previous[k],v)+j*rank_penalty for j,(k,v) in enumerate(zip(choices,current))]
        pointers.append(choices)
    index = min(range(len(costs)),key=costs.__getitem__)
    path = [layers[-1][index]]
    for i in range(len(pointers)-1,-1,-1):
        index = pointers[i][index]
        path.append(layers[i][index])
    return path[::-1]


def optimize_voice_leading(chord_sequence: list[str], capo: int = 0) -> list[dict]:
    """Select the minimum-cost fretboard voicing per chord via Viterbi dynamic programming over hand centroid shift, common-finger anchor retention, per-string movement, and soprano voice-leading continuity."""
    layers = []
    for index, chord in enumerate(chord_sequence):
        layer = generate_voicings(chord,capo)
        if not layer:
            raise ValueError(f'No voicings for chord {chord!r} at index {index}')
        layers.append(layer)
    return _optimize_candidates(layers)
