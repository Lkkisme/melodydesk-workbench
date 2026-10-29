"""Extended & Slash Chord Parser, CAGED/Open/Shell Voicing Generator, and Parameter-Free Voice Leading DP."""
from __future__ import annotations

import re
from functools import lru_cache

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
EXTENDED_INTERVALS = {
    "": (0,4,7), "m": (0,3,7), "7": (0,4,7,10),
    "maj7": (0,4,7,11), "m7": (0,3,7,10), "m7b5": (0,3,6,10),
    "dim": (0,3,6), "dim7": (0,3,6,9), "aug": (0,4,8),
    "sus2": (0,2,7), "sus4": (0,5,7), "7sus4": (0,5,7,10),
    "add9": (0,4,7,14), "madd9": (0,3,7,14), "6": (0,4,7,9),
    "m6": (0,3,7,9), "9": (0,4,7,10,14),
    "maj9": (0,4,7,11,14), "m9": (0,3,7,10,14),
}


def _note_pc(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Ga-g][#b]?', name):
        raise ValueError(f'Invalid note: {name!r}')
    return ({'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[name[0].upper()]
            + (1 if name.endswith('#') else -1 if name.endswith('b') else 0)) % 12


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'Invalid {name}: {value!r}')


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
    elif (len(active) < 3 or active != sorted(active)
          or min(active) % 12 != parsed['bass_pc']
          or not {p%12 for p in active} <= set(parsed['pcs']) | {parsed['bass_pc']}
          or not _required(parsed) <= {p%12 for p in active}
          or span > 4 or fingers > 4):
        raise ValueError(f'Unplayable or harmonically invalid voicing for {symbol}: {frets}')
    return dict(frets=list(frets), pitches=pitches, base_fret=1 if 0 in frets else min(positive) if positive else 1,
                barre=barre, span=span, label=label)


_SHAPES = {
    'C':[-1,3,2,0,1,0], 'Cm':[-1,3,5,5,4,3], 'Cmaj7':[-1,3,2,0,0,0],
    'D':[-1,-1,0,2,3,2], 'Dm':[-1,-1,0,2,3,1], 'D7':[-1,-1,0,2,1,2],
    'Dm7':[-1,-1,0,2,1,1], 'E':[0,2,2,1,0,0], 'Em':[0,2,2,0,0,0],
    'E7':[0,2,0,1,0,0], 'Em7':[0,2,0,0,0,0], 'F':[1,3,3,2,1,1],
    'Fmaj7':[1,3,3,2,1,0], 'Fm6':[1,-1,0,1,1,1], 'G':[3,2,0,0,0,3],
    'G7':[3,2,0,0,0,1], 'G7sus4':[3,-1,0,0,1,1], 'A':[-1,0,2,2,2,0],
    'Am':[-1,0,2,2,1,0], 'Am7':[-1,0,2,0,1,0], 'C/E':[0,3,2,0,1,0],
    'G/B':[-1,2,0,0,0,3], 'Am7/G':[3,0,2,0,1,0], 'D/F#':[2,-1,0,2,3,2],
    'F/G':[3,-1,3,2,1,1],
}


@lru_cache(maxsize=512)
def _search(symbol, capo):
    parsed = parse_extended_chord(symbol)
    allowed = set(parsed['pcs']) | {parsed['bass_pc']}
    required = _required(parsed) | {parsed['bass_pc']}
    found = {}

    def accept(frets):
        try:
            voicing = _make_voicing(symbol, frets, capo, symbol)
        except ValueError:
            return
        found[tuple(frets)] = voicing

    for name, shape in _SHAPES.items():
        template = parse_extended_chord(name)
        if template['quality'] == parsed['quality']:
            shift = (parsed['root']-template['root']-capo) % 12
            accept([f+shift if f >= 0 else -1 for f in shape])
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
                if pitch >= 0 and (pitch < last or (not active and pitch%12 != parsed['bass_pc'])):
                    continue
                visit(i+1, frets+[fret], pcs | ({pitch%12} if pitch >= 0 else set()),
                      active+([pitch] if pitch >= 0 else []), pitch if pitch >= 0 else last)
        visit(0, [], set(), [], -1)
    def key(item):
        frets, v = item
        return (len(set(parsed['pcs'])-{p%12 for p in v['pitches'] if p >= 0}),
                v['span'], _fingers(frets)[0], max(0,*frets), sum(max(f,0) for f in frets), frets)
    return tuple(frets for frets, _ in sorted(found.items(), key=key))


def parse_extended_chord(symbol: str) -> dict | None:
    """Parse any chord symbol (including Maj7, m7, 7, sus2, sus4, add9, m7b5, dim7, 6, m6, Maj9, m9, and Slash Chords like C/E, G/B, Am7/G, D/F#, F/G)."""
    if not isinstance(symbol, str):
        return None
    symbol = symbol.strip().replace('♯','#').replace('♭','b')
    match = re.fullmatch(r'([A-Ga-g](?:#|b)?)([^/]*)(?:/([A-Ga-g](?:#|b)?))?', symbol)
    if not match:
        return None
    root, quality, bass = match.groups()
    root = root[0].upper()+root[1:]
    bass = bass[0].upper()+bass[1:] if bass else None
    quality = {'Maj7':'maj7','M7':'maj7','Maj9':'maj9','M9':'maj9'}.get(quality,quality)
    if quality not in EXTENDED_INTERVALS:
        return None
    intervals = EXTENDED_INTERVALS[quality]
    pc = _note_pc(root)
    return dict(root=pc, root_name=root, quality=quality, bass=bass,
                bass_pc=_note_pc(bass) if bass else pc,
                pcs=sorted({(pc+x)%12 for x in intervals}), intervals=intervals)


def generate_voicings(symbol: str, capo: int = 0, max_candidates: int = 6) -> list[dict]:
    """Return playable 6-string fretboard voicings sorted by physical ergonomics."""
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


def _transition_cost(a,b):
    return sum(abs(x-y) for x,y in zip(a['frets'],b['frets'])) - len(
        {p for p in a['pitches'] if p >= 0} & {p for p in b['pitches'] if p >= 0})


def _optimize_candidates(layers):
    if not layers:
        return []
    if any(not layer for layer in layers):
        raise ValueError('Empty candidate layer')
    costs, pointers = [0]*len(layers[0]), []
    for previous, current in zip(layers,layers[1:]):
        choices = [min(range(len(previous)), key=lambda k: costs[k]+_transition_cost(previous[k],v)) for v in current]
        costs = [costs[k]+_transition_cost(previous[k],v) for k,v in zip(choices,current)]
        pointers.append(choices)
    index = min(range(len(costs)),key=costs.__getitem__)
    path = [layers[-1][index]]
    for i in range(len(pointers)-1,-1,-1):
        index = pointers[i][index]
        path.append(layers[i][index])
    return path[::-1]


def optimize_voice_leading(chord_sequence: list[str], capo: int = 0) -> list[dict]:
    """Select optimal fretboard voicing per chord via dynamic programming using pure integer fretboard Manhattan distance minus shared pitch count."""
    layers = []
    for index, chord in enumerate(chord_sequence):
        layer = generate_voicings(chord,capo)
        if not layer:
            raise ValueError(f'No voicings for chord {chord!r} at index {index}')
        layers.append(layer)
    return _optimize_candidates(layers)
