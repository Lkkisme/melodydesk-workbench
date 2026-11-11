"""Phrase-level rhythm inference; never changes pitch or source timing.

Small dynamic program balances onset fit, interval fit and simple durations.
The user supplies tempo/beat origin. This is not key or harmony inference.
"""
import copy
import math
import numpy as np
from subpkgs.stem_mir.beat_tracker import seconds_to_beats_piecewise


def _raw_to_beats(notes, bpm, origin, beat_times=None):
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=np.float64).ravel()
        if len(bt) >= 2:
            if not np.all(np.isfinite(bt)) or not np.all(np.diff(bt) > 0):
                raise ValueError('拍点时间列表必须包含严格递增的有限数值')
            origin_beat = float(seconds_to_beats_piecewise(origin, bt, bpm, origin))
            starts = [float(seconds_to_beats_piecewise(n.raw_start, bt, bpm, origin)) - origin_beat for n in notes]
            ends = [float(seconds_to_beats_piecewise(n.raw_start + n.raw_duration, bt, bpm, origin)) - origin_beat for n in notes]
            lengths = [max(0.0, e - s) for s, e in zip(starts, ends)]
            return starts, lengths
    starts = [(n.raw_start - origin) * bpm / 60 for n in notes]
    lengths = [n.raw_duration * bpm / 60 for n in notes]
    return starts, lengths


def _phrase(notes, bpm, origin, division, beat_times=None, min_beat=0.0, max_beat=None):
    unit = 1 / division
    starts, lengths = _raw_to_beats(notes, bpm, origin, beat_times)
    floor_b = max(0.0, float(min_beat or 0.0))
    ceil_b = None if max_beat is None else float(max_beat)
    k_min = max(0, math.ceil((floor_b - 1e-8) / unit))
    if ceil_b is not None:
        k_max_last = math.floor((ceil_b - unit + 1e-8) / unit)
        if k_max_last - k_min + 1 < len(notes):
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
    candidates = []
    for idx, t in enumerate(starts):
        rem_after = len(notes) - 1 - idx
        k_lo = k_min + idx
        k_hi = None if ceil_b is None else math.floor((ceil_b - (rem_after + 1) * unit + 1e-8) / unit)
        if k_hi is not None and k_hi < k_lo:
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
        center = round(t / unit)
        ks = [k for k in range(center - 2, center + 3) if k >= k_lo and (k_hi is None or k <= k_hi)]
        if not ks:
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
        candidates.append(sorted(round(k * unit, 10) for k in ks))
    states = {s: ((s - starts[0]) ** 2, [s]) for s in candidates[0]}
    common = [unit, 2 * unit, 3 * unit, 1, 1.5, 2, 3, 4, 6, 8]
    for i in range(1, len(notes)):
        next_states = {}
        for s in candidates[i]:
            options = []
            for previous, (cost, path) in states.items():
                interval = s - previous
                if interval < unit - 1e-8:
                    continue
                observed = starts[i] - starts[i - 1]
                complexity = min(abs(interval - d) for d in common)
                options.append((cost + (s - starts[i]) ** 2 + .5 * (interval - observed) ** 2 + .05 * complexity, path + [s]))
            if options:
                next_states[s] = min(options, key=lambda x: x[0])
        if not next_states:
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
        states = next_states
    cost, path = min(states.values(), key=lambda x: x[0])
    result = copy.deepcopy(notes)
    for i, n in enumerate(result):
        n.start = path[i]
        limit = path[i + 1] - path[i] if i + 1 < len(path) else max(8, math.ceil(lengths[i]) + 1)
        if i + 1 == len(path) and ceil_b is not None:
            limit = min(limit, ceil_b - path[i])
        durations = sorted({round(k * unit, 10) for k in range(1, max(2, math.ceil(limit / unit) + 1)) if k * unit <= limit + 1e-8})
        if not durations:
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
        gap = (starts[i + 1] - starts[i] - lengths[i]) if i + 1 < len(path) else 1
        # Small articulation gaps belong inside a notated note; real rests do not.
        target = limit if i + 1 < len(path) and gap < .22 else lengths[i]
        n.duration = min(durations, key=lambda d: (d - target) ** 2 + .025 * (0 if any(abs(d - c) < 1e-6 for c in common) else 1))
        cost += (n.duration - lengths[i]) ** 2 * .15
    return result, cost / len(notes)


def organize(notes, bpm, origin=0, mode='auto', beat_times=None, min_beat=0.0, max_beat=None):
    if not notes:
        return []
    if mode not in ('auto', 'binary', 'triplet'):
        raise ValueError('未知节奏类型')
    ordered = sorted(copy.deepcopy(notes), key=lambda n: n.raw_start)
    phrases = []
    for n in ordered:
        if not phrases or n.raw_start - (phrases[-1][-1].raw_start + phrases[-1][-1].raw_duration) > .45:
            phrases.append([])
        phrases[-1].append(n)
    result = []
    cur_min = max(0.0, float(min_beat or 0.0))
    ceil_b = None if max_beat is None else float(max_beat)
    divisions = [2, 3, 4, 6] if mode == 'auto' else [2, 4] if mode == 'binary' else [3, 6]
    finest_unit = 1.0 / max(divisions)
    for p_idx, phrase in enumerate(phrases):
        rem_notes = sum(len(p) for p in phrases[p_idx + 1:])
        p_max = None if ceil_b is None else (ceil_b - rem_notes * finest_unit)
        options = []
        last_exc = None
        for division in divisions:
            try:
                arranged, cost = _phrase(
                    phrase,
                    bpm,
                    origin,
                    division,
                    beat_times=beat_times,
                    min_beat=cur_min,
                    max_beat=p_max,
                )
                # Prefer simple binary rhythm unless finer/triplet fit is better.
                options.append((cost + {2: 0, 4: .006, 3: .009, 6: .015}[division], arranged))
            except ValueError as exc:
                # Candidate grid division is too coarse for this phrase density; try finer divisions.
                last_exc = exc
        if not options:
            raise ValueError('此乐句音符太密，请先合并误切音符再整理。') from last_exc
        chosen = min(options, key=lambda p: p[0])[1]
        result.extend(chosen)
        cur_min = max(cur_min, chosen[-1].start + chosen[-1].duration)
    for a, b in zip(result, result[1:]):
        if a.start + a.duration > b.start + 1e-6:
            raise ValueError('整理后乐句边界冲突，请调整拍点或缩小选择范围。')
    if ceil_b is not None and result and result[-1].start + result[-1].duration > ceil_b + 1e-6:
        raise ValueError('整理后乐句超出可用拍区间，请调整拍点或缩小选择范围。')
    return result
