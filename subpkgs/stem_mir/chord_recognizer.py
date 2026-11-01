"""Pure Vocal-Melody-Conditioned Functional Harmony & Vocal Dynamics Generator.

100% driven by the singer's vocal melody notes, phrase cadences, and vocal register/density dynamics.
Zero accompaniment audio or background CQT is ever read or used.
"""
from __future__ import annotations
import math
import numpy as np

MAJOR_TRANS_MATRIX = np.array([
    [-0.04, 0.08, 0.04, 0.14, 0.14, 0.10, 0.04],
    [ 0.04,-0.06, 0.02, 0.06, 0.16, 0.04, 0.02],
    [ 0.02, 0.06,-0.06, 0.10, 0.04, 0.12, 0.02],
    [ 0.12, 0.06, 0.02,-0.04, 0.14, 0.06, 0.02],
    [ 0.18, 0.02, 0.02, 0.06,-0.06, 0.12, 0.02],
    [ 0.04, 0.10, 0.06, 0.12, 0.08,-0.06, 0.02],
    [ 0.10, 0.02, 0.02, 0.08, 0.06, 0.04,-0.04],
])

MINOR_TRANS_MATRIX = np.array([
    [-0.04, 0.08, 0.12, 0.06, 0.06, 0.12, 0.04],
    [ 0.06,-0.06, 0.08, 0.04, 0.06, 0.08, 0.02],
    [ 0.08, 0.06,-0.04, 0.10, 0.06, 0.08, 0.10],
    [ 0.10, 0.06, 0.06,-0.06, 0.08, 0.06, 0.04],
    [ 0.06, 0.08, 0.08, 0.06,-0.04, 0.10, 0.06],
    [ 0.12, 0.08, 0.06, 0.04, 0.06,-0.04, 0.04],
    [ 0.16, 0.04, 0.04, 0.02, 0.08, 0.04,-0.06],
])


def _timing(bpm: float, beat_origin: float, meter: str) -> tuple[float, float, float]:
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be finite and positive")
    if not np.isfinite(beat_origin) or beat_origin < 0:
        raise ValueError("Beat origin must be finite and nonnegative")
    if meter not in ("4/4", "3/4"):
        raise ValueError("Unsupported meter")
    measure_beats = 4.0 if meter == "4/4" else 3.0
    seconds_per_beat = 60.0 / bpm
    return measure_beats, seconds_per_beat, measure_beats * seconds_per_beat


def _gttm_note_salience(rel_onset_beats: float, overlap_beats: float, measure_beats: float) -> float:
    sigma = 0.14
    grid = [(0.0, 1.45), (measure_beats * 0.5, 1.22)]
    for q in range(int(measure_beats)):
        grid.append((float(q), 1.05))
    for q in range(int(measure_beats * 2)):
        grid.append((q * 0.5, 0.85))
    best = 0.85
    for pos, weight in grid:
        dist = abs(rel_onset_beats - pos)
        if dist > measure_beats / 2:
            dist = measure_beats - dist
        gauss = math.exp(-0.5 * (dist / sigma) ** 2)
        val = weight * gauss
        if val > best:
            best = val
    dur_factor = min(1.45, max(0.35, 0.40 + 0.65 * math.log(1 + 2.2 * overlap_beats)))
    return best * dur_factor


_W_TIS = np.array([2.0, 11.0, 17.0, 16.0, 19.0, 7.0], dtype=np.float64)


def _tiv(chroma: np.ndarray) -> np.ndarray:
    total = float(np.sum(chroma))
    if total <= 1e-12:
        return np.zeros(12, dtype=np.float64)
    dft = np.fft.rfft(chroma / total)[1:7] * _W_TIS
    return np.concatenate([dft.real, dft.imag])


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _degree_index(root: int, quality: str, tonic_root: int, is_minor_key: bool) -> int:
    offset = (root - tonic_root) % 12
    if is_minor_key:
        if offset == 7 and quality != "m":
            return 6
        return {0: 0, 3: 1, 5: 2, 7: 3, 8: 4, 10: 5}.get(offset, -1)
    return {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 10: 6}.get(offset, -1)


def synthesize_melody_conditioned_harmony(
    notes: list,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    tonic: str = "C",
    minor: bool = False,
    clip_duration: float = 0.0,
    bar_start_seconds: list[float] | None = None,
) -> dict:
    """Pure Vocal-Melody-Conditioned Functional Harmony & Dynamics Engine.

    Scores diatonic, modal, and secondary-dominant guitar chords per bar 100% from the
    singer's vocal melody notes (`notes`), resolves harmonic progressions via functional
    Hooktheory-grounded Viterbi decoding with GTTM metrical salience and TIV emission,
    and derives per-bar guitar dynamics (`bar_dynamics`) from the singer's vocal note
    density and pitch register.
    """
    NAMES = "C C# D D# E F F# G G# A A# B".split()
    if tonic not in NAMES:
        raise ValueError("Malformed tonic")
    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    R = NAMES.index(tonic)
    tonic_root = (R + 9) % 12 if minor else R
    trans_matrix = MINOR_TRANS_MATRIX if minor else MAJOR_TRANS_MATRIX

    def _chord_name(root: int, is_minor: bool) -> str:
        return NAMES[root % 12] + ("m" if is_minor else "")

    candidates = []
    seen_names = set()

    def _add(root: int, quality: str, tones: tuple[int, ...], prior_bias: float) -> None:
        r = root % 12
        is_minor = quality == "m"
        name = _chord_name(r, is_minor)
        key = (name, tones)
        if key not in seen_names:
            seen_names.add(key)
            deg = _degree_index(r, quality, tonic_root, minor)
            candidates.append((r, quality, tuple(sorted(set(((r + t) % 12) for t in tones))), prior_bias, name, deg))

    if minor:
        R_min = (R + 9) % 12
        _add(R_min, "m", (0, 3, 7), 0.08)
        _add((R_min + 3) % 12, "", (0, 4, 7), 0.06)
        _add((R_min + 5) % 12, "m", (0, 3, 7), 0.06)
        _add((R_min + 5) % 12, "", (0, 4, 7), 0.05)
        _add((R_min + 7) % 12, "m", (0, 3, 7), 0.05)
        _add((R_min + 7) % 12, "", (0, 4, 7), 0.02)
        _add((R_min + 8) % 12, "", (0, 4, 7), 0.06)
        _add((R_min + 10) % 12, "", (0, 4, 7), 0.07)
    else:
        _add(R, "", (0, 4, 7), 0.07)
        _add((R + 2) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 4) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 5) % 12, "", (0, 4, 7), 0.06)
        _add((R + 7) % 12, "", (0, 4, 7), 0.06)
        _add((R + 9) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 4) % 12, "", (0, 4, 7), -0.02)
        _add((R + 2) % 12, "", (0, 4, 7), -0.03)
        _add((R + 10) % 12, "", (0, 4, 7), -0.03)

    chord_names = [c[4] for c in candidates]
    candidate_roots = [c[0] for c in candidates]
    candidate_tones = [c[2] for c in candidates]
    candidate_biases = [c[3] for c in candidates]
    candidate_qualities = [c[1] for c in candidates]
    candidate_degrees = [c[5] for c in candidates]

    max_note_end = max((n.start * seconds_per_beat + beat_origin + n.duration * seconds_per_beat
                        for n in notes), default=0.0) if notes else 0.0
    total_duration = max(clip_duration, max_note_end, beat_origin + bar_seconds)
    n_bars = max(1, int(math.ceil((total_duration - beat_origin) / bar_seconds)))

    if bar_start_seconds is not None and len(bar_start_seconds) >= 2:
        n_bars = len(bar_start_seconds) - 1
        computed_bar_starts = [float(t) for t in bar_start_seconds]
    else:
        computed_bar_starts = [beat_origin + b * bar_seconds for b in range(n_bars + 1)]

    tonic_qual = "m" if minor else ""
    dom_root = (tonic_root + 7) % 12

    chord_tivs = {}
    for ci in range(len(candidates)):
        name = chord_names[ci]
        if name not in chord_tivs:
            chroma = np.zeros(12)
            for pc in candidate_tones[ci]:
                chroma[pc] = 1.0
            chord_tivs[name] = _tiv(chroma)

    bar_vocal_w = []
    bar_has_vocal = []
    first_vocal_bar = -1
    for b in range(n_bars):
        bs_beat = b * measure_beats
        be_beat = (b + 1) * measure_beats
        vocal_w = np.zeros(12)
        for n in notes:
            overlap = max(0.0, min(n.start + n.duration, be_beat) - max(n.start, bs_beat))
            if overlap <= 0:
                continue
            rel_onset = (n.start - bs_beat) % measure_beats
            salience = _gttm_note_salience(rel_onset, overlap, measure_beats)
            vocal_w[n.pitch % 12] += salience
        has_vocal = float(np.sum(vocal_w)) > 0
        bar_vocal_w.append(vocal_w)
        bar_has_vocal.append(has_vocal)
        if has_vocal and first_vocal_bar < 0:
            first_vocal_bar = b

    emission_scores = []
    for b in range(n_bars):
        vocal_w = bar_vocal_w[b]
        has_vocal = bar_has_vocal[b]
        rel_b = (b - first_vocal_bar) if (first_vocal_bar >= 0 and b >= first_vocal_bar) else b

        is_cadence_bar = (rel_b % 4 == 3) or (has_vocal and b + 1 < n_bars and not bar_has_vocal[b + 1])

        bar_tiv = _tiv(vocal_w) if has_vocal else None

        bar_scores = {}
        for ci in range(len(candidates)):
            root = candidate_roots[ci]
            tones = set(candidate_tones[ci])
            quality = candidate_qualities[ci]
            name = chord_names[ci]
            bias = candidate_biases[ci]
            is_minor_q = quality == "m"
            third = (root + 3) % 12 if is_minor_q else (root + 4) % 12
            avoid_pcs = {(third + 1) % 12, (third - 1) % 12} if not is_minor_q else set()

            if has_vocal:
                s = 0.0
                for pc in range(12):
                    if pc in tones:
                        s += vocal_w[pc] * 1.0
                    elif pc in avoid_pcs:
                        s += vocal_w[pc] * -0.85
                    else:
                        s += vocal_w[pc] * -0.25
                total_w = float(np.sum(vocal_w))
                vocal_score = s / total_w if total_w > 0 else 0.0
                tiv_sim = _cosine_sim(bar_tiv, chord_tivs[name])
                combined = 0.75 * vocal_score + 0.17 * tiv_sim + bias
            else:
                t_root = tonic_root
                turnaround_roots = (
                    (t_root, (t_root + 10) % 12, (t_root + 3) % 12, (t_root + 7) % 12)
                    if minor
                    else (t_root, (t_root + 7) % 12, (t_root + 9) % 12, (t_root + 5) % 12)
                )
                turnaround_prior = 0.14 if root == turnaround_roots[rel_b % 4] else 0.0
                tonic_prior = 0.22 if (b == 0 and root == t_root) else (0.05 if root == t_root else 0.0)
                cadence_prior = 0.08 if (b == n_bars - 1 or rel_b % 4 == 0) and root == t_root else 0.0
                combined = tonic_prior + cadence_prior + turnaround_prior + bias

            is_tonic_chord = (root == tonic_root and quality == tonic_qual)

            if is_tonic_chord and (b == 0 or b == first_vocal_bar):
                tonic_triad = set(candidate_tones[ci])
                vocal_on_triad = sum(vocal_w[pc] for pc in range(12) if pc in tonic_triad)
                avoid_clash = not any(vocal_w[pc] > 0 for pc in avoid_pcs)
                triad_frac = vocal_on_triad / float(np.sum(vocal_w)) if has_vocal and float(np.sum(vocal_w)) > 0 else 1.0
                if not has_vocal or (avoid_clash and triad_frac >= 0.35):
                    combined += 0.28
            if is_tonic_chord and b == n_bars - 1:
                combined += 0.22

            if is_cadence_bar:
                if root == dom_root and quality != "m":
                    combined += 0.16
                if is_tonic_chord and b == n_bars - 1:
                    combined += 0.06

            if minor and root == (tonic_root + 7) % 12 and not is_minor_q:
                leading_pc = (tonic_root + 11) % 12
                if not (has_vocal and vocal_w[leading_pc] > 0.05):
                    combined -= 0.22

            if name in bar_scores:
                bar_scores[name] = max(bar_scores[name], combined)
            else:
                bar_scores[name] = combined
        emission_scores.append(bar_scores)

    tonic_name = _chord_name(tonic_root, minor)

    name_to_degree = {}
    name_to_root = {}
    for ci in range(len(candidates)):
        n = chord_names[ci]
        if n not in name_to_degree or candidate_degrees[ci] >= 0:
            name_to_degree[n] = candidate_degrees[ci]
        if n not in name_to_root:
            name_to_root[n] = candidate_roots[ci]

    unique_names = list(dict.fromkeys(chord_names))

    states = {None: (0.0, [])}
    for b in range(n_bars):
        updated = {}
        rel_b_v = (b - first_vocal_bar) if (first_vocal_bar >= 0 and b >= first_vocal_bar) else b
        for name in unique_names:
            if name not in emission_scores[b]:
                continue
            fit = emission_scores[b][name]
            best = None
            curr_deg = name_to_degree.get(name, -1)
            for prev, (cost, path) in states.items():
                transition = 0.0
                if prev is not None and prev != "N.C." and name != "N.C.":
                    prev_deg = name_to_degree.get(prev, -1)
                    if prev_deg >= 0 and curr_deg >= 0:
                        transition = float(trans_matrix[prev_deg, curr_deg])
                    else:
                        prev_root = name_to_root.get(prev, 0)
                        curr_root = name_to_root.get(name, 0)
                        dist = (curr_root - prev_root) % 12
                        if dist in (5, 7):
                            transition = 0.10
                        elif dist in (2, 10):
                            transition = 0.06

                    if prev == name:
                        transition -= 0.03
                        if len(path) >= 2 and path[-2] == name:
                            transition -= 0.14

                    if (rel_b_v % 4 == 0 or b == n_bars - 1) and name == tonic_name:
                        transition += 0.12

                total = cost + fit + transition
                if best is None or total > best[0]:
                    best = (total, path + [name])
            if best is not None:
                if name not in updated or best[0] > updated[name][0]:
                    updated[name] = best
        states = updated

    chords = max(states.values(), key=lambda x: x[0])[1] if states else ["N.C."] * n_bars

    if notes:
        vocal_energy = np.zeros(n_bars, dtype=np.float64)
        for b in range(n_bars):
            bs_beat, be_beat = b * measure_beats, (b + 1) * measure_beats
            bar_ns = [n for n in notes if min(n.start + n.duration, be_beat) > max(n.start, bs_beat)]
            if bar_ns:
                cov = sum(min(n.start + n.duration, be_beat) - max(n.start, bs_beat) for n in bar_ns) / measure_beats
                reg = np.clip((np.mean([n.pitch for n in bar_ns]) - 54.0) / 22.0, 0.0, 1.0)
                vocal_energy[b] = float(np.clip(0.55 * min(1.0, cov) + 0.45 * reg, 0.15, 1.0))
            else:
                vocal_energy[b] = 0.25
        smoothed = np.convolve(vocal_energy, np.ones(3) / 3, mode="same")
        bar_dynamics = [round(float(np.clip(0.66 + 0.44 * v, 0.66, 1.10)), 4) for v in smoothed]
    else:
        bar_dynamics = [0.85] * n_bars

    return {
        "chords": chords,
        "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
        "bar_dynamics": bar_dynamics,
    }
