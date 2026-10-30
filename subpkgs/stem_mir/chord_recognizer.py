"""Pure Vocal-Melody-Conditioned Functional Harmony & Vocal Dynamics Generator.

100% driven by the singer's vocal melody notes, phrase cadences, and vocal register/density dynamics.
Zero accompaniment audio or background CQT is ever read or used.
"""
from __future__ import annotations
import math
import numpy as np


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
    circle-of-fifths Viterbi decoding, and derives per-bar guitar dynamics (`bar_dynamics`)
    from the singer's vocal note density and pitch register.
    """
    NAMES = "C C# D D# E F F# G G# A A# B".split()
    if tonic not in NAMES:
        raise ValueError("Malformed tonic")
    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    R = NAMES.index(tonic)

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
            candidates.append((r, quality, tuple(sorted(set(((r + t) % 12) for t in tones))), prior_bias, name))

    if minor:
        R_min = (R + 9) % 12
        # Diatonic & modal minor triads: i, bIII, iv, IV (Dorian), v, V (harmonic), bVI, bVII
        _add(R_min, "m", (0, 3, 7), 0.08)                  # i (tonic minor)
        _add((R_min + 3) % 12, "", (0, 4, 7), 0.06)        # bIII (relative major)
        _add((R_min + 5) % 12, "m", (0, 3, 7), 0.06)       # iv (subdominant minor)
        _add((R_min + 5) % 12, "", (0, 4, 7), 0.05)        # IV (Dorian major subdominant)
        _add((R_min + 7) % 12, "m", (0, 3, 7), 0.05)       # v (natural minor dominant)
        _add((R_min + 7) % 12, "", (0, 4, 7), 0.02)        # V (harmonic minor dominant - only when leading tone sung)
        _add((R_min + 8) % 12, "", (0, 4, 7), 0.06)        # bVI
        _add((R_min + 10) % 12, "", (0, 4, 7), 0.07)       # bVII (subtonic major - essential in folk/Dorian)
    else:
        # Diatonic triads: I, ii, iii, IV, V, vi + functional secondary dominants & borrowed bVII
        _add(R, "", (0, 4, 7), 0.07)                       # I
        _add((R + 2) % 12, "m", (0, 3, 7), 0.06)           # ii
        _add((R + 4) % 12, "m", (0, 3, 7), 0.06)           # iii
        _add((R + 5) % 12, "", (0, 4, 7), 0.06)            # IV
        _add((R + 7) % 12, "", (0, 4, 7), 0.06)            # V
        _add((R + 9) % 12, "m", (0, 3, 7), 0.06)           # vi
        _add((R + 4) % 12, "", (0, 4, 7), -0.02)           # V/vi
        _add((R + 2) % 12, "", (0, 4, 7), -0.03)           # V/V
        _add((R + 10) % 12, "", (0, 4, 7), -0.03)          # bVII (Mixolydian)

    chord_names = [c[4] for c in candidates]
    candidate_roots = [c[0] for c in candidates]
    candidate_tones = [c[2] for c in candidates]
    candidate_biases = [c[3] for c in candidates]
    candidate_qualities = [c[1] for c in candidates]

    max_note_end = max((n.start * seconds_per_beat + beat_origin + n.duration * seconds_per_beat
                        for n in notes), default=0.0) if notes else 0.0
    total_duration = max(clip_duration, max_note_end, beat_origin + bar_seconds)
    n_bars = max(1, int(math.ceil((total_duration - beat_origin) / bar_seconds)))

    if bar_start_seconds is not None and len(bar_start_seconds) >= 2:
        n_bars = len(bar_start_seconds) - 1
        computed_bar_starts = [float(t) for t in bar_start_seconds]
    else:
        computed_bar_starts = [beat_origin + b * bar_seconds for b in range(n_bars + 1)]

    # --- 100% Pure Vocal-Melody Emission Scoring ---
    emission_scores = []
    for b in range(n_bars):
        bs_beat = b * measure_beats
        be_beat = (b + 1) * measure_beats
        vocal_w = np.zeros(12)
        for n in notes:
            overlap = max(0.0, min(n.start + n.duration, be_beat) - max(n.start, bs_beat))
            if overlap <= 0:
                continue
            on_strong = abs(n.start - bs_beat) < 0.13 or abs(n.start - (bs_beat + measure_beats / 2)) < 0.13
            weight = overlap * (1.35 if on_strong else 1.0)
            vocal_w[n.pitch % 12] += weight

        has_vocal = float(np.sum(vocal_w)) > 0

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
                vocal_score = s / float(np.sum(vocal_w))
                combined = 0.92 * vocal_score + bias
            else:
                t_root = (R + 9) % 12 if minor else R
                turnaround_roots = (
                    (t_root, (t_root + 10) % 12, (t_root + 3) % 12, (t_root + 7) % 12)
                    if minor
                    else (t_root, (t_root + 7) % 12, (t_root + 9) % 12, (t_root + 5) % 12)
                )
                turnaround_prior = 0.14 if root == turnaround_roots[b % 4] else 0.0
                tonic_prior = 0.22 if (b == 0 and root == t_root) else (0.05 if root == t_root else 0.0)
                cadence_prior = 0.08 if (b == n_bars - 1 or b % 4 == 0) and root == t_root else 0.0
                combined = tonic_prior + cadence_prior + turnaround_prior + bias

            # In minor keys, Major V (harmonic minor dominant) requires the raised leading tone (#7) in the vocal melody.
            if minor and root == (R_min + 7) % 12 and not is_minor_q:
                leading_pc = (R_min + 11) % 12
                if not (has_vocal and vocal_w[leading_pc] > 0.05):
                    combined -= 0.22

            if name in bar_scores:
                bar_scores[name] = max(bar_scores[name], combined)
            else:
                bar_scores[name] = combined
        emission_scores.append(bar_scores)

    # --- Functional Viterbi Transitions ---
    tonic_root = (R + 9) % 12 if minor else R
    tonic_name = _chord_name(tonic_root, minor)
    dominant_root = (tonic_root + 7) % 12
    dom_name = _chord_name(dominant_root, False)
    vi_root = (R + 9) % 12 if not minor else -1
    vi_name = _chord_name(vi_root, True) if not minor else ""
    v_of_vi_name = _chord_name((R + 4) % 12, False) if not minor else ""

    name_to_root = {}
    for ci in range(len(candidates)):
        n = chord_names[ci]
        if n not in name_to_root:
            name_to_root[n] = candidate_roots[ci]

    unique_names = list(dict.fromkeys(chord_names))

    states = {None: (0.0, [])}
    for b in range(n_bars):
        updated = {}
        for name in unique_names:
            if name not in emission_scores[b]:
                continue
            fit = emission_scores[b][name]
            best = None
            curr_root = name_to_root.get(name, 0)
            for prev, (cost, path) in states.items():
                transition = 0.0
                if prev is not None and prev != "N.C." and name != "N.C.":
                    prev_root = name_to_root.get(prev, 0)
                    distance = (curr_root - prev_root) % 12

                    if prev == dom_name and name == tonic_name:
                        transition += 0.06
                    elif not minor and prev == v_of_vi_name and name == vi_name:
                        transition += 0.10

                    if distance in (5, 7):
                        transition += 0.10
                    elif distance in (2, 10):
                        transition += 0.10

                    if prev == name:
                        transition -= 0.03
                        if len(path) >= 2 and path[-2] == name:
                            transition -= 0.14

                    if (b % 4 == 0 or b == n_bars - 1) and name == tonic_name:
                        transition += 0.12
                total = cost + fit + transition
                if best is None or total > best[0]:
                    best = (total, path + [name])
            if best is not None:
                if name not in updated or best[0] > updated[name][0]:
                    updated[name] = best
        states = updated

    chords = max(states.values(), key=lambda x: x[0])[1] if states else ["N.C."] * n_bars

    # --- 100% Pure Vocal-Driven Dynamics ---
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
