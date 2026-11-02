"""Vocal-melody-conditioned functional harmony and vocal dynamics generator.

Chords are decoded per bar with a Dirichlet-smoothed functional Markov chain and
a Tonal Interval Space (TIS) Viterbi harmonizer, driven only by the singer's
vocal melody notes, phrase cadences, and vocal register/density dynamics
(returning N.C. and 0.0 dynamics when notes is empty).
Zero accompaniment audio or background CQT is ever read or used.

Transition model
----------------
The 7x7 functional-degree pseudo-count tables below are hand-specified expert
priors (they are NOT fitted to Hooktheory or any other corpus). Each row is
Dirichlet-smoothed with a symmetric concentration ``_DIRICHLET_ALPHA`` and
row-normalized into a proper stochastic matrix (``sum_j P_ij = 1``, ``P_ij > 0``).
The additive Viterbi transition weight is the scaled log-likelihood ratio
against a uniform chain, ``c * log(P_ij / (1/K))``, so a transition that is more
likely than chance scores > 0 and a less likely one scores < 0.
"""
from __future__ import annotations
import math
import numpy as np

_DIRICHLET_ALPHA = 1.0
_TRANS_LOG_SCALE = 0.06

# Degree order (major): I, ii, iii, IV, V, vi, bVII.
_MAJOR_TRANS_COUNTS = np.array([
    [  2,  21,  10,  59,  59,  30,  10],
    [ 15,   2,  10,  21, 116,  15,  10],
    [ 10,  21,   2,  42,  15,  59,  10],
    [ 42,  15,   7,   2,  59,  15,   7],
    [163,  10,  10,  21,   2,  59,  10],
    [ 15,  42,  21,  59,  30,   2,  10],
    [ 30,   7,   7,  21,  15,  10,   2],
], dtype=np.float64)

# Degree order (minor): i, III, iv/IV, v, VI, VII, V.
_MINOR_TRANS_COUNTS = np.array([
    [  2,  21,  42,  15,  15,  42,  10],
    [ 21,   2,  30,  15,  21,  30,  10],
    [ 21,  15,   2,  30,  15,  21,  30],
    [ 42,  21,  21,   2,  30,  21,  15],
    [ 15,  21,  21,  15,   2,  30,  15],
    [ 42,  21,  15,  10,  15,   2,  10],
    [116,  15,  15,  10,  30,  15,   2],
], dtype=np.float64)


def _dirichlet_row_normalize(counts: np.ndarray, alpha: float = _DIRICHLET_ALPHA) -> np.ndarray:
    """Posterior-mean transition matrix: P_ij = (n_ij + alpha) / sum_k (n_ik + alpha)."""
    smoothed = np.asarray(counts, dtype=np.float64) + float(alpha)
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _log_likelihood_ratio(probs: np.ndarray, scale: float = _TRANS_LOG_SCALE) -> np.ndarray:
    """Scaled log-likelihood ratio c * log(P_ij / P_uniform) of each transition."""
    p_uniform = 1.0 / probs.shape[1]
    return scale * np.log(probs / p_uniform)


MAJOR_TRANS_PROBS = _dirichlet_row_normalize(_MAJOR_TRANS_COUNTS)
MINOR_TRANS_PROBS = _dirichlet_row_normalize(_MINOR_TRANS_COUNTS)
MAJOR_TRANS_MATRIX = _log_likelihood_ratio(MAJOR_TRANS_PROBS)
MINOR_TRANS_MATRIX = _log_likelihood_ratio(MINOR_TRANS_PROBS)


def _timing(bpm: float, beat_origin: float, meter: str) -> tuple[float, float, float]:
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be finite and positive")
    if not np.isfinite(beat_origin):
        raise ValueError("Beat origin must be finite")
    if not isinstance(meter, str) or "/" not in meter:
        raise ValueError("Unsupported meter")
    try:
        num_s, den_s = meter.strip().split("/", 1)
        num, den = int(num_s), int(den_s)
    except ValueError as exc:
        raise ValueError("Unsupported meter") from exc
    if num <= 0 or den <= 0 or (den & (den - 1)) != 0:
        raise ValueError("Unsupported meter")
    measure_beats = float(num) * (4.0 / float(den))
    seconds_per_beat = 60.0 / bpm
    return measure_beats, seconds_per_beat, measure_beats * seconds_per_beat


def _gttm_note_salience(rel_onset_beats: float, overlap_beats: float, measure_beats: float, meter: str = "4/4") -> float:
    sigma = 0.14
    num_m, den_m = 4, 4
    if isinstance(meter, str) and "/" in meter:
        try:
            num_m, den_m = map(int, meter.strip().split("/", 1))
        except ValueError:
            num_m, den_m = 4, 4
    sub_unit = 4.0 / den_m if den_m > 0 else 1.0
    is_compound = (den_m >= 8 and num_m % 3 == 0 and num_m >= 6)
    if is_compound:
        grid = [(0.0, 1.45)]
        n_comp_beats = num_m // 3
        comp_step = 3.0 * sub_unit
        for cb in range(1, n_comp_beats):
            w_cb = 1.30 if (n_comp_beats % 2 == 0 and cb == n_comp_beats // 2) else 1.25
            grid.append((cb * comp_step, w_cb))
        for s_idx in range(num_m):
            if s_idx % 3 != 0:
                grid.append((s_idx * sub_unit, 0.88))
    else:
        grid = [(0.0, 1.45)]
        if num_m % 2 == 0 and num_m >= 2:
            grid.append((measure_beats * 0.5, 1.22))
        for q in range(1, num_m):
            grid.append((q * sub_unit, 1.05))
        for q in range(num_m * 2):
            grid.append((q * sub_unit * 0.5, 0.85))
    # Circular (von Mises) metrical kernel: the phase coherence term
    # cos(2*pi*(rel_onset - pos) / measure) makes salience smooth and periodic in
    # the bar; kappa is matched so the kernel equals a Gaussian of width sigma
    # near each grid position.
    period = measure_beats if measure_beats > 0 else 4.0
    kappa = (period / (2.0 * math.pi * sigma)) ** 2
    max_salience = 0.85
    for pos, weight in grid:
        phase_coherence = math.cos(2.0 * math.pi * (rel_onset_beats - pos) / period)
        val = weight * math.exp(kappa * (phase_coherence - 1.0))
        if val > max_salience:
            max_salience = val
    dur_factor = min(1.45, max(0.35, 0.40 + 0.65 * math.log(1 + 2.2 * overlap_beats)))
    return max_salience * dur_factor


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
    is_min_like = quality.startswith(("m", "dim")) and not quality.startswith("maj")
    if is_minor_key:
        if offset == 7 and not is_min_like:
            return 6
        return {0: 0, 2: 2, 3: 1, 5: 2, 7: 3, 8: 4, 10: 5}.get(offset, -1)
    if offset == 4 and not is_min_like:
        return 4
    if offset == 2 and not is_min_like:
        return 4
    return {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 10: 6, 11: 4}.get(offset, -1)


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
    """Dirichlet-Smoothed Functional Markov & Tonal Interval Space (TIS) Viterbi Harmonizer.

    Scores diatonic, modal, and secondary-dominant guitar chords per bar from the
    singer's vocal melody notes (`notes`). Each bar's emission combines weighted
    pitch-class coverage with the cosine similarity of 6-coefficient Tonal Interval
    Vectors (bar chroma vs. chord template), using GTTM-style circular metrical
    salience as note weights. Progressions are decoded by Viterbi over the
    log-likelihood-ratio transitions of row-normalized, Dirichlet-smoothed
    functional Markov matrices (``MAJOR_TRANS_PROBS`` / ``MINOR_TRANS_PROBS``,
    hand-specified priors, not corpus-fitted). Per-bar guitar dynamics
    (`bar_dynamics`) are derived from the singer's vocal note density and register.
    """
    NAMES = "C C# D D# E F F# G G# A A# B".split()
    if not isinstance(tonic, str):
        raise ValueError("Malformed tonic")
    t_clean = tonic.strip().replace("♯", "#").replace("♭", "b")
    if not t_clean or t_clean[0].upper() not in "CDEFGAB" or len(t_clean) > 2 or (len(t_clean) == 2 and t_clean[1] not in ("#", "b")):
        raise ValueError("Malformed tonic")
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[t_clean[0].upper()]
    R = (base_pc + (1 if len(t_clean) == 2 and t_clean[1] == "#" else (-1 if len(t_clean) == 2 and t_clean[1] == "b" else 0))) % 12
    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    tonic_root = (R + 9) % 12 if minor else R
    trans_matrix = MINOR_TRANS_MATRIX if minor else MAJOR_TRANS_MATRIX

    def _chord_name(root: int, quality) -> str:
        suf = quality if isinstance(quality, str) else ("m" if quality else "")
        return NAMES[root % 12] + suf

    candidates = []
    seen_names = set()

    def _add(root: int, quality: str, tones: tuple[int, ...], prior_bias: float) -> None:
        r = root % 12
        name = _chord_name(r, quality)
        key = (name, tones)
        if key not in seen_names:
            seen_names.add(key)
            deg = _degree_index(r, quality, tonic_root, minor)
            candidates.append((r, quality, tuple(sorted(set(((r + t) % 12) for t in tones))), prior_bias, name, deg))

    if minor:
        R_min = (R + 9) % 12
        _add(R_min, "m", (0, 3, 7), 0.08)
        _add((R_min + 2) % 12, "dim", (0, 3, 6), 0.01)
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
        _add((R + 11) % 12, "dim", (0, 3, 6), 0.01)
        _add((R + 4) % 12, "", (0, 4, 7), -0.02)
        _add((R + 2) % 12, "", (0, 4, 7), -0.03)
        _add((R + 10) % 12, "", (0, 4, 7), -0.03)

    if notes:
        from subpkgs.guitar_reharm.chord_voicing import EXTENDED_INTERVALS, parse_extended_chord
        observed_pcs = {int(round(float(getattr(n, "pitch", 60)))) % 12 for n in notes}
        seen_ext_pcs = {(0, 4, 7), (0, 3, 7)}
        ext_q_list = []
        for q_str, ivs in EXTENDED_INTERVALS.items():
            norm_pcs = tuple(sorted({iv % 12 for iv in ivs}))
            if len(norm_pcs) < 3 or norm_pcs in seen_ext_pcs:
                continue
            seen_ext_pcs.add(norm_pcs)
            ext_q_list.append((q_str, norm_pcs))
        n_b_est = max(1, math.ceil(max((float(n.start) + float(n.duration) for n in notes), default=0.0) / measure_beats))
        for b_i in range(n_b_est):
            b_pcs = {int(round(float(n.pitch))) % 12 for n in notes if min(float(n.start) + float(n.duration), (b_i + 1) * measure_beats) - max(float(n.start), b_i * measure_beats) > 0}
            if len(b_pcs) < 3:
                continue
            for r_c in range(12):
                if r_c not in b_pcs:
                    continue
                rel = tuple(sorted((p - r_c) % 12 for p in b_pcs))
                if rel in seen_ext_pcs:
                    continue
                rel_s = set(rel)
                if 4 in rel_s:
                    base_q = "7" if 10 in rel_s else ("maj7" if 11 in rel_s else "")
                elif 3 in rel_s:
                    base_q = "m7b5" if (10 in rel_s and 6 in rel_s and 7 not in rel_s) else ("m7" if 10 in rel_s else ("mmaj7" if 11 in rel_s else ("dim7" if (6 in rel_s and 7 not in rel_s and 9 in rel_s) else ("dim" if (6 in rel_s and 7 not in rel_s) else "m"))))
                elif 5 in rel_s:
                    base_q = "7sus4" if 10 in rel_s else "sus4"
                elif 2 in rel_s:
                    base_q = "7sus2" if 10 in rel_s else "sus2"
                else:
                    continue
                bpcs = {x % 12 for x in EXTENDED_INTERVALS[base_q]}
                mods = []
                if 7 in bpcs and 7 not in rel_s:
                    if 6 in rel_s and 6 not in bpcs:
                        mods.append("b5"); bpcs.discard(7); bpcs.add(6)
                    elif 8 in rel_s and 8 not in bpcs:
                        mods.append("#5"); bpcs.discard(7); bpcs.add(8)
                    else:
                        mods.append("no5"); bpcs.discard(7)
                has_7 = (10 in rel_s or 11 in rel_s)
                for p_rel in sorted(rel_s - bpcs):
                    if p_rel == 1:
                        mods.append("b9" if has_7 else "addb9")
                    elif p_rel == 2:
                        mods.append("add9")
                    elif p_rel == 3 and 4 in rel_s:
                        mods.append("#9" if has_7 else "add#9")
                    elif p_rel == 5:
                        mods.append("add11")
                    elif p_rel == 6:
                        mods.append("#11" if has_7 else "add#11")
                    elif p_rel == 8:
                        mods.append("b13" if has_7 else "addb13")
                    elif p_rel == 9:
                        mods.append("add13" if has_7 else "6")
                    elif p_rel == 11:
                        mods.append("add#7")
                p_res = parse_extended_chord("C" + base_q + "".join(mods))
                if p_res is not None and tuple(sorted({x % 12 for x in p_res["intervals"]})) == rel:
                    seen_ext_pcs.add(rel)
                    ext_q_list.append((p_res["quality"], rel))
        for off in range(12):
            r_pc = (tonic_root + off) % 12
            for q_str, ivs in (("", (0, 4, 7)), ("m", (0, 3, 7))):
                if sum(1 for iv in ivs if (r_pc + iv) % 12 in observed_pcs) >= 2:
                    _add(r_pc, q_str, ivs, -0.04)
            for q_str, ivs in ext_q_list:
                if all((r_pc + iv) % 12 in observed_pcs for iv in ivs):
                    _add(r_pc, q_str, ivs, -0.02 - 0.03 * max(0, len(ivs) - 3))

    chord_names = [c[4] for c in candidates]
    candidate_roots = [c[0] for c in candidates]
    candidate_tones = [c[2] for c in candidates]
    candidate_biases = [c[3] for c in candidates]
    candidate_qualities = [c[1] for c in candidates]
    candidate_degrees = [c[5] for c in candidates]

    max_note_end = max((n.start * seconds_per_beat + beat_origin + n.duration * seconds_per_beat
                        for n in notes), default=0.0) if notes else 0.0
    note_bars = math.ceil(max((float(n.start) + float(n.duration) for n in notes), default=0.0) / measure_beats) if notes else 0
    total_duration = max(clip_duration, max_note_end, beat_origin + bar_seconds)
    n_bars = max(1, note_bars, int(math.ceil((total_duration - beat_origin) / bar_seconds)))

    if bar_start_seconds is not None:
        if len(bar_start_seconds) < 2:
            raise ValueError("bar_start_seconds must contain at least 2 boundary timestamps")
        computed_bar_starts = [float(t) for t in bar_start_seconds]
        if any(not math.isfinite(t) for t in computed_bar_starts):
            raise ValueError("bar_start_seconds must be finite")
        if any(computed_bar_starts[i + 1] <= computed_bar_starts[i] for i in range(len(computed_bar_starts) - 1)):
            raise ValueError("bar_start_seconds must be strictly increasing")
        max_raw_end = max((float(n.raw_start) + float(n.raw_duration) for n in notes if getattr(n, "raw_duration", 0.0) > 0), default=0.0) if notes else 0.0
        target_end_sec = max(float(clip_duration), max_raw_end)
        while (len(computed_bar_starts) - 1 < max(1, note_bars)) or (target_end_sec > computed_bar_starts[-1] + 1e-6):
            computed_bar_starts.append(computed_bar_starts[-1] + bar_seconds)
        n_bars = len(computed_bar_starts) - 1
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
    bar_cov_ratios = []
    bar_mean_pitches = []
    first_vocal_bar = -1
    custom_bounds = bar_start_seconds is not None and len(bar_start_seconds) >= 2
    for b in range(n_bars):
        bs_sec = computed_bar_starts[b]
        be_sec = computed_bar_starts[b + 1]
        bar_dur_sec = max(1e-6, be_sec - bs_sec)
        bs_beat_musical = b * measure_beats
        be_beat_musical = (b + 1) * measure_beats
        vocal_w = np.zeros(12)
        cov_beats = 0.0
        pitches_in_bar = []
        for n in notes:
            has_raw = getattr(n, "raw_duration", 0.0) > 0
            if custom_bounds and has_raw:
                ns_sec = float(n.raw_start)
                ne_sec = ns_sec + float(n.raw_duration)
                ov_sec = max(0.0, min(ne_sec, be_sec) - max(ns_sec, bs_sec))
                if ov_sec <= 0:
                    continue
                overlap = (ov_sec / bar_dur_sec) * measure_beats
                rel_onset = ((max(ns_sec, bs_sec) - bs_sec) / bar_dur_sec) * measure_beats
            else:
                ns_b = float(n.start)
                ne_b = ns_b + float(n.duration)
                overlap = max(0.0, min(ne_b, be_beat_musical) - max(ns_b, bs_beat_musical))
                if overlap <= 0:
                    continue
                rel_onset = (max(ns_b, bs_beat_musical) - bs_beat_musical) % measure_beats
            cov_beats += overlap
            pitches_in_bar.append(float(n.pitch))
            salience = _gttm_note_salience(rel_onset, overlap, measure_beats, meter)
            vocal_w[int(round(float(n.pitch))) % 12] += salience
        bar_len_beats = measure_beats
        has_vocal = float(np.sum(vocal_w)) > 0
        bar_vocal_w.append(vocal_w)
        bar_has_vocal.append(has_vocal)
        bar_cov_ratios.append(cov_beats / bar_len_beats)
        bar_mean_pitches.append(float(np.mean(pitches_in_bar)) if pitches_in_bar else 0.0)
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
            is_minor_q = quality.startswith(("m", "dim")) and not quality.startswith("maj")
            third = (root + 3) % 12 if is_minor_q else (root + 4) % 12
            avoid_pcs = set() if quality.startswith("sus") else ({(third + 1) % 12, (third - 1) % 12} if not is_minor_q else set())
            missing_tones = 0
            cov_ratio = 0.0

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
                cov_ratio = sum(vocal_w[pc] for pc in tones) / total_w if total_w > 0 else 0.0
                missing_tones = sum(1 for pc in tones if vocal_w[pc] <= 0.02 * total_w)
                is_dia_triad = (quality in ("", "m") or (quality == "dim" and candidate_degrees[ci] >= 0))
                if not is_dia_triad and (missing_tones > 0 or cov_ratio < (0.92 if len(tones) >= 4 else 0.85)):
                    continue
                tiv_sim = _cosine_sim(bar_tiv, chord_tivs[name])
                combined = 0.75 * vocal_score + 0.17 * tiv_sim + bias - 0.04 * missing_tones
            else:
                is_dia_triad = (quality in ("", "m") or (quality == "dim" and candidate_degrees[ci] >= 0))
                if not is_dia_triad and candidate_degrees[ci] < 0:
                    continue
                t_root = tonic_root
                ctx_w = np.zeros(12)
                for ob in range(n_bars):
                    if bar_has_vocal[ob]:
                        ctx_w += bar_vocal_w[ob] / max(1.0, float(abs(ob - b)))
                ctx_tiv_sim = _cosine_sim(_tiv(ctx_w), chord_tivs[name]) if float(np.sum(ctx_w)) > 0 else 0.0
                tonic_prior = 0.22 if (b == 0 and root == t_root) else (0.05 if root == t_root else 0.0)
                cadence_prior = 0.08 if (b == n_bars - 1 or rel_b % 4 == 0) and root == t_root else 0.0
                combined = 0.25 * ctx_tiv_sim + tonic_prior + cadence_prior + bias

            is_tonic_chord = (root == tonic_root and (quality == tonic_qual or (has_vocal and missing_tones == 0 and cov_ratio >= 0.95)))

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
                if root == dom_root and not is_minor_q:
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

    states = {(None, False): (0.0, [])}
    for b in range(n_bars):
        updated = {}
        rel_b_v = (b - first_vocal_bar) if (first_vocal_bar >= 0 and b >= first_vocal_bar) else b
        for name in unique_names:
            if name not in emission_scores[b]:
                continue
            fit = emission_scores[b][name]
            curr_deg = name_to_degree.get(name, -1)
            for (prev, prev_rep), (cost, path) in states.items():
                transition = 0.0
                is_rep = (prev == name)
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

                    if is_rep:
                        transition -= 0.03
                        if prev_rep:
                            transition -= 0.14

                    if (rel_b_v % 4 == 0 or b == n_bars - 1) and name == tonic_name:
                        transition += 0.12

                total = cost + fit + transition
                state_key = (name, is_rep)
                if state_key not in updated or total > updated[state_key][0]:
                    updated[state_key] = (total, path + [name])
        states = updated

    chords = max(states.values(), key=lambda x: x[0])[1] if states else ["N.C."] * n_bars

    if notes and any(bar_has_vocal):
        vocal_energy = np.zeros(n_bars, dtype=np.float64)
        for b in range(n_bars):
            if bar_has_vocal[b]:
                cov = bar_cov_ratios[b]
                reg = np.clip((bar_mean_pitches[b] - 54.0) / 22.0, 0.0, 1.0)
                vocal_energy[b] = float(np.clip(0.55 * min(1.0, cov) + 0.45 * reg, 0.15, 1.0))
            else:
                vocal_energy[b] = 0.25
        if n_bars == 1:
            smoothed = vocal_energy.copy()
        else:
            padded = np.pad(vocal_energy, (1, 1), mode="edge")
            smoothed = np.convolve(padded, np.ones(3) / 3, mode="valid")
        bar_dynamics = [round(float(np.clip(0.66 + 0.44 * v, 0.66, 1.10)), 4) for v in smoothed[:n_bars]]
    else:
        chords = ["N.C."] * n_bars
        bar_dynamics = [0.0] * n_bars

    return {
        "chords": chords,
        "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
        "bar_dynamics": bar_dynamics,
    }
