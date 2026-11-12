"""Guitar accompaniment and fingerstyle chord-melody arrangement engine.

Supports split-bar chords, X slap, P.M. palm mute, H/P licks, passing bass and
intro/interlude chord-melody.  Three decision layers are used:

* ``_decode_viterbi_texture_states`` - a quantile-regularised 4-state Viterbi
  chain decoder over Gaussian emission potentials and row-normalised transition
  potentials that assigns each bar a texture density level.
* ``_compute_metrical_groove_profile`` - a metrical-hierarchy salience field
  (downbeat / secondary accent / beat / eighth offbeat) from which the active
  subdivision slots and per-slot velocities of a bar are derived.
* ``_solve_bar_treble_dag`` - a layered DAG (trellis) over every treble
  request of a bar, solved exactly with forward DP + backpointer traceback,
  combining vocal-aware emission costs with voice-leading / counterpoint edge
  costs against the bass layer.
"""
from __future__ import annotations

ADVANCED_STYLES = (
    "分解和弦",
    "扫弦伴奏",
    "指弹律动",
    "Neo-Soul 律动",
    "抒情琶音",
)


import math
from .chord_voicing import (STANDARD_TUNING, _assign_fingers, _fingers, _integer, _make_voicing,
                            generate_voicings, _optimize_candidates, parse_extended_chord, _note_pc)
from .melodyt5_bridge import generate_melodyt5_solo_cells
from .reharmonizer import _measure_beats, _overlapping_notes, roman_numeral


# ---------------------------------------------------------------------------
# Texture-state HMM
# ---------------------------------------------------------------------------

# Structural anchor priors for the four texture levels (sparse -> dense).  They
# act as the prior mean of the emission centres and dominate for short songs.
_TEXTURE_ANCHORS = (0.18, 0.42, 0.66, 0.88)
_TEXTURE_QUANTILES = (0.125, 0.375, 0.625, 0.875)
_TEXTURE_PRIOR_SIGMA = 0.155
_TEXTURE_SHRINKAGE_BARS = 12.0
_TEXTURE_MIN_GAP = 0.06
_TEXTURE_JUMP_COST = (0.0, 0.35, 1.20, 2.50)


def _quantile(sorted_values, q):
    """Linear-interpolated empirical quantile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _bar_texture_energy(bar, project, measure, idx, song_low, song_range):
    """Per-bar observation: onset density, register, harmonic rhythm and dynamics, attenuated on vocal rests (x0.45) and 4-bar cadence boundaries (x0.78)."""
    bar_start = idx * measure
    bar_end = (idx + 1) * measure
    ov_notes = _overlapping_notes(project.notes, bar_start, bar_end)
    onset_notes = [n for n in ov_notes if bar_start - 1e-6 <= n.start < bar_end]

    density_norm = min(1.0, len(onset_notes) / max(1.0, measure))
    if ov_notes:
        mean_p = sum(n.pitch for n in ov_notes) / len(ov_notes)
        pitch_norm = max(0.0, min(1.0, (mean_p - song_low) / max(1.0, float(song_range))))
    else:
        pitch_norm = 0.35
    n_segments = len(bar.get('split_segments') or ())
    harmonic_rhythm_norm = max(0.0, min(1.0, (n_segments - 1) / 2.0))
    raw_dyn = float(bar.get('dynamic_scale', 0.88))
    dyn_norm = max(0.0, min(1.0, (raw_dyn - 0.65) / 0.45))

    energy = (0.36 * density_norm + 0.30 * pitch_norm
              + 0.12 * harmonic_rhythm_norm + 0.22 * dyn_norm)
    if not ov_notes:
        energy *= 0.45
    if (idx + 1) % 4 == 0:
        energy *= 0.78
    return energy


def _texture_emission_model(energies):
    """Blend empirical energy quantiles with the anchor priors.

    The shrinkage weight ``n / (n + k)`` lets long songs place the four state
    centres on their own dynamic distribution while short excerpts stay close
    to the structural anchors.  Centres are kept strictly increasing, and the
    emission width ``sigma`` blends ``_TEXTURE_PRIOR_SIGMA`` with the clamped
    empirical standard deviation ``0.7 * clip(std, 0.08, 0.30)``.
    """
    n = len(energies)
    if n == 0:
        return list(_TEXTURE_ANCHORS), _TEXTURE_PRIOR_SIGMA
    ordered = sorted(energies)
    weight = n / (n + _TEXTURE_SHRINKAGE_BARS)
    centres = []
    for anchor, q in zip(_TEXTURE_ANCHORS, _TEXTURE_QUANTILES):
        centres.append((1.0 - weight) * anchor + weight * _quantile(ordered, q))
    for s in range(1, len(centres)):
        centres[s] = max(centres[s], centres[s - 1] + _TEXTURE_MIN_GAP)

    mean_e = sum(energies) / n
    spread = math.sqrt(sum((e - mean_e) ** 2 for e in energies) / n)
    spread = max(0.08, min(0.30, spread))
    sigma = (1.0 - weight) * _TEXTURE_PRIOR_SIGMA + weight * spread * 0.7
    return centres, sigma


def _decode_viterbi_texture_states(bars, project, measure):
    """Quantile-regularised 4-state Viterbi chain decoder for per-bar texture levels.

    Each bar's observation is a weighted energy built from vocal onset
    density, normalised pitch register, harmonic rhythm (number of split
    segments) and the bar dynamic scale, with multiplicative attenuation on
    vocal-rest bars (x0.45) and 4-bar cadence bars (x0.78).  State emission
    centres are a shrinkage blend of the song's own energy quantiles and fixed
    structural anchors (see ``_texture_emission_model``).  Row-normalised
    transition potentials penalise texture jumps, relax at 4-bar phrase starts
    and favour thinning into cadence bars.  The minimum-cost state sequence is
    decoded with forward Viterbi DP and backpointer traceback.  Returns one
    state index (0 = sparse ... 3 = dense) per bar.
    """
    n_bars = len(bars)
    if n_bars == 0:
        return []

    all_pitches = [n.pitch for n in project.notes]
    song_low = min(all_pitches) if all_pitches else 60
    song_high = max(all_pitches) if all_pitches else 72
    song_range = max(1.0, float(song_high - song_low))

    energies = [_bar_texture_energy(bars[idx], project, measure, idx, song_low, song_range)
                for idx in range(n_bars)]
    centres, sigma = _texture_emission_model(energies)
    n_states = len(centres)

    def emission(energy, s):
        z = (energy - centres[s]) / sigma
        return 0.5 * z * z + math.log(sigma)

    def _row_normalized_trans(is_phrase_boundary: bool, is_cadence: bool) -> list[list[float]]:
        rows = []
        for ps in range(n_states):
            raw_row = []
            for s in range(n_states):
                tc = _TEXTURE_JUMP_COST[abs(s - ps)]
                if is_phrase_boundary:
                    tc *= 0.5
                if is_cadence and s <= ps:
                    tc -= 0.35
                raw_row.append(tc)
            log_z = math.log(sum(math.exp(-c) for c in raw_row))
            rows.append([c + log_z for c in raw_row])
        return rows

    INF = float('inf')
    dp = [[INF] * n_states for _ in range(n_bars)]
    back = [[0] * n_states for _ in range(n_bars)]

    for s in range(n_states):
        dp[0][s] = emission(energies[0], s)

    for b in range(1, n_bars):
        trans_mat = _row_normalized_trans(b % 4 == 0, (b + 1) % 4 == 0)
        for s in range(n_states):
            emit = emission(energies[b], s)
            min_cost = INF
            prev_state = 0
            for ps in range(n_states):
                cost = dp[b - 1][ps] + trans_mat[ps][s] + emit
                if cost < min_cost:
                    min_cost = cost
                    prev_state = ps
            dp[b][s] = min_cost
            back[b][s] = prev_state

    last_state = min(range(n_states), key=lambda s: dp[n_bars - 1][s])
    bar_states = [0] * n_bars
    bar_states[n_bars - 1] = last_state
    for b in range(n_bars - 2, -1, -1):
        bar_states[b] = back[b + 1][bar_states[b + 1]]
    return bar_states


# ---------------------------------------------------------------------------
# Metrical groove profile
# ---------------------------------------------------------------------------

_FINGERPICK_STYLES = {'分解和弦', '抒情琶音', '指弹律动'}
_ARPEGGIO_STYLES = {'分解和弦', '抒情琶音'}
_STRUM_STYLE = '扫弦伴奏'
_NEOSOUL_STYLE = 'Neo-Soul 律动'

# Lerdahl-Jackendoff style weights per metrical depth:
# 0 downbeat, 1 secondary accent (half-bar / grouping accent), 2 beat, 3 eighth offbeat.
_DEPTH_WEIGHT = (1.0, 0.75, 0.5, 0.25)
_VEL_FLOOR = 66.0
_VEL_SPAN = 26.0


def _compute_metrical_groove_profile(
    measure,
    style,
    state,
    fast_tempo,
    cm,
    meter='4/4',
    rho: float = 0.0,
    tau_slot: int = 0,
    exempt_pickup: bool = False,
    n_segments: int = 1,
):
    """Derive active eighth-note slots, stroke directions, and slot velocities from a metrical salience field."""
    n_slots = max(1, int(math.ceil(measure * 2 - 1e-9)))
    n_beats = max(1, int(math.ceil(measure - 1e-9)))
    is_strum = style == _STRUM_STYLE
    is_picked = style in _FINGERPICK_STYLES or style == _NEOSOUL_STYLE

    is_compound = str(meter).strip().endswith('/8') and n_slots >= 6 and n_slots % 3 == 0
    if is_compound:
        secondary = n_slots // 2 if (n_slots // 3) % 2 == 0 else (n_slots - 3 if (is_strum or fast_tempo) else 3)
    elif n_beats >= 2 and n_beats % 2 == 0:
        secondary = n_beats
    elif n_beats >= 3:
        secondary = 2 * (n_beats - 1) if (is_strum or fast_tempo) else 2
    else:
        secondary = None
    bass_pivot = secondary if is_compound else (n_beats if n_beats % 2 == 0 else (2 * (n_beats - 1) if n_beats >= 2 else None))

    depths = []
    for slot in range(n_slots):
        if slot == 0:
            depths.append(0)
        elif slot == secondary:
            depths.append(1)
        elif (slot % 3 == 0 if is_compound else slot % 2 == 0):
            depths.append(2)
        else:
            depths.append(3)
    weights = [_DEPTH_WEIGHT[d] for d in depths]
    directions = ['↓' if d <= 2 else '↑' for d in depths]

    if fast_tempo:
        a_next, b_prev, gamma, beat_coupling = 0.20, 0.07, 0.72, 0.35
    else:
        a_next, b_prev, gamma, beat_coupling = 0.22, -0.08, 0.62, 0.0

    def anticipation(slot, gamma_override=None):
        for nxt in range(slot + 1, n_slots):
            if depths[nxt] < depths[slot]:
                return weights[nxt]
        return weights[0] * (gamma if gamma_override is None else gamma_override)

    salience = []
    for slot in range(n_slots):
        depth, w = depths[slot], weights[slot]
        if depth == 3:
            prev_w = weights[slot - 1]
            if is_strum:
                rebound = 0.22 * prev_w if depths[slot - 1] == 1 else (-0.05 if depths[slot - 1] == 0 else 0.0)
                sal = w + rebound + 0.18 * anticipation(slot, gamma_override=0.80)
            else:
                sal = w + a_next * anticipation(slot) + b_prev * prev_w
        elif depth == 2:
            sal = w + beat_coupling * (anticipation(slot) - 0.5) - 0.01 * (slot // 2)
        else:
            sal = w
        salience.append(sal)

    s_idx = min(3, max(0, int(state)))
    if cm:
        budget = n_slots
    elif is_strum:
        if fast_tempo:
            budget = max(n_beats, n_slots - 3)
        else:
            budget = (
                max(n_beats + 1, n_slots - 3),
                max(n_beats + 2, n_slots - 2),
                max(n_beats + 2, n_slots - 2),
                max(n_beats + 3, n_slots - 1),
            )[s_idx]
    elif style in _ARPEGGIO_STYLES:
        if is_compound:
            budget = max(4, n_slots - 2) if fast_tempo else n_slots
        elif fast_tempo:
            budget = (max(3, n_beats), max(4, n_beats + 1), max(5, n_slots - 2), n_slots)[s_idx]
        else:
            budget = (max(4, n_slots - 2), n_slots, n_slots, n_slots)[s_idx]
    elif style == '指弹律动':
        if fast_tempo:
            budget = (max(4, n_beats), max(5, n_beats + 1), max(6, n_slots - 2), n_slots)[s_idx]
        else:
            budget = (max(5, n_slots - 2), max(6, n_slots - 1), n_slots, n_slots)[s_idx]
    elif is_picked:
        # Neo-Soul pocket groove
        if fast_tempo:
            budget = (max(4, n_beats), max(5, n_beats + 1), max(6, n_slots - 2), n_slots)[s_idx]
        else:
            budget = (max(5, n_slots - 3), max(6, n_slots - 2), max(7, n_slots - 1), n_slots)[s_idx]
    else:
        budget = n_slots
    budget = max(1, min(n_slots, budget))

    if rho >= 0.38 and not cm:
        eff_rho = min(1.0, max(0.0, (float(rho) - 0.25) / 0.75))
        k_min = min(n_slots, max(2 if is_compound else 1, int(n_segments)))
        budget = max(k_min, min(n_slots, k_min + int(round(max(0, budget - k_min) * ((1.0 - eff_rho) ** 1.5)))))
        t0 = max(1, int(tau_slot))
        denom = max(1, n_slots - t0)
        for s in range(t0, n_slots):
            if exempt_pickup and s == n_slots - 1 and rho < 1.0 - 1e-6:
                continue
            salience[s] = salience[s] - rho * ((s - t0 + 1) / denom)

    ranked = sorted(range(n_slots), key=lambda s: (-salience[s], s))
    active_slots = {s for s in ranked[:budget]}
    active_slots.add(0)
    if is_compound and budget >= 2 and secondary is not None and secondary not in active_slots:
        if len(active_slots) >= budget:
            removable = [s for s in ranked if s in active_slots and s != 0]
            if removable:
                active_slots.discard(removable[-1])
        active_slots.add(secondary)
    threshold = salience[ranked[min(len(ranked) - 1, budget - 1)]]

    velocities = []
    for slot in range(n_slots):
        w = weights[slot]
        metrical = math.sqrt(max(0.0, (w - 0.25) / 0.75))
        syncopation = max(0.0, max(0.0, salience[slot]) - w)
        velocities.append(int(round(_VEL_FLOOR + _VEL_SPAN * metrical + 10.0 * syncopation)))

    backbeats = {s for s in (range(3, n_slots, 3) if is_compound else range(2, n_slots, 2)) if s != bass_pivot} or ({secondary} if is_compound and secondary else set())
    full_chord_slots = {0} | ({bass_pivot} if bass_pivot is not None else set())
    return dict(n_slots=n_slots, depths=depths, directions=directions, salience=salience, threshold=threshold,
                active_slots=active_slots, velocities=velocities, bass_pivot=bass_pivot,
                backbeat_slots=backbeats, full_chord_slots=full_chord_slots)


# ---------------------------------------------------------------------------
# Layered counterpoint DAG for the treble voice
# ---------------------------------------------------------------------------

# Harmonic cost of the treble-over-bass interval class (negative = preferred).
# Imperfect consonances (-0.4) are favoured, perfect fifths (0.0) neutral,
# fourths (0.5) and doubled octaves/unisons (0.6) mildly penalized, dissonances (0.8..1.2) discouraged.
_BASS_INTERVAL_COST = (0.6, 1.2, 1.0, -0.4, -0.4, 0.5, 1.0, 0.0, -0.4, -0.4, 0.8, 1.2)
_VL_SEMITONE_COST = 0.32
_REPEAT_STRING_COST = 4.5
_STRING_SKIP_COST = 0.65
_PARALLEL_PERFECT_COST = 2.5
_VOICE_CROSSING_COST = 3.0


def _vocal_context(ov_notes, abs_time, beat, measure):
    vocal_onset_here = any(abs(n.start - abs_time) < 0.35 for n in ov_notes)
    active_n = next((n for n in ov_notes if n.start <= abs_time < n.start + n.duration), None)
    if active_n is None:
        active_n = min((n for n in ov_notes if abs(n.start - abs_time) < 0.35), key=lambda n: abs(n.start - abs_time), default=None)
    v_pitch = active_n.pitch if active_n is not None else None
    vocal_gap_here = (0.0 <= beat <= measure) and v_pitch is None and not vocal_onset_here
    return v_pitch, vocal_onset_here, vocal_gap_here


def _default_target_string(beat: float, measure: float, style: str = '', kind='single') -> float:
    """Right-hand target string contour: 1.4 for pinch, 2.2 for dyad, half-bar 3 -> 2 -> 1 roll for '指弹律动' (measure >= 3.5), and full-bar 3 -> 2 -> 1 -> 2 -> 3 arch otherwise."""
    if kind == 'pinch':
        return 1.4
    if isinstance(kind, tuple) and kind[0] == 'dyad':
        return 2.2
    span = max(0.5, float(measure))
    if style == '指弹律动' and span >= 3.5:
        half = span / 2.0
        local = (float(beat) % half) / max(0.5, half)
        return max(1.0, min(3.2, 3.4 - 2.8 * local))
    t_span = max(0.5, span - 1.0)
    phase = min(1.0, max(0.0, (float(beat) - 0.5) / t_span))
    arch = 1.0 - abs(2.0 * phase - 1.0)
    return 3.0 - 2.0 * arch


def _treble_emission_cost(cand, context, bass_pitch, target_string: float = 2.0, is_arpeggio: bool = False):
    """Node cost: right-hand metrical string arch, gentle high-register vocal clearance, and bass consonance."""
    string, _fret, pitch = cand
    v_pitch, vocal_onset_here, vocal_gap_here = context
    str_weight = 1.85 if is_arpeggio else 1.05
    cost = str_weight * abs(float(string) - float(target_string))
    if v_pitch is not None:
        ceil_pitch = max(float(v_pitch) + 5.0, 71.0)
        if pitch > ceil_pitch:
            cost += (2.4 if vocal_onset_here else 1.5) * ((pitch - ceil_pitch) / 3.0)
        if vocal_onset_here and abs(pitch - v_pitch) <= 1:
            cost += 1.6
    if vocal_gap_here and not is_arpeggio:
        cost += -0.85 * (3.5 - string)
    if bass_pitch is not None:
        if pitch <= bass_pitch:
            cost += _VOICE_CROSSING_COST
        else:
            cost += _BASS_INTERVAL_COST[(pitch - bass_pitch) % 12]
    return cost


def _treble_transition_cost(prev, cur, prev_bass, cur_bass, is_arpeggio: bool = False):
    """Edge cost: semitone voice-leading distance, repeated-string refractory, string skips, and parallel perfect 5ths/8ves above sounding bass."""
    p_string, _pf, p_pitch = prev
    c_string, _cf, c_pitch = cur
    vl_w = 0.08 if is_arpeggio else _VL_SEMITONE_COST
    cost = vl_w * abs(c_pitch - p_pitch)
    if c_string == p_string:
        cost += _REPEAT_STRING_COST
    cost += _STRING_SKIP_COST * max(0, abs(c_string - p_string) - 1)
    if prev_bass is not None and cur_bass is not None and p_pitch > prev_bass and c_pitch > cur_bass:
        iv_prev = (p_pitch - prev_bass) % 12
        iv_cur = (c_pitch - cur_bass) % 12
        moved_together = (c_pitch - p_pitch) * (cur_bass - prev_bass) > 0
        if iv_prev == iv_cur and iv_cur in (0, 7) and moved_together:
            cost += _PARALLEL_PERFECT_COST
    return cost


def _solve_bar_treble_dag(slot_requests, ov_notes, bar_idx, measure, init_pitch=None, init_string=None, style: str = ''):
    """Exact minimum-cost treble line over all treble requests of one bar.

    ``slot_requests`` is a chronologically ordered list of dicts with keys
    ``beat``, ``candidates`` (``(string, fret, pitch)`` tuples), ``bass_pitch``
    (sounding bass pitch or ``None``), and optional ``target_string``.  Layer ``t``
    of the DAG holds the candidates of request ``t``; edges connect every node of
    layer ``t-1`` to every node of layer ``t``.  A virtual source carries the
    previous bar's last treble note (``init_pitch`` / ``init_string``) so the
    line stays continuous across bar lines.
    """
    layer_ids = [t for t, req in enumerate(slot_requests) if req.get('candidates')]
    result = [None] * len(slot_requests)
    if not layer_ids:
        return result

    is_arpeggio = style in _ARPEGGIO_STYLES
    layers = [list(slot_requests[t]['candidates']) for t in layer_ids]
    basses = [slot_requests[t].get('bass_pitch') for t in layer_ids]
    targets = [
        float(slot_requests[t]['target_string'])
        if slot_requests[t].get('target_string') is not None
        else _default_target_string(slot_requests[t]['beat'], measure, style=style)
        for t in layer_ids
    ]
    contexts = []
    for t in layer_ids:
        beat = slot_requests[t]['beat']
        contexts.append(_vocal_context(ov_notes, bar_idx * measure + beat, beat, measure))

    source = None
    if init_pitch is not None:
        source = (init_string if init_string is not None else 0, 0, init_pitch)

    INF = float('inf')
    dp = [[INF] * len(layer) for layer in layers]
    back = [[-1] * len(layer) for layer in layers]

    src_vl_w = 0.18 if is_arpeggio else _VL_SEMITONE_COST
    first_beat = float(slot_requests[layer_ids[0]]['beat'])
    for k, cand in enumerate(layers[0]):
        cost = _treble_emission_cost(cand, contexts[0], basses[0], targets[0], is_arpeggio=is_arpeggio)
        if source is not None:
            src_cost = src_vl_w * abs(cand[2] - source[2])
            if init_string is not None and cand[0] == init_string and first_beat < 0.25:
                src_cost += _REPEAT_STRING_COST
            cost += src_cost
        dp[0][k] = cost

    for t in range(1, len(layers)):
        for k, cand in enumerate(layers[t]):
            emit = _treble_emission_cost(cand, contexts[t], basses[t], targets[t], is_arpeggio=is_arpeggio)
            min_cost = INF
            prev_k = 0
            for j, prev_cand in enumerate(layers[t - 1]):
                cost = dp[t - 1][j] + _treble_transition_cost(prev_cand, cand, basses[t - 1], basses[t], is_arpeggio=is_arpeggio)
                if cost < min_cost:
                    min_cost = cost
                    prev_k = j
            dp[t][k] = min_cost + emit
            back[t][k] = prev_k

    last = len(layers) - 1
    k = min(range(len(layers[last])), key=lambda i: dp[last][i])
    for t in range(last, -1, -1):
        result[layer_ids[t]] = layers[t][k]
        k = back[t][k]
    return result


def _wdag_select(mid_treble, ov_notes, bar_idx, measure, beat,
                 prev_treble_pitch, prev_treble_string, bass_pitch=None):
    """Single-slot wrapper around ``_solve_bar_treble_dag`` (kept for external callers)."""
    if not mid_treble:
        return None
    request = dict(beat=beat, candidates=list(mid_treble), bass_pitch=bass_pitch)
    return _solve_bar_treble_dag([request], ov_notes, bar_idx, measure,
                                 prev_treble_pitch, prev_treble_string)[0]


# ---------------------------------------------------------------------------
# Breathing space (B1) and motif-echo chord-melody (B2)
# ---------------------------------------------------------------------------

_MAJ_SCALE_PCS = (0, 2, 4, 5, 7, 9, 11)


def _pitch_to_deg_alter(pitch: int, tonic_pc: int) -> tuple[int, int]:
    """Map MIDI pitch to continuous diatonic scale degree and chromatic alteration (-1, 0, +1)."""
    oct_idx, rel_pc = divmod(int(pitch) - int(tonic_pc), 12)
    best_d, best_alt, best_abs = 0, 0, 99
    for d_idx, d_pc in enumerate(_MAJ_SCALE_PCS):
        diff = rel_pc - d_pc
        if diff > 6:
            diff -= 12
        elif diff < -6:
            diff += 12
        if abs(diff) < best_abs or (abs(diff) == best_abs and diff >= 0):
            best_d, best_alt, best_abs = d_idx, diff, abs(diff)
    return oct_idx * 7 + best_d, max(-1, min(1, best_alt))


def _compute_bar_breathing(bars, project, measure: float, spb: float, explicit_cm_bars: set[int]) -> list[dict]:
    """Compute per-bar breathing coefficient ``rho_b`` and ramp onset ``tau_slot``."""
    count = len(bars)
    if count == 0:
        return []
    tonic_pc = _note_pc(project.tonic)
    sorted_notes = sorted(project.notes, key=lambda n: (n.start, n.pitch))
    comma_gap_beats = max(1.0, 0.55 / max(1e-4, float(spb)))
    section_gap_beats = max(2.0, 0.85 * float(measure))

    phrase_end_weight: dict[int, float] = {}
    for i, n in enumerate(sorted_notes):
        end_beat = n.start + n.duration
        end_bar = max(0, int(math.floor((end_beat - 1e-6) / measure)))
        if i == len(sorted_notes) - 1:
            phrase_end_weight[end_bar] = 1.0
        else:
            gap_next = sorted_notes[i + 1].start - end_beat
            if gap_next >= section_gap_beats - 1e-9:
                phrase_end_weight[end_bar] = max(phrase_end_weight.get(end_bar, 0.0), 1.0)
            elif gap_next >= comma_gap_beats - 1e-9:
                phrase_end_weight[end_bar] = max(phrase_end_weight.get(end_bar, 0.0), 0.30)

    n_slots = max(1, int(math.ceil(measure * 2 - 1e-9)))
    out = []
    for b in range(count):
        bar_start = b * measure
        bar_end = (b + 1) * measure
        is_final_roll = (count >= 2 and b == count - 1 and b not in explicit_cm_bars)
        if is_final_roll:
            out.append(dict(rho=1.0, tau_slot=1, exempt_pickup=False, is_final_roll=True))
            continue
        if not sorted_notes:
            out.append(dict(rho=0.0, tau_slot=1, exempt_pickup=False, is_final_roll=False))
            continue

        e_b = phrase_end_weight.get(b, 0.0)
        ov_wide = _overlapping_notes(sorted_notes, bar_start, bar_end + 1.0)
        bar_ov = [n for n in ov_wide if n.start < bar_end - 1e-9 and n.start + n.duration > bar_start + 1e-9]
        rest_cnt = 0
        hold_cnt = 0
        for s in range(n_slots):
            t_s = bar_start + 0.5 * s
            sounding = any(n.start <= t_s < n.start + n.duration - 1e-6 for n in ov_wide)
            soon_onset = any(t_s - 1e-6 <= n.start < t_s + 1.0 - 1e-6 for n in ov_wide)
            if not sounding and not soon_onset:
                rest_cnt += 1
            if any(
                n.duration >= 2.0 - 1e-6 and n.start + 1.0 - 1e-6 <= t_s < n.start + n.duration - 1e-6
                for n in ov_wide
            ):
                hold_cnt += 1
        rest_b = rest_cnt / n_slots
        hold_b = hold_cnt / n_slots

        cad_b = 0.0
        p_cur = parse_extended_chord(bars[b].get('chord', ''))
        if p_cur is not None and b + 1 < count:
            p_nxt = parse_extended_chord(bars[b + 1].get('chord', ''))
            if p_nxt is not None:
                deg_c = (p_cur['root'] - tonic_pc) % 12
                deg_n = (p_nxt['root'] - tonic_pc) % 12
                if deg_c == 7 and deg_n in (0, 9):
                    cad_b = 1.0
                elif deg_c == 5 and deg_n == 0:
                    cad_b = 0.5

        rho = min(1.0, max(0.0, 0.5 * e_b + 0.25 * rest_b + 0.15 * hold_b + 0.2 * e_b * cad_b))
        if not bar_ov and b > 0 and rest_b >= 0.75:
            rho = max(rho, 0.65)
        bar_onsets = [n.start - bar_start for n in bar_ov if bar_start - 1e-6 <= n.start < bar_end - 1e-6]
        if bar_onsets:
            tau_slot = min(n_slots, max(1, int(math.ceil((max(bar_onsets) + 1.0 - 1e-6) * 2.0))))
        elif bar_ov:
            rel_end = max(min(measure, n.start + n.duration - bar_start) for n in bar_ov)
            tau_slot = min(n_slots, max(1, int(math.ceil((rel_end - 1e-6) * 2.0))))
        else:
            tau_slot = 1

        exempt_pickup = (b + 1 < count) and any(
            bar_end - 1e-6 <= n.start < bar_end + 1.0 - 1e-6 for n in sorted_notes
        )
        out.append(dict(rho=rho, tau_slot=tau_slot, exempt_pickup=exempt_pickup, is_final_roll=False))
    return out


def _step_edit_distance(a: list[int], b: list[int]) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            sub = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + sub)
            prev = cur
    return dp[lb]


def _extract_motif_cells(project, measure: float, spb: float) -> tuple[dict, dict]:
    """Extract primary head motif ``M_head`` and phrase-ending motif ``M_tail`` from vocal notes."""
    tonic_pc = _note_pc(project.tonic)
    sorted_notes = sorted(project.notes, key=lambda n: (n.start, n.pitch))

    def _fallback_cell() -> dict:
        if measure >= 3.5:
            onsets = [0.0, 1.5, 2.5]
            durs = [1.5, 1.0, 1.5]
        elif measure >= 2.5:
            onsets = [0.0, 1.5]
            durs = [1.5, measure - 1.5]
        else:
            onsets = [0.0, max(0.5, measure * 0.5)]
            durs = [max(0.5, measure * 0.5), max(0.5, measure - max(0.5, measure * 0.5))]
        steps = [0, 1, -1][: len(onsets)]
        return dict(onsets=onsets, durs=durs, steps=steps, is_guide_fallback=True)

    if len(sorted_notes) < 4:
        fb = _fallback_cell()
        return fb, fb

    gap_beats = max(0.75, 0.45 / max(1e-4, float(spb)))
    phrases: list[list] = []
    cur_p = [sorted_notes[0]]
    for n in sorted_notes[1:]:
        if n.start - (cur_p[-1].start + cur_p[-1].duration) >= gap_beats - 1e-9:
            phrases.append(cur_p)
            cur_p = [n]
        else:
            cur_p.append(n)
    phrases.append(cur_p)

    cells = []
    for ph in phrases:
        first_bar = int(math.floor((ph[0].start + 1e-6) / measure))
        last_bar = int(math.floor((ph[-1].start + 1e-6) / measure))
        by_bar: dict[int, list] = {}
        for n in ph:
            b = int(math.floor((n.start + 1e-6) / measure))
            by_bar.setdefault(b, []).append(n)
        for b, b_notes in sorted(by_bar.items()):
            if len(b_notes) < 2:
                continue
            onsets, durs, degs = [], [], []
            for n in b_notes:
                rel = max(0.0, min(measure - 0.25, round((n.start - b * measure) * 4.0) / 4.0))
                if onsets and rel <= onsets[-1] + 1e-6:
                    continue
                dur = max(0.25, min(float(n.duration), measure - rel))
                onsets.append(rel)
                durs.append(dur)
                degs.append(_pitch_to_deg_alter(n.pitch, tonic_pc)[0])
            if len(onsets) < 2:
                continue
            steps = [0] + [degs[k] - degs[k - 1] for k in range(1, len(degs))]
            cells.append(dict(
                onsets=onsets,
                durs=durs,
                steps=steps,
                qset={int(round(o * 4)) for o in onsets},
                is_head=(b == first_bar),
                is_tail=(b == last_bar),
            ))

    if not cells:
        fb = _fallback_cell()
        return fb, fb

    for idx_c, c in enumerate(cells):
        rec = 0
        for idx_o, o in enumerate(cells):
            if idx_o == idx_c:
                continue
            u = len(c['qset'] | o['qset'])
            jacc = (len(c['qset'] & o['qset']) / u) if u > 0 else 0.0
            if jacc >= 0.75 and _step_edit_distance(c['steps'][1:], o['steps'][1:]) <= 1:
                rec += 1
        has_leap_long = any(abs(st) >= 2 for st in c['steps'][1:]) and any(d >= 1.0 - 1e-6 for d in c['durs'])
        npb = len(c['onsets']) / max(1.0, measure)
        c['rec'] = rec
        c['head_score'] = (
            1.0 * math.log(1.0 + rec)
            + (0.6 if c['is_head'] else 0.0)
            + (0.4 if has_leap_long else 0.0)
            - 0.8 * max(0.0, npb - 1.5)
        )
        c['tail_score'] = (
            1.0 * math.log(1.0 + rec)
            + (0.8 if c['is_tail'] else 0.0)
            + (0.3 if any(d >= 1.0 - 1e-6 for d in c['durs']) else 0.0)
            - 0.8 * max(0.0, npb - 1.5)
        )

    m_head = max(cells, key=lambda c: c['head_score'])
    m_tail = max(cells, key=lambda c: c['tail_score'])
    return m_head, m_tail


def _is_playable_solo_hand(
    bass_info: tuple[int, int, int] | None,
    lead_sf: tuple[int, int],
    ds_sf: tuple[int, int] | None,
    capo: int,
) -> bool:
    """Hard 4-finger left-hand ergonomic gate simultaneously holding sustained bass + lead + optional DS."""
    s_l, f_l = lead_sf
    if f_l < 0 or f_l > 24 - capo or not (1 <= s_l <= 6):
        return False
    frets6 = [-1] * 6
    if bass_info is not None:
        s_b, f_b, _ = bass_info
        if 1 <= s_b <= 6 and f_b >= 0:
            if s_b == s_l:
                return False
            frets6[6 - s_b] = int(f_b)
    frets6[6 - s_l] = int(f_l)
    if ds_sf is not None:
        s_h, f_h = ds_sf
        if f_h < 0 or f_h > 24 - capo or not (1 <= s_h <= 6) or s_h == s_l:
            return False
        if bass_info is not None and s_h >= bass_info[0]:
            return False
        frets6[6 - s_h] = int(f_h)
    pos = [f for f in frets6 if f > 0]
    if pos and (max(pos) - min(pos)) > 4:
        return False
    f_cnt, _ = _fingers(frets6)
    if f_cnt > 4:
        return False
    _, overflow, strain = _assign_fingers(tuple(frets6), capo)
    return overflow == 0 and math.isfinite(strain)


def _apply_tasteful_anticipation(
    specs: list[tuple[float, float, int]],
    role: int,
    measure: float,
    meter: str,
    w_lo: float,
    is_final_handoff: bool,
    enable_syncopation: bool = True,
) -> list[tuple[float, float, int]]:
    """Apply at most 1 musically idiomatic eighth-note syncopated anticipation inside [w_lo, measure)."""
    if not enable_syncopation or len(specs) < 3 or role == 0 or (role == 3 and is_final_handoff):
        return specs
    is_comp = str(meter).strip().endswith('/8') and int(round(measure * 2.0)) % 3 == 0
    out = list(specs)
    if is_comp and abs(measure - 3.0) < 0.25:
        # In 6/8, allow moving pulse 1.5 -> 1.0 only on Climax role=2 when preceding note keeps >= 0.5b
        if role != 2:
            return out
        for k in range(1, len(out)):
            if k == len(out) - 1 and is_final_handoff:
                continue
            o_prev, d_prev, st_prev = out[k - 1]
            o_cur, d_cur, st_cur = out[k]
            if abs(o_cur - 1.5) < 1e-6 and abs((o_prev + d_prev) - 1.5) < 1e-6 and d_prev >= 1.0 - 1e-6 and o_cur - 0.5 >= w_lo - 1e-6:
                out[k - 1] = (o_prev, d_prev - 0.5, st_prev)
                out[k] = (o_cur - 0.5, d_cur + 0.5, st_cur)
                break
        return out
    if measure >= 3.5 and role in (1, 2):
        # In 4/4, allow moving beat 2.0 -> 1.5 when preceding note keeps >= 0.5b
        for k in range(1, len(out)):
            if k == len(out) - 1 and is_final_handoff:
                continue
            o_prev, d_prev, st_prev = out[k - 1]
            o_cur, d_cur, st_cur = out[k]
            if abs(o_cur - 2.0) < 1e-6 and abs((o_prev + d_prev) - 2.0) < 1e-6 and d_prev >= 1.0 - 1e-6 and o_cur - 0.5 >= w_lo - 1e-6:
                out[k - 1] = (o_prev, d_prev - 0.5, st_prev)
                out[k] = (o_cur - 0.5, d_cur + 0.5, st_cur)
                break
    return out


def _adapt_cell_viterbi(
    bar: dict,
    cell_specs: list[tuple[float, float, int]],
    capo: int,
    tonic_pc: int,
    diatonic: set[int],
    depths: list[int],
    measure: float,
    v0_scale: float = 0.96,
    allowed_strings: tuple[int, ...] = (1, 2, 3),
    start_pc_hint: int | None = None,
    target_pitch: int | None = None,
    prev_pitch_hint: int | None = None,
    register_center: float = 70.0,
    allow_double_stops: bool = False,
    allow_slides: bool = False,
    role: int = 0,
    prefer_home_position: bool = False,
) -> list[dict]:
    """Adapt ``[(beat, duration, step_deg), ...]`` via Joint Fingerboard Viterbi over (pitch, string, DS, slide)."""
    def _seg_at(b_pos: float):
        return next((s for s in bar['split_segments'] if s['beat'] <= b_pos < s['beat'] + s['duration']), None)

    def _seg_bass_info(seg: dict) -> tuple[int, int, int] | None:
        played_all = [(6 - i, int(f), STANDARD_TUNING[i] + capo + int(f)) for i, f in enumerate(seg.get('frets', [])) if int(f) >= 0]
        bass_pool = [x for x in played_all if x[0] >= 4] or played_all
        return min(bass_pool, key=lambda x: x[2]) if bass_pool else None

    valid_specs = []
    for b_pos, dur, st_deg in cell_specs:
        if b_pos < 0.0 or b_pos >= measure - 1e-6 or dur <= 1e-6:
            continue
        seg = _seg_at(b_pos)
        if seg is None or seg.get('chord') == 'N.C.' or all(int(f) < 0 for f in seg.get('frets', [-1])):
            continue
        valid_specs.append((float(b_pos), min(float(dur), measure - float(b_pos)), int(st_deg), seg))
    if not valid_specs:
        return []

    pentatonic = {(tonic_pc + x) % 12 for x in (0, 2, 4, 7, 9)}
    n_slots = len(depths)
    layers = []
    meta = []
    INF = float('inf')
    low_floor = 64.0 if float(register_center) >= 68.0 else 60.0
    n_specs = len(valid_specs)

    for idx_i, (b_pos, dur, st_deg, seg) in enumerate(valid_specs):
        parsed = parse_extended_chord(seg['chord'])
        chord_pcs = (set(parsed['pcs']) | {parsed['bass_pc']}) if parsed else set(diatonic)
        forbid_b3_pcs = set()
        if parsed and 4 in parsed.get('intervals', ()) and 3 not in parsed.get('intervals', ()):
            forbid_b3_pcs.add((parsed['root'] + 3) % 12)
        # Check every split-bar segment overlapped by the sustained note span [b_pos, b_pos + dur)
        later_chord_pcs_list = []
        later_bass_infos = []
        for s_ov in bar.get('split_segments', ()):
            if s_ov is seg or s_ov.get('chord') == 'N.C.':
                continue
            ov_len = min(b_pos + dur, float(s_ov['beat']) + float(s_ov['duration'])) - max(b_pos, float(s_ov['beat']))
            if ov_len > 0.05:
                p_ov = parse_extended_chord(s_ov['chord'])
                if p_ov is not None:
                    later_chord_pcs_list.append(set(p_ov['pcs']) | {p_ov['bass_pc']})
                    if 4 in p_ov.get('intervals', ()) and 3 not in p_ov.get('intervals', ()):
                        forbid_b3_pcs.add((p_ov['root'] + 3) % 12)
                    b_ov = _seg_bass_info(s_ov)
                    if b_ov is not None:
                        later_bass_infos.append(b_ov)

        sweet_pcs = set()
        if parsed:
            for iv in parsed.get('intervals', ()):
                if (iv % 12) in (2, 3, 4, 9, 10, 11):
                    sweet_pcs.add((parsed['root'] + iv) % 12)
        bass_info = _seg_bass_info(seg)
        bass_p = bass_info[2] if bass_info is not None else 48
        base_fret = int(seg.get('base_fret', 1))
        slot_i = min(max(0, int(round(b_pos * 2.0))), n_slots - 1)
        depth_i = depths[slot_i]

        max_fret_pref = min(max(5, base_fret + 3) if prefer_home_position else 12, 24 - capo)
        base_cands = []
        fret_caps = (max_fret_pref, min(14, 24 - capo), 24 - capo) if prefer_home_position else (max_fret_pref, 24 - capo)
        for enforce_hand in (True, False):
            for fret_cap in fret_caps:
                for s in allowed_strings:
                    open_p = STANDARD_TUNING[6 - s] + capo
                    for f in range(0, fret_cap + 1):
                        p = open_p + f
                        if p <= bass_p or (enforce_hand and p < low_floor):
                            continue
                        if enforce_hand and not _is_playable_solo_hand(bass_info, (s, f), None, capo):
                            continue
                        if enforce_hand and any(not _is_playable_solo_hand(b_ov, (s, f), None, capo) for b_ov in later_bass_infos):
                            continue
                        pc = p % 12
                        if pc in forbid_b3_pcs:
                            continue
                        semi_above_ct = (((pc - 1) % 12 in chord_pcs) and (pc not in chord_pcs)) or any(
                            ((pc - 1) % 12 in l_pcs) and (pc not in l_pcs) for l_pcs in later_chord_pcs_list
                        )
                        if semi_above_ct:
                            continue
                        if pc in chord_pcs:
                            harm = -0.30 if pc in sweet_pcs else -0.08
                        elif depth_i == 0:
                            continue
                        elif pc in pentatonic and depth_i >= 2:
                            harm = 0.35
                        elif pc in pentatonic and depth_i == 1:
                            harm = 0.85
                        elif depth_i >= 2 and pc in diatonic:
                            harm = 1.25 if depth_i == 3 else 1.85
                        else:
                            continue
                        base_cands.append((p, s, f, harm))
                if base_cands:
                    break
            if base_cands:
                break

        if not base_cands:
            for s in (1, 2, 3):
                open_p = STANDARD_TUNING[6 - s] + capo
                for f in range(0, 24 - capo + 1):
                    p = open_p + f
                    if p > bass_p and (p % 12) in chord_pcs:
                        base_cands.append((p, s, f, 0.0))
        if not base_cands:
            return []

        # Expand joint states: (p, s, f, harm, ds_info, slide_delta)
        # ds_info is None or (s_h, f_h, p_h, iv_class); slide_delta in (0, 1, 2)
        is_cadence_note = (idx_i == n_specs - 1 and role in (1, 3))
        accent_i = min(
            1.0,
            (0.45 if depth_i <= 1 else 0.0)
            + (0.35 if dur >= 1.0 - 1e-6 else 0.0)
            + (0.40 if is_cadence_note else 0.0)
            + (0.20 if role == 2 else 0.0),
        )
        can_ds_here = allow_double_stops and dur >= 1.0 - 1e-6 and (n_specs >= 2 or is_cadence_note)
        can_slide_here = allow_slides and dur >= 1.0 - 1e-6 and (idx_i > 0 or n_specs == 1 or role == 2)

        joint_cands = []
        for p, s, f, harm in base_cands:
            joint_cands.append((p, s, f, harm, None, 0))
            delta_s = 2 if ((p - 2) % 12) in diatonic else 1
            if can_slide_here and f - delta_s >= 1 and _is_playable_solo_hand(bass_info, (s, f - delta_s), None, capo):
                joint_cands.append((p, s, f, harm, None, delta_s))

            if can_ds_here and s in (1, 2):
                best_ds_for_lead = []
                iv_specs = ((3, s + 1, 3), (4, s + 1, 3)) + (((8, 3, 6), (9, 3, 6)) if s == 1 else ())
                for iv, s_h, iv_cls in iv_specs:
                    p_h = p - iv
                    f_h = p_h - (STANDARD_TUNING[6 - s_h] + capo)
                    if f_h < 0 or f_h > min(14, 24 - capo):
                        continue
                    if bass_info is not None and (s_h >= bass_info[0] or p_h <= bass_info[2] + 2):
                        continue
                    pc_h = p_h % 12
                    if pc_h not in diatonic or pc_h in forbid_b3_pcs:
                        continue
                    if depth_i <= 1 and pc_h not in chord_pcs:
                        continue
                    if any(((pc_h - ct) % 12) == 1 and pc_h not in chord_pcs for ct in chord_pcs):
                        continue
                    if any(
                        ((pc_h - ct) % 12) == 1 and pc_h not in l_pcs
                        for l_pcs in later_chord_pcs_list
                        for ct in l_pcs
                    ):
                        continue
                    if not _is_playable_solo_hand(bass_info, (s, f), (s_h, f_h), capo):
                        continue
                    if any(not _is_playable_solo_hand(b_ov, (s, f), (s_h, f_h), capo) for b_ov in later_bass_infos):
                        continue
                    in_ch_bonus = -0.12 if pc_h in chord_pcs else 0.08
                    best_ds_for_lead.append((in_ch_bonus, (s_h, f_h, p_h, iv_cls)))
                best_ds_for_lead.sort(key=lambda item: item[0])
                for _b_sc, ds_tuple in best_ds_for_lead[:2]:
                    joint_cands.append((p, s, f, harm, ds_tuple, 0))

        layers.append(joint_cands)
        meta.append((b_pos, dur, st_deg, bass_p, base_fret, depth_i, accent_i))

    n_n = len(layers)
    pos_w = 0.18 if prefer_home_position else 0.04
    has_interior_long = any(m_it[1] >= 1.0 - 1e-6 for m_it in meta[1:])

    def _run_viterbi(mu_ds: float) -> list[tuple]:
        dp = [[INF] * len(layer) for layer in layers]
        back = [[-1] * len(layer) for layer in layers]
        b_pos0, dur0, _, bass_p0, bf0, _, accent0 = meta[0]
        for k, (p, s, f, harm, ds_info, sl_d) in enumerate(layers[0]):
            pos_pen0 = pos_w * abs(f - bf0) if f > 0 else 0.0
            reg_pen = (
                0.85 * max(0.0, low_floor - p)
                + 0.16 * max(0.0, p - 77.0)
                + 0.075 * abs(p - float(register_center))
            )
            cost = (
                1.2 * harm
                + 0.06 * max(0.0, f - 9.0)
                + pos_pen0
                + _BASS_INTERVAL_COST[(p - bass_p0) % 12]
                + reg_pen
            )
            if start_pc_hint is not None and (p % 12) != start_pc_hint:
                cost += 0.45
            if prev_pitch_hint is not None:
                cost += 0.16 * abs(p - prev_pitch_hint)
                if p == prev_pitch_hint:
                    cost += 1.55
            if ds_info is not None:
                cost += (mu_ds - 0.95 * accent0) + _BASS_INTERVAL_COST[(ds_info[2] - bass_p0) % 12] * 0.5
            if sl_d > 0:
                can_open_slide = (not has_interior_long) and dur0 >= 1.25 - 1e-6 and (prev_pitch_hint is None or p > prev_pitch_hint)
                cost += -0.14 if can_open_slide else 0.32
            dp[0][k] = cost

        for i in range(1, n_n):
            b_posi, duri, st_deg, bass_pi, bfi, _, accenti = meta[i]
            gap_prev = max(0.25, b_posi - meta[i - 1][0])
            rc_i = float(register_center) + (0.8 if st_deg > 0 else (-0.4 if st_deg < 0 else 0.0))
            for k, (p, s, f, harm, ds_info, sl_d) in enumerate(layers[i]):
                pos_pen_i = pos_w * abs(f - bfi) if f > 0 else 0.0
                reg_pen = (
                    0.85 * max(0.0, low_floor - p)
                    + 0.16 * max(0.0, p - 77.0)
                    + 0.065 * abs(p - rc_i)
                )
                emit_base = (
                    1.2 * harm
                    + 0.06 * max(0.0, f - 9.0)
                    + pos_pen_i
                    + _BASS_INTERVAL_COST[(p - bass_pi) % 12]
                    + reg_pen
                )
                if ds_info is not None:
                    emit_base += (mu_ds - 0.95 * accenti) + _BASS_INTERVAL_COST[(ds_info[2] - bass_pi) % 12] * 0.5
                deg_cur = _pitch_to_deg_alter(p, tonic_pc)[0]
                best_c = INF
                best_j = 0
                for j, (p_prev, s_prev, f_prev, _, ds_prev, sl_prev) in enumerate(layers[i - 1]):
                    if sl_d > 0:
                        if sl_prev > 0 or p <= p_prev:
                            continue
                        if gap_prev < 0.5 - 1e-6 and (s == s_prev or (ds_prev is not None and s == ds_prev[0])):
                            continue
                    deg_prev = _pitch_to_deg_alter(p_prev, tonic_pc)[0]
                    step_cost = 0.95 * abs((deg_cur - deg_prev) - st_deg)
                    repeat_pen = (2.35 if st_deg != 0 else 1.15) if p == p_prev else 0.0
                    fret_jump = abs(f - f_prev) if (f > 0 and f_prev > 0) else 0
                    shift_mult = 0.45 if sl_d > 0 else 1.0
                    trans = (
                        0.24 * abs(p - p_prev)
                        + 0.55 * max(0, abs(s - s_prev) - 1)
                        + shift_mult * (0.12 * max(0, fret_jump - 3) + (1.5 if (s == s_prev and fret_jump > 4) else 0.0))
                        + repeat_pen
                    )
                    if ds_info is not None and ds_prev is not None:
                        trans += 1.65 + (0.35 if ds_info[3] != ds_prev[3] else 0.0)
                    if sl_d > 0:
                        leap_i = p - p_prev
                        trans += 0.24 - (0.44 if (1 <= leap_i <= 5 and duri >= 1.0 - 1e-6) else 0.0) - (0.12 if role == 2 else 0.0)
                    c_tot = dp[i - 1][j] + step_cost + trans + emit_base
                    if c_tot < best_c:
                        best_c = c_tot
                        best_j = j
                dp[i][k] = best_c
                back[i][k] = best_j

        if target_pitch is not None:
            for k, (p, *_rest) in enumerate(layers[-1]):
                dp[-1][k] += 0.65 * abs(p - target_pitch)

        best_last = min(range(len(layers[-1])), key=lambda k: dp[-1][k])
        path = [None] * n_n
        cur_k = best_last
        for i in range(n_n - 1, -1, -1):
            path[i] = layers[i][cur_k]
            cur_k = back[i][cur_k]
        return path

    if allow_double_stops and n_n >= 2:
        # Bisect mu in [0.15, 1.85] so at most 1 sustained note per bar blooms into a warm 3rd/6th Double-Stop
        mu_lo, mu_hi = 0.15, 1.85
        chosen = _run_viterbi(0.75)
        max_ds_bar = 1 if n_n <= 4 else 2
        for _ in range(5):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            cand_path = _run_viterbi(mu_mid)
            ds_cnt = sum(1 for st in cand_path if st[4] is not None)
            chosen = cand_path
            if 1 <= ds_cnt <= max_ds_bar:
                break
            if ds_cnt > max_ds_bar:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
    else:
        chosen = _run_viterbi(0.85)

    # Enforce at most 1 slide-in per bar (keep the strongest/longest one if backpointer picked >1 non-adjacent slides)
    slide_indices = [i for i, st in enumerate(chosen) if st[5] > 0]
    keep_slide_idx = max(slide_indices, key=lambda idx: (meta[idx][1], meta[idx][6])) if len(slide_indices) > 1 else (slide_indices[0] if slide_indices else -1)

    lead_notes = []
    companion_by_idx: dict[int, tuple] = {}
    base_v = 91.0 * float(v0_scale)
    for i, ((p, s, f, _h, ds_info, sl_d), (b_pos, dur, _st, _bp, _bf, depth_i, _acc)) in enumerate(zip(chosen, meta)):
        phase = i / max(1, n_n - 1) if n_n > 1 else 0.5
        arch = 1.0 - abs(2.0 * phase - 1.0)
        vel = int(round(base_v * (0.98 + 0.09 * arch) + (4.0 if depth_i <= 1 else 0.0)))
        vel = max(78, min(106, vel))
        use_sl = (sl_d > 0 and i == keep_slide_idx)
        ev_lead = dict(beat=b_pos, duration=dur, pitch=int(p), string=int(s), velocity=vel, technique='CM')
        if use_sl and f - sl_d >= 1:
            ev_lead['slide_from_fret'] = int(f - sl_d)
        lead_notes.append(ev_lead)
        if ds_info is not None:
            companion_by_idx[i] = (ds_info, sl_d if use_sl else 0, max(58, vel - 9))

    for i in range(len(lead_notes) - 1):
        gap_to_next = lead_notes[i + 1]['beat'] - lead_notes[i]['beat']
        lead_notes[i]['duration'] = min(lead_notes[i]['duration'], max(0.25, gap_to_next))

    notes_out = []
    for i, ev_lead in enumerate(lead_notes):
        notes_out.append(ev_lead)
        if i in companion_by_idx:
            (s_h, f_h, p_h, _iv_cls), sl_h, vel_ds = companion_by_idx[i]
            ev_ds = dict(
                beat=ev_lead['beat'],
                duration=ev_lead['duration'],
                pitch=int(p_h),
                string=int(s_h),
                velocity=int(vel_ds),
                technique='DS',
            )
            if sl_h > 0 and f_h - sl_h >= 1:
                ev_ds['slide_from_fret'] = int(f_h - sl_h)
            notes_out.append(ev_ds)
    return notes_out


def _fold_into_guitar_melody_range(pitch: int, lo: int = 65, hi: int = 77) -> int:
    p = int(pitch)
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def _solo_role_specs(
    role: int,
    measure: float,
    meter: str,
    m_head: dict | None,
    m_tail: dict | None,
    w_lo: float,
    w_hi: float,
    is_final_handoff: bool,
) -> list[tuple[float, float, int]]:
    """Derive 4-bar Call/Answer/Climax/Resolution ('起-承-转-合') solo cell specs inside [w_lo, w_hi] from ``m_head`` and ``m_tail``, falling back to metrical archetypes only when no observed vocal motif exists."""
    span = w_hi - w_lo
    if span < 1.0 - 1e-6:
        return []
    if is_final_handoff and w_hi >= measure - 1e-6 and span >= 2.5:
        eff_hi = w_hi - 0.5
    elif is_final_handoff and w_lo > 0.24 and span >= 1.75:
        eff_hi = w_hi - 0.25
    else:
        eff_hi = w_hi
    eff_span = eff_hi - w_lo
    if eff_span < 0.5 - 1e-6:
        return []

    has_vocal_motif = bool(m_head and not m_head.get('is_guide_fallback') and len(m_head.get('onsets', ())) >= 2)
    tail_cell = m_tail if (m_tail and not m_tail.get('is_guide_fallback') and len(m_tail.get('onsets', ())) >= 2) else m_head

    if has_vocal_motif:
        head_steps = [max(-4, min(4, int(s))) for s in m_head.get('steps', [0, 1])]
        tail_steps = [max(-4, min(4, int(s))) for s in (tail_cell.get('steps', [0, -1]) if tail_cell else [0, -1])]
        nz_head = [s for s in head_steps[1:] if s != 0] or [1]
        nz_tail = [s for s in tail_steps[1:] if s != 0] or [-1]
        d1 = nz_head[0]
        t_end = nz_tail[-1]

        src_cell = tail_cell if (role == 3 and tail_cell is not None) else m_head
        c_onsets = [float(o) for o in src_cell['onsets']]
        c_steps = [max(-4, min(4, int(s))) for s in src_cell['steps']]
        o0 = c_onsets[0]
        rel_o = [max(0.0, o - o0) for o in c_onsets]
        max_rel = rel_o[-1] if rel_o else 0.0
        scale_t = (
            max(0.25, (eff_span - 0.5) / max(0.5, max_rel))
            if max_rel > eff_span - 0.49
            else 1.0
        )

        proj_onsets: list[float] = []
        proj_steps: list[int] = []
        for idx_o, (ro, st_val) in enumerate(zip(rel_o, c_steps)):
            pos = w_lo if idx_o == 0 else (w_lo + round((ro * scale_t) * 2.0) / 2.0)
            pos = min(eff_hi - 0.5, max(w_lo, pos))
            if proj_onsets and pos < proj_onsets[-1] + 0.5 - 1e-6:
                proj_steps[-1] = max(-4, min(4, proj_steps[-1] + st_val))
            else:
                proj_onsets.append(pos)
                proj_steps.append(st_val)

        if not proj_onsets:
            proj_onsets = [w_lo]
            proj_steps = [0]

        specs: list[tuple[float, float, int]] = []
        for k_s, (o_k, st_k) in enumerate(zip(proj_onsets, proj_steps)):
            nxt_o = proj_onsets[k_s + 1] if k_s + 1 < len(proj_onsets) else eff_hi
            specs.append((o_k, max(0.25, nxt_o - o_k), st_k))

        # Apply phrase-role motivic variation ('起-承-转-合') preserving extracted motif rhythms & contours
        ans_step = t_end if t_end < 0 else -max(1, abs(t_end))
        if role == 1:
            if len(specs) >= 4 and eff_span >= 2.4:
                specs = specs[:2] + [(specs[2][0], eff_hi - specs[2][0], ans_step)]
            elif len(specs) >= 2:
                specs[-1] = (specs[-1][0], specs[-1][1], ans_step)
        elif role == 2:
            varied = []
            for k_s, (o_k, d_k, st_k) in enumerate(specs):
                if k_s == 0:
                    new_st = st_k
                elif k_s == 1:
                    new_st = max(1, d1) if st_k >= 0 else max(1, abs(st_k))
                elif k_s < len(specs) - 1:
                    new_st = st_k if st_k != 0 else nz_head[(k_s - 1) % len(nz_head)]
                else:
                    new_st = t_end if st_k == 0 else st_k
                varied.append((o_k, d_k, max(-4, min(4, new_st))))
            specs = varied
        elif role == 3:
            if len(specs) >= 4 and eff_span >= 2.4 and specs[-1][1] < 0.75:
                pen_o = specs[-2][0]
                merged_cad = max(-4, min(4, specs[-2][2] + ans_step))
                specs = specs[:-2] + [(pen_o, eff_hi - pen_o, merged_cad if merged_cad != 0 else ans_step)]
            elif specs:
                specs[-1] = (specs[-1][0], specs[-1][1], ans_step if is_final_handoff or specs[-1][2] == 0 else specs[-1][2])

        max_notes = (4 if measure >= 3.5 else 3) if role in (0, 2) else (3 if measure >= 3.5 else 2)
        while len(specs) > max_notes and len(specs) >= 2:
            idx_short = min(range(len(specs) - 1), key=lambda idx: specs[idx][1] + specs[idx + 1][1])
            o_a, d_a, st_a = specs[idx_short]
            _o_b, d_b, st_b = specs[idx_short + 1]
            specs[idx_short : idx_short + 2] = [(o_a, d_a + d_b, max(-4, min(4, st_a + st_b)))]

        min_notes = (
            (3 if role in (0, 2) else 2)
            if eff_span >= 2.4
            else (2 if eff_span >= 1.45 else 1)
        )
        step_pool = (nz_tail + nz_head) if role == 3 else (nz_head + nz_tail)
        while len(specs) < min_notes:
            idx_long = max(range(len(specs)), key=lambda idx: specs[idx][1])
            o_l, d_l, st_l = specs[idx_long]
            if d_l < 1.0 - 1e-6:
                break
            if role in (1, 3) and idx_long == len(specs) - 1 and idx_long > 0 and d_l >= 1.5 - 1e-6:
                d_first = 0.5
            else:
                d_first = max(0.5, min(d_l - 0.5, round((d_l * 0.5) * 2.0) / 2.0))
            d_second = d_l - d_first
            ins_st = step_pool[(len(specs) - 1) % len(step_pool)]
            if role in (1, 3) and idx_long == len(specs) - 1:
                ins_st = ans_step
            specs[idx_long : idx_long + 1] = [(o_l, d_first, st_l), (o_l + d_first, d_second, ins_st)]
        return specs

    # Instrumental fallback when no vocal motif cells exist (e.g., empty notes or < 4 vocal notes)
    fb_steps = [s for s in (m_head.get('steps', [0, 1])[1:] if m_head else [1]) if s != 0]
    d1_fb = 1 if (not fb_steps or fb_steps[0] >= 0) else -1
    is_comp = str(meter).strip().endswith('/8') and int(round(measure * 2.0)) % 3 == 0

    if w_lo > 0.24:
        if eff_span >= 2.4:
            return [
                (w_lo, 0.5, d1_fb),
                (w_lo + 0.5, 0.5, 1),
                (w_lo + 1.0, 0.5, 1),
                (w_lo + 1.5, max(0.5, eff_span - 1.5), -1),
            ]
        if eff_span >= 1.45:
            return [(w_lo, 0.5, d1_fb), (w_lo + 0.5, max(0.5, eff_span - 0.5), -1 if is_final_handoff else 1)]
        return [(w_lo, max(0.5, eff_span), -1 if is_final_handoff else d1_fb)]

    if is_comp and abs(measure - 3.0) < 0.25:
        if role == 0:
            raw = [(0.0, 1.0, 0), (1.0, 0.5, d1_fb), (1.5, 1.0, 1), (2.5, 0.5, 1)]
        elif role == 1:
            raw = [(0.0, 0.5, 0), (0.5, 0.5, 1), (1.0, 0.5, -1), (1.5, 1.5, -1)]
        elif role == 2:
            raw = [(0.0, 0.5, 1), (0.5, 0.5, 1), (1.0, 0.5, 1), (1.5, 1.0, 1), (2.5, 0.5, -1)]
        else:
            raw = [(0.0, 1.0, 0), (1.0, 0.5, 1), (1.5, 1.5, -2)]
    elif abs(measure - 3.0) < 0.25:
        if role == 0:
            raw = [(0.0, 1.0, 0), (1.0, 1.0, d1_fb), (2.0, 0.5, 1), (2.5, 0.5, 1)]
        elif role == 1:
            raw = [(0.0, 1.5, 1), (1.5, 0.5, -1), (2.0, 1.0, -1)]
        elif role == 2:
            raw = [(0.0, 0.5, 1), (0.5, 0.5, 1), (1.0, 1.0, 1), (2.0, 1.0, -1)]
        else:
            raw = [(0.0, 1.0, 1), (1.0, 2.0, -2)]
    elif measure >= 3.5:
        if role == 0:
            raw = [(0.0, 1.0, 0), (1.0, 0.5, d1_fb), (1.5, 1.0, 1), (2.5, 0.5, 1), (3.0, 1.0, -1)]
        elif role == 1:
            raw = [(0.0, 0.5, 0), (0.5, 0.5, 1), (1.0, 1.0, -1), (2.0, 2.0, -1)]
        elif role == 2:
            raw = [(0.0, 0.5, 1), (0.5, 0.5, 1), (1.0, 0.5, 1), (1.5, 1.0, 1), (2.5, 0.5, -1), (3.0, 1.0, -1)]
        else:
            raw = [(0.0, 1.0, 0), (1.0, 0.5, 1), (1.5, 2.0, -2)]
    else:
        mid = max(0.5, round(measure * 0.5 * 2.0) / 2.0)
        raw = [(0.0, mid, 0), (mid, max(0.5, measure - mid), d1_fb if role != 3 else -1)]

    clipped = []
    for o, d, st in raw:
        if o >= eff_hi - 0.24:
            break
        clipped.append((o, min(d, eff_hi - o), st))
    if not clipped and eff_hi >= 0.5:
        clipped = [(0.0, eff_hi, -1 if is_final_handoff else 0)]
    return clipped


def _build_motif_chord_melody(
    bars: list[dict],
    project,
    measure: float,
    capo: int,
    spb: float,
    meter: str = '4/4',
    fast_tempo: bool = False,
    explicit_cm_bars: set[int] | None = None,
) -> tuple[dict[int, list[dict]], set[int], set[int]]:
    """Build lyrical voice-led solo chord-melody across Intro, Interlude, Turnaround, and Outro gaps.

    Returns ``(motif_cm_map, roll_slot0_bars, breath_silent_bars)``.
    """
    explicit_set = set(explicit_cm_bars or ())
    count = len(bars)
    if count == 0:
        return {}, set(), set()

    tonic_pc = _note_pc(project.tonic)
    diatonic = {(tonic_pc + x) % 12 for x in _MAJ_SCALE_PCS}
    sorted_notes = sorted(project.notes, key=lambda n: (n.start, n.pitch))
    total_beats = count * measure

    m_head, m_tail = _extract_motif_cells(project, measure, spb)
    mt5_cells_raw = (
        generate_melodyt5_solo_cells(sorted_notes, meter, str(project.tonic), False, num_bars=4)
        if len(sorted_notes) >= 12
        else []
    )
    mt5_cells = [
        dict(
            onsets=[float(x[0]) for x in bc],
            durs=[float(x[1]) for x in bc],
            steps=[int(x[2]) for x in bc],
            is_guide_fallback=False,
        )
        for bc in mt5_cells_raw
        if len(bc) >= 2
    ]

    def _select_bar_motifs(j_idx: int, role_idx: int, is_end: bool) -> tuple[dict, dict]:
        if not mt5_cells:
            return m_head, m_tail
        if j_idx == 0:
            return m_head, m_tail
        if is_end:
            return (mt5_cells[-1] if j_idx >= 4 else m_head), m_tail
        var_cell = mt5_cells[(j_idx - 1) % len(mt5_cells)]
        return var_cell, m_tail

    def _last_lead_pitch(ev_list: list[dict], default_p: int | None) -> int | None:
        leads = [e['pitch'] for e in ev_list if e.get('technique', 'CM') == 'CM']
        return leads[-1] if leads else default_p

    head_nz = [max(-3, min(3, int(st))) for st in m_head.get('steps', [])[1:] if st != 0]
    tail_nz = [max(-3, min(3, int(st))) for st in m_tail.get('steps', [])[1:] if st != 0]
    pk_step1 = head_nz[0] if head_nz else 1
    pk_step2 = tail_nz[-1] if tail_nz else pk_step1
    base_groove = _compute_metrical_groove_profile(measure, '指弹律动', 1, fast_tempo, True, meter)
    depths = base_groove['depths']

    motif_cm_map: dict[int, list[dict]] = {}
    roll_slot0_bars: set[int] = set()
    breath_silent_bars: set[int] = set()

    def _pickup_specs_for_measure(hi_b: float) -> list[tuple[float, float, int]]:
        if measure >= 3.5 and hi_b >= measure - 0.25:
            return [(measure - 2.0, 1.0, 0), (measure - 1.0, 1.0, pk_step1)]
        if measure >= 2.5 and hi_b >= measure - 0.25:
            return [(measure - 1.5, 1.5, 0)]
        return [(max(0.5, hi_b - 1.0), min(1.0, max(0.5, hi_b - max(0.5, hi_b - 1.0))), 0)]

    def _intro_bell_specs(
        j_idx: int,
        n_wb_tot: int,
        w_lo: float,
        w_hi: float,
    ) -> list[tuple[float, float, int]]:
        """Calm, flowing home-position chord-melody bell phrases (2-4 lyrical notes/bar) for multi-bar Intros."""
        is_end = (j_idx == n_wb_tot - 1)
        is_comp_m = str(meter).strip().endswith('/8') and int(round(measure * 2.0)) % 3 == 0
        if j_idx == 0 and w_lo <= 1e-6:
            if n_wb_tot <= 1:
                return _pickup_specs_for_measure(w_hi)
            if abs(measure - 3.0) < 0.25:
                raw = [(1.5, 1.0, 0), (2.5, 0.5, 1)] if is_comp_m else [(1.0, 1.0, 0), (2.0, 1.0, 1)]
            elif measure >= 3.5:
                raw = [(1.0, 1.0, 0), (2.0, 1.0, 1), (3.0, 1.0, -1)]
            else:
                half_m = max(0.5, round(measure * 0.5 * 2.0) / 2.0)
                raw = [(half_m, max(0.5, measure - half_m), 0)]
        elif abs(measure - 3.0) < 0.25:
            if is_end:
                raw = [(0.0, 1.5, 0), (1.5, 1.5, -1)]
            elif is_comp_m:
                raw = [(0.0, 1.5, 0), (1.5, 1.0, 1), (2.5, 0.5, -1)] if (j_idx % 2 == 1) else [(0.0, 1.0, 0), (1.0, 0.5, 1), (1.5, 1.5, -1)]
            else:
                raw = [(0.0, 2.0, 0), (2.0, 1.0, 1)] if (j_idx % 2 == 1) else [(0.0, 1.0, 0), (1.0, 2.0, -1)]
        elif measure >= 3.5:
            if is_end:
                raw = [(0.0, 1.5, 0), (1.5, 0.5, 1), (2.0, 2.0, -1)]
            elif j_idx % 2 == 1:
                raw = [(0.0, 1.5, 0), (1.5, 0.5, 1), (2.0, 1.0, -1), (3.0, 1.0, 0)]
            else:
                raw = [(0.0, 1.0, 0), (1.0, 1.0, 1), (2.0, 1.5, -1), (3.5, 0.5, 0)]
        else:
            half_m = max(0.5, round(measure * 0.5 * 2.0) / 2.0)
            raw = [(0.0, half_m, 0), (half_m, max(0.5, measure - half_m), -1 if is_end else 1)]
        out = []
        for o_i, d_i, s_i in raw:
            o_eff = max(w_lo, o_i)
            if o_eff >= w_hi - 0.49:
                continue
            out.append((o_eff, min(d_i, w_hi - o_eff), s_i))
        return out or [(w_lo, max(0.5, w_hi - w_lo), 0)]

    if not sorted_notes:
        # Purely instrumental project with no vocal notes: weave lyrical solo across all bars except final roll
        prev_p = None
        solo_end = max(1, count - 1) if count >= 2 else count
        for b in range(solo_end):
            if b in explicit_set:
                continue
            if b == 0:
                roll_slot0_bars.add(0)
            role = 3 if (b == solo_end - 1 and solo_end > 1) else (b % 4)
            arch_pos = b / max(1, solo_end - 1)
            rc = 69.0 + 5.0 * math.sin(math.pi * arch_pos)
            specs = _solo_role_specs(role, measure, meter, None, None, 0.0, measure, b == solo_end - 1)
            specs = _apply_tasteful_anticipation(specs, role, measure, meter, 0.0, b == solo_end - 1, enable_syncopation=(solo_end >= 3))
            adapted = _adapt_cell_viterbi(
                bars[b], specs, capo, tonic_pc, diatonic, depths, measure,
                v0_scale=0.98, allowed_strings=(1, 2, 3), prev_pitch_hint=prev_p, register_center=rc,
                allow_double_stops=(b > 0 and solo_end >= 2), allow_slides=(b > 0), role=role,
            )
            if adapted:
                motif_cm_map[b] = adapted
                prev_p = _last_lead_pitch(adapted, prev_p)
        if count >= 2:
            breath_silent_bars.add(count - 1)
        return motif_cm_map, roll_slot0_bars, breath_silent_bars

    # Collect continuous vocal-free intervals [g_s, g_e] across [0, total_beats], tracking preceding sung span
    vocal_gaps: list[tuple[float, float, str, float]] = []
    first_v_start = max(0.0, float(sorted_notes[0].start))
    if first_v_start >= 1.5 - 1e-6:
        vocal_gaps.append((0.0, min(total_beats, first_v_start), 'intro', 0.0))

    seg_sung_start = first_v_start
    cur_v_end = max(0.0, float(sorted_notes[0].start + sorted_notes[0].duration))
    min_turnaround_beats = max(2.25, 0.85 * measure)
    for n in sorted_notes[1:]:
        ns = float(n.start)
        ne = float(n.start + n.duration)
        if ns - cur_v_end >= min_turnaround_beats - 1e-6 and cur_v_end < total_beats - 1e-6:
            g_s = max(0.0, cur_v_end)
            g_e = min(total_beats, ns)
            if g_e - g_s >= min_turnaround_beats - 1e-6:
                vocal_gaps.append((g_s, g_e, 'interior', max(0.0, cur_v_end - seg_sung_start)))
            seg_sung_start = ns
        cur_v_end = max(cur_v_end, ne)

    if total_beats - cur_v_end >= measure - 1e-6:
        vocal_gaps.append((max(0.0, cur_v_end), total_beats, 'outro', max(0.0, cur_v_end - seg_sung_start)))

    last_turnaround_end_bar = -99

    for g_s, g_e, kind, sung_span_beats in vocal_gaps:
        gap_len = g_e - g_s
        b_first = max(0, min(count - 1, int(math.floor((g_s + 1e-6) / measure))))
        b_last = max(b_first, min(count - 1, int(math.floor((g_e - 1e-6) / measure))))
        full_silent_bars = [
            b for b in range(b_first, b_last + 1)
            if b not in explicit_set and not _overlapping_notes(sorted_notes, b * measure, (b + 1) * measure)
        ]

        if kind == 'interior':
            is_multi_bar_interlude = (gap_len >= 1.65 * measure - 1e-6) or (len(full_silent_bars) >= 2)
            is_section_turnaround = (
                gap_len >= max(2.5, 0.90 * measure) - 1e-6
                and sung_span_beats >= 3.5 * measure - 1e-6
                and (b_first - last_turnaround_end_bar) >= 4
            )
            if not (is_multi_bar_interlude or is_section_turnaround):
                for fb in full_silent_bars:
                    breath_silent_bars.add(fb)
                continue

        if kind == 'outro':
            # Final bar of song (count - 1) is always B1 sustained rolled chord
            if b_last == count - 1:
                breath_silent_bars.add(count - 1)
            win_bars = []
            for b in range(b_first, b_last):
                if b in explicit_set:
                    continue
                w_lo = 0.0 if g_s <= b * measure + 1e-6 else math.ceil((g_s - b * measure + 0.35) * 2.0 - 1e-6) / 2.0
                w_hi = measure
                if w_hi - w_lo >= 1.25 - 1e-6:
                    win_bars.append((b, w_lo, w_hi))
            if not win_bars:
                continue
            prev_p = _fold_into_guitar_melody_range(sorted_notes[-1].pitch)
            n_wb = len(win_bars)
            for j, (b, w_lo, w_hi) in enumerate(win_bars):
                if j > 0 and w_lo <= 1e-6:
                    roll_slot0_bars.add(b)
                role = 3 if j == n_wb - 1 else (j % 3)
                v0 = max(0.84, 0.98 - 0.05 * j)
                rc = max(67.0, 71.5 - 1.2 * j)
                ch_m, ct_m = _select_bar_motifs(j, role, j == n_wb - 1)
                specs = _solo_role_specs(role, measure, meter, ch_m, ct_m, w_lo, w_hi, j == n_wb - 1)
                specs = _apply_tasteful_anticipation(specs, role, measure, meter, w_lo, j == n_wb - 1, enable_syncopation=(len(sorted_notes) >= 12))
                adapted = _adapt_cell_viterbi(
                    bars[b], specs, capo, tonic_pc, diatonic, depths, measure,
                    v0_scale=v0, allowed_strings=(1, 2, 3), prev_pitch_hint=prev_p, register_center=rc,
                    allow_double_stops=(b > 0 and n_wb >= 2),
                    allow_slides=(b > 0 and ((n_wb >= 3 and role == 2) or (n_wb == 2 and j == 1))),
                    role=role,
                )
                if adapted:
                    motif_cm_map[b] = adapted
                    prev_p = _last_lead_pitch(adapted, prev_p)
            continue

        # Build valid solo windows (b, w_lo, w_hi) across [b_first..b_last]
        # For multi-bar Intros where vocals enter mid-bar inside b_last, leave b_last as pure chordal accompaniment
        intro_bar_end = (
            b_last - 1
            if (kind == 'intro' and (b_last - b_first) >= 2 and g_e < (b_last + 1) * measure - 0.25)
            else b_last
        )
        win_bars = []
        for b in range(b_first, intro_bar_end + 1):
            if b in explicit_set or (count >= 2 and b == count - 1):
                continue
            w_lo = 0.0 if g_s <= b * measure + 1e-6 else math.ceil((g_s - b * measure + 0.35) * 2.0 - 1e-6) / 2.0
            w_hi = measure if g_e >= (b + 1) * measure - 1e-6 else math.floor((g_e - b * measure - 0.35) * 2.0 + 1e-6) / 2.0
            min_w = 1.0 if kind == 'intro' else 1.25
            if w_hi - w_lo >= min_w - 1e-6:
                win_bars.append((b, w_lo, w_hi))

        if not win_bars:
            for fb in full_silent_bars:
                breath_silent_bars.add(fb)
            continue

        after_notes = [n for n in sorted_notes if n.start >= g_e - 1e-6]
        before_notes = [n for n in sorted_notes if n.start + n.duration <= g_s + 1e-6]
        entry_pitch = _fold_into_guitar_melody_range(after_notes[0].pitch) if after_notes else (67 + (tonic_pc % 7))
        if entry_pitch <= 67:
            approach_target = entry_pitch + (2 if ((entry_pitch + 2) % 12) in diatonic else 1)
        else:
            approach_target = entry_pitch - (2 if ((entry_pitch - 2) % 12) in diatonic else 1)

        if kind == 'intro':
            if win_bars[0][0] == 0 and win_bars[0][1] <= 1e-6:
                roll_slot0_bars.add(0)
            # Strict 1-bar intro contract: Beat-0 rolled chord + concise home-position pickup lead into Bar 1
            if len(win_bars) == 1 and win_bars[0][0] == 0:
                b0, _w0, w_hi0 = win_bars[0]
                specs = _pickup_specs_for_measure(w_hi0)
                notes_b0 = _adapt_cell_viterbi(
                    bars[b0], specs, capo, tonic_pc, diatonic, depths, measure,
                    v0_scale=0.90, allowed_strings=(1, 2, 3), target_pitch=approach_target, register_center=67.5,
                    allow_double_stops=False, allow_slides=False, role=0, prefer_home_position=True,
                )
                if notes_b0:
                    motif_cm_map[b0] = notes_b0
                continue

            prev_p = None
            n_wb = len(win_bars)
            for j, (b, w_lo, w_hi) in enumerate(win_bars):
                is_last = (j == n_wb - 1)
                arch_pos = j / max(1, n_wb - 1)
                v0 = 0.88 + 0.06 * math.sin(math.pi * arch_pos)
                rc = 66.5 + 2.0 * math.sin(math.pi * arch_pos)
                role = 0 if j == 0 else (3 if is_last else (2 if j == n_wb // 2 else 1))
                specs = _intro_bell_specs(j, n_wb, w_lo, w_hi)
                adapted = _adapt_cell_viterbi(
                    bars[b], specs, capo, tonic_pc, diatonic, depths, measure,
                    v0_scale=v0, allowed_strings=(1, 2, 3),
                    target_pitch=(approach_target if is_last else None),
                    prev_pitch_hint=prev_p,
                    register_center=rc,
                    allow_double_stops=(b > 0 and n_wb >= 2 and j in (n_wb // 2, n_wb - 1)),
                    allow_slides=(b > 0 and n_wb >= 3 and j == 1),
                    role=role,
                    prefer_home_position=True,
                )
                if adapted:
                    motif_cm_map[b] = adapted
                    prev_p = _last_lead_pitch(adapted, prev_p)
        else:
            # Interior Interlude or Section Turnaround Solo
            last_pc = (before_notes[-1].pitch % 12) if before_notes else None
            prev_p = _fold_into_guitar_melody_range(before_notes[-1].pitch) if before_notes else None
            n_wb = len(win_bars)
            for j, (b, w_lo, w_hi) in enumerate(win_bars):
                is_last = (j == n_wb - 1)
                if n_wb == 1:
                    role = 1
                    rc = 69.5
                    v0 = 0.95
                else:
                    role = 3 if is_last else (j % 4)
                    arch_pos = j / max(1, n_wb - 1)
                    rc = 68.0 + 3.8 * math.sin(math.pi * arch_pos)
                    v0 = 0.94 + 0.08 * math.sin(math.pi * arch_pos)
                if is_last and n_wb >= 3 and w_lo <= 1e-6 and w_hi >= measure - 1e-6:
                    roll_slot0_bars.add(b)
                ch_m, ct_m = _select_bar_motifs(j, role, is_last)
                specs = _solo_role_specs(role, measure, meter, ch_m, ct_m, w_lo, w_hi, is_last)
                specs = _apply_tasteful_anticipation(
                    specs, role, measure, meter, w_lo, is_last,
                    enable_syncopation=(len(sorted_notes) >= 12 and n_wb >= 2),
                )
                adapted = _adapt_cell_viterbi(
                    bars[b], specs, capo, tonic_pc, diatonic, depths, measure,
                    v0_scale=v0, allowed_strings=(1, 2),
                    start_pc_hint=(last_pc if j == 0 else None),
                    target_pitch=(approach_target if is_last else None),
                    prev_pitch_hint=prev_p,
                    register_center=rc,
                    allow_double_stops=(b > 0 and n_wb >= 2),
                    allow_slides=(b > 0 and ((n_wb >= 3 and role == 2) or (n_wb == 2 and j == 1))),
                    role=role,
                )
                if adapted:
                    motif_cm_map[b] = adapted
                    prev_p = _last_lead_pitch(adapted, prev_p)
            last_turnaround_end_bar = b_last

    return motif_cm_map, roll_slot0_bars, breath_silent_bars


# ---------------------------------------------------------------------------
# Arrangement
# ---------------------------------------------------------------------------

def arrange_advanced(
    project,
    chords: list[str] | None = None,
    style: str = "指弹律动",
    capo: int | None = None,
    custom_voicings: list[list[int] | None] | None = None,
    split_chords: list[list[dict]] | None = None,
    passing_bass: bool = True,
    slap_x: bool = False,
    palm_mute: bool = False,
    embellish_hp: bool = False,
    intro_mode: str = "chord_melody",
    instrumental_melody: dict[int, list[dict]] | None = None,
    bar_dynamics: list[float] | None = None,
) -> dict:
    """Generate 6-string guitar tab events with split-bar harmony, X/PM/HP/PB articulations and intro/interlude chord-melody."""
    if style not in ADVANCED_STYLES:
        raise ValueError(f'Unknown style: {style!r}')
    if intro_mode not in {'none', 'nc', 'groove', 'chord_melody'}:
        raise ValueError(f'Unknown intro mode: {intro_mode!r}')
    capo = 0 if capo is None else capo
    _integer(capo,0,12,'capo')
    measure = _measure_beats(project)
    if not math.isfinite(project.bpm) or project.bpm <= 0:
        raise ValueError('BPM must be positive and finite')
    if chords is None:
        end = max((n.start+n.duration for n in project.notes),default=0.0)
        if getattr(project, 'source', None):
            end = max(end, getattr(project, 'clip_duration', 0.0)*project.bpm/60)
        chords = [project.tonic.strip() if isinstance(project.tonic, str) else project.tonic]*math.ceil(end/measure)
    else:
        chords = [c.strip() if isinstance(c, str) else c for c in chords]
    count = len(chords)
    for name, supplied in [('custom_voicings',custom_voicings),('split_chords',split_chords)]:
        if supplied is not None and len(supplied) != count:
            raise ValueError(f'{name} must match bar count')
    bars, layers, flat = [], [], []
    for index, chord in enumerate(chords):
        custom = None
        if custom_voicings is not None and custom_voicings[index] is not None:
            custom = _make_voicing(chord,custom_voicings[index],capo,chord)
        supplied = ([dict(beat=0.0,duration=measure,chord=chord)] if split_chords is None
                    else [dict(s) for s in split_chords[index]])
        segments = sorted(supplied,key=lambda s:s['beat'])
        end = 0.0
        for j, segment in enumerate(segments):
            beat, duration = segment['beat'], segment['duration']
            if (not math.isfinite(beat) or not math.isfinite(duration) or duration <= 0
                    or abs(beat-end) > 1e-9):
                raise ValueError(f'Invalid segment coverage in bar {index}')
            end = beat+duration
            symbol = segment['chord'].strip() if isinstance(segment['chord'], str) else segment['chord']
            segment['chord'] = symbol
            fixed = None
            if 'frets' in segment:
                provenance = segment['capo'] if 'capo' in segment else capo
                _integer(provenance,0,12,'segment capo')
                supplied_voicing = _make_voicing(symbol,segment['frets'],provenance,symbol)
                if provenance == capo:
                    fixed = supplied_voicing
            elif j == 0 and custom is not None:
                fixed = _make_voicing(symbol,custom['frets'],capo,symbol)
            candidates = [fixed] if fixed is not None else generate_voicings(symbol,capo)
            if not candidates:
                raise ValueError(f'No voicing for {symbol!r} in bar {index}')
            segment.update(beat=float(beat),duration=float(duration),roman=roman_numeral(symbol,project.tonic),capo=capo)
            flat.append(segment)
            layers.append(candidates)
        if segments and abs(end-measure) > 1e-9:
            raise ValueError(f'Incomplete segment coverage in bar {index}')
        bars.append(dict(bar=index,chord=segments[0]['chord'] if segments else 'N.C.',
                         roman=segments[0]['roman'] if segments else 'N.C.',frets=[-1]*6,split_segments=segments))
    for segment, voicing in zip(flat,_optimize_candidates(layers)):
        segment['frets'] = list(voicing['frets'])
        segment['base_fret'] = voicing['base_fret']
    for bar in bars:
        if bar['split_segments']:
            bar['frets'] = list(bar['split_segments'][0]['frets'])
        if bar_dynamics is not None and bar['bar'] < len(bar_dynamics):
            dyn = max(0.76, min(1.08, float(bar_dynamics[bar['bar']])))
            bar['dynamic_scale'] = round(dyn, 3)

    bar_states = _decode_viterbi_texture_states(bars, project, measure)

    _GENTLE_STYLES = {'民谣扫弦', '流行扫弦', '拍弦律动', '扫弦伴奏', '指弹律动'}

    events = {}
    slaps = []
    def active(bar, beat):
        return next((s for s in bar['split_segments'] if s['beat'] <= beat < s['beat']+s['duration']),None)

    def strings(segment):
        return [(6-i,f,STANDARD_TUNING[i]+capo+f) for i,f in enumerate(segment['frets']) if f >= 0]

    def bass_of(segment):
        candidates = [x for x in strings(segment) if x[0] >= 4] or strings(segment)
        return min(candidates,key=lambda x:x[2]) if candidates else None

    def insert(bar, beat, duration, played, velocity, technique='', replace=False, direction=None, roll_step_sec=None, slide_from_fret=None):
        segment = active(bar,beat)
        if segment is None:
            return
        string, fret, pitch = played
        if technique in ('CM', 'DS', 'SL'):
            duration = min(duration, measure - beat)
        else:
            duration = min(duration, segment['beat'] + segment['duration'] - beat, measure - beat)
        if duration <= 0:
            return
        dyn_scale = bar.get('dynamic_scale')
        if dyn_scale is not None:
            if technique in ('CM', 'SL'):
                floor_cm = min(int(velocity), 84)
                cm_dyn = 0.94 + 0.5 * (float(dyn_scale) - 0.94)
                velocity = max(floor_cm, min(127, round(velocity * cm_dyn)))
            elif technique == 'DS':
                cm_dyn = 0.94 + 0.5 * (float(dyn_scale) - 0.94)
                velocity = max(54, min(115, round(velocity * cm_dyn)))
            else:
                velocity = max(1, min(127, round(velocity * dyn_scale)))
        start = bar['bar']*measure+beat
        event = dict(bar=bar['bar'],beat=beat,start=start,duration=duration,string=string,fret=fret,
                     pitch=pitch,velocity=velocity,technique=technique,chord=segment['chord'],
                     segment=bar['split_segments'].index(segment))
        if direction is not None:
            event['direction'] = direction
        if roll_step_sec is not None:
            event['roll_step_sec'] = float(roll_step_sec)
        if slide_from_fret is not None:
            event['slide_from_fret'] = int(slide_from_fret)
        key = (start,string)
        if replace or key not in events:
            events[key] = event

    def clear_beat(bar_idx, beat_pos):
        """Remove technique-free accompaniment events at a specific beat to make room for ornaments."""
        bar_start = bar_idx * measure + beat_pos
        to_remove = []
        for key, ev in events.items():
            if abs(key[0] - bar_start) < 1e-9 and ev['technique'] == '':
                to_remove.append(key)
        for key in to_remove:
            del events[key]

    tonic_pc = _note_pc(project.tonic)
    diatonic = {(tonic_pc+x)%12 for x in (0,2,4,5,7,9,11)}
    _PHRASE_VEL_CURVE = [0.97, 1.00, 1.04, 0.98]

    spb = 60.0 / project.bpm
    half_beat_sec = 0.5 * spb
    fast_tempo = half_beat_sec < 0.195
    step_dur = 1.0 if fast_tempo else 0.5
    roll_sec = max(0.012, min(0.030, 0.05 * spb))

    vocal_counts = {}
    for bar in bars:
        idx = bar['bar']
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        vocal_counts[idx] = len(_overlapping_notes(project.notes, bar_start, bar_end))

    explicit_cm_bars = set(instrumental_melody.keys()) if instrumental_melody is not None else set()
    bar_breathing = _compute_bar_breathing(bars, project, measure, spb, explicit_cm_bars)
    if intro_mode == 'chord_melody':
        motif_cm_map, roll_slot0_bars, breath_silent_bars = _build_motif_chord_melody(
            bars, project, measure, capo, spb, project.meter, fast_tempo, explicit_cm_bars
        )
    else:
        motif_cm_map, roll_slot0_bars, breath_silent_bars = {}, set(), set()

    prev_treble_pitch = None
    prev_treble_string = None
    last_orn_bar = -99
    last_hp_bar = -99
    last_pb_bar = -99

    for bar in bars:
        index = bar['bar']
        bar_index = index
        silent = not _overlapping_notes(project.notes,index*measure,(index+1)*measure)
        if intro_mode == 'nc' and silent:
            continue
        explicit_cm = instrumental_melody is not None and index in instrumental_melody
        has_motif_cm = index in motif_cm_map and bool(motif_cm_map[index])
        cm = intro_mode == 'chord_melody' and (explicit_cm or has_motif_cm)
        b_info = bar_breathing[index]
        rho_b = b_info['rho']
        if index in breath_silent_bars and not cm:
            rho_b = max(rho_b, 0.72)

        if b_info['is_final_roll']:
            for seg in bar['split_segments']:
                if seg.get('chord') == 'N.C.' or all(int(f) < 0 for f in seg.get('frets', [-1])):
                    continue
                seg_played = sorted(strings(seg), key=lambda x: -x[0])
                for rank_idx, item in enumerate(seg_played):
                    vel_roll = 80 if rank_idx == 0 else max(66, 76 - rank_idx)
                    insert(
                        bar, seg['beat'], seg['duration'], item, vel_roll, '',
                        replace=False, direction='↓', roll_step_sec=roll_sec,
                    )
            continue

        bar_dyn = bar.get('dynamic_scale')
        use_gentle = bar_dyn is not None and bar_dyn < 0.78 and style in _GENTLE_STYLES
        state = 0 if use_gentle else (bar_states[index] if index < len(bar_states) else 1)
        phrase_vel_scale = _PHRASE_VEL_CURVE[bar_index % 4]
        dyn_scale = bar.get('dynamic_scale')
        is_cadence_bar = (bar_index + 1) % 4 == 0
        is_fingerpick = style in _FINGERPICK_STYLES
        is_4_4 = abs(measure - 4.0) < 1e-9
        segments = bar['split_segments']
        segment_beats = {s['beat'] for s in segments}
        grid = sorted(set([i*0.5 for i in range(math.ceil(measure*2)) if i*0.5 < measure]
                          +[s['beat'] for s in segments]))

        is_fingerpick_or_neosoul = style in _FINGERPICK_STYLES or style == _NEOSOUL_STYLE
        ov_notes = _overlapping_notes(project.notes, index * measure, (index + 1) * measure) if not silent else []
        eff_groove_rho = (
            0.0
            if (style in _ARPEGGIO_STYLES and ov_notes and rho_b < 0.72 and not cm)
            else rho_b
        )
        groove = _compute_metrical_groove_profile(
            measure, style, state, fast_tempo, cm, project.meter,
            rho=eff_groove_rho, tau_slot=b_info['tau_slot'],
            exempt_pickup=b_info['exempt_pickup'], n_segments=len(segments),
        )
        active_slots = groove['active_slots']
        slot_vels = groove['velocities']
        slot_dirs = groove['directions']
        last_slot = groove['n_slots'] - 1

        # Place Chord-Melody (CM) lead first so accompaniment respects string/register reservations.
        cm_reservations: list[tuple[int, float, float, int, int]] = []
        cm_onset_slots: set[int] = set()
        ds_onset_slots: set[int] = set()
        if cm:
            lead = list(instrumental_melody[index]) if explicit_cm else list(motif_cm_map.get(index, []))
            for note in lead:
                beat_cm = note['beat'] if 'beat' in note else note['beat_offset']
                duration_cm, pitch_cm = note['duration'], note['pitch']
                tech_cm = str(note.get('technique') or 'CM')
                _integer(pitch_cm, 0, 127, 'melody pitch')
                if (
                    not math.isfinite(beat_cm)
                    or not math.isfinite(duration_cm)
                    or beat_cm < 0
                    or beat_cm >= measure
                    or duration_cm <= 0
                    or beat_cm + duration_cm > measure + 1e-9
                ):
                    raise ValueError(f'Invalid instrumental melody timing in bar {index}')
                seg_cm = active(bar, beat_cm)
                if seg_cm is None:
                    raise ValueError('Instrumental melody falls in an accompaniment rest')
                if seg_cm['chord'] == 'N.C.' or all(int(f) < 0 for f in seg_cm.get('frets', [-1])):
                    continue
                pref_s = note.get('string')
                min_i = 2 if tech_cm == 'DS' else 3
                choices = [
                    (6 - i, pitch_cm - t - capo, pitch_cm)
                    for i, t in enumerate(STANDARD_TUNING)
                    if i >= min_i and 0 <= pitch_cm - t - capo <= 24 - capo
                ]
                if not choices:
                    if tech_cm == 'DS':
                        continue
                    raise ValueError(f'Unplayable treble melody pitch {pitch_cm}')
                if pref_s is not None and any(c[0] == int(pref_s) for c in choices):
                    item_cm = next(c for c in choices if c[0] == int(pref_s))
                else:
                    item_cm = min(choices, key=lambda x: (abs(x[1] - seg_cm['base_fret']), x[1], x[0]))
                vel_cm = int(note.get('velocity', 88))
                eff_dur_cm = min(float(duration_cm), measure - float(beat_cm))
                raw_sl_from = note.get('slide_from_fret')
                sl_from = None
                if (
                    raw_sl_from is not None
                    and (pref_s is None or item_cm[0] == int(pref_s))
                    and 0 <= int(raw_sl_from) <= 24 - capo
                    and int(raw_sl_from) != int(item_cm[1])
                ):
                    sl_from = int(raw_sl_from)
                insert(bar, float(beat_cm), eff_dur_cm, item_cm, vel_cm, tech_cm, True, slide_from_fret=sl_from)
                cm_reservations.append((item_cm[0], float(beat_cm), float(beat_cm) + eff_dur_cm, int(pitch_cm), int(item_cm[1])))
                sl_idx = min(max(0, int(round(float(beat_cm) * 2))), last_slot)
                cm_onset_slots.add(sl_idx)
                if tech_cm == 'DS':
                    ds_onset_slots.add(sl_idx)

        def is_cm_reserved(s_num: int, b_pos: float) -> bool:
            return any(rs == s_num and rb0 - 1e-6 <= b_pos < rb1 - 1e-6 for rs, rb0, rb1, _rp, _rf in cm_reservations)

        def active_cm_at(b_pos: float) -> list[tuple[int, float, float, int, int]]:
            return [item_r for item_r in cm_reservations if item_r[1] - 1e-6 <= b_pos < item_r[2] - 1e-6]

        def cm_ceiling_at(b_pos: float) -> int:
            act_r = active_cm_at(b_pos)
            if act_r:
                return min(it[3] for it in act_r)
            near_r = [it[3] for it in cm_reservations if it[1] - 0.26 <= b_pos < it[2] + 0.26]
            if near_r:
                return min(near_r)
            return 999

        def is_playable_cm_inner(
            bass_item: tuple[int, int, int] | None,
            act_cm_list: list[tuple[int, float, float, int, int]],
            cand_item: tuple[int, int, int],
        ) -> bool:
            s_c, f_c, _ = cand_item
            if f_c < 0 or f_c > 24 - capo or not (1 <= s_c <= 6):
                return False
            frets6 = [-1] * 6
            if bass_item is not None and 1 <= bass_item[0] <= 6 and bass_item[1] >= 0:
                if bass_item[0] == s_c:
                    return False
                frets6[6 - bass_item[0]] = int(bass_item[1])
            for rs, _rb0, _rb1, _rp, rf in act_cm_list:
                if rs == s_c or not (1 <= rs <= 6) or rf < 0:
                    return False
                frets6[6 - rs] = int(rf)
            frets6[6 - s_c] = int(f_c)
            pos = [f for f in frets6 if f > 0]
            if pos and (max(pos) - min(pos)) > 4:
                return False
            f_cnt, _ = _fingers(frets6)
            if f_cnt > 4:
                return False
            _, overflow, strain = _assign_fingers(tuple(frets6), capo)
            return overflow == 0 and math.isfinite(strain)

        # Plan entries are executed in order after the bar's treble DAG is solved:
        # ('note', beat, duration, item, velocity, technique, direction, roll_step) or
        # ('treble', beat, request_index, velocity, kind, direction)
        plan, treble_requests, bass_pitch = [], [], None
        is_1bar_intro = (
            index == 0
            and not explicit_cm
            and index in roll_slot0_bars
            and 1 not in motif_cm_map
            and vocal_counts.get(0, 0) == 0
            and (len(bars) == 1 or vocal_counts.get(1, 0) > 0 or 1 in explicit_cm_bars)
        )
        is_comp_bar = str(project.meter).strip().endswith('/8') and int(round(measure * 2.0)) % 3 == 0
        first_cm_beat = min((it[1] for it in cm_reservations), default=0.0)
        cm_onset_beats = sorted({it[1] for it in cm_reservations})
        min_cm_bar_vel = min(
            (int(e['velocity']) for e in events.values() if e['bar'] == index and e.get('technique') == 'CM'),
            default=88,
        )
        last_cm_acc_treble_pitch = None

        def request_treble(beat, candidates, bass_pitch, velocity, kind, direction=None):
            t_str = _default_target_string(beat, measure, style=style, kind=kind)
            treble_requests.append(dict(
                beat=beat,
                candidates=list(candidates),
                bass_pitch=bass_pitch,
                target_string=t_str,
            ))
            plan.append(('treble', beat, len(treble_requests) - 1, velocity, kind, direction))

        for beat in grid:
            segment = active(bar,beat)
            if segment is None or segment.get('chord') == 'N.C.':
                continue
            played = strings(segment)
            if not played:
                continue
            bass = bass_of(segment)
            treble = sorted((x for x in played if x[0] <= 3 and x != bass),key=lambda x:x[2]) or sorted((x for x in played if x[0] <= 3),key=lambda x:x[2])
            mid_treble = sorted((x for x in played if x[0] <= 4 and x != bass),key=lambda x:x[2]) or treble
            arpeggio_treble = treble if len(treble) >= 3 else mid_treble
            slot = min(int(round(beat*2)), last_slot)
            slot_dir = slot_dirs[slot]
            is_segment_boundary = any(abs(beat - sb) < 1e-9 for sb in segment_beats)
            is_pm_beat = palm_mute and (rho_b < 0.5) and any(abs(beat - b) < 1e-9 for b in (0.0, measure / 2))

            slot_in_active = slot in active_slots or is_segment_boundary
            if is_pm_beat and not slot_in_active:
                slot_in_active = True
            if not slot_in_active:
                continue

            # Rolled chord on beat 0 for silent/section-ending breathing bars (rho >= 0.55) or rolled CM intros/interludes.
            if slot == 0 and abs(beat) < 1e-9 and (
                (rho_b >= 0.55 and not cm and (style not in _ARPEGGIO_STYLES or not ov_notes))
                or (cm and index in roll_slot0_bars)
            ):
                roll_dur = segment['duration']
                roll_items = sorted(played, key=lambda x: -x[0])
                if cm:
                    ceil_p = cm_ceiling_at(beat)
                    roll_items = [x for x in roll_items if not is_cm_reserved(x[0], beat) and x[2] < ceil_p]
                    if not roll_items and bass is not None and not is_cm_reserved(bass[0], beat) and bass[2] < ceil_p:
                        roll_items = [bass]
                    rolled_tr = [x for x in roll_items if x[0] <= 3]
                    if rolled_tr:
                        last_cm_acc_treble_pitch = max(rolled_tr, key=lambda x: x[0])[2]
                if bass is not None and (not cm or bass[2] < cm_ceiling_at(beat)):
                    bass_pitch = bass[2]
                for r_idx, item in enumerate(roll_items):
                    item_roll_dur = roll_dur
                    if cm and not is_1bar_intro and item != bass and item[0] >= 4 and item[1] > 0:
                        for rb0 in cm_onset_beats:
                            if rb0 > 1e-6 and not is_playable_cm_inner(bass, active_cm_at(rb0), item):
                                item_roll_dur = min(item_roll_dur, rb0)
                                break
                    v_base = 84 if r_idx == 0 else max(68, 78 - r_idx)
                    vel_r = max(1, min(127, round(v_base * phrase_vel_scale)))
                    plan.append(('note', beat, item_roll_dur, item, vel_r, '', '↓', roll_sec))
                continue

            if cm:
                if is_1bar_intro:
                    continue
                ceil_p = cm_ceiling_at(beat)
                dyn_div = max(1.0, float(dyn_scale)) if dyn_scale is not None else 1.0
                is_cm_bass_slot = (
                    (slot == 0)
                    or is_segment_boundary
                    or (
                        slot == groove['bass_pivot']
                        and slot not in ds_onset_slots
                        and (not is_comp_bar or slot not in cm_onset_slots)
                    )
                )
                emitted_cm_bass = False
                if is_cm_bass_slot and bass is not None and not is_cm_reserved(bass[0], beat) and bass[2] < ceil_p:
                    bass_pitch = bass[2]
                    technique = 'PM' if palm_mute and any(abs(beat - b) < 1e-9 for b in (0, measure / 2)) else ''
                    bass_vel = 74 if slot == 0 else 66
                    bass_cap = max(50, int((min_cm_bar_vel - 8) / dyn_div)) if bass[0] <= 4 else 127
                    vel_b = max(1, min(127, bass_cap, round(bass_vel * phrase_vel_scale)))
                    bass_dur = max(step_dur, 1.5 if is_comp_bar else 1.0)
                    plan.append(('note', beat, bass_dur, bass, vel_b, technique, '↓', None))
                    emitted_cm_bass = True

                allow_cm_inner_slot = (
                    slot > 0
                    and slot not in cm_onset_slots
                    and slot not in ds_onset_slots
                    and not emitted_cm_bass
                    and (not fast_tempo or slot % 2 == 0)
                )
                if allow_cm_inner_slot:
                    act_cm = active_cm_at(beat)
                    avail_inner = [
                        c for c in mid_treble
                        if not is_cm_reserved(c[0], beat)
                        and c[2] <= ceil_p - 3
                        and (bass_pitch is None or c[2] >= bass_pitch + 3)
                        and is_playable_cm_inner(bass, act_cm, c)
                    ]
                    if not avail_inner and beat < first_cm_beat - 1e-6:
                        avail_inner = [
                            c for c in arpeggio_treble
                            if not is_cm_reserved(c[0], beat)
                            and c[2] < ceil_p
                            and is_playable_cm_inner(bass, act_cm, c)
                        ]
                    if avail_inner:
                        vel_cap = max(48, int((min_cm_bar_vel - 10) / dyn_div))
                        vel_in = max(48, min(60, vel_cap, round(slot_vels[slot] * 0.66 * phrase_vel_scale)))
                        request_treble(beat, avail_inner, bass_pitch, vel_in, 'cm_inner', slot_dir)
                continue

            is_bass_slot = (slot == 0) or is_segment_boundary or is_pm_beat or (
                slot == groove['bass_pivot'] and style in {'指弹律动', _NEOSOUL_STYLE}
            )
            use_alt_bass = False
            alt_bass_item = None
            if (is_fingerpick and is_bass_slot and is_4_4 and len(segments) == 1
                    and (style == '指弹律动' or is_cadence_bar or bar_index % 2 == 0)
                    and abs(beat - 2.0) < 1e-9 and bass is not None):
                bass_s = bass[0]
                for candidate in sorted(played, key=lambda x: -x[0]):
                    if candidate[0] in (4, 5) and candidate[1] >= 0 and candidate[0] != bass_s:
                        alt_bass_item = candidate
                        use_alt_bass = True
                        break
            if is_segment_boundary or bass_pitch is None:
                bass_pitch = bass[2] if bass is not None else None

            if style == _STRUM_STYLE:
                if slot in groove['full_chord_slots'] or is_pm_beat:
                    selection = list(played)
                elif slot in groove['backbeat_slots']:
                    selection = list(mid_treble) or list(played)
                else:
                    selection = list(treble) or list(mid_treble)
                if use_alt_bass and alt_bass_item is not None and bass in selection:
                    selection = [alt_bass_item if x == bass else x for x in selection]
                is_strum = len(selection) >= 3
                if is_segment_boundary and bass is not None and bass not in selection and not use_alt_bass:
                    selection.append(bass)
                for item in selection:
                    is_bass_note = bass is not None and (item == bass or (use_alt_bass and item == alt_bass_item))
                    technique = 'PM' if palm_mute and is_bass_note and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                    velocity = (90 if slot == 0 else slot_vels[slot]) if is_strum else (90 if (is_bass_note and slot == 0) else 84 if is_bass_note else slot_vels[slot])
                    velocity = max(1, min(127, round(velocity * phrase_vel_scale)))
                    plan.append(('note', beat, step_dur, item, velocity, technique, slot_dir, None))

            elif is_fingerpick_or_neosoul:
                if is_bass_slot:
                    bass_item = alt_bass_item if (use_alt_bass and alt_bass_item is not None) else bass
                    if bass_item is not None:
                        bass_pitch = bass_item[2]
                        technique = 'PM' if palm_mute and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                        bass_vel = 90 if slot == 0 else 84
                        velocity = max(1, min(127, round(bass_vel * phrase_vel_scale)))
                        plan.append(('note', beat, step_dur, bass_item, velocity, technique, slot_dir, None))

                    if slot == 0 and style in {'指弹律动', _NEOSOUL_STYLE} and state >= 1 and arpeggio_treble:
                        vel_pinch = max(1, min(127, round(slot_vels[slot] * 0.94 * phrase_vel_scale)))
                        request_treble(beat, arpeggio_treble, bass_pitch, vel_pinch, 'pinch', slot_dir)
                else:
                    if style == _NEOSOUL_STYLE and slot in groove['backbeat_slots'] and len(mid_treble) >= 2:
                        vel_primary = max(1, min(127, round(slot_vels[slot] * phrase_vel_scale)))
                        request_treble(beat, mid_treble, bass_pitch, vel_primary, ('dyad', tuple(mid_treble)), slot_dir)
                    elif arpeggio_treble:
                        vel_t = max(1, min(127, round(slot_vels[slot] * phrase_vel_scale)))
                        request_treble(beat, arpeggio_treble, bass_pitch, vel_t, 'single', slot_dir)

            else:
                selection = list(played)
                if use_alt_bass and alt_bass_item is not None and bass in selection:
                    selection = [alt_bass_item if x == bass else x for x in selection]
                is_strum = len(selection) >= 3
                if is_segment_boundary and bass is not None and bass not in selection and not use_alt_bass:
                    selection.append(bass)
                for item in selection:
                    is_bass_note = bass is not None and (item == bass or (use_alt_bass and item == alt_bass_item))
                    technique = 'PM' if palm_mute and is_bass_note and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                    velocity = (90 if slot == 0 else slot_vels[slot]) if is_strum else (90 if (is_bass_note and slot == 0) else 84 if is_bass_note else slot_vels[slot])
                    velocity = max(1, min(127, round(velocity * phrase_vel_scale)))
                    plan.append(('note', beat, step_dur, item, velocity, technique, slot_dir, None))

        treble_line = _solve_bar_treble_dag(treble_requests, ov_notes, index, measure,
                                            prev_treble_pitch, prev_treble_string, style=style)
        for entry in plan:
            if entry[0] == 'note':
                _, beat, duration, item, velocity, technique, dir_tag, r_step = entry
                insert(bar, beat, duration, item, velocity, technique, direction=dir_tag, roll_step_sec=r_step)
                continue
            _, beat, req_index, velocity, kind, dir_tag = entry
            sel_treble = treble_line[req_index]
            if sel_treble is None:
                continue
            dur_treble = step_dur
            if kind == 'cm_inner':
                if (
                    sel_treble[0] <= 3
                    and last_cm_acc_treble_pitch is not None
                    and sel_treble[2] == last_cm_acc_treble_pitch
                ):
                    req_cands = treble_requests[req_index]['candidates']
                    t_str_req = float(treble_requests[req_index].get('target_string', 2.0))
                    alt_cands = [
                        c for c in req_cands
                        if c[0] > 3 or c[2] != last_cm_acc_treble_pitch
                    ]
                    if alt_cands:
                        sel_treble = min(alt_cands, key=lambda c: (abs(float(c[0]) - t_str_req), abs(c[2] - sel_treble[2])))
                    else:
                        continue
                next_same_s = min(
                    (it[1] for it in cm_reservations if it[0] == sel_treble[0] and it[1] > beat + 1e-6),
                    default=None,
                )
                if next_same_s is not None:
                    dur_treble = min(dur_treble, next_same_s - beat)
                if sel_treble[0] >= 4:
                    next_any_cm = min((b_on for b_on in cm_onset_beats if b_on > beat + 1e-6), default=None)
                    if next_any_cm is not None:
                        dur_treble = min(dur_treble, next_any_cm - beat)
                if dur_treble <= 1e-6:
                    continue
                if sel_treble[0] <= 3:
                    last_cm_acc_treble_pitch = sel_treble[2]
            insert(bar, beat, dur_treble, sel_treble, velocity, direction=dir_tag)
            if isinstance(kind, tuple) and kind[0] == 'dyad':
                sel_primary = sel_treble
                adjacent = [c for c in kind[1] if c[0] != sel_primary[0] and abs(c[0] - sel_primary[0]) == 1]
                if adjacent:
                    dyad_note = min(adjacent, key=lambda c: abs(c[2] - sel_primary[2]))
                    insert(bar, beat, step_dur, dyad_note, max(1, min(127, round(velocity * 0.88))), direction=dir_tag)
            prev_treble_pitch = sel_treble[2]
            prev_treble_string = sel_treble[0]

        # --- Ornament, passing-bass, and slap insertion (suppressed on breathing bars and CM lead bars) ---
        if rho_b < 0.5 and not cm:
            tail = measure - 0.5
            segment = active(bar, tail) if tail >= 0 else None
            ornament_candidates = []
            tail_done = False
            tail_abs = index * measure + tail
            tail_vocal_clear = not any(abs(n.start - tail_abs) < 0.30 for n in ov_notes)
            is_pre_breath = (index + 1 < len(bars) and bar_breathing[index + 1]['rho'] >= 0.5)
            is_arp_style = style in _ARPEGGIO_STYLES
            orn_ready = (index - last_orn_bar) >= (2 if is_arp_style else 1)
            min_hp_gap = 3 if is_arp_style else 2
            allow_hp_here = (
                orn_ready
                and tail_vocal_clear
                and (index - last_hp_bar) >= min_hp_gap
                and (not is_arp_style or is_pre_breath or (index + 1) % 2 == 0 or last_hp_bar < 0)
            )
            allow_pb_here = orn_ready and tail_vocal_clear and (index - last_pb_bar) >= 2
            tail_slot_idx = min(max(0, int(round(tail * 2))), last_slot)
            hp_vel = max(50, min(82, round(slot_vels[tail_slot_idx] * 0.96 * phrase_vel_scale)))
            pb_vel = max(56, min(84, round(72 * phrase_vel_scale)))

            if embellish_hp and allow_hp_here and segment is not None and segment['chord'] != 'N.C.':
                parsed = parse_extended_chord(segment['chord'])
                tones = set(parsed['pcs']) | {parsed['bass_pc']} if parsed else set()
                for string in (1, 2, 3):
                    if any(e['bar'] == index and e['string'] == string and e['technique'] == 'CM' and e['beat'] <= tail < e['beat'] + e['duration'] - 1e-9 for e in events.values()):
                        continue
                    previous = [e for e in events.values() if e['bar'] == index and e['string'] == string and e['beat'] < tail]
                    if not previous:
                        continue
                    previous_event = max(previous, key=lambda e: e['beat'])
                    for fret in range(max(0, previous_event['fret'] - 2), min(24 - capo, previous_event['fret'] + 2) + 1):
                        pitch = STANDARD_TUNING[6 - string] + capo + fret
                        if fret != previous_event['fret'] and pitch % 12 in tones:
                            ornament_candidates.append((abs(fret - previous_event['fret']), string, fret, pitch, previous_event['fret']))
                if ornament_candidates:
                    _, string, fret, pitch, old = min(ornament_candidates)
                    existing = events.get((index * measure + tail, string))
                    if existing is None or existing['technique'] not in {'CM', 'H', 'P', 'PB'}:
                        clear_beat(index, tail)
                        insert(bar, tail, step_dur, (string, fret, pitch), hp_vel, 'H' if fret > old else 'P', True)
                        tail_done = True
                        last_orn_bar = index
                        last_hp_bar = index
            if not tail_done and passing_bass and allow_pb_here and segment is not None and segment['chord'] != 'N.C.' and index + 1 < len(bars):
                next_segment = active(bars[index + 1], 0)
                target = bass_of(next_segment) if next_segment is not None else None
                current = bass_of(segment)
                candidates = []
                if target is not None and current is not None and current[2] != target[2] and (current[2] % 12) != (target[2] % 12):
                    lo_b, hi_b = min(current[2], target[2]), max(current[2], target[2])
                    for string in (4, 5, 6):
                        if (index * measure + tail, string) in events:
                            continue
                        for fret in range(25 - capo):
                            pitch = STANDARD_TUNING[6 - string] + capo + fret
                            if pitch in (current[2], target[2]):
                                continue
                            step_to_tgt = abs(pitch - target[2])
                            if step_to_tgt not in (1, 2):
                                continue
                            in_scale = (pitch % 12) in diatonic
                            is_chrom_half = (step_to_tgt == 1 and lo_b < pitch < hi_b and (hi_b - lo_b) == 2)
                            if not (in_scale or is_chrom_half):
                                continue
                            if abs(pitch - current[2]) > 5 and not (lo_b < pitch < hi_b):
                                continue
                            candidates.append((
                                0 if (lo_b < pitch < hi_b) else 1,
                                abs(pitch - current[2]),
                                fret,
                                string,
                                pitch,
                            ))
                if candidates:
                    _, _, fret, string, pitch = min(candidates)
                    clear_beat(index, tail)
                    insert(bar, tail, step_dur, (string, fret, pitch), pb_vel, 'PB')
                    tail_done = True
                    last_orn_bar = index
                    last_pb_bar = index
            vnc = vocal_counts.get(index, 0)
            if (
                not tail_done
                and orn_ready
                and measure >= 3.5
                and vnc <= 1
                and not cm
                and not is_arp_style
                and (passing_bass or embellish_hp)
                and segment is not None
            ):
                fill_beat = measure - 1.0
                fill_abs = index * measure + fill_beat
                fill_vocal_clear = not any(abs(n.start - fill_abs) < 0.30 for n in ov_notes)
                fill_seg = active(bar, fill_beat) if (fill_beat >= 0 and fill_vocal_clear) else None
                if fill_seg is not None and fill_seg['chord'] != 'N.C.':
                    fill_done = False
                    tail_pitches = {e['pitch'] for e in events.values() if e['bar'] == index and abs(e['beat'] - (measure - 0.5)) < 1e-6}
                    if embellish_hp and (index - last_hp_bar) >= min_hp_gap:
                        parsed_fill = parse_extended_chord(fill_seg['chord'])
                        tones_fill = set(parsed_fill['pcs']) | {parsed_fill['bass_pc']} if parsed_fill else set()
                        for fs in (1, 2, 3):
                            if any(e['bar'] == index and e['string'] == fs and e['technique'] == 'CM' and e['beat'] <= fill_beat < e['beat'] + e['duration'] - 1e-9 for e in events.values()):
                                continue
                            prev_evts = [e for e in events.values() if e['bar'] == index and e['string'] == fs and e['beat'] < fill_beat]
                            if not prev_evts:
                                continue
                            prev_e = max(prev_evts, key=lambda e: e['beat'])
                            chosen_orn = None
                            for ff in range(max(0, prev_e['fret'] - 2), min(24 - capo, prev_e['fret'] + 2) + 1):
                                pp = STANDARD_TUNING[6 - fs] + capo + ff
                                if ff != prev_e['fret'] and pp not in tail_pitches and pp % 12 in tones_fill:
                                    dist = abs(ff - prev_e['fret'])
                                    if chosen_orn is None or dist < chosen_orn[0]:
                                        chosen_orn = (dist, fs, ff, pp, prev_e['fret'])
                            if chosen_orn is not None:
                                _, fs2, ff2, pp2, old2 = chosen_orn
                                key_check = (index * measure + fill_beat, fs2)
                                if key_check not in events or events[key_check].get('technique') not in {'CM', 'H', 'P', 'PB'}:
                                    clear_beat(index, fill_beat)
                                    insert(bar, fill_beat, step_dur, (fs2, ff2, pp2), hp_vel, 'H' if ff2 > old2 else 'P', True)
                                    fill_done = True
                                    last_orn_bar = index
                                    last_hp_bar = index
                                    break
                    if not fill_done and passing_bass and (index - last_pb_bar) >= 2 and index + 1 < len(bars):
                        next_seg_fill = active(bars[index + 1], 0)
                        target_fill = bass_of(next_seg_fill) if next_seg_fill is not None else None
                        current_fill = bass_of(fill_seg)
                        cands_fill = []
                        if target_fill is not None and current_fill is not None and current_fill[2] != target_fill[2] and (current_fill[2] % 12) != (target_fill[2] % 12):
                            lo_bf, hi_bf = min(current_fill[2], target_fill[2]), max(current_fill[2], target_fill[2])
                            for fs in (4, 5, 6):
                                if (index * measure + fill_beat, fs) in events:
                                    continue
                                for ff in range(25 - capo):
                                    pp = STANDARD_TUNING[6 - fs] + capo + ff
                                    if pp in tail_pitches or pp in (current_fill[2], target_fill[2]):
                                        continue
                                    step_to_tgt_f = abs(pp - target_fill[2])
                                    if step_to_tgt_f not in (1, 2):
                                        continue
                                    in_scale_f = (pp % 12) in diatonic
                                    is_chrom_half_f = (step_to_tgt_f == 1 and lo_bf < pp < hi_bf and (hi_bf - lo_bf) == 2)
                                    if not (in_scale_f or is_chrom_half_f):
                                        continue
                                    if abs(pp - current_fill[2]) > 5 and not (lo_bf < pp < hi_bf):
                                        continue
                                    cands_fill.append((
                                        0 if (lo_bf < pp < hi_bf) else 1,
                                        abs(pp - current_fill[2]),
                                        ff,
                                        fs,
                                        pp,
                                    ))
                        if cands_fill:
                            _, _, ff3, fs3, pp3 = min(cands_fill)
                            clear_beat(index, fill_beat)
                            insert(bar, fill_beat, step_dur, (fs3, ff3, pp3), pb_vel, 'PB')
                            last_orn_bar = index
                            last_pb_bar = index

            if slap_x and bar['split_segments']:
                for beat in ([1.5 * k for k in range(1, int(measure // 1.5) + 1) if 1.5 * k < measure] if (str(project.meter).strip().endswith('/8') and int(round(measure * 2)) % 3 == 0) else (1.0, 3.0)):
                    segment = active(bar,beat) if beat < measure else None
                    if segment is not None and segment['chord'] != 'N.C.':
                        slap_vel = max(1, min(127, round(82 * phrase_vel_scale * (float(dyn_scale) if dyn_scale is not None else 1.0))))
                        slaps.append(dict(bar=index,beat=beat,start=index*measure+beat,duration=min(0.5,measure-beat),
                                          string=6,fret=0,pitch=40+capo,velocity=slap_vel,technique='X',chord=segment['chord'],
                                          segment=bar['split_segments'].index(segment)))
    ordered = sorted(events.values(),key=lambda e:(e['start'],e['string'],e['technique']))
    cm_bar_set = {e['bar'] for e in ordered if e.get('technique') in ('CM', 'DS', 'SL')}
    for event in ordered:
        b_idx = event['bar']
        b_rho = bar_breathing[b_idx]['rho'] if b_idx < len(bar_breathing) else 0.0
        vnc = vocal_counts.get(b_idx, 0)
        if event.get('technique') in ('CM', 'DS', 'SL'):
            comp_scale = 1.0
        elif b_rho >= 0.5:
            comp_scale = 0.96
        elif vnc >= 5:
            comp_scale = 0.94
        elif vnc >= 3:
            comp_scale = 0.97
        elif vnc <= 1:
            comp_scale = 1.03 if b_idx not in cm_bar_set else 0.96
        else:
            comp_scale = 1.0
        event['velocity'] = max(46, min(122, round(event['velocity'] * comp_scale)))
        is_beat1_bass = abs(event['beat']) < 1e-9 and event.get('string', 0) >= 4
        min_b1_vel = 70 if (
            (b_idx < len(bar_breathing) and bar_breathing[b_idx]['is_final_roll'])
            or b_idx in cm_bar_set
        ) else 76
        if is_beat1_bass and event['velocity'] < min_b1_vel:
            event['velocity'] = min_b1_vel
    nc_starts = sorted({
        float(b['bar'] * measure + seg['beat'])
        for b in bars
        for seg in b.get('split_segments', [])
        if seg.get('chord') == 'N.C.' or all(int(f) < 0 for f in seg.get('frets', [-1]))
    } | ({
        float(b['bar'] * measure)
        for b in bars
        if intro_mode == 'nc' and not _overlapping_notes(project.notes, b['bar'] * measure, (b['bar'] + 1) * measure)
    }))
    previous_by_string = {}
    for event in ordered:
        previous = previous_by_string.get(event['string'])
        if event['technique'] in ('H', 'P'):
            event['technique'] = ('H' if event['fret'] > previous['fret'] else 'P') if (previous and previous['bar'] == event['bar'] and event['fret'] != previous['fret']) else ''
        if previous is not None:
            previous['duration'] = min(previous['duration'],event['start']-previous['start'])
        next_nc = next((ns for ns in nc_starts if ns >= event['start'] + 1e-9), None)
        if next_nc is not None:
            event['duration'] = min(event['duration'], max(1e-3, next_nc - event['start']))
            event['nc_bound'] = next_nc
        previous_by_string[event['string']] = event
    result = dict(title=project.title,tonic=project.tonic,meter=project.meter,bpm=project.bpm,
                  beat_origin=project.beat_origin,style=style,capo=capo,bar_count=count,measure_beats=measure,
                  chords=[b['chord'] for b in bars],bars=bars,
                  events=sorted(ordered+slaps,key=lambda e:(e['start'],e['string'],e['technique'])))
    if getattr(project, 'bar_start_seconds', None) is not None:
        result['bar_start_seconds'] = list(project.bar_start_seconds)
    if getattr(project, 'beat_times', None) is not None:
        result['beat_times'] = list(project.beat_times)
    return result
