"""Guitar accompaniment and fingerstyle chord-melody arrangement engine.

Supports split-bar chords, X slap, P.M. palm mute, H/P licks, passing bass and
intro/interlude chord-melody.  Three decision layers are used:

* ``_decode_viterbi_texture_states`` - an empirical, quantile-regularised
  4-state Gaussian HMM decoded with Viterbi DP that assigns each bar a texture
  density level.
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
from .chord_voicing import (STANDARD_TUNING, _integer, _make_voicing, generate_voicings,
                            _optimize_candidates, parse_extended_chord, _note_pc)
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
    """Per-bar observation: onset density, register, harmonic rhythm and dynamics."""
    bar_start = idx * measure
    bar_end = (idx + 1) * measure
    ov_notes = _overlapping_notes(project.notes, bar_start, bar_end)
    onset_notes = [n for n in ov_notes if bar_start - 1e-6 <= n.start < bar_end]

    density_norm = min(1.0, len(onset_notes) / max(1.0, measure))
    if ov_notes:
        mean_p = sum(n.pitch for n in ov_notes) / len(ov_notes)
        pitch_norm = max(0.0, min(1.0, (mean_p - song_low) / song_range))
    else:
        pitch_norm = 0.35
    n_segments = len(bar.get('split_segments') or ())
    harmonic_rhythm_norm = max(0.0, min(1.0, (n_segments - 1) / 2.0))
    raw_dyn = float(bar.get('dynamic_scale', 0.85))
    dyn_norm = max(0.0, min(1.0, (raw_dyn - 0.55) / 0.60))

    energy = (0.36 * density_norm + 0.30 * pitch_norm
              + 0.12 * harmonic_rhythm_norm + 0.22 * dyn_norm)
    if not ov_notes:
        energy *= 0.45
    if (idx + 1) % 4 == 0:
        energy *= 0.72
    return energy


def _texture_emission_model(energies):
    """Blend empirical energy quantiles with the anchor priors.

    The shrinkage weight ``n / (n + k)`` lets long songs place the four state
    centres on their own dynamic distribution while short excerpts stay close
    to the structural anchors.  Centres are kept strictly increasing.
    """
    n = len(energies)
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
    """Empirical quantile-regularised 4-state Gaussian HMM texture decoder.

    Each bar's observation is a weighted energy built from vocal onset
    density, normalised pitch register, harmonic rhythm (number of split
    segments) and the bar dynamic scale.  State emission centres are a
    shrinkage blend of the song's own energy quantiles and fixed structural
    anchors (see ``_texture_emission_model``).  Transition costs penalise
    texture jumps, relax at 4-bar phrase starts and favour thinning into
    cadence bars.  The single most likely state path is recovered with forward
    Viterbi DP and backpointer traceback.  Returns one state index (0 = sparse
    ... 3 = dense) per bar.
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
_STRUM_STYLE = '扫弦伴奏'
_NEOSOUL_STYLE = 'Neo-Soul 律动'

# Lerdahl-Jackendoff style weights per metrical depth:
# 0 downbeat, 1 secondary accent (half-bar / grouping accent), 2 beat, 3 eighth offbeat.
_DEPTH_WEIGHT = (1.0, 0.75, 0.5, 0.25)
_VEL_FLOOR = 44.0
_VEL_SPAN = 46.0


def _compute_metrical_groove_profile(measure, style, state, fast_tempo, cm):
    """Derive active eighth-note slots and slot velocities from a metrical salience field.

    Slots index the eighth-note grid of the bar (``slot = round(beat * 2)``).
    Every slot receives a metrical depth (downbeat 0, secondary accent 1,
    beat 2, offbeat 3).  The secondary accent is the half-bar in even meters;
    in odd meters it follows the grouping preference: early (beat 2) for slow
    arpeggios, late (last beat) for strumming and fast tempi.

    The salience of an offbeat adds a syncopation term: coupling ``a`` to the
    strength of the next stronger slot it anticipates (cross-barline
    anticipation scaled by ``gamma``) and coupling ``b`` to the slot it follows
    (negative at slow tempi = masking by a preceding accent, positive at fast
    tempi = rebound stroke).  For strumming the offbeat that rebounds off the
    secondary accent receives the upstroke weight instead.

    A bar's density budget ``k`` is derived from its tactus count ``B`` and
    the texture state; the activation threshold is the k-th order statistic
    of the salience field, so the k most salient slots are active.  Velocities
    are a concave map of the metrical weight plus a small syncopation accent.
    """
    n_slots = max(1, int(math.ceil(measure * 2 - 1e-9)))
    n_beats = max(1, int(math.ceil(measure - 1e-9)))
    is_strum = style == _STRUM_STYLE
    is_picked = style in _FINGERPICK_STYLES or style == _NEOSOUL_STYLE

    if n_beats >= 2 and n_beats % 2 == 0:
        secondary = n_beats
    elif n_beats >= 3:
        late_grouping = is_strum or fast_tempo
        secondary = 2 * (n_beats - 1) if late_grouping else 2
    else:
        secondary = None
    bass_pivot = n_beats if n_beats % 2 == 0 else 2 * (n_beats - 1)
    if n_beats < 2:
        bass_pivot = None

    depths = []
    for slot in range(n_slots):
        if slot == 0:
            depths.append(0)
        elif slot == secondary:
            depths.append(1)
        elif slot % 2 == 0:
            depths.append(2)
        else:
            depths.append(3)
    weights = [_DEPTH_WEIGHT[d] for d in depths]

    if fast_tempo:
        a_next, b_prev, gamma, beat_coupling = 0.20, 0.07, 0.70, 0.40
    else:
        a_next, b_prev, gamma, beat_coupling = 0.20, -0.10, 0.45, 0.0

    def anticipation(slot):
        for nxt in range(slot + 1, n_slots):
            if depths[nxt] < depths[slot]:
                return weights[nxt]
        return weights[0] * gamma

    salience = []
    for slot in range(n_slots):
        depth, w = depths[slot], weights[slot]
        if depth == 3:
            prev_w = weights[slot - 1]
            if is_strum:
                rebound = 0.20 * weights[slot - 1] if depths[slot - 1] == 1 else 0.0
                sal = w + rebound
            else:
                sal = w + a_next * anticipation(slot) + b_prev * prev_w
        elif depth == 2:
            sal = w + beat_coupling * (anticipation(slot) - 0.5) - 0.01 * (slot // 2)
        else:
            sal = w
        salience.append(sal)

    if is_strum:
        budget = n_beats if (fast_tempo or state <= 1) else n_beats + 1
    elif is_picked and not cm:
        half = n_beats // 2
        if fast_tempo:
            budget = (2, n_beats, n_beats, n_beats + 1)[state]
        else:
            budget = (max(2, n_beats - 1), n_beats + half - 1, n_beats + half, n_slots)[state]
    else:
        budget = n_slots
    budget = max(1, min(n_slots, budget))

    ranked = sorted(range(n_slots), key=lambda s: (-salience[s], s))
    threshold = salience[ranked[budget - 1]]
    active_slots = {s for s in ranked[:budget]}

    velocities = []
    for slot in range(n_slots):
        w = weights[slot]
        metrical = math.sqrt(max(0.0, (w - 0.25) / 0.75))
        syncopation = max(0.0, salience[slot] - w)
        velocities.append(int(round(_VEL_FLOOR + _VEL_SPAN * metrical + 12.0 * syncopation)))

    backbeats = {s for s in range(2, n_slots, 2) if s != bass_pivot}
    full_chord_slots = {0} | ({bass_pivot} if bass_pivot is not None else set())
    return dict(n_slots=n_slots, depths=depths, salience=salience, threshold=threshold,
                active_slots=active_slots, velocities=velocities, bass_pivot=bass_pivot,
                backbeat_slots=backbeats, full_chord_slots=full_chord_slots)


# ---------------------------------------------------------------------------
# Layered counterpoint DAG for the treble voice
# ---------------------------------------------------------------------------

# Harmonic cost of the treble-over-bass interval class (negative = preferred).
# Imperfect consonances are favoured, perfect fifths neutral, doubled
# octaves/unisons slightly hollow, fourths and dissonances discouraged.
_BASS_INTERVAL_COST = (0.6, 1.2, 1.0, -0.4, -0.4, 0.5, 1.0, 0.0, -0.4, -0.4, 0.8, 1.2)
_VL_SEMITONE_COST = 0.45
_REPEAT_STRING_COST = 4.5
_STRING_SKIP_COST = 0.8
_PARALLEL_PERFECT_COST = 2.5
_VOICE_CROSSING_COST = 3.0


def _vocal_context(ov_notes, abs_time, beat, measure):
    vocal_onset_here = any(abs(n.start - abs_time) < 0.45 for n in ov_notes)
    v_pitch = None
    for n in ov_notes:
        if n.start <= abs_time < n.start + n.duration or abs(n.start - abs_time) < 0.45:
            v_pitch = n.pitch
            break
    vocal_gap_here = (v_pitch is None) or (not vocal_onset_here and beat >= measure * 0.5)
    return v_pitch, vocal_onset_here, vocal_gap_here


def _treble_emission_cost(cand, context, bass_pitch):
    """Node cost: vocal masking avoidance, call-and-response contour, bass consonance."""
    string, _fret, pitch = cand
    v_pitch, vocal_onset_here, vocal_gap_here = context
    cost = 0.0
    if vocal_onset_here and v_pitch is not None:
        cost += 5.0 * max(0.0, (pitch - (v_pitch - 5)) / 3.0)
    if vocal_gap_here:
        cost += -3.5 * (4 - string)
    else:
        cost += 1.2 * (4 - string)
    if bass_pitch is not None:
        if pitch <= bass_pitch:
            cost += _VOICE_CROSSING_COST
        else:
            cost += _BASS_INTERVAL_COST[(pitch - bass_pitch) % 12]
    return cost


def _treble_transition_cost(prev, cur, prev_bass, cur_bass):
    """Edge cost: voice-leading distance, repeated-string refractory, string skips, parallels."""
    p_string, _pf, p_pitch = prev
    c_string, _cf, c_pitch = cur
    cost = _VL_SEMITONE_COST * abs(c_pitch - p_pitch)
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


def _solve_bar_treble_dag(slot_requests, ov_notes, bar_idx, measure, init_pitch=None, init_string=None):
    """Exact minimum-cost treble line over all treble requests of one bar.

    ``slot_requests`` is a chronologically ordered list of dicts with keys
    ``beat``, ``candidates`` (``(string, fret, pitch)`` tuples) and
    ``bass_pitch`` (sounding bass pitch or ``None``).  Layer ``t`` of the DAG
    holds the candidates of request ``t``; edges connect every node of layer
    ``t-1`` to every node of layer ``t``.  A virtual source carries the
    previous bar's last treble note (``init_pitch`` / ``init_string``) so the
    line stays continuous across bar lines.  ``dp[t][k]`` is the minimum cost
    of any path ending at candidate ``k`` of layer ``t``; ``back[t][k]``
    stores its predecessor.  Returns the chosen candidate per request
    (``None`` for a request without candidates).
    """
    layer_ids = [t for t, req in enumerate(slot_requests) if req.get('candidates')]
    result = [None] * len(slot_requests)
    if not layer_ids:
        return result

    layers = [list(slot_requests[t]['candidates']) for t in layer_ids]
    basses = [slot_requests[t].get('bass_pitch') for t in layer_ids]
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

    for k, cand in enumerate(layers[0]):
        cost = _treble_emission_cost(cand, contexts[0], basses[0])
        if source is not None:
            src_cost = _VL_SEMITONE_COST * abs(cand[2] - source[2])
            if init_string is not None and cand[0] == init_string:
                src_cost += _REPEAT_STRING_COST
            cost += src_cost
        dp[0][k] = cost

    for t in range(1, len(layers)):
        for k, cand in enumerate(layers[t]):
            emit = _treble_emission_cost(cand, contexts[t], basses[t])
            min_cost = INF
            prev_k = 0
            for j, prev_cand in enumerate(layers[t - 1]):
                cost = dp[t - 1][j] + _treble_transition_cost(prev_cand, cand, basses[t - 1], basses[t])
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
        chords = [project.tonic]*math.ceil(end/measure)
    else:
        chords = list(chords)
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
            symbol = segment['chord']
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
            dyn = max(0.55, min(1.15, float(bar_dynamics[bar['bar']])))
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
        candidates = [x for x in strings(segment) if x[0] >= 4]
        return min(candidates,key=lambda x:x[2]) if candidates else None

    def insert(bar, beat, duration, played, velocity, technique='', replace=False):
        segment = active(bar,beat)
        if segment is None:
            return
        string, fret, pitch = played
        if technique == 'CM':
            duration = min(duration, measure - beat)
        else:
            duration = min(duration, segment['beat'] + segment['duration'] - beat, measure - beat)
        if duration <= 0:
            return
        dyn_scale = bar.get('dynamic_scale')
        if dyn_scale is not None:
            velocity = max(1, min(127, round(velocity * dyn_scale)))
        start = bar['bar']*measure+beat
        event = dict(bar=bar['bar'],beat=beat,start=start,duration=duration,string=string,fret=fret,
                     pitch=pitch,velocity=velocity,technique=technique,chord=segment['chord'],
                     segment=bar['split_segments'].index(segment))
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
    pentatonic = {(tonic_pc+x)%12 for x in (0,2,4,7,9)}
    diatonic = {(tonic_pc+x)%12 for x in (0,2,4,5,7,9,11)}
    _PHRASE_VEL_CURVE = [0.97, 1.00, 1.04, 0.98]

    spb = 60.0 / project.bpm
    half_beat_sec = 0.5 * spb
    fast_tempo = half_beat_sec < 0.285
    step_dur = 1.0 if fast_tempo else 0.5

    vocal_counts = {}
    for bar in bars:
        idx = bar['bar']
        bar_start = idx * measure
        bar_end = (idx + 1) * measure
        vocal_counts[idx] = len(_overlapping_notes(project.notes, bar_start, bar_end))

    prev_treble_pitch = None
    prev_treble_string = None

    for bar in bars:
        index = bar['bar']
        bar_index = index
        silent = not _overlapping_notes(project.notes,index*measure,(index+1)*measure)
        if intro_mode == 'nc' and silent:
            continue
        cm = intro_mode == 'chord_melody' and silent
        bar_dyn = bar.get('dynamic_scale')
        use_gentle = bar_dyn is not None and bar_dyn < 0.68 and style in _GENTLE_STYLES
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
        groove = _compute_metrical_groove_profile(measure, style, state, fast_tempo, cm)
        active_slots = groove['active_slots']
        slot_vels = groove['velocities']
        last_slot = groove['n_slots'] - 1

        ov_notes = _overlapping_notes(project.notes, index * measure, (index + 1) * measure) if not silent else []

        # Plan entries are executed in order after the bar's treble DAG is solved:
        # ('note', beat, duration, item, velocity, technique) or
        # ('treble', beat, request_index, velocity, kind)
        plan = []
        treble_requests = []

        def request_treble(beat, candidates, bass_pitch, velocity, kind):
            treble_requests.append(dict(beat=beat, candidates=list(candidates), bass_pitch=bass_pitch))
            plan.append(('treble', beat, len(treble_requests) - 1, velocity, kind))

        for beat in grid:
            segment = active(bar,beat)
            if segment is None:
                continue
            played = strings(segment)
            bass = bass_of(segment)
            treble = sorted((x for x in played if x[0] <= 3),key=lambda x:x[2])
            mid_treble = sorted((x for x in played if x[0] <= 4),key=lambda x:x[2]) or treble
            slot = min(int(round(beat*2)), last_slot)
            is_segment_boundary = any(abs(beat - sb) < 1e-9 for sb in segment_beats)
            is_pm_beat = palm_mute and any(abs(beat - b) < 1e-9 for b in (0.0, measure / 2))

            slot_in_active = slot in active_slots or is_segment_boundary
            if is_pm_beat and not slot_in_active:
                slot_in_active = True
            if not slot_in_active:
                continue

            use_alt_bass = False
            alt_bass_item = None
            if (is_fingerpick and is_4_4 and len(segments) == 1
                    and (is_cadence_bar or bar_index % 2 == 0)
                    and abs(beat - 2.0) < 1e-9 and bass is not None):
                bass_s = bass[0]
                for candidate in sorted(played, key=lambda x: -x[0]):
                    if candidate[0] in (4, 5) and candidate[1] >= 0 and candidate[0] != bass_s:
                        alt_bass_item = candidate
                        use_alt_bass = True
                        break
            sounding_bass = alt_bass_item if use_alt_bass else bass
            bass_pitch = sounding_bass[2] if sounding_bass is not None else None

            if style == _STRUM_STYLE:
                if slot in groove['full_chord_slots']:
                    selection = list(played)
                elif slot in groove['backbeat_slots']:
                    selection = list(mid_treble)
                else:
                    selection = list(treble)
                if use_alt_bass and alt_bass_item is not None and bass in selection:
                    selection = [alt_bass_item if x == bass else x for x in selection]
                is_strum = len(selection) >= 3
                if is_segment_boundary and bass is not None and bass not in selection and not use_alt_bass:
                    selection.append(bass)
                for item in selection:
                    if cm and item[0] <= 3:
                        continue
                    is_bass_note = bass is not None and (item == bass or (use_alt_bass and item == alt_bass_item))
                    technique = 'PM' if palm_mute and is_bass_note and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                    velocity = (92 if slot == 0 else slot_vels[slot]) if is_strum else (92 if (is_bass_note and slot == 0) else 86 if is_bass_note else slot_vels[slot])
                    velocity = max(1, min(127, round(velocity * phrase_vel_scale)))
                    plan.append(('note', beat, step_dur, item, velocity, technique))

            elif is_fingerpick_or_neosoul:
                is_bass_slot = (slot == 0) or (slot == groove['bass_pivot'] and style in {'指弹律动', _NEOSOUL_STYLE}
                                               and (is_segment_boundary or state >= 1))

                if is_bass_slot:
                    bass_item = alt_bass_item if (use_alt_bass and alt_bass_item is not None) else bass
                    if bass_item is not None:
                        technique = 'PM' if palm_mute and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                        velocity = max(1, min(127, round(92 * phrase_vel_scale)))
                        plan.append(('note', beat, step_dur, bass_item, velocity, technique))

                    if slot == 0 and style in {'指弹律动', _NEOSOUL_STYLE} and state >= 2 and mid_treble:
                        vel_pinch = max(1, min(127, round(slot_vels[slot] * phrase_vel_scale)))
                        request_treble(beat, mid_treble, bass_pitch, vel_pinch, 'pinch')
                else:
                    if cm:
                        continue

                    if style == _NEOSOUL_STYLE and slot in groove['backbeat_slots'] and len(mid_treble) >= 2:
                        vel_primary = max(1, min(127, round(slot_vels[slot] * phrase_vel_scale)))
                        request_treble(beat, mid_treble, bass_pitch, vel_primary, ('dyad', tuple(mid_treble)))
                    elif mid_treble:
                        vel_t = max(1, min(127, round(slot_vels[slot] * phrase_vel_scale)))
                        request_treble(beat, mid_treble, bass_pitch, vel_t, 'single')

            else:
                selection = list(played)
                if use_alt_bass and alt_bass_item is not None and bass in selection:
                    selection = [alt_bass_item if x == bass else x for x in selection]
                is_strum = len(selection) >= 3
                if is_segment_boundary and bass is not None and bass not in selection and not use_alt_bass:
                    selection.append(bass)
                for item in selection:
                    if cm and item[0] <= 3:
                        continue
                    is_bass_note = bass is not None and (item == bass or (use_alt_bass and item == alt_bass_item))
                    technique = 'PM' if palm_mute and is_bass_note and any(abs(beat-b) < 1e-9 for b in (0, measure/2)) else ''
                    velocity = (92 if slot == 0 else slot_vels[slot]) if is_strum else (92 if (is_bass_note and slot == 0) else 86 if is_bass_note else slot_vels[slot])
                    velocity = max(1, min(127, round(velocity * phrase_vel_scale)))
                    plan.append(('note', beat, step_dur, item, velocity, technique))

        treble_line = _solve_bar_treble_dag(treble_requests, ov_notes, index, measure,
                                            prev_treble_pitch, prev_treble_string)
        for entry in plan:
            if entry[0] == 'note':
                _, beat, duration, item, velocity, technique = entry
                insert(bar, beat, duration, item, velocity, technique)
                continue
            _, beat, req_index, velocity, kind = entry
            sel_treble = treble_line[req_index]
            if sel_treble is None:
                continue
            insert(bar, beat, step_dur, sel_treble, velocity)
            if isinstance(kind, tuple) and kind[0] == 'dyad':
                sel_primary = sel_treble
                adjacent = [c for c in kind[1] if c[0] != sel_primary[0] and abs(c[0] - sel_primary[0]) == 1]
                if adjacent:
                    dyad_note = min(adjacent, key=lambda c: abs(c[2] - sel_primary[2]))
                    insert(bar, beat, step_dur, dyad_note, max(1, min(127, round(velocity * 0.85))))
            prev_treble_pitch = sel_treble[2]
            prev_treble_string = sel_treble[0]

        if cm:
            explicit = instrumental_melody is not None and index in instrumental_melody
            lead = list(instrumental_melody[index]) if explicit else []
            if not explicit:
                for cycle, beat in enumerate(range(math.ceil(measure))):
                    if beat >= measure:
                        continue
                    segment = active(bar,beat)
                    if segment is None or segment['chord'] == 'N.C.':
                        continue
                    pcs = set(parse_extended_chord(segment['chord'])['pcs'])
                    preferred = pcs & pentatonic or pcs
                    pitches = [p for p in range(60,77) if p%12 in preferred and any(0 <= p-t-capo <= 24-capo for t in STANDARD_TUNING[3:])]
                    if pitches:
                        route = pitches+pitches[-2:0:-1]
                        lead.append(dict(beat=float(beat),duration=min(1.0, measure - float(beat)),pitch=route[cycle%len(route)]))
            for note in lead:
                beat = note['beat'] if 'beat' in note else note['beat_offset']
                duration, pitch = note['duration'], note['pitch']
                _integer(pitch,0,127,'melody pitch')
                if not math.isfinite(beat) or not math.isfinite(duration) or beat < 0 or beat >= measure or duration <= 0 or beat+duration > measure+1e-9:
                    raise ValueError(f'Invalid instrumental melody timing in bar {index}')
                segment = active(bar,beat)
                if segment is None:
                    raise ValueError('Instrumental melody falls in an accompaniment rest')
                choices = [(6-i,pitch-t-capo,pitch) for i,t in enumerate(STANDARD_TUNING) if i >= 3 and 0 <= pitch-t-capo <= 24-capo]
                if not choices:
                    raise ValueError(f'Unplayable treble melody pitch {pitch}')
                item = min(choices,key=lambda x:(abs(x[1]-segment['base_fret']),x[1],x[0]))
                insert(bar,beat,duration,item,88,'CM',True)

        # --- Zero-collision ornament insertion ---
        tail = measure-0.5
        segment = active(bar,tail) if tail >= 0 else None
        ornament_candidates = []
        tail_done = False
        if embellish_hp and segment is not None:
            parsed = parse_extended_chord(segment['chord'])
            tones = set(parsed['pcs']) | {parsed['bass_pc']} if parsed else set()
            for string in (1,2,3):
                previous = [e for e in events.values() if e['bar'] == index and e['string'] == string and e['beat'] < tail]
                if not previous:
                    continue
                previous_event = max(previous,key=lambda e:e['beat'])
                for fret in range(max(0,previous_event['fret']-2),min(24-capo,previous_event['fret']+2)+1):
                    pitch = STANDARD_TUNING[6-string]+capo+fret
                    if fret != previous_event['fret'] and pitch%12 in tones:
                        ornament_candidates.append((abs(fret-previous_event['fret']),string,fret,pitch,previous_event['fret']))
            if ornament_candidates:
                _, string, fret, pitch, old = min(ornament_candidates)
                existing = events.get((index*measure+tail,string))
                if existing is None or existing['technique'] not in {'CM','H','P','PB'}:
                    clear_beat(index, tail)
                    insert(bar,tail,step_dur,(string,fret,pitch),78,'H' if fret > old else 'P',True)
                    tail_done = True
        if not tail_done and passing_bass and segment is not None and index+1 < len(bars):
            next_segment = active(bars[index+1],0)
            target = bass_of(next_segment) if next_segment is not None else None
            current = bass_of(segment)
            candidates = []
            if target is not None and current is not None:
                for string in (4,5,6):
                    if (index*measure+tail,string) in events:
                        continue
                    for fret in range(25-capo):
                        pitch = STANDARD_TUNING[6-string]+capo+fret
                        if abs(pitch-target[2]) in (1,2) and pitch%12 in diatonic:
                            candidates.append((abs(pitch-current[2]),0,fret,string,pitch))
            if candidates:
                _, _, fret, string, pitch = min(candidates)
                clear_beat(index, tail)
                insert(bar,tail,step_dur,(string,fret,pitch),88,'PB')
        vnc = vocal_counts.get(index, 0)
        if vnc <= 1 and (passing_bass or embellish_hp) and segment is not None:
            fill_beat = measure - 1.0
            fill_seg = active(bar, fill_beat) if fill_beat >= 0 else None
            if fill_seg is not None:
                fill_done = False
                if embellish_hp:
                    parsed_fill = parse_extended_chord(fill_seg['chord'])
                    tones_fill = set(parsed_fill['pcs']) | {parsed_fill['bass_pc']} if parsed_fill else set()
                    for fs in (1, 2, 3):
                        prev_evts = [e for e in events.values() if e['bar'] == index and e['string'] == fs and e['beat'] < fill_beat]
                        if not prev_evts:
                            continue
                        prev_e = max(prev_evts, key=lambda e: e['beat'])
                        chosen_orn = None
                        for ff in range(max(0, prev_e['fret'] - 2), min(24 - capo, prev_e['fret'] + 2) + 1):
                            pp = STANDARD_TUNING[6 - fs] + capo + ff
                            if ff != prev_e['fret'] and pp % 12 in tones_fill:
                                dist = abs(ff - prev_e['fret'])
                                if chosen_orn is None or dist < chosen_orn[0]:
                                    chosen_orn = (dist, fs, ff, pp, prev_e['fret'])
                        if chosen_orn is not None:
                            _, fs2, ff2, pp2, old2 = chosen_orn
                            key_check = (index * measure + fill_beat, fs2)
                            if key_check not in events or events[key_check].get('technique') not in {'CM', 'H', 'P', 'PB'}:
                                clear_beat(index, fill_beat)
                                insert(bar, fill_beat, 1.0, (fs2, ff2, pp2), 78, 'H' if ff2 > old2 else 'P', True)
                                fill_done = True
                                break
                if not fill_done and passing_bass and index + 1 < len(bars):
                    next_seg_fill = active(bars[index + 1], 0)
                    target_fill = bass_of(next_seg_fill) if next_seg_fill is not None else None
                    current_fill = bass_of(fill_seg)
                    tail_pitches = {e['pitch'] for e in events.values() if e['bar'] == index and abs(e['beat'] - (measure - 0.5)) < 1e-6}
                    cands_fill = []
                    if target_fill is not None and current_fill is not None:
                        for fs in (4, 5, 6):
                            if (index * measure + fill_beat, fs) in events:
                                continue
                            for ff in range(25 - capo):
                                pp = STANDARD_TUNING[6 - fs] + capo + ff
                                if pp not in tail_pitches and pp != current_fill[2] and abs(pp - target_fill[2]) in (2, 3, 4) and pp % 12 in diatonic:
                                    cands_fill.append((abs(pp - current_fill[2]), 0, ff, fs, pp))
                    if cands_fill:
                        _, _, ff3, fs3, pp3 = min(cands_fill)
                        clear_beat(index, fill_beat)
                        insert(bar, fill_beat, step_dur, (fs3, ff3, pp3), 84, 'PB')

        if slap_x and bar['split_segments']:
            for beat in (1.0,3.0):
                segment = active(bar,beat) if beat < measure else None
                if segment is not None:
                    slap_vel = max(1, min(127, round(82 * phrase_vel_scale * (float(dyn_scale) if dyn_scale is not None else 1.0))))
                    slaps.append(dict(bar=index,beat=beat,start=index*measure+beat,duration=min(0.5,measure-beat),
                                      string=6,fret=0,pitch=40,velocity=slap_vel,technique='X',chord=segment['chord'],
                                      segment=bar['split_segments'].index(segment)))
    ordered = sorted(events.values(),key=lambda e:(e['start'],e['string'],e['technique']))
    for event in ordered:
        vnc = vocal_counts.get(event['bar'], 0)
        if vnc >= 5:
            comp_scale = 0.88
        elif vnc >= 3:
            comp_scale = 0.94
        elif vnc <= 1:
            comp_scale = 1.05
        else:
            comp_scale = 1.0
        event['velocity'] = max(28, min(122, round(event['velocity'] * comp_scale)))
        is_beat1_bass = abs(event['beat']) < 1e-9 and event.get('string', 0) >= 4
        if is_beat1_bass and event['velocity'] < 72:
            event['velocity'] = 72
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
