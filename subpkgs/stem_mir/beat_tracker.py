"""Beat & downbeat tracker wrapping CPJKU/beat_this with a librosa fallback.

Pipeline:
  1. Neural tier (``beat_this.inference.File2Beats``) or librosa ``beat_track``.
  2. Metrical-level / tactus disambiguation for the neural downbeat sequence via an
     additive log-domain score combining onset-envelope autocorrelation at beat/bar lags,
     vocal syllable inter-onset-interval (IOI) proximity, and a quadratic log2-BPM penalty.
  3. Grid construction:
       * neural tier: downbeat-locked bars whose interior beats keep the native
         ``beat_this`` detections (or snap per subdivision, then interpolate monotonically);
       * fallback tier: a 2-state Kalman tempo-phase filter ``[t_k, T_k]`` with a
         Rauch-Tung-Striebel (RTS) backward smoothing pass.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Heuristic target ratio of median sung-syllable IOI to tactus period.
_SYLLABLE_IOI_PER_BEAT = 0.68
# Scale parameter (in octaves) for the quadratic vocal-IOI log2-deviation penalty.
_SYLLABLE_IOI_SIGMA_OCT = 1.05
# Plausible sung-syllable inter-onset-interval passband in seconds (excludes sub-160ms trills and >2.4s phrase pauses).
_SYLLABLE_IOI_MIN_SEC = 0.16
_SYLLABLE_IOI_MAX_SEC = 2.4
# Kalman observation noise standard deviations, as fractions of the beat period.
# Ordering R_downbeat < R_beat < R_vocal reflects decreasing reliability of the cue.
_R_DOWNBEAT_FRAC = 0.03
_R_BEAT_FRAC = 0.05
_R_VOCAL_FRAC = 0.09
# Kalman validation gate half-widths, as fractions of the beat period (downbeat > beat > vocal).
_GATE_DOWNBEAT_FRAC = 0.45
_GATE_BEAT_FRAC = 0.35
_GATE_VOCAL_FRAC = 0.24
# Kalman process noise standard deviations (per beat step), as fractions of the beat period.
_Q_PHASE_FRAC = 0.01
_Q_PERIOD_FRAC = 0.015


def _timestamps(values, duration):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite beat timestamps")
    return np.unique(values[(values >= 0) & (values < duration)])


def seconds_to_beats_piecewise(
    sec: float | np.ndarray,
    beat_times: list[float] | np.ndarray | None,
    bpm: float,
    beat_origin: float = 0.0,
) -> float | np.ndarray:
    """Map timestamp(s) in seconds to continuous beat coordinates using piecewise grid."""
    spb = 60.0 / bpm
    scalar = np.ndim(sec) == 0
    sec_arr = np.atleast_1d(np.asarray(sec, dtype=np.float64))
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=np.float64).ravel()
        if len(bt) >= 2 and np.all(np.diff(bt) > 0):
            beat_indices = np.arange(len(bt), dtype=np.float64)
            result = np.interp(sec_arr, bt, beat_indices)
            below = sec_arr < bt[0]
            if np.any(below):
                result[below] = (sec_arr[below] - bt[0]) / spb
            above = sec_arr > bt[-1]
            if np.any(above):
                result[above] = beat_indices[-1] + (sec_arr[above] - bt[-1]) / spb
            return float(result[0]) if scalar else result
    result = (sec_arr - beat_origin) / spb
    return float(result[0]) if scalar else result


def beats_to_seconds_piecewise(
    beat: float | np.ndarray,
    beat_times: list[float] | np.ndarray | None,
    bpm: float,
    beat_origin: float = 0.0,
) -> float | np.ndarray:
    """Inverse mapping from continuous beat coordinate(s) to seconds."""
    spb = 60.0 / bpm
    scalar = np.ndim(beat) == 0
    beat_arr = np.atleast_1d(np.asarray(beat, dtype=np.float64))
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=np.float64).ravel()
        if len(bt) >= 2 and np.all(np.diff(bt) > 0):
            beat_indices = np.arange(len(bt), dtype=np.float64)
            result = np.interp(beat_arr, beat_indices, bt)
            below = beat_arr < 0
            if np.any(below):
                result[below] = bt[0] + beat_arr[below] * spb
            above = beat_arr > beat_indices[-1]
            if np.any(above):
                result[above] = bt[-1] + (beat_arr[above] - beat_indices[-1]) * spb
            return float(result[0]) if scalar else result
    result = beat_origin + beat_arr * spb
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Onset-autocorrelation & vocal-IOI heuristic meter/tactus disambiguator
# ---------------------------------------------------------------------------

def _onset_autocorrelation(onset_env: np.ndarray) -> np.ndarray:
    """Empirical (biased, mean-removed) autocorrelation of the onset envelope, normalised to acf[0] = 1."""
    x = np.asarray(onset_env, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return np.ones(max(1, n), dtype=np.float64)
    x = x - float(np.mean(x))
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    if acf[0] <= 1e-12:
        return np.zeros(n, dtype=np.float64)
    return acf / acf[0]


def _acf_at(acf: np.ndarray, lag_frames: float, halfwidth: int = 2) -> float:
    """Peak autocorrelation within +-halfwidth frames of a (fractional) lag; 0 if out of range."""
    centre = int(round(lag_frames))
    if centre < 1 or centre >= len(acf):
        return 0.0
    lo, hi = max(1, centre - halfwidth), min(len(acf), centre + halfwidth + 1)
    return float(np.max(acf[lo:hi]))


def _tempo_range_penalty(bpm: float, lo_bpm: float, hi_bpm: float) -> float:
    """Quadratic log2-BPM penalty centred on the geometric mean of [lo_bpm, hi_bpm] (scale = max(1e-3, range/4) octaves)."""
    if bpm <= 0.0 or bpm < lo_bpm or bpm > hi_bpm:
        return -np.inf
    centre = math.sqrt(lo_bpm * hi_bpm)
    sigma_oct = max(1e-3, math.log2(hi_bpm / lo_bpm) / 4.0) if hi_bpm > lo_bpm else 0.1
    return -0.5 * (math.log2(bpm / centre) / sigma_oct) ** 2


def _syllable_ioi_penalty(syl_arr: np.ndarray, spb: float) -> float:
    """Quadratic log2-deviation penalty between median vocal syllable IOI (within [_SYLLABLE_IOI_MIN_SEC, _SYLLABLE_IOI_MAX_SEC]) and target beat-scaled syllable IOI; returns 0.0 when fewer than 4 syllables or 2 in-band IOIs are available."""
    if len(syl_arr) < 4:
        return 0.0
    iois = np.diff(syl_arr)
    iois = iois[(iois >= _SYLLABLE_IOI_MIN_SEC) & (iois <= _SYLLABLE_IOI_MAX_SEC)]
    if len(iois) < 2:
        return 0.0
    med_ioi = float(np.median(iois))
    target = _SYLLABLE_IOI_PER_BEAT * spb
    z = math.log2(max(1e-3, med_ioi) / max(1e-3, target)) / _SYLLABLE_IOI_SIGMA_OCT
    return -0.5 * z * z


def _metrical_level_score(
    acf: np.ndarray,
    frames_per_sec: float,
    bar_sec: float,
    measure_beats: float,
    syl_arr: np.ndarray,
    lo_bpm: float,
    hi_bpm: float,
) -> float:
    """Additive log-domain metrical score combining onset autocorrelation at beat/bar lags, syllable-IOI alignment, and log-tempo range preference."""
    spb = bar_sec / measure_beats
    bpm = 60.0 / max(1e-6, spb)
    range_pen = _tempo_range_penalty(bpm, lo_bpm, hi_bpm)
    if not np.isfinite(range_pen):
        return -np.inf
    acf_beat = max(1e-3, _acf_at(acf, spb * frames_per_sec))
    acf_bar = max(1e-3, _acf_at(acf, bar_sec * frames_per_sec))
    log_acf = math.log(acf_beat) + math.log(acf_bar)
    return log_acf + _syllable_ioi_penalty(syl_arr, spb) + range_pen


def _harmonic_change_median_sec(
    pulse_times: np.ndarray,
    pulse_chroma: np.ndarray | None,
) -> float | None:
    """Estimate median harmonic-change interval D_hc (seconds) from 2-pulse cosine novelty on body chroma."""
    if pulse_chroma is None or pulse_chroma.ndim != 2 or pulse_chroma.shape[0] != 12:
        return None
    n_p = min(len(pulse_times), pulse_chroma.shape[1])
    if n_p < 6:
        return None
    nov = np.zeros(n_p, dtype=np.float64)
    for k in range(2, n_p - 1):
        left = np.mean(pulse_chroma[:, k - 2 : k], axis=1)
        right = np.mean(pulse_chroma[:, k : min(n_p, k + 2)], axis=1)
        nl = float(np.linalg.norm(left))
        nr = float(np.linalg.norm(right))
        if nl > 1e-9 and nr > 1e-9:
            cos_sim = float(np.clip(np.dot(left, right) / (nl * nr), -1.0, 1.0))
            nov[k] = 1.0 - cos_sim
    interior = nov[2 : n_p - 1]
    if interior.size < 3:
        return None
    med = float(np.median(interior))
    mad = float(np.median(np.abs(interior - med)))
    thresh = max(med + 3.0 * 1.4826 * mad, med + 0.04)
    peaks: list[float] = []
    for k in range(2, n_p - 1):
        if nov[k] > thresh and nov[k] >= nov[k - 1] and nov[k] > nov[k + 1]:
            peaks.append(float(pulse_times[k]))
    if len(peaks) < 2:
        return None
    diffs = np.diff(np.asarray(peaks, dtype=np.float64))
    pos = diffs[diffs > 0.25]
    if pos.size < 1:
        return None
    return float(np.median(pos))


def _tactus_octave_log_posterior(
    acf_sub: np.ndarray,
    frames_per_sec: float,
    median_bar: float,
    bar_q_beats: float,
    beat_arr: np.ndarray,
    cleaned_db: np.ndarray,
    syl_arr: np.ndarray,
    lo_bpm: float,
    hi_bpm: float,
    is_compound_meter: bool = False,
    d_hc_sec: float | None = None,
    sigma_trk: float = 0.50,
    sigma_hc: float = 1.00,
) -> dict[float, float]:
    """Compute log posterior across metrical octave hypotheses m in H ({1.0} if compound else {0.5, 1.0, 2.0})."""
    hypotheses = (1.0,) if is_compound_meter else (0.5, 1.0, 2.0)
    phi_2 = 1.0
    if len(cleaned_db) >= 2 and len(beat_arr) >= 2:
        db_p_idx = np.asarray([int(np.argmin(np.abs(beat_arr - t))) for t in cleaned_db], dtype=np.int64)
        p_diffs = np.diff(db_p_idx)
        valid_p = p_diffs[p_diffs >= 1]
        if valid_p.size >= 1:
            phi_2 = float(np.mean((valid_p % 2) == 0))
    scores: dict[float, float] = {}
    for m in hypotheses:
        t_m = (m * median_bar) / max(1e-6, bar_q_beats)
        bpm_m = 60.0 / max(1e-6, t_m)
        prior_pen = _tempo_range_penalty(bpm_m, lo_bpm, hi_bpm)
        if not np.isfinite(prior_pen):
            scores[m] = -np.inf
            continue
        phi_m = phi_2 if m == 2.0 else 1.0
        lam_lat = math.log(max(phi_m, 1e-3))
        a_1 = max(0.0, _acf_at(acf_sub, t_m * frames_per_sec))
        a_2 = max(0.0, _acf_at(acf_sub, 0.5 * t_m * frames_per_sec))
        a_3 = max(0.0, _acf_at(acf_sub, (t_m / 3.0) * frames_per_sec))
        max_a = max(a_1, a_2, a_3)
        r_m = max(a_2, a_3) / max_a if max_a > 1e-6 else 1.0
        lam_sub = math.log(max(r_m, 0.1))
        lam_hc = 0.0
        if d_hc_sec is not None and math.isfinite(d_hc_sec) and d_hc_sec > 0.0:
            lam_hc = -((math.log2(d_hc_sec / max(1e-6, m * median_bar)) + 0.5) ** 2) / (2.0 * sigma_hc ** 2)
        lam_v = _syllable_ioi_penalty(syl_arr, t_m)
        trust_pen = -((math.log2(m)) ** 2) / (2.0 * sigma_trk ** 2)
        scores[m] = prior_pen + lam_lat + lam_sub + lam_hc + lam_v + trust_pen
    return scores


# ---------------------------------------------------------------------------
# Neural-tier interior beat placement
# ---------------------------------------------------------------------------

def _bar_beat_positions(
    beat_arr: np.ndarray,
    db_start: float,
    db_end: float,
    measure_beats: int,
    is_compound: bool = False,
) -> list[float]:
    """Beat positions k = 0..measure_beats-1 for one bar spanning [db_start, db_end).

    * Exactly ``measure_beats - 1`` strictly increasing neural interior beats -> keep them verbatim
      (unless ``is_compound`` with a single half-bar dotted-quarter anchor).
    * Exactly ``2 * measure_beats - 1`` eighth-note neural interior beats (e.g. compound 6/8 where
      ``beat_this`` tracks all 6 eighth-note pulses while ``bar_grid_beats == 3`` quarter-note units)
      -> keep every second subdivision verbatim (``interior_sub[1::2]``).
    * In compound meters (``is_compound=True``), match interior neural beats onto the half-integer
      eighth-note / dotted-quarter lattice ``k = 0.5 * m`` (``m = 1 .. 2*measure_beats - 1``) so a
      half-bar dotted-quarter pulse at ``k = 1.5`` or partial eighth-note subdivisions interpolate
      cleanly onto integer quarter-note grid points ``0 .. measure_beats - 1``.
    * Otherwise snap each expected subdivision to the nearest neural beat within 0.25 of a local
      beat period, and fill unmatched subdivisions by monotonic piecewise-linear interpolation
      between the matched anchors (bar boundaries included).
    """
    bar_span = db_end - db_start
    local_spb = bar_span / measure_beats
    interior = beat_arr[(beat_arr > db_start + 0.3 * local_spb) & (beat_arr < db_end - 0.3 * local_spb)]
    if measure_beats == 1:
        return [db_start]
    if not is_compound and len(interior) == measure_beats - 1 and np.all(np.diff(interior) >= 0.45 * local_spb):
        return [db_start] + [float(t) for t in interior]
    sub_spb = bar_span / (2 * measure_beats)
    interior_sub = beat_arr[(beat_arr > db_start + 0.3 * sub_spb) & (beat_arr < db_end - 0.3 * sub_spb)]
    if len(interior_sub) == 2 * measure_beats - 1 and np.all(np.diff(interior_sub) >= 0.45 * sub_spb):
        return [db_start] + [float(t) for t in interior_sub[1::2]]
    inside = beat_arr[(beat_arr > db_start + 1e-4) & (beat_arr < db_end - 1e-4)]
    if is_compound and len(inside):
        anchor_k_c: list[float] = [0.0]
        anchor_t_c: list[float] = [db_start]
        if len(inside) == 1 and abs(float(inside[0]) - (db_start + 0.5 * bar_span)) <= 0.22 * bar_span:
            anchor_k_c.append(0.5 * measure_beats)
            anchor_t_c.append(float(inside[0]))
        else:
            tol_sub = 0.35 * sub_spb
            for m in range(1, 2 * measure_beats):
                k_half = 0.5 * m
                expected = db_start + k_half * local_spb
                j = int(np.argmin(np.abs(inside - expected)))
                cand = float(inside[j])
                if abs(cand - expected) <= tol_sub and cand > anchor_t_c[-1] + 1e-4:
                    anchor_k_c.append(k_half)
                    anchor_t_c.append(cand)
        anchor_k_c.append(float(measure_beats))
        anchor_t_c.append(db_end)
        return [float(t) for t in np.interp(np.arange(measure_beats, dtype=np.float64), anchor_k_c, anchor_t_c)]
    tol = 0.25 * local_spb
    anchor_k: list[float] = [0.0]
    anchor_t: list[float] = [db_start]
    for k in range(1, measure_beats):
        expected = db_start + k * local_spb
        if not len(inside):
            break
        j = int(np.argmin(np.abs(inside - expected)))
        cand = float(inside[j])
        # Windows of half-width 0.25*T around consecutive subdivisions never overlap, so the
        # matched anchors are increasing; the explicit check guards against degenerate spans.
        if abs(cand - expected) <= tol and cand > anchor_t[-1] + 1e-4:
            anchor_k.append(float(k))
            anchor_t.append(cand)
    anchor_k.append(float(measure_beats))
    anchor_t.append(db_end)
    return [float(t) for t in np.interp(np.arange(measure_beats, dtype=np.float64), anchor_k, anchor_t)]


# ---------------------------------------------------------------------------
# Fallback tier: Kalman tempo-phase filter + RTS smoother
# ---------------------------------------------------------------------------

def _kalman_rts_beat_grid(
    beat_origin: float,
    spb: float,
    duration: float,
    measure_beats: float,
    downbeat_arr: np.ndarray,
    beat_arr: np.ndarray,
    syl_arr: np.ndarray,
    lo_bpm: float,
    hi_bpm: float,
) -> tuple[list[float], float]:
    """Track beats with a constant-tempo state-space model and smooth with RTS.

    State x_k = [t_k, T_k]^T (beat time, beat period); transition t_{k+1} = t_k + T_k, T_{k+1} = T_k.
    Observation z_k = t_k + v_k with v_k ~ N(0, R), where R is chosen per cue type
    (downbeat < neural/librosa beat < vocal syllable onset). Unobserved steps are pure predictions.
    Returns (beat_times, final_period).
    """
    t_min, t_max = 60.0 / hi_bpm, 60.0 / lo_bpm
    spb = float(np.clip(spb, t_min, t_max))
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = np.diag([(_Q_PHASE_FRAC * spb) ** 2, (_Q_PERIOD_FRAC * spb) ** 2])
    r_downbeat = (_R_DOWNBEAT_FRAC * spb) ** 2
    r_beat = (_R_BEAT_FRAC * spb) ** 2
    r_vocal = (_R_VOCAL_FRAC * spb) ** 2
    downbeat_gate = _GATE_DOWNBEAT_FRAC * spb
    beat_gate = _GATE_BEAT_FRAC * spb
    vocal_gate = _GATE_VOCAL_FRAC * spb

    x = np.array([float(beat_origin), spb])
    P = np.diag([r_downbeat, (0.1 * spb) ** 2])
    xs_filt, ps_filt = [x.copy()], [P.copy()]
    xs_pred, ps_pred = [x.copy()], [P.copy()]  # index 0 unused by the smoother

    def _nearest(cands: np.ndarray, target: float) -> float:
        return float(cands[np.argmin(np.abs(cands - target))])

    while x[0] + x[1] * 0.5 < duration:
        pos, cur_spb = float(x[0]), float(x[1])
        x_pred = F @ x
        x_pred[1] = float(np.clip(x_pred[1], t_min, t_max))
        P_pred = F @ P @ F.T + Q
        pred_t = float(x_pred[0])
        beat_idx = len(xs_filt)

        z, r = None, None
        is_bar_beat = measure_beats > 0 and abs(round(beat_idx / measure_beats) * measure_beats - beat_idx) < 1e-4
        if is_bar_beat and len(downbeat_arr):
            cands = downbeat_arr[(downbeat_arr > pos + cur_spb * 0.35) & (downbeat_arr < pos + cur_spb * 1.65)]
            if len(cands):
                c = _nearest(cands, pred_t)
                if abs(c - pred_t) <= downbeat_gate:
                    z, r = c, r_downbeat
        if z is None and len(beat_arr):
            cands = beat_arr[(beat_arr > pos + cur_spb * 0.3) & (beat_arr < pos + cur_spb * 1.7)]
            if len(cands):
                c = _nearest(cands, pred_t)
                if abs(c - pred_t) <= beat_gate:
                    z, r = c, r_beat
        if z is None and len(syl_arr):
            cands = syl_arr[(syl_arr > pos + cur_spb * 0.4) & (syl_arr < pos + cur_spb * 1.6)]
            if len(cands):
                c = _nearest(cands, pred_t)
                if abs(c - pred_t) <= vocal_gate:
                    z, r = c, r_vocal

        if z is not None:
            s = float(H @ P_pred @ H) + r
            K = (P_pred @ H) / s
            x_new = x_pred + K * (z - pred_t)
            I_KH = np.eye(2) - np.outer(K, H)
            P_new = I_KH @ P_pred @ I_KH.T + r * np.outer(K, K)  # Joseph form
        else:
            x_new, P_new = x_pred.copy(), P_pred.copy()

        x_new[1] = float(np.clip(x_new[1], t_min, t_max))
        if x_new[0] <= pos + 1e-4:
            x_new[0] = pos + x_new[1]
        xs_pred.append(x_pred)
        ps_pred.append(P_pred)
        xs_filt.append(x_new)
        ps_filt.append(P_new)
        x, P = x_new, P_new

    # Rauch-Tung-Striebel backward pass.
    n = len(xs_filt)
    xs_smooth = [None] * n
    ps_smooth = [None] * n
    xs_smooth[-1], ps_smooth[-1] = xs_filt[-1].copy(), ps_filt[-1].copy()
    for k in range(n - 2, -1, -1):
        try:
            C = np.linalg.solve(ps_pred[k + 1].T, (ps_filt[k] @ F.T).T).T
        except np.linalg.LinAlgError:
            xs_smooth[k], ps_smooth[k] = xs_filt[k].copy(), ps_filt[k].copy()
            continue
        xs_smooth[k] = xs_filt[k] + C @ (xs_smooth[k + 1] - xs_pred[k + 1])
        ps_smooth[k] = ps_filt[k] + C @ (ps_smooth[k + 1] - ps_pred[k + 1]) @ C.T

    beat_times = [float(beat_origin)]
    for k in range(1, n):
        period = float(np.clip(xs_smooth[k][1], t_min, t_max))
        t = float(xs_smooth[k][0])
        if not np.isfinite(t) or t <= beat_times[-1] + 1e-4:
            t = beat_times[-1] + period
        beat_times.append(t)
    final_period = float(np.clip(xs_smooth[-1][1], t_min, t_max))
    return beat_times, final_period


def track_beats_and_downbeats(
    audio_path: str | Path,
    prefer_cuda: bool = True,
    bpm_range: tuple[float, float] = (60.0, 180.0),
    meter_hint: str | None = None,
    vocal_onsets_sec: list[float] | None = None,
    pre_vocal_bars: int = 1,
) -> dict:
    """Track beats, downbeats, meter, bpm, and beat_origin from audio.

    Returns dict:
        - `bpm`: float (octave-folded into bpm_range when compatible; otherwise physical grid tempo)
        - `bpm_in_range`: bool (whether bpm lies within bpm_range)
        - `beat_origin`: float (`pre_vocal_bars`-bar pre-vocal pickup origin or first downbeat/beat in seconds >= 0.0)
        - `meter`: str (inferred 'num/den' or validated meter_hint 'num/den')
        - `beats`: list[float]
        - `downbeats`: list[float]
        - `backend`: str ('beat_this' or 'librosa')
        - `beat_times`: list[float] — piecewise quarter-note beat grid snapped to detected beats
        - `bar_start_seconds`: list[float] — bar boundary timestamps
    """
    import librosa
    from .separator_engine import resolve_device

    if not isinstance(bpm_range, (tuple, list)) or len(bpm_range) != 2:
        raise ValueError(f"Invalid bpm_range: {bpm_range!r}")
    lo_bpm, hi_bpm = float(bpm_range[0]), float(bpm_range[1])
    if not (math.isfinite(lo_bpm) and math.isfinite(hi_bpm) and 0.0 < lo_bpm <= hi_bpm):
        raise ValueError(f"Invalid bpm_range: {bpm_range!r}")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if not y.size or not np.all(np.isfinite(y)):
        raise ValueError("Empty or nonfinite audio")
    duration = len(y) / sr
    beats, downbeats, backend = None, None, "librosa"
    try:
        from beat_this.inference import File2Beats
        from runtime_paths import resolve_model_path

        ckpt_file = resolve_model_path("hub", "checkpoints", "beat_this-final0.ckpt")
        ckpt_arg = str(ckpt_file) if ckpt_file.is_file() else "final0"
        tracker = File2Beats(checkpoint_path=ckpt_arg, device=resolve_device(prefer_cuda))
        candidate_beats, candidate_downbeats = tracker(str(audio_path))
        candidate_beats = _timestamps(candidate_beats, duration)
        candidate_downbeats = _timestamps(candidate_downbeats, duration)
        if len(candidate_beats) >= 2:
            beats, downbeats, backend = candidate_beats, candidate_downbeats, "beat_this"
        else:
            logger.warning("beat_this returned fewer than two distinct beats")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("beat_this tier failed: %s", exc)
    hop, n_fft = 256, 2048
    frames_per_sec = sr / hop
    padded = np.pad(y, (n_fft, 0))
    onset_env = librosa.onset.onset_strength(y=padded, sr=sr, hop_length=hop, n_fft=n_fft)
    if beats is None:
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False)
        beats = _timestamps(librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop) - n_fft / sr, duration)
        if len(beats) < 2:
            frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
            beats = _timestamps(librosa.frames_to_time(frames, sr=sr, hop_length=hop) - n_fft / sr, duration)
        if len(beats) < 2:
            raise ValueError("Insufficient periodic onsets to estimate tempo")
    native_beats = np.asarray(beats, dtype=np.float64)
    raw_diffs = np.diff(native_beats)
    native_bpm = 60.0 / float(np.median(raw_diffs))
    bpm = native_bpm
    folded_beats = native_beats.copy()
    while bpm < lo_bpm and bpm * 2.0 <= hi_bpm:
        folded_beats = np.sort(np.concatenate((folded_beats, (folded_beats[:-1] + folded_beats[1:]) / 2)))
        bpm *= 2.0
    while bpm > hi_bpm and bpm * 0.5 >= lo_bpm:
        if len(folded_beats) >= 4:
            parity = 0
            if downbeats is not None and len(downbeats):
                parity = int(np.argmin(np.abs(folded_beats - downbeats[0]))) % 2
            target_spb = 120.0 / bpm
            decimated = [float(folded_beats[parity])]
            for b_t in folded_beats[parity + 1:]:
                if float(b_t) - decimated[-1] >= 0.7 * target_spb:
                    decimated.append(float(b_t))
            if len(decimated) >= 2:
                folded_beats = np.asarray(decimated, dtype=np.float64)
        bpm /= 2.0
    has_neural_downbeats = backend == "beat_this" and downbeats is not None and len(downbeats) >= 2
    syl_arr = (
        np.asarray(
            sorted(
                float(t)
                for t in vocal_onsets_sec
                if t is not None and math.isfinite(float(t)) and 0.0 <= float(t) < duration
            ),
            dtype=np.float64,
        )
        if vocal_onsets_sec
        else np.empty(0, dtype=np.float64)
    )
    # When tracking a vocal stem (vocal_onsets_sec supplied) with held vowels/pauses or out-of-range BPM, fold multi-beat IOIs (k=1..4) using a globally consistent base period in [60/hi_bpm, 60/lo_bpm]
    high_cv = len(raw_diffs) >= 4 and float(np.std(raw_diffs) / max(1e-6, np.mean(raw_diffs))) > 0.25
    if (not has_neural_downbeats) and len(syl_arr) >= 2 and (bpm < lo_bpm or bpm > hi_bpm or high_cv):
        t_min, t_max = 60.0 / hi_bpm, 60.0 / lo_bpm
        pos_diffs = [float(d) for d in raw_diffs if d > 0]
        cand_periods: list[float] = []
        for d in pos_diffs:
            for mult in (1, 2, 3, 4):
                p_cand = d / mult
                if t_min <= p_cand <= t_max:
                    cand_periods.append(p_cand)
        best_folded: list[float] = []
        best_loss = np.inf
        for t_cand in cand_periods:
            cur_folded: list[float] = []
            loss_sum = 0.0
            for d in pos_diffs:
                k = max(1, min(4, int(round(d / t_cand))))
                p = d / k
                rel_err = abs(p - t_cand) / t_cand
                if t_min <= p <= t_max:
                    cur_folded.append(p)
                    loss_sum += rel_err + 0.02 * (k - 1)
                else:
                    loss_sum += 1.0 + rel_err
            mean_loss = loss_sum / len(pos_diffs)
            if len(cur_folded) >= 4 and mean_loss < best_loss:
                best_loss = mean_loss
                best_folded = cur_folded
        if len(best_folded) >= 4:
            spb_refold = float(np.median(best_folded))
            bpm = 60.0 / spb_refold
            refolded_pts: list[float] = [float(native_beats[0])]
            for idx_b in range(len(native_beats) - 1):
                b0 = float(native_beats[idx_b])
                b1 = float(native_beats[idx_b + 1])
                d = b1 - b0
                if d > 0:
                    k = max(1, min(4, int(round(d / spb_refold))))
                    if t_min <= d / k <= t_max:
                        for j in range(1, k):
                            refolded_pts.append(b0 + j * d / k)
                refolded_pts.append(b1)
            folded_beats = np.asarray(refolded_pts, dtype=np.float64)
    beats = native_beats if has_neural_downbeats else folded_beats
    if 0.95 * lo_bpm <= bpm <= 1.05 * hi_bpm:
        bpm = float(np.clip(bpm, lo_bpm, hi_bpm))
    else:
        bpm = float(bpm)
    accent = np.zeros(len(beats), dtype=np.float64)
    for i, time in enumerate(beats):
        frame = int(round((time * sr + n_fft) / hop))
        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
        accent[i] = float(np.max(window)) if window.size else 0.0
    measure_beats, meter_den, phase, peak_corr = 4, 4, 0, -np.inf
    neural_pulses_per_bar: float | None = None
    if has_neural_downbeats:
        indices = np.asarray([np.argmin(np.abs(beats - t)) for t in downbeats])
        diff_idx = np.diff(indices)
        valid_diffs = diff_idx[diff_idx >= 2]
        if len(diff_idx):
            neural_pulses_per_bar = float(np.median(valid_diffs if len(valid_diffs) else diff_idx))
    has_meter_hint = meter_hint is not None and str(meter_hint).strip() != ""
    if has_meter_hint:
        try:
            m_num, m_den = map(int, str(meter_hint).strip().split("/"))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid meter_hint: {meter_hint!r}") from exc
        if m_num < 2 or m_den not in (2, 4, 8):
            raise ValueError(f"Invalid meter_hint: {meter_hint!r}")
        measure_beats, meter_den = m_num, m_den
    elif has_neural_downbeats and neural_pulses_per_bar is not None:
        measure_beats = max(2, min(12, int(math.floor(neural_pulses_per_bar + 0.5))))
        if measure_beats in (6, 9, 12) and native_bpm >= 120.0:
            meter_den = 8
        elif measure_beats == 2 and native_bpm <= 90.0:
            # Distinguish compound 6/8 tracked at 2 dotted-quarter beats/bar from simple 2/4 via ternary subdivision ACF
            acf_pre = _onset_autocorrelation(onset_env)
            t_beat = 60.0 / max(1e-6, native_bpm)
            c_tern = 0.5 * (
                _acf_at(acf_pre, frames_per_sec * (t_beat / 3.0))
                + _acf_at(acf_pre, frames_per_sec * (2.0 * t_beat / 3.0))
            )
            c_duple = _acf_at(acf_pre, frames_per_sec * (0.5 * t_beat))
            if c_tern > max(0.08, 1.15 * c_duple):
                measure_beats, meter_den = 6, 8
    else:
        for m in range(2, 8):
            for p in range(m):
                indicator = (np.arange(len(beats)) % m == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                corr = float(a @ t / denominator) if denominator > 0 else 0.0
                if corr > peak_corr:
                    measure_beats, phase, peak_corr = m, p, corr

    beat_scale = 4.0 / meter_den
    bar_q_beats = measure_beats * beat_scale
    is_compound_meter = meter_den == 8 and measure_beats in (6, 9, 12)
    # Express bpm / spb_q in quarter-note units when neural pulses tracked eighth-notes or dotted-quarters in compound /8
    if is_compound_meter and has_neural_downbeats and neural_pulses_per_bar is not None:
        if abs(neural_pulses_per_bar - measure_beats) <= 1.5:
            q_bpm = native_bpm * beat_scale
        elif abs(neural_pulses_per_bar - (measure_beats / 3.0)) <= 0.6:
            q_bpm = native_bpm * 1.5
        else:
            q_bpm = bpm
        while q_bpm < lo_bpm and q_bpm * 2.0 <= hi_bpm:
            q_bpm *= 2.0
        while q_bpm > hi_bpm and q_bpm * 0.5 >= lo_bpm:
            q_bpm *= 0.5
        bpm = float(np.clip(q_bpm, lo_bpm, hi_bpm)) if (0.95 * lo_bpm <= q_bpm <= 1.05 * hi_bpm) else float(q_bpm)
    is_int_bar_q = bar_q_beats >= 1.0 and abs(bar_q_beats - round(bar_q_beats)) < 1e-6
    bar_grid_beats = max(1, int(round(bar_q_beats)))
    spb_q = 60.0 / bpm
    if not has_neural_downbeats:
        if has_meter_hint and is_int_bar_q:
            for p in range(bar_grid_beats):
                indicator = (np.arange(len(beats)) % bar_grid_beats == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                corr = float(a @ t / denominator) if denominator > 0 else 0.0
                if corr > peak_corr:
                    phase, peak_corr = p, corr
        downbeats = np.asarray([float(beats[min(phase, len(beats) - 1)])], dtype=np.float64)

    # Neural downbeat sequence cleaning & parity disambiguation
    beat_arr = np.asarray(beats, dtype=np.float64)

    db = np.asarray(downbeats, dtype=np.float64) if downbeats is not None else np.empty(0, dtype=np.float64)
    median_bar = float(np.median(np.diff(db))) if len(db) >= 2 else float(bar_q_beats * spb_q)
    use_neural_grid = has_neural_downbeats and len(db) >= 3

    if use_neural_grid:
        median_bar = float(np.median(np.diff(db)))
        # Clean spurious duplicate downbeats
        cleaned = [float(db[0])]
        for i in range(1, len(db)):
            if float(db[i]) - cleaned[-1] >= 0.62 * median_bar:
                cleaned.append(float(db[i]))
        db = np.asarray(cleaned, dtype=np.float64)
        if len(db) >= 2:
            median_bar = float(np.median(np.diff(db)))

        # Interpolate missed downbeats across gaps
        filled: list[float] = [float(db[0])]
        for i in range(len(db) - 1):
            gap = float(db[i + 1] - db[i])
            if gap > 1.65 * median_bar:
                k = max(2, int(round(gap / median_bar)))
                for j in range(1, k):
                    filled.append(float(db[i]) + j * gap / k)
            filled.append(float(db[i + 1]))
        db = np.asarray(filled, dtype=np.float64)
        if len(db) >= 2:
            median_bar = float(np.median(np.diff(db)))

        # Multi-cue metrical tactus octave posterior over m in {0.5, 1.0, 2.0} ({1.0} for compound /8):
        # combines tempo prior, downbeat pulse-index parity lattice, 1024-win subdivision ACF,
        # 2-pulse harmonic-change novelty, vocal syllable IOI, and relaxed tracker trust (sigma_trk = 0.5).
        n_fft_sub = 1024
        padded_sub = np.pad(y, (n_fft_sub, 0))
        onset_env_sub = librosa.onset.onset_strength(y=padded_sub, sr=sr, hop_length=hop, n_fft=n_fft_sub)
        acf_sub = _onset_autocorrelation(onset_env_sub)
        d_hc_sec: float | None = None
        if not is_compound_meter and len(beat_arr) >= 6:
            hop_chr = 512
            chr_fps = sr / hop_chr
            body_chr = librosa.feature.chroma_cqt(
                y=y,
                sr=sr,
                hop_length=hop_chr,
                fmin=float(librosa.note_to_hz("G2")),
                n_octaves=4,
            )
            n_chr = body_chr.shape[1]
            pulse_chr = np.zeros((12, len(beat_arr)), dtype=np.float64)
            med_p = float(np.median(np.diff(beat_arr)))
            for pk, pt in enumerate(beat_arr):
                pt_next = float(beat_arr[pk + 1]) if pk + 1 < len(beat_arr) else float(pt + med_p)
                c0 = max(0, min(n_chr - 1, int(round(float(pt) * chr_fps))))
                c1 = max(c0 + 1, min(n_chr, int(round(pt_next * chr_fps))))
                pulse_chr[:, pk] = np.mean(body_chr[:, c0:c1], axis=1)
            d_hc_sec = _harmonic_change_median_sec(beat_arr, pulse_chr)
        score_by_m = _tactus_octave_log_posterior(
            acf_sub=acf_sub,
            frames_per_sec=frames_per_sec,
            median_bar=median_bar,
            bar_q_beats=bar_q_beats,
            beat_arr=beat_arr,
            cleaned_db=db,
            syl_arr=syl_arr,
            lo_bpm=lo_bpm,
            hi_bpm=hi_bpm,
            is_compound_meter=is_compound_meter,
            d_hc_sec=d_hc_sec,
        )
        if not any(np.isfinite(v) for v in score_by_m.values()):
            centre_bpm = math.sqrt(lo_bpm * hi_bpm)
            for m in ((1.0,) if is_compound_meter else (0.5, 1.0, 2.0)):
                cand_bpm = 60.0 * bar_q_beats / max(1e-6, m * median_bar)
                if 0.85 * lo_bpm <= cand_bpm <= 1.15 * hi_bpm:
                    score_by_m[m] = -abs(math.log2(max(1e-6, cand_bpm) / centre_bpm))
        chosen_m = max(score_by_m, key=lambda m: score_by_m[m]) if score_by_m else 1.0
        if chosen_m == 0.5 and np.isfinite(score_by_m[0.5]):
            subdivided: list[float] = [float(db[0])]
            for i in range(len(db) - 1):
                db_s, db_e = float(db[i]), float(db[i + 1])
                mid_exp = 0.5 * (db_s + db_e)
                tol_mid = 0.25 * (db_e - db_s)
                inside_mid = beat_arr[(beat_arr > db_s + 1e-4) & (beat_arr < db_e - 1e-4)]
                mid_t = mid_exp
                if len(inside_mid):
                    j_m = int(np.argmin(np.abs(inside_mid - mid_exp)))
                    if abs(float(inside_mid[j_m]) - mid_exp) <= tol_mid:
                        mid_t = float(inside_mid[j_m])
                subdivided.append(mid_t)
                subdivided.append(db_e)
            db = np.asarray(subdivided, dtype=np.float64)
            if len(db) >= 2:
                median_bar = float(np.median(np.diff(db)))
        elif chosen_m == 2.0 and np.isfinite(score_by_m[2.0]):
            # Downbeat phase parity: combine low-band (< 240 Hz) bass/kick downbeat accent with
            # vocal syllable-to-downbeat alignment when vocal onsets are available.
            from scipy.signal import butter, sosfilt
            y_low = sosfilt(butter(2, 240.0, fs=sr, output="sos"), padded)
            low_env = librosa.onset.onset_strength(y=y_low, sr=sr, hop_length=hop, n_fft=n_fft)
            env_scale = max(1e-6, float(np.percentile(onset_env, 90)))
            low_scale = max(1e-6, float(np.percentile(low_env, 90)))
            chosen_parity, min_parity_loss = 0, np.inf
            for par in (0, 1):
                db_cand = db[par::2]
                if len(db_cand) < 1:
                    continue
                db_accent = np.zeros(len(db_cand))
                for ai, t in enumerate(db_cand):
                    frame = int(round((t * sr + n_fft) / hop))
                    w_full = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
                    w_low = low_env[max(0, frame - 1):min(len(low_env), frame + 2)]
                    a_full = (float(np.max(w_full)) / env_scale) if w_full.size else 0.0
                    a_low = (float(np.max(w_low)) / low_scale) if w_low.size else 0.0
                    db_accent[ai] = 0.45 * a_full + 0.55 * a_low
                acoustic_loss = -float(np.mean(db_accent))
                if len(syl_arr):
                    dists = np.array([float(np.min(np.abs(db_cand - s))) for s in syl_arr])
                    vocal_loss = float(np.median(dists)) / max(1e-3, 2.0 * median_bar)
                    loss = vocal_loss + 0.35 * acoustic_loss
                else:
                    loss = acoustic_loss
                if loss < min_parity_loss:
                    chosen_parity, min_parity_loss = par, loss
            demoted_db = db[1 - chosen_parity::2]
            if len(demoted_db):
                min_sep = 0.35 * max(0.15, float(spb_q))
                keep_demoted = [
                    float(d_t)
                    for d_t in demoted_db
                    if len(beat_arr) == 0 or float(np.min(np.abs(beat_arr - float(d_t)))) >= min_sep
                ]
                if keep_demoted:
                    beat_arr = np.unique(np.sort(np.concatenate([beat_arr, np.asarray(keep_demoted, dtype=np.float64)])))
            db = db[chosen_parity::2]
            if len(db) == 1:
                db = np.array([float(db[0]), float(db[0]) + 2.0 * median_bar], dtype=np.float64)
            if len(db) >= 2:
                median_bar = float(np.median(np.diff(db)))

    use_neural_grid = has_neural_downbeats and len(db) >= 2

    # Pre-Vocal Pickup Origin (retaining `pre_bars` audible intro bars while preserving detected downbeat phase)
    pre_bars = max(0, int(pre_vocal_bars))
    first_downbeat = float(db[0]) if len(db) else float(downbeats[0] if len(downbeats) else beats[0])
    bar_dur_init = bar_q_beats * spb_q
    win_rms = max(1, int(round(0.050 * sr)))
    if len(y) >= win_rms:
        n_w = len(y) // win_rms
        frame_rms_arr = np.sqrt(np.mean(y[: n_w * win_rms].astype(np.float64).reshape(n_w, win_rms) ** 2, axis=1))
        audible_bar_floor = max(1e-4, 0.04 * float(np.percentile(frame_rms_arr, 85.0)))
    else:
        audible_bar_floor = 1e-4

    def _bar_is_audible(t_s: float, t_e: float) -> bool:
        i0 = max(0, min(len(y), int(round(t_s * sr))))
        i1 = max(i0 + 1, min(len(y), int(round(t_e * sr))))
        if i0 >= len(y):
            return False
        seg = y[i0:i1].astype(np.float64)
        return float(np.sqrt(np.mean(seg * seg))) >= audible_bar_floor

    if use_neural_grid:
        while float(db[0]) - median_bar >= 0.0:
            db = np.insert(db, 0, float(db[0]) - median_bar)
        while float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)
        if len(syl_arr) >= 1:
            first_vocal = float(syl_arr[0])
            if first_vocal > float(db[0]) + (pre_bars + 0.2) * median_bar:
                anticipation_tol = min(0.5 * spb_q, 0.2 * median_bar)
                vocal_bar_idx = 0
                for idx in range(len(db)):
                    if float(db[idx]) <= first_vocal + anticipation_tol and float(db[idx]) < duration - 1e-3:
                        vocal_bar_idx = idx
                    else:
                        break
                start_bar_idx = max(0, vocal_bar_idx - pre_bars)
                while start_bar_idx < max(0, vocal_bar_idx - 1) and not _bar_is_audible(float(db[start_bar_idx]), float(db[start_bar_idx + 1])):
                    start_bar_idx += 1
                db = db[start_bar_idx:]
        while len(db) < 3 or float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)
        beat_origin = round(min(float(db[0]), max(0.0, duration - 1e-3)), 6)
        db[0] = beat_origin
        if len(db) >= 2:
            median_bar = float(np.median(np.diff(db)))
    else:
        n_pre_bars = int(first_downbeat // bar_dur_init) if bar_dur_init > 1e-6 else 0
        phase_origin = min(max(0.0, first_downbeat - n_pre_bars * bar_dur_init), max(0.0, duration - 1e-3))
        if len(syl_arr) >= 1 and bar_dur_init > 1e-6:
            first_vocal = float(syl_arr[0])
            if first_vocal > phase_origin + (pre_bars + 0.2) * bar_dur_init:
                anticipation_tol = min(0.5 * spb_q, 0.2 * bar_dur_init)
                max_valid_bar_idx = max(0, int(math.ceil((duration - 1e-3 - phase_origin) / bar_dur_init)) - 1)
                vocal_bar_idx = min(
                    max_valid_bar_idx,
                    int(math.floor((first_vocal + anticipation_tol - phase_origin) / bar_dur_init)),
                )
                start_bar_idx = max(0, vocal_bar_idx - pre_bars)
                while start_bar_idx < max(0, vocal_bar_idx - 1) and not _bar_is_audible(
                    phase_origin + start_bar_idx * bar_dur_init,
                    phase_origin + (start_bar_idx + 1) * bar_dur_init,
                ):
                    start_bar_idx += 1
                beat_origin = round(min(phase_origin + start_bar_idx * bar_dur_init, max(0.0, duration - 1e-3)), 6)
            else:
                beat_origin = round(phase_origin, 6)
        else:
            beat_origin = round(phase_origin, 6)

    use_neural_grid = has_neural_downbeats and len(db) >= 3

    # Downbeat-Locked quarter-note beat_times & bar_start_seconds Construction
    if use_neural_grid:
        # Extrapolate db forward until db[-1] >= duration
        while float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)

        bpm = float(60.0 * bar_q_beats / float(np.median(np.diff(db))))
        spb_q = 60.0 / bpm

        if is_int_bar_q:
            beat_times_list: list[float] = []
            for b in range(len(db) - 1):
                for t in _bar_beat_positions(
                    beat_arr, float(db[b]), float(db[b + 1]), bar_grid_beats, is_compound=is_compound_meter
                ):
                    if beat_times_list and t <= beat_times_list[-1] + 1e-4:
                        t = beat_times_list[-1] + 1e-4
                    beat_times_list.append(float(t))
            beat_times_list.append(max(beat_times_list[-1] + 1e-4, float(db[-1])) if beat_times_list else float(db[-1]))
            max_bar_idx = (len(beat_times_list) - 1) // bar_grid_beats
            bar_start_seconds = [float(beat_times_list[b * bar_grid_beats]) for b in range(max_bar_idx + 1)]
        else:
            n_bars_db = len(db) - 1
            total_q_beats = n_bars_db * bar_q_beats
            n_q_grid = max(1, int(math.ceil(total_q_beats)))
            anchor_q: list[float] = [0.0]
            anchor_t: list[float] = [float(db[0])]
            for b in range(n_bars_db):
                b_q_start = b * bar_q_beats
                b_q_end = (b + 1) * bar_q_beats
                db_s, db_e = float(db[b]), float(db[b + 1])
                local_spb_q = (db_e - db_s) / max(1e-6, bar_q_beats)
                tol_q = 0.25 * local_spb_q
                inside_q = beat_arr[(beat_arr > db_s + 1e-4) & (beat_arr < db_e - 1e-4)]
                k_lo = int(math.floor(b_q_start)) + 1
                k_hi = int(math.ceil(b_q_end))
                for k_q in range(k_lo, k_hi):
                    expected_t = db_s + (k_q - b_q_start) * local_spb_q
                    if len(inside_q):
                        j_q = int(np.argmin(np.abs(inside_q - expected_t)))
                        cand_t = float(inside_q[j_q])
                        if abs(cand_t - expected_t) <= tol_q and cand_t > anchor_t[-1] + 1e-4 and cand_t < db_e - 1e-4:
                            anchor_q.append(float(k_q))
                            anchor_t.append(cand_t)
                if db_e > anchor_t[-1] + 1e-4:
                    anchor_q.append(float(b_q_end))
                    anchor_t.append(db_e)
            if n_q_grid > anchor_q[-1]:
                ext_q = float(n_q_grid) - anchor_q[-1]
                anchor_q.append(float(n_q_grid))
                anchor_t.append(anchor_t[-1] + ext_q * spb_q)
            bt_arr = np.interp(np.arange(n_q_grid + 1, dtype=np.float64), anchor_q, anchor_t)
            for b in range(1, n_bars_db + 1):
                b_q = b * bar_q_beats
                k = int(math.floor(b_q))
                alpha = b_q - k
                if 1e-4 < alpha < 1.0 - 1e-4 and k + 1 <= n_q_grid:
                    target_t = float(db[b])
                    lo_bound = float(bt_arr[k - 1]) if k >= 1 else (float(db[0]) - spb_q)
                    hi_bound = float(bt_arr[k + 2]) if (k + 2) <= n_q_grid else (float(bt_arr[-1]) + spb_q)
                    h_init = max(1e-3, float(bt_arr[k + 1] - bt_arr[k]))
                    h_max_lo = max(1e-4, 0.9 * (target_t - lo_bound) / alpha)
                    h_max_hi = max(1e-4, 0.9 * (hi_bound - target_t) / (1.0 - alpha))
                    h = min(h_init, h_max_lo, h_max_hi)
                    bt_arr[k] = target_t - alpha * h
                    bt_arr[k + 1] = target_t + (1.0 - alpha) * h
            beat_times_list = [float(t) for t in bt_arr]
            bar_start_seconds = [
                beats_to_seconds_piecewise(b * bar_q_beats, beat_times_list, bpm, beat_origin)
                for b in range(n_bars_db + 1)
            ]
        db = np.asarray([t for t in db if 0.0 <= float(t) < duration], dtype=np.float64)
        downbeats = db
    else:
        # Fallback: Kalman tempo-phase filter + RTS smoother in quarter-note beat units
        if has_neural_downbeats:
            beats = folded_beats
            beat_arr = folded_beats
        downbeat_arr = db if has_neural_downbeats else np.empty(0, dtype=np.float64)
        eff_lo_bpm = min(lo_bpm, bpm * 0.95)
        eff_hi_bpm = max(hi_bpm, bpm * 1.05)
        beat_times_list, cur_spb_q = _kalman_rts_beat_grid(
            beat_origin, spb_q, duration, bar_q_beats, downbeat_arr, beat_arr, syl_arr,
            eff_lo_bpm, eff_hi_bpm,
        )
        if cur_spb_q > 0:
            bpm = float(60.0 / cur_spb_q)

        # Build bar_start_seconds for fallback path (extending beat_times_list to cover duration)
        while beat_times_list[-1] < duration:
            beat_times_list.append(beat_times_list[-1] + cur_spb_q)
        n_beat_times = len(beat_times_list)
        n_bars = max(1, math.ceil((n_beat_times - 1) / max(1e-6, bar_q_beats)))
        needed = int(math.ceil(n_bars * bar_q_beats)) + 1
        while len(beat_times_list) < needed:
            beat_times_list.append(beat_times_list[-1] + cur_spb_q)
        if is_int_bar_q:
            bar_start_seconds = [beat_times_list[b * bar_grid_beats] for b in range(n_bars + 1)]
        else:
            bar_start_seconds = [
                beats_to_seconds_piecewise(b * bar_q_beats, beat_times_list, bpm, beat_origin)
                for b in range(n_bars + 1)
            ]
        db = np.asarray([t for t in bar_start_seconds if 0.0 <= t < duration], dtype=np.float64)
        downbeats = db

    return {
        "bpm": float(bpm),
        "bpm_in_range": bool(lo_bpm <= bpm <= hi_bpm),
        "beat_origin": beat_origin,
        "meter": f"{measure_beats}/{meter_den}",
        "beats": [float(t) for t in beats],
        "downbeats": [float(t) for t in downbeats if 0.0 <= float(t) < duration],
        "backend": backend,
        "beat_times": beat_times_list,
        "bar_start_seconds": bar_start_seconds,
    }
