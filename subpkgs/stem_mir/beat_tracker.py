"""Beat & downbeat tracker wrapping CPJKU/beat_this with a librosa fallback.

Pipeline:
  1. Neural tier (``beat_this.inference.File2Beats``) or librosa ``beat_track``.
  2. Metrical-level / tactus disambiguation for the neural downbeat sequence via an
     onset-envelope autocorrelation likelihood, a vocal syllable inter-onset-interval
     (IOI) log-likelihood and a log-normal tempo-range prior.
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

# Empirical ratio of the median sung-syllable IOI to the tactus period (calibrated on the
# repository's vocal repertoire); used as the centre of the vocal-IOI log-likelihood.
_SYLLABLE_IOI_PER_BEAT = 0.68
# Standard deviation (in octaves) of the vocal-IOI log-likelihood.
_SYLLABLE_IOI_SIGMA_OCT = 1.05
# Kalman observation noise standard deviations, as fractions of the beat period.
# Ordering R_downbeat < R_beat < R_vocal reflects decreasing reliability of the cue.
_R_DOWNBEAT_FRAC = 0.03
_R_BEAT_FRAC = 0.05
_R_VOCAL_FRAC = 0.09
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
# Onset-autocorrelation & vocal-IOI log-likelihood meter/tactus disambiguator
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
    if centre < 1 or centre >= len(acf) - 1:
        return 0.0
    lo, hi = max(1, centre - halfwidth), min(len(acf), centre + halfwidth + 1)
    return float(np.max(acf[lo:hi]))


def _tempo_range_log_prior(bpm: float, lo_bpm: float, hi_bpm: float) -> float:
    """Quadratic log2-BPM penalty centred on the geometric mean of [lo_bpm, hi_bpm] (sigma = range/4 octaves)."""
    if bpm <= 0.0 or bpm < 0.95 * lo_bpm or bpm > 1.05 * hi_bpm:
        return -np.inf
    centre = math.sqrt(lo_bpm * hi_bpm)
    sigma_oct = max(0.1, math.log2(hi_bpm / lo_bpm) / 4.0)
    return -0.5 * (math.log2(bpm / centre) / sigma_oct) ** 2


def _syllable_ioi_log_likelihood(syl_arr: np.ndarray, spb: float) -> float:
    """Quadratic log2-deviation score between the median vocal syllable IOI and the expected beat-scaled syllable IOI."""
    if len(syl_arr) < 4:
        return 0.0
    iois = np.diff(syl_arr)
    iois = iois[(iois >= 0.16) & (iois <= 2.4)]
    if len(iois) < 2:
        return 0.0
    med_ioi = float(np.median(iois))
    target = _SYLLABLE_IOI_PER_BEAT * spb
    z = math.log2(max(1e-3, med_ioi) / max(1e-3, target)) / _SYLLABLE_IOI_SIGMA_OCT
    return -0.5 * z * z


def _metrical_level_log_posterior(
    acf: np.ndarray,
    frames_per_sec: float,
    bar_sec: float,
    measure_beats: int,
    syl_arr: np.ndarray,
    lo_bpm: float,
    hi_bpm: float,
) -> float:
    """Additive log-domain metrical score combining onset autocorrelation at beat/bar lags, syllable-IOI alignment, and log-tempo range preference."""
    spb = bar_sec / measure_beats
    bpm = 60.0 / max(1e-6, spb)
    log_prior = _tempo_range_log_prior(bpm, lo_bpm, hi_bpm)
    if not np.isfinite(log_prior):
        return -np.inf
    acf_beat = max(1e-3, _acf_at(acf, spb * frames_per_sec))
    acf_bar = max(1e-3, _acf_at(acf, bar_sec * frames_per_sec))
    log_acf = math.log(acf_beat) + math.log(acf_bar)
    return log_acf + _syllable_ioi_log_likelihood(syl_arr, spb) + log_prior


# ---------------------------------------------------------------------------
# Neural-tier interior beat placement
# ---------------------------------------------------------------------------

def _bar_beat_positions(
    beat_arr: np.ndarray, db_start: float, db_end: float, measure_beats: int
) -> list[float]:
    """Beat positions k = 0..measure_beats-1 for one bar spanning [db_start, db_end).

    * Exactly ``measure_beats - 1`` strictly increasing neural interior beats -> keep them verbatim.
    * Otherwise snap each expected subdivision to the nearest neural beat within 0.25 of a local
      beat period, and fill unmatched subdivisions by monotonic piecewise-linear interpolation
      between the matched anchors (bar boundaries included).
    """
    bar_span = db_end - db_start
    local_spb = bar_span / measure_beats
    interior = beat_arr[(beat_arr > db_start + 0.3 * local_spb) & (beat_arr < db_end - 0.3 * local_spb)]
    if measure_beats == 1:
        return [db_start]
    if len(interior) == measure_beats - 1 and np.all(np.diff(interior) > 0):
        return [db_start] + [float(t) for t in interior]
    inside = beat_arr[(beat_arr > db_start + 1e-4) & (beat_arr < db_end - 1e-4)]
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
    measure_beats: int,
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
    downbeat_gate, beat_gate, vocal_gate = 0.45 * spb, 0.35 * spb, 0.24 * spb

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
        if beat_idx % measure_beats == 0 and len(downbeat_arr):
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
) -> dict:
    """Track beats, downbeats, meter, bpm, and beat_origin from audio.

    Returns dict:
        - `bpm`: float (normalized to bpm_range)
        - `beat_origin`: float (first downbeat or earliest beat in seconds >= 0.0)
        - `meter`: str ('4/4' or '3/4')
        - `beats`: list[float]
        - `downbeats`: list[float]
        - `backend`: str ('beat_this' or 'librosa')
        - `beat_times`: list[float] — piecewise beat grid snapped to detected beats
        - `bar_start_seconds`: list[float] — bar boundary timestamps
    """
    import librosa
    from .separator_engine import resolve_device

    lo_bpm, hi_bpm = float(bpm_range[0]), float(bpm_range[1])

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if not y.size or not np.all(np.isfinite(y)):
        raise ValueError("Empty or nonfinite audio")
    duration = len(y) / sr
    beats, downbeats, backend = None, None, "librosa"
    try:
        from beat_this.inference import File2Beats

        tracker = File2Beats(device=resolve_device(prefer_cuda))
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
    raw_diffs = np.diff(beats)
    bpm = 60.0 / float(np.median(raw_diffs))
    while bpm < lo_bpm:
        beats = np.sort(np.concatenate((beats, (beats[:-1] + beats[1:]) / 2)))
        bpm *= 2.0
    while bpm > hi_bpm:
        if len(beats) >= 4:
            parity = 0
            if downbeats is not None and len(downbeats):
                parity = int(np.argmin(np.abs(beats - downbeats[0]))) % 2
            beats = beats[parity::2]
        bpm /= 2.0
    # When tracking a pure vocal stem (with held vowels and breath pauses), fold multi-beat IOIs (k=1..4) into [lo_bpm, hi_bpm]
    high_cv = len(raw_diffs) >= 4 and float(np.std(raw_diffs) / max(1e-6, np.mean(raw_diffs))) > 0.25
    if bpm < lo_bpm or bpm > hi_bpm or vocal_onsets_sec is not None or high_cv:
        folded_spbs = []
        for d in raw_diffs:
            for mult in (1, 2, 3, 4):
                cand_bpm = 60.0 / (d / mult) if d > 0 else 0.0
                if lo_bpm <= cand_bpm <= hi_bpm:
                    folded_spbs.append(float(d / mult))
                    break
        if len(folded_spbs) >= 4:
            bpm = 60.0 / float(np.median(folded_spbs))
    bpm = float(np.clip(bpm, lo_bpm, hi_bpm))
    accent = np.zeros(len(beats), dtype=np.float64)
    for i, time in enumerate(beats):
        frame = int(round((time * sr + n_fft) / hop))
        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
        accent[i] = float(np.max(window)) if window.size else 0.0
    measure_beats, meter_den, phase, peak_corr = 4, 4, 0, -np.inf
    has_meter_hint = meter_hint is not None and str(meter_hint).strip() != ""
    if has_meter_hint:
        try:
            m_num, m_den = map(int, str(meter_hint).strip().split("/"))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid meter_hint: {meter_hint!r}") from exc
        if m_num <= 0 or m_den not in (1, 2, 4, 8, 16):
            raise ValueError(f"Invalid meter_hint: {meter_hint!r}")
        measure_beats, meter_den = m_num, m_den
        if backend == "beat_this" and downbeats is not None and len(downbeats) >= 2:
            phase = int(np.argmin(np.abs(beats - downbeats[0]))) % measure_beats
        else:
            for p in range(measure_beats):
                indicator = (np.arange(len(beats)) % measure_beats == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                corr = float(a @ t / denominator) if denominator > 0 else 0.0
                if corr > peak_corr:
                    phase, peak_corr = p, corr
    else:
        for m in (4, 3):
            for p in range(m):
                indicator = (np.arange(len(beats)) % m == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                corr = float(a @ t / denominator) if denominator > 0 else 0.0
                if corr > peak_corr:
                    measure_beats, phase, peak_corr = m, p, corr
    if backend == "beat_this" and downbeats is not None and len(downbeats) >= 2:
        if not has_meter_hint:
            indices = np.asarray([np.argmin(np.abs(beats - t)) for t in downbeats])
            intervals = float(np.median(np.diff(indices)))
            measure_beats = min((4, 3), key=lambda m: abs(intervals - m))
    else:
        downbeats = beats[phase::measure_beats]

    # Neural downbeat sequence cleaning & half-bar pairing
    spb = 60.0 / bpm
    syl_arr = np.asarray(sorted(vocal_onsets_sec), dtype=np.float64) if vocal_onsets_sec else np.empty(0, dtype=np.float64)
    beat_arr = np.asarray(beats, dtype=np.float64)

    db = np.asarray(downbeats, dtype=np.float64) if downbeats is not None else np.empty(0, dtype=np.float64)
    median_bar = float(np.median(np.diff(db))) if len(db) >= 2 else float(measure_beats * spb)
    use_neural_grid = backend == "beat_this" and len(db) >= 3

    if use_neural_grid:
        median_bar = float(np.median(np.diff(db)))
        # Clean spurious duplicate downbeats
        cleaned = [float(db[0])]
        for i in range(1, len(db)):
            if float(db[i]) - cleaned[-1] >= 0.55 * median_bar:
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

        # Onset-autocorrelation & vocal-IOI log-likelihood meter/tactus disambiguator.
        # Hypothesis m in {1, 2}: the neural "bar" is one bar (m=1) or half a bar (m=2).
        acf = _onset_autocorrelation(onset_env)
        log_post_by_m: dict[int, float] = {}
        for m in (1, 2):
            if m == 2 and len(db) < 6:
                continue
            log_post_by_m[m] = _metrical_level_log_posterior(
                acf, frames_per_sec, m * median_bar, measure_beats, syl_arr, lo_bpm, hi_bpm
            )
        chosen_m = max(log_post_by_m, key=lambda m: log_post_by_m[m]) if log_post_by_m else 1
        if chosen_m == 2 and np.isfinite(log_post_by_m[2]):
            # Downbeat phase parity: minimise the median syllable-to-downbeat distance, or, without
            # vocal onsets, maximise the summed onset accent on the retained downbeats.
            chosen_parity, min_parity_loss = 0, np.inf
            for par in (0, 1):
                db_cand = db[par::2]
                if len(db_cand) < 2:
                    continue
                if len(syl_arr):
                    dists = np.array([float(np.min(np.abs(db_cand - s))) for s in syl_arr])
                    loss = float(np.median(dists))
                else:
                    db_accent = np.zeros(len(db_cand))
                    for ai, t in enumerate(db_cand):
                        frame = int(round((t * sr + n_fft) / hop))
                        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
                        db_accent[ai] = float(np.max(window)) if window.size else 0.0
                    loss = -float(np.sum(db_accent))
                if loss < min_parity_loss:
                    chosen_parity, min_parity_loss = par, loss
            beat_arr = np.unique(np.sort(np.concatenate([beat_arr, db[1 - chosen_parity::2]])))
            db = db[chosen_parity::2]
            if len(db) >= 2:
                median_bar = float(np.median(np.diff(db)))

    use_neural_grid = backend == "beat_this" and len(db) >= 3

    # 1-Bar Pre-Vocal Pickup Origin
    first_downbeat = float(db[0]) if len(db) else float(downbeats[0] if len(downbeats) else beats[0])
    bar_dur_init = measure_beats * spb

    if use_neural_grid:
        while float(db[0]) - median_bar >= 0.0:
            db = np.insert(db, 0, float(db[0]) - median_bar)
        while float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)
        if len(syl_arr) >= 1:
            first_vocal = float(syl_arr[0])
            if first_vocal > float(db[0]) + 1.2 * median_bar:
                vocal_bar_idx = 0
                for idx in range(len(db)):
                    if float(db[idx]) <= first_vocal + 0.38:
                        vocal_bar_idx = idx
                    else:
                        break
                start_bar_idx = max(0, vocal_bar_idx - 1)
                db = db[start_bar_idx:]
        while len(db) < 3 or float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)
        beat_origin = round(float(db[0]), 6)
        db[0] = beat_origin
        if len(db) >= 2:
            median_bar = float(np.median(np.diff(db)))
    elif len(syl_arr) >= 1:
        first_vocal = float(syl_arr[0])
        beat_origin = round(max(0.0, first_vocal - bar_dur_init), 6)
    elif first_downbeat >= bar_dur_init:
        n_pre_bars = int(first_downbeat // bar_dur_init)
        beat_origin = round(max(0.0, first_downbeat - n_pre_bars * bar_dur_init), 6)
    else:
        beat_origin = first_downbeat

    use_neural_grid = backend == "beat_this" and len(db) >= 3

    # Downbeat-Locked beat_times & bar_start_seconds Construction
    if use_neural_grid:
        # Extrapolate db forward until db[-1] >= duration
        while float(db[-1]) < duration:
            db = np.append(db, float(db[-1]) + median_bar)

        bpm = float(np.clip(60.0 * measure_beats / float(np.median(np.diff(db))), lo_bpm, hi_bpm))
        spb = 60.0 / bpm

        beat_times_list: list[float] = []
        for b in range(len(db) - 1):
            for t in _bar_beat_positions(beat_arr, float(db[b]), float(db[b + 1]), measure_beats):
                if beat_times_list and t <= beat_times_list[-1] + 1e-4:
                    t = beat_times_list[-1] + 1e-4
                beat_times_list.append(float(t))
        beat_times_list.append(max(beat_times_list[-1] + 1e-4, float(db[-1])) if beat_times_list else float(db[-1]))

        max_bar_idx = (len(beat_times_list) - 1) // measure_beats
        bar_start_seconds = [float(beat_times_list[b * measure_beats]) for b in range(max_bar_idx + 1)]
    else:
        # Fallback: Kalman tempo-phase filter + RTS smoother (librosa backend or sparse downbeats)
        downbeat_arr = np.asarray(downbeats, dtype=np.float64)
        beat_times_list, cur_spb = _kalman_rts_beat_grid(
            beat_origin, spb, duration, measure_beats, downbeat_arr, beat_arr, syl_arr, lo_bpm, hi_bpm
        )
        if cur_spb > 0:
            bpm = float(np.clip(60.0 / cur_spb, lo_bpm, hi_bpm))

        # Build bar_start_seconds for fallback path
        n_beat_times = len(beat_times_list)
        n_bars = max(1, math.ceil((n_beat_times - 1) / measure_beats))
        needed = n_bars * measure_beats + 1
        while len(beat_times_list) < needed:
            beat_times_list.append(beat_times_list[-1] + cur_spb)
        bar_start_seconds = [beat_times_list[b * measure_beats] for b in range(n_bars + 1)]

    return {
        "bpm": float(bpm),
        "beat_origin": beat_origin,
        "meter": f"{measure_beats}/{meter_den}",
        "beats": [float(t) for t in beats],
        "downbeats": [float(t) for t in downbeats],
        "backend": backend,
        "beat_times": beat_times_list,
        "bar_start_seconds": bar_start_seconds,
    }
