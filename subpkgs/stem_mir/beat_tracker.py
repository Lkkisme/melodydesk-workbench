"""SOTA Beat & Downbeat Tracker wrapping CPJKU/beat_this (ISMIR SOTA) with librosa fallback."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    while bpm > hi_bpm and len(beats) >= 4:
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
    accent = np.zeros(len(beats), dtype=np.float64)
    for i, time in enumerate(beats):
        frame = int(round((time * sr + n_fft) / hop))
        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
        accent[i] = float(np.max(window)) if window.size else 0.0
    measure_beats, phase, best_score = 4, 0, -np.inf
    if meter_hint in ('3/4', '4/4'):
        measure_beats = 3 if meter_hint == '3/4' else 4
        if backend == "beat_this" and downbeats is not None and len(downbeats) >= 2:
            phase = int(np.argmin(np.abs(beats - downbeats[0]))) % measure_beats
        else:
            for p in range(measure_beats):
                indicator = (np.arange(len(beats)) % measure_beats == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                score = float(a @ t / denominator) if denominator > 0 else 0.0
                if score > best_score:
                    phase, best_score = p, score
    else:
        for m in (4, 3):
            for p in range(m):
                indicator = (np.arange(len(beats)) % m == p).astype(np.float64)
                a, t = accent - accent.mean(), indicator - indicator.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(t)
                score = float(a @ t / denominator) if denominator > 0 else 0.0
                if score > best_score:
                    measure_beats, phase, best_score = m, p, score
    if backend == "beat_this" and downbeats is not None and len(downbeats) >= 2:
        if meter_hint not in ('3/4', '4/4'):
            indices = np.asarray([np.argmin(np.abs(beats - t)) for t in downbeats])
            intervals = float(np.median(np.diff(indices)))
            measure_beats = min((4, 3), key=lambda m: abs(intervals - m))
    else:
        downbeats = beats[phase::measure_beats]

    # Neural Downbeat Sequence Cleaning & Half-Bar Pairing
    spb = 60.0 / bpm
    syl_arr = np.asarray(sorted(vocal_onsets_sec), dtype=np.float64) if vocal_onsets_sec else np.empty(0, dtype=np.float64)
    beat_arr = np.asarray(beats, dtype=np.float64)

    db = np.asarray(downbeats, dtype=np.float64)
    use_neural_grid = backend == "beat_this" and len(db) >= 4

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

        # Fast Cut-Time / Half-Bar Consolidation for 4/4
        if measure_beats == 4 and median_bar < 1.65:
            best_parity, best_parity_score = 0, np.inf
            for par in (0, 1):
                db_cand = db[par::2]
                if len(db_cand) < 2:
                    continue
                if len(syl_arr):
                    dists = np.array([float(np.min(np.abs(db_cand - s))) for s in syl_arr])
                    score = float(np.median(dists))
                else:
                    db_accent = np.zeros(len(db_cand))
                    for ai, t in enumerate(db_cand):
                        frame = int(round((t * sr + n_fft) / hop))
                        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
                        db_accent[ai] = float(np.max(window)) if window.size else 0.0
                    score = -float(np.sum(db_accent))
                if score < best_parity_score:
                    best_parity, best_parity_score = par, score
            db = db[best_parity::2]
            if len(db) >= 2:
                median_bar = float(np.median(np.diff(db)))

    use_neural_grid = backend == "beat_this" and len(db) >= 3

    # 1-Bar Pre-Vocal Pickup Origin
    first_downbeat = float(db[0]) if len(db) else float(downbeats[0] if len(downbeats) else beats[0])
    bar_dur_init = measure_beats * spb

    if use_neural_grid:
        while float(db[0]) - median_bar >= 0.0:
            db = np.insert(db, 0, float(db[0]) - median_bar)
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
        beat_origin = round(float(db[0]), 6)
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
            db_start = float(db[b])
            db_end = float(db[b + 1])
            bar_span = db_end - db_start
            local_spb = bar_span / measure_beats
            interior_beats = beat_arr[(beat_arr > db_start + local_spb * 0.3) & (beat_arr < db_end - local_spb * 0.3)]
            snap_interior = len(interior_beats) == measure_beats - 1
            snap_tol_bar = 0.25 * local_spb
            for k in range(measure_beats):
                t = db_start + k * local_spb
                if snap_interior and k > 0:
                    det_beat = float(interior_beats[k - 1])
                    if abs(det_beat - t) <= snap_tol_bar:
                        candidate = det_beat
                        if (not beat_times_list or candidate > beat_times_list[-1] + 1e-4) and candidate < db_end - 1e-4:
                            t = candidate
                if beat_times_list and t <= beat_times_list[-1] + 1e-4:
                    t = beat_times_list[-1] + 1e-4
                beat_times_list.append(t)
        beat_times_list.append(max(beat_times_list[-1] + 1e-4, float(db[-1])) if beat_times_list else float(db[-1]))

        max_bar_idx = (len(beat_times_list) - 1) // measure_beats
        bar_start_seconds = [float(beat_times_list[b * measure_beats]) for b in range(max_bar_idx + 1)]
    else:
        # Fallback: adaptive snap grid (librosa backend or sparse downbeats)
        downbeat_arr = np.asarray(downbeats, dtype=np.float64)
        snap_tol = 0.35 * spb
        syl_tol = 0.24 * spb
        beat_times_list = [beat_origin]
        cur_spb = spb
        pos = beat_origin
        while pos + cur_spb * 0.5 < duration:
            next_pos = pos + cur_spb
            snapped = False
            beat_idx = len(beat_times_list)
            if beat_idx % measure_beats == 0 and len(downbeat_arr):
                db_cands = downbeat_arr[(downbeat_arr > pos + cur_spb * 0.35) & (downbeat_arr < pos + cur_spb * 1.65)]
                if len(db_cands):
                    closest_db = db_cands[np.argmin(np.abs(db_cands - next_pos))]
                    if abs(closest_db - next_pos) <= 0.45 * spb:
                        next_pos = float(closest_db)
                        cur_spb = float(np.clip(0.75 * cur_spb + 0.25 * (next_pos - pos), 60.0 / hi_bpm, 60.0 / lo_bpm))
                        snapped = True
            if not snapped:
                candidates = beat_arr[(beat_arr > pos + cur_spb * 0.3) & (beat_arr < pos + cur_spb * 1.7)]
                if len(candidates):
                    closest = candidates[np.argmin(np.abs(candidates - next_pos))]
                    if abs(closest - next_pos) <= snap_tol:
                        next_pos = float(closest)
                        step = next_pos - pos
                        cur_spb = float(np.clip(0.75 * cur_spb + 0.25 * step, 60.0 / hi_bpm, 60.0 / lo_bpm))
                        snapped = True
            if not snapped and len(syl_arr):
                syl_cands = syl_arr[(syl_arr > pos + cur_spb * 0.4) & (syl_arr < pos + cur_spb * 1.6)]
                if len(syl_cands):
                    closest_syl = syl_cands[np.argmin(np.abs(syl_cands - next_pos))]
                    if abs(closest_syl - next_pos) <= syl_tol:
                        next_pos = float(closest_syl)
                        step = next_pos - pos
                        cur_spb = float(np.clip(0.82 * cur_spb + 0.18 * step, 60.0 / hi_bpm, 60.0 / lo_bpm))
            if next_pos <= pos + 1e-4:
                next_pos = pos + cur_spb
            pos = next_pos
            beat_times_list.append(pos)

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
        "meter": f"{measure_beats}/4",
        "beats": [float(t) for t in beats],
        "downbeats": [float(t) for t in downbeats],
        "backend": backend,
        "beat_times": beat_times_list,
        "bar_start_seconds": bar_start_seconds,
    }
