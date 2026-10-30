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
    if bpm < lo_bpm or bpm > hi_bpm or vocal_onsets_sec is not None:
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

    # Build piecewise beat_times grid anchored to the singer's vocal downbeats & syllable onsets
    spb = 60.0 / bpm
    first_downbeat = float(downbeats[0] if len(downbeats) else beats[0])
    bar_dur_init = measure_beats * spb
    if first_downbeat >= bar_dur_init:
        n_pre_bars = int(first_downbeat // bar_dur_init)
        beat_origin = round(max(0.0, first_downbeat - n_pre_bars * bar_dur_init), 6)
    else:
        beat_origin = first_downbeat
    snap_tol = 0.35 * spb
    syl_tol = 0.24 * spb
    beat_arr = np.asarray(beats, dtype=np.float64)
    syl_arr = np.asarray(sorted(vocal_onsets_sec), dtype=np.float64) if vocal_onsets_sec else np.empty(0, dtype=np.float64)
    beat_times_list: list[float] = [beat_origin]
    cur_spb = spb
    pos = beat_origin
    while pos + cur_spb * 0.5 < duration:
        next_pos = pos + cur_spb
        snapped = False
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
        pos = next_pos
        beat_times_list.append(pos)

    # Build bar_start_seconds
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
