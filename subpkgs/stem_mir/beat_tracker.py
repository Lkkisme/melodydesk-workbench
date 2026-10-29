"""SOTA Beat & Downbeat Tracker wrapping CPJKU/beat_this (ISMIR SOTA) with librosa fallback."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _timestamps(values, duration):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite beat timestamps")
    return np.unique(values[(values >= 0) & (values < duration)])


def track_beats_and_downbeats(
    audio_path: str | Path,
    prefer_cuda: bool = True,
) -> dict:
    """Track beats, downbeats, meter, bpm, and beat_origin from audio.

    Returns dict:
        - `bpm`: float (60.0 to 180.0 normalized)
        - `beat_origin`: float (first downbeat or earliest beat in seconds >= 0.0)
        - `meter`: str ('4/4' or '3/4')
        - `beats`: list[float]
        - `downbeats`: list[float]
        - `backend`: str ('beat_this' or 'librosa')
    """
    import librosa
    from .separator_engine import resolve_device

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
    bpm = 60.0 / float(np.median(np.diff(beats)))
    while bpm < 60.0:
        beats = np.sort(np.concatenate((beats, (beats[:-1] + beats[1:]) / 2)))
        bpm *= 2.0
    while bpm > 180.0:
        parity = 0
        if downbeats is not None and len(downbeats):
            parity = int(np.argmin(np.abs(beats - downbeats[0]))) % 2
        beats = beats[parity::2]
        bpm /= 2.0
    accent = np.zeros(len(beats), dtype=np.float64)
    for i, time in enumerate(beats):
        frame = int(round((time * sr + n_fft) / hop))
        window = onset_env[max(0, frame - 1):min(len(onset_env), frame + 2)]
        accent[i] = float(np.max(window)) if window.size else 0.0
    measure_beats, phase, best_score = 4, 0, -np.inf
    for m in (4, 3):
        for p in range(m):
            indicator = (np.arange(len(beats)) % m == p).astype(np.float64)
            a, t = accent - accent.mean(), indicator - indicator.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(t)
            score = float(a @ t / denominator) if denominator > 0 else 0.0
            if score > best_score:
                measure_beats, phase, best_score = m, p, score
    if backend == "beat_this" and len(downbeats) >= 2:
        indices = np.asarray([np.argmin(np.abs(beats - t)) for t in downbeats])
        intervals = float(np.median(np.diff(indices)))
        measure_beats = min((4, 3), key=lambda m: abs(intervals - m))
    else:
        downbeats = beats[phase::measure_beats]
    return {
        "bpm": float(bpm),
        "beat_origin": float(downbeats[0] if len(downbeats) else beats[0]),
        "meter": f"{measure_beats}/4",
        "beats": [float(t) for t in beats],
        "downbeats": [float(t) for t in downbeats],
        "backend": backend,
    }
