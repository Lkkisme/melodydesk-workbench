"""GPU Singing Voice Transcription Engine (RMVPE + ROSVOT with fixed 0.5 note_bd_threshold & word_bd joint boundary regulation)."""
from __future__ import annotations
from pathlib import Path
import math
import sys

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from music import performance_timing
import pitch_notes
import singing_model


def transcribe_vocal_notes(
    vocal_path: str | Path,
    bpm: float = 120.0,
    beat_origin: float = 0.0,
    word_boundaries_sec: list[float] | None = None,
    prefer_cuda: bool = True,
) -> dict:
    """Transcribe lead vocal audio into Note objects + detected word boundaries.

    Fixes upstream ROSVOT `note_bd_threshold` from 0.85 to 0.5, runs on CUDA when available,
    and optionally injects external `word_boundaries_sec` (from Qwen3-ForcedAligner / Whisper)
    to regulate note onset/offset boundaries.
    """
    import librosa
    import torch

    vocal_path = Path(vocal_path)
    if not vocal_path.is_file():
        raise FileNotFoundError(vocal_path)
    bpm, beat_origin = float(bpm), float(beat_origin)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if not math.isfinite(beat_origin):
        raise ValueError("beat_origin must be finite")
    boundaries = None
    if word_boundaries_sec is not None:
        boundaries = [float(value) for value in word_boundaries_sec]
        if any(not math.isfinite(value) or value < 0 for value in boundaries):
            raise ValueError("Word boundaries must be finite, nonnegative seconds")
        boundaries = sorted(set(boundaries))
    y, sr = librosa.load(str(vocal_path), sr=22050, mono=True)
    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    returned_boundaries = [] if boundaries is None else list(boundaries)
    backend_used, device_used = "pyin", "cpu"
    notes = []
    if singing_model.ready():
        result = singing_model.track_singing(
            y, sr, lambda: None, lambda p, s: None, device=device,
            word_boundaries_sec=boundaries, return_dict=True,
        )
        notes = performance_timing(result["notes"], bpm, beat_origin)
        returned_boundaries = result["word_boundaries_sec"]
        backend_used, device_used = result["backend"], result["device"]
    else:
        notes = pitch_notes.extract_vocal_notes(str(vocal_path), bpm, beat_origin)
    return {"notes": notes, "word_boundaries_sec": returned_boundaries,
            "backend": backend_used, "device": device_used}
