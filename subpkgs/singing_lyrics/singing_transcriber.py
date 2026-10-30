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
    beat_times: list[float] | None = None,
) -> dict:
    """Transcribe lead vocal audio into Note objects + detected word boundaries.

    Fixes upstream ROSVOT `note_bd_threshold` from 0.85 to 0.5, runs on CUDA when available,
    and optionally injects external `word_boundaries_sec` (from Qwen3-ForcedAligner / Whisper)
    to regulate note onset/offset boundaries.

    When `beat_times` is provided, uses piecewise beat mapping for tempo-varying scores.
    """
    import librosa
    import torch
    from subpkgs.stem_mir.beat_tracker import seconds_to_beats_piecewise

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
        if beat_times is not None:
            from music import Note
            raw_notes = result["notes"]
            converted = []
            for n in raw_notes:
                raw_s = n.raw_start
                raw_e = n.raw_start + n.raw_duration
                start_beat = seconds_to_beats_piecewise(raw_s, beat_times, bpm, beat_origin)
                end_beat = seconds_to_beats_piecewise(raw_e, beat_times, bpm, beat_origin)
                dur_beat = max(0.0, end_beat - start_beat)
                if dur_beat > 0:
                    converted.append(Note(n.pitch, max(0.0, start_beat), dur_beat,
                                          n.raw_start, n.raw_duration, n.confidence, n.edited))
            notes = converted
        else:
            notes = performance_timing(result["notes"], bpm, beat_origin)
        returned_boundaries = result["word_boundaries_sec"]
        backend_used, device_used = result["backend"], result["device"]
    else:
        notes = pitch_notes.extract_vocal_notes(str(vocal_path), bpm, beat_origin)
    return {"notes": notes, "word_boundaries_sec": returned_boundaries,
            "backend": backend_used, "device": device_used}


def regulate_singing_notes_for_score(
    notes: list,
    bpm: float,
    beat_origin: float,
    clip_duration: float = 0.0,
    word_boundaries_sec: list[float] | None = None,
    min_grid_beat: float = 0.25,
    beat_times: list[float] | None = None,
) -> list:
    """Clean raw Note objects for score layout: merge micro-glitches, snap to
    grid, organize phrases, and synchronize raw timing.

    When `beat_times` has >= 2 monotonically increasing timestamps and notes have
    valid raw timing, uses piecewise beat mapping for tempo-varying scores.
    """
    import copy
    import numpy as np
    import score_cleanup
    from subpkgs.stem_mir.beat_tracker import seconds_to_beats_piecewise, beats_to_seconds_piecewise

    bpm = float(bpm)
    beat_origin = float(beat_origin)
    spb = 60.0 / bpm
    grid = float(min_grid_beat)

    use_piecewise = False
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=np.float64).ravel()
        if len(bt) >= 2 and np.all(np.diff(bt) > 0):
            use_piecewise = True

    ordered = sorted(copy.deepcopy(notes), key=lambda n: n.raw_start)
    filtered = [n for n in ordered if n.raw_start >= beat_origin and n.raw_duration >= 0.075]
    if not filtered:
        return []

    merged: list = []
    for n in filtered:
        if (merged
                and (n.raw_start - (merged[-1].raw_start + merged[-1].raw_duration)) < 0.05
                and abs(n.pitch - merged[-1].pitch) <= 1
                and merged[-1].raw_duration < 0.16):
            prev_dur = merged[-1].raw_duration
            merged[-1].raw_duration = (n.raw_start + n.raw_duration) - merged[-1].raw_start
            if n.raw_duration > prev_dur:
                merged[-1].pitch = n.pitch
        else:
            merged.append(n)

    for n in merged:
        if use_piecewise and n.raw_start >= 0 and n.raw_duration > 0:
            beats_from_origin = seconds_to_beats_piecewise(n.raw_start, beat_times, bpm, beat_origin)
        else:
            beats_from_origin = (n.raw_start - beat_origin) / spb
        n.start = max(0.0, round(round(beats_from_origin / grid) * grid, 10))

    deduped: list = []
    for n in merged:
        if deduped and abs(deduped[-1].start - n.start) < 1e-9:
            if n.raw_duration > deduped[-1].raw_duration:
                deduped[-1] = n
        else:
            deduped.append(n)

    for i, n in enumerate(deduped):
        if use_piecewise and n.raw_start >= 0 and n.raw_duration > 0:
            raw_end_beat = seconds_to_beats_piecewise(n.raw_start + n.raw_duration, beat_times, bpm, beat_origin)
            raw_dur_beats = min(4.0, max(grid, round(round((raw_end_beat - n.start) / grid) * grid, 10)))
        else:
            raw_dur_beats = min(4.0, max(grid, round(round((n.raw_duration / spb) / grid) * grid, 10)))
        if i + 1 < len(deduped):
            gap_beats = deduped[i + 1].start - n.start
            n.duration = max(grid, min(raw_dur_beats if gap_beats - raw_dur_beats > 0.25 else gap_beats, gap_beats))
        else:
            n.duration = raw_dur_beats

    phrases: list[list] = []
    for n in deduped:
        if not phrases or (n.raw_start - (phrases[-1][-1].raw_start + phrases[-1][-1].raw_duration)) > 0.45:
            phrases.append([])
        phrases[-1].append(n)

    if use_piecewise:
        for n in deduped:
            n.raw_start = beat_origin + n.start * spb
            n.raw_duration = n.duration * spb

    raw_result: list = []
    for phrase in phrases:
        try:
            organized = score_cleanup.organize(phrase, bpm, beat_origin, 'auto')
            raw_result.extend(organized)
        except ValueError:
            raw_result.extend(phrase)

    raw_result.sort(key=lambda x: x.start)
    result: list = []
    for n in raw_result:
        if result and n.start < result[-1].start + grid - 1e-6:
            if n.duration > result[-1].duration:
                n.start = result[-1].start
                result[-1] = n
            continue
        result.append(n)

    if clip_duration > beat_origin:
        max_beats = max(grid, (clip_duration - beat_origin) / spb)
        result = [n for n in result if n.start < max_beats - grid]

    cleaned_result: list = []
    for i, n in enumerate(result):
        if i + 1 < len(result):
            max_dur = result[i + 1].start - n.start
            if max_dur <= 1e-6:
                continue
            if n.duration > max_dur:
                n.duration = max_dur
        n.duration = max(1e-4, float(n.duration))
        if use_piecewise:
            n.raw_start = round(beats_to_seconds_piecewise(n.start, beat_times, bpm, beat_origin), 4)
            n.raw_duration = round(beats_to_seconds_piecewise(n.start + n.duration, beat_times, bpm, beat_origin) - n.raw_start, 4)
        else:
            n.raw_start = round(beat_origin + n.start * spb, 4)
            n.raw_duration = round(n.duration * spb, 4)
        cleaned_result.append(n)

    return cleaned_result
