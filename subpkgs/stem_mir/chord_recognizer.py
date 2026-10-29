"""Dual-band CQT chord recognition with binary cosine templates and melody extraction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import re
from numbers import Integral

QUALITIES = (
    ("", (0, 4, 7)), ("m", (0, 3, 7)),
    ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)),
    ("7", (0, 4, 7, 10)), ("sus4", (0, 5, 7)),
    ("sus2", (0, 2, 7)), ("add9", (0, 2, 4, 7)),
    ("dim", (0, 3, 6)),
)
_CHORDS = tuple((root, quality, intervals) for root in range(12) for quality, intervals in QUALITIES)
_MASKS = np.zeros((len(_CHORDS), 12), dtype=np.float64)
for _index, (_root, _quality, _intervals) in enumerate(_CHORDS):
    _MASKS[_index, (_root + np.asarray(_intervals, dtype=np.int64)) % 12] = 1.0
_TEMPLATES = _MASKS / np.linalg.norm(_MASKS, axis=1, keepdims=True)


def _timing(bpm, beat_origin, meter):
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be finite and positive")
    if not np.isfinite(beat_origin) or beat_origin < 0:
        raise ValueError("Beat origin must be finite and nonnegative")
    if meter not in ("4/4", "3/4"):
        raise ValueError("Unsupported meter")
    measure_beats = 4.0 if meter == "4/4" else 3.0
    seconds_per_beat = 60.0 / bpm
    return measure_beats, seconds_per_beat, measure_beats * seconds_per_beat


def _load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if not y.size or not np.all(np.isfinite(y)):
        raise ValueError("Empty or nonfinite audio")
    return y, sr


def _pitch_names(tonic):
    if not isinstance(tonic, str) or re.fullmatch(r"[A-G](?:#|b)?", tonic) is None:
        raise ValueError("Malformed tonic")
    return ("C Db D Eb E F Gb G Ab A Bb B" if "b" in tonic or tonic == "F" else "C C# D D# E F F# G G# A A# B").split()


def _segment(start, end, frame_times, bass_cqt, upper_cqt, pitch_names):
    bass_chroma, upper_chroma = np.zeros(12), np.zeros(12)
    symbol, score, bass_pc = "N.C.", 0.0, None
    selected = (frame_times >= start) & (frame_times < end)
    if np.any(selected):
        bass_spectrum = np.median(bass_cqt[:, selected], axis=1)
        upper_spectrum = np.median(upper_cqt[:, selected], axis=1)
        spectrum = np.concatenate((bass_spectrum[:12], upper_spectrum))
        neighbors = np.pad(spectrum, (1, 1))
        peaks = (spectrum > neighbors[:-2]) & (spectrum >= neighbors[2:])
        spectrum = np.where(peaks, spectrum, 0.0)
        bass_chroma = spectrum[:24].reshape(2, 12).sum(axis=0)
        upper_chroma = spectrum[12:].reshape(3, 12).sum(axis=0)
        norm = np.linalg.norm(upper_chroma)
        if norm > 0:
            scores = _TEMPLATES @ (upper_chroma / norm)
            maximum = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, maximum, rtol=0, atol=np.finfo(np.float64).eps))
            chosen = max(tied, key=lambda i: (upper_chroma[_CHORDS[i][0]], -len(_CHORDS[i][2]), -int(i)))
            root, quality, _ = _CHORDS[chosen]
            score = float(scores[chosen])
            bass_pc = int(np.argmax(bass_chroma)) if np.any(bass_chroma > 0) else None
            symbol = pitch_names[root] + quality
            if bass_pc is not None and bass_pc != root:
                symbol += "/" + pitch_names[bass_pc]
    return {"chord": symbol, "score": score, "bass_pc": bass_pc, "bass_chroma": bass_chroma.tolist(), "upper_chroma": upper_chroma.tolist()}


def recognize_accompaniment_chords(
    audio_path: str | Path,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    bar_count: int | None = None,
    tonic: str = "C",
    split_bar: bool = False,
) -> dict:
    """Recognize real accompaniment chords per bar (or per half-bar when split_bar=True).

    Uses bass C2..B3 and upper C3..B5 Constant-Q transforms with binary
    pitch-class cosine scoring and independent slash-bass detection.

    Returns dict:
        - `chords`: list[str] (1 primary chord per bar, e.g. ['C', 'G/B', 'Am7', 'Fmaj7'])
        - `split_chords`: list[list[dict]] (per bar list of segments:
              if split_bar=False: `[{'beat': 0.0, 'duration': 4.0, 'chord': 'C'}]`
              if split_bar=True:  `[{'beat': 0.0, 'duration': 2.0, 'chord': 'Fmaj7'}, {'beat': 2.0, 'duration': 2.0, 'chord': 'Fm6'}]`
          )
        - `bar_details`: list[dict]
    """
    import librosa

    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    if bar_count is not None and (isinstance(bar_count, bool) or not isinstance(bar_count, Integral) or bar_count < 0):
        raise ValueError("Bar count must be a nonnegative integer")
    pitch_names = _pitch_names(tonic)
    y, sr = _load_audio(audio_path)
    duration = len(y) / sr
    if bar_count is None:
        bar_count = int(np.ceil(max(0.0, duration - beat_origin) / bar_seconds))
    primary_chords, per_bar_segments, details = [], [], []
    if bar_count == 0:
        return {"tonic": tonic, "chords": primary_chords, "split_chords": per_bar_segments, "bar_details": details}
    transform_y = np.pad(y, (0, max(0, 4096 - len(y))))
    hop = 256
    bass_cqt = np.abs(librosa.cqt(transform_y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_bins=24, bins_per_octave=12, tuning=0.0, scale=False))
    upper_cqt = np.abs(librosa.cqt(transform_y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C3"), n_bins=36, bins_per_octave=12, tuning=0.0, scale=False))
    frame_times = librosa.frames_to_time(np.arange(upper_cqt.shape[1]), sr=sr, hop_length=hop)
    for bar_index in range(bar_count):
        bar_start = beat_origin + bar_index * bar_seconds
        definitions = ((0.0, measure_beats / 2), (measure_beats / 2, measure_beats / 2)) if split_bar and measure_beats >= 4 else ((0.0, measure_beats),)
        segments, segment_details = [], []
        for offset, length in definitions:
            start = bar_start + offset * seconds_per_beat
            end = min(start + length * seconds_per_beat, duration)
            result = _segment(start, end, frame_times, bass_cqt, upper_cqt, pitch_names)
            segment = {"beat": float(offset), "duration": float(length), "chord": result["chord"]}
            segments.append(segment)
            segment_details.append({**result, "beat": float(offset), "duration": float(length)})
        primary_chords.append(segments[0]["chord"])
        per_bar_segments.append(segments)
        details.append({"bar_index": int(bar_index), "start_time": float(bar_start), "end_time": float(bar_start + bar_seconds), "chord": segments[0]["chord"], "segments": segment_details})
    return {"tonic": tonic, "chords": primary_chords, "split_chords": per_bar_segments, "bar_details": details}


def extract_instrumental_melody(
    audio_path: str | Path,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    silent_bars: list[int] | None = None,
) -> dict[int, list[dict]]:
    """Extract top-voice instrumental lead melody notes for intro/interlude/outro bars where vocals are silent.

    Returns dict mapping `bar_index -> list[{'beat_offset': float, 'duration': float, 'pitch': int}]`.
    """
    import librosa

    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    y, sr = _load_audio(audio_path)
    inferred_bar_count = int(np.ceil(max(0.0, len(y) / sr - beat_origin) / bar_seconds))
    requested = list(range(inferred_bar_count) if silent_bars is None else silent_bars)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in requested):
        raise ValueError("Bar indices must be nonnegative integers")
    result = {int(i): [] for i in requested}
    if not requested:
        return result
    transform_y = np.pad(y, (0, max(0, 4096 - len(y))))
    melody_cqt = np.abs(librosa.cqt(transform_y, sr=sr, hop_length=256, fmin=librosa.note_to_hz("C4"), n_bins=25, bins_per_octave=12, tuning=0.0, scale=False))
    frame_times = librosa.frames_to_time(np.arange(melody_cqt.shape[1]), sr=sr, hop_length=256)
    for bar_index in result:
        for beat_index in range(int(measure_beats)):
            start = beat_origin + bar_index * bar_seconds + beat_index * seconds_per_beat
            end = min(start + seconds_per_beat, len(y) / sr)
            s0, s1 = int(start * sr), min(len(y), int(end * sr))
            if s1 <= s0 or not np.any(y[s0:s1]):
                continue
            selected = (frame_times >= start) & (frame_times < end)
            if not np.any(selected):
                continue
            spectrum = np.median(melody_cqt[:, selected], axis=1)
            if not np.any(spectrum > 0):
                continue
            result[bar_index].append({"beat": float(beat_index), "beat_offset": float(beat_index), "duration": float(min(1.0, (s1 - s0) / sr / seconds_per_beat)), "pitch": int(60 + np.argmax(spectrum))})
    return result
