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


def synthesize_melody_conditioned_harmony(
    notes: list,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    tonic: str = "C",
    accompaniment_path: str | Path | None = None,
    minor: bool = False,
    clip_duration: float = 0.0,
    bar_start_seconds: list[float] | None = None,
) -> dict:
    """Dual-Band (Bass CQT + Upper CQT) Melody-Conditioned Functional Harmony Engine
    with expanded 24-chord vocabulary and Viterbi smoothing across bars."""
    import math

    NAMES = "C C# D D# E F F# G G# A A# B".split()
    if tonic not in NAMES:
        raise ValueError("Malformed tonic")
    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    R = NAMES.index(tonic)

    def _chord_name(root, is_minor):
        return NAMES[root % 12] + ("m" if is_minor else "")

    # --- Expanded Functional & Acoustic Guitar Chord Vocabulary ---
    # Each candidate: (root, quality, tones_set, prior_bias, emit_name)
    # emit_name uses only major/minor to stay parseable by guitar.parse_chord
    candidates = []
    seen_names = set()

    def _add(root, quality, tones, prior_bias):
        r = root % 12
        is_minor = quality == "m"
        name = _chord_name(r, is_minor)
        key = (name, tones)
        if key not in seen_names:
            seen_names.add(key)
            candidates.append((r, quality, tuple(sorted(set(((r + t) % 12) for t in tones))), prior_bias, name))

    if minor:
        R_min = (R + 9) % 12
        # Diatonic & modal minor triads: i, III, iv, v, V, VI, VII, Dorian IV
        _add(R_min, "m", (0, 3, 7), 0.08)                 # i (tonic minor)
        _add((R_min + 3) % 12, "", (0, 4, 7), 0.06)       # bIII (relative major)
        _add((R_min + 5) % 12, "m", (0, 3, 7), 0.06)      # iv (subdominant minor)
        _add((R_min + 5) % 12, "", (0, 4, 7), 0.05)       # IV (Dorian major subdominant)
        _add((R_min + 7) % 12, "m", (0, 3, 7), 0.05)      # v (natural minor dominant)
        _add((R_min + 7) % 12, "", (0, 4, 7), 0.02)       # V (harmonic minor dominant - only when leading tone present)
        _add((R_min + 8) % 12, "", (0, 4, 7), 0.06)       # bVI
        _add((R_min + 10) % 12, "", (0, 4, 7), 0.07)      # bVII (subtonic major - essential in folk/Dorian)
    else:
        # Diatonic triads: I, ii, iii, IV, V, vi + functional secondary dominants & borrowed bVII
        _add(R, "", (0, 4, 7), 0.07)                  # I
        _add((R + 2) % 12, "m", (0, 3, 7), 0.06)      # ii
        _add((R + 4) % 12, "m", (0, 3, 7), 0.06)      # iii
        _add((R + 5) % 12, "", (0, 4, 7), 0.06)        # IV
        _add((R + 7) % 12, "", (0, 4, 7), 0.06)        # V
        _add((R + 9) % 12, "m", (0, 3, 7), 0.06)       # vi
        # Secondary dominant V/vi
        _add((R + 4) % 12, "", (0, 4, 7), -0.02)
        # Secondary dominant V/V
        _add((R + 2) % 12, "", (0, 4, 7), -0.03)
        # Borrowed bVII (Mixolydian)
        _add((R + 10) % 12, "", (0, 4, 7), -0.03)

    chord_names = [c[4] for c in candidates]
    candidate_roots = [c[0] for c in candidates]
    candidate_tones = [c[2] for c in candidates]
    candidate_biases = [c[3] for c in candidates]
    candidate_qualities = [c[1] for c in candidates]

    max_note_end = max((n.start * seconds_per_beat + beat_origin + n.duration * seconds_per_beat
                        for n in notes), default=0.0) if notes else 0.0
    total_duration = max(clip_duration, max_note_end, beat_origin + bar_seconds)
    n_bars = max(1, int(math.ceil((total_duration - beat_origin) / bar_seconds)))

    # --- True Bar-Boundary Audio Slicing ---
    if bar_start_seconds is not None and len(bar_start_seconds) >= 2:
        n_bars = len(bar_start_seconds) - 1
        computed_bar_starts = [float(t) for t in bar_start_seconds]
    else:
        computed_bar_starts = [beat_origin + b * bar_seconds for b in range(n_bars + 1)]

    # --- Dual-Band CQT Extraction ---
    acc_bass_per_bar = None
    acc_chroma_per_bar = None
    acc_rms_per_bar = None
    if accompaniment_path is not None:
        acc_path = Path(accompaniment_path)
        if acc_path.is_file():
            import librosa
            y_acc, sr_acc = librosa.load(str(acc_path), sr=22050, mono=True)
            if y_acc.size and np.all(np.isfinite(y_acc)):
                hop = 512
                pad_acc = np.pad(y_acc, (0, max(0, 4096 - len(y_acc))))
                bass_cqt = np.abs(librosa.cqt(pad_acc, sr=sr_acc, hop_length=hop,
                                               fmin=librosa.note_to_hz("C2"), n_bins=24,
                                               bins_per_octave=12, tuning=0.0, scale=False))
                upper_cqt = np.abs(librosa.cqt(pad_acc, sr=sr_acc, hop_length=hop,
                                                fmin=librosa.note_to_hz("C3"), n_bins=36,
                                                bins_per_octave=12, tuning=0.0, scale=False))
                ft = librosa.frames_to_time(np.arange(bass_cqt.shape[1]), sr=sr_acc, hop_length=hop)
                rms_frames = librosa.feature.rms(y=y_acc, frame_length=2048, hop_length=hop)[0]
                acc_bass_per_bar = []
                acc_chroma_per_bar = []
                acc_rms_per_bar = []
                for b in range(n_bars):
                    bs = float(computed_bar_starts[b])
                    be = float(computed_bar_starts[b + 1])
                    sel = (ft >= bs) & (ft < be)
                    bass_chroma = np.zeros(12)
                    upper_chroma = np.zeros(12)
                    bar_rms = 0.0
                    if np.any(sel):
                        bass_spec = np.median(bass_cqt[:, sel], axis=1)
                        bass_chroma = bass_spec[:12] + bass_spec[12:24]
                        upper_spec = np.median(upper_cqt[:, sel], axis=1)
                        for oct_start in range(0, 36, 12):
                            upper_chroma += upper_spec[oct_start:oct_start + 12]
                        rms_sel = rms_frames[:len(ft)][sel] if len(rms_frames) >= len(ft) else rms_frames[sel[:len(rms_frames)]]
                        bar_rms = float(np.mean(rms_sel)) if len(rms_sel) else 0.0
                    acc_bass_per_bar.append(bass_chroma)
                    acc_chroma_per_bar.append(upper_chroma)
                    acc_rms_per_bar.append(bar_rms)

    # --- Emission Scoring ---
    emission_scores = []
    vocal_present = []
    for b in range(n_bars):
        bs_beat = b * measure_beats
        be_beat = (b + 1) * measure_beats
        vocal_w = np.zeros(12)
        for n in notes:
            overlap = max(0.0, min(n.start + n.duration, be_beat) - max(n.start, bs_beat))
            if overlap <= 0:
                continue
            on_strong = abs(n.start - bs_beat) < 0.13 or abs(n.start - (bs_beat + measure_beats / 2)) < 0.13
            weight = overlap * (1.35 if on_strong else 1.0)
            vocal_w[n.pitch % 12] += weight

        has_vocal = float(np.sum(vocal_w)) > 0
        vocal_present.append(has_vocal)

        acc_bass = acc_bass_per_bar[b] if acc_bass_per_bar is not None else None
        acc_upper = acc_chroma_per_bar[b] if acc_chroma_per_bar is not None else None

        bar_scores = {}
        for ci in range(len(candidates)):
            root = candidate_roots[ci]
            tones = set(candidate_tones[ci])
            quality = candidate_qualities[ci]
            name = chord_names[ci]
            bias = candidate_biases[ci]
            is_minor_q = quality == "m"
            third = (root + 3) % 12 if is_minor_q else (root + 4) % 12
            avoid_pcs = {(third + 1) % 12, (third - 1) % 12} if not is_minor_q else set()

            vocal_score = 0.0
            if has_vocal:
                s = 0.0
                for pc in range(12):
                    if pc in tones:
                        s += vocal_w[pc] * 1.0
                    elif pc in avoid_pcs:
                        s += vocal_w[pc] * -0.85
                    else:
                        s += vocal_w[pc] * -0.25
                vocal_score = s / float(np.sum(vocal_w))

            bass_score = 0.0
            if acc_bass is not None:
                max_bass = float(np.max(acc_bass))
                if max_bass > 0:
                    bass_score = float(acc_bass[root] / max_bass)

            upper_score = 0.0
            if acc_upper is not None and float(np.sum(acc_upper)) > 0:
                s = 0.0
                for pc in range(12):
                    s += acc_upper[pc] * (1.0 if pc in tones else -0.3)
                upper_score = s / float(np.sum(acc_upper))

            if has_vocal:
                if acc_bass is not None or acc_upper is not None:
                    combined = 0.60 * vocal_score + 0.18 * upper_score + 0.22 * bass_score + bias
                else:
                    combined = 0.92 * vocal_score + bias
            else:
                t_root = (R + 9) % 12 if minor else R
                turnaround_roots = (
                    (t_root, (t_root + 10) % 12, (t_root + 3) % 12, (t_root + 7) % 12)
                    if minor
                    else (t_root, (t_root + 7) % 12, (t_root + 9) % 12, (t_root + 5) % 12)
                )
                turnaround_prior = 0.14 if (acc_upper is None and root == turnaround_roots[b % 4]) else 0.0
                tonic_prior = 0.22 if (b == 0 and root == t_root) else (0.05 if root == t_root else 0.0)
                cadence_prior = 0.08 if (b == n_bars - 1 or b % 4 == 0) and root == t_root else 0.0
                combined = 0.55 * upper_score + 0.35 * bass_score + tonic_prior + cadence_prior + turnaround_prior + bias

            # In minor keys, Major V (harmonic minor dominant) contains the raised leading tone (#7).
            # Penalize Major V in bars where #7 is absent so Dorian/Aeolian songs use bVII / v instead of false V.
            if minor and root == (R_min + 7) % 12 and not is_minor_q:
                leading_pc = (R_min + 11) % 12
                natural_subtonic_pc = (R_min + 10) % 12
                has_leading_vocal = has_vocal and vocal_w[leading_pc] > 0.05
                has_leading_acc = (
                    acc_upper is not None
                    and float(np.sum(acc_upper)) > 0
                    and acc_upper[leading_pc] > 1.15 * acc_upper[natural_subtonic_pc]
                )
                if not (has_leading_vocal or has_leading_acc):
                    combined -= 0.22

            if name in bar_scores:
                bar_scores[name] = max(bar_scores[name], combined)
            else:
                bar_scores[name] = combined
        emission_scores.append(bar_scores)

    # --- Functional Viterbi Transitions ---
    tonic_root = (R + 9) % 12 if minor else R
    tonic_name = _chord_name(tonic_root, minor)
    dominant_root = (tonic_root + 7) % 12
    dom_name = _chord_name(dominant_root, False)
    vi_root = (R + 9) % 12 if not minor else -1
    vi_name = _chord_name(vi_root, True) if not minor else ""
    v_of_vi_name = _chord_name((R + 4) % 12, False) if not minor else ""

    name_to_root = {}
    for ci in range(len(candidates)):
        n = chord_names[ci]
        if n not in name_to_root:
            name_to_root[n] = candidate_roots[ci]

    unique_names = list(dict.fromkeys(chord_names))

    states = {None: (0.0, [])}
    for b in range(n_bars):
        updated = {}
        for name in unique_names:
            if name not in emission_scores[b]:
                continue
            fit = emission_scores[b][name]
            best = None
            curr_root = name_to_root.get(name, 0)
            for prev, (cost, path) in states.items():
                transition = 0.0
                if prev is not None and prev != "N.C." and name != "N.C.":
                    prev_root = name_to_root.get(prev, 0)
                    distance = (curr_root - prev_root) % 12

                    # Authentic cadences
                    if prev == dom_name and name == tonic_name:
                        transition += 0.06
                    elif not minor and prev == v_of_vi_name and name == vi_name:
                        transition += 0.10

                    # Circle-of-fifths and whole-step (modal bVII <-> i, IV <-> V) progressions
                    if distance in (5, 7):
                        transition += 0.10
                    elif distance in (2, 10):
                        transition += 0.10

                    # Repetition penalty
                    if prev == name:
                        transition -= 0.03
                        if len(path) >= 2 and path[-2] == name:
                            transition -= 0.14

                    if (b % 4 == 0 or b == n_bars - 1) and name == tonic_name:
                        transition += 0.12
                total = cost + fit + transition
                if best is None or total > best[0]:
                    best = (total, path + [name])
            if best is not None:
                if name not in updated or best[0] > updated[name][0]:
                    updated[name] = best
        states = updated

    chords = max(states.values(), key=lambda x: x[0])[1] if states else ["N.C."] * n_bars

    bar_dynamics = []
    if acc_rms_per_bar is not None:
        raw = np.array(acc_rms_per_bar, dtype=np.float64)
        peak = float(np.max(raw)) if np.any(raw > 0) else 1.0
        normalized = raw / peak if peak > 0 else raw
        smoothed = np.convolve(normalized, np.ones(3) / 3, mode="same")
        bar_dynamics = [round(float(np.clip(0.60 + 0.52 * v, 0.60, 1.12)), 4) for v in smoothed]
    elif notes:
        vocal_energy = np.zeros(n_bars, dtype=np.float64)
        for b in range(n_bars):
            bs_beat, be_beat = b * measure_beats, (b + 1) * measure_beats
            bar_ns = [n for n in notes if min(n.start + n.duration, be_beat) > max(n.start, bs_beat)]
            if bar_ns:
                cov = sum(min(n.start + n.duration, be_beat) - max(n.start, bs_beat) for n in bar_ns) / measure_beats
                reg = np.clip((np.mean([n.pitch for n in bar_ns]) - 54.0) / 22.0, 0.0, 1.0)
                vocal_energy[b] = float(np.clip(0.55 * min(1.0, cov) + 0.45 * reg, 0.15, 1.0))
            else:
                vocal_energy[b] = 0.25
        smoothed = np.convolve(vocal_energy, np.ones(3) / 3, mode="same")
        bar_dynamics = [round(float(np.clip(0.66 + 0.44 * v, 0.66, 1.10)), 4) for v in smoothed]
    else:
        bar_dynamics = [0.85] * n_bars

    return {
        "chords": chords,
        "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
        "bar_dynamics": bar_dynamics,
    }


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
