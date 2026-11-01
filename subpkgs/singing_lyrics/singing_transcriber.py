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


AS_MAJOR = (0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081)
AS_MINOR = (0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052)


def estimate_key_albrecht_shanahan(notes: list) -> dict:
    if not notes:
        return {'tonic_pc': 0, 'is_minor': False, 'rel_major_pc': 0,
                'confidence': 0.0, 'profile': list(AS_MAJOR)}

    chroma = [0.0] * 12
    n_notes = len(notes)
    for idx, n in enumerate(notes):
        conf = max(0.35, float(getattr(n, 'confidence', 1.0)))
        dur = float(getattr(n, 'duration', 1.0))
        w = dur * conf
        beat_pos = float(getattr(n, 'start', 0.0)) % 4.0
        if beat_pos < 0.1:
            w *= 1.3
        elif beat_pos % 2.0 < 0.1:
            w *= 1.15
        elif beat_pos % 1.0 < 0.1:
            w *= 1.05
        chroma[int(n.pitch) % 12] += w

    last_pc = int(notes[-1].pitch) % 12
    first_pc = int(notes[0].pitch) % 12

    best_k, best_minor, best_corr = 0, False, -2.0
    best_profile = list(AS_MAJOR)
    for k in range(12):
        for is_minor, profile in ((False, AS_MAJOR), (True, AS_MINOR)):
            rotated = [chroma[(k + d) % 12] for d in range(12)]
            mean_r = sum(rotated) / 12.0
            mean_p = sum(profile) / 12.0
            cov, var_r, var_p = 0.0, 0.0, 0.0
            for i in range(12):
                dr = rotated[i] - mean_r
                dp = profile[i] - mean_p
                cov += dr * dp
                var_r += dr * dr
                var_p += dp * dp
            denom = math.sqrt(var_r * var_p) if var_r > 0 and var_p > 0 else 1e-12
            corr = cov / denom

            if last_pc == k:
                corr += 0.12
            if first_pc == k:
                corr += 0.06

            if corr > best_corr:
                best_corr = corr
                best_k = k
                best_minor = is_minor
                best_profile = list(profile)

    rel_major_pc = (best_k + 3) % 12 if best_minor else best_k
    return {'tonic_pc': best_k, 'is_minor': best_minor, 'rel_major_pc': rel_major_pc,
            'confidence': best_corr, 'profile': best_profile}


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

    for n in ordered:
        if n.raw_start < beat_origin and beat_origin - n.raw_start <= 0.25 and n.raw_start + n.raw_duration > beat_origin + 0.06:
            n.raw_duration = (n.raw_start + n.raw_duration) - beat_origin
            n.raw_start = beat_origin

    filtered = [n for n in ordered if n.raw_start >= beat_origin and n.raw_duration >= 0.065]
    if not filtered:
        return []

    def _has_wb_between(a, b, wbs_list):
        a_end = a.raw_start + a.raw_duration
        for wb in wbs_list:
            if a_end - 0.03 < wb < b.raw_start + 0.03:
                return True
        return False

    wb_set = sorted(set(float(w) for w in (word_boundaries_sec or []) if math.isfinite(w))) if word_boundaries_sec else []

    merged: list = []
    for n in filtered:
        if merged:
            prev = merged[-1]
            gap_t = n.raw_start - (prev.raw_start + prev.raw_duration)
            dp = abs(n.pitch - prev.pitch)
            prev_conf = max(0.35, float(getattr(prev, 'confidence', 1.0)))
            cur_conf = max(0.35, float(getattr(n, 'confidence', 1.0)))
            dur_ratio = min(prev.raw_duration, n.raw_duration) / max(prev.raw_duration, n.raw_duration, 1e-6)
            has_wb = _has_wb_between(prev, n, wb_set)

            log_odds = -2.0 * max(0.0, gap_t) / 0.05 - 1.5 * dp + 1.2 * dur_ratio + 0.8 * (prev_conf + cur_conf) - 1.5
            if has_wb:
                log_odds -= 3.0

            if not has_wb and dp <= 1 and log_odds > 0.0 and prev.raw_duration < 0.16:
                prev_dur = prev.raw_duration
                prev.raw_duration = (n.raw_start + n.raw_duration) - prev.raw_start
                if n.raw_duration > prev_dur:
                    prev.pitch = n.pitch
                continue
        merged.append(n)

    scooped: list = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged):
            prev = merged[i]
            cur = merged[i + 1]
            gap_t = cur.raw_start - (prev.raw_start + prev.raw_duration)
            dp = abs(cur.pitch - prev.pitch)
            prev_conf = max(0.35, float(getattr(prev, 'confidence', 1.0)))
            cur_conf = max(0.35, float(getattr(cur, 'confidence', 1.0)))
            has_wb = _has_wb_between(prev, cur, wb_set)

            scoop_log_odds = (-2.0 * max(0.0, gap_t) / 0.07
                              + 1.5 * (cur.raw_duration / max(prev.raw_duration, 1e-6) - 1.0)
                              + 0.8 * (cur_conf - prev_conf)
                              - 0.5 * max(0, dp - 1))
            if has_wb:
                scoop_log_odds -= 3.0

            if (not has_wb and 1 <= dp <= 3
                    and prev.raw_duration < 0.21 and gap_t < 0.07
                    and cur.raw_duration >= prev.raw_duration * 1.35
                    and scoop_log_odds > 0.0):
                old_end = cur.raw_start + cur.raw_duration
                cur.raw_start = prev.raw_start
                cur.raw_duration = old_end - cur.raw_start
                scooped.append(cur)
                i += 2
                continue
        scooped.append(merged[i])
        i += 1
    merged = scooped

    if word_boundaries_sec:
        from music import Note
        wbs = sorted(set(float(w) for w in word_boundaries_sec if math.isfinite(w) and w >= beat_origin))
        if wbs:
            split_merged: list = []
            for n in merged:
                if n.raw_duration < 0.46:
                    split_merged.append(n)
                    continue
                n_start = n.raw_start
                n_end = n.raw_start + n.raw_duration
                interior = [wb for wb in wbs if n_start + 0.20 <= wb <= n_end - 0.20]
                cuts = [n_start] + interior[:4] + [n_end]
                valid_cuts = [cuts[0]]
                for c in cuts[1:]:
                    if c - valid_cuts[-1] >= 0.20:
                        valid_cuts.append(c)
                    else:
                        valid_cuts[-1] = c
                if len(valid_cuts) < 2 or valid_cuts[-1] - valid_cuts[0] < 0.20:
                    split_merged.append(n)
                    continue
                appended = False
                for j in range(len(valid_cuts) - 1):
                    seg_start = valid_cuts[j]
                    seg_dur = valid_cuts[j + 1] - seg_start
                    if seg_dur >= 0.20:
                        split_merged.append(Note(n.pitch, n.start, n.duration, seg_start, seg_dur, n.confidence, n.edited))
                        appended = True
                if not appended:
                    split_merged.append(n)
            merged = split_merged

    for n in merged:
        if use_piecewise and n.raw_start >= 0 and n.raw_duration > 0:
            cont_start = float(seconds_to_beats_piecewise(n.raw_start, beat_times, bpm, beat_origin))
            cont_end = float(seconds_to_beats_piecewise(n.raw_start + n.raw_duration, beat_times, bpm, beat_origin))
        else:
            cont_start = (n.raw_start - beat_origin) / spb
            cont_end = (n.raw_start + n.raw_duration - beat_origin) / spb
        n.start = max(0.0, cont_start)
        n.duration = min(4.0, max(0.12, cont_end - cont_start))

    deduped: list = []
    for n in merged:
        if deduped and abs(deduped[-1].start - n.start) < 0.08:
            if n.raw_duration > deduped[-1].raw_duration:
                deduped[-1] = n
        else:
            deduped.append(n)

    for i, n in enumerate(deduped):
        if i + 1 < len(deduped):
            gap_beats = deduped[i + 1].start - n.start
            if gap_beats > 0.08:
                n.duration = max(0.12, min(n.duration if gap_beats - n.duration > 0.25 else gap_beats, gap_beats))

    phrases: list[list] = []
    for n in deduped:
        if not phrases or (n.raw_start - (phrases[-1][-1].raw_start + phrases[-1][-1].raw_duration)) > 0.45:
            phrases.append([])
        phrases[-1].append(n)

    for n in deduped:
        n.raw_start = beat_origin + n.start * spb
        n.raw_duration = max(0.065, n.duration * spb)
        n.start = max(0.0, round(round(n.start / grid) * grid, 10))
        n.duration = min(4.0, max(grid, round(round(n.duration / grid) * grid, 10)))

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

    if len(result) >= 8:
        key_info = estimate_key_albrecht_shanahan(result)
        if key_info['confidence'] >= 0.45:
            profile = key_info['profile']
            tonic_k = key_info['tonic_pc']
            for i, n in enumerate(result):
                if getattr(n, 'edited', False):
                    continue
                conf_i = max(0.35, float(getattr(n, 'confidence', 1.0)))
                if n.duration > 0.8 or conf_i > 0.85:
                    continue
                p0 = int(n.pitch)
                pc0 = p0 % 12
                rel0 = (pc0 - tonic_k) % 12
                if profile[rel0] > 0.05:
                    continue
                prev_p = int(result[i - 1].pitch) if i > 0 else p0
                next_p = int(result[i + 1].pitch) if i + 1 < len(result) else p0
                best_p, best_score = p0, -math.inf
                lam = max(0.3, min(1.5, 1.0 - 0.8 * conf_i + 0.5 * (1.0 - n.duration)))
                for candidate_p in (p0 - 1, p0, p0 + 1):
                    cp_rel = (candidate_p % 12 - tonic_k) % 12
                    log_prior = math.log(max(1e-6, profile[cp_rel]))
                    obs_score = -abs(candidate_p - p0) * (0.5 + 0.5 * conf_i)
                    vl = -0.1 * (abs(candidate_p - prev_p) + abs(candidate_p - next_p))
                    score = obs_score + lam * log_prior + vl
                    if score > best_score:
                        best_score = score
                        best_p = candidate_p
                if best_p != p0:
                    n.pitch = best_p

    if clip_duration > beat_origin:
        if use_piecewise:
            max_beats = max(grid, float(seconds_to_beats_piecewise(clip_duration, beat_times, bpm, beat_origin)))
        else:
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
            s_sec = float(beats_to_seconds_piecewise(n.start, beat_times, bpm, beat_origin))
            e_sec = float(beats_to_seconds_piecewise(n.start + n.duration, beat_times, bpm, beat_origin))
            n.raw_start = round(s_sec, 6)
            n.raw_duration = round(max(1e-4, e_sec - s_sec), 6)
        else:
            n.raw_start = round(beat_origin + n.start * spb, 6)
            n.raw_duration = round(n.duration * spb, 6)
        cleaned_result.append(n)

    return cleaned_result
