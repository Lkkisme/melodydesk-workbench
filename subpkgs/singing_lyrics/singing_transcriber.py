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
        chroma[int(n.pitch) % 12] += w

    last_pc = int(notes[-1].pitch) % 12
    first_pc = int(notes[0].pitch) % 12

    top_k, top_minor, max_corr = 0, False, -2.0
    top_profile = list(AS_MAJOR)
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

            if corr > max_corr:
                max_corr = corr
                top_k = k
                top_minor = is_minor
                top_profile = list(profile)

    rel_major_pc = (top_k + 3) % 12 if top_minor else top_k
    return {'tonic_pc': top_k, 'is_minor': top_minor, 'rel_major_pc': rel_major_pc,
            'confidence': max_corr, 'profile': top_profile}


def transcribe_vocal_notes(
    vocal_path: str | Path,
    bpm: float = 120.0,
    beat_origin: float = 0.0,
    word_boundaries_sec: list[float] | None = None,
    prefer_cuda: bool = True,
    beat_times: list[float] | None = None,
) -> dict:
    """Transcribe lead vocal audio into Note objects + detected word boundaries.

    Delegates to `singing_model.track_singing` (which configures ROSVOT `note_bd_threshold=0.5`), runs on CUDA when available,
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
        raw_notes = result["notes"]
        returned_boundaries = result["word_boundaries_sec"]
        backend_used, device_used = result["backend"], result["device"]
    else:
        raw_notes = pitch_notes.extract_vocal_notes(str(vocal_path), bpm, beat_origin)
        if boundaries:
            from music import Note
            spb_local = 60.0 / bpm
            min_wb_seg = 0.20 - 1e-9
            split_raw = []
            for n in raw_notes:
                n_s = float(n.raw_start)
                n_e = n_s + float(n.raw_duration)
                if float(n.raw_duration) < 2 * min_wb_seg or getattr(n, "edited", False):
                    split_raw.append(n)
                    continue
                interior = [wb for wb in boundaries if n_s + min_wb_seg <= wb <= n_e - min_wb_seg]
                if not interior:
                    split_raw.append(n)
                    continue
                cuts = [n_s] + interior + [n_e]
                valid_cuts = [cuts[0]]
                for c in cuts[1:]:
                    if c - valid_cuts[-1] >= min_wb_seg:
                        valid_cuts.append(c)
                    else:
                        valid_cuts[-1] = c
                if len(valid_cuts) < 2:
                    split_raw.append(n)
                    continue
                for j in range(len(valid_cuts) - 1):
                    seg_s = valid_cuts[j]
                    seg_d = valid_cuts[j + 1] - seg_s
                    if seg_d >= min_wb_seg:
                        b_s = max(0.0, (seg_s - beat_origin) / spb_local)
                        b_e = max(b_s + 1e-4, (seg_s + seg_d - beat_origin) / spb_local)
                        split_raw.append(Note(n.pitch, b_s, b_e - b_s, seg_s, seg_d, n.confidence, n.edited))
            raw_notes = split_raw

    if beat_times is not None:
        from music import Note
        origin_beat = float(seconds_to_beats_piecewise(beat_origin, beat_times, bpm, beat_origin))
        converted = []
        for n in raw_notes:
            raw_s = float(n.raw_start)
            raw_e = raw_s + float(n.raw_duration)
            start_beat = float(seconds_to_beats_piecewise(raw_s, beat_times, bpm, beat_origin)) - origin_beat
            end_beat = float(seconds_to_beats_piecewise(raw_e, beat_times, bpm, beat_origin)) - origin_beat
            clamped_start = max(0.0, start_beat)
            dur_beat = max(0.0, end_beat - clamped_start)
            if dur_beat > 0:
                converted.append(Note(n.pitch, clamped_start, dur_beat,
                                      n.raw_start, n.raw_duration, n.confidence, n.edited))
        notes = converted
    elif singing_model.ready():
        notes = performance_timing(raw_notes, bpm, beat_origin)
    else:
        notes = raw_notes
    return {"notes": notes, "word_boundaries_sec": returned_boundaries,
            "backend": backend_used, "device": device_used}


VITERBI_TRANSITION_WEIGHT = 0.1
VITERBI_PHRASE_GAP_SEC = 0.45


def _pitch_candidates(n, profile: list, tonic_k: int) -> list[int]:
    """Candidate pitch states for one note; the observed pitch is always first (tie preference)."""
    p0 = int(n.pitch)
    conf_i = max(0.35, float(getattr(n, 'confidence', 1.0)))
    rel0 = (p0 % 12 - tonic_k) % 12
    if (getattr(n, 'edited', False) or float(n.duration) > 0.8 or conf_i > 0.85
            or profile[rel0] > 0.05):
        return [p0]
    return [p0] + [p for p in (p0 - 1, p0 + 1) if 21 <= p <= 108]


def _pitch_node_potential(n, p: int, profile: list, tonic_k: int) -> float:
    """Log-potential of state p: observation fidelity + key-profile prior weighted by lambda_i."""
    p0 = int(n.pitch)
    conf_i = max(0.35, float(getattr(n, 'confidence', 1.0)))
    lam = max(0.3, min(1.5, 1.0 - 0.8 * conf_i + 0.5 * (1.0 - float(n.duration))))
    obs = -abs(p - p0) * (0.5 + 0.5 * conf_i)
    log_prior = math.log(max(1e-6, profile[(p - tonic_k) % 12]))
    return obs + lam * log_prior


def _viterbi_decode_phrase(phrase: list, profile: list, tonic_k: int) -> list[int]:
    """First-order HMM Viterbi over one phrase: dp[i][s] / back[i][s] with -w*|p_i - p_{i-1}| transitions."""
    states = [_pitch_candidates(n, profile, tonic_k) for n in phrase]
    if all(len(s) == 1 for s in states):
        return [s[0] for s in states]
    dp = [[_pitch_node_potential(phrase[0], p, profile, tonic_k) for p in states[0]]]
    back = [[-1] * len(states[0])]
    for i in range(1, len(phrase)):
        row, ptr = [], []
        for p in states[i]:
            node = _pitch_node_potential(phrase[i], p, profile, tonic_k)
            arg_s, max_path_score = 0, -math.inf
            for s_prev, q in enumerate(states[i - 1]):
                path_score = dp[i - 1][s_prev] - VITERBI_TRANSITION_WEIGHT * abs(p - q)
                if path_score > max_path_score:
                    max_path_score, arg_s = path_score, s_prev
            row.append(max_path_score + node)
            ptr.append(arg_s)
        dp.append(row)
        back.append(ptr)
    last = dp[-1]
    s_cur = max(range(len(last)), key=lambda s: (last[s], -s))
    decoded = [0] * len(phrase)
    for i in range(len(phrase) - 1, -1, -1):
        decoded[i] = states[i][s_cur]
        s_cur = back[i][s_cur]
    return decoded


def _viterbi_regularize_pitch_phrases(result: list, profile: list, tonic_k: int, spb: float) -> None:
    """Phrase-level Viterbi pitch-sequence regularizer (in place).

    Notes are split into phrases at rests longer than VITERBI_PHRASE_GAP_SEC; within each
    phrase the pitch sequence is decoded jointly, with candidates {p0-1, p0, p0+1} for
    uncertain out-of-key notes and {p0} for edited/long/confident/in-key notes.
    """
    if not result:
        return
    phrases: list[list] = [[result[0]]]
    for prev, cur in zip(result, result[1:]):
        rest_sec = (float(cur.start) - (float(prev.start) + float(prev.duration))) * spb
        if rest_sec > VITERBI_PHRASE_GAP_SEC:
            phrases.append([])
        phrases[-1].append(cur)
    for phrase in phrases:
        for n, cand_opt_p in zip(phrase, _viterbi_decode_phrase(phrase, profile, tonic_k)):
            if cand_opt_p != int(n.pitch):
                n.pitch = cand_opt_p


def regulate_singing_notes_for_score(
    notes: list,
    bpm: float,
    beat_origin: float = 0.0,
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
    clip_duration = float(clip_duration)
    if not math.isfinite(clip_duration) or clip_duration < 0.0:
        raise ValueError("clip_duration must be finite and non-negative")
    if 0.0 < clip_duration <= beat_origin:
        return []
    spb = 60.0 / bpm
    grid = float(min_grid_beat)

    use_piecewise = False
    origin_beat = 0.0
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=np.float64).ravel()
        if len(bt) >= 2 and np.all(np.diff(bt) > 0):
            use_piecewise = True
            origin_beat = float(seconds_to_beats_piecewise(beat_origin, beat_times, bpm, beat_origin))

    ordered = sorted(copy.deepcopy(notes), key=lambda n: n.raw_start)

    for n in ordered:
        is_ed = bool(getattr(n, 'edited', False))
        if n.raw_start < beat_origin and (n.raw_start + n.raw_duration >= beat_origin + 0.065 or (is_ed and n.raw_start + n.raw_duration > beat_origin)):
            n.raw_duration = (n.raw_start + n.raw_duration) - beat_origin
            n.raw_start = beat_origin

    filtered = [n for n in ordered if n.raw_start >= beat_origin and (n.raw_duration >= 0.065 or (bool(getattr(n, 'edited', False)) and n.raw_duration > 0))]
    if not filtered:
        return []

    def _has_wb_between(a, b, wbs_list):
        a_end = a.raw_start + a.raw_duration
        for wb in wbs_list:
            if a_end - 0.03 < wb < b.raw_start + 0.03:
                return True
        return False

    wb_set = sorted(set(float(w) for w in (word_boundaries_sec or []) if math.isfinite(w))) if word_boundaries_sec else []

    # Robust articulation-tempo statistics of the singer (medians are insensitive
    # to the very micro-glitches we are about to merge). Inside the reference
    # band the scales are exactly 1.0, so standard inputs keep the original
    # time constants; only unusually fast/slow articulation rescales them.
    durs_sorted = sorted(float(n.raw_duration) for n in filtered)
    median_dur = durs_sorted[len(durs_sorted) // 2] if len(durs_sorted) % 2 else \
        0.5 * (durs_sorted[len(durs_sorted) // 2 - 1] + durs_sorted[len(durs_sorted) // 2])
    gaps_pos = sorted(g for g in (b.raw_start - (a.raw_start + a.raw_duration)
                                  for a, b in zip(filtered, filtered[1:])) if g > 0.0)
    if gaps_pos:
        median_gap = gaps_pos[len(gaps_pos) // 2] if len(gaps_pos) % 2 else \
            0.5 * (gaps_pos[len(gaps_pos) // 2 - 1] + gaps_pos[len(gaps_pos) // 2])
    else:
        median_gap = 0.0
    DUR_BAND_LO, DUR_BAND_HI = 0.15, 0.60
    if median_dur < DUR_BAND_LO:
        dur_scale = max(0.6, median_dur / DUR_BAND_LO)
    elif median_dur > DUR_BAND_HI:
        dur_scale = min(1.5, median_dur / DUR_BAND_HI)
    else:
        dur_scale = 1.0
    GAP_BAND_HI = 0.25
    gap_scale = min(1.4, median_gap / GAP_BAND_HI) if median_gap > GAP_BAND_HI else 1.0
    merge_gap_tau = 0.05 * dur_scale * gap_scale
    merge_max_prev_dur = 0.16 * dur_scale
    scoop_gap_tau = 0.07 * dur_scale * gap_scale
    scoop_max_prev_dur = 0.21 * dur_scale

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

            # Uncalibrated linear affinity (not a probability): > 0 favours merging.
            merge_affinity = (-2.0 * max(0.0, gap_t) / merge_gap_tau - 1.5 * dp + 1.2 * dur_ratio
                              + 0.8 * (prev_conf + cur_conf) - 1.5)
            if has_wb:
                merge_affinity -= 3.0

            if (not getattr(prev, 'edited', False) and not getattr(n, 'edited', False)
                    and not has_wb and dp <= 1 and merge_affinity > 0.0
                    and prev.raw_duration < merge_max_prev_dur):
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

            # Uncalibrated linear affinity (not a probability): > 0 favours absorbing the scoop.
            scoop_affinity = (-2.0 * max(0.0, gap_t) / scoop_gap_tau
                              + 1.5 * (cur.raw_duration / max(prev.raw_duration, 1e-6) - 1.0)
                              + 0.8 * (cur_conf - prev_conf)
                              - 0.5 * max(0, dp - 1))
            if has_wb:
                scoop_affinity -= 3.0

            if (not getattr(prev, 'edited', False) and not getattr(cur, 'edited', False)
                    and not has_wb and 1 <= dp <= 3
                    and prev.raw_duration < scoop_max_prev_dur and gap_t < scoop_gap_tau
                    and cur.raw_duration >= prev.raw_duration * 1.35
                    and scoop_affinity > 0.0):
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
            min_wb_seg = 0.20 - 1e-9
            for n in merged:
                if n.raw_duration < 2 * min_wb_seg or getattr(n, 'edited', False):
                    split_merged.append(n)
                    continue
                n_start = n.raw_start
                n_end = n.raw_start + n.raw_duration
                interior = [wb for wb in wbs if n_start + min_wb_seg <= wb <= n_end - min_wb_seg]
                cuts = [n_start] + interior + [n_end]
                valid_cuts = [cuts[0]]
                for c in cuts[1:]:
                    if c - valid_cuts[-1] >= min_wb_seg:
                        valid_cuts.append(c)
                    else:
                        valid_cuts[-1] = c
                if len(valid_cuts) < 2 or valid_cuts[-1] - valid_cuts[0] < min_wb_seg:
                    split_merged.append(n)
                    continue
                appended = False
                for j in range(len(valid_cuts) - 1):
                    seg_start = valid_cuts[j]
                    seg_dur = valid_cuts[j + 1] - seg_start
                    if seg_dur >= min_wb_seg:
                        split_merged.append(Note(n.pitch, n.start, n.duration, seg_start, seg_dur, n.confidence, n.edited))
                        appended = True
                if not appended:
                    split_merged.append(n)
            merged = split_merged

    for n in merged:
        if use_piecewise and n.raw_start >= 0 and n.raw_duration > 0:
            cont_start = float(seconds_to_beats_piecewise(n.raw_start, beat_times, bpm, beat_origin)) - origin_beat
            cont_end = float(seconds_to_beats_piecewise(n.raw_start + n.raw_duration, beat_times, bpm, beat_origin)) - origin_beat
        else:
            cont_start = (n.raw_start - beat_origin) / spb
            cont_end = (n.raw_start + n.raw_duration - beat_origin) / spb
        n.start = max(0.0, cont_start)
        n.duration = max(0.12, cont_end - n.start)

    deduped: list = []
    for n in merged:
        if deduped and abs(deduped[-1].start - n.start) < 0.08:
            prev_ed = bool(getattr(deduped[-1], 'edited', False))
            cur_ed = bool(getattr(n, 'edited', False))
            if (cur_ed and not prev_ed) or (cur_ed == prev_ed and n.raw_duration > deduped[-1].raw_duration):
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

    max_beats_early = None
    if clip_duration > 0.0 and clip_duration > beat_origin:
        if use_piecewise:
            max_beats_early = max(1e-4, float(seconds_to_beats_piecewise(clip_duration, beat_times, bpm, beat_origin)) - origin_beat)
        else:
            max_beats_early = max(1e-4, (clip_duration - beat_origin) / spb)

    eff_grid = grid
    if len(deduped) >= 2:
        min_sep = min((b.start - a.start for a, b in zip(deduped, deduped[1:]) if b.start > a.start), default=grid)
        while eff_grid > min_sep * 0.9 and eff_grid > 0.03125:
            eff_grid *= 0.5
    if max_beats_early is not None and eff_grid > max_beats_early * 0.5:
        while eff_grid > max_beats_early * 0.5 and eff_grid > 0.03125:
            eff_grid *= 0.5

    for n in deduped:
        n.raw_start = max(0.0, n.start * spb)
        n.raw_duration = max(0.065, n.duration * spb)
        n.start = max(0.0, round(round(n.start / eff_grid) * eff_grid, 10))
        n.duration = max(eff_grid, round(round(n.duration / eff_grid) * eff_grid, 10))

    raw_result: list = []
    for phrase in phrases:
        try:
            if abs(eff_grid - 0.25) > 1e-6 or any(getattr(x, 'edited', False) for x in phrase):
                raise ValueError("Custom grid or edited note preservation required")
            organized = score_cleanup.organize(phrase, bpm, 0.0, 'auto')
            if any(abs(round(x.start / eff_grid) * eff_grid - x.start) > 1e-6
                   or abs(round(x.duration / eff_grid) * eff_grid - x.duration) > 1e-6
                   for x in organized):
                raise ValueError("Organizer violated eff_grid")
            if max_beats_early is not None:
                in_before = sum(1 for x in phrase if x.start < max_beats_early - 1e-6)
                in_after = sum(1 for x in organized if x.start < max_beats_early - 1e-6)
                if in_after < in_before:
                    raise ValueError("Organizer pushed note past clip boundary")
            raw_result.extend(organized)
        except ValueError:
            cur_s = 0.0
            for idx_p, np_item in enumerate(phrase):
                if idx_p > 0 and np_item.start < cur_s + eff_grid - 1e-6:
                    np_item.start = round(cur_s + eff_grid, 10)
                cur_s = np_item.start
                raw_result.append(np_item)

    raw_result.sort(key=lambda x: x.start)
    result: list = []
    for n in raw_result:
        if result and n.start < result[-1].start + eff_grid - 1e-6:
            prev_ed = bool(getattr(result[-1], 'edited', False))
            cur_ed = bool(getattr(n, 'edited', False))
            if (cur_ed and not prev_ed) or (cur_ed == prev_ed and n.duration > result[-1].duration):
                n.start = result[-1].start
                result[-1] = n
            continue
        result.append(n)

    if len(result) >= 4:
        key_info = estimate_key_albrecht_shanahan(result)
        if key_info['confidence'] >= 0.45:
            _viterbi_regularize_pitch_phrases(result, key_info['profile'], key_info['tonic_pc'], spb)

    max_beats = None
    if clip_duration > 0.0 and clip_duration > beat_origin:
        if use_piecewise:
            max_beats = max(1e-4, float(seconds_to_beats_piecewise(clip_duration, beat_times, bpm, beat_origin)) - origin_beat)
        else:
            max_beats = max(1e-4, (clip_duration - beat_origin) / spb)
        result = [n for n in result if n.start < max_beats - 1e-6]

    cleaned_result: list = []
    for i, n in enumerate(result):
        if i + 1 < len(result):
            max_dur = result[i + 1].start - n.start
            if max_dur <= 1e-6:
                continue
            if n.duration > max_dur:
                n.duration = max_dur
        if max_beats is not None:
            rem_beats = max_beats - n.start
            if rem_beats <= 1e-6:
                continue
            if n.duration > rem_beats:
                n.duration = rem_beats
        n.duration = max(1e-4, float(n.duration))
        if use_piecewise:
            s_sec = float(beats_to_seconds_piecewise(n.start + origin_beat, beat_times, bpm, beat_origin))
            e_sec = float(beats_to_seconds_piecewise(n.start + n.duration + origin_beat, beat_times, bpm, beat_origin))
        else:
            s_sec = beat_origin + n.start * spb
            e_sec = s_sec + n.duration * spb
        s_sec_c = max(0.0, s_sec)
        e_sec_c = max(s_sec_c + 1e-4, e_sec)
        n.raw_start = round(s_sec_c, 6)
        n.raw_duration = round(max(1e-4, e_sec_c - s_sec_c), 6)
        cleaned_result.append(n)

    return cleaned_result
