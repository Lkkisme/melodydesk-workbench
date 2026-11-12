"""Lyric tokenizer, multi-backend neural forced aligner (Qwen3-ForcedAligner / MMS-FA CTC / faster-whisper),
and monotone sequence-assignment DP that maps lyric syllables onto transcribed notes.

Word timestamps come from whichever neural backend is available; Needleman-Wunsch projects them onto the
ground-truth lyric tokens, an LCS-style DP picks monotone anchors, and a monotone assignment DP places each
syllable onset on a note. Melisma continuation ('-') uses a heuristic log-affinity whose time constants are
scaled by the song's median note duration; its weights are hand-set, not learned."""
from __future__ import annotations
from pathlib import Path
import logging
import math
import re
import sys
import unicodedata
from statistics import median

_CJK_PATTERN = (
    r"[㐀-䶿一-鿿豈-﫿"
    r"\U00020000-\U0002fa1f\U00030000-\U000323af"
    r"\u3040-\u30ff\u3005-\u3007\u303b\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"
)
_LATIN_PATTERN = (
    r"(?:(?!" + _CJK_PATTERN + r")(?:[^\W_]|["
    + "".join(chr(i) for i in range(0x10000) if unicodedata.category(chr(i)).startswith("M") or i in (0x200C, 0x200D))
    + r"]))+(?:['’](?:(?!" + _CJK_PATTERN + r")(?:[^\W_]|["
    + "".join(chr(i) for i in range(0x10000) if unicodedata.category(chr(i)).startswith("M") or i in (0x200C, 0x200D))
    + r"]))+)*"
)
_SCAN = re.compile(_CJK_PATTERN + r"[\u3099\u309a\ufe00-\ufe0f]*|" + _LATIN_PATTERN + r"|\r\n|[\r\n，,.。！？!?;；、:：…]")
_LOG = logging.getLogger(__name__)
_MMS_FA_CACHE: dict[str, tuple] = {}


def tokenize_lyrics(raw_text: str) -> list[dict]:
    """Tokenize Chinese/English mixed lyrics into singable units (1 Chinese char = 1 syllable, 1 English word = 1 token) preserving line/clause breaks."""
    raw_text = unicodedata.normalize("NFC", raw_text) if isinstance(raw_text, str) else raw_text
    tokens = []
    line_index = 0
    for match in _SCAN.finditer(raw_text):
        item = match.group(0)
        if item in ("\r\n", "\r", "\n"):
            if tokens:
                tokens[-1]["is_line_end"] = True
            line_index += 1
        elif item in "，,.。！？!?;；、:：…":
            if tokens:
                tokens[-1]["is_line_end"] = True
        else:
            tokens.append({"text": item, "line_index": line_index, "is_line_end": False})
    if tokens:
        tokens[-1]["is_line_end"] = True
    return tokens


def _expand_word_timestamps(records):
    expanded = []
    previous_start = -math.inf
    records_list = list(records)
    for rec_idx, record in enumerate(records_list):
        word = record["word"]
        start, end = float(record["start"]), float(record["end"])
        probability = float(record.get("probability", 0.0))
        if not isinstance(word, str):
            raise ValueError("Timestamp word must be text")
        if (not all(math.isfinite(x) for x in (start, end, probability))
                or not 0 <= start <= end or not 0 <= probability <= 1
                or start < previous_start):
            raise ValueError("Invalid or unordered word timestamps")
        previous_start = start
        pieces = tokenize_lyrics(word)
        if not pieces:
            continue
        eff_end = end
        if len(pieces) > 1 and rec_idx + 1 < len(records_list):
            next_raw = records_list[rec_idx + 1].get("start") if isinstance(records_list[rec_idx + 1], dict) else None
            if isinstance(next_raw, (int, float, str)) and not isinstance(next_raw, bool):
                next_s = float(next_raw)
                if math.isfinite(next_s) and next_s >= start:
                    eff_end = min(end, next_s)
        is_unsung = bool(record.get("unsung", False))
        is_interp = bool(record.get("interpolated", False)) or is_unsung
        is_subdiv = (not is_interp) and (len(pieces) > 1 or bool(record.get("subdivided", False)))
        g_start = float(record.get("group_start", start)) if record.get("subdivided") else start
        g_end = float(record.get("group_end", eff_end)) if record.get("subdivided") else eff_end
        piece_prob = 0.0 if is_interp else probability
        for index, token in enumerate(pieces):
            item = {"word": token["text"],
                    "start": start + (eff_end - start) * index / len(pieces),
                    "end": start + (eff_end - start) * (index + 1) / len(pieces),
                    "probability": piece_prob}
            if is_interp:
                item["interpolated"] = True
                if is_unsung:
                    item["unsung"] = True
            elif is_subdiv:
                item["subdivided"] = True
                item["group_start"] = g_start
                item["group_end"] = g_end
            expanded.append(item)
    return expanded


def _is_faster_whisper_real():
    """Return True if faster_whisper is available or not explicitly disabled."""
    fw = sys.modules.get("faster_whisper")
    if fw is None:
        return True
    return getattr(fw, "WhisperModel", None) is not None


_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]")
_MMS_DEFAULT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz'")


def _token_to_mms_chars(token_text: str, dict_chars: set[str] | frozenset[str]) -> str:
    if re.search(_CJK_PATTERN, token_text):
        if not _HANZI_RE.search(token_text):
            return ""
        try:
            from pypinyin import Style, lazy_pinyin
            # uroman maps Mandarin ü (pypinyin 'v' in lv/nv/lve/nve) to ASCII 'u'.
            text = "".join(lazy_pinyin(token_text, style=Style.NORMAL)).lower().replace("v", "u")
        except ImportError:
            return ""
    else:
        nfkd = unicodedata.normalize("NFKD", token_text)
        text = "".join(c for c in nfkd if not unicodedata.category(c).startswith("M")).lower()
    return "".join(c for c in text if c in dict_chars)


def _can_mms_align(ground_truth_text: str) -> bool:
    gt_tokens = tokenize_lyrics(ground_truth_text)
    if not gt_tokens:
        return False
    valid = sum(1 for tok in gt_tokens if bool(_token_to_mms_chars(tok["text"], _MMS_DEFAULT_CHARS)))
    return (valid / len(gt_tokens)) >= 0.90


def _free_end_ctc_prefix_len(
    emission_2d,
    transcript_tokens: list[list[int]],
    blank_id: int = 0,
    char_prior_beta: float = 4.8,
    star_id: int | None = None,
) -> int:
    """Forward-only CTC Viterbi over blank-interleaved transcript tokens to select sung prefix k* in 0..m."""
    import numpy as np
    import torch

    m = len(transcript_tokens)
    if m == 0:
        return 0
    emis_t = torch.as_tensor(emission_2d, dtype=torch.float32)
    log_probs = torch.log_softmax(emis_t, dim=-1).cpu().numpy()
    n_frames = int(log_probs.shape[0])
    vocab_n = int(log_probs.shape[1]) if log_probs.ndim >= 2 else 0
    has_star = star_id is not None and 0 <= int(star_id) < vocab_n
    flat_chars: list[int] = []
    cum_chars = [0]
    for tok_ids in transcript_tokens:
        ids = tok_ids if tok_ids else [blank_id]
        flat_chars.extend(int(c) for c in ids)
        cum_chars.append(len(flat_chars))
    n_chars = len(flat_chars)
    if n_frames == 0 or n_chars == 0:
        return 0
    s_len = 2 * n_chars + 1
    z = np.full(s_len, int(blank_id), dtype=np.int64)
    z[1::2] = np.asarray(flat_chars, dtype=np.int64)
    skip2_idx = np.asarray(
        [s for s in range(3, s_len, 2) if z[s] != z[s - 2]],
        dtype=np.int64,
    )
    alpha = np.full(s_len, -np.inf, dtype=np.float64)
    lead0 = float(log_probs[0, z[0]])
    if has_star:
        lead0 = max(lead0, float(log_probs[0, int(star_id)]) - 1.6)
    alpha[0] = lead0
    if s_len > 1:
        alpha[1] = float(log_probs[0, z[1]])
    step1 = np.empty(s_len, dtype=np.float64)
    step2 = np.empty(s_len, dtype=np.float64)
    for t in range(1, n_frames):
        prev_a0 = alpha[0]
        step1[0] = -np.inf
        step1[1:] = alpha[:-1]
        best_prev = np.maximum(alpha, step1)
        if skip2_idx.size:
            step2.fill(-np.inf)
            step2[skip2_idx] = alpha[skip2_idx - 2]
            best_prev = np.maximum(best_prev, step2)
        alpha = best_prev + log_probs[t, z]
        if has_star:
            lead_t = max(float(log_probs[t, z[0]]), float(log_probs[t, int(star_id)]) - 1.6)
            alpha[0] = prev_a0 + lead_t

    best_k = 0
    best_sc = float(alpha[0])
    for k in range(1, m + 1):
        r_end = cum_chars[k] - 1
        f_k = max(float(alpha[2 * r_end + 1]), float(alpha[2 * r_end + 2])) + char_prior_beta * cum_chars[k]
        if f_k >= best_sc:
            best_sc = f_k
            best_k = k
    return best_k


def _try_mms_fa(audio_path: Path, ground_truth_text: str, prefer_cuda: bool) -> dict | None:
    """Tier 2: Meta MMS_FA CTC forced alignment for Latin/English and CJK text with free-end unsung detection."""
    gt = ground_truth_text.strip()
    if not gt:
        return None
    try:
        import torch
        import torchaudio
        import librosa

        device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"

        if device in _MMS_FA_CACHE:
            bundle, dictionary, model, aligner = _MMS_FA_CACHE[device]
        else:
            bundle = torchaudio.pipelines.MMS_FA
            dictionary = bundle.get_dict()
            model = bundle.get_model(with_star=True).to(device).eval()
            aligner = bundle.get_aligner()
            _MMS_FA_CACHE[device] = (bundle, dictionary, model, aligner)

        sr = bundle.sample_rate
        waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        gt_tokens = tokenize_lyrics(ground_truth_text)
        dict_chars = set(dictionary.keys()) - {"*", "-"}
        transcript_tokens = []
        any_valid = False
        for tok in gt_tokens:
            normalized = _token_to_mms_chars(tok["text"], dict_chars)
            if normalized:
                transcript_tokens.append([dictionary[c] for c in normalized])
                any_valid = True
            else:
                transcript_tokens.append([dictionary.get("*", 0)])

        if not any_valid:
            return None

        chunk_sec = 30.0
        context_sec = 1.0
        total_samples = len(waveform)

        with torch.inference_mode():
            if total_samples <= int(chunk_sec * sr):
                waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
                emission, _ = model(waveform_tensor)
            else:
                chunk_samples = int(chunk_sec * sr)
                context_samples = int(context_sec * sr)
                chunk_emissions = []
                pos = 0
                while pos < total_samples:
                    chunk_end = min(pos + chunk_samples, total_samples)
                    pad_left = min(context_samples, pos)
                    pad_right = min(context_samples, total_samples - chunk_end)
                    window_start = pos - pad_left
                    window_end = chunk_end + pad_right
                    window_tensor = torch.from_numpy(
                        waveform[window_start:window_end]
                    ).unsqueeze(0).to(device)
                    window_emission, _ = model(window_tensor)
                    total_window_frames = window_emission.shape[1]
                    window_samples = window_end - window_start
                    left_frac = pad_left / window_samples
                    right_frac = pad_right / window_samples
                    left_frames = int(round(left_frac * total_window_frames))
                    right_frames = int(round(right_frac * total_window_frames))
                    end_frame = total_window_frames - right_frames if right_frames > 0 else total_window_frames
                    chunk_emissions.append(window_emission[:, left_frames:end_frame, :].cpu())
                    pos = chunk_end
                emission = torch.cat(chunk_emissions, dim=1)

        emis_cpu = emission[0].cpu() if emission.device.type != "cpu" else emission[0]
        blank_id = int(dictionary.get("-", 0))
        star_raw = dictionary.get("*")
        star_id = int(star_raw) if star_raw is not None else None
        k_star = _free_end_ctc_prefix_len(
            emis_cpu, transcript_tokens, blank_id=blank_id, star_id=star_id
        )
        if k_star <= 0:
            return None
        active_tokens = transcript_tokens[:k_star]
        if star_id is not None:
            raw_spans = aligner(emis_cpu, [[star_id]] + active_tokens + [[star_id]])
            token_spans = raw_spans[1:-1]
        else:
            token_spans = aligner(emis_cpu, active_tokens)
        total_duration = len(waveform) / float(sr)
        num_frames = emission.shape[1]
        frame_dur = total_duration / float(num_frames)

        records = []
        for i, spans in enumerate(token_spans):
            if not spans:
                continue
            s_frame = spans[0].start
            e_frame = spans[-1].end
            score = sum(s.score for s in spans) / len(spans)
            records.append({
                "word": gt_tokens[i]["text"],
                "start": float(s_frame * frame_dur),
                "end": float(e_frame * frame_dur),
                "probability": max(0.0, min(1.0, float(score))),
            })

        if not records:
            return None
        if k_star < len(gt_tokens):
            tail_t = float(records[-1]["end"])
            for i_u in range(k_star, len(gt_tokens)):
                records.append({
                    "word": gt_tokens[i_u]["text"],
                    "start": tail_t,
                    "end": tail_t,
                    "probability": 0.0,
                    "interpolated": True,
                    "unsung": True,
                })
        return {
            "raw_lyrics": ground_truth_text,
            "word_timestamps": _expand_word_timestamps(records),
            "backend": f"mms-fa-{device}",
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _LOG.warning("MMS-FA CTC backend failed: %s", exc)
        return None


def _edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1,
                          prev[i - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        prev = curr
    return prev[la]


def _canonical_phonetic_key(text: str) -> frozenset[str]:
    norm_txt = unicodedata.normalize("NFKC", text).strip()
    if not norm_txt:
        return frozenset()
    if re.search(_CJK_PATTERN, norm_txt):
        if not _HANZI_RE.search(norm_txt):
            return frozenset([norm_txt.casefold()])
        try:
            from pypinyin import Style, lazy_pinyin, pinyin
            if len(norm_txt) == 1:
                readings = pinyin(norm_txt, style=Style.NORMAL, heteronym=True)
                if readings and readings[0]:
                    return frozenset(r.lower() for r in readings[0] if r)
            joined = "".join(lazy_pinyin(norm_txt, style=Style.NORMAL)).lower()
            return frozenset([joined]) if joined else frozenset([norm_txt.casefold()])
        except ImportError:
            return frozenset([norm_txt.casefold()])
    nfkd = unicodedata.normalize("NFKD", norm_txt)
    stripped = "".join(
        c for c in nfkd
        if not unicodedata.category(c).startswith("M") and c not in ("'", "’")
    ).casefold().strip()
    return frozenset([stripped]) if stripped else frozenset([norm_txt.casefold()])


def _token_match(gt_text: str, asr_text: str) -> tuple[float, float]:
    g = unicodedata.normalize("NFKC", gt_text).casefold().strip()
    a = unicodedata.normalize("NFKC", asr_text).casefold().strip()
    if not g or not a:
        return -2.0, 0.0
    if g == a:
        return 3.0, 1.0
    k_g = _canonical_phonetic_key(gt_text)
    k_a = _canonical_phonetic_key(asr_text)
    if k_g and k_a and (k_g & k_a):
        return 2.6, 0.9
    if not re.search(_CJK_PATTERN, gt_text) and not re.search(_CJK_PATTERN, asr_text):
        dist = _edit_distance(g, a)
        sim = max(0.0, 1.0 - dist / max(len(g), len(a)))
        if sim >= 0.5:
            return 1.2 * sim, 0.0
    return -2.0, 0.0


def _nw_score(gt_text: str, asr_text: str) -> float:
    return _token_match(gt_text, asr_text)[0]


def _project_records_to_ground_truth(records: list[dict], ground_truth_text: str) -> list[dict]:
    """Semi-global Needleman-Wunsch DP alignment projecting ASR timestamps onto ground-truth tokens."""
    gt_tokens = tokenize_lyrics(ground_truth_text)
    if not gt_tokens:
        return records
    records = _expand_word_timestamps(records)
    if not records:
        return []

    m, n = len(gt_tokens), len(records)
    GAP = -0.8
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + GAP
        bt[i][0] = 1
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + GAP
        bt[0][j] = 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = dp[i - 1][j - 1] + _nw_score(gt_tokens[i - 1]["text"], records[j - 1]["word"])
            up = dp[i - 1][j] + GAP
            left = dp[i][j - 1] + GAP
            max_cell = max(diag, up, left)
            dp[i][j] = max_cell
            if max_cell == diag:
                bt[i][j] = 0
            elif max_cell == up:
                bt[i][j] = 1
            else:
                bt[i][j] = 2

    best_end_i = m
    best_end_sc = -math.inf
    for i_cand in range(0, m + 1):
        cand_sc = dp[i_cand][n] - (0.8 if i_cand < m else 0.0)
        if cand_sc >= best_end_sc - 1e-12:
            best_end_sc = cand_sc
            best_end_i = i_cand

    matched = {}
    i, j = best_end_i, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            matched[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bt[i][j] == 1):
            i -= 1
        else:
            j -= 1

    result = []
    has_positive_match = False
    last_pos_gt_idx = -1
    last_pos_rec_idx = -1
    for i, tok in enumerate(gt_tokens):
        if i in matched:
            r_idx = matched[i]
            r = records[r_idx]
            sc, trust = _token_match(tok["text"], r["word"])
            if sc > 0.0:
                has_positive_match = True
                last_pos_gt_idx = i
                last_pos_rec_idx = max(last_pos_rec_idx, r_idx)
                is_unsung = bool(r.get("unsung", False))
                is_interp = bool(r.get("interpolated", False)) or is_unsung or (trust <= 0.0)
                rec = {
                    "word": tok["text"],
                    "start": float(r["start"]),
                    "end": float(r["end"]),
                    "probability": 0.0 if is_interp else round(float(r.get("probability", 0.0)) * trust, 4),
                }
                if is_interp:
                    rec["interpolated"] = True
                    if is_unsung:
                        rec["unsung"] = True
                elif r.get("subdivided"):
                    rec["subdivided"] = True
                    rec["group_start"] = float(r.get("group_start", r["start"]))
                    rec["group_end"] = float(r.get("group_end", r["end"]))
                result.append(rec)
            else:
                result.append({"word": tok["text"], "start": -1.0, "end": -1.0, "probability": 0.0, "interpolated": True})
        else:
            result.append({"word": tok["text"], "start": -1.0, "end": -1.0, "probability": 0.0, "interpolated": True})
    if not has_positive_match:
        return []

    observed_durs = [max(1e-3, float(r["end"]) - float(r["start"])) for r in records if float(r["end"]) > float(r["start"])]
    med_word_dur = float(median(observed_durs)) if observed_durs else 0.16

    rem_asr_dur = 0.0
    if 0 <= last_pos_rec_idx < len(records) - 1:
        rem_asr_dur = max(0.0, float(records[-1]["end"]) - float(records[last_pos_rec_idx]["end"]))
    max_trailing_interp = math.ceil(rem_asr_dur / max(0.05, med_word_dur)) if rem_asr_dur > 1e-6 else 0
    unsung_start_idx = last_pos_gt_idx + 1 + max_trailing_interp

    i = 0
    prev_matched_idx = -1
    while i < len(result):
        if result[i]["start"] >= 0:
            prev_matched_idx = i
            i += 1
            continue
        next_matched_idx = -1
        for k in range(i + 1, len(result)):
            if result[k]["start"] >= 0:
                next_matched_idx = k
                break
        gap_start = prev_matched_idx + 1
        gap_end = next_matched_idx if next_matched_idx >= 0 else len(result)
        if next_matched_idx < 0 and prev_matched_idx >= 0:
            interp_end = min(gap_end, unsung_start_idx)
            interp_count = max(0, interp_end - gap_start)
            left_time = float(result[prev_matched_idx]["end"])
            if interp_count > 0:
                right_time = left_time + med_word_dur * interp_count
                span = right_time - left_time
                for offset, idx in enumerate(range(gap_start, interp_end)):
                    s = left_time + span * offset / interp_count
                    e = left_time + span * (offset + 1) / interp_count
                    result[idx]["start"] = max(0.0, float(s))
                    result[idx]["end"] = max(result[idx]["start"] + 1e-4, float(e))
                    result[idx]["probability"] = 0.0
                    result[idx]["interpolated"] = True
                tail_anchor_t = float(result[interp_end - 1]["end"])
            else:
                tail_anchor_t = max(0.0, left_time)
            for idx in range(interp_end, gap_end):
                result[idx]["start"] = tail_anchor_t
                result[idx]["end"] = tail_anchor_t
                result[idx]["probability"] = 0.0
                result[idx]["interpolated"] = True
                result[idx]["unsung"] = True
            break
        gap_count = gap_end - gap_start
        if prev_matched_idx >= 0:
            left_time = result[prev_matched_idx]["end"]
            right_time = result[next_matched_idx]["start"] if next_matched_idx >= 0 else (left_time + med_word_dur * gap_count)
        elif next_matched_idx >= 0:
            right_time = result[next_matched_idx]["start"]
            left_time = max(0.0, right_time - med_word_dur * gap_count)
        else:
            left_time = records[0]["start"] if records else 0.0
            right_time = left_time + med_word_dur * gap_count
        if right_time > left_time:
            span = right_time - left_time
            for offset, idx in enumerate(range(gap_start, gap_end)):
                s = left_time + span * offset / gap_count
                e = left_time + span * (offset + 1) / gap_count
                result[idx]["start"] = max(0.0, float(s))
                result[idx]["end"] = max(result[idx]["start"] + 1e-4, float(e))
                result[idx]["probability"] = 0.0
                result[idx]["interpolated"] = True
        elif prev_matched_idx >= 0 and next_matched_idx >= 0:
            s_lo = float(result[prev_matched_idx]["start"])
            s_hi = float(result[next_matched_idx]["start"])
            if s_hi > s_lo:
                for offset, idx in enumerate(range(gap_start, gap_end)):
                    s = s_lo + (s_hi - s_lo) * (offset + 1) / (gap_count + 1)
                    e = s_lo + (s_hi - s_lo) * (offset + 2) / (gap_count + 1)
                    result[idx]["start"] = max(0.0, float(s))
                    result[idx]["end"] = max(result[idx]["start"] + 1e-4, float(e))
                    result[idx]["probability"] = 0.0
                    result[idx]["interpolated"] = True
            else:
                for idx in range(gap_start, gap_end):
                    result[idx]["start"] = max(0.0, s_lo)
                    result[idx]["end"] = max(0.0, s_lo) + 1e-4
                    result[idx]["probability"] = 0.0
                    result[idx]["interpolated"] = True
        else:
            for idx in range(gap_start, gap_end):
                result[idx]["start"] = max(0.0, left_time)
                result[idx]["end"] = max(0.0, left_time) + 1e-4
                result[idx]["probability"] = 0.0
                result[idx]["interpolated"] = True
        prev_matched_idx = next_matched_idx
        i = gap_end + 1 if next_matched_idx >= 0 else len(result)

    prev_start = 0.0
    for r in result:
        if r["start"] < prev_start:
            r["start"] = prev_start
            r["probability"] = 0.0
            r["interpolated"] = True
        if r["end"] <= r["start"] and not r.get("unsung", False):
            r["end"] = r["start"] + 1e-4
            r["probability"] = 0.0
            r["interpolated"] = True
        elif r.get("unsung", False) and r["end"] < r["start"]:
            r["end"] = r["start"]
        prev_start = r["start"]

    return result


def transcribe_word_timestamps(
    audio_path: str | Path,
    prefer_cuda: bool = True,
    ground_truth_text: str | None = "",
) -> dict:
    """Run Qwen3-ASR / Qwen3-ForcedAligner-0.6B (or MMS-FA CTC / faster-whisper CUDA fallback) to obtain character/word-level timestamps."""
    ground_truth_text = str(ground_truth_text or "")
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    failures = []
    last_failure = None
    text, records, backend = "", [], ""
    try:
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

        device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        if ground_truth_text:
            model = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B", dtype=dtype, device_map=device)
            try:
                result = model.align(audio=str(audio_path), text=ground_truth_text)
            except TypeError:
                result = model.align(audio=str(audio_path), text=ground_truth_text, language=None)
            text = ground_truth_text
            first_res = result[0] if result else []
            items = getattr(first_res, "items", first_res)
            backend = "qwen3-forced-aligner"
        else:
            model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B", dtype=dtype, device_map=device,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs={"dtype": dtype, "device_map": device})
            result = model.transcribe(audio=str(audio_path), return_time_stamps=True)
            text = result[0].text if result else ""
            items = result[0].time_stamps if result else []
            backend = "qwen3-asr"
        records = [
            {
                "word": item.text,
                "start": item.start_time,
                "end": item.end_time,
                "probability": float(getattr(item, "probability", getattr(item, "confidence", 0.0)) or 0.0),
            }
            for item in (items or [])
        ]
        if (text.strip() or ground_truth_text.strip()) and not records:
            raise RuntimeError("Qwen returned nonempty text without timestamps")
        expanded_qwen = _expand_word_timestamps(records)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        last_failure = exc
        failures.append(f"qwen3: {exc}")
        _LOG.warning("Qwen timestamp backend failed: %s", exc)
    else:
        return {"raw_lyrics": text, "word_timestamps": expanded_qwen,
                "backend": backend}

    # Tier 2: MMS-FA CTC when >= 90% of ground-truth tokens romanize cleanly (Latin, CJK via Pinyin, or mixed)
    if ground_truth_text.strip() and _can_mms_align(ground_truth_text):
        mms_result = _try_mms_fa(audio_path, ground_truth_text, prefer_cuda)
        if mms_result is not None:
            return mms_result
        failures.append("mms-fa: fell through")

    devices = ["cuda", "cpu"] if prefer_cuda else ["cpu"]
    for device in devices:
        try:
            import librosa
            import numpy as np
            from faster_whisper import WhisperModel

            audio_16k, _ = librosa.load(str(audio_path), sr=16000, mono=True)
            model = WhisperModel(
                "small", device=device,
                compute_type="float16" if device == "cuda" else "int8",
                download_root=str(Path(__file__).resolve().parents[2].parents[1]
                                  / "work" / "model-cache" / "whisper"))
            gt_toks = tokenize_lyrics(ground_truth_text) if ground_truth_text else []
            prompt_tokens = [tok["text"] for tok in gt_toks]
            prompt_hint = _join_tokens(prompt_tokens) if prompt_tokens else None
            try:
                segments, info = model.transcribe(
                    audio_16k.astype(np.float32), language=None, word_timestamps=True,
                    condition_on_previous_text=False, vad_filter=True,
                    temperature=0.0, initial_prompt=prompt_hint)
            except TypeError:
                segments, info = model.transcribe(
                    audio_16k.astype(np.float32), word_timestamps=True, condition_on_previous_text=False,
                    temperature=0.0, initial_prompt=ground_truth_text or None)
            segments = list(segments)
            records = [{"word": word.word, "start": word.start, "end": word.end,
                        "probability": getattr(word, "probability", 0.0)}
                       for segment in segments for word in (segment.words or [])]
            min_expected = min(len(gt_toks), max(1, len(gt_toks) // 4)) if gt_toks else 1
            if len(records) < min_expected:
                try:
                    segments_no_vad, info = model.transcribe(
                        audio_16k.astype(np.float32), language=None, word_timestamps=True,
                        condition_on_previous_text=False, vad_filter=False,
                        temperature=0.0, initial_prompt=None)
                    segments_no_vad = list(segments_no_vad)
                    records_no_vad = [{"word": word.word, "start": word.start, "end": word.end,
                                       "probability": getattr(word, "probability", 0.0)}
                                      for segment in segments_no_vad for word in (segment.words or [])]
                    if len(records_no_vad) > len(records):
                        segments = segments_no_vad
                        records = records_no_vad
                except Exception as vad_exc:
                    _LOG.warning("Whisper no-VAD retry skipped after error: %s", vad_exc)
            text = "".join(segment.text for segment in segments).strip()
            if (text or ground_truth_text.strip()) and not records:
                raise RuntimeError("Whisper returned nonempty text without timestamps")
            if ground_truth_text.strip() and records:
                projected = _project_records_to_ground_truth(records, ground_truth_text)
                if projected:
                    return {
                        "raw_lyrics": ground_truth_text,
                        "word_timestamps": _expand_word_timestamps(projected),
                        "backend": f"faster-whisper-{device}+gt-projection",
                    }
            expanded_whisper = _expand_word_timestamps(records)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            last_failure = exc
            failures.append(f"faster-whisper-{device}: {exc}")
            _LOG.warning("Whisper %s timestamp backend failed: %s", device, exc)
            continue

        return {"raw_lyrics": text, "word_timestamps": expanded_whisper,
                "backend": f"faster-whisper-{device}"}
    raise RuntimeError("All timestamp backends failed: " + "; ".join(failures)) from last_failure


def _align_text_tokens(tokens, notes):
    n = len(notes)
    if len(tokens) > n:
        _LOG.debug("Truncating %d excess lyric tokens to match %d notes", len(tokens) - n, n)
    tokens = tokens[:n]
    m = len(tokens)
    lyrics, owners = [""] * n, [None] * n
    if not m or not n:
        return lyrics, owners
    durations = [float(note.duration) for note in notes]
    gaps = [0.0] + [max(0.0, float(notes[j].start) -
                       (float(notes[j - 1].start) + durations[j - 1])) for j in range(1, n)]
    base, total = median(durations), sum(durations)
    elapsed = [0.0]
    for duration in durations:
        elapsed.append(elapsed[-1] + duration)
    costs = [[math.inf] * (n + 1) for _ in range(m + 1)]
    transitions = [[None] * (n + 1) for _ in range(m + 1)]
    costs[0][0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, min(m, j) + 1):
            penalty = gaps[j - 1] / base if i > 1 and not tokens[i - 2]["is_line_end"] else 0.0
            new = costs[i - 1][j - 1] + (elapsed[j - 1] / total - (i - 1) / m) ** 2 + penalty
            extend = math.inf
            if j >= 2:
                extend = (costs[i][j - 1] + base / durations[j - 1]
                          + abs(float(notes[j - 1].pitch) - float(notes[j - 2].pitch)) / 12.0
                          + gaps[j - 1] / base)
            costs[i][j] = min(new, extend)
            transitions[i][j] = "new" if new <= extend else "extend"
    i, j = m, n
    while j and i > 0:
        transition = transitions[i][j] or "new"
        owners[j - 1] = i - 1
        lyrics[j - 1] = tokens[i - 1]["text"] if transition == "new" else "-"
        if transition == "new":
            i -= 1
        j -= 1
    return lyrics, owners


def _match_anchors(tokens, records, intervals=None, max_allow_gap=1.5):
    valid_records = [r for r in records if not r.get("interpolated", False) and not r.get("unsung", False)]
    m, k = len(tokens), len(valid_records)
    rec_bonus = [0.0] * k
    if intervals:
        denom = float(max(1, m))
        n_iv = len(intervals)
        for j, r in enumerate(valid_records):
            rs, re_t = float(r["start"]), float(r["end"])
            ov = max((max(0.0, min(ne, re_t) - max(ns, rs)) for ns, ne in intervals), default=0.0)
            min_gap = min((max(0.0, ns - re_t, rs - ne) for ns, ne in intervals), default=math.inf)
            is_elig = (ov > 1e-6) or (m <= n_iv and min_gap <= max_allow_gap)
            rec_bonus[j] = (float(m) if is_elig else 0.0) + ((0.4 if ov > 1e-6 else 0.0) + 0.1 / (1.0 + min_gap)) / denom
    lengths = [[0.0] * (k + 1) for _ in range(m + 1)]
    choices = [[0] * (k + 1) for _ in range(m + 1)]
    anchors = [None] * m
    for i in range(m - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            max_cell = lengths[i][j + 1]
            choice = 2
            if lengths[i + 1][j] > max_cell:
                max_cell = lengths[i + 1][j]
                choice = 1
            _sc_m, tr_m = _token_match(tokens[i]["text"], valid_records[j]["word"])
            if tr_m > 0.0:
                match_gain = 1.0 if tr_m >= 0.99 else (1.0 - 0.25 / max(1, m))
                cand = match_gain + rec_bonus[j] + lengths[i + 1][j + 1]
                if cand >= max_cell - 1e-12:
                    max_cell = cand
                    choice = 0
            lengths[i][j] = max_cell
            choices[i][j] = choice
    i, j = 0, 0
    while i < m and j < k:
        c = choices[i][j]
        if c == 0:
            anchors[i] = valid_records[j]
            i, j = i + 1, j + 1
        elif c == 1:
            i += 1
        else:
            j += 1
    return anchors


MELISMA_DUR_BAND = (0.15, 0.60)
MELISMA_LEGATO_GAP = 0.06
MELISMA_TAIL_TAU_FLOOR = 0.12
MELISMA_NEXT_ANCHOR_TOL = 0.02
MELISMA_OV_REF = 0.18
MELISMA_TAIL_REF = 0.55
MELISMA_STEP_FREE = 4


def _melisma_log_affinity(ov_ratio, tail_decay, pitch_step, lead, tempo_scale=1.0):
    """Continuous log-affinity (>= 0 continues the melisma) for extending a syllable onto the next note.

    Two evidence channels combined as a log-domain OR (max):
      overlap:  log(ov_ratio / OV_REF)                      -- note still sits inside the word's ASR span
      tail:     log(tail_decay / TAIL_REF)                  -- exponential decay past the word end
                - max(0, pitch_step - STEP_FREE)            -- semitone-step continuity
                - lead / (0.01 * tempo_scale)               -- note starting before the word is not a tail
    """
    e_ov = math.log(max(ov_ratio, 1e-9) / MELISMA_OV_REF)
    e_tail = (math.log(max(tail_decay, 1e-12) / MELISMA_TAIL_REF)
              - max(0, pitch_step - MELISMA_STEP_FREE)
              - lead / (0.01 * tempo_scale))
    return max(e_ov, e_tail)


def _align_anchored_tokens(tokens, notes, intervals, anchors, baseline, spb=None):
    n, m = len(notes), len(tokens)
    if n == 0 or m == 0:
        return [""] * n, [None] * n
    if m > n:
        _LOG.debug("Truncating %d excess anchored lyric tokens to match %d notes", m - n, n)
        tokens = tokens[:n]
        anchors = anchors[:n]
        m = n
    baseline_onsets = [
        baseline.index(i) if i in baseline else min(n - 1, int(round(i * (n - 1) / max(1, m - 1))))
        for i in range(m)
    ]
    subdiv_targets = {}
    subdiv_melismas = {}
    subdiv_group_end = {}
    subdiv_not_last = {}
    idx_tok = 0
    while idx_tok < m:
        anc = anchors[idx_tok]
        if anc is not None and anc.get("subdivided"):
            gs = float(anc.get("group_start", anc["start"]))
            ge = float(anc.get("group_end", anc["end"]))
            end_tok = idx_tok + 1
            while end_tok < m and anchors[end_tok] is not None and anchors[end_tok].get("subdivided"):
                if (abs(float(anchors[end_tok].get("group_start", -1.0)) - gs) < 1e-6
                        and abs(float(anchors[end_tok].get("group_end", -1.0)) - ge) < 1e-6):
                    end_tok += 1
                else:
                    break
            g_notes = [j_n for j_n, (ns, ne) in enumerate(intervals) if min(ne, ge) - max(ns, gs) > 1e-6]
            g_count = end_tok - idx_tok
            if len(g_notes) >= g_count:
                g_lyr, g_own = _align_text_tokens(tokens[idx_tok:end_tok], [notes[j_n] for j_n in g_notes])
                for offset_t in range(g_count):
                    tok_pos = idx_tok + offset_t
                    subdiv_group_end[tok_pos] = ge
                    if offset_t < g_count - 1:
                        subdiv_not_last[tok_pos] = ge
                    if offset_t in g_own:
                        subdiv_targets[tok_pos] = g_notes[g_own.index(offset_t)]
                        subdiv_melismas[tok_pos] = {
                            g_notes[k_n] for k_n, ow in enumerate(g_own) if ow == offset_t and g_lyr[k_n] == "-"
                        }
            idx_tok = end_tok
        else:
            idx_tok += 1

    note_durs = [max(0.0, float(ne) - float(ns)) for ns, ne in intervals]
    med_note_dur = median(note_durs) if note_durs else MELISMA_DUR_BAND[0]
    if med_note_dur < MELISMA_DUR_BAND[0]:
        tempo_scale = max(0.6, med_note_dur / MELISMA_DUR_BAND[0])
    elif med_note_dur > MELISMA_DUR_BAND[1]:
        tempo_scale = min(1.5, med_note_dur / MELISMA_DUR_BAND[1])
    else:
        tempo_scale = 1.0
    eff_spb = float(spb) if (spb is not None and math.isfinite(spb) and spb > 0.0) else max(0.25, float(med_note_dur))
    g_leg = max(MELISMA_LEGATO_GAP * tempo_scale, 0.5 * eff_spb)
    g_phr = max(1.5, 2.0 * eff_spb)
    tail_tau_floor = MELISMA_TAIL_TAU_FLOOR * tempo_scale
    tau_gap = max(0.08, (0.5 * eff_spb) / math.log(2.0))

    # Build monotone onset estimates s_hat[i] and syllable territories W_i = [w_lo[i], w_hi[i]).
    s_hat = [None] * m
    for i in range(m):
        if anchors[i] is not None:
            if i in subdiv_targets:
                s_hat[i] = float(intervals[subdiv_targets[i]][0])
            else:
                s_hat[i] = float(anchors[i]["start"])
    meas_idx = [i for i, v in enumerate(s_hat) if v is not None]
    if meas_idx:
        first_m = meas_idx[0]
        first_s = float(s_hat[first_m])
        t0_sec = min(float(intervals[0][0]), first_s)
        for i in range(0, first_m):
            s_hat[i] = t0_sec + max(0.0, first_s - t0_sec) * i / max(1, first_m)
        for p_k in range(len(meas_idx) - 1):
            i_l, i_r = meas_idx[p_k], meas_idx[p_k + 1]
            s_l, s_r = float(s_hat[i_l]), float(s_hat[i_r])
            for i in range(i_l + 1, i_r):
                s_hat[i] = s_l + max(0.0, s_r - s_l) * (i - i_l) / (i_r - i_l)
        last_m = meas_idx[-1]
        last_s = float(s_hat[last_m])
        t1_sec = max(float(intervals[-1][1]), last_s)
        for i in range(last_m + 1, m):
            s_hat[i] = last_s + max(0.0, t1_sec - last_s) * (i - last_m) / max(1, m - last_m)
    else:
        t0_sec, t1_sec = float(intervals[0][0]), float(intervals[-1][1])
        for i in range(m):
            s_hat[i] = t0_sec + max(0.0, t1_sec - t0_sec) * i / max(1, m)
    for i in range(1, m):
        if s_hat[i] < s_hat[i - 1]:
            s_hat[i] = s_hat[i - 1]

    max_syl_span = max(2.5, 4.0 * eff_spb)
    w_lo = [0.0] * m
    w_hi = [0.0] * m
    for i in range(m):
        w_lo[i] = float(s_hat[i])
        anc_end = float(anchors[i]["end"]) if anchors[i] is not None else w_lo[i]
        nat_cap = max(anc_end + eff_spb, w_lo[i] + max_syl_span)
        if i + 1 < m:
            w_hi[i] = max(w_lo[i], min(float(s_hat[i + 1]), nat_cap))
        else:
            w_hi[i] = max(w_lo[i], min(max(float(intervals[-1][1]), anc_end) + 0.05, nat_cap))

    # Precompute phi_O(i, j) and phi_C(i, j) for joint 3-label Viterbi over {PRE_B, O_i, C_i, B_i}.
    phi_o = [[0.0] * n for _ in range(m)]
    phi_c = [[-math.inf] * n for _ in range(m)]
    for i in range(m):
        anchor = anchors[i]
        for j in range(n):
            start, end = intervals[j]
            duration = max(1e-6, end - start)
            if anchor is not None:
                if i in subdiv_targets:
                    gs = float(anchor.get("group_start", anchor["start"]))
                    ge = float(anchor.get("group_end", anchor["end"]))
                    g_ov = max(0.0, min(end, ge) - max(start, gs))
                    if g_ov > 1e-6:
                        onset_cost = abs(j - subdiv_targets[i]) / max(1, n)
                    else:
                        onset_cost = abs(start - gs) / duration + 1.0
                else:
                    overlap = max(0.0, min(end, float(anchor["end"])) - max(start, float(anchor["start"])))
                    onset_cost = abs(start - float(anchor["start"])) / duration + 1.0 - overlap / duration
            else:
                base_cost = abs(j - baseline_onsets[i]) / max(1, n)
                ter_cost = 0.35 * (abs(start - w_lo[i]) / duration)
                onset_cost = 0.5 * base_cost + ter_cost
            phi_o[i][j] = -onset_cost

            if j == 0:
                continue
            gap = max(0.0, start - intervals[j - 1][1])
            if gap > g_phr:
                phi_c[i][j] = -math.inf
                continue
            if i in subdiv_melismas and (i in subdiv_not_last or start < subdiv_group_end[i] - 1e-6):
                phi_c[i][j] = 1.0 if j in subdiv_melismas[i] else -math.inf
                continue
            pitch_step = abs(float(notes[j].pitch) - float(notes[j - 1].pitch))
            ter_ov = max(0.0, min(end, w_hi[i]) - max(start, w_lo[i]))
            rho = ter_ov / duration
            if i in subdiv_not_last and start >= subdiv_not_last[i] - 1e-6:
                rho = 0.0
            e_ter = (
                math.log(max(rho, 1e-9) / 0.5)
                - max(0.0, gap - g_leg) / tau_gap
                - ((1.0 if gap > g_leg else 0.0) * max(0.0, pitch_step - MELISMA_STEP_FREE))
            )
            if anchor is not None:
                overlap = max(0.0, min(end, float(anchor["end"])) - max(start, float(anchor["start"])))
                ov_ratio = overlap / duration
                tail_ext = max(0.0, start - float(anchor["end"]))
                tail_decay = math.exp(-tail_ext / max(tail_tau_floor, 0.45 * duration))
                lead = max(0.0, float(anchor["start"]) - start)
                e_anc = _melisma_log_affinity(ov_ratio, tail_decay, pitch_step, lead, tempo_scale)
                if gap > g_leg:
                    e_anc -= (gap - g_leg) / tau_gap
                raw_aff = max(e_ter, e_anc)
            else:
                if baseline[j] != i and gap > g_leg and pitch_step > MELISMA_STEP_FREE:
                    raw_aff = -2.0
                else:
                    raw_aff = e_ter
            phi_c[i][j] = max(-2.0, min(2.0, raw_aff))

    # State indexing: 0 = PRE_B; for token i in 0..m-1: 3*i+1 = O_i, 3*i+2 = C_i, 3*i+3 = B_i.
    n_states = 1 + 3 * m
    dp_v = [[-math.inf] * n_states for _ in range(n)]
    back = [[-1] * n_states for _ in range(n)]
    dp_v[0][0] = 0.0
    dp_v[0][1] = phi_o[0][0]
    for j in range(1, n):
        prev_v = dp_v[j - 1]
        cur_v = dp_v[j]
        cur_b = back[j]
        if prev_v[0] > -math.inf:
            cur_v[0] = prev_v[0]
            cur_b[0] = 0
            cur_v[1] = prev_v[0] + phi_o[0][j]
            cur_b[1] = 0
        for i in range(m):
            s_o = 3 * i + 1
            s_c = 3 * i + 2
            s_b = 3 * i + 3
            if i > 0:
                p_o = 3 * (i - 1) + 1
                p_c = 3 * (i - 1) + 2
                p_b = 3 * (i - 1) + 3
                best_p, best_idx = prev_v[p_o], p_o
                if prev_v[p_c] > best_p:
                    best_p, best_idx = prev_v[p_c], p_c
                if prev_v[p_b] > best_p:
                    best_p, best_idx = prev_v[p_b], p_b
                if best_p > -math.inf:
                    cur_v[s_o] = best_p + phi_o[i][j]
                    cur_b[s_o] = best_idx
            # C_i: contiguous continuation from O_i or C_i only
            if phi_c[i][j] > -math.inf:
                best_cont, cont_idx = prev_v[s_o], s_o
                if prev_v[s_c] > best_cont:
                    best_cont, cont_idx = prev_v[s_c], s_c
                if best_cont > -math.inf:
                    cur_v[s_c] = best_cont + phi_c[i][j]
                    cur_b[s_c] = cont_idx
            # B_i: blank after token i from O_i, C_i, or B_i
            best_any, any_idx = prev_v[s_o], s_o
            if prev_v[s_c] > best_any:
                best_any, any_idx = prev_v[s_c], s_c
            if prev_v[s_b] > best_any:
                best_any, any_idx = prev_v[s_b], s_b
            if best_any > -math.inf:
                cur_v[s_b] = best_any
                cur_b[s_b] = any_idx

    end_candidates = (3 * (m - 1) + 1, 3 * (m - 1) + 2, 3 * (m - 1) + 3)
    best_state = max(end_candidates, key=lambda st: dp_v[n - 1][st])
    lyrics, owners = [""] * n, [None] * n
    st = best_state
    for j in range(n - 1, -1, -1):
        if st > 0:
            tok_i = (st - 1) // 3
            kind = (st - 1) % 3
            if kind == 0:
                lyrics[j], owners[j] = tokens[tok_i]["text"], tok_i
            elif kind == 1:
                lyrics[j], owners[j] = "-", tok_i
        st = back[j][st]
    return lyrics, owners


def _join_tokens(texts):
    result, previous = "", ""
    for text in texts:
        if result and (re.fullmatch(_LATIN_PATTERN, previous)
                       or re.fullmatch(_LATIN_PATTERN, text)):
            result += " "
        result += text
        previous = text
    return result


def _use_raw_intervals(notes, origin):
    """Check if all notes have explicit non-negative raw_start and positive raw_duration intervals ordered from origin."""
    prev_raw_end = max(-1e-6, float(origin) - 0.5)
    has_any = False
    for note in notes:
        if getattr(note, "raw_start", None) is None or getattr(note, "raw_duration", None) is None:
            return False
        rs = float(note.raw_start)
        rd = float(note.raw_duration)
        if not (math.isfinite(rs) and math.isfinite(rd)) or rs < 0.0 or rd <= 0.0 or rs < prev_raw_end - 1e-6:
            return False
        prev_raw_end = rs + rd
        has_any = True
    return has_any


def align_lyrics_to_notes(
    project,
    raw_lyrics: str = "",
    word_timestamps: list[dict] | None = None,
    vocal_audio_path: str | Path | None = None,
) -> dict:
    """Align up to ``len(project.notes)`` lyric tokens to ``project.notes`` with 1-to-1 syllabic and 1-to-many melisma ('-') markers."""
    notes = list(project.notes)
    bpm = float(project.bpm)
    origin = float(getattr(project, "beat_origin", 0.0))
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(origin):
        raise ValueError("BPM must be finite and positive; origin must be finite")
    previous_end = -math.inf
    intervals = []
    spb = 60.0 / bpm
    raw_bar_starts = getattr(project, "bar_start_seconds", None)
    valid_bar_starts = None
    raw_beat_times = getattr(project, "beat_times", None)
    valid_beat_times = None
    raw_meter = getattr(project, "meter", None)
    meter_str = "4/4" if raw_meter is None or str(raw_meter).strip() == "" else str(raw_meter).strip()
    try:
        m_num, m_den = map(int, meter_str.split("/"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid meter: {raw_meter!r}") from exc
    if m_num <= 0 or m_den <= 0 or (m_den & (m_den - 1)) != 0:
        raise ValueError(f"Invalid meter: {raw_meter!r}")
    measure_beats = m_num * 4.0 / m_den
    if raw_bar_starts is not None and len(raw_bar_starts) >= 1:
        try:
            bs_list = [float(x) for x in raw_bar_starts]
        except (ValueError, TypeError) as exc:
            raise ValueError("bar_start_seconds must contain finite numbers") from exc
        if not all(math.isfinite(x) for x in bs_list) or not all(bs_list[k] > bs_list[k - 1] for k in range(1, len(bs_list))):
            raise ValueError("bar_start_seconds must be strictly increasing finite timestamps")
        valid_bar_starts = bs_list
    if raw_beat_times is not None and len(raw_beat_times) >= 2:
        try:
            bt_list = [float(x) for x in raw_beat_times]
        except (ValueError, TypeError) as exc:
            raise ValueError("beat_times must contain finite numbers") from exc
        if not all(math.isfinite(x) for x in bt_list) or not all(bt_list[k] > bt_list[k - 1] for k in range(1, len(bt_list))):
            raise ValueError("beat_times must be strictly increasing finite timestamps")
        valid_beat_times = bt_list
    from subpkgs.stem_mir.beat_tracker import beats_to_seconds_piecewise, seconds_to_beats_piecewise
    origin_beat = seconds_to_beats_piecewise(origin, valid_beat_times, bpm, origin) if valid_beat_times is not None else 0.0
    use_raw = _use_raw_intervals(notes, origin) and (valid_bar_starts is not None or valid_beat_times is not None or bool(getattr(project, "source", "")))
    for note in notes:
        start, duration = float(note.start), float(note.duration)
        if (not math.isfinite(start) or not math.isfinite(duration) or duration <= 0
                or start < previous_end - 1e-6 or not math.isfinite(float(note.pitch))
                or int(note.pitch) != note.pitch):
            raise ValueError("Notes must have finite pitches and ordered nonoverlapping positive durations")
        previous_end = start + duration
        if use_raw and not getattr(note, "edited", False):
            rs = float(note.raw_start)
            re = rs + float(note.raw_duration)
            if intervals and rs < intervals[-1][1]:
                rs = intervals[-1][1]
            intervals.append((rs, max(rs + 1e-4, re)))
        elif valid_beat_times is not None:
            p0 = beats_to_seconds_piecewise(origin_beat + start, valid_beat_times, bpm, origin)
            p1 = beats_to_seconds_piecewise(origin_beat + start + duration, valid_beat_times, bpm, origin)
            s_pt = max(intervals[-1][1], p0) if intervals else p0
            intervals.append((s_pt, max(s_pt + 1e-4, p1)))
        elif valid_bar_starts is not None:
            pts = []
            for b_val in (start, float(note.start + note.duration)):
                if b_val < 0.0:
                    pts.append(valid_bar_starts[0] + b_val * spb)
                else:
                    k_idx = int(b_val // measure_beats)
                    if k_idx < len(valid_bar_starts) - 1:
                        frac = (b_val - k_idx * measure_beats) / measure_beats
                        pts.append(valid_bar_starts[k_idx] + frac * (valid_bar_starts[k_idx + 1] - valid_bar_starts[k_idx]))
                    else:
                        rem_beats = b_val - (len(valid_bar_starts) - 1) * measure_beats
                        pts.append(valid_bar_starts[-1] + rem_beats * spb)
            s_pt = max(intervals[-1][1], pts[0]) if intervals else pts[0]
            intervals.append((s_pt, max(s_pt + 1e-4, pts[1])))
        else:
            s_pt = origin + start * spb
            e_pt = origin + float(note.start + note.duration) * spb
            if intervals and s_pt < intervals[-1][1]:
                s_pt = intervals[-1][1]
            intervals.append((s_pt, max(s_pt + 1e-4, e_pt)))
    resolved_raw_lyrics = raw_lyrics
    records = []
    if word_timestamps is None and vocal_audio_path is not None:
        transcription = transcribe_word_timestamps(vocal_audio_path, ground_truth_text=raw_lyrics)
        word_timestamps = transcription["word_timestamps"]
        if not resolved_raw_lyrics:
            resolved_raw_lyrics = transcription["raw_lyrics"]
    if word_timestamps is not None:
        records = _expand_word_timestamps(word_timestamps)
    if not resolved_raw_lyrics and records:
        resolved_raw_lyrics = _join_tokens([record["word"] for record in records])
    all_tokens = tokenize_lyrics(resolved_raw_lyrics)
    unsung_token_set: set[int] = set()
    if records and all_tokens and any(r.get("unsung", False) for r in records):
        if len(records) == len(all_tokens) and all(
            _token_match(all_tokens[i]["text"], records[i]["word"])[1] >= 0.99
            for i in range(len(all_tokens))
        ):
            unsung_token_set = {i for i, r in enumerate(records) if r.get("unsung", False)}
        else:
            sung_only = [r for r in records if not r.get("unsung", False)]
            if sung_only:
                reprojected = _project_records_to_ground_truth(sung_only, resolved_raw_lyrics)
                records = _expand_word_timestamps(reprojected)
                unsung_token_set = {
                    i for i, r in enumerate(records)
                    if i < len(all_tokens) and r.get("unsung", False)
                }
            else:
                unsung_token_set = set(range(len(all_tokens)))
    sung_records = [r for r in records if not r.get("unsung", False)]
    max_allow_gap = max(1.5, 2.0 * spb)
    all_anchors = _match_anchors(all_tokens, sung_records, intervals, max_allow_gap)
    n_notes = len(notes)
    g_leg_sec = max(MELISMA_LEGATO_GAP, 0.5 * spb)
    if n_notes == 0:
        selected_idx = []
    elif any(a is not None for a in all_anchors):
        anchored_scored = []
        for idx_t, anc in enumerate(all_anchors):
            if anc is not None and idx_t not in unsung_token_set:
                ov = max((max(0.0, min(ne, anc["end"]) - max(ns, anc["start"])) for ns, ne in intervals), default=0.0)
                min_gap = min((max(0.0, ns - anc["end"], anc["start"] - ne) for ns, ne in intervals), default=math.inf)
                if ov > 1e-6 or (len(all_tokens) - len(unsung_token_set) <= n_notes and min_gap <= max_allow_gap):
                    anchored_scored.append((1 if ov > 1e-6 else 0, ov, -min_gap, -idx_t, idx_t))
        if len(anchored_scored) > n_notes:
            anchored_scored = sorted(anchored_scored, reverse=True)[:n_notes]
        kept_anchored = sorted(item[4] for item in anchored_scored)
        kept_set = set(kept_anchored)
        rejected_anchor_set = {idx_t for idx_t, anc in enumerate(all_anchors) if anc is not None and idx_t not in kept_set}
        k_anc = len(kept_anchored)
        if k_anc > 0:
            anc_tokens = [all_tokens[idx_t] for idx_t in kept_anchored]
            anc_records = [all_anchors[idx_t] for idx_t in kept_anchored]
            _, anc_base = _align_text_tokens(anc_tokens, notes)
            _, anc_owners = _align_anchored_tokens(anc_tokens, notes, intervals, anc_records, anc_base, spb=spb)
            anc_note_idx = [next((j_n for j_n, ow in enumerate(anc_owners) if ow == r_i), r_i) for r_i in range(k_anc)]
            selected_set = set(kept_anchored)
            for r_i in range(k_anc + 1):
                t_lo = 0 if r_i == 0 else kept_anchored[r_i - 1] + 1
                t_hi = len(all_tokens) if r_i == k_anc else kept_anchored[r_i]
                n_lo = 0 if r_i == 0 else anc_note_idx[r_i - 1] + 1
                if r_i == k_anc and not unsung_token_set and len(all_tokens) > n_notes:
                    while n_lo < n_notes and (intervals[n_lo][0] - intervals[n_lo - 1][1]) <= g_leg_sec:
                        n_lo += 1
                n_hi = n_notes if r_i == k_anc else anc_note_idx[r_i]
                cap = max(0, n_hi - n_lo)
                if cap > 0:
                    span_start_sec = float(intervals[n_lo][0])
                    span_end_sec = float(intervals[n_hi - 1][1])
                    eff_lo, eff_hi = t_lo, t_hi
                    for t_idx in range(t_lo, t_hi):
                        anc_t = all_anchors[t_idx]
                        if anc_t is not None:
                            if float(anc_t["end"]) <= span_start_sec + 1e-6:
                                eff_lo = max(eff_lo, t_idx + 1)
                            if float(anc_t["start"]) >= span_end_sec - 1e-6:
                                eff_hi = min(eff_hi, t_idx)
                    unanc_in_span = [
                        t_idx for t_idx in range(eff_lo, max(eff_lo, eff_hi))
                        if t_idx not in selected_set and t_idx not in rejected_anchor_set and t_idx not in unsung_token_set
                    ]
                    for t_idx in unanc_in_span[:cap]:
                        selected_set.add(t_idx)
            selected_idx = sorted(selected_set)[:n_notes]
        else:
            eligible_unanc = [
                idx_t for idx_t in range(len(all_tokens))
                if idx_t not in rejected_anchor_set and idx_t not in unsung_token_set
            ]
            selected_idx = eligible_unanc[:n_notes]
    else:
        eligible_all = [idx_t for idx_t in range(len(all_tokens)) if idx_t not in unsung_token_set]
        selected_idx = eligible_all[:n_notes]

    sub_tokens = [all_tokens[k] for k in selected_idx]
    sub_anchors = [all_anchors[k] for k in selected_idx]
    note_lyrics, sub_owners = _align_text_tokens(sub_tokens, notes)
    if any(anchor is not None for anchor in sub_anchors) and len(sub_tokens) <= n_notes:
        note_lyrics, sub_owners = _align_anchored_tokens(sub_tokens, notes, intervals, sub_anchors, sub_owners, spb=spb)
    owners = [selected_idx[o] if o is not None and o < len(selected_idx) else None for o in sub_owners]

    pairs, blocks, unaligned_tokens = [], [], []
    active_tokens = all_tokens
    for i, token in enumerate(active_tokens):
        indices = [j for j, owner in enumerate(owners) if owner == i]
        if indices:
            p_start = max(0.0, float(intervals[indices[0]][0]))
            p_end = max(p_start, float(intervals[indices[-1]][1]))
            pairs.append({"token_index": i, "text": token["text"],
                          "line_index": token["line_index"], "note_indices": indices,
                          "start": p_start, "end": p_end})
        else:
            anc = all_anchors[i]
            if anc is not None:
                p_start, p_end = float(anc["start"]), float(anc["end"])
            else:
                prev_end = pairs[-1]["end"] if pairs else (float(intervals[0][0]) if intervals else max(0.0, float(origin)))
                p_start, p_end = max(0.0, prev_end), max(0.0, prev_end)
            unaligned_entry = {"token_index": i, "text": token["text"],
                               "line_index": token["line_index"], "note_indices": [],
                               "start": p_start, "end": p_end, "unaligned": True}
            if i in unsung_token_set:
                unaligned_entry["reason"] = "unsung"
            pairs.append(unaligned_entry)
            unaligned_tokens.append(unaligned_entry)
    group = []
    for pair in pairs:
        group.append(pair)
        if active_tokens[pair["token_index"]]["is_line_end"] or pair is pairs[-1]:
            timed_in_group = [item for item in group if item["note_indices"] or all_anchors[item["token_index"]] is not None]
            span_items = timed_in_group if timed_in_group else group
            blk_start = min(item["start"] for item in span_items)
            blk_end = max(item["end"] for item in span_items)
            blocks.append({"text": _join_tokens([item["text"] for item in group]),
                           "line_index": group[0]["line_index"],
                           "start": blk_start, "end": blk_end,
                           "note_indices": list(dict.fromkeys(j for item in group for j in item["note_indices"]))})
            group = []
    return {"note_lyrics": note_lyrics, "lyrics_blocks": blocks,
            "aligned_pairs": pairs, "unaligned_tokens": unaligned_tokens,
            "raw_lyrics": resolved_raw_lyrics}


def detect_pre_lyric_vocalise_onsets(
    audio_path: str | Path | None,
    first_lyric_sec: float,
) -> list[float]:
    """Detect sustained pre-lyric human vocalise/humming phrase onsets before the first scripted word."""
    hop_sec = 0.050
    pre_guard_sec = 1.0
    min_seg_sec = 1.20
    min_seg_frames = int(round(min_seg_sec / hop_sec))
    if audio_path is None:
        return []
    first_sec = float(first_lyric_sec)
    if not math.isfinite(first_sec) or first_sec < pre_guard_sec + min_seg_sec:
        return []
    wav_p = Path(audio_path)
    if not wav_p.is_file():
        return []
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
    if arr.size == 0 or sr <= 0:
        return []
    mono = np.mean(arr.astype(np.float64), axis=1)
    hop = max(1, int(round(hop_sec * sr)))
    n_f = len(mono) // hop
    if n_f < min_seg_frames:
        return []
    f_rms = np.sqrt(np.mean(mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    act = f_rms[f_rms > 0.003]
    if act.size == 0:
        return []
    ref_rms = float(np.percentile(act, 80.0))
    thr = max(0.014, 0.25 * ref_rms)
    limit_f = min(n_f, int((first_sec - pre_guard_sec) / hop_sec))
    if limit_f < min_seg_frames:
        return []
    mask = f_rms[:limit_f] >= thr
    bridged = mask.copy()
    for i in range(1, max(1, limit_f - 3)):
        if bridged[i - 1] and not bridged[i]:
            nxt = np.where(mask[i : min(limit_f, i + 4)])[0]
            if nxt.size:
                bridged[i : i + int(nxt[0])] = True
    onsets: list[float] = []
    in_seg = False
    seg_s = 0
    for i in range(limit_f + 1):
        val = bool(bridged[i]) if i < limit_f else False
        if val and not in_seg:
            in_seg = True
            seg_s = i
        elif not val and in_seg:
            in_seg = False
            if (i - seg_s) >= min_seg_frames:
                onsets.append(round(seg_s * hop_sec, 3))
    return onsets


def filter_pre_lyric_bleed_notes(
    raw_notes: list,
    audio_path: str | Path | None,
    first_lyric_sec: float | None,
    beat_origin: float,
    pickup_tol_sec: float,
) -> list:
    """Reject low-energy instrumental bleed before the first lyric while retaining real sung vocalise/humming."""
    if not raw_notes:
        return []
    origin = float(beat_origin)
    if first_lyric_sec is None:
        return [n for n in raw_notes if float(n.raw_start) >= origin - 1e-6]
    cutoff = max(origin, float(first_lyric_sec) - float(pickup_tol_sec))
    wav_p = Path(audio_path) if audio_path is not None else None
    if wav_p is None or not wav_p.is_file():
        return [n for n in raw_notes if float(n.raw_start) >= cutoff]

    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
    if arr.size == 0 or sr <= 0:
        return [n for n in raw_notes if float(n.raw_start) >= cutoff]
    mono = np.mean(arr.astype(np.float64), axis=1)
    hop = max(1, int(round(0.050 * sr)))
    n_f = len(mono) // hop
    if n_f < 4:
        return [n for n in raw_notes if float(n.raw_start) >= cutoff]
    f_rms = np.sqrt(np.mean(mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    act = f_rms[f_rms > 0.003]
    ref_rms = float(np.percentile(act, 80.0)) if act.size else 0.05
    voc_thr = max(0.015, 0.28 * ref_rms)

    kept = []
    for n in raw_notes:
        ns = float(n.raw_start)
        nd = float(n.raw_duration)
        if ns < origin - 1e-6:
            continue
        if ns >= cutoff:
            kept.append(n)
            continue
        if nd < 0.20 or float(getattr(n, "confidence", 1.0)) < 0.55:
            continue
        s0 = max(0, min(len(mono) - 1, int(round(ns * sr))))
        s1 = max(s0 + 1, min(len(mono), int(round((ns + nd) * sr))))
        note_rms = float(np.sqrt(np.mean(mono[s0:s1] ** 2)))
        if note_rms >= voc_thr:
            kept.append(n)
    return kept
