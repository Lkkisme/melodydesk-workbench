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
        is_interp = bool(record.get("interpolated", False))
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


def _token_to_mms_chars(token_text: str, dict_chars: set[str]) -> str:
    text = token_text.lower()
    if re.search(_CJK_PATTERN, token_text):
        try:
            from pypinyin import Style, lazy_pinyin
            text = "".join(lazy_pinyin(token_text, style=Style.NORMAL)).lower()
        except ImportError:
            return ""
    return "".join(c for c in text if c in dict_chars)


def _try_mms_fa(audio_path: Path, ground_truth_text: str, prefer_cuda: bool) -> dict | None:
    """Tier 2: Meta MMS_FA CTC forced alignment for Latin/English and CJK text."""
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
        dict_chars = set(dictionary.keys()) - {"*"}
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

        token_spans = aligner(emission[0].cpu() if emission.device.type != "cpu" else emission[0], transcript_tokens)
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


def _nw_score(gt_text: str, asr_text: str) -> float:
    g = gt_text.casefold().strip()
    a = asr_text.casefold().strip()
    if not g or not a:
        return -2.0
    if g == a:
        return 3.0
    dist = _edit_distance(g, a)
    sim = max(0.0, 1.0 - dist / max(len(g), len(a)))
    if sim >= 0.5:
        return 1.2 * sim
    return -2.0


def _project_records_to_ground_truth(records: list[dict], ground_truth_text: str) -> list[dict]:
    """Needleman-Wunsch DP alignment projecting ASR timestamps onto ground-truth tokens."""
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

    matched = {}
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            matched[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and bt[i][j] == 1:
            i -= 1
        else:
            j -= 1

    result = []
    has_positive_match = False
    for i, tok in enumerate(gt_tokens):
        if i in matched:
            r = records[matched[i]]
            sc = _nw_score(tok["text"], r["word"])
            if sc > 0.0:
                has_positive_match = True
                sim = 1.0 if sc >= 2.99 else round(sc / 1.2, 4)
                is_exact = (tok["text"].casefold().strip() == r["word"].casefold().strip())
                is_interp = bool(r.get("interpolated", False)) or not is_exact
                rec = {"word": tok["text"], "start": float(r["start"]),
                       "end": float(r["end"]),
                       "probability": 0.0 if is_interp else round(float(r.get("probability", 0.0)) * sim, 4)}
                if is_interp:
                    rec["interpolated"] = True
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
        if r["end"] <= r["start"]:
            r["end"] = r["start"] + 1e-4
            r["probability"] = 0.0
            r["interpolated"] = True
        prev_start = r["start"]

    return result


def transcribe_word_timestamps(
    audio_path: str | Path,
    prefer_cuda: bool = True,
    ground_truth_text: str = "",
) -> dict:
    """Run Qwen3-ASR / Qwen3-ForcedAligner-0.6B (or MMS-FA CTC / faster-whisper CUDA fallback) to obtain character/word-level timestamps."""
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

    # Tier 2: MMS-FA CTC for Latin ground truth (skip when text is CJK/Kana/Hangul)
    if ground_truth_text.strip() and not re.search(_CJK_PATTERN, ground_truth_text):
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
    if ground_truth_text.strip() and re.search(_CJK_PATTERN, ground_truth_text):
        mms_cjk = _try_mms_fa(audio_path, ground_truth_text, prefer_cuda)
        if mms_cjk is not None:
            return mms_cjk
        failures.append("mms-fa-cjk: fell through")
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
    valid_records = [r for r in records if not r.get("interpolated", False)]
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
            if tokens[i]["text"].casefold() == valid_records[j]["word"].casefold():
                cand = 1.0 + rec_bonus[j] + lengths[i + 1][j + 1]
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


def _align_anchored_tokens(tokens, notes, intervals, anchors, baseline):
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
                    if offset_t in g_own:
                        subdiv_targets[idx_tok + offset_t] = g_notes[g_own.index(offset_t)]
                        subdiv_melismas[idx_tok + offset_t] = {
                            g_notes[k_n] for k_n, ow in enumerate(g_own) if ow == offset_t and g_lyr[k_n] == "-"
                        }
            idx_tok = end_tok
        else:
            idx_tok += 1
    costs = [[math.inf] * (n + 1) for _ in range(m + 1)]
    placed = [[False] * (n + 1) for _ in range(m + 1)]
    costs[0] = [0.0] * (n + 1)
    for i in range(1, m + 1):
        anchor = anchors[i - 1]
        for j in range(1, n + 1):
            start, end = intervals[j - 1]
            duration = max(1e-6, end - start)
            onset_cost = abs(j - 1 - baseline_onsets[i - 1]) / max(1, n)
            if anchor is not None:
                if (i - 1) in subdiv_targets:
                    gs = float(anchor.get("group_start", anchor["start"]))
                    ge = float(anchor.get("group_end", anchor["end"]))
                    g_ov = max(0.0, min(end, ge) - max(start, gs))
                    if g_ov > 1e-6:
                        onset_cost = abs((j - 1) - subdiv_targets[i - 1]) / max(1, n)
                    else:
                        onset_cost = abs(start - gs) / duration + 1.0
                else:
                    overlap = max(0.0, min(end, anchor["end"]) - max(start, anchor["start"]))
                    onset_cost = abs(start - anchor["start"]) / duration + 1 - overlap / duration
            candidate = costs[i - 1][j - 1] + onset_cost
            costs[i][j] = costs[i][j - 1]
            if candidate < costs[i][j]:
                costs[i][j] = candidate
                placed[i][j] = True
    onsets = [0] * m
    i, j = m, n
    while i and j > 0:
        if placed[i][j] or j == i:
            onsets[i - 1] = j - 1
            i -= 1
        j -= 1
    # Tempo-adaptive melisma scales: the empirical median note duration rescales the
    # legato gap and tail-decay floor; inside the reference band the scale is exactly 1.
    note_durs = [max(0.0, float(ne) - float(ns)) for ns, ne in intervals]
    med_note_dur = median(note_durs) if note_durs else MELISMA_DUR_BAND[0]
    if med_note_dur < MELISMA_DUR_BAND[0]:
        tempo_scale = max(0.6, med_note_dur / MELISMA_DUR_BAND[0])
    elif med_note_dur > MELISMA_DUR_BAND[1]:
        tempo_scale = min(1.5, med_note_dur / MELISMA_DUR_BAND[1])
    else:
        tempo_scale = 1.0
    legato_gap = MELISMA_LEGATO_GAP * tempo_scale
    tail_tau_floor = MELISMA_TAIL_TAU_FLOOR * tempo_scale
    lyrics, owners = [""] * n, [None] * n
    for i, onset in enumerate(onsets):
        lyrics[onset], owners[onset] = tokens[i]["text"], i
        stop = onsets[i + 1] if i + 1 < m else n
        anchor = anchors[i]
        next_anchor_start = math.inf
        if i + 1 < m and anchors[i + 1] is not None:
            next_anchor_start = anchors[i + 1]["start"]
        for j in range(onset + 1, stop):
            start, end = intervals[j]
            duration = max(1e-6, end - start)
            gap = start - intervals[j - 1][1]
            if anchor is not None:
                if i in subdiv_melismas:
                    if j not in subdiv_melismas[i]:
                        break
                    lyrics[j], owners[j] = "-", i
                    continue
                if gap > legato_gap:
                    break
                if end > next_anchor_start + MELISMA_NEXT_ANCHOR_TOL:
                    break
                overlap = max(0.0, min(end, anchor["end"]) - max(start, anchor["start"]))
                ov_ratio = overlap / duration
                tail_ext = max(0.0, start - anchor["end"])
                tail_decay = math.exp(-tail_ext / max(tail_tau_floor, 0.45 * duration))
                pitch_step = abs(float(notes[j].pitch) - float(notes[j - 1].pitch))
                lead = max(0.0, float(anchor["start"]) - start)
                if _melisma_log_affinity(ov_ratio, tail_decay, pitch_step, lead, tempo_scale) < 0.0:
                    break
            else:
                if gap > legato_gap or end > next_anchor_start + MELISMA_NEXT_ANCHOR_TOL:
                    break
                if baseline[j] != i and abs(float(notes[j].pitch) - float(notes[j - 1].pitch)) > MELISMA_STEP_FREE:
                    break
            lyrics[j], owners[j] = "-", i
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
    max_allow_gap = max(1.5, 2.0 * spb)
    all_anchors = _match_anchors(all_tokens, records, intervals, max_allow_gap)
    n_notes = len(notes)
    if n_notes == 0:
        selected_idx = []
    elif any(a is not None for a in all_anchors):
        anchored_scored = []
        for idx_t, anc in enumerate(all_anchors):
            if anc is not None:
                ov = max((max(0.0, min(ne, anc["end"]) - max(ns, anc["start"])) for ns, ne in intervals), default=0.0)
                min_gap = min((max(0.0, ns - anc["end"], anc["start"] - ne) for ns, ne in intervals), default=math.inf)
                if ov > 1e-6 or (len(all_tokens) <= n_notes and min_gap <= max_allow_gap):
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
            _, anc_owners = _align_anchored_tokens(anc_tokens, notes, intervals, anc_records, anc_base)
            anc_note_idx = [next((j_n for j_n, ow in enumerate(anc_owners) if ow == r_i), r_i) for r_i in range(k_anc)]
            selected_set = set(kept_anchored)
            for r_i in range(k_anc + 1):
                t_lo = 0 if r_i == 0 else kept_anchored[r_i - 1] + 1
                t_hi = len(all_tokens) if r_i == k_anc else kept_anchored[r_i]
                n_lo = 0 if r_i == 0 else anc_note_idx[r_i - 1] + 1
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
                        if t_idx not in selected_set and t_idx not in rejected_anchor_set
                    ]
                    for t_idx in unanc_in_span[:cap]:
                        selected_set.add(t_idx)
            selected_idx = sorted(selected_set)[:n_notes]
        else:
            eligible_unanc = [idx_t for idx_t in range(len(all_tokens)) if idx_t not in rejected_anchor_set]
            selected_idx = eligible_unanc[:n_notes]
    else:
        selected_idx = list(range(min(len(all_tokens), n_notes)))

    sub_tokens = [all_tokens[k] for k in selected_idx]
    sub_anchors = [all_anchors[k] for k in selected_idx]
    note_lyrics, sub_owners = _align_text_tokens(sub_tokens, notes)
    if any(anchor is not None for anchor in sub_anchors) and len(sub_tokens) <= n_notes:
        note_lyrics, sub_owners = _align_anchored_tokens(sub_tokens, notes, intervals, sub_anchors, sub_owners)
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
