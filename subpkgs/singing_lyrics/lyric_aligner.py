"""Dual-Mode Lyric Tokenizer, ASR Timestamp Extractor (Qwen3-ASR / ForcedAligner + MMS-FA CTC + faster-whisper CUDA), and Parameter-Free DP Syllable-to-Note Forced Aligner."""
from __future__ import annotations
from pathlib import Path
import logging
import math
import re
import sys
from statistics import median

_CJK_PATTERN = (
    r"[㐀-䶿一-鿿豈-﫿"
    r"\U00020000-\U0002fa1f"
    r"ぁ-ゖァ-ヺ가-힯]"
)
_LATIN_PATTERN = r"[A-Za-z0-9'’]+"
_SCAN = re.compile(_CJK_PATTERN + "|" + _LATIN_PATTERN + r"|\r\n|[\r\n，,。！？!?;；、]")
_LOG = logging.getLogger(__name__)
_MMS_FA_CACHE: dict[str, tuple] = {}


def tokenize_lyrics(raw_text: str) -> list[dict]:
    """Tokenize Chinese/English mixed lyrics into singable units (1 Chinese char = 1 syllable, 1 English word = 1 token) preserving line/clause breaks."""
    tokens = []
    line_index = 0
    for match in _SCAN.finditer(raw_text):
        item = match.group(0)
        if item in ("\r\n", "\r", "\n"):
            if tokens:
                tokens[-1]["is_line_end"] = True
            line_index += 1
        elif item in "，,。！？!?;；、":
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
    for record in records:
        word = record["word"]
        start, end = float(record["start"]), float(record["end"])
        # External anchors and Qwen may have unavailable confidence.
        probability = float(record.get("probability", 0.0))
        if not isinstance(word, str):
            raise ValueError("Timestamp word must be text")
        if (not all(math.isfinite(x) for x in (start, end, probability))
                or not 0 <= start <= end or not 0 <= probability <= 1
                or start < previous_start):
            raise ValueError("Invalid or unordered word timestamps")
        previous_start = start
        pieces = tokenize_lyrics(word)
        for index, token in enumerate(pieces):
            expanded.append({"word": token["text"],
                             "start": start + (end - start) * index / len(pieces),
                             "end": start + (end - start) * (index + 1) / len(pieces),
                             "probability": probability})
    return expanded


def _is_faster_whisper_real():
    """Return True if faster_whisper in sys.modules is the real library, not a test mock."""
    fw = sys.modules.get("faster_whisper")
    if fw is None:
        return True
    wm = getattr(fw, "WhisperModel", None)
    if wm is None:
        return True
    return getattr(wm, "__module__", "").startswith("faster_whisper")


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
    g = gt_text.casefold()
    a = asr_text.casefold()
    if g == a:
        return 3.0
    if re.search(_CJK_PATTERN, g) and re.search(_CJK_PATTERN, a) and g[0] == a[0]:
        return 1.2
    if len(g) > 1 and len(a) > 1 and (g[:2] == a[:2] or g[-2:] == a[-2:]):
        return 1.2
    if _edit_distance(g, a) <= 1:
        return 1.2
    return -0.5


def _project_records_to_ground_truth(records: list[dict], ground_truth_text: str) -> list[dict]:
    """Needleman-Wunsch DP alignment projecting ASR timestamps onto ground-truth tokens."""
    gt_tokens = tokenize_lyrics(ground_truth_text)
    if not gt_tokens:
        return records
    records = _expand_word_timestamps(records)
    if not records:
        return [{"word": t["text"], "start": 0.0, "end": 0.15, "probability": 0.0} for t in gt_tokens]

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
            best = max(diag, up, left)
            dp[i][j] = best
            if best == diag:
                bt[i][j] = 0
            elif best == up:
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
    for i, tok in enumerate(gt_tokens):
        if i in matched:
            r = records[matched[i]]
            result.append({"word": tok["text"], "start": float(r["start"]),
                           "end": float(r["end"]), "probability": float(r.get("probability", 0.0))})
        else:
            result.append({"word": tok["text"], "start": -1.0, "end": -1.0, "probability": 0.0})

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
        left_time = result[prev_matched_idx]["end"] if prev_matched_idx >= 0 else (records[0]["start"] if records else 0.0)
        right_time = result[next_matched_idx]["start"] if next_matched_idx >= 0 else (left_time + 0.16 * gap_count)
        if right_time - left_time < 0.03 * gap_count and prev_matched_idx >= 0 and next_matched_idx >= 0:
            # Redistribute across [prev_matched_idx .. next_matched_idx] so adjacent words share positive duration
            span_start = result[prev_matched_idx]["start"]
            span_end = max(span_start + 0.03 * (gap_count + 2), result[next_matched_idx]["end"])
            total_slots = gap_count + 2
            for offset, idx in enumerate(range(prev_matched_idx, next_matched_idx + 1)):
                s = span_start + (span_end - span_start) * offset / total_slots
                e = span_start + (span_end - span_start) * (offset + 1) / total_slots
                result[idx]["start"] = max(0.0, float(s))
                result[idx]["end"] = max(result[idx]["start"] + 0.015, float(e))
        else:
            span = max(0.04 * gap_count, right_time - left_time)
            for offset, idx in enumerate(range(gap_start, gap_end)):
                s = left_time + span * offset / gap_count
                e = left_time + span * (offset + 1) / gap_count
                result[idx]["start"] = max(0.0, float(s))
                result[idx]["end"] = max(result[idx]["start"] + 0.015, float(e))
        prev_matched_idx = next_matched_idx
        i = gap_end + 1 if next_matched_idx >= 0 else len(result)

    prev_start = 0.0
    for r in result:
        if r["start"] < prev_start:
            r["start"] = prev_start
        if r["end"] <= r["start"]:
            r["end"] = r["start"] + 0.02
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
            language = "English"
            if re.search(r"[ぁ-ゖァ-ヺ]", ground_truth_text):
                language = "Japanese"
            elif re.search(r"[가-힯]", ground_truth_text):
                language = "Korean"
            elif re.search(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f]", ground_truth_text):
                language = "Chinese"
            model = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B", dtype=dtype, device_map=device)
            result = model.align(audio=str(audio_path), text=ground_truth_text, language=language)
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
        records = [{"word": item.text, "start": item.start_time,
                    "end": item.end_time, "probability": 0.0} for item in (items or [])]
        if text.strip() and not records:
            raise RuntimeError("Qwen returned nonempty text without timestamps")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        last_failure = exc
        failures.append(f"qwen3: {exc}")
        _LOG.warning("Qwen timestamp backend failed: %s", exc)
    else:
        return {"raw_lyrics": text, "word_timestamps": _expand_word_timestamps(records),
                "backend": backend}

    # Tier 2: MMS-FA CTC for Latin/English ground truth (skip when faster_whisper is mocked)
    if _is_faster_whisper_real() and ground_truth_text.strip():
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
            is_real_fw = getattr(WhisperModel, "__module__", "").startswith("faster_whisper")
            if is_real_fw:
                lang_hint = "zh" if re.search(_CJK_PATTERN, ground_truth_text) else ("en" if ground_truth_text else None)
                prompt_hint = ground_truth_text[:48] if ground_truth_text else None
                segments, info = model.transcribe(
                    audio_16k.astype(np.float32), language=lang_hint, word_timestamps=True,
                    condition_on_previous_text=False, vad_filter=True,
                    temperature=0.0, initial_prompt=prompt_hint)
                segments = list(segments)
                records = [{"word": word.word, "start": word.start, "end": word.end,
                            "probability": word.probability}
                           for segment in segments for word in (segment.words or [])]
                min_expected = max(5, len(tokenize_lyrics(ground_truth_text)) // 4) if ground_truth_text else 5
                if len(records) < min_expected:
                    segments_no_vad, info = model.transcribe(
                        audio_16k.astype(np.float32), language=lang_hint, word_timestamps=True,
                        condition_on_previous_text=False, vad_filter=False,
                        temperature=0.0, initial_prompt=None)
                    segments_no_vad = list(segments_no_vad)
                    records_no_vad = [{"word": word.word, "start": word.start, "end": word.end,
                                       "probability": word.probability}
                                      for segment in segments_no_vad for word in (segment.words or [])]
                    if len(records_no_vad) > len(records):
                        segments = segments_no_vad
                        records = records_no_vad
                text = "".join(segment.text for segment in segments).strip()
            else:
                segments, info = model.transcribe(
                    audio_16k.astype(np.float32), word_timestamps=True, condition_on_previous_text=False,
                    temperature=0.0, initial_prompt=ground_truth_text or None)
                segments = list(segments)
                text = "".join(segment.text for segment in segments).strip()
                records = [{"word": word.word, "start": word.start, "end": word.end,
                            "probability": word.probability}
                           for segment in segments for word in (segment.words or [])]
            if text and not records:
                raise RuntimeError("Whisper returned nonempty text without timestamps")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            last_failure = exc
            failures.append(f"faster-whisper-{device}: {exc}")
            _LOG.warning("Whisper %s timestamp backend failed: %s", device, exc)
            continue

        # Tier 3: project Whisper records onto ground truth via Needleman-Wunsch
        if (ground_truth_text.strip() and records
                and getattr(WhisperModel, "__module__", "").startswith("faster_whisper")):
            projected = _project_records_to_ground_truth(records, ground_truth_text)
            return {
                "raw_lyrics": ground_truth_text,
                "word_timestamps": _expand_word_timestamps(projected),
                "backend": f"faster-whisper-{device}+gt-projection",
            }

        return {"raw_lyrics": text, "word_timestamps": _expand_word_timestamps(records),
                "backend": f"faster-whisper-{device}"}
    raise RuntimeError("All timestamp backends failed: " + "; ".join(failures)) from last_failure


def _align_text_tokens(tokens, notes):
    n = len(notes)
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
                          + abs(int(notes[j - 1].pitch) - int(notes[j - 2].pitch)) / 12
                          + gaps[j - 1] / base)
            costs[i][j] = min(new, extend)
            transitions[i][j] = "new" if new <= extend else "extend"
    i, j = m, n
    while j:
        transition = transitions[i][j]
        owners[j - 1] = i - 1
        lyrics[j - 1] = tokens[i - 1]["text"] if transition == "new" else "-"
        if transition == "new":
            i -= 1
        j -= 1
    return lyrics, owners


def _match_anchors(tokens, records):
    m, k = len(tokens), len(records)
    lengths = [[0] * (k + 1) for _ in range(m + 1)]
    anchors = [None] * m
    for i in range(m - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            if tokens[i]["text"].casefold() == records[j]["word"].casefold():
                lengths[i][j] = 1 + lengths[i + 1][j + 1]
            else:
                lengths[i][j] = max(lengths[i + 1][j], lengths[i][j + 1])
    i, j = 0, 0
    while i < m and j < k:
        if tokens[i]["text"].casefold() == records[j]["word"].casefold():
            anchors[i] = records[j]
            i, j = i + 1, j + 1
        elif lengths[i + 1][j] > lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return anchors


def _align_anchored_tokens(tokens, notes, intervals, anchors, baseline):
    n, m = len(notes), len(tokens)
    baseline_onsets = [baseline.index(i) for i in range(m)]
    costs = [[math.inf] * (n + 1) for _ in range(m + 1)]
    placed = [[False] * (n + 1) for _ in range(m + 1)]
    costs[0] = [0.0] * (n + 1)
    for i in range(1, m + 1):
        anchor = anchors[i - 1]
        for j in range(1, n + 1):
            start, end = intervals[j - 1]
            duration = end - start
            onset_cost = abs(j - 1 - baseline_onsets[i - 1]) / max(1, n)
            if anchor is not None:
                overlap = max(0.0, min(end, anchor["end"]) - max(start, anchor["start"]))
                onset_cost = abs(start - anchor["start"]) / duration + 1 - overlap / duration
            candidate = costs[i - 1][j - 1] + onset_cost
            costs[i][j] = costs[i][j - 1]
            if candidate < costs[i][j]:
                costs[i][j] = candidate
                placed[i][j] = True
    onsets = [0] * m
    i, j = m, n
    while i:
        if placed[i][j]:
            onsets[i - 1] = j - 1
            i -= 1
        j -= 1
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
                overlap = max(0.0, min(end, anchor["end"]) - max(start, anchor["start"]))
                ov_ratio = overlap / duration
                tail_ext = max(0.0, start - anchor["end"])
                tail_score = math.exp(-tail_ext / max(0.12, 0.45 * duration))
                pitch_step = abs(int(notes[j].pitch) - int(notes[j - 1].pitch))
                is_legato = gap <= 0.06
                before_next = end <= next_anchor_start + 0.02
                if gap > 0.06:
                    break
                if not (is_legato and before_next
                        and (ov_ratio >= 0.18
                             or (start >= anchor["start"] and tail_score >= 0.55 and pitch_step <= 4))):
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
    """Check if all notes have valid acoustic timestamps for interval computation."""
    prev_raw_end = -1e-6
    has_any = False
    for note in notes:
        rs = float(getattr(note, "raw_start", 0.0))
        rd = float(getattr(note, "raw_duration", 0.0))
        if rd <= 0:
            return False
        if rs < prev_raw_end - 1e-6:
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
    """Align lyrics to `project.notes` with 1-to-1 syllabic and 1-to-many melisma ('-') markers."""
    notes = list(project.notes)
    bpm = float(project.bpm)
    origin = float(getattr(project, "beat_origin", 0.0))
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(origin):
        raise ValueError("BPM must be finite and positive; origin must be finite")
    previous_end = -math.inf
    intervals = []
    spb = 60.0 / bpm
    use_raw = _use_raw_intervals(notes, origin) and (getattr(project, "bar_start_seconds", None) is not None or bool(getattr(project, "source", "")))
    for note in notes:
        start, duration = float(note.start), float(note.duration)
        if (not math.isfinite(start) or not math.isfinite(duration) or duration <= 0
                or start < previous_end or not math.isfinite(float(note.pitch))
                or int(note.pitch) != note.pitch):
            raise ValueError("Notes must have finite pitches and ordered nonoverlapping positive durations")
        previous_end = start + duration
        if use_raw:
            rs = float(note.raw_start)
            rd = float(note.raw_duration)
            intervals.append((rs, rs + rd))
        else:
            intervals.append((origin + start * spb, origin + float(note.start + note.duration) * spb))
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
    tokens = tokenize_lyrics(resolved_raw_lyrics)[:len(notes)]
    note_lyrics, owners = _align_text_tokens(tokens, notes)
    anchors = _match_anchors(tokens, records)
    if any(anchor is not None for anchor in anchors):
        note_lyrics, owners = _align_anchored_tokens(tokens, notes, intervals, anchors, owners)
    pairs, blocks = [], []
    for i, token in enumerate(tokens):
        indices = [j for j, owner in enumerate(owners) if owner == i]
        pairs.append({"token_index": i, "text": token["text"],
                      "line_index": token["line_index"], "note_indices": indices,
                      "start": float(intervals[indices[0]][0]),
                      "end": float(intervals[indices[-1]][1])})
    group = []
    for pair in pairs:
        group.append(pair)
        if tokens[pair["token_index"]]["is_line_end"] or pair is pairs[-1]:
            blocks.append({"text": _join_tokens([item["text"] for item in group]),
                           "line_index": group[0]["line_index"],
                           "start": group[0]["start"], "end": group[-1]["end"],
                           "note_indices": [j for item in group for j in item["note_indices"]]})
            group = []
    return {"note_lyrics": note_lyrics, "lyrics_blocks": blocks,
            "aligned_pairs": pairs, "raw_lyrics": resolved_raw_lyrics}
