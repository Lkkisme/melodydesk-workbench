"""Dual-Mode Lyric Tokenizer, ASR Timestamp Extractor (Qwen3-ASR / ForcedAligner + faster-whisper CUDA), and Parameter-Free DP Syllable-to-Note Forced Aligner."""
from __future__ import annotations
from pathlib import Path
import logging
import math
import re
from statistics import median

_CJK_PATTERN = (
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002fa1f"
    r"\u3041-\u3096\u30a1-\u30fa\uac00-\ud7af]"
)
_LATIN_PATTERN = r"[A-Za-z0-9']+"
_SCAN = re.compile(_CJK_PATTERN + "|" + _LATIN_PATTERN + r"|\r\n|[\r\n，,。！？!?;；、]")
_LOG = logging.getLogger(__name__)


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


def transcribe_word_timestamps(
    audio_path: str | Path,
    prefer_cuda: bool = True,
    ground_truth_text: str = "",
) -> dict:
    """Run Qwen3-ASR / Qwen3-ForcedAligner-0.6B (or faster-whisper CUDA fallback) to obtain character/word-level timestamps."""
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
            if re.search(r"[\u3041-\u3096\u30a1-\u30fa]", ground_truth_text):
                language = "Japanese"
            elif re.search(r"[\uac00-\ud7af]", ground_truth_text):
                language = "Korean"
            elif re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]", ground_truth_text):
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
        for j in range(onset + 1, stop):
            start, end = intervals[j]
            if anchor is not None:
                overlap = max(0.0, min(end, anchor["end"]) - max(start, anchor["start"]))
                if (start > intervals[j - 1][1] or overlap <= 0.0
                        or overlap < 0.25 * (end - start)):
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
    for note in notes:
        start, duration = float(note.start), float(note.duration)
        if (not math.isfinite(start) or not math.isfinite(duration) or duration <= 0
                or start < previous_end or not math.isfinite(float(note.pitch))
                or int(note.pitch) != note.pitch):
            raise ValueError("Notes must have finite pitches and ordered nonoverlapping positive durations")
        previous_end = start + duration
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
