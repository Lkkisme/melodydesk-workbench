"""Hermetic Pretrained ISMIR 2024 `MelodyT5` Symbolic Phrase-Variation Bridge (`melodyt5_bridge.py`).

Converts transcribed vocal melodies into key-signature-aware ABC notation, queries the local
pretrained `MelodyT5` hierarchical patch-level + character-level Transformer (`weights.pth`) with
zero `os.chdir()` or global `sys.path` side effects, validates musical bar/range invariants, caches
generated variations on disk + memory, and exports relative `(beat_offset, dur_beats, step_delta)`
phrase cells for singing-gap & interlude acoustic guitar solo generation.
"""
from __future__ import annotations

from fractions import Fraction as Fr
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import numpy as np

_PATCH_SIZE = 64
_PATCH_LENGTH = 256
_PATCH_NUM_LAYERS = 9
_CHAR_NUM_LAYERS = 3

_NAT = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LET = "CDEFGAB"
_LOF = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
_MODE = {
    "maj": 0, "ion": 0, "min": -3, "aeo": -3,
    "dor": -2, "mix": -1, "lyd": 1, "phr": -4, "loc": -5,
}
_CLEAN = [
    (r'"[^"]*"', ""),
    (r"![^!]*!", ""),
    (r"\{[^}]*\}", ""),
    (r"\[[A-Za-z]:[^\]]*\]", ""),
    (r"\[([\^_=]*[A-Ga-g][,']*)[^\]]*\]", r"\1"),
]
_TOK = re.compile(
    r"(?P<tup>\((?P<p>\d))|(?P<acc>\^\^|\^|__|_|=)?(?P<n>[A-Ga-gzx])(?P<oct>[,']*)"
    r"(?P<num>\d*)(?P<sl>/*)(?P<den>\d*)(?P<tie>\s*-)?|(?P<br>[<>])|(?P<bar>\|)"
)

_MODEL_LOCK = threading.RLock()
_MELODYT5_SINGLETON: tuple[object, object, object] | None = None
_MELODYT5_FAILED = False
_MEM_CACHE: dict[str, list[list[tuple[float, float, int]]]] = {}


def _resolve_melodyt5_dir() -> Path | None:
    env_p = os.environ.get("MELODYDESK_MELODYT5_DIR")
    candidates = []
    if env_p:
        candidates.append(Path(env_p))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root.parent.parent / "work" / "model-cache" / "melodyt5",
        repo_root / "model-cache" / "melodyt5",
        Path.home() / ".cache" / "melodydesk" / "melodyt5",
    ])
    for cand in candidates:
        if (cand / "weights.pth").is_file() and (cand / "random_model").is_dir():
            return cand.resolve()
    return None


def _get_field(note_obj, field: str):
    return note_obj[field] if isinstance(note_obj, dict) else getattr(note_obj, field)


_MAJ_FIFTHS_NAME = {
    -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#",
}
_MIN_FIFTHS_NAME = {
    -6: "Ebm", -5: "Bbm", -4: "Fm", -3: "Cm", -2: "Gm", -1: "Dm",
    0: "Am", 1: "Em", 2: "Bm", 3: "F#m", 4: "C#m", 5: "G#m", 6: "D#m",
}


def _wrapped_key_fifths(key: str) -> tuple[int, str]:
    m = re.match(r"\s*([A-Ga-g])([#b]?)\s*([A-Za-z]*)", str(key))
    mode_raw = m.group(3).lower() if m else ""
    mode = "min" if mode_raw == "m" else (mode_raw[:3] or "maj")
    if not m or mode not in _MODE:
        raise ValueError(f"Invalid key specification: {key!r}")
    fifths = _LOF[m.group(1).upper()] + 7 * {"#": 1, "b": -1, "": 0}[m.group(2)] + _MODE[mode]
    while fifths > 6:
        fifths -= 12
    while fifths < -6:
        fifths += 12
    return fifths, mode


def canonical_abc_key(key: str) -> str:
    """Normalize enharmonic sharp/flat tonics (e.g. `'A#'` -> `'Bb'`, `'A#m'` -> `'Bbm'`, `'Ddor'` -> `'Ddor'`) into standard ABC `K:`."""
    fifths, mode = _wrapped_key_fifths(key)
    if mode == "min":
        return _MIN_FIFTHS_NAME[fifths]
    if mode == "maj":
        return _MAJ_FIFTHS_NAME[fifths]
    tonic_lof = fifths - _MODE[mode]
    acc_idx, let_idx = divmod(tonic_lof + 1, 7)
    tonic_name = "FCGDAEB"[let_idx] + {-1: "b", 0: "", 1: "#"}[acc_idx]
    return f"{tonic_name}{mode}"


def key_signature(key: str) -> tuple[dict[str, int], float]:
    """Parse `'D'`, `'F#m'`, `'Bb'`, `'A#'`, `'Em'` -> `({letter: -1/0/+1}, line_of_fifths_center)`."""
    fifths, mode = _wrapped_key_fifths(key)
    sig = dict.fromkeys(_LET, 0)
    for ch in "FCGDAEB"[: max(0, fifths)]:
        sig[ch] = 1
    for ch in "BEADGCF"[: max(0, -fifths)]:
        sig[ch] = -1
    center = fifths + 2.25 + (1.0 if _MODE[mode] <= -2 else 0.0)
    return sig, center


def spell_midi(midi: int, center: float) -> tuple[str, int, int]:
    """Convert MIDI pitch -> `(letter, accidental -1..+1, letter_octave)`."""
    _, letter, acc = min(
        (abs(_LOF[ch] + 7 * a - center), ch, a)
        for ch in _LET
        for a in (0, 1, -1)
        if (_NAT[ch] + a - midi) % 12 == 0
    )
    return letter, acc, (midi - acc) // 12 - 1


def _abc_len(ticks_16th: int) -> str:
    frac = Fr(ticks_16th, 2)
    num_s = "" if frac.numerator == 1 else str(frac.numerator)
    return num_s if frac.denominator == 1 else f"{num_s}/{frac.denominator}"


def notes_to_abc(notes: list, meter: str, key: str, origin_beat: float = 0.0) -> str:
    """Quantize MelodyDesk notes (`start`, `duration` in quarter-beats, `pitch` MIDI) into valid `L:1/8` ABC."""
    sig, center = key_signature(key)
    num, den = map(int, str(meter).strip().split("/"))
    bar_16ths = max(4, num * 16 // den)
    grp_16ths = 6 if (den == 8 and num % 3 == 0) else 4

    quant = sorted(
        (
            round((float(_get_field(n, "start")) - origin_beat) * 4.0),
            round((float(_get_field(n, "start")) + float(_get_field(n, "duration")) - origin_beat) * 4.0),
            int(round(float(_get_field(n, "pitch")))),
        )
        for n in notes
    )
    if not quant:
        raise ValueError("Cannot convert empty note list to ABC")

    segs: list[tuple[int, int, int | None]] = []
    cur = (quant[0][0] // bar_16ths) * bar_16ths
    for idx, (s_q, e_q, pitch) in enumerate(quant):
        if idx + 1 < len(quant):
            e_q = min(e_q, quant[idx + 1][0])
        s_q = max(s_q, cur)
        if e_q <= s_q:
            continue
        if s_q > cur:
            segs.append((cur, s_q, None))
        segs.append((s_q, e_q, pitch))
        cur = e_q
    end_bar = -(-cur // bar_16ths) * bar_16ths
    if end_bar > cur:
        segs.append((cur, end_bar, None))

    out: list[str] = []
    state: dict[tuple[str, int], int] = {}
    for s_q, e_q, pitch in segs:
        a = s_q
        while a < e_q:
            b = min(e_q, (a // bar_16ths + 1) * bar_16ths)
            if a % bar_16ths == 0:
                state = {}
                if out:
                    out.append(" | ")
            elif (a % bar_16ths) % grp_16ths == 0:
                out.append(" ")
            if pitch is None:
                tok = "z"
            else:
                letter, acc, octv = spell_midi(pitch, center)
                tok = ""
                if state.get((letter, octv), sig[letter]) != acc:
                    tok = {1: "^", -1: "_", 0: "="}[acc]
                    state[(letter, octv)] = acc
                if octv >= 5:
                    tok += letter.lower() + ("'" * (octv - 5))
                else:
                    tok += letter + ("," * max(0, 4 - octv))
            out.append(tok + _abc_len(b - a) + ("-" if (pitch is not None and b < e_q) else ""))
            a = b
    return "".join(out) + " |"


def abc_to_notes(abc_text: str, meter: str, key: str, origin_beat: float = 0.0) -> list[dict]:
    """Parse `L:1/8` ABC body back into `[{'start': float, 'duration': float, 'pitch': int}, ...]`."""
    sig, _ = key_signature(key)
    num, den = map(int, str(meter).strip().split("/"))
    compound = (den == 8 and num % 3 == 0)
    body = "\n".join(
        line for line in str(abc_text).splitlines()
        if not re.match(r"\s*(%|[A-Za-z]:)", line)
    )
    for pat, rep in _CLEAN:
        body = re.sub(pat, rep, body)

    events: list[list] = []
    state: dict[tuple[str, int], int] = {}
    tie: tuple[tuple[str, int], int] | None = None
    tup_rem = 0
    tup_factor = Fr(1)
    brk = Fr(1)

    for m in _TOK.finditer(body):
        if m["bar"]:
            state = {}
            continue
        if m["tup"]:
            p = int(m["p"])
            if p >= 2:
                q_num = 3 if p in (2, 4, 8) else (2 if p in (3, 6) else (3 if compound else 2))
                tup_rem, tup_factor = p, Fr(q_num, p)
            continue
        if m["br"]:
            if events:
                events[-1][0] *= Fr(3, 2) if m["br"] == ">" else Fr(1, 2)
                brk = Fr(1, 2) if m["br"] == ">" else Fr(3, 2)
            continue
        if not m["n"]:
            continue
        dur_eighth = Fr(int(m["num"] or 1), int(m["den"]) if m["den"] else (2 ** len(m["sl"]))) * brk
        brk = Fr(1)
        if tup_rem > 0:
            dur_eighth *= tup_factor
            tup_rem -= 1
        if m["n"] in "zx":
            events.append([dur_eighth, None, False])
            tie = None
            continue
        up = m["n"].upper()
        octv = (5 if m["n"].islower() else 4) + m["oct"].count("'") - m["oct"].count(",")
        note_key = (up, octv)
        if m["acc"]:
            state[note_key] = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}[m["acc"]]
        midi = 12 * (octv + 1) + _NAT[up] + state.get(note_key, sig[up])
        if tie is not None and tie[0] == note_key and not m["acc"]:
            midi = tie[1]
        is_tied = bool(tie is not None and tie[1] == midi)
        events.append([dur_eighth, int(midi), is_tied])
        tie = (note_key, int(midi)) if m["tie"] else None

    out: list[dict] = []
    pos_eighth = Fr(0)
    for dur_e, midi, is_tied in events:
        if midi is not None:
            if is_tied and out:
                out[-1]["duration"] += float(dur_e) * 0.5
            else:
                out.append({
                    "start": origin_beat + float(pos_eighth) * 0.5,
                    "duration": float(dur_e) * 0.5,
                    "pitch": int(midi),
                })
        pos_eighth += dur_e
    return out


def phrase_cells(notes: list, key: str, cumulative: bool = False) -> list[tuple[float, float, int]]:
    """Convert notes into `[(beat_offset, dur_beats, diatonic_step_delta), ...]`."""
    if not notes:
        return []
    _, center = key_signature(key)
    diatonic_idx = [
        7 * octv + _LET.index(letter)
        for letter, _acc, octv in (spell_midi(int(round(float(_get_field(n, "pitch")))), center) for n in notes)
    ]
    t0 = float(_get_field(notes[0], "start"))
    cells: list[tuple[float, float, int]] = []
    for j, (n, d_i) in enumerate(zip(notes, diatonic_idx)):
        ref = diatonic_idx[0] if cumulative else diatonic_idx[max(0, j - 1)]
        cells.append((
            round(float(_get_field(n, "start")) - t0, 4),
            round(max(0.25, float(_get_field(n, "duration"))), 4),
            int(d_i - ref),
        ))
    return cells


class _Patchilizer:
    """Bar-patch tokenizer matching ISMIR 2024 MelodyT5 checkpoint layout (`PATCH_SIZE=64`)."""

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regex_pattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def split_bars(self, body: str) -> list[str]:
        bars = [b for b in re.split(self.regex_pattern, "".join(body)) if b]
        if not bars:
            return []
        if bars[0] in self.delimiters and len(bars) >= 2:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        paired = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        if len(bars) % 2 == 1:
            paired.append(bars[-1])
        return paired

    def bar2patch(self, bar: str, patch_size: int = _PATCH_SIZE) -> list[int]:
        patch = [self.bos_token_id] + [min(127, ord(c)) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch

    def patch2bar(self, patch: list[int]) -> str:
        return "".join(chr(idx) if idx > self.eos_token_id else "" for idx in patch if idx != self.eos_token_id)

    def encode(self, abc_code: str, add_special_patches: bool = False) -> list[list[int]]:
        from unidecode import unidecode
        lines = [ln for ln in unidecode(abc_code).split("\n") if ln]
        body = ""
        patches: list[list[int]] = []
        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ":") or line.startswith("%%")):
                if body:
                    bars = self.split_bars(body)
                    patches.extend(
                        self.bar2patch(bar + "\n" if idx == len(bars) - 1 else bar)
                        for idx, bar in enumerate(bars)
                    )
                    body = ""
                patches.append(self.bar2patch(line + "\n"))
            else:
                body += line + "\n"
        if body:
            patches.extend(self.bar2patch(bar) for bar in self.split_bars(body))
        if add_special_patches:
            bos_patch = [self.bos_token_id] * (_PATCH_SIZE - 1) + [self.eos_token_id]
            eos_patch = [self.bos_token_id] + [self.eos_token_id] * (_PATCH_SIZE - 1)
            patches = [bos_patch] + patches + [eos_patch]
        return patches[:_PATCH_LENGTH]

    def decode(self, patches: list[list[int]]) -> str:
        return "".join(self.patch2bar(p) for p in patches)


def _sample_next_token(probs: np.ndarray, top_p: float, top_k: int, temperature: float, rng: random.Random) -> int:
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 0.0)
    if p.sum() <= 0:
        return 2
    p /= p.sum()
    if 0 < top_k < len(p):
        top_idx = np.argpartition(p, -top_k)[-top_k:]
        mask = np.zeros_like(p)
        mask[top_idx] = p[top_idx]
        p = mask / max(1e-12, mask.sum())
    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        csum = np.cumsum(p[order])
        cutoff = int(np.searchsorted(csum, top_p)) + 1
        keep = order[: max(1, cutoff)]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / max(1e-12, mask.sum())
    if temperature != 1.0 and temperature > 1e-3:
        logits = np.log(np.maximum(p, 1e-15)) / float(temperature)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= max(1e-12, p.sum())
    r = rng.random()
    return int(min(len(p) - 1, np.searchsorted(np.cumsum(p), r)))


def _get_melodyt5_runtime():
    global _MELODYT5_SINGLETON, _MELODYT5_FAILED
    if _MELODYT5_SINGLETON is not None:
        return _MELODYT5_SINGLETON
    if _MELODYT5_FAILED:
        return None
    with _MODEL_LOCK:
        if _MELODYT5_SINGLETON is not None:
            return _MELODYT5_SINGLETON
        if _MELODYT5_FAILED:
            return None
        model_dir = _resolve_melodyt5_dir()
        if model_dir is None:
            _MELODYT5_FAILED = True
            return None
        try:
            import torch
            from transformers import EncoderDecoderModel, GPT2Config, GPT2LMHeadModel, PreTrainedModel
            from transformers import logging as hf_logging

            rand_dir = str(model_dir / "random_model")
            prev_hf_verb = hf_logging.get_verbosity()
            try:
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()

                class _PatchEnDec(PreTrainedModel):
                    def __init__(self, cfg):
                        super().__init__(cfg)
                        self.patch_embedding = torch.nn.Linear(_PATCH_SIZE * 128, cfg.n_embd)
                        self.base = EncoderDecoderModel.from_encoder_decoder_pretrained(
                            rand_dir, rand_dir, tie_encoder_decoder=True
                        )
                        self.base.config.pad_token_id = 0
                        self.base.config.decoder_start_token_id = 1

                    def forward(self, patches: torch.Tensor, decoder_patches: torch.Tensor) -> torch.Tensor:
                        p_oh = torch.nn.functional.one_hot(patches, num_classes=128).float()
                        p_emb = self.patch_embedding(p_oh.reshape(len(patches), -1, _PATCH_SIZE * 128))
                        d_oh = torch.nn.functional.one_hot(decoder_patches, num_classes=128).float()
                        d_emb = self.patch_embedding(d_oh.reshape(len(decoder_patches), -1, _PATCH_SIZE * 128))
                        return self.base(
                            inputs_embeds=p_emb,
                            decoder_inputs_embeds=d_emb,
                            output_hidden_states=True,
                        )["decoder_hidden_states"][-1]

                class _CharDec(PreTrainedModel):
                    def __init__(self, cfg):
                        super().__init__(cfg)
                        self.base = GPT2LMHeadModel(cfg)

                    def next_probs(self, encoded_patch: torch.Tensor, tokens: torch.Tensor) -> np.ndarray:
                        enc = encoded_patch.reshape(1, 1, -1)
                        tok_ids = tokens.reshape(1, -1)
                        emb = torch.nn.functional.embedding(tok_ids, self.base.transformer.wte.weight)
                        inp = torch.cat((enc, emb[:, 1:, :]), dim=1)
                        logits = self.base(inputs_embeds=inp).logits.squeeze(0)[-1]
                        return torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

                class _MelodyT5Wrapper(PreTrainedModel):
                    def __init__(self, enc_cfg, dec_cfg):
                        super().__init__(enc_cfg)
                        self.patch_level_decoder = _PatchEnDec(enc_cfg)
                        self.char_level_decoder = _CharDec(dec_cfg)

                p_cfg = GPT2Config(
                    num_hidden_layers=_PATCH_NUM_LAYERS,
                    max_length=_PATCH_LENGTH,
                    max_position_embeddings=_PATCH_LENGTH,
                    vocab_size=1,
                )
                c_cfg = GPT2Config(
                    num_hidden_layers=_CHAR_NUM_LAYERS,
                    max_length=_PATCH_SIZE,
                    max_position_embeddings=_PATCH_SIZE,
                    vocab_size=128,
                )
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = _MelodyT5Wrapper(p_cfg, c_cfg)
                ckpt = torch.load(str(model_dir / "weights.pth"), map_location="cpu", weights_only=False)
                raw_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                legacy_suffixes = (
                    ".attn.bias",
                    ".attn.masked_bias",
                    ".crossattention.bias",
                    ".crossattention.masked_bias",
                )
                clean_sd = {k: v for k, v in raw_sd.items() if not k.endswith(legacy_suffixes)}
                model.load_state_dict(clean_sd, strict=True)
                model = model.to(device)
                model.eval()
            finally:
                hf_logging.set_verbosity(prev_hf_verb)
            _MELODYT5_SINGLETON = (model, _Patchilizer(), device)
            return _MELODYT5_SINGLETON
        except Exception:
            _MELODYT5_FAILED = True
            return None


def generate_melodyt5_solo_cells(
    notes: list,
    meter: str,
    tonic: str,
    minor: bool = False,
    num_bars: int = 4,
) -> list[list[tuple[float, float, int]]]:
    """Generate 1-bar relative `(beat_offset, dur_beats, step_delta)` phrase cells from vocal melody via MelodyT5.

    Returns an empty list on synthetic/short test snippets (`< 12` notes) or if weights are absent,
    allowing the caller to blend or fall back cleanly without side effects.
    """
    if len(notes) < 12:
        return []
    num_m, den_m = map(int, str(meter).strip().split("/"))
    measure_beats = num_m * (4.0 / den_m)
    raw_key = str(tonic).strip()
    if minor and not raw_key.lower().endswith("m"):
        raw_key = raw_key + "m"
    key_str = canonical_abc_key(raw_key)

    # Extract the most representative 4-to-8 bar vocal excerpt (highest note density without long silence)
    sorted_notes = sorted(notes, key=lambda n: float(_get_field(n, "start")))
    first_bar = int(float(_get_field(sorted_notes[0], "start")) // measure_beats)
    last_bar = int(float(_get_field(sorted_notes[-1], "start")) // measure_beats)
    if last_bar - first_bar < 3:
        return []

    best_span_notes: list = []
    best_score = -1.0
    win_bars = min(8, max(4, last_bar - first_bar + 1))
    for b0 in range(first_bar, max(first_bar + 1, last_bar - win_bars + 2)):
        s_b = b0 * measure_beats
        e_b = (b0 + win_bars) * measure_beats
        sub = [
            n for n in sorted_notes
            if s_b - 1e-6 <= float(_get_field(n, "start")) < e_b - 1e-6
        ]
        if len(sub) < 6:
            continue
        active_bars = len({int(float(_get_field(n, "start")) // measure_beats) for n in sub})
        pitch_var = len({int(round(float(_get_field(n, "pitch")))) % 12 for n in sub})
        sc = active_bars * 2.0 + min(6, pitch_var) + 0.1 * len(sub)
        if sc > best_score:
            best_score = sc
            best_span_notes = sub
            best_origin = s_b

    if len(best_span_notes) < 6:
        return []

    abc_in = notes_to_abc(best_span_notes, meter, key_str, origin_beat=best_origin)
    target_bars = max(2, min(8, int(num_bars)))
    cache_id = hashlib.sha256(f"v2|{meter}|{key_str}|{target_bars}|{abc_in}".encode("utf-8")).hexdigest()[:24]

    with _MODEL_LOCK:
        if cache_id in _MEM_CACHE:
            return [list(c) for c in _MEM_CACHE[cache_id]]

    model_dir = _resolve_melodyt5_dir()
    cache_file = (model_dir / "phrase_cache_v1.json") if model_dir is not None else None
    if cache_file is not None and cache_file.is_file():
        try:
            disk_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if cache_id in disk_data:
                parsed_cells = [
                    [(float(o), float(d), int(s)) for o, d, s in bar_c]
                    for bar_c in disk_data[cache_id]
                ]
                with _MODEL_LOCK:
                    _MEM_CACHE[cache_id] = parsed_cells
                return [list(c) for c in parsed_cells]
        except Exception:
            pass

    rt = _get_melodyt5_runtime()
    if rt is None:
        return []
    model, patchilizer, device = rt

    import torch
    header = f"L:1/8\nM:{meter}\nK:{key_str}\n"
    input_abc = f"%%variation\n{header}{abc_in}\n"
    out_prefix = f"E:{target_bars}\n{header}"
    det_seed = int(cache_id[:8], 16) % 900000 + 17
    rng = random.Random(det_seed)

    with _MODEL_LOCK, torch.no_grad():
        patches = torch.tensor([patchilizer.encode(input_abc, add_special_patches=True)], device=device)
        dec_patches = torch.tensor([patchilizer.encode(out_prefix, add_special_patches=True)[:-1]], device=device)
        gen_bars: list[str] = []
        max_steps = target_bars + 4
        for _ in range(max_steps):
            if len(gen_bars) >= target_bars or dec_patches.shape[1] >= _PATCH_LENGTH - 2:
                break
            enc_patches = model.patch_level_decoder(patches, dec_patches)
            enc_last = enc_patches[0, -1]
            tok_list = [patchilizer.bos_token_id]
            while len(tok_list) < _PATCH_SIZE - 1:
                tok_tensor = torch.tensor(tok_list, dtype=torch.long, device=device)
                probs = model.char_level_decoder.next_probs(enc_last, tok_tensor)
                nxt = _sample_next_token(probs, top_p=0.78, top_k=5, temperature=0.85, rng=rng)
                tok_list.append(nxt)
                if nxt == patchilizer.eos_token_id:
                    break
            if tok_list[1] == patchilizer.eos_token_id:
                break
            bar_str = patchilizer.patch2bar(tok_list[1:])
            if not bar_str:
                break
            if not bar_str.startswith(("S:", "B:", "E:", "L:", "M:", "K:")):
                gen_bars.append(bar_str)
            bar_patch = torch.tensor([patchilizer.bar2patch(bar_str)], dtype=torch.long, device=device)
            dec_patches = torch.cat([dec_patches, bar_patch.unsqueeze(0)], dim=1)

    gen_abc = "".join(gen_bars)
    parsed_notes = abc_to_notes(gen_abc, meter, key_str, origin_beat=0.0)
    if len(parsed_notes) < 3:
        return []
    pitches = [n["pitch"] for n in parsed_notes]
    if max(pitches) - min(pitches) > 19:
        return []

    bar_cells: list[list[tuple[float, float, int]]] = []
    for b_idx in range(target_bars):
        b_s = b_idx * measure_beats
        b_e = (b_idx + 1) * measure_beats
        b_notes = [
            dict(
                start=max(0.0, n["start"] - b_s),
                duration=min(b_e - max(b_s, n["start"]), n["duration"]),
                pitch=n["pitch"],
            )
            for n in parsed_notes
            if b_s - 1e-4 <= n["start"] < b_e - 0.12
        ]
        if 2 <= len(b_notes) <= 5:
            cell = phrase_cells(b_notes, key_str, cumulative=False)
            steps_only = [st for _o, _d, st in cell[1:]]
            has_sawtooth = (
                len(steps_only) >= 3
                and all(
                    steps_only[m] * steps_only[m + 1] < 0 and abs(steps_only[m]) >= 2
                    for m in range(len(steps_only) - 1)
                )
            )
            if not has_sawtooth and all(abs(st) <= 4 for st in steps_only):
                bar_cells.append(cell)

    if not bar_cells:
        return []

    with _MODEL_LOCK:
        _MEM_CACHE[cache_id] = bar_cells
        if cache_file is not None:
            try:
                existing = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else {}
                existing[cache_id] = [[list(item) for item in bc] for bc in bar_cells]
                tmp_file = cache_file.with_suffix(".tmp")
                tmp_file.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
                tmp_file.replace(cache_file)
            except Exception:
                pass
    return [list(c) for c in bar_cells]
