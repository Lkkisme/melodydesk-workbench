"""Hybrid Accompaniment CQT-Chroma Chord Recognizer & Vocal-Melody-Conditioned Harmony Fallback.

Plan A Hybrid Architecture:
  1. ``recognize_accompaniment_chords`` (primary when accompaniment stem has audible energy):
     Harmonic-percussive separation (``librosa.effects.harmonic``) -> tuning-calibrated 36-bin/octave
     CQT split into Bass Chroma (C1..F#3) and Harmonic Body Chroma (G2..G5) -> key-aware diatonic,
     modal-borrowed, secondary-dominant, and slash-inversion template scoring -> chroma-continuity
     Viterbi decoding + acoustic/vocal bar dynamics.
  2. ``synthesize_melody_conditioned_harmony`` (automatic fallback for pure a-cappella vocals):
     Dirichlet-smoothed functional Markov chain and Tonal Interval Space (TIS) Viterbi harmonizer
     driven purely by vocal melody notes, phrase cadences, and vocal register/density dynamics.

Transition model
----------------
The 7x7 functional-degree pseudo-count tables below are hand-specified expert
priors (they are NOT fitted to Hooktheory or any other corpus). Each row is
Dirichlet-smoothed with a symmetric concentration ``_DIRICHLET_ALPHA`` and
row-normalized into a proper stochastic matrix (``sum_j P_ij = 1``, ``P_ij > 0``).
The additive Viterbi transition weight is the scaled log-likelihood ratio
against a uniform chain, ``c * log(P_ij / (1/K))``, so a transition that is more
likely than chance scores > 0 and a less likely one scores < 0.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np

_DIRICHLET_ALPHA = 1.0
_TRANS_LOG_SCALE = 0.06

# Degree order (major): I, ii, iii, IV, V, vi, bVII/vii°.
_MAJOR_TRANS_COUNTS = np.array([
    [  2,  20,  10,  60,  60,  30,  10],
    [ 15,   2,  10,  20, 120,  15,  10],
    [ 10,  20,   2,  40,  15,  60,  10],
    [ 40,  15,  10,   2,  60,  15,  10],
    [160,  10,  10,  20,   2,  60,  10],
    [ 15,  40,  20,  60,  30,   2,  10],
    [ 30,  10,  10,  20,  15,  10,   2],
], dtype=np.float64)

# Degree order (minor, ascending scale degree 0..6): i, ii°, III, iv/IV, v/V, VI, VII.
_MINOR_TRANS_COUNTS = np.array([
    [  2,  10,  20,  40,  15,  15,  40],
    [ 10,   2,  15,  20,  40,  15,  15],
    [ 20,  10,   2,  30,  15,  20,  30],
    [ 20,  15,  15,   2,  40,  15,  20],
    [ 80,  10,  20,  20,   2,  30,  20],
    [ 15,  15,  20,  20,  15,   2,  30],
    [ 40,  10,  20,  15,  10,  15,   2],
], dtype=np.float64)

# Clean acoustic/pop/jazz extension qualities permitted in automatic vocal harmony synthesis.
_CLEAN_MELODY_EXT_QUALITIES = (
    "maj7", "m7", "7", "add9", "madd9", "sus2", "sus4", "7sus4", "6", "m6",
)

# Albrecht & Shanahan (2013) major/minor pitch-class profiles for concert key detection.
_AS_MAJOR_PROFILE = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081],
    dtype=np.float64,
)
_AS_MINOR_PROFILE = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.072, 0.043],
    dtype=np.float64,
)


def _dirichlet_row_normalize(counts: np.ndarray, alpha: float = _DIRICHLET_ALPHA) -> np.ndarray:
    """Posterior-mean transition matrix: P_ij = (n_ij + alpha) / sum_k (n_ik + alpha)."""
    smoothed = np.asarray(counts, dtype=np.float64) + float(alpha)
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _log_likelihood_ratio(probs: np.ndarray, scale: float = _TRANS_LOG_SCALE) -> np.ndarray:
    """Scaled log-likelihood ratio c * log(P_ij / P_uniform) of each transition."""
    p_uniform = 1.0 / probs.shape[1]
    return scale * np.log(probs / p_uniform)


MAJOR_TRANS_PROBS = _dirichlet_row_normalize(_MAJOR_TRANS_COUNTS)
MINOR_TRANS_PROBS = _dirichlet_row_normalize(_MINOR_TRANS_COUNTS)
MAJOR_TRANS_MATRIX = _log_likelihood_ratio(MAJOR_TRANS_PROBS)
MINOR_TRANS_MATRIX = _log_likelihood_ratio(MINOR_TRANS_PROBS)

import os
import sys
import threading

_BTC_LOCK = threading.RLock()
_BTC_RUNTIME: tuple[object, np.ndarray, np.ndarray, str] | None = None
_BTC_LOAD_FAILED = False
_BTC_DT_SEC = 10.0 / 108.0
_BTC_QUAL_TO_OFFSET = {
    "m": 0, "": 1, "dim": 2, "aug": 3, "m6": 4, "6": 5, "m7": 6,
    "mmaj7": 7, "maj7": 8, "7": 9, "dim7": 10, "m7b5": 11, "sus2": 12, "sus4": 13,
}
_BTC_EXT_PENALTY = {
    "": 0.0, "m": 0.0, "m7": 0.036, "7": 0.045, "maj7": 0.048,
    "sus4": 0.055, "7sus4": 0.062, "sus2": 0.068, "6": 0.088, "m6": 0.092,
    "m7b5": 0.090, "dim": 0.115, "mmaj7": 0.180, "aug": 0.210, "dim7": 0.220,
}


def _get_btc_runtime():
    """Load cached ISMIR 170-chord Bi-Directional Transformer (`btc_model_large_voca.pt`) if installed."""
    global _BTC_RUNTIME, _BTC_LOAD_FAILED
    if _BTC_RUNTIME is not None:
        return _BTC_RUNTIME
    if _BTC_LOAD_FAILED:
        return None
    with _BTC_LOCK:
        if _BTC_RUNTIME is not None:
            return _BTC_RUNTIME
        if _BTC_LOAD_FAILED:
            return None
        cands = []
        env_dir = os.environ.get("MELODYDESK_BTC_DIR")
        if env_dir:
            cands.append(Path(env_dir))
        repo_root = Path(__file__).resolve().parents[2]
        cands.extend([
            repo_root.parent.parent / "work" / "model-cache" / "btc-chord",
            repo_root / "model-cache" / "btc-chord",
            Path.home() / ".cache" / "melodydesk" / "btc-chord",
        ])
        btc_dir = next((p.resolve() for p in cands if (p / "btc_model_large_voca.pt").is_file() and (p / "btc_src").is_dir()), None)
        if btc_dir is None:
            _BTC_LOAD_FAILED = True
            return None
        try:
            import torch
            added_path = False
            if str(btc_dir) not in sys.path:
                sys.path.insert(0, str(btc_dir))
                added_path = True
            try:
                from btc_src.btc_model import BTC_model
            finally:
                if added_path and str(btc_dir) in sys.path:
                    sys.path.remove(str(btc_dir))
            cfg = dict(
                feature_size=144, timestep=108,
                input_dropout=0.2, layer_dropout=0.2, attention_dropout=0.2, relu_dropout=0.2,
                num_layers=8, num_heads=4, hidden_size=128,
                total_key_depth=128, total_value_depth=128, filter_size=128,
                loss="ce", probs_out=False, num_chords=170,
            )
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = BTC_model(config=cfg).to(dev)
            ckpt = torch.load(str(btc_dir / "btc_model_large_voca.pt"), map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            mean_arr = np.asarray(ckpt["mean"], dtype=np.float32)
            std_arr = np.asarray(ckpt["std"], dtype=np.float32)
            _BTC_RUNTIME = (model, mean_arr, std_arr, dev)
            return _BTC_RUNTIME
        except Exception:
            _BTC_LOAD_FAILED = True
            return None


def _compute_btc_frame_posteriors(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute 50%-overlapping Hann-windowed 170-chord frame log-posteriors and probabilities at 10.8 Hz."""
    if y.size < int(4.5 * sr):
        return None
    rt = _get_btc_runtime()
    if rt is None:
        return None
    model, mean_arr, std_arr, dev = rt
    try:
        import librosa
        import torch

        sr_btc = 22050
        wav = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr_btc) if sr != sr_btc else y.astype(np.float32)
        win_samples = int(sr_btc * 10.0)
        parts = []
        cur = 0
        while len(wav) > cur + win_samples:
            parts.append(librosa.cqt(wav[cur : cur + win_samples], sr=sr_btc, n_bins=144, bins_per_octave=24, hop_length=2048))
            cur += win_samples
        if cur < len(wav):
            parts.append(librosa.cqt(wav[cur:], sr=sr_btc, n_bins=144, bins_per_octave=24, hop_length=2048))
        if not parts:
            return None
        feat = np.concatenate(parts, axis=1)
        feat = np.log(np.abs(feat) + 1e-6).T.astype(np.float32)
        n_valid = feat.shape[0]
        if n_valid < 12:
            return None
        feat = (feat - mean_arr) / np.maximum(std_arr, 1e-6)

        win_len, stride = 108, 54
        starts = list(range(0, max(1, n_valid - win_len + 1), stride))
        if not starts or starts[-1] + win_len < n_valid:
            starts.append(max(0, n_valid - win_len))
        # Deduplicate start indices when n_valid <= 108
        starts = sorted(set(starts))
        chunks = []
        valid_lens = []
        for st in starts:
            sl = feat[st : st + win_len]
            v_len = len(sl)
            if v_len < win_len:
                sl = np.pad(sl, ((0, win_len - v_len), (0, 0)), mode="constant", constant_values=0.0)
            chunks.append(sl)
            valid_lens.append(v_len)

        batch_np = np.stack(chunks, axis=0)
        with _BTC_LOCK, torch.no_grad():
            bx = torch.tensor(batch_np, dtype=torch.float32, device=dev)
            attn_out, _ = model.self_attn_layers(bx)
            logits = model.output_layer.output_projection(attn_out)  # [B, 108, 170]
            log_p = torch.nn.functional.log_softmax(logits / 1.45, dim=-1).detach().cpu().numpy()
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        w_win = (np.sin(math.pi * (np.arange(win_len, dtype=np.float64) + 0.5) / win_len) ** 2)
        ell_acc = np.zeros((n_valid, 170), dtype=np.float64)
        prob_acc = np.zeros((n_valid, 170), dtype=np.float64)
        w_sum = np.zeros(n_valid, dtype=np.float64)
        for ch_i, st in enumerate(starts):
            vl = valid_lens[ch_i]
            wv = w_win[:vl]
            ell_acc[st : st + vl] += wv[:, None] * log_p[ch_i, :vl]
            prob_acc[st : st + vl] += wv[:, None] * probs[ch_i, :vl]
            w_sum[st : st + vl] += wv
        denom = np.maximum(w_sum[:, None], 1e-9)
        return ell_acc / denom, prob_acc / denom
    except Exception:
        return None


def _timing(bpm: float, beat_origin: float, meter: str) -> tuple[float, float, float]:
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be finite and positive")
    if not np.isfinite(beat_origin):
        raise ValueError("Beat origin must be finite")
    if not isinstance(meter, str) or "/" not in meter:
        raise ValueError("Unsupported meter")
    try:
        num_s, den_s = meter.strip().split("/", 1)
        num, den = int(num_s), int(den_s)
    except ValueError as exc:
        raise ValueError("Unsupported meter") from exc
    if num <= 0 or den <= 0 or (den & (den - 1)) != 0:
        raise ValueError("Unsupported meter")
    measure_beats = float(num) * (4.0 / float(den))
    seconds_per_beat = 60.0 / bpm
    return measure_beats, seconds_per_beat, measure_beats * seconds_per_beat


def _gttm_note_salience(rel_onset_beats: float, overlap_beats: float, measure_beats: float, meter: str = "4/4") -> float:
    sigma = 0.14
    if not isinstance(meter, str) or "/" not in meter:
        raise ValueError(f"Unsupported meter: {meter!r}")
    try:
        num_m, den_m = map(int, meter.strip().split("/", 1))
    except ValueError as exc:
        raise ValueError(f"Unsupported meter: {meter!r}") from exc
    if num_m <= 0 or den_m <= 0 or (den_m & (den_m - 1)) != 0:
        raise ValueError(f"Unsupported meter: {meter!r}")
    sub_unit = 4.0 / den_m
    is_compound = (den_m >= 8 and num_m % 3 == 0 and num_m >= 6)
    if is_compound:
        grid = [(0.0, 1.45)]
        n_comp_beats = num_m // 3
        comp_step = 3.0 * sub_unit
        for cb in range(1, n_comp_beats):
            w_cb = 1.30 if (n_comp_beats % 2 == 0 and cb == n_comp_beats // 2) else 1.25
            grid.append((cb * comp_step, w_cb))
        for s_idx in range(num_m):
            if s_idx % 3 != 0:
                grid.append((s_idx * sub_unit, 0.88))
    else:
        grid = [(0.0, 1.45)]
        if num_m % 2 == 0 and num_m >= 2:
            grid.append((measure_beats * 0.5, 1.22))
        for q in range(1, num_m):
            grid.append((q * sub_unit, 1.05))
        for q in range(num_m * 2):
            grid.append((q * sub_unit * 0.5, 0.85))
    # Circular (von Mises) metrical kernel: the phase coherence term
    # cos(2*pi*(rel_onset - pos) / measure) makes salience smooth and periodic in
    # the bar; kappa is matched so the kernel equals a Gaussian of width sigma
    # near each grid position.
    period = measure_beats if measure_beats > 0 else 4.0
    kappa = (period / (2.0 * math.pi * sigma)) ** 2
    max_salience = 0.70
    for pos, weight in grid:
        phase_coherence = math.cos(2.0 * math.pi * (rel_onset_beats - pos) / period)
        val = weight * math.exp(kappa * (phase_coherence - 1.0))
        if val > max_salience:
            max_salience = val
    dur_factor = min(1.45, max(0.35, 0.40 + 0.65 * math.log(1 + 2.2 * overlap_beats)))
    return max_salience * dur_factor


# Bernardes et al. (2016, JNMR Table 1) 6-component DFT weights for k=1..6 pitch-class interval cycles.
_W_TIS = np.array([2.0, 11.0, 17.0, 16.0, 19.0, 7.0], dtype=np.float64)


def _tiv(chroma: np.ndarray) -> np.ndarray:
    """Compute a 12-D Tonal Interval Vector from a 12-bin chroma using DFT k=1..6 weights _W_TIS."""
    chroma_12 = np.asarray(chroma, dtype=np.float64)
    if chroma_12.ndim != 1 or len(chroma_12) != 12:
        raise ValueError("chroma must be a 12-element 1-D pitch-class vector")
    total = float(np.sum(chroma_12))
    if total <= 1e-12:
        return np.zeros(12, dtype=np.float64)
    dft = np.fft.rfft(chroma_12 / total)[1:7] * _W_TIS
    return np.concatenate([dft.real, dft.imag])


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, returning 0.0 when either norm is near zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _degree_index(root: int, quality: str, tonic_root: int, is_minor_key: bool, tones: tuple[int, ...] = ()) -> int:
    """Map chord root/quality to scale-degree index 0..6 (or -1 when routed to the interval fallback).

    In minor keys, degree 1 at offset 2 requires a diminished chord (ii°) and degree 6 accepts
    either a non-diminished subtonic at offset 10 (VII) or a diminished leading-tone chord at
    offset 11 (vii°); in major keys, degree 6 at offset 11 requires a diminished chord (vii°).
    """
    offset = (root - tonic_root) % 12
    ivs = {iv % 12 for iv in tones} if tones else set()
    if not ivs:
        from subpkgs.guitar_reharm.chord_voicing import EXTENDED_INTERVALS
        ivs = {iv % 12 for iv in EXTENDED_INTERVALS.get(quality.strip(), (0, 4, 7))}
    is_dim = (3 in ivs and 6 in ivs and 4 not in ivs and 7 not in ivs)
    if is_minor_key:
        if (offset in (2, 11) and not is_dim) or (offset == 10 and is_dim):
            return -1
        return {0: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8: 5, 10: 6, 11: 6}.get(offset, -1)
    if offset == 11 and not is_dim:
        return -1
    return {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}.get(offset, -1)


def _parse_tonic_spec(tonic: str, minor: bool) -> tuple[int, bool, tuple[str, ...]]:
    """Parse tonic string (e.g. 'C', 'Am', 'F#') into (tonic_pc, is_minor, pitch_names_table)."""
    sharp_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    flat_names = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    if not isinstance(tonic, str):
        raise ValueError("Malformed tonic")
    t_clean = tonic.strip().replace("♯", "#").replace("♭", "b")
    if len(t_clean) >= 2 and t_clean[-1] in ("m", "M"):
        if t_clean[-1] == "m":
            minor = True
        t_clean = t_clean[:-1]
    if (
        not t_clean
        or t_clean[0].upper() not in "CDEFGAB"
        or len(t_clean) > 2
        or (len(t_clean) == 2 and t_clean[1] not in ("#", "b"))
    ):
        raise ValueError("Malformed tonic")
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[t_clean[0].upper()]
    r_pc = (base_pc + (1 if len(t_clean) == 2 and t_clean[1] == "#" else (-1 if len(t_clean) == 2 and t_clean[1] == "b" else 0))) % 12
    use_flats = ("b" in t_clean) or ("#" not in t_clean and ((not minor and r_pc in (5, 10, 3, 8, 1)) or (minor and r_pc in (2, 7, 0, 5, 10))))
    return r_pc, bool(minor), (flat_names if use_flats else sharp_names)


def _resolve_bar_boundaries(
    notes: list | None,
    bpm: float,
    beat_origin: float,
    measure_beats: float,
    seconds_per_beat: float,
    bar_seconds: float,
    clip_duration: float,
    bar_start_seconds: list[float] | None,
    beat_times: list[float] | None,
) -> tuple[list[float], list[float] | None, bool]:
    """Compute strictly increasing bar boundary timestamps [t_0, ..., t_n_bars]."""
    valid_bt: list[float] | None = None
    if beat_times is not None:
        if len(beat_times) < 2:
            raise ValueError("beat_times must contain at least 2 timestamps")
        bt_floats = [float(t) for t in beat_times]
        if any(not math.isfinite(t) for t in bt_floats):
            raise ValueError("beat_times must be finite")
        if any(bt_floats[i + 1] <= bt_floats[i] for i in range(len(bt_floats) - 1)):
            raise ValueError("beat_times must be strictly increasing")
        valid_bt = bt_floats
    from subpkgs.stem_mir.beat_tracker import beats_to_seconds_piecewise, seconds_to_beats_piecewise
    origin_beat = float(seconds_to_beats_piecewise(beat_origin, valid_bt, bpm, beat_origin)) if valid_bt is not None else 0.0

    note_list = notes or []
    max_note_end = max((n.start * seconds_per_beat + beat_origin + n.duration * seconds_per_beat
                        for n in note_list), default=0.0) if note_list else 0.0
    note_bars = math.ceil(max((float(n.start) + float(n.duration) for n in note_list), default=0.0) / measure_beats) if note_list else 0
    total_duration = max(clip_duration, max_note_end, beat_origin + bar_seconds)
    n_bars = max(1, note_bars, int(math.ceil((total_duration - beat_origin) / bar_seconds)))

    if bar_start_seconds is not None:
        if len(bar_start_seconds) < 2:
            raise ValueError("bar_start_seconds must contain at least 2 boundary timestamps")
        computed_bar_starts = [float(t) for t in bar_start_seconds]
        if any(not math.isfinite(t) for t in computed_bar_starts):
            raise ValueError("bar_start_seconds must be finite")
        if any(computed_bar_starts[i + 1] <= computed_bar_starts[i] for i in range(len(computed_bar_starts) - 1)):
            raise ValueError("bar_start_seconds must be strictly increasing")
        max_raw_end = max((float(n.raw_start) + float(n.raw_duration) for n in note_list if getattr(n, "raw_duration", 0.0) > 0), default=0.0) if note_list else 0.0
        non_raw_bars = math.ceil(max((float(n.start) + float(n.duration) for n in note_list if getattr(n, "raw_duration", 0.0) <= 0), default=0.0) / measure_beats) if note_list else 0
        target_end_sec = max(float(clip_duration), max_raw_end)
        while (len(computed_bar_starts) - 1 < max(1, non_raw_bars)) or (target_end_sec > computed_bar_starts[-1] + 1e-6):
            if valid_bt is not None:
                cand_sec = float(beats_to_seconds_piecewise(origin_beat + len(computed_bar_starts) * measure_beats, valid_bt, bpm, beat_origin))
                computed_bar_starts.append(cand_sec if cand_sec > computed_bar_starts[-1] + 1e-6 else computed_bar_starts[-1] + bar_seconds)
            else:
                computed_bar_starts.append(computed_bar_starts[-1] + bar_seconds)
    elif valid_bt is not None:
        max_raw_end = max((float(n.raw_start) + float(n.raw_duration) for n in note_list if getattr(n, "raw_duration", 0.0) > 0), default=0.0) if note_list else 0.0
        non_raw_bars = math.ceil(max((float(n.start) + float(n.duration) for n in note_list if getattr(n, "raw_duration", 0.0) <= 0), default=0.0) / measure_beats) if note_list else 0
        target_end_sec = max(float(clip_duration), max_raw_end, beat_origin + 1e-6)
        end_beats = max(
            float(non_raw_bars) * measure_beats,
            seconds_to_beats_piecewise(target_end_sec, valid_bt, bpm, beat_origin) - origin_beat,
        )
        n_bars = max(1, non_raw_bars, int(math.ceil(end_beats / measure_beats)))
        computed_bar_starts = [
            float(beats_to_seconds_piecewise(origin_beat + b * measure_beats, valid_bt, bpm, beat_origin))
            for b in range(n_bars + 1)
        ]
    else:
        computed_bar_starts = [beat_origin + b * bar_seconds for b in range(n_bars + 1)]

    custom_bounds = (bar_start_seconds is not None and len(bar_start_seconds) >= 2) or (valid_bt is not None)
    return computed_bar_starts, valid_bt, custom_bounds


def _extract_bar_vocal_profiles(
    notes: list | None,
    computed_bar_starts: list[float],
    valid_bt: list[float] | None,
    custom_bounds: bool,
    bpm: float,
    beat_origin: float,
    measure_beats: float,
    seconds_per_beat: float,
    bar_seconds: float,
    meter: str,
) -> tuple[list[np.ndarray], list[bool], list[float], list[float], int]:
    """Compute per-bar GTTM-weighted vocal pitch-class distributions and coverage stats."""
    from subpkgs.stem_mir.beat_tracker import seconds_to_beats_piecewise

    note_list = notes or []
    n_bars = len(computed_bar_starts) - 1
    bar_vocal_w: list[np.ndarray] = []
    bar_has_vocal: list[bool] = []
    bar_cov_ratios: list[float] = []
    bar_mean_pitches: list[float] = []
    first_vocal_bar = -1
    for b in range(n_bars):
        bs_sec = computed_bar_starts[b]
        be_sec = computed_bar_starts[b + 1]
        bar_dur_sec = max(1e-6, be_sec - bs_sec)
        bs_beat_musical = b * measure_beats
        be_beat_musical = (b + 1) * measure_beats
        bar_in_sync = (
            abs(bs_sec - (beat_origin + b * bar_seconds)) <= 1e-3
            and abs(bar_dur_sec - bar_seconds) <= 1e-3
        )
        vocal_w = np.zeros(12)
        cov_beats = 0.0
        pitches_in_bar = []
        for n in note_list:
            has_raw = getattr(n, "raw_duration", 0.0) > 0
            if custom_bounds and has_raw:
                ns_sec = float(n.raw_start)
                ne_sec = ns_sec + float(n.raw_duration)
                ov_sec = max(0.0, min(ne_sec, be_sec) - max(ns_sec, bs_sec))
                if ov_sec <= 0:
                    continue
                if valid_bt is not None:
                    b0 = seconds_to_beats_piecewise(max(ns_sec, bs_sec), valid_bt, bpm, beat_origin)
                    b1 = seconds_to_beats_piecewise(min(ne_sec, be_sec), valid_bt, bpm, beat_origin)
                    b_bar = seconds_to_beats_piecewise(bs_sec, valid_bt, bpm, beat_origin)
                    overlap = max(0.0, b1 - b0)
                    if overlap <= 0:
                        continue
                    rel_onset = max(0.0, b0 - b_bar) % measure_beats
                else:
                    ns_b = float(n.start)
                    ne_b = ns_b + float(n.duration)
                    ov_b = max(0.0, min(ne_b, be_beat_musical) - max(ns_b, bs_beat_musical))
                    nom_start = (ns_sec - beat_origin) / seconds_per_beat
                    nom_dur = (ne_sec - ns_sec) / seconds_per_beat
                    is_unwarped_nominal = (
                        abs(ns_b - nom_start) <= 1e-3
                        and abs(float(n.duration) - nom_dur) <= 1e-3
                    )
                    if ov_b > 0 and (not is_unwarped_nominal or bar_in_sync):
                        overlap = ov_b
                        rel_onset = (max(ns_b, bs_beat_musical) - bs_beat_musical) % measure_beats
                    else:
                        overlap = (ov_sec / bar_dur_sec) * measure_beats
                        rel_onset = ((max(ns_sec, bs_sec) - bs_sec) / bar_dur_sec) * measure_beats
            else:
                ns_b = float(n.start)
                ne_b = ns_b + float(n.duration)
                overlap = max(0.0, min(ne_b, be_beat_musical) - max(ns_b, bs_beat_musical))
                if overlap <= 0:
                    continue
                rel_onset = (max(ns_b, bs_beat_musical) - bs_beat_musical) % measure_beats
            cov_beats += overlap
            pitches_in_bar.append(float(n.pitch))
            salience = _gttm_note_salience(rel_onset, overlap, measure_beats, meter)
            vocal_w[int(round(float(n.pitch))) % 12] += salience
        has_vocal = float(np.sum(vocal_w)) > 0
        bar_vocal_w.append(vocal_w)
        bar_has_vocal.append(has_vocal)
        bar_cov_ratios.append(cov_beats / measure_beats)
        bar_mean_pitches.append(float(np.mean(pitches_in_bar)) if pitches_in_bar else 0.0)
        if has_vocal and first_vocal_bar < 0:
            first_vocal_bar = b
    return bar_vocal_w, bar_has_vocal, bar_cov_ratios, bar_mean_pitches, first_vocal_bar


def recognize_accompaniment_chords(
    accompaniment_path: str | Path | None,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    tonic: str | None = None,
    minor: bool | None = None,
    clip_duration: float = 0.0,
    bar_start_seconds: list[float] | None = None,
    beat_times: list[float] | None = None,
    notes: list | None = None,
    vocal_audio_path: str | Path | None = None,
) -> dict:
    """Recognize per-bar guitar/band chords directly from an accompaniment stem (`no_vocals.wav`).

    When `accompaniment_path` is missing or silent (a-cappella recording), returns
    `{"available": False, ...}` so the caller can fall back to `synthesize_melody_conditioned_harmony`.
    """
    import librosa

    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    if accompaniment_path is None or not Path(accompaniment_path).is_file():
        return {"available": False, "reason": "missing_accompaniment"}

    y, sr = librosa.load(str(accompaniment_path), sr=22050, mono=True)
    if y.size < sr // 2 or not np.all(np.isfinite(y)):
        return {"available": False, "reason": "empty_accompaniment"}

    overall_rms = float(np.sqrt(np.mean(y ** 2)))
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
    p90_rms = float(np.percentile(frame_rms, 90)) if frame_rms.size else 0.0
    if overall_rms < 0.004 or p90_rms < 0.008:
        return {"available": False, "reason": "accompaniment_silent"}

    audio_dur = len(y) / sr
    eff_clip_dur = max(float(clip_duration), audio_dur)
    computed_bar_starts, valid_bt, custom_bounds = _resolve_bar_boundaries(
        notes, bpm, beat_origin, measure_beats, seconds_per_beat, bar_seconds,
        eff_clip_dur, bar_start_seconds, beat_times,
    )
    n_bars = len(computed_bar_starts) - 1
    bar_vocal_w, bar_has_vocal, bar_cov_ratios, bar_mean_pitches, _ = _extract_bar_vocal_profiles(
        notes, computed_bar_starts, valid_bt, custom_bounds,
        bpm, beat_origin, measure_beats, seconds_per_beat, bar_seconds, meter,
    )

    # 1. Strip drums/percussion transients before CQT pitch-class analysis.
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    try:
        tuning = float(np.clip(librosa.estimate_tuning(y=y_harm, sr=sr), -0.45, 0.45))
    except Exception:
        tuning = 0.0

    hop_cqt = 512
    cqt_mag = np.abs(
        librosa.cqt(
            y=y_harm,
            sr=sr,
            hop_length=hop_cqt,
            fmin=float(librosa.note_to_hz("C1")),
            n_bins=216,
            bins_per_octave=36,
            tuning=tuning,
        )
    )
    n_frames = cqt_mag.shape[1]
    semi_mag = np.zeros((72, n_frames), dtype=np.float64)
    for m in range(72):
        b_c = 3 * m
        center = cqt_mag[b_c]
        left = cqt_mag[b_c - 1] if b_c > 0 else 0.0
        right = cqt_mag[b_c + 1] if b_c + 1 < 216 else 0.0
        semi_mag[m] = center + 0.45 * (left + right)

    ref_scale = float(np.percentile(semi_mag, 95)) + 1e-9
    semi_log = np.log1p(15.0 * (semi_mag / ref_scale))

    bass_chroma = np.zeros((12, n_frames), dtype=np.float64)
    for m in range(0, 31):  # C1 (32.7 Hz) .. F#3 (185 Hz)
        w = 1.0 if m <= 24 else 0.5
        bass_chroma[m % 12] += w * semi_log[m]

    body_chroma = np.zeros((12, n_frames), dtype=np.float64)
    for m in range(19, 68):  # G2 (98 Hz) .. G6 (1568 Hz)
        body_chroma[m % 12] += semi_log[m]

    frames_per_sec = sr / hop_cqt

    def _summarize_chroma_span(start_sec: float, end_sec: float) -> tuple[np.ndarray, np.ndarray, float]:
        s_t = max(0.0, float(start_sec))
        e_t = max(s_t + 0.05, float(end_sec))
        f0_s = max(0, min(n_frames - 1, int(round(s_t * frames_per_sec))))
        f1_s = max(f0_s + 1, min(n_frames, int(round(e_t * frames_per_sec))))
        s0_s = max(0, min(len(y) - 1, int(round(s_t * sr))))
        s1_s = max(s0_s + 1, min(len(y), int(round(e_t * sr))))
        rms_s = float(np.sqrt(np.mean(y[s0_s:s1_s] ** 2))) if s1_s > s0_s else 0.0
        sub_bdy = body_chroma[:, f0_s:f1_s]
        sub_bss = bass_chroma[:, f0_s:f1_s]
        r_body = 0.60 * np.median(sub_bdy, axis=1) + 0.40 * np.mean(sub_bdy, axis=1)
        r_bass = 0.55 * np.median(sub_bss, axis=1) + 0.45 * np.mean(sub_bss, axis=1)
        return r_body, r_bass, rms_s

    # 2. Concert Key Resolution (explicit tonic/minor if given, else Stage A Albrecht-Shanahan + Stage B Third-Polarity Mode Estimator).
    mode_evidence: dict = {"source": "explicit"}
    if tonic is not None and str(tonic).strip():
        is_min_flag = bool(minor) if minor is not None else ("m" in str(tonic).strip()[1:])
        tonic_root, is_minor, NAMES = _parse_tonic_spec(str(tonic), is_min_flag)
    else:
        glob_bass = np.sum(bass_chroma, axis=1)
        glob_body = np.sum(body_chroma, axis=1)
        glob_bass = glob_bass / (np.sum(glob_bass) + 1e-12)
        glob_body = glob_body / (np.sum(glob_body) + 1e-12)
        acc_joint_pc = 0.32 * glob_bass + 0.68 * glob_body
        acc_joint_pc = acc_joint_pc / (np.sum(acc_joint_pc) + 1e-12)
        joint_pc = acc_joint_pc.copy()
        if any(bar_has_vocal):
            voc_glob = np.sum(bar_vocal_w, axis=0)
            voc_glob = voc_glob / (np.sum(voc_glob) + 1e-12)
            joint_pc = 0.78 * joint_pc + 0.22 * voc_glob
            joint_pc = joint_pc / (np.sum(joint_pc) + 1e-12)

        best_key_sc = -np.inf
        tonic_root, is_minor = 0, False
        for r_cand in range(12):
            maj_prof = np.roll(_AS_MAJOR_PROFILE, r_cand)
            min_prof = np.roll(_AS_MINOR_PROFILE, r_cand)
            sc_maj = _cosine_sim(joint_pc, maj_prof) + 0.08 * float(glob_bass[r_cand])
            sc_min = _cosine_sim(joint_pc, min_prof) + 0.08 * float(glob_bass[r_cand])
            if minor is False:
                sc_min = -np.inf
            elif minor is True:
                sc_maj = -np.inf
            if sc_maj > best_key_sc:
                best_key_sc = sc_maj
                tonic_root, is_minor = r_cand, False
            if sc_min > best_key_sc:
                best_key_sc = sc_min
                tonic_root, is_minor = r_cand, True

        if minor is None and n_bars >= 1:
            acc_maj_sc = [
                _cosine_sim(acc_joint_pc, np.roll(_AS_MAJOR_PROFILE, r_c)) + 0.08 * float(glob_bass[r_c])
                for r_c in range(12)
            ]
            acc_min_sc = [
                _cosine_sim(acc_joint_pc, np.roll(_AS_MINOR_PROFILE, r_c)) + 0.08 * float(glob_bass[r_c])
                for r_c in range(12)
            ]
            all_24 = np.asarray(acc_maj_sc + acc_min_sc, dtype=np.float64)
            z_prof = float(acc_maj_sc[tonic_root] - acc_min_sc[tonic_root]) / max(1e-6, float(np.std(all_24)))

            segs_per_bar = max(1, int(round(measure_beats)))
            seg_betas: list[np.ndarray] = []
            seg_us: list[np.ndarray] = []
            seg_rms_list: list[float] = []
            seg_bar_ids: list[int] = []
            for b_i in range(n_bars):
                bs_t = max(0.0, computed_bar_starts[b_i])
                be_t = max(bs_t + 0.05, computed_bar_starts[b_i + 1])
                span_b = (be_t - bs_t) / segs_per_bar
                for s_i in range(segs_per_bar):
                    rb, rbs, r_val = _summarize_chroma_span(bs_t + s_i * span_b, bs_t + (s_i + 1) * span_b)
                    seg_betas.append(rbs / (np.sum(rbs) + 1e-12))
                    seg_us.append(rb / (np.max(rb) + 1e-12))
                    seg_rms_list.append(r_val)
                    seg_bar_ids.append(b_i)

            n_segs = len(seg_rms_list)
            seg_rms_arr = np.asarray(seg_rms_list, dtype=np.float64)
            rms_p90 = max(1e-6, float(np.percentile(seg_rms_arr, 90))) if n_segs else 1.0
            e_arr = np.minimum(1.0, seg_rms_arr / rms_p90)

            def _bass_dom(beta_vec: np.ndarray, pc: int) -> float:
                bp = float(beta_vec[pc])
                if bp <= 1e-9:
                    return 0.0
                other = float(np.max(np.delete(beta_vec, pc)))
                return max(0.0, 1.0 - other / bp)

            dom_pc = (tonic_root + 7) % 12
            sup_pc = (tonic_root + 2) % 12
            maj3_pc = (tonic_root + 4) % 12
            min3_pc = (tonic_root + 3) % 12

            g_tonic = np.asarray([_bass_dom(seg_betas[s], tonic_root) for s in range(n_segs)], dtype=np.float64)
            a_arr = e_arr * g_tonic
            k_arr = np.zeros(n_segs, dtype=np.float64)
            run_start = -1
            for s in range(n_segs):
                if g_tonic[s] > 0.0:
                    if run_start < 0:
                        run_start = s
                    pi = run_start - 1
                    if pi >= 0:
                        k_arr[s] = _bass_dom(seg_betas[pi], dom_pc) * float(seg_us[pi][sup_pc])
                else:
                    run_start = -1

            w_arr = a_arr * (1.0 + 1.0 * k_arr)
            u_mat = np.asarray(seg_us, dtype=np.float64)
            num_acc = float(np.sum(w_arr * (u_mat[:, maj3_pc] - u_mat[:, min3_pc])))
            den_acc = float(np.sum(w_arr * (u_mat[:, maj3_pc] + u_mat[:, min3_pc])) + 0.02 * np.sum(e_arr))
            pi_acc = num_acc / max(1e-12, den_acc)

            pi_voc = 0.0
            if any(bar_has_vocal):
                seg_bar_np = np.asarray(seg_bar_ids, dtype=np.int64)
                num_voc = 0.0
                den_voc = 0.0
                for b_i in range(n_bars):
                    if not bar_has_vocal[b_i]:
                        continue
                    mask_b = seg_bar_np == b_i
                    a_bar = float(np.mean(a_arr[mask_b])) if np.any(mask_b) else 0.0
                    e_bar = float(np.mean(e_arr[mask_b])) if np.any(mask_b) else 0.0
                    vw = bar_vocal_w[b_i]
                    v_u = vw / (float(np.max(vw)) + 1e-12)
                    num_voc += a_bar * float(v_u[maj3_pc] - v_u[min3_pc])
                    den_voc += a_bar * float(v_u[maj3_pc] + v_u[min3_pc]) + 0.02 * e_bar
                pi_voc = num_voc / max(1e-12, den_voc)

            mode_L = z_prof + 3.0 * pi_acc + 1.5 * max(0.0, pi_voc)
            is_minor = bool(mode_L < 0.0)
            tonic_bass_share = float(np.sum(a_arr) / max(1e-12, np.sum(e_arr)))
            mode_evidence = {
                "source": "stage_b_third_polarity",
                "pi_acc": round(pi_acc, 4),
                "pi_voc": round(pi_voc, 4),
                "z_prof": round(z_prof, 4),
                "L": round(mode_L, 4),
                "tonic_bass_share": round(tonic_bass_share, 4),
            }
        else:
            mode_evidence = {"source": "stage_a_profile"}

        sharp_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        flat_names = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
        use_flats = (not is_minor and tonic_root in (5, 10, 3, 8, 1)) or (is_minor and tonic_root in (2, 7, 0, 5, 10))
        NAMES = flat_names if use_flats else sharp_names

    trans_matrix = MINOR_TRANS_MATRIX if is_minor else MAJOR_TRANS_MATRIX

    # 3. Build Clean Idiomatic Chord Templates in Concert Key.
    # Tuple layout: (root, quality, bass_pc, tpl_vec, tones_set, complexity, ext_pc, prior_bias, name, degree_idx)
    candidates = []
    seen_names: set[str] = set()

    def _add_tpl(
        r_pc: int,
        qual: str,
        intervals_weights: tuple[tuple[int, float], ...],
        prior_bias: float,
        complexity: float = 0.0,
        ext_semi: int | None = None,
        slash_semi: int | None = None,
    ) -> None:
        r = r_pc % 12
        bass_pc = (r + slash_semi) % 12 if slash_semi is not None else r
        name = NAMES[r] + qual + ("/" + NAMES[bass_pc] if slash_semi is not None else "")
        if name in seen_names:
            return
        seen_names.add(name)
        tpl = np.zeros(12, dtype=np.float64)
        tones = []
        for semi, w in intervals_weights:
            pc = (r + semi) % 12
            tpl[pc] = max(tpl[pc], w)
            tones.append(pc)
        if slash_semi is not None:
            tpl[bass_pc] = max(tpl[bass_pc], 0.95)
        tpl = tpl / (np.linalg.norm(tpl) + 1e-12)
        ext_pc = (r + ext_semi) % 12 if ext_semi is not None else None
        deg = _degree_index(r, qual, tonic_root, is_minor, tuple(iv for iv, _ in intervals_weights))
        candidates.append((r, qual, bass_pc, tpl, frozenset(tones), complexity, ext_pc, prior_bias, name, deg))

    from subpkgs.guitar_reharm.chord_voicing import generate_voicings

    _playable_cache: dict[str, bool] = {}

    def _is_playable(sym: str) -> bool:
        if sym in _playable_cache:
            return _playable_cache[sym]
        try:
            ok = bool(generate_voicings(sym, 0))
        except Exception:
            ok = False
        _playable_cache[sym] = ok
        return ok

    TRI_MAJ = ((0, 1.0), (4, 0.92), (7, 0.80))
    TRI_MIN = ((0, 1.0), (3, 0.92), (7, 0.80))
    TRI_DIM = ((0, 1.0), (3, 0.90), (6, 0.82))
    SEV_DOM = ((0, 1.0), (4, 0.88), (7, 0.72), (10, 0.78))
    SEV_MAJ = ((0, 1.0), (4, 0.88), (7, 0.72), (11, 0.78))
    SEV_MIN = ((0, 1.0), (3, 0.88), (7, 0.72), (10, 0.78))
    SUS_2 = ((0, 1.0), (2, 0.90), (7, 0.80))
    SUS_4 = ((0, 1.0), (5, 0.92), (7, 0.80))
    SEV_SUS4 = ((0, 1.0), (5, 0.86), (7, 0.72), (10, 0.78))
    SIX_MAJ = ((0, 1.0), (4, 0.88), (7, 0.72), (9, 0.74))
    SIX_MIN = ((0, 1.0), (3, 0.88), (7, 0.72), (9, 0.74))
    HALF_DIM = ((0, 1.0), (3, 0.88), (6, 0.82), (10, 0.76))

    for off in range(12):
        r = (tonic_root + off) % 12
        if is_minor:
            if off == 0:
                _add_tpl(r, "m", TRI_MIN, 0.13)
                _add_tpl(r, "m7", SEV_MIN, 0.10, complexity=0.045, ext_semi=10)
                _add_tpl(r, "m6", SIX_MIN, 0.05, complexity=0.060, ext_semi=9)
                _add_tpl(r, "sus2", SUS_2, 0.04, complexity=0.058, ext_semi=2)
                _add_tpl(r, "sus4", SUS_4, 0.04, complexity=0.058, ext_semi=5)
            elif off == 2:
                _add_tpl(r, "m7b5", HALF_DIM, 0.05, complexity=0.055, ext_semi=6)
                _add_tpl(r, "dim", TRI_DIM, 0.04, complexity=0.055, ext_semi=6)
                _add_tpl(r, "m", TRI_MIN, 0.03)
                _add_tpl(r, "", TRI_MAJ, 0.02)
            elif off == 3:
                _add_tpl(r, "", TRI_MAJ, 0.11)
                _add_tpl(r, "maj7", SEV_MAJ, 0.08, complexity=0.055, ext_semi=11)
                _add_tpl(r, "6", SIX_MAJ, 0.05, complexity=0.060, ext_semi=9)
                _add_tpl(r, "sus2", SUS_2, 0.05, complexity=0.058, ext_semi=2)
                _add_tpl(r, "", TRI_MAJ, 0.07, complexity=0.045, slash_semi=4)
            elif off == 5:
                _add_tpl(r, "m", TRI_MIN, 0.10)
                _add_tpl(r, "m7", SEV_MIN, 0.08, complexity=0.045, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.09)
                _add_tpl(r, "7", SEV_DOM, 0.06, complexity=0.050, ext_semi=10)
            elif off == 7:
                _add_tpl(r, "m", TRI_MIN, 0.09)
                _add_tpl(r, "m7", SEV_MIN, 0.08, complexity=0.045, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.10)
                _add_tpl(r, "7", SEV_DOM, 0.10, complexity=0.045, ext_semi=10)
                _add_tpl(r, "sus4", SUS_4, 0.06, complexity=0.050, ext_semi=5)
                _add_tpl(r, "", TRI_MAJ, 0.07, complexity=0.045, slash_semi=4)
            elif off == 8:
                _add_tpl(r, "", TRI_MAJ, 0.10)
                _add_tpl(r, "maj7", SEV_MAJ, 0.08, complexity=0.055, ext_semi=11)
            elif off == 10:
                _add_tpl(r, "", TRI_MAJ, 0.11)
                _add_tpl(r, "7", SEV_DOM, 0.08, complexity=0.050, ext_semi=10)
                _add_tpl(r, "sus2", SUS_2, 0.05, complexity=0.058, ext_semi=2)
                _add_tpl(r, "sus4", SUS_4, 0.05, complexity=0.055, ext_semi=5)
                _add_tpl(r, "", TRI_MAJ, 0.07, complexity=0.045, slash_semi=4)
            else:
                _add_tpl(r, "", TRI_MAJ, -0.06)
                _add_tpl(r, "m", TRI_MIN, -0.06)
                _add_tpl(r, "7", SEV_DOM, -0.05, complexity=0.055, ext_semi=10)
        else:
            if off == 0:
                _add_tpl(r, "", TRI_MAJ, 0.13)
                _add_tpl(r, "maj7", SEV_MAJ, 0.10, complexity=0.052, ext_semi=11)
                _add_tpl(r, "6", SIX_MAJ, 0.05, complexity=0.060, ext_semi=9)
                _add_tpl(r, "sus2", SUS_2, 0.06, complexity=0.055, ext_semi=2)
                _add_tpl(r, "sus4", SUS_4, 0.06, complexity=0.055, ext_semi=5)
                _add_tpl(r, "7", SEV_DOM, 0.04, complexity=0.055, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.09, complexity=0.042, slash_semi=4)
            elif off == 2:
                _add_tpl(r, "m", TRI_MIN, 0.10)
                _add_tpl(r, "m7", SEV_MIN, 0.09, complexity=0.042, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.03)
                _add_tpl(r, "7", SEV_DOM, 0.04, complexity=0.050, ext_semi=10)
            elif off == 4:
                _add_tpl(r, "m", TRI_MIN, 0.09)
                _add_tpl(r, "m7", SEV_MIN, 0.09, complexity=0.042, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.04)
                _add_tpl(r, "7", SEV_DOM, 0.05, complexity=0.048, ext_semi=10)
            elif off == 5:
                _add_tpl(r, "", TRI_MAJ, 0.11)
                _add_tpl(r, "maj7", SEV_MAJ, 0.09, complexity=0.052, ext_semi=11)
                _add_tpl(r, "6", SIX_MAJ, 0.05, complexity=0.060, ext_semi=9)
                _add_tpl(r, "sus2", SUS_2, 0.05, complexity=0.058, ext_semi=2)
                _add_tpl(r, "m", TRI_MIN, 0.04)
                _add_tpl(r, "m7", SEV_MIN, 0.03, complexity=0.050, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.07, complexity=0.045, slash_semi=4)
            elif off == 7:
                _add_tpl(r, "", TRI_MAJ, 0.12)
                _add_tpl(r, "7", SEV_DOM, 0.10, complexity=0.045, ext_semi=10)
                _add_tpl(r, "sus2", SUS_2, 0.06, complexity=0.055, ext_semi=2)
                _add_tpl(r, "sus4", SUS_4, 0.08, complexity=0.050, ext_semi=5)
                _add_tpl(r, "7sus4", SEV_SUS4, 0.07, complexity=0.055, ext_semi=5)
                _add_tpl(r, "m", TRI_MIN, 0.03)
                _add_tpl(r, "m7", SEV_MIN, 0.04, complexity=0.048, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.09, complexity=0.042, slash_semi=4)
            elif off == 9:
                _add_tpl(r, "m", TRI_MIN, 0.11)
                _add_tpl(r, "m7", SEV_MIN, 0.10, complexity=0.042, ext_semi=10)
                _add_tpl(r, "", TRI_MAJ, 0.03)
                _add_tpl(r, "7", SEV_DOM, 0.05, complexity=0.048, ext_semi=10)
            elif off == 10:
                _add_tpl(r, "", TRI_MAJ, 0.05)
                _add_tpl(r, "7", SEV_DOM, 0.03, complexity=0.055, ext_semi=10)
            elif off == 11:
                _add_tpl(r, "m7b5", HALF_DIM, 0.04, complexity=0.055, ext_semi=6)
                _add_tpl(r, "dim", TRI_DIM, 0.03, complexity=0.055, ext_semi=6)
                _add_tpl(r, "7", SEV_DOM, 0.05, complexity=0.050, ext_semi=10)
                _add_tpl(r, "m", TRI_MIN, 0.02)
            else:
                _add_tpl(r, "", TRI_MAJ, -0.06)
                _add_tpl(r, "m", TRI_MIN, -0.06)
                _add_tpl(r, "7", SEV_DOM, -0.05, complexity=0.055, ext_semi=10)

    candidates = [c for c in candidates if _is_playable(c[8])]

    # 3b. Pretrained 170-Chord BTC Neural Posterior Extraction (runs AFTER Concert Key/Mode is locked!).
    btc_post = _compute_btc_frame_posteriors(y, sr)
    scale_pcs = frozenset(
        (tonic_root + iv) % 12 for iv in ((0, 2, 3, 5, 7, 8, 10) if is_minor else (0, 2, 4, 5, 7, 9, 11))
    )

    def _btc_span_summary(s_sec: float, e_sec: float) -> tuple[np.ndarray | None, float, float]:
        if btc_post is None:
            return None, 0.0, 0.0
        ell_mat, prob_mat = btc_post
        n_btc = ell_mat.shape[0]
        t_lo = max(0.0, float(s_sec))
        t_hi = max(t_lo + 0.04, float(e_sec))
        k0 = max(0, min(n_btc - 1, int(math.floor(t_lo / _BTC_DT_SEC))))
        k1 = max(k0 + 1, min(n_btc, int(math.ceil(t_hi / _BTC_DT_SEC))))
        f_starts = np.arange(k0, k1, dtype=np.float64) * _BTC_DT_SEC
        f_ends = f_starts + _BTC_DT_SEC
        ov = np.maximum(0.0, np.minimum(f_ends, t_hi) - np.maximum(f_starts, t_lo)) / _BTC_DT_SEC
        ov_sum = float(np.sum(ov))
        if ov_sum <= 1e-6:
            return None, 0.0, 0.0
        ell_bar = (ov[:, None] * ell_mat[k0:k1]).sum(axis=0) / ov_sum
        max_p_per_frame = np.max(prob_mat[k0:k1, :168], axis=1)
        m_span = float(np.sum(ov * max_p_per_frame) / ov_sum)
        g_span = float(np.clip((m_span - 0.28) / 0.30, 0.0, 1.0))
        top_ell = float(np.max(ell_bar[:168]))
        l_tilde = np.maximum(-6.0, ell_bar - top_ell)
        return l_tilde, g_span, ov_sum

    def _btc_cand_delta(
        l_tilde: np.ndarray | None,
        g_span: float,
        r: int,
        qual: str,
        tones: frozenset[int],
        body_norm_l1: np.ndarray,
    ) -> float:
        if l_tilde is None or g_span <= 1e-6:
            return 0.0
        if qual == "7sus4":
            lt_val = 0.5 * (float(l_tilde[14 * r + 9]) + float(l_tilde[14 * r + 13])) - 0.20
        else:
            q_off = _BTC_QUAL_TO_OFFSET.get(qual)
            if q_off is None:
                return 0.0
            lt_val = float(l_tilde[14 * r + q_off])
        pen_scale = 0.5 if tones.issubset(scale_pcs) else 1.0
        pen = _BTC_EXT_PENALTY.get(qual, 0.08) * pen_scale
        normal_delta = g_span * (0.125 * lt_val - pen)
        # Hard acoustic chroma sanity guards preventing neural extension hallucinations:
        # strictly cap below both normal_delta and the parent triad's BTC contribution.
        is_min_th = float(body_norm_l1[(r + 3) % 12]) > float(body_norm_l1[(r + 4) % 12])
        parent_qual = "m" if (qual in ("m7", "m6", "mmaj7") or (qual in ("sus4", "7sus4", "sus2") and is_min_th)) else ""
        parent_off = _BTC_QUAL_TO_OFFSET[parent_qual]
        triad_delta = g_span * 0.125 * float(l_tilde[14 * r + parent_off])
        guard_ceil = min(normal_delta, triad_delta, 0.0)
        if qual in ("m7", "7") and float(body_norm_l1[(r + 10) % 12]) < 0.062:
            return guard_ceil - 0.10 * g_span
        if qual == "maj7" and float(body_norm_l1[(r + 11) % 12]) < 0.062:
            return guard_ceil - 0.10 * g_span
        if qual in ("sus4", "7sus4"):
            th_e = max(float(body_norm_l1[(r + 3) % 12]), float(body_norm_l1[(r + 4) % 12]))
            if float(body_norm_l1[(r + 5) % 12]) <= 0.94 * th_e:
                return guard_ceil - 0.12 * g_span
        if qual == "sus2":
            th_e = max(float(body_norm_l1[(r + 3) % 12]), float(body_norm_l1[(r + 4) % 12]))
            if float(body_norm_l1[(r + 2) % 12]) <= 0.94 * th_e:
                return guard_ceil - 0.12 * g_span
        return normal_delta

    def _score_span_candidates(
        s_sec: float,
        e_sec: float,
        vocal_w: np.ndarray | None,
        has_voc: bool,
    ) -> tuple[dict[str, float], np.ndarray, float, float, float]:
        raw_body, raw_bass, span_rms = _summarize_chroma_span(s_sec, e_sec)
        body_norm_l2 = raw_body / (np.linalg.norm(raw_body) + 1e-12)
        body_norm_l1 = raw_body / (np.sum(raw_body) + 1e-12)
        body_max = raw_body / (np.max(raw_body) + 1e-12)
        bass_max = raw_bass / (np.max(raw_bass) + 1e-12)
        l_tilde, g_span, n_btc_f = _btc_span_summary(s_sec, e_sec)
        voc_total = float(np.sum(vocal_w)) if (has_voc and vocal_w is not None) else 0.0
        low_energy_bar = span_rms < 0.0025 and not has_voc
        scores_out: dict[str, float] = {}

        for (r, qual, bass_pc, tpl, tones, comp, ext_pc, bias, name, _deg) in candidates:
            if low_energy_bar:
                is_ton = (r == tonic_root and bass_pc == r and qual == ("m" if is_minor else ""))
                scores_out[name] = (0.35 if is_ton else 0.0) + bias - comp
                continue

            cos_fit = _cosine_sim(body_norm_l2, tpl)
            out_leak = sum(float(body_norm_l1[pc]) for pc in range(12) if pc not in tones)
            score = cos_fit - 0.22 * out_leak + bias - comp

            if ext_pc is not None and float(body_max[ext_pc]) < 0.34:
                score -= 0.085
            if "sus4" in qual:
                third_energy = max(float(body_max[(r + 4) % 12]), float(body_max[(r + 3) % 12]))
                if third_energy > 0.52 and float(body_max[(r + 5) % 12]) < third_energy:
                    score -= 0.16
            elif qual == "sus2":
                third_energy = max(float(body_max[(r + 4) % 12]), float(body_max[(r + 3) % 12]))
                if third_energy > 0.52 and float(body_max[(r + 2) % 12]) < third_energy:
                    score -= 0.16

            if bass_pc != r:
                if float(bass_max[bass_pc]) >= 0.55 and float(bass_max[bass_pc]) > 1.22 * float(bass_max[r]):
                    score += 0.17 * float(bass_max[bass_pc])
                else:
                    score -= 0.18
            else:
                score += 0.18 * float(bass_max[r])

            if has_voc and vocal_w is not None and voc_total > 0:
                voc_in = sum(float(vocal_w[pc]) for pc in tones) / voc_total
                score += 0.10 * voc_in

            score += _btc_cand_delta(l_tilde, g_span, r, qual, tones, body_norm_l1)
            scores_out[name] = score
        return scores_out, body_norm_l2, span_rms, g_span, n_btc_f

    # 4. Per-Bar Chroma + Neural BTC Summarization & Emission Scoring.
    bar_body_vecs: list[np.ndarray] = []
    bar_acc_rms: list[float] = []
    bar_btc_gates: list[float] = []
    emission_scores: list[dict[str, float]] = []

    for b in range(n_bars):
        bs_sec = max(0.0, computed_bar_starts[b])
        be_sec = max(bs_sec + 0.05, computed_bar_starts[b + 1])
        b_scores, b_l2, b_rms, g_b, _ = _score_span_candidates(
            bs_sec, be_sec, bar_vocal_w[b], bar_has_vocal[b]
        )
        emission_scores.append(b_scores)
        bar_body_vecs.append(b_l2)
        bar_acc_rms.append(b_rms)
        bar_btc_gates.append(g_b)

    # 5. Chroma-Continuity Viterbi Decoding.
    name_to_deg = {c[8]: c[9] for c in candidates}
    name_to_root = {c[8]: c[0] for c in candidates}
    name_to_qual = {c[8]: c[1] for c in candidates}
    name_to_bass = {c[8]: c[2] for c in candidates}
    name_to_tones = {c[8]: c[4] for c in candidates}
    unique_names = [c[8] for c in candidates]
    tonic_name = NAMES[tonic_root] + ("m" if is_minor else "")

    dp: dict[str, tuple[float, list[str]]] = {
        name: (emission_scores[0][name] + (0.06 if name == tonic_name else 0.0), [name])
        for name in unique_names
    }
    for b in range(1, n_bars):
        chroma_cont = _cosine_sim(bar_body_vecs[b - 1], bar_body_vecs[b])
        next_dp: dict[str, tuple[float, list[str]]] = {}
        for name in unique_names:
            emit = emission_scores[b][name]
            curr_deg = name_to_deg[name]
            curr_root = name_to_root[name]
            curr_bass = name_to_bass[name]
            best_sc = -np.inf
            best_path: list[str] = []
            for prev_name, (prev_sc, prev_path) in dp.items():
                if prev_name == name:
                    trans = 0.10 * (chroma_cont - 0.70)
                else:
                    prev_deg = name_to_deg[prev_name]
                    if prev_deg >= 0 and curr_deg >= 0:
                        trans = 0.65 * float(trans_matrix[prev_deg, curr_deg])
                    else:
                        dist = (curr_root - name_to_root[prev_name]) % 12
                        trans = 0.05 if dist in (5, 7) else (0.03 if dist in (2, 10) else -0.02)
                    bass_step = (curr_bass - name_to_bass[prev_name]) % 12
                    if bass_step in (1, 2, 10, 11):
                        trans += 0.025
                if b == n_bars - 1 and name == tonic_name:
                    trans += 0.06
                cand_sc = prev_sc + emit + trans
                if cand_sc > best_sc:
                    best_sc = cand_sc
                    best_path = prev_path
            next_dp[name] = (best_sc, best_path + [name])
        dp = next_dp

    chords = max(dp.values(), key=lambda item: item[0])[1] if dp else [tonic_name] * n_bars

    # 5b. Dual-Band Harmonic-Decontaminated Bass-CQT Slash Promotion & Half-Bar `split_chords`.
    root_pos_pcs_by_root: dict[int, list[frozenset[int]]] = {}
    for c_row in candidates:
        if c_row[2] == c_row[0]:
            root_pos_pcs_by_root.setdefault(c_row[0], []).append(c_row[4])

    def _maybe_promote_slash(base_sym: str, s_sec: float, e_sec: float, prev_bass_pc: int | None) -> tuple[str, int]:
        if "/" in base_sym or base_sym == "N.C." or base_sym not in name_to_root:
            b_pc = name_to_bass.get(base_sym, tonic_root)
            return base_sym, b_pc
        r_pc = name_to_root[base_sym]
        tones = name_to_tones[base_sym]
        f0_b = max(0, min(n_frames - 1, int(round(max(0.0, s_sec) * frames_per_sec))))
        f1_b = max(f0_b + 1, min(n_frames, int(round(max(s_sec + 0.05, e_sec) * frames_per_sec))))
        if f1_b - f0_b < 4:
            return base_sym, r_pc

        t_rel = np.arange(f1_b - f0_b, dtype=np.float64) / max(1.0, frames_per_sec)
        tau_b = min(0.38, max(0.20, 0.65 * float(seconds_per_beat)))
        u_w = np.exp(-t_rel / tau_b)
        bass_pow = semi_mag[0:31, f0_b:f1_b] ** 2   # C1 (24) .. F#3 (54)
        lo_pow = bass_pow[0:17]                     # C1 (24) .. E2 (40)

        e_lo = np.zeros(12, dtype=np.float64)
        e_hi = np.zeros(12, dtype=np.float64)
        frame_lo_hi = np.zeros((12, f1_b - f0_b), dtype=np.float64)
        lowest_bin_e = np.full(12, 99, dtype=np.int32)
        bin_energy = np.array([float(np.sum(u_w * bass_pow[m])) for m in range(31)], dtype=np.float64)
        peak_bin_e = float(np.max(bin_energy)) if bin_energy.size else 0.0
        for m_lo in range(17):
            pc = m_lo % 12
            e_lo[pc] += bin_energy[m_lo]
            frame_lo_hi[pc] += bass_pow[m_lo]
            if peak_bin_e > 0 and bin_energy[m_lo] >= 0.18 * peak_bin_e and (24 + m_lo) < lowest_bin_e[pc]:
                lowest_bin_e[pc] = 24 + m_lo
        for m_hi in range(17, 31):
            midi_m = 24 + m_hi
            w_reg = math.exp(-max(0.0, midi_m - 43.0) / 6.0)
            pc = m_hi % 12
            e_hi[pc] += w_reg * bin_energy[m_hi]
            frame_lo_hi[pc] += w_reg * bass_pow[m_hi]
            if peak_bin_e > 0 and bin_energy[m_hi] >= 0.18 * peak_bin_e and midi_m < lowest_bin_e[pc]:
                lowest_bin_e[pc] = midi_m

        # Subtract 3rd (19 st -> +7 mod 12) and 5th (28 st -> +4 mod 12) harmonics of low roots
        e_tilde = np.zeros(12, dtype=np.float64)
        for x_pc in range(12):
            decontam_hi = max(0.0, e_hi[x_pc] - 0.35 * e_lo[(x_pc - 7) % 12] - 0.15 * e_lo[(x_pc - 4) % 12])
            e_tilde[x_pc] = e_lo[x_pc] + 0.85 * decontam_hi

        tot_e = float(np.sum(e_tilde))
        if tot_e <= 1e-9:
            return base_sym, r_pc
        frame_dom = np.argmax(frame_lo_hi, axis=0)

        best_sym = base_sym
        best_bass = r_pc
        best_margin = 0.0
        for x_pc in sorted(tones):
            if x_pc == r_pc:
                continue
            iv = (x_pc - r_pc) % 12
            if iv not in (3, 4, 7, 10, 11):
                continue
            # Do not rewrite root-position chords disguised as inversions (e.g. C6/A == Am7)
            if any(tones == rp_set for rp_set in root_pos_pcs_by_root.get(x_pc, ())):
                continue
            # A 2nd-inversion 5th-in-bass requires the 5th to sound strictly lower in pitch than the root
            # (suppressing inner arpeggio 4th-string D3/E3 false positives above G2/A2).
            if iv == 7 and lowest_bin_e[x_pc] >= lowest_bin_e[r_pc]:
                continue
            d_val = math.log(e_tilde[x_pc] + 1e-9) - math.log(e_tilde[r_pc] + 1e-9)
            step_from_prev = min((prev_bass_pc - x_pc) % 12, (x_pc - prev_bass_pc) % 12) if prev_bass_pc is not None else 99
            if iv in (3, 4):
                tau_req = 0.80 if step_from_prev in (1, 2) else 0.95
            elif iv == 7:
                tau_req = 1.45
            else:
                tau_req = 1.05 if step_from_prev in (1, 2) else 1.55
            share = float(e_tilde[x_pc] / tot_e)
            dom_ratio = float(np.mean(frame_dom == x_pc))
            slash_cand = f"{base_sym}/{NAMES[x_pc]}"
            if d_val > tau_req and share >= 0.38 and dom_ratio >= 0.55 and _is_playable(slash_cand):
                margin = d_val - tau_req
                if margin > best_margin:
                    best_margin = margin
                    best_sym = slash_cand
                    best_bass = x_pc
        return best_sym, best_bass

    prev_b_pc: int | None = None
    base_bar_chords = list(chords)
    for b in range(n_bars):
        bs_sec = max(0.0, computed_bar_starts[b])
        be_sec = max(bs_sec + 0.05, computed_bar_starts[b + 1])
        if bar_acc_rms[b] >= 0.004:
            promoted, prev_b_pc = _maybe_promote_slash(chords[b], bs_sec, be_sec, prev_b_pc)
            chords[b] = promoted
        else:
            prev_b_pc = name_to_bass.get(chords[b], tonic_root)

    split_chords: dict[int, list[str]] = {}
    if btc_post is not None and measure_beats >= 3.0:
        is_simple_triple = (round(measure_beats) == 3 and not meter.strip().endswith("/8"))
        split_frac = (2.0 / 3.0) if is_simple_triple else 0.5
        clean_pool_all = [c[8] for c in candidates if "/" not in c[8]]

        def _span_vocal_profile(
            s_sec: float,
            e_sec: float,
            bs_full_sec: float,
            be_full_sec: float,
            b_lo_m: float,
            b_hi_m: float,
            b_base_m: float,
        ) -> tuple[np.ndarray, bool]:
            vw = np.zeros(12, dtype=np.float64)
            if not notes:
                return vw, False
            full_dur_s = max(1e-3, be_full_sec - bs_full_sec)
            for n in notes:
                if getattr(n, "raw_start", None) is not None and getattr(n, "raw_duration", None) is not None and float(n.raw_duration) > 0:
                    ns_s = float(n.raw_start)
                    ne_s = ns_s + float(n.raw_duration)
                    ov_s = max(0.0, min(ne_s, e_sec) - max(ns_s, s_sec))
                    if ov_s <= 0.0:
                        continue
                    ov_b = (ov_s / full_dur_s) * measure_beats
                    rel_on = ((max(ns_s, bs_full_sec) - bs_full_sec) / full_dur_s) * measure_beats
                else:
                    ns_b = float(n.start)
                    ne_b = ns_b + float(n.duration)
                    ov_b = max(0.0, min(ne_b, b_hi_m) - max(ns_b, b_lo_m))
                    if ov_b <= 0.0:
                        continue
                    rel_on = (max(ns_b, b_base_m) - b_base_m) % measure_beats
                sal = _gttm_note_salience(rel_on, ov_b, measure_beats, meter)
                vw[int(round(float(n.pitch))) % 12] += sal
            return vw, bool(np.sum(vw) > 0.0)

        for b in range(n_bars):
            if bar_btc_gates[b] < 0.30 or bar_acc_rms[b] < 0.005:
                continue
            bs_sec = max(0.0, computed_bar_starts[b])
            be_sec = max(bs_sec + 0.10, computed_bar_starts[b + 1])
            mid_sec = bs_sec + split_frac * (be_sec - bs_sec)
            b_base_m = b * measure_beats
            b_mid_m = (b + split_frac) * measure_beats
            b_end_m = (b + 1.0) * measure_beats
            vw_h1, hv_h1 = _span_vocal_profile(bs_sec, mid_sec, bs_sec, be_sec, b_base_m, b_mid_m, b_base_m)
            vw_h2, hv_h2 = _span_vocal_profile(mid_sec, be_sec, bs_sec, be_sec, b_mid_m, b_end_m, b_base_m)
            sc_h1, _, r_h1, g_h1, nf_h1 = _score_span_candidates(bs_sec, mid_sec, vw_h1, hv_h1)
            sc_h2, _, r_h2, g_h2, nf_h2 = _score_span_candidates(mid_sec, be_sec, vw_h2, hv_h2)
            if nf_h1 < 5.0 or nf_h2 < 5.0 or g_h1 < 0.32 or g_h2 < 0.32 or r_h1 < 0.0035 or r_h2 < 0.0035:
                continue
            c_star_base = base_bar_chords[b].split("/")[0]
            top_bar = sorted(clean_pool_all, key=lambda nm: emission_scores[b].get(nm, -99.0), reverse=True)[:5]
            top_h1 = sorted(clean_pool_all, key=lambda nm: sc_h1.get(nm, -99.0), reverse=True)[:4]
            top_h2 = sorted(clean_pool_all, key=lambda nm: sc_h2.get(nm, -99.0), reverse=True)[:4]
            pool = list(dict.fromkeys(
                top_bar
                + top_h1
                + top_h2
                + [c_star_base]
                + ([base_bar_chords[b - 1].split("/")[0]] if b > 0 else [])
                + ([base_bar_chords[b + 1].split("/")[0]] if b + 1 < n_bars else [])
            ))
            pool = [p for p in pool if p in sc_h1 and p in sc_h2]
            best_pair: tuple[str, str] | None = None
            best_gain = 0.0
            base_full_sc = split_frac * sc_h1.get(c_star_base, -9.0) + (1.0 - split_frac) * sc_h2.get(c_star_base, -9.0)
            for c1 in pool:
                for c2 in pool:
                    if c1 == c2:
                        continue
                    r1, r2 = name_to_root.get(c1, -1), name_to_root.get(c2, -2)
                    q1, q2 = name_to_qual.get(c1, ""), name_to_qual.get(c2, "")
                    # Require a genuine root change or sus->resolved triad resolution
                    if r1 == r2 and not ("sus" in q1 and q2 in ("", "m")):
                        continue
                    if (sc_h1[c1] - sc_h1[c2]) < 0.095 or (sc_h2[c2] - sc_h2[c1]) < 0.095:
                        continue
                    gain = (split_frac * sc_h1[c1] + (1.0 - split_frac) * sc_h2[c2]) - base_full_sc
                    if gain > 0.115 and gain > best_gain:
                        best_gain = gain
                        best_pair = (c1, c2)
            if best_pair is not None:
                c1_s, b1_pc = _maybe_promote_slash(best_pair[0], bs_sec, mid_sec, name_to_bass.get(chords[b - 1], None) if b > 0 else None)
                c2_s, _ = _maybe_promote_slash(best_pair[1], mid_sec, be_sec, b1_pc)
                if _is_playable(c1_s) and _is_playable(c2_s):
                    split_chords[b] = [c1_s, c2_s]

    # 6. Joint Accompaniment + Vocal Bar Dynamics (smooth musical arc in [0.76, 1.06]).
    acc_arr = np.asarray(bar_acc_rms, dtype=np.float64)
    acc_p15 = float(np.percentile(acc_arr, 15)) if acc_arr.size else 0.0
    acc_p90 = float(np.percentile(acc_arr, 90)) if acc_arr.size else 1.0
    acc_span = max(1e-5, acc_p90 - acc_p15)
    voc_rms_arr: np.ndarray | None = None
    if vocal_audio_path is not None and Path(vocal_audio_path).is_file():
        y_voc, sr_voc = librosa.load(str(vocal_audio_path), sr=sr, mono=True)
        if y_voc.size > 0 and np.all(np.isfinite(y_voc)):
            v_bars = []
            for b in range(n_bars):
                bs_sec = max(0.0, computed_bar_starts[b])
                be_sec = max(bs_sec + 0.05, computed_bar_starts[b + 1])
                vs0 = max(0, min(len(y_voc) - 1, int(round(bs_sec * sr_voc))))
                vs1 = max(vs0 + 1, min(len(y_voc), int(round(be_sec * sr_voc))))
                v_bars.append(float(np.sqrt(np.mean(y_voc[vs0:vs1] ** 2))) if vs1 > vs0 else 0.0)
            v_arr = np.asarray(v_bars, dtype=np.float64)
            v_p15 = float(np.percentile(v_arr, 15)) if v_arr.size else 0.0
            v_p90 = float(np.percentile(v_arr, 90)) if v_arr.size else 1.0
            if v_p90 - v_p15 > 1e-5:
                voc_rms_arr = np.clip((v_arr - v_p15) / (v_p90 - v_p15), 0.0, 1.0)
    raw_dyn = np.zeros(n_bars, dtype=np.float64)
    for b in range(n_bars):
        a_norm = float(np.clip((acc_arr[b] - acc_p15) / acc_span, 0.0, 1.0))
        if bar_has_vocal[b]:
            cov = min(1.0, bar_cov_ratios[b])
            reg = float(np.clip((bar_mean_pitches[b] - 54.0) / 22.0, 0.0, 1.0))
            v_norm = 0.55 * cov + 0.45 * reg
            if voc_rms_arr is not None:
                v_norm = 0.55 * v_norm + 0.45 * float(voc_rms_arr[b])
            raw_dyn[b] = 0.58 * a_norm + 0.42 * v_norm
        else:
            raw_dyn[b] = 0.72 * a_norm + 0.10
    if n_bars > 1:
        raw_dyn = np.convolve(np.pad(raw_dyn, (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid")[:n_bars]
    bar_dynamics = [round(float(np.clip(0.76 + 0.30 * v, 0.76, 1.06)), 4) for v in raw_dyn]

    return {
        "available": True,
        "chords": chords,
        "split_chords": split_chords,
        "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
        "bar_dynamics": bar_dynamics,
        "tonic": NAMES[tonic_root],
        "tonic_pc": int(tonic_root),
        "is_minor": bool(is_minor),
        "minor": bool(is_minor),
        "tuning_semitones": round(tuning, 3),
        "mode_evidence": mode_evidence,
    }


def synthesize_melody_conditioned_harmony(
    notes: list,
    bpm: float,
    beat_origin: float = 0.0,
    meter: str = "4/4",
    tonic: str = "C",
    minor: bool = False,
    clip_duration: float = 0.0,
    bar_start_seconds: list[float] | None = None,
    beat_times: list[float] | None = None,
) -> dict:
    """Harmonize vocal melody notes per bar via Viterbi decoding over candidate guitar chords.

    Used as the pure-vocal harmonizer when no accompaniment stem is present (a-cappella singing).
    """
    tonic_root, minor, NAMES = _parse_tonic_spec(tonic, minor)
    measure_beats, seconds_per_beat, bar_seconds = _timing(bpm, beat_origin, meter)
    trans_matrix = MINOR_TRANS_MATRIX if minor else MAJOR_TRANS_MATRIX

    def _chord_name(root: int, quality) -> str:
        suf = quality if isinstance(quality, str) else ("m" if quality else "")
        return NAMES[root % 12] + suf

    candidates = []
    seen_names = set()

    def _add(root: int, quality: str, tones: tuple[int, ...], prior_bias: float) -> None:
        r = root % 12
        name = _chord_name(r, quality)
        key = (name, tones)
        if key not in seen_names:
            seen_names.add(key)
            deg = _degree_index(r, quality, tonic_root, minor, tones)
            candidates.append((r, quality, tuple(sorted(set(((r + t) % 12) for t in tones))), prior_bias, name, deg))

    if minor:
        R_min = tonic_root
        _add(R_min, "m", (0, 3, 7), 0.08)
        _add((R_min + 2) % 12, "m7b5", (0, 3, 6, 10), 0.01)
        _add((R_min + 3) % 12, "", (0, 4, 7), 0.06)
        _add((R_min + 5) % 12, "m", (0, 3, 7), 0.06)
        _add((R_min + 5) % 12, "", (0, 4, 7), 0.05)
        _add((R_min + 7) % 12, "m", (0, 3, 7), 0.05)
        _add((R_min + 7) % 12, "", (0, 4, 7), 0.04)
        _add((R_min + 8) % 12, "", (0, 4, 7), 0.06)
        _add((R_min + 10) % 12, "", (0, 4, 7), 0.07)
    else:
        R = tonic_root
        _add(R, "", (0, 4, 7), 0.07)
        _add((R + 2) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 4) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 5) % 12, "", (0, 4, 7), 0.06)
        _add((R + 7) % 12, "", (0, 4, 7), 0.06)
        _add((R + 9) % 12, "m", (0, 3, 7), 0.06)
        _add((R + 11) % 12, "m7b5", (0, 3, 6, 10), 0.01)
        _add((R + 4) % 12, "", (0, 4, 7), -0.02)
        _add((R + 2) % 12, "", (0, 4, 7), -0.03)
        _add((R + 10) % 12, "", (0, 4, 7), -0.03)

    computed_bar_starts, valid_bt, custom_bounds = _resolve_bar_boundaries(
        notes, bpm, beat_origin, measure_beats, seconds_per_beat, bar_seconds,
        clip_duration, bar_start_seconds, beat_times,
    )
    n_bars = len(computed_bar_starts) - 1

    if notes:
        from subpkgs.guitar_reharm.chord_voicing import EXTENDED_INTERVALS
        observed_pcs = {int(round(float(getattr(n, "pitch", 60)))) % 12 for n in notes}
        seen_ext_pcs = {(0, 4, 7), (0, 3, 7)}
        ext_q_list = []
        for q_str in _CLEAN_MELODY_EXT_QUALITIES:
            ivs = EXTENDED_INTERVALS.get(q_str)
            if not ivs:
                continue
            norm_pcs = tuple(sorted({iv % 12 for iv in ivs}))
            if len(norm_pcs) < 3 or norm_pcs in seen_ext_pcs:
                continue
            seen_ext_pcs.add(norm_pcs)
            ext_q_list.append((q_str, norm_pcs))
        for off in range(12):
            r_pc = (tonic_root + off) % 12
            for q_str, ivs in (("", (0, 4, 7)), ("m", (0, 3, 7))):
                if sum(1 for iv in ivs if (r_pc + iv) % 12 in observed_pcs) >= 2:
                    _add(r_pc, q_str, ivs, -0.04)
            for q_str, ivs in ext_q_list:
                if all((r_pc + iv) % 12 in observed_pcs for iv in ivs):
                    _add(r_pc, q_str, ivs, -0.02 - 0.03 * max(0, len(ivs) - 3))

    chord_names = [c[4] for c in candidates]
    candidate_roots = [c[0] for c in candidates]
    candidate_tones = [c[2] for c in candidates]
    candidate_biases = [c[3] for c in candidates]
    candidate_qualities = [c[1] for c in candidates]
    candidate_degrees = [c[5] for c in candidates]

    tonic_qual = "m" if minor else ""
    dom_root = (tonic_root + 7) % 12

    chord_tivs = {}
    for ci in range(len(candidates)):
        name = chord_names[ci]
        if name not in chord_tivs:
            chroma = np.zeros(12)
            for pc in candidate_tones[ci]:
                chroma[pc] = 1.0
            chord_tivs[name] = _tiv(chroma)

    bar_vocal_w, bar_has_vocal, bar_cov_ratios, bar_mean_pitches, first_vocal_bar = _extract_bar_vocal_profiles(
        notes, computed_bar_starts, valid_bt, custom_bounds,
        bpm, beat_origin, measure_beats, seconds_per_beat, bar_seconds, meter,
    )

    if not notes or not any(bar_has_vocal):
        return {
            "chords": ["N.C."] * n_bars,
            "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
            "bar_dynamics": [0.0] * n_bars,
        }

    emission_scores = []
    for b in range(n_bars):
        vocal_w = bar_vocal_w[b]
        has_vocal = bar_has_vocal[b]
        rel_b = (b - first_vocal_bar) if (first_vocal_bar >= 0 and b >= first_vocal_bar) else b

        is_cadence_bar = (rel_b % 4 == 3) or (has_vocal and b + 1 < n_bars and not bar_has_vocal[b + 1])

        bar_tiv = _tiv(vocal_w) if has_vocal else None

        bar_scores = {}
        for ci in range(len(candidates)):
            root = candidate_roots[ci]
            tones = set(candidate_tones[ci])
            quality = candidate_qualities[ci]
            name = chord_names[ci]
            bias = candidate_biases[ci]
            has_maj3 = ((root + 4) % 12 in tones) and ((root + 3) % 12 not in tones)
            has_min3 = ((root + 3) % 12 in tones) and ((root + 4) % 12 not in tones)
            is_minor_q = has_min3 or quality in ("dim", "dim7", "m7b5")
            b9_pcs = {(t + 1) % 12 for t in tones} - tones
            avoid_pcs = (b9_pcs | ({(root + 3) % 12, (root + 5) % 12} if has_maj3 else set())) - tones
            missing_tones = 0
            cov_ratio = 0.0

            if has_vocal:
                s = 0.0
                for pc in range(12):
                    if pc in tones:
                        s += vocal_w[pc] * 1.0
                    elif pc in avoid_pcs:
                        s += vocal_w[pc] * -0.85
                    else:
                        s += vocal_w[pc] * -0.25
                total_w = float(np.sum(vocal_w))
                vocal_score = s / total_w if total_w > 0 else 0.0
                cov_ratio = sum(vocal_w[pc] for pc in tones) / total_w if total_w > 0 else 0.0
                missing_tones = sum(1 for pc in tones if vocal_w[pc] <= 0.02 * total_w)
                is_dia_triad = (candidate_degrees[ci] >= 0 and quality in ("", "m", "dim", "m7b5"))
                if not is_dia_triad and (missing_tones > 0 or cov_ratio < (0.92 if len(tones) >= 4 else 0.85)):
                    continue
                tiv_sim = _cosine_sim(bar_tiv, chord_tivs[name])
                combined = 0.75 * vocal_score + 0.17 * tiv_sim + bias - 0.04 * missing_tones
            else:
                is_dia_triad = (candidate_degrees[ci] >= 0 and quality in ("", "m", "dim", "m7b5"))
                if not is_dia_triad and candidate_degrees[ci] < 0:
                    continue
                t_root = tonic_root
                ctx_w = np.zeros(12)
                for ob in range(n_bars):
                    if bar_has_vocal[ob]:
                        ctx_w += bar_vocal_w[ob] / max(1.0, float(abs(ob - b)))
                ctx_tiv_sim = _cosine_sim(_tiv(ctx_w), chord_tivs[name]) if float(np.sum(ctx_w)) > 0 else 0.0
                tonic_prior = 0.22 if (b == 0 and root == t_root) else (0.05 if root == t_root else 0.0)
                cadence_prior = 0.08 if (b == n_bars - 1 or rel_b % 4 == 0) and root == t_root else 0.0
                combined = 0.25 * ctx_tiv_sim + tonic_prior + cadence_prior + bias

            is_tonic_chord = (root == tonic_root and "no5" not in quality and (quality == tonic_qual or (has_vocal and missing_tones == 0 and cov_ratio >= 0.95)))

            if is_tonic_chord and (b == 0 or b == first_vocal_bar):
                tonic_triad = set(candidate_tones[ci])
                vocal_on_triad = sum(vocal_w[pc] for pc in range(12) if pc in tonic_triad)
                avoid_clash = not any(vocal_w[pc] > 0 for pc in avoid_pcs)
                triad_frac = vocal_on_triad / float(np.sum(vocal_w)) if has_vocal and float(np.sum(vocal_w)) > 0 else 1.0
                if not has_vocal or (avoid_clash and triad_frac >= 0.35):
                    combined += 0.28
            if is_tonic_chord and b == n_bars - 1:
                combined += 0.22

            if is_cadence_bar:
                if root == dom_root and not is_minor_q:
                    combined += 0.16
                if is_tonic_chord and b == n_bars - 1:
                    combined += 0.06

            if minor and root == (tonic_root + 7) % 12 and not is_minor_q:
                leading_pc = (tonic_root + 11) % 12
                if not (has_vocal and vocal_w[leading_pc] > 0.05):
                    combined -= 0.22

            if name in bar_scores:
                bar_scores[name] = max(bar_scores[name], combined)
            else:
                bar_scores[name] = combined
        emission_scores.append(bar_scores)

    tonic_name = _chord_name(tonic_root, minor)

    name_to_degree = {}
    name_to_root = {}
    for ci in range(len(candidates)):
        n = chord_names[ci]
        if n not in name_to_degree or candidate_degrees[ci] >= 0:
            name_to_degree[n] = candidate_degrees[ci]
        if n not in name_to_root:
            name_to_root[n] = candidate_roots[ci]

    unique_names = list(dict.fromkeys(chord_names))

    states = {(None, False): (0.0, [])}
    for b in range(n_bars):
        updated = {}
        rel_b_v = (b - first_vocal_bar) if (first_vocal_bar >= 0 and b >= first_vocal_bar) else b
        for name in unique_names:
            if name not in emission_scores[b]:
                continue
            fit = emission_scores[b][name]
            curr_deg = name_to_degree.get(name, -1)
            for (prev, prev_rep), (cost, path) in states.items():
                transition = 0.0
                is_rep = (prev == name)
                if prev is not None and prev != "N.C." and name != "N.C.":
                    prev_deg = name_to_degree.get(prev, -1)
                    if prev_deg >= 0 and curr_deg >= 0:
                        transition = float(trans_matrix[prev_deg, curr_deg])
                    else:
                        prev_root = name_to_root.get(prev, 0)
                        curr_root = name_to_root.get(name, 0)
                        dist = (curr_root - prev_root) % 12
                        if dist in (5, 7):
                            transition = 0.10
                        elif dist in (2, 10):
                            transition = 0.06

                    if is_rep:
                        transition -= 0.03
                        if prev_rep:
                            transition -= 0.14

                    if (rel_b_v % 4 == 0 or b == n_bars - 1) and name == tonic_name:
                        transition += 0.12

                total = cost + fit + transition
                state_key = (name, is_rep)
                if state_key not in updated or total > updated[state_key][0]:
                    updated[state_key] = (total, path + [name])
        states = updated

    chords = max(states.values(), key=lambda x: x[0])[1] if states else ["N.C."] * n_bars

    vocal_energy = np.zeros(n_bars, dtype=np.float64)
    for b in range(n_bars):
        if bar_has_vocal[b]:
            cov = bar_cov_ratios[b]
            reg = np.clip((bar_mean_pitches[b] - 54.0) / 22.0, 0.0, 1.0)
            vocal_energy[b] = float(np.clip(0.55 * min(1.0, cov) + 0.45 * reg, 0.15, 1.0))
        else:
            vocal_energy[b] = 0.25
    if n_bars == 1:
        smoothed = vocal_energy.copy()
    else:
        padded = np.pad(vocal_energy, (1, 1), mode="edge")
        smoothed = np.convolve(padded, np.ones(3) / 3, mode="valid")
    bar_dynamics = [round(float(np.clip(0.76 + 0.30 * v, 0.76, 1.06)), 4) for v in smoothed[:n_bars]]

    return {
        "chords": chords,
        "bar_start_seconds": [round(t, 6) for t in computed_bar_starts],
        "bar_dynamics": bar_dynamics,
    }
