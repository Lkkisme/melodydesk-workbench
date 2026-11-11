"""Studio Multi-Round-Robin 6-String Acoustic Guitar Synthesizer & Vocal/Guitar Stem Mixer.

Architecture (Route 1 Acoustic Physics & Multi-Sample Engine):
1. Hybrid Multi-RR Sample Engine: 272 pure microphone-recorded acoustic plucks from `Shinyguitar`
   (17 minor-3rd zones x 4 velocity layers x 4 Round-Robins) fused with per-string minimum-phase FIR
   timbre transfer (`_minphase_fir`) from 375 `Meta Steel Guitar` per-string recordings, plus `FSS`
   and Karplus-Strong physical-modeling fallback when sample libraries are absent.
2. Formant-Preserving Phase-Continuous Fret-Slide (`'SL'` / `slide_from_fret`): separates static wood-body
   formants (98/196/385/2800 Hz RBJ modes) from string excitation, warps along a fret-quantized C^1 glide
   with integer phase-locked splice `D`, fret-wire micro-dip/buzz, and amplitude-complementary crossfade.
3. Real Studio Articulation Noise Staging: `Shinyguitar` 5-RR string-release noises (`release_rr`),
   16-RR wooden body taps (`perc_rr`), per-string muted chops (`mute_string`), and `Emilyguitar` 5-RR
   wound-string fret-shift squeaks (`fingering1_rr`) & palm-rest damps (`pickrest1_rr`) on a guarded noise bus.
4. Closed-Form 6-String Coupled Sympathetic String + Soundboard Resonance (`_sympathetic_ir`) folded into
   a single shared `scipy.fft.rfft` wooden-room convolution (`O(N log N)` zero-overhead post-bus).
5. Polyphonic 6-String Scheduler with `'CM'/'DS'/'SL'` Double-Stop dyad cluster sustain (no premature self-choke).
"""
from __future__ import annotations

from collections import OrderedDict
import math
import os
from pathlib import Path
import re
import threading
import numpy as np
import scipy.fft as sp_fft
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt
import soundfile as sf
from .chord_voicing import STANDARD_TUNING, _integer


def _resolve_samples_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get("MELODYDESK_GUITAR_SAMPLES")
    if env_dir:
        cand = Path(env_dir)
        if cand.is_dir():
            dirs.append(cand.resolve())
    repo_cand = (Path(__file__).resolve().parents[2] / "guitar_samples").resolve()
    if repo_cand.is_dir() and repo_cand not in dirs:
        dirs.append(repo_cand)
    return dirs


_CACHE_LOCK = threading.RLock()

_NOTE_TO_SEMI = {
    "C": 0, "Cs": 1, "D": 2, "Ds": 3, "E": 4, "F": 5,
    "Fs": 6, "G": 7, "Gs": 8, "A": 9, "As": 10, "B": 11,
}
_SHINY_NOTE_TO_SEMI = {
    "c": 0, "db": 1, "d": 2, "eb": 3, "e": 4, "f": 5,
    "gb": 6, "g": 7, "ab": 8, "a": 9, "bb": 10, "b": 11,
}
_STRING_NAME_TO_NUM = {
    "StringLowE": 6,
    "StringA": 5,
    "StringD": 4,
    "StringG": 3,
    "StringB": 2,
    "StringHighE": 1,
}

_META_INDEX: dict[tuple[int, int, int], Path] = {}
_FSS_INDEX: dict[tuple[int, str], Path] = {}
_SHINY_PITCHED: dict[tuple[int, int, int], Path] = {}      # (zone_midi, vl 1..4, rr 1..4) -> Path
_SHINY_ZONES: list[int] = []
_SHINY_RELEASE: dict[tuple[int, int], Path] = {}           # (zone_midi, rr 1..5) -> Path
_SHINY_MUTE: dict[tuple[int, int], Path] = {}              # (string 1..6, rr 1..5) -> Path
_SHINY_PERC: dict[tuple[int, int], Path] = {}              # (perc_id 1..4, rr 1..4) -> Path
_EMILY_SQUEAK: dict[int, Path] = {}                        # rr 1..5 -> Path
_EMILY_PICKREST: dict[int, Path] = {}                      # rr 1..5 -> Path

_WAVE_CACHE: OrderedDict[tuple[str, int, bool], np.ndarray] = OrderedDict()
_SHINY_ZONE_STATS_CACHE: dict[tuple[int, int, int], tuple[float, dict[int, int], np.ndarray]] = {}
_TIMBRE_FIR_CACHE: dict[tuple[int, int, int, int], np.ndarray | None] = {}
_HYBRID_BASE_CACHE: OrderedDict[tuple[int, int, int, int, int, bool], np.ndarray] = OrderedDict()
_BLENDED_NOTE_CACHE: OrderedDict[tuple[int, int, int, int, bool], np.ndarray] = OrderedDict()
_NOISE_ASSET_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_SLAP_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_ROOM_IR_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_SYMP_IR_CACHE: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}
_BODY_SOS_CACHE: dict[tuple[int, int], np.ndarray] = {}
_SOS_FILTER_CACHE: dict[tuple, np.ndarray] = {}
_INDEX_BUILT = False
_INDEX_DIRS_SCANNED: tuple[str, ...] | None = None

_MAX_WAVE_CACHE = 320
_MAX_HYBRID_CACHE = 384
_TARGET_GUITAR_ACTIVE_RMS = float(10.0 ** (-20.5 / 20.0))  # -20.5 dBFS studio accompaniment level
_TARGET_VOCAL_ACTIVE_RMS = float(10.0 ** (-16.0 / 20.0))   # -16.0 dBFS upfront lead vocal level
_BODY_MODES = ((98.0, 6.0, 4.0), (196.0, 7.0, 5.0), (385.0, 4.5, 5.0), (2800.0, 3.0, 1.2))


class _RoundRobin:
    """Deterministic shuffle-bag Round-Robin selector preventing back-to-back take repetition."""

    def __init__(self, seed: int = 0x20265EED):
        self.rng = np.random.default_rng(seed)
        self.bag: dict[tuple, list[int]] = {}
        self.last: dict[tuple, int] = {}

    def next(self, key: tuple, n: int = 4) -> int:
        if n <= 1:
            return 1
        bag = self.bag.get(key)
        if not bag:
            bag = [int(i) + 1 for i in self.rng.permutation(n)]
            if bag[-1] == self.last.get(key):
                bag[0], bag[-1] = bag[-1], bag[0]
            self.bag[key] = bag
        choice = bag.pop()
        self.last[key] = choice
        return choice


def _sample_rate(sr):
    return _integer(sr, 8001, float("inf"), "sample rate")


def _cached_butter_sos(order: int, freqs: float | tuple[float, float], btype: str, sr: int) -> np.ndarray | None:
    nyq_guard = 0.46 * float(sr)
    if isinstance(freqs, tuple):
        hi = min(float(freqs[1]), nyq_guard)
        lo = min(float(freqs[0]), 0.65 * hi)
        if lo <= 5.0 or hi <= lo + 10.0:
            return None
        w_key: float | tuple[float, float] = (round(lo, 2), round(hi, 2))
    else:
        fc = min(float(freqs), nyq_guard)
        if fc <= 5.0:
            return None
        w_key = round(fc, 2)
    key = (order, w_key, btype, int(sr))
    with _CACHE_LOCK:
        if key in _SOS_FILTER_CACHE:
            return _SOS_FILTER_CACHE[key]
    sos = butter(order, w_key, btype=btype, fs=sr, output="sos")
    with _CACHE_LOCK:
        _SOS_FILTER_CACHE[key] = sos
    return sos


def _peaking_eq_sos(fc: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Robert Bristow-Johnson peaking EQ biquad formatted as a single second-order section."""
    if abs(gain_db) < 1e-6 or fc <= 0.0 or fc >= 0.48 * sr:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / float(sr)
    alpha = math.sin(w0) / (2.0 * max(0.1, float(q)))
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _body_sos(sr: int, sign: int) -> np.ndarray:
    key = (int(sr), int(sign))
    with _CACHE_LOCK:
        if key in _BODY_SOS_CACHE:
            return _BODY_SOS_CACHE[key]
    sos = np.vstack([_peaking_eq_sos(fc, sign * gain, q, sr) for fc, gain, q in _BODY_MODES])
    with _CACHE_LOCK:
        _BODY_SOS_CACHE[key] = sos
    return sos


def _get_studio_room_ir(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached uncorrelated stereo wooden-room impulse response (RT60 ~ 0.34s, pre-delay 11ms L / 15ms R)."""
    with _CACHE_LOCK:
        if sr in _ROOM_IR_CACHE:
            return _ROOM_IR_CACHE[sr]
    ir_len = max(64, round(0.38 * sr))
    t = np.arange(ir_len, dtype=np.float64) / float(sr)
    env = np.exp(-20.3 * t)
    fade_in = min(ir_len, max(1, round(0.006 * sr)))
    env[:fade_in] *= np.sin(np.linspace(0.0, math.pi / 2.0, fade_in)) ** 2

    sos_room = _cached_butter_sos(2, (160.0, 5200.0), "bandpass", sr)
    channels = []
    for seed, pre_ms in ((202601, 0.011), (202602, 0.015)):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(ir_len) * env
        shaped = sosfilt(sos_room, noise) if sos_room is not None else noise
        pre = round(pre_ms * sr)
        ir = np.zeros(ir_len + pre, dtype=np.float64)
        ir[pre:] = shaped
        norm = float(np.sqrt(np.sum(ir ** 2)))
        if norm > 1e-12:
            ir /= norm
        ir.flags.writeable = False
        channels.append(ir)
    pair = (channels[0], channels[1])
    with _CACHE_LOCK:
        _ROOM_IR_CACHE[sr] = pair
    return pair


def _get_sympathetic_ir(
    sr: int,
    capo: int = 0,
    tuning: tuple[int, ...] = tuple(STANDARD_TUNING),
    gain_db: float = -25.5,
    kappa: float = 0.35,
) -> np.ndarray:
    """Closed-form impulse response of 6 open/capo strings (+ 2nd harmonics) coupled through 3 wood body modes."""
    capo_clamped = max(0, min(12, int(capo)))
    key = (int(sr), capo_clamped, tuple(int(x) for x in tuning))
    with _CACHE_LOCK:
        if key in _SYMP_IR_CACHE:
            return _SYMP_IR_CACHE[key]
    ir_len = max(64, round(2.1 * sr))
    nyq_max = 0.45 * float(sr)

    def _modal_bank(freqs: np.ndarray, t60s: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
        valid = (freqs > 25.0) & (freqs < nyq_max)
        if not np.any(valid):
            return np.zeros(length, dtype=np.float64)
        f = freqs[valid][:, None]
        t60 = t60s[valid][:, None]
        w = weights[valid][:, None]
        r = 10.0 ** (-3.0 / np.maximum(1.0, t60 * sr))
        theta = 2.0 * math.pi * f / float(sr)
        sin_th = np.maximum(1e-6, np.abs(np.sin(theta)))
        k = np.arange(length, dtype=np.float64)[None, :]
        p = np.exp(k * np.log(r)) * np.sin((k + 1.0) * theta) / sin_th
        h = p.copy()
        if length > 2:
            h[:, 2:] -= p[:, :-2]
        # Delay sympathetic onset by 4ms so it blooms behind the direct transient without comb-coloring attack
        out = (w * 0.5 * (1.0 - r * r) * h).sum(axis=0)
        fade = min(length, max(1, round(0.006 * sr)))
        out[:fade] *= np.sin(np.linspace(0.0, math.pi / 2.0, fade)) ** 2
        return out

    f_open = 440.0 * (2.0 ** ((np.asarray(tuning, dtype=np.float64) + capo_clamped - 69.0) / 12.0))
    s_ir = _modal_bank(
        np.concatenate([f_open, 2.0 * f_open]),
        np.concatenate([np.linspace(1.9, 1.15, len(f_open)), np.linspace(1.25, 0.75, len(f_open))]),
        np.concatenate([np.linspace(1.0, 0.60, len(f_open)), np.linspace(0.48, 0.28, len(f_open))]),
        ir_len,
    )
    fb = np.array([98.0, 196.0, 385.0], dtype=np.float64)
    qb = np.array([15.0, 20.0, 15.0], dtype=np.float64)
    b_ir = _modal_bank(fb, 6.9078 * qb / (math.pi * fb), np.array([1.0, 0.8, 0.5]), max(16, round(0.38 * sr)))
    coupled = s_ir + kappa * fftconvolve(s_ir, b_ir)[:ir_len]
    tail = max(1, ir_len // 8)
    coupled[-tail:] *= np.cos(np.linspace(0.0, math.pi / 2.0, tail)) ** 2
    symp = (10.0 ** (gain_db / 20.0)) * coupled
    symp.flags.writeable = False
    with _CACHE_LOCK:
        _SYMP_IR_CACHE[key] = symp
    return symp


def _active_rms(audio: np.ndarray, sr: int, gate_db: float = -46.0) -> tuple[float, int]:
    """Compute gated RMS over active 50ms frames and return (active_rms, active_frame_count)."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0
    mono = arr.mean(axis=-1) if arr.ndim >= 2 else arr
    win = max(1, round(0.050 * sr))
    if len(mono) < win:
        trimmed = mono.reshape(1, -1)
    else:
        n_frames = len(mono) // win
        trimmed = mono[: n_frames * win].reshape(n_frames, win)
    frame_rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    max_rms = float(np.max(frame_rms)) if frame_rms.size else 0.0
    if max_rms <= 1e-9:
        return 0.0, 0
    abs_gate = 10.0 ** (gate_db / 20.0)
    thresh = max(abs_gate, 0.10 * max_rms)
    active = frame_rms[frame_rms >= thresh]
    if active.size == 0:
        return max_rms, 0
    return float(np.sqrt(np.mean(active ** 2))), int(active.size)


def _ensure_sample_index():
    global _INDEX_BUILT, _INDEX_DIRS_SCANNED, _SHINY_ZONES
    roots = _resolve_samples_dirs()
    sig = tuple(str(r) for r in roots)
    if _INDEX_BUILT and _INDEX_DIRS_SCANNED == sig:
        return
    with _CACHE_LOCK:
        if _INDEX_BUILT and _INDEX_DIRS_SCANNED == sig:
            return
        _META_INDEX.clear()
        _FSS_INDEX.clear()
        _SHINY_PITCHED.clear()
        _SHINY_RELEASE.clear()
        _SHINY_MUTE.clear()
        _SHINY_PERC.clear()
        _EMILY_SQUEAK.clear()
        _EMILY_PICKREST.clear()
        for base in roots:
            meta_dir = base / "meta-steel-guitar" / "ogg"
            if meta_dir.is_dir():
                for p in meta_dir.glob("*.ogg"):
                    m = re.match(r"INSGuitarSteel_(String\w+)_Vel0([123])_([A-G]s?)(\d)_01\.ogg", p.name)
                    if m:
                        key = (
                            _STRING_NAME_TO_NUM[m.group(1)],
                            (int(m.group(4)) + 1) * 12 + _NOTE_TO_SEMI[m.group(3)],
                            int(m.group(2)),
                        )
                        _META_INDEX.setdefault(key, p)
            fss_dir = base / "FSS-SteelStringGuitar-SFZ-20200521" / "samples"
            if fss_dir.is_dir():
                for p in fss_dir.glob("*.wav"):
                    m = re.match(r"(MV|HV)_(\d+)\.wav", p.name)
                    if m:
                        _FSS_INDEX.setdefault((int(m.group(2)), m.group(1)), p)
            shiny_dir = base / "shinyguitar" / "Samples" / "acoustic"
            if shiny_dir.is_dir():
                for p in shiny_dir.glob("*.wav"):
                    m_vl = re.match(r"([a-g]b?)(\d)_vl([1-4])_rr([1-4])_1\.wav", p.name)
                    if m_vl:
                        z_midi = (int(m_vl.group(2)) + 1) * 12 + _SHINY_NOTE_TO_SEMI[m_vl.group(1)]
                        _SHINY_PITCHED.setdefault((z_midi, int(m_vl.group(3)), int(m_vl.group(4))), p)
                        continue
                    m_rel = re.match(r"([a-g]b?)(\d)_release_rr([1-5])_1\.wav", p.name)
                    if m_rel:
                        z_midi = (int(m_rel.group(2)) + 1) * 12 + _SHINY_NOTE_TO_SEMI[m_rel.group(1)]
                        _SHINY_RELEASE.setdefault((z_midi, int(m_rel.group(3))), p)
                        continue
                    m_mute = re.match(r"mute_string([1-6])_rr([1-5])_1\.wav", p.name)
                    if m_mute:
                        _SHINY_MUTE.setdefault((int(m_mute.group(1)), int(m_mute.group(2))), p)
                        continue
                    m_perc = re.match(r"perc_([1-4])_rr([1-4])_1\.wav", p.name)
                    if m_perc:
                        _SHINY_PERC.setdefault((int(m_perc.group(1)), int(m_perc.group(2))), p)
            emily_noises = base / "emilyguitar" / "noises"
            if emily_noises.is_dir():
                for p in emily_noises.glob("*.wav"):
                    m_sq = re.match(r"fingering1_rr([1-5])\.wav", p.name)
                    if m_sq:
                        _EMILY_SQUEAK.setdefault(int(m_sq.group(1)), p)
                        continue
                    m_pr = re.match(r"pickrest1_rr([1-5])\.wav", p.name)
                    if m_pr:
                        _EMILY_PICKREST.setdefault(int(m_pr.group(1)), p)
        _SHINY_ZONES = sorted({z for (z, _, _) in _SHINY_PITCHED})
        _INDEX_BUILT = bool(_SHINY_PITCHED or _META_INDEX or _FSS_INDEX)
        _INDEX_DIRS_SCANNED = sig


def _read_cached_audio(path: Path, target_sr: int, trim_onset: bool = True) -> np.ndarray:
    key = (str(path), int(target_sr), bool(trim_onset))
    with _CACHE_LOCK:
        if key in _WAVE_CACHE:
            _WAVE_CACHE.move_to_end(key)
            return _WAVE_CACHE[key]
    y, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if src_sr != target_sr:
        g = math.gcd(int(src_sr), int(target_sr))
        y = resample_poly(y, int(target_sr) // g, int(src_sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if trim_onset and peak > 1e-6:
        idx = int(np.argmax(np.abs(y) > 0.025 * peak))
        trim = max(0, idx - int(0.0008 * target_sr))
        if trim > 0:
            y = y[trim:]
    y.flags.writeable = False
    with _CACHE_LOCK:
        _WAVE_CACHE[key] = y
        while len(_WAVE_CACHE) > _MAX_WAVE_CACHE:
            _WAVE_CACHE.popitem(last=False)
    return y


def _pitch_shift_resample(wave: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-4 or wave.size < 4:
        return wave
    ratio = 2.0 ** (semitones / 12.0)
    new_len = max(2, int(round(len(wave) / ratio)))
    src_x = np.linspace(0.0, len(wave) - 1, new_len, dtype=np.float64)
    idx = np.floor(src_x).astype(np.int64)
    frac = src_x - idx
    idx_next = np.minimum(idx + 1, len(wave) - 1)
    return ((1.0 - frac) * wave[idx] + frac * wave[idx_next]).astype(np.float64)


def _smooth_power_spectrum(power: np.ndarray, freqs: np.ndarray, f0: float) -> np.ndarray:
    bw = np.maximum(0.115 * freqs, 0.75 * max(40.0, f0))
    lo = np.searchsorted(freqs, freqs - bw)
    hi = np.searchsorted(freqs, freqs + bw)
    csum = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)


def _minphase_fir(mag: np.ndarray, n_taps: int = 256) -> np.ndarray:
    """Convert non-negative rfft magnitude response into a causal minimum-phase FIR via cepstral folding."""
    nfft = 2 * (len(mag) - 1)
    cep = sp_fft.irfft(np.log(np.maximum(mag, 1e-4)), nfft)
    cep[1 : nfft // 2] *= 2.0
    cep[nfft // 2 + 1 :] = 0.0
    h = sp_fft.irfft(np.exp(sp_fft.rfft(cep, nfft)), nfft)[:n_taps].copy()
    half = n_taps // 2
    h[half:] *= np.hanning(n_taps)[half:]
    return h


def _get_shiny_zone_stats(zone_midi: int, vl: int, sr: int) -> tuple[float, dict[int, int], np.ndarray]:
    """Return (median_0.5s_rms, rr_envelope_lags, mean_smoothed_power_spectrum) for `(zone_midi, vl, sr)`."""
    key = (int(zone_midi), int(vl), int(sr))
    with _CACHE_LOCK:
        if key in _SHINY_ZONE_STATS_CACHE:
            return _SHINY_ZONE_STATS_CACHE[key]
    nfft = 16384
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    f0 = 440.0 * (2.0 ** ((zone_midi - 69.0) / 12.0))
    w_start, w_len = round(0.030 * sr), round(0.500 * sr)
    rms_vals: list[float] = []
    lags: dict[int, int] = {}
    env_ref: np.ndarray | None = None
    max_lag = max(1, round(0.002 * sr))
    env_len = max(16, round(0.028 * sr))
    box = np.ones(max(1, round(0.001 * sr)), dtype=np.float64)
    box /= len(box)
    spec_acc = np.zeros(len(freqs), dtype=np.float64)
    spec_cnt = 0

    for rr in range(1, 5):
        p = _SHINY_PITCHED.get((zone_midi, vl, rr))
        if p is None:
            continue
        w = _read_cached_audio(p, sr)
        n_rms = min(len(w), max(1, round(0.50 * sr)))
        rms_vals.append(float(np.sqrt(np.mean(w[:n_rms].astype(np.float64) ** 2))))
        head = np.abs(w[: min(len(w), env_len)].astype(np.float64))
        if len(head) < env_len:
            head = np.pad(head, (0, env_len - len(head)))
        env = np.convolve(head, box, mode="same")
        if env_ref is None:
            env_ref = env
            lags[rr] = 0
        else:
            best_lag, best_corr = 0, -1.0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    c = float(np.dot(env[lag:], env_ref[: len(env) - lag]))
                else:
                    c = float(np.dot(env[: len(env) + lag], env_ref[-lag:]))
                if c > best_corr:
                    best_corr, best_lag = c, lag
            lags[rr] = best_lag
        seg = w[w_start : min(len(w), w_start + w_len)].astype(np.float64)
        if len(seg) >= 256:
            win_seg = seg * np.hanning(len(seg))
            pw = np.abs(sp_fft.rfft(win_seg, nfft)) ** 2
            spec_acc += _smooth_power_spectrum(pw, freqs, f0)
            spec_cnt += 1

    med_rms = float(np.median(rms_vals)) if rms_vals else 0.095
    mean_spec = (spec_acc / spec_cnt) if spec_cnt > 0 else np.ones(len(freqs), dtype=np.float64)
    res = (max(1e-4, med_rms), lags, mean_spec)
    with _CACHE_LOCK:
        _SHINY_ZONE_STATS_CACHE[key] = res
    return res


def _get_string_timbre_fir(string: int, zone_midi: int, vl: int, sr: int) -> np.ndarray | None:
    """Compute cached 256-tap minimum-phase FIR transferring Meta per-string wound/plain spectral envelope onto Shiny."""
    key = (int(string), int(zone_midi), int(vl), int(sr))
    with _CACHE_LOCK:
        if key in _TIMBRE_FIR_CACHE:
            return _TIMBRE_FIR_CACHE[key]
    if not _META_INDEX or sr < 16000:
        with _CACHE_LOCK:
            _TIMBRE_FIR_CACHE[key] = None
        return None
    meta_layer = 1 if vl == 1 else (2 if vl in (2, 3) else 3)
    cands = [m for (s, m, v) in _META_INDEX if s == string and v == meta_layer]
    if not cands:
        cands = [m for (s, m, _) in _META_INDEX if s == string]
    if not cands:
        with _CACHE_LOCK:
            _TIMBRE_FIR_CACHE[key] = None
        return None
    nearest_m = min(cands, key=lambda m: abs(m - zone_midi))
    if abs(nearest_m - zone_midi) > 7:
        with _CACHE_LOCK:
            _TIMBRE_FIR_CACHE[key] = None
        return None
    meta_p = _META_INDEX.get((string, nearest_m, meta_layer))
    if meta_p is None:
        meta_p = next(p for (s, m, _), p in _META_INDEX.items() if s == string and m == nearest_m)
    m_wave = _read_cached_audio(meta_p, sr)
    m_seg = _pitch_shift_resample(
        m_wave[: min(len(m_wave), round(0.75 * sr))].astype(np.float64),
        float(zone_midi - nearest_m),
    )
    w_start, w_len = round(0.030 * sr), round(0.500 * sr)
    m_win = m_seg[w_start : min(len(m_seg), w_start + w_len)]
    if len(m_win) < 256:
        with _CACHE_LOCK:
            _TIMBRE_FIR_CACHE[key] = None
        return None
    nfft = 16384
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    f0 = 440.0 * (2.0 ** ((zone_midi - 69.0) / 12.0))
    m_pow = np.abs(sp_fft.rfft(m_win * np.hanning(len(m_win)), nfft)) ** 2
    e_meta = _smooth_power_spectrum(m_pow, freqs, f0)
    _, _, e_shiny = _get_shiny_zone_stats(zone_midi, vl, sr)

    ratio_db = 10.0 * np.log10(np.maximum(e_meta, 1e-12) / np.maximum(e_shiny, 1e-12))
    band = (freqs >= 110.0) & (freqs <= min(5800.0, 0.40 * sr))
    if np.any(band):
        ratio_db -= float(np.mean(ratio_db[band]))
    ratio_db = np.clip(ratio_db, -5.0, 5.0)
    lo_taper = np.clip((freqs - 70.0) / 70.0, 0.0, 1.0)
    hi_cut = min(11500.0, 0.44 * sr)
    hi_taper = np.clip((hi_cut - freqs) / max(400.0, 0.25 * hi_cut), 0.0, 1.0)
    ratio_db *= lo_taper * hi_taper
    fir = _minphase_fir(10.0 ** (ratio_db / 20.0), n_taps=256)
    fir.flags.writeable = False
    with _CACHE_LOCK:
        _TIMBRE_FIR_CACHE[key] = fir
    return fir


def _select_shiny_layer_and_gain(velocity: float, technique: str) -> tuple[int, float]:
    if velocity <= 0:
        return 1, 0.0
    v = float(np.clip(velocity, 1.0, 127.0))
    vn = v / 127.0
    if technique == "PM":
        return 1, float(0.84 * (0.30 + 0.70 * vn))
    if technique in {"CM", "SL"}:
        vl = 4 if v >= 104.0 else (3 if v >= 78.0 else 2)
        return vl, float(1.08 + 0.26 * vn)
    if technique == "DS":
        vl = 3 if v >= 86.0 else 2
        return vl, float(0.94 + 0.22 * vn)
    vl = 1 + int(v > 46.0) + int(v > 76.0) + int(v > 104.0)
    db_offset = 20.0 * math.log10(0.32 + 0.68 * vn) + 1.2
    return vl, float(10.0 ** (np.clip(db_offset, -8.0, 3.2) / 20.0))


def _get_hybrid_shiny_base(
    pitch: int,
    string: int,
    vl: int,
    rr: int,
    sr: int,
    is_pm: bool,
) -> np.ndarray | None:
    if not _SHINY_ZONES:
        return None
    zone_midi = min(_SHINY_ZONES, key=lambda z: abs(z - pitch))
    if abs(pitch - zone_midi) > 6:
        return None
    rr_idx = ((int(rr) - 1) % 4) + 1
    key = (int(string), int(pitch), int(vl), rr_idx, int(sr), bool(is_pm))
    with _CACHE_LOCK:
        if key in _HYBRID_BASE_CACHE:
            _HYBRID_BASE_CACHE.move_to_end(key)
            return _HYBRID_BASE_CACHE[key]

    wav_path = _SHINY_PITCHED.get((zone_midi, vl, rr_idx))
    if wav_path is None:
        wav_path = next((p for (z, l, _), p in _SHINY_PITCHED.items() if z == zone_midi and l == vl), None)
    if wav_path is None:
        return None

    raw = _read_cached_audio(wav_path, sr)
    med_rms_l, lags, _ = _get_shiny_zone_stats(zone_midi, vl, sr)
    med_rms_3, _, _ = _get_shiny_zone_stats(zone_midi, 3, sr)
    lag = lags.get(rr_idx, 0)
    max_len = round(3.8 * sr)
    if lag > 0 and lag < len(raw) // 4:
        seg = raw[lag : min(len(raw), lag + max_len)].astype(np.float64)
    elif lag < 0:
        seg = np.pad(raw[: min(len(raw), max_len)].astype(np.float64), (-lag, 0))[:max_len]
    else:
        seg = raw[: min(len(raw), max_len)].astype(np.float64)

    # Calibrate vl3 median RMS to 0.147 (matching Meta 0.060 * 2.45) while preserving natural take variance
    g_cal = (0.147 / max(1e-4, med_rms_3)) * ((med_rms_l / max(1e-4, med_rms_3)) ** (-0.38))
    seg *= g_cal

    fir = _get_string_timbre_fir(string, zone_midi, vl, sr)
    if fir is not None:
        seg = fftconvolve(seg, fir, mode="full")[: len(seg)]

    shift_semi = float(pitch - zone_midi)
    if abs(shift_semi) > 1e-4:
        seg = _pitch_shift_resample(seg, shift_semi)
        if abs(shift_semi) > 0.5:
            sos_lp = _cached_butter_sos(2, 10800.0, "lowpass", sr)
            if sos_lp is not None:
                seg = sosfilt(sos_lp, seg)

    t = np.arange(len(seg), dtype=np.float64) / float(sr)
    if is_pm:
        seg *= np.exp(-t * 9.5)
        sos_pm = _cached_butter_sos(2, 1150.0, "lowpass", sr)
        if sos_pm is not None:
            seg = sosfilt(sos_pm, seg)
    else:
        decay_rate = 0.10 if string >= 4 else (0.18 if vl == 1 else 0.13)
        seg *= np.exp(-t * decay_rate)
        if vl == 1 and string <= 3:
            sos_soft = _cached_butter_sos(1, 9600.0, "lowpass", sr)
            if sos_soft is not None:
                seg = sosfilt(sos_soft, seg)

    seg32 = seg.astype(np.float32)
    seg32.flags.writeable = False
    with _CACHE_LOCK:
        _HYBRID_BASE_CACHE[key] = seg32
        while len(_HYBRID_BASE_CACHE) > _MAX_HYBRID_CACHE:
            _HYBRID_BASE_CACHE.popitem(last=False)
    return seg32


def _select_vel_layer(velocity: float, technique: str) -> tuple[int, float]:
    if velocity <= 0:
        return 1, 0.0
    vel_norm = float(np.clip(velocity / 127.0, 0.0, 1.0))
    if technique == "PM":
        return 1, float(0.85 * (0.30 + 0.70 * vel_norm))
    if technique in {"CM", "SL"}:
        return (3 if velocity >= 84 else 2), float(1.06 + 0.28 * vel_norm)
    if technique == "DS":
        return (3 if velocity >= 90 else 2), float(0.92 + 0.24 * vel_norm)
    if velocity <= 54:
        layer = 1
    elif velocity <= 88:
        layer = 2
    else:
        layer = 3
    span_pos = np.clip((velocity - 1.0) / 126.0, 0.0, 1.0)
    db_offset = -1.8 + 3.0 * span_pos
    return layer, float(10.0 ** (db_offset / 20.0))


def _get_blended_studio_base(pitch: int, string: int, layer: int, sr: int, is_pm: bool) -> np.ndarray | None:
    key = (string, int(pitch), layer, sr, is_pm)
    with _CACHE_LOCK:
        if key in _BLENDED_NOTE_CACHE:
            _BLENDED_NOTE_CACHE.move_to_end(key)
            return _BLENDED_NOTE_CACHE[key]
    primary_wave = None
    shift_semi = 0.0
    if _META_INDEX:
        if (string, pitch, layer) in _META_INDEX:
            primary_wave = _read_cached_audio(_META_INDEX[(string, pitch, layer)], sr)
        else:
            same_str = [m for (s, m, v) in _META_INDEX if s == string and v == layer]
            if same_str:
                nearest_m = min(same_str, key=lambda m: abs(m - pitch))
                if abs(nearest_m - pitch) <= 3:
                    primary_wave = _read_cached_audio(_META_INDEX[(string, nearest_m, layer)], sr)
                    shift_semi = float(pitch - nearest_m)
            if primary_wave is None:
                any_str = [(s, m) for (s, m, v) in _META_INDEX if v == layer]
                if any_str:
                    nearest_s, nearest_m = min(any_str, key=lambda sm: (abs(sm[1] - pitch), abs(sm[0] - string)))
                    primary_wave = _read_cached_audio(_META_INDEX[(nearest_s, nearest_m, layer)], sr)
                    shift_semi = float(pitch - nearest_m)

    fss_wave = None
    fss_shift = 0.0
    if _FSS_INDEX:
        pref_tag = "HV" if layer == 3 else "MV"
        alt_tag = "MV" if pref_tag == "HV" else "HV"
        if (pitch, pref_tag) in _FSS_INDEX:
            fss_wave = _read_cached_audio(_FSS_INDEX[(pitch, pref_tag)], sr)
        elif (pitch, alt_tag) in _FSS_INDEX:
            fss_wave = _read_cached_audio(_FSS_INDEX[(pitch, alt_tag)], sr)
        else:
            avail = sorted({m for (m, _) in _FSS_INDEX})
            nearest_m = min(avail, key=lambda m: abs(m - pitch))
            tag = pref_tag if (nearest_m, pref_tag) in _FSS_INDEX else alt_tag
            fss_wave = _read_cached_audio(_FSS_INDEX[(nearest_m, tag)], sr)
            fss_shift = float(pitch - nearest_m)

    max_len = round(4.0 * sr)
    if primary_wave is not None:
        seg = primary_wave[: min(len(primary_wave), max_len)].astype(np.float64)
        seg = _pitch_shift_resample(seg, shift_semi) * 2.45
        if abs(shift_semi) > 0.5:
            sos_lp = _cached_butter_sos(2, 10500.0, "lowpass", sr)
            if sos_lp is not None:
                seg = sosfilt(sos_lp, seg)
        if fss_wave is not None:
            f_seg = fss_wave[: min(len(fss_wave), max_len)].astype(np.float64)
            f_seg = _pitch_shift_resample(f_seg, fss_shift)
            n_blend = min(len(seg), len(f_seg))
            seg[:n_blend] = 0.88 * seg[:n_blend] + 0.12 * (f_seg[:n_blend] * 0.45)
    elif fss_wave is not None:
        seg = fss_wave[: min(len(fss_wave), max_len)].astype(np.float64)
        seg = _pitch_shift_resample(seg, fss_shift) * 0.55
    else:
        return None

    t = np.arange(len(seg), dtype=np.float64) / sr
    if is_pm:
        seg *= np.exp(-t * 9.5)
        sos_pm = _cached_butter_sos(2, 1150.0, "lowpass", sr)
        if sos_pm is not None:
            seg = sosfilt(sos_pm, seg)
    else:
        decay_rate = 0.12 if string >= 4 else (0.22 if layer == 1 else 0.16)
        seg *= np.exp(-t * decay_rate)
        if layer == 1 and string <= 3:
            sos_s = _cached_butter_sos(1, 9800.0, "lowpass", sr)
            if sos_s is not None:
                seg = sosfilt(sos_s, seg)
    seg32 = seg.astype(np.float32)
    seg32.flags.writeable = False
    with _CACHE_LOCK:
        _BLENDED_NOTE_CACHE[key] = seg32
        while len(_BLENDED_NOTE_CACHE) > _MAX_HYBRID_CACHE:
            _BLENDED_NOTE_CACHE.popitem(last=False)
    return seg32


def _slide_head(base: np.ndarray, skip: int, delta_semi: int, tau_sec: float, sr: int) -> tuple[np.ndarray, int]:
    """Formant-preserving, phase-continuous fret-slide head returning `(warped_head, int_offset_D)`."""
    n_hold = max(2, round(0.014 * sr))
    n_glide = max(8, round(tau_sec * sr))
    n_xf = max(8, round(0.024 * sr))

    # Compute glide phase deviation first (zero outside n_hold + n_glide) so n_settle > |d_int| is exact for all delta_semi
    n_active = n_hold + n_glide
    n_act_arr = np.arange(n_active, dtype=np.float64)
    u_act = np.clip((n_act_arr - n_hold) / float(n_glide), 0.0, 1.0)
    g_act = float(delta_semi) * (1.0 - u_act * u_act * (3.0 - 2.0 * u_act))
    semi_act = g_act - 0.70 * np.sin(2.0 * math.pi * g_act) / (2.0 * math.pi)
    dev_act = np.exp2(semi_act / 12.0) - 1.0
    d_tot = float(dev_act.sum())
    d_int = int(round(d_tot))
    if abs(d_tot) > 1e-9:
        dev_act *= d_int / d_tot

    n_settle = max(round(0.030 * sr), abs(d_int) + 16)
    n_head = n_active + n_settle + n_xf
    n = np.arange(n_head, dtype=np.float64)
    g = np.zeros(n_head, dtype=np.float64)
    g[:n_active] = g_act
    dev = np.zeros(n_head, dtype=np.float64)
    dev[:n_active] = dev_act
    phi = np.maximum(0.0, n + np.concatenate(([0.0], np.cumsum(dev)[:-1])))

    need = n_head + max(d_int, 0) + 8
    raw_slice = base[skip : min(len(base), skip + need)]
    src = np.asarray(raw_slice, dtype=np.float64)
    if len(src) < need:
        src = np.pad(src, (0, need - len(src)))

    # Whiten static wood-body formants -> warp string excitation -> restore static wood-body formants
    exc = sosfilt(_body_sos(sr, -1), src)
    idx = np.minimum(phi.astype(np.int64), len(exc) - 2)
    frac = phi - idx
    warped_exc = (1.0 - frac) * exc[idx] + frac * exc[idx + 1]

    fret_gate = np.sin(math.pi * g) ** 4
    warped_exc *= 1.0 - (0.20 if abs(delta_semi) >= 2 else 0.08) * fret_gate
    if abs(delta_semi) >= 2:
        sos_buzz = _cached_butter_sos(2, 1500.0, "highpass", sr)
        if sos_buzz is not None:
            warped_exc += 0.045 * fret_gate * sosfilt(sos_buzz, np.abs(warped_exc))

    y = sosfilt(_body_sos(sr, +1), warped_exc)
    w = 0.5 - 0.5 * np.cos(math.pi * np.arange(1, n_xf + 1, dtype=np.float64) / (n_xf + 1))
    splice_start = max(0, n_head - n_xf + d_int)
    ref_tail = src[splice_start : splice_start + n_xf]
    if len(ref_tail) < n_xf:
        ref_tail = np.pad(ref_tail, (0, n_xf - len(ref_tail)))
    y[-n_xf:] = (1.0 - w) * y[-n_xf:] + w * ref_tail
    return y, d_int


def _fetch_studio_guitar_note(
    pitch: int,
    string: int,
    duration_samples: int,
    sr: int,
    velocity: float,
    technique: str,
    seed: int,
    rr_idx: int | None = None,
    slide_delta: int = 0,
    slide_tau: float = 0.065,
) -> np.ndarray | None:
    _ensure_sample_index()
    if not _SHINY_PITCHED and not _META_INDEX and not _FSS_INDEX:
        return None
    eff_rr = int(rr_idx) if rr_idx is not None else (1 + (abs(int(seed)) % 4))
    base: np.ndarray | None = None
    if _SHINY_PITCHED:
        vl, vel_gain = _select_shiny_layer_and_gain(velocity, technique)
        if vel_gain <= 0.0:
            return np.zeros(duration_samples, dtype=np.float64)
        base = _get_hybrid_shiny_base(pitch, string, vl, eff_rr, sr, technique == "PM")
    if base is None:
        layer, vel_gain = _select_vel_layer(velocity, technique)
        if vel_gain <= 0.0:
            return np.zeros(duration_samples, dtype=np.float64)
        base = _get_blended_studio_base(pitch, string, layer, sr, technique == "PM")
    if base is None:
        return None

    rng = np.random.default_rng(seed)
    amp_human = float(10.0 ** (rng.uniform(-0.35, 0.35) / 20.0))

    is_solo_voice = technique in {"CM", "DS", "SL"}
    use_cm_vib = technique in {"CM", "SL"} and string <= 3 and duration_samples >= round(0.36 * sr)
    vib_pad = round(0.002 * sr) if use_cm_vib else 0
    need_samples = duration_samples + vib_pad
    skip = min(len(base) // 4, round(0.011 * sr)) if technique in {"H", "P"} else 0

    do_slide = (slide_delta != 0) and (duration_samples >= round(0.085 * sr)) and (technique not in {"X", "PM"})
    if do_slide:
        tau_eff = float(np.clip(slide_tau, 0.042, min(0.092, 0.38 * (duration_samples / float(sr)))))
        head, d_int = _slide_head(base, skip, int(slide_delta), tau_eff, sr)
        if len(head) >= need_samples:
            seg = head[:need_samples].copy()
        else:
            rem = need_samples - len(head)
            cont_start = max(0, skip + len(head) + d_int)
            tail_src = np.asarray(base[cont_start : min(len(base), cont_start + rem)], dtype=np.float64)
            if len(tail_src) < rem:
                tail_src = np.pad(tail_src, (0, rem - len(tail_src)))
            seg = np.concatenate([head, tail_src])
    else:
        avail = max(0, len(base) - skip)
        if avail >= need_samples:
            seg = np.asarray(base[skip : skip + need_samples], dtype=np.float64).copy()
        else:
            seg = np.pad(np.asarray(base[skip:], dtype=np.float64), (0, need_samples - avail))

    atk_s = 0.014 if technique in {"H", "P"} else (0.0022 if is_solo_voice else 0.0008)
    attack = min(duration_samples, max(1, round(sr * atk_s)))
    if attack > 1:
        seg[:attack] *= np.sin(np.linspace(0.0, math.pi / 2.0, attack)) ** 2

    if is_solo_voice and string <= 3 and len(seg) >= 32:
        sos_cm = _peaking_eq_sos(min(0.42 * sr, 1550.0), 1.4 if technique != "DS" else 0.9, 0.9, sr)
        if pitch >= 69 and sr > 18000:
            sos_lp_cm = _cached_butter_sos(1, 8600.0, "lowpass", sr)
            if sos_lp_cm is not None:
                sos_cm = np.vstack([sos_cm, sos_lp_cm])
        seg = sosfilt(sos_cm, seg)
        if use_cm_vib:
            vib_start = 0.16 if do_slide else 0.12
            t_v = np.arange(duration_samples, dtype=np.float64) / float(sr)
            f_vib = 5.0 + 0.35 * (((seed % 7) / 6.0) - 0.5)
            ramp = np.clip((t_v - vib_start) / 0.18, 0.0, 1.0)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            d_semi = 0.11 * ramp * np.sin(2.0 * math.pi * f_vib * np.maximum(0.0, t_v - vib_start))
            tau = np.cumsum((2.0 ** (d_semi / 12.0)) - 1.0)
            r_pos = np.clip(np.arange(duration_samples, dtype=np.float64) + tau, 0.0, float(len(seg) - 1))
            r_idx = np.floor(r_pos).astype(np.int64)
            r_frac = r_pos - r_idx
            r_nxt = np.minimum(r_idx + 1, len(seg) - 1)
            seg = (1.0 - r_frac) * seg[r_idx] + r_frac * seg[r_nxt]
        else:
            seg = seg[:duration_samples]
    else:
        seg = seg[:duration_samples]

    rel_s = 0.048 if string >= 4 else 0.036
    release = min(duration_samples // 2, max(1, round(rel_s * sr)))
    if release > 1:
        seg[-release:] *= np.cos(np.linspace(0.0, math.pi / 2.0, release)) ** 2

    return seg * (vel_gain * amp_human)


def _pluck_fallback(pitch, string, duration_samples, sr, velocity, technique, seed):
    if duration_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    omega = 2 * np.pi * frequency / sr
    alpha = 0.18 + 0.30 * (string - 1) / 5
    dispersion_a = 0.95 - 0.25 * (string - 1) / 5
    z = np.exp(-1j * omega)
    response = ((1 - alpha) + alpha * z) * (dispersion_a + z) / (1 + dispersion_a * z)
    remaining = sr / frequency - (-np.angle(response) / omega)
    delay_length = int(np.floor(remaining))
    if delay_length < 1:
        raise ValueError("Pitch is too high for the sample rate")
    fractional = remaining - delay_length
    fractional_a = None
    if fractional > 1e-12:
        ratio = np.tan(omega * fractional / 2) / np.tan(omega / 2)
        fractional_a = (1 - ratio) / (1 + ratio)
    rng = np.random.default_rng(seed)
    phase = np.arange(delay_length, dtype=np.float64) / delay_length
    excitation = np.zeros(delay_length, dtype=np.float64)
    for harmonic in range(1, min(delay_length // 2, int(sr / (2 * frequency))) + 1):
        excitation += np.sin(np.pi * harmonic * 0.18) / harmonic ** 2 * np.cos(2 * np.pi * harmonic * phase)
    if technique not in {"H", "P"}:
        noise = rng.standard_normal(delay_length)
        excitation += 0.035 * (noise - np.roll(noise, round(0.18 * delay_length)))
    excitation -= excitation.mean()
    rms = np.sqrt(np.mean(excitation ** 2))
    if rms > 0:
        excitation /= rms
    output = np.zeros(duration_samples, dtype=np.float64)
    initial = min(delay_length, duration_samples)
    output[:initial] = excitation[:initial]
    fir_state = np.zeros(1)
    dispersion_state = np.zeros(1)
    fractional_state = np.zeros(1)
    gain = 10 ** (-3 / (frequency * (2.8 + 0.3 * (string - 1))))
    for start in range(delay_length, duration_samples, delay_length):
        end = min(start + delay_length, duration_samples)
        block = output[start - delay_length : end - delay_length]
        block, fir_state = lfilter([1 - alpha, alpha], [1], block, zi=fir_state)
        block, dispersion_state = lfilter([dispersion_a, 1], [1, dispersion_a], block, zi=dispersion_state)
        if fractional_a is not None:
            block, fractional_state = lfilter([fractional_a, 1], [1, fractional_a], block, zi=fractional_state)
        output[start:end] = gain * block
    t = np.arange(duration_samples, dtype=np.float64) / sr
    output *= np.exp(-t * (16 if technique == "PM" else 16 / 3.5))
    if technique == "PM":
        sos_pm = _cached_butter_sos(2, 700.0, "lowpass", sr)
        if sos_pm is not None:
            output = sosfilt(sos_pm, output)
    attack = min(duration_samples, max(1, round(sr * (0.015 if technique in {"H", "P"} else 0.002))))
    output[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    release = min(duration_samples, max(1, round(0.015 * sr)))
    output[-release:] *= np.linspace(1, 0, release) ** 2
    return output * (velocity / 127.0)


def _pluck(
    pitch,
    string,
    duration_samples,
    sr,
    velocity,
    technique,
    seed,
    *,
    rr_idx: int | None = None,
    slide_delta: int = 0,
    slide_tau: float = 0.065,
):
    if duration_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    if sr / frequency < 2.2:
        raise ValueError("Pitch is too high for the sample rate")
    studio = _fetch_studio_guitar_note(
        pitch,
        string,
        duration_samples,
        sr,
        velocity,
        technique,
        seed,
        rr_idx=rr_idx,
        slide_delta=slide_delta,
        slide_tau=slide_tau,
    )
    if studio is not None:
        return studio
    return _pluck_fallback(pitch, string, duration_samples, sr, velocity, technique, seed)


def _get_calibrated_noise_asset(kind: str, subkey: int, rr: int, sr: int) -> np.ndarray | None:
    """Peak-normalized, band-limited, click-free studio articulation noise sample."""
    cache_key = (f"{kind}:{subkey}", int(rr), int(sr))
    with _CACHE_LOCK:
        if cache_key in _NOISE_ASSET_CACHE:
            return _NOISE_ASSET_CACHE[cache_key]
    path: Path | None = None
    if kind == "release" and _SHINY_ZONES:
        z_midi = min(_SHINY_ZONES, key=lambda z: abs(z - subkey))
        path = _SHINY_RELEASE.get((z_midi, ((rr - 1) % 5) + 1))
    elif kind == "squeak":
        path = _EMILY_SQUEAK.get(((rr - 1) % 5) + 1)
    elif kind == "pickrest":
        path = _EMILY_PICKREST.get(((rr - 1) % 5) + 1)
    elif kind == "mute":
        path = _SHINY_MUTE.get((subkey, ((rr - 1) % 5) + 1))
    if path is None:
        return None
    raw = _read_cached_audio(path, sr, trim_onset=(kind != "squeak")).astype(np.float64)
    if raw.size < 16:
        return None
    if kind == "squeak":
        sos_bp = _cached_butter_sos(2, (1200.0, 6800.0), "bandpass", sr)
        if sos_bp is not None:
            raw = sosfilt(sos_bp, raw)
    elif kind in {"release", "pickrest"}:
        sos_hp = _cached_butter_sos(2, 150.0, "highpass", sr)
        if sos_hp is not None:
            raw = sosfilt(sos_hp, raw)
    peak = float(np.max(np.abs(raw)))
    if peak <= 1e-8:
        return None
    raw /= peak
    fi = min(len(raw) // 3, max(1, round(0.0025 * sr)))
    fo = min(len(raw) // 2, max(1, round(0.012 * sr)))
    raw[:fi] *= np.sin(np.linspace(0.0, math.pi / 2.0, fi)) ** 2
    raw[-fo:] *= np.cos(np.linspace(0.0, math.pi / 2.0, fo)) ** 2
    out = raw.astype(np.float32)
    out.flags.writeable = False
    with _CACHE_LOCK:
        _NOISE_ASSET_CACHE[cache_key] = out
    return out


_SLAP_BODY_GAIN = 0.38
_SLAP_CLICK_GAIN = 0.45


def _find_slap_click_sample() -> Path | None:
    if not _META_INDEX:
        return None
    selected_key = min(_META_INDEX.keys(), key=lambda k: (-k[0], k[1], -k[2]))
    return _META_INDEX[selected_key]


def _slap(sr, velocity, seed, rr_idx: int | None = None):
    """Real multi-RR wooden guitar body percussion (`perc_1..4_rr1..4`) reinforced by warm 95 Hz Helmholtz thump."""
    _ensure_sample_index()
    eff_rr = int(rr_idx) if rr_idx is not None else (1 + (abs(int(seed)) % 4))
    perc_id = 1 + ((abs(int(seed)) // 7) % 4)
    perc_path = _SHINY_PERC.get((perc_id, ((eff_rr - 1) % 4) + 1))
    cache_key = (int(sr), perc_id if perc_path is not None else 0, ((eff_rr - 1) % 4) + 1 if perc_path is not None else 0)
    with _CACHE_LOCK:
        cached = _SLAP_CACHE.get(cache_key)
    if cached is not None:
        return cached * ((velocity / 127.0) ** 1.25)

    n_samples = round(0.16 * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    body = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.045) + 0.45 * np.sin(2 * np.pi * 205 * t) * np.exp(-t / 0.028)
    if perc_path is not None:
        real_perc = _read_cached_audio(perc_path, sr)[:n_samples].astype(np.float64)
        if len(real_perc) < n_samples:
            real_perc = np.pad(real_perc, (0, n_samples - len(real_perc)))
        sos_hp = _cached_butter_sos(2, 65.0, "highpass", sr)
        if sos_hp is not None:
            real_perc = sosfilt(sos_hp, real_perc)
        p_peak = max(1e-9, float(np.max(np.abs(real_perc))))
        real_perc /= p_peak
        result = 0.28 * real_perc + 0.16 * body
    else:
        click_sample = _find_slap_click_sample()
        if click_sample is not None:
            source = _read_cached_audio(click_sample, sr)[:n_samples].astype(np.float64)
            if len(source) < n_samples:
                source = np.pad(source, (0, n_samples - len(source)))
        else:
            source = np.random.default_rng(seed).standard_normal(n_samples)
        sos_bp = _cached_butter_sos(2, (700.0, 2600.0), "bandpass", sr)
        click = (sosfilt(sos_bp, source) if sos_bp is not None else source) * np.exp(-t / 0.008)
        click /= max(1e-9, float(np.max(np.abs(click))))
        result = _SLAP_BODY_GAIN * (body + _SLAP_CLICK_GAIN * click)
        if click_sample is None:
            attack = min(n_samples, max(1, round(0.0012 * sr)))
            result[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
            rel = max(1, round(0.012 * sr))
            result[-rel:] *= np.linspace(1, 0, rel)
            return result * (velocity / 127.0)

    attack = min(n_samples, max(1, round(0.0010 * sr)))
    result[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    rel = max(1, round(0.014 * sr))
    result[-rel:] *= np.linspace(1, 0, rel)
    result.flags.writeable = False
    with _CACHE_LOCK:
        _SLAP_CACHE[cache_key] = result
    return result * ((velocity / 127.0) ** 1.25)


_SOLO_NON_RAKE_TECHS = {"X", "CM", "DS", "SL", "H", "P", "PB"}


def _schedule(events, bpm, sr, measure_beats=4.0, bars=None, chords=None, beat_to_sec=None):
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be positive and finite")
    groups, scheduled = {}, []
    for event in events:
        for field in ("start", "duration", "velocity"):
            if not math.isfinite(event[field]):
                raise ValueError(f"Nonfinite event {field}")
        if event["start"] < 0 or event["duration"] <= 0 or not 0 <= event["velocity"] <= 127:
            raise ValueError("Invalid event timing or velocity")
        _integer(event["string"], 1, 6, "string")
        _integer(event["pitch"], 0, 127, "pitch")
        _integer(event["fret"], 0, 24, "fret")
        if event.get("technique", "") not in _SOLO_NON_RAKE_TECHS:
            groups.setdefault(event["start"], []).append(event["string"])

    spb, b2s = 60.0 / bpm, (beat_to_sec or (lambda b: float(b) * (60.0 / bpm)))
    max_event_end_beat = max((e["start"] + e["duration"] for e in events), default=measure_beats)
    natural_ring_beats = min(4.0, max(2.5, 2.4 / spb))

    boundary_beats = set()
    nc_boundary_beats = set()
    sorted_by_start = sorted(events, key=lambda e: e["start"])
    prev_chord_id = None
    for e in sorted_by_start:
        bar_idx = e.get("bar", int(e["start"] // measure_beats))
        seg_idx = e.get("segment", 0)
        chord_sym = str(e.get("chord", "")).strip()
        cid = (bar_idx, seg_idx, chord_sym)
        if prev_chord_id is not None and cid != prev_chord_id:
            if chord_sym != prev_chord_id[2]:
                boundary_beats.add(float(e["start"]))
        prev_chord_id = cid
        nc_b = e.get("nc_bound")
        if nc_b is not None and math.isfinite(float(nc_b)):
            nc_boundary_beats.add(float(nc_b))
            boundary_beats.add(float(nc_b))
    bars_by_idx = {int(b.get("bar", idx)): b for idx, b in enumerate(bars or [])}
    n_harm_bars = max(len(chords or []), (max(bars_by_idx) + 1) if bars_by_idx else 0)
    prev_seg_chord = None
    for b_idx in range(n_harm_bars):
        b = bars_by_idx.get(b_idx)
        segs = (b.get("split_segments") or []) if b is not None else []
        if segs:
            for seg in sorted(segs, key=lambda s: float(s.get("beat", 0.0))):
                seg_t = float(b_idx * measure_beats + float(seg.get("beat", 0.0)))
                seg_ch = str(seg.get("chord", "")).strip()
                if seg_ch == "N.C." or ("frets" in seg and all(int(f) < 0 for f in seg.get("frets", [-1]))):
                    nc_boundary_beats.add(seg_t)
                    boundary_beats.add(seg_t)
                elif prev_seg_chord is not None and seg_ch != prev_seg_chord:
                    boundary_beats.add(seg_t)
                prev_seg_chord = seg_ch
        else:
            b_ch = str(b.get("chord", "") if b is not None else (chords[b_idx] if chords and b_idx < len(chords) else "")).strip()
            seg_t = float(b_idx * measure_beats)
            if b_ch == "N.C." or (b is not None and "frets" in b and all(int(f) < 0 for f in b.get("frets", [-1]))):
                nc_boundary_beats.add(seg_t)
                boundary_beats.add(seg_t)
            elif prev_seg_chord is not None and b_ch != prev_seg_chord:
                boundary_beats.add(seg_t)
            prev_seg_chord = b_ch
    sorted_boundaries = sorted(boundary_beats)
    sorted_nc_boundaries = sorted(nc_boundary_beats)

    for index, event in enumerate(events):
        rake = 0.0
        tech = event.get("technique", "")
        ev_dir = event.get("direction")
        roll_step = event.get("roll_step_sec")
        if tech not in _SOLO_NON_RAKE_TECHS:
            played = groups[event["start"]]
            unique_strings = sorted(set(played))
            if len(unique_strings) >= 2:
                if roll_step is not None and float(roll_step) > 0.0:
                    is_downstroke = True
                    step_sec = float(roll_step)
                else:
                    if ev_dir in ("↓", "↑"):
                        is_downstroke = (ev_dir == "↓")
                    else:
                        is_downstroke = abs((event["start"] * 2.0) % 2.0) < 1e-6
                    if len(unique_strings) >= 3:
                        step_sec = 0.0052 if is_downstroke else 0.0040
                    else:
                        step_sec = 0.0020
                rake = ((max(unique_strings) - event["string"]) if is_downstroke else (event["string"] - min(unique_strings))) * step_sec
        onset = max(0, round((b2s(event["start"]) + rake) * sr))
        if tech not in _SOLO_NON_RAKE_TECHS:
            played = groups[event["start"]]
            unique_strings = sorted(set(played))
            is_offbeat_single = (ev_dir == "↑") if ev_dir in ("↓", "↑") else ((event["start"] % 1.0) > 0.1)
            if len(unique_strings) == 1 and is_offbeat_single:
                micro = (((index * 7 + event["string"] * 13) % 17) / 17.0 - 0.5) * 2 * 0.0020
                onset = max(0, onset + round(micro * sr))
        next_bound = None
        if tech == "X":
            ring_beats = event["duration"]
        elif tech == "PM":
            ring_beats = min(max(event["duration"] * 1.35, 0.65), 1.4)
        elif tech == "PB":
            ring_beats = min(event["duration"] + 0.08, 0.75)
        elif tech in {"CM", "DS", "SL"}:
            ring_beats = event["duration"] + 0.12
        else:
            next_bound = next((b for b in sorted_boundaries if b > event["start"] + 0.05), event["start"] + natural_ring_beats)
            overlap_beats = 0.0 if (next_bound in nc_boundary_beats or beat_to_sec is not None) else (0.048 / spb)
            chord_span_beats = (next_bound - event["start"]) + overlap_beats
            remaining_song_beats = max(event["duration"], max_event_end_beat - event["start"] + 1.0)
            string_ring_cap = natural_ring_beats if event["string"] >= 4 else min(natural_ring_beats, max(2.2, 2.0 / spb))
            ring_beats = min(max(event["duration"], chord_span_beats), string_ring_cap, remaining_song_beats)
        extra_tail = (
            0.048
            if (beat_to_sec is not None and tech not in {"X", "PM", "PB", "CM", "DS", "SL"} and next_bound not in nc_boundary_beats)
            else 0.0
        )
        duration = max(1, round((b2s(event["start"] + ring_beats) - b2s(event["start"]) + extra_tail) * sr))
        ev_nc = event.get("nc_bound")
        next_nc = next((b for b in sorted_nc_boundaries if b > event["start"] + 1e-6), None)
        if ev_nc is not None and float(ev_nc) > event["start"] + 1e-6:
            next_nc = float(ev_nc) if next_nc is None else min(next_nc, float(ev_nc))
        if next_nc is not None:
            nc_sample = round(b2s(next_nc) * sr)
            if onset >= nc_sample:
                continue
            duration = min(duration, nc_sample - onset)
        scheduled.append(dict(event=event, onset=onset, duration=duration, index=index, same_string_choked=False))

    # 1. Physical 6-String Choke: a new pluck on the SAME string cuts the previous note on that string
    pitched = sorted((s for s in scheduled if s["event"].get("technique", "") != "X"), key=lambda s: (s["onset"], s["index"]))
    last_on_string: dict[int, dict] = {}
    choke_overlap = round(0.036 * sr)
    for item in pitched:
        string = item["event"]["string"]
        if string in last_on_string:
            previous = last_on_string[string]
            gap_samples = item["onset"] - previous["onset"]
            if gap_samples < previous["duration"]:
                previous["same_string_choked"] = (gap_samples < round(0.12 * sr))
            choke_len = max(1, gap_samples + choke_overlap)
            previous["duration"] = min(previous["duration"], choke_len)
        last_on_string[string] = item

    # 2. Polyphonic Solo Cluster ('CM' / 'DS' / 'SL') Choke:
    # Simultaneous Double-Stop dyads within 12ms share one cluster so both voices ring full duration!
    solo_items = sorted(
        (s for s in pitched if s["event"].get("technique", "") in {"CM", "DS", "SL"}),
        key=lambda s: (s["onset"], s["index"]),
    )
    cm_tol = round(0.012 * sr)
    cm_overlap = round(0.024 * sr)
    clusters: list[list[dict]] = []
    for it in solo_items:
        if clusters and (it["onset"] - clusters[-1][0]["onset"]) <= cm_tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    for prev_cluster, next_cluster in zip(clusters, clusters[1:]):
        t_next = next_cluster[0]["onset"]
        for it in prev_cluster:
            it["duration"] = min(it["duration"], max(1, t_next - it["onset"] + cm_overlap))

    # 3. Percussive 'X' Slap: thumb slap lets strings 4-6 ring through and gently damps treble strings 1-3
    slap_onsets = sorted(s["onset"] for s in scheduled if s["event"].get("technique", "") == "X")
    if slap_onsets:
        slap_overlap = round(0.12 * sr)
        for item in (s for s in pitched if s["event"]["string"] <= 3):
            for s_on in slap_onsets:
                if s_on > item["onset"] + round(0.02 * sr):
                    item["duration"] = min(item["duration"], max(1, s_on - item["onset"] + slap_overlap))
                    break

    return scheduled


def _room(mono, left, right, left_gain, right_gain, sr):
    delays = [round(left * sr), round(right * sr)]
    stereo = np.zeros((len(mono) + max(delays), 2), dtype=np.float64)
    stereo[: len(mono), :] = mono[:, None]
    for channel, delay, gain in ((0, delays[0], left_gain), (1, delays[1], right_gain)):
        stereo[delay : delay + len(mono), channel] += gain * mono
    return stereo


def _sf2_attacks(path, scheduled, sr, length):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    import fluidsynth
    synth = fluidsynth.Synth(samplerate=float(sr))
    result = np.zeros((length, 2), dtype=np.float64)
    try:
        soundfont = synth.sfload(str(path))
        if soundfont < 0 or synth.program_select(0, soundfont, 0, 25) < 0:
            raise RuntimeError("SoundFont load or guitar program selection failed")
        for item in scheduled:
            event = item["event"]
            if event["technique"] in {"X", "H", "P"} or item["duration"] <= 0:
                continue
            synth.all_sounds_off(0)
            synth.noteon(0, event["pitch"], int(event["velocity"]))
            size = min(round(0.08 * sr), item["duration"], length - item["onset"])
            attack = np.asarray(synth.get_samples(size), dtype=np.float64).reshape(-1, 2) / 32768
            synth.noteoff(0, event["pitch"])
            attack *= np.linspace(1, 0, len(attack))[:, None]
            result[item["onset"] : item["onset"] + size] += 0.12 * attack
    finally:
        synth.delete()
    return result


def _recording_origin_sec(project, arrangement: dict) -> float:
    if arrangement.get("time_origin_shifted"):
        return float(arrangement.get("warp_origin", getattr(project, "beat_origin", 0.0)))
    bss = arrangement.get("bar_start_seconds")
    if bss is None:
        bss = getattr(project, "bar_start_seconds", None)
    has_warp = bss is not None and hasattr(bss, "__len__") and len(bss) >= 2
    bt = arrangement.get("beat_times")
    if bt is None:
        bt = getattr(project, "beat_times", None)
    has_beat_warp = bt is not None and hasattr(bt, "__len__") and len(bt) >= 2
    if has_beat_warp and has_warp:
        return min(float(bt[0]), float(bss[0]))
    if has_beat_warp:
        return float(bt[0])
    if has_warp:
        return float(bss[0])
    return float(getattr(project, "beat_origin", 0.0))


def _resolve_slide_delta(event: dict) -> int:
    fret = int(event["fret"])
    if fret <= 0:
        return 0
    s_from = event.get("slide_from_fret")
    if s_from is not None:
        delta = int(np.clip(int(s_from) - fret, -4, 4))
        return max(delta, -fret)
    if event.get("technique") == "SL":
        return -2 if fret >= 2 else -1
    return 0


def synthesize_realistic_guitar(
    project,
    arrangement: dict,
    out_wav_path: str | Path | None = None,
    sr: int = 44100,
    sf2_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize stereo (N, 2) float32 multi-RR studio acoustic guitar audio at `sr` Hz."""
    _sample_rate(sr)
    _ensure_sample_index()
    measure_beats = float(arrangement.get("measure_beats", 4.0))
    if not math.isfinite(measure_beats) or measure_beats <= 0:
        raise ValueError("measure_beats must be positive and finite")
    bpm = float(project.bpm)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be positive and finite")
    spb = 60.0 / bpm
    beats_per_bar = measure_beats
    n_bars = int(arrangement.get("bar_count", max((e.get("bar", 0) for e in arrangement["events"]), default=0) + 1))
    capo = int(arrangement.get("capo", 0) or 0)

    bar_start_seconds = arrangement.get("bar_start_seconds")
    if bar_start_seconds is None:
        bar_start_seconds = getattr(project, "bar_start_seconds", None)
    has_warp = (bar_start_seconds is not None
                and hasattr(bar_start_seconds, "__len__")
                and len(bar_start_seconds) >= 2)
    raw_bss = []
    if has_warp:
        raw_bss = [float(x) for x in bar_start_seconds]
        if not all(math.isfinite(x) for x in raw_bss) or any(b <= a for a, b in zip(raw_bss, raw_bss[1:])):
            raise ValueError("bar_start_seconds must be finite and strictly increasing")

    beat_times_raw = arrangement.get("beat_times")
    if beat_times_raw is None:
        beat_times_raw = getattr(project, "beat_times", None)
    has_beat_warp = (beat_times_raw is not None
                     and hasattr(beat_times_raw, "__len__")
                     and len(beat_times_raw) >= 2)
    raw_bt = []
    if has_beat_warp:
        raw_bt = [float(x) for x in beat_times_raw]
        if not all(math.isfinite(x) for x in raw_bt) or any(b <= a for a, b in zip(raw_bt, raw_bt[1:])):
            raise ValueError("beat_times must be finite and strictly increasing")

    if arrangement.get("time_origin_shifted"):
        warp_origin = 0.0
    elif has_beat_warp or has_warp:
        warp_origin = _recording_origin_sec(project, arrangement)
    else:
        warp_origin = 0.0
    if has_warp:
        bar_start_seconds = [x - warp_origin for x in raw_bss]
        avg_bar_dur = (bar_start_seconds[-1] - bar_start_seconds[0]) / max(1, len(bar_start_seconds) - 1)
    if has_beat_warp:
        bt = [x - warp_origin for x in raw_bt]
        bt_avg_dur = (bt[-1] - bt[0]) / max(1, len(bt) - 1)

    bar_dynamics = arrangement.get("bar_dynamics")
    bars_list = arrangement.get("bars", [])
    already_scaled = bool(bars_list and isinstance(bars_list[0], dict) and "dynamic_scale" in bars_list[0])
    has_dynamics = (not already_scaled) and bar_dynamics is not None and hasattr(bar_dynamics, "__len__") and len(bar_dynamics) > 0

    def _beat_to_warped_sec(beat_pos: float) -> float:
        if has_beat_warp:
            max_idx = len(bt) - 1
            if beat_pos <= max_idx:
                return float(np.interp(beat_pos, np.arange(len(bt), dtype=np.float64), bt))
            return bt[-1] + (beat_pos - max_idx) * bt_avg_dur
        if not has_warp:
            return beat_pos * spb
        b_idx = max(0, int(beat_pos // beats_per_bar))
        local_beat = beat_pos - b_idx * beats_per_bar
        if b_idx + 1 < len(bar_start_seconds):
            bar_t0 = bar_start_seconds[b_idx]
            bar_t1 = bar_start_seconds[b_idx + 1]
        elif b_idx < len(bar_start_seconds):
            bar_t0 = bar_start_seconds[b_idx]
            bar_t1 = bar_t0 + avg_bar_dur
        else:
            bar_t0 = bar_start_seconds[-1] + (b_idx - len(bar_start_seconds) + 1) * avg_bar_dur
            bar_t1 = bar_t0 + avg_bar_dur
        bar_dur = max(0.0, bar_t1 - bar_t0)
        return bar_t0 + (local_beat / beats_per_bar) * bar_dur

    scheduled = _schedule(
        arrangement["events"],
        bpm,
        sr,
        measure_beats=measure_beats,
        bars=arrangement.get("bars"),
        chords=arrangement.get("chords"),
        beat_to_sec=_beat_to_warped_sec if (has_warp or has_beat_warp) else None,
    )

    if has_dynamics:
        for item in scheduled:
            event = item["event"]
            b_idx = event.get("bar", int(event["start"] // beats_per_bar))
            if 0 <= b_idx < len(bar_dynamics):
                dyn = float(np.clip(bar_dynamics[b_idx], 0.76, 1.08))
                orig_vel = int(event["velocity"])
                new_vel = 0 if orig_vel <= 0 else max(1, min(127, round(orig_vel * dyn)))
                item["event"] = dict(event, velocity=new_vel)

    nominal_sec = _beat_to_warped_sec(max(0.0, n_bars * beats_per_bar))
    total_sec = (max(float(bar_start_seconds[-1]), nominal_sec) + 0.5) if has_warp else nominal_sec
    length = max(
        [1, round(total_sec * sr)]
        + [s["onset"] + (min(round(0.16 * sr), s["duration"]) if s["event"]["technique"] == "X" else s["duration"]) for s in scheduled]
    )

    dry_l = np.zeros(length, dtype=np.float64)
    dry_r = np.zeros(length, dtype=np.float64)
    noise_l = np.zeros(length, dtype=np.float64)
    noise_r = np.zeros(length, dtype=np.float64)

    rr_gen = _RoundRobin(seed=0x20265EED)
    noise_rng = np.random.default_rng(0x51EA2026)
    slide_tau_default = float(np.clip(0.125 * spb, 0.045, 0.088))
    rel_cluster_counts: dict[int, int] = {}
    last_wound_fretted: dict[int, tuple[int, int, float]] = {}  # string -> (end_sample, fret, peak)
    last_squeak_sample = -sr

    ordered_items = sorted(scheduled, key=lambda s: (s["onset"], s["index"]))
    for item in ordered_items:
        event = item["event"]
        tech = event.get("technique", "")
        seed = item["index"] * 100003 + event["pitch"] * 101 + event["string"]
        pan_pos = (3.5 - event["string"]) * 0.065
        gain_l = math.sqrt(0.5 * (1.0 - pan_pos))
        gain_r = math.sqrt(0.5 * (1.0 + pan_pos))

        if tech == "X":
            rr_perc = rr_gen.next(("perc",), 4)
            signal = _slap(sr, event["velocity"], seed, rr_idx=rr_perc)[: item["duration"]]
            slide_delta = 0
        elif item["duration"] > 0:
            z_key = min(_SHINY_ZONES, key=lambda z: abs(z - event["pitch"])) if _SHINY_ZONES else event["pitch"]
            rr_note = rr_gen.next(("zone", z_key), 4)
            slide_delta = _resolve_slide_delta(event)
            slide_tau = float(event.get("slide_dur_sec", slide_tau_default))
            signal = _pluck(
                event["pitch"],
                event["string"],
                item["duration"],
                sr,
                event["velocity"],
                tech,
                seed,
                rr_idx=rr_note,
                slide_delta=slide_delta,
                slide_tau=slide_tau,
            )
        else:
            signal = np.zeros(0, dtype=np.float64)
            slide_delta = 0

        if len(signal) == 0:
            continue
        end_idx = min(length, item["onset"] + len(signal))
        seg = signal[: end_idx - item["onset"]]
        if len(seg) == 0:
            continue
        dry_l[item["onset"] : end_idx] += gain_l * seg
        dry_r[item["onset"] : end_idx] += gain_r * seg

        if tech == "X":
            continue

        note_peak = float(np.max(np.abs(seg[: min(len(seg), round(0.15 * sr))])))
        if note_peak <= 1e-5:
            continue

        # A. Fret-squeak injection on multi-fret slide ('SL' / slide_from_fret) or rapid wound-string shift
        s_num = int(event["string"])
        fret = int(event["fret"])
        if abs(slide_delta) >= 2 and s_num >= 3 and _EMILY_SQUEAK and noise_rng.random() < 0.65:
            sq = _get_calibrated_noise_asset("squeak", 0, rr_gen.next(("squeak",), 5), sr)
            if sq is not None:
                sq_db = -28.5 if s_num >= 5 else (-30.5 if s_num == 4 else -35.5)
                sq_len = min(len(sq), max(16, round((slide_tau_default + 0.030) * sr)))
                sq_start = item["onset"] + round(0.012 * sr)
                sq_end = min(length, sq_start + sq_len)
                if sq_end > sq_start + 8:
                    win = np.sin(np.linspace(0.0, math.pi, sq_end - sq_start)) ** 2
                    sq_seg = sq[: sq_end - sq_start] * win * (note_peak * (10.0 ** (sq_db / 20.0)))
                    noise_l[sq_start:sq_end] += gain_l * sq_seg
                    noise_r[sq_start:sq_end] += gain_r * sq_seg
                    last_squeak_sample = sq_start
        elif s_num >= 4 and fret > 0 and _EMILY_SQUEAK and (item["onset"] - last_squeak_sample) >= round(0.85 * sr):
            prev_info = last_wound_fretted.get(s_num)
            if prev_info is not None:
                prev_end, prev_fret, prev_peak = prev_info
                gap_s = (item["onset"] - prev_end) / float(sr)
                if -0.04 <= gap_s <= 0.32 and abs(fret - prev_fret) >= 3 and noise_rng.random() < 0.38:
                    sq = _get_calibrated_noise_asset("squeak", 0, rr_gen.next(("squeak",), 5), sr)
                    if sq is not None:
                        sq_len = min(len(sq), round(0.075 * sr))
                        sq_end = max(0, item["onset"] - round(0.004 * sr))
                        sq_start = max(0, sq_end - sq_len)
                        if sq_end > sq_start + 8 and sq_end <= length:
                            win = np.sin(np.linspace(0.0, math.pi, sq_end - sq_start)) ** 2
                            sq_seg = sq[: sq_end - sq_start] * win * (min(note_peak, prev_peak) * (10.0 ** (-34.0 / 20.0)))
                            noise_l[sq_start:sq_end] += gain_l * sq_seg
                            noise_r[sq_start:sq_end] += gain_r * sq_seg
                            last_squeak_sample = sq_start
        if s_num >= 4 and fret > 0:
            last_wound_fretted[s_num] = (end_idx, fret, note_peak)

        # B. Subtle real string-release / finger-lift noise at note-off when string is genuinely damped (not re-plucked)
        if (
            _SHINY_RELEASE
            and fret > 0
            and not item.get("same_string_choked", False)
            and tech not in {"PM", "H", "P"}
            and item["duration"] >= round(0.18 * sr)
        ):
            rel_start = max(0, end_idx - round(0.036 * sr))
            bucket = rel_start // max(1, round(0.030 * sr))
            if rel_cluster_counts.get(bucket, 0) < 2 and noise_rng.random() < 0.65:
                rel_wave = _get_calibrated_noise_asset("release", event["pitch"], rr_gen.next(("rel",), 5), sr)
                if rel_wave is not None:
                    rel_end = min(length, rel_start + len(rel_wave))
                    if rel_end > rel_start + 8:
                        tail_rms = float(np.sqrt(np.mean(seg[max(0, len(seg) - round(0.04 * sr)) :] ** 2)))
                        e_ratio = float(np.clip(tail_rms / max(1e-6, 0.35 * note_peak), 0.12, 1.0))
                        rel_gain = note_peak * (10.0 ** (-27.5 / 20.0)) * math.sqrt(e_ratio)
                        r_seg = rel_wave[: rel_end - rel_start] * rel_gain
                        noise_l[rel_start:rel_end] += gain_l * r_seg
                        noise_r[rel_start:rel_end] += gain_r * r_seg
                        rel_cluster_counts[bucket] = rel_cluster_counts.get(bucket, 0) + 1

    # Global studio noise-bus guard: ensure mechanical noises stay <= -31 dB below pitched guitar RMS
    pitched_rms = float(np.sqrt(0.5 * (np.mean(dry_l ** 2) + np.mean(dry_r ** 2)))) if length > 0 else 0.0
    noise_rms = float(np.sqrt(0.5 * (np.mean(noise_l ** 2) + np.mean(noise_r ** 2)))) if length > 0 else 0.0
    if noise_rms > 1e-9 and pitched_rms > 1e-9:
        max_noise_rms = pitched_rms * (10.0 ** (-31.0 / 20.0))
        if noise_rms > max_noise_rms:
            n_scale = max_noise_rms / noise_rms
            noise_l *= n_scale
            noise_r *= n_scale
        dry_l += noise_l
        dry_r += noise_r

    stereo_dry = np.column_stack([dry_l, dry_r])

    # Studio acoustic EQ chain + Coupled 6-String Sympathetic & Wooden-Room Shared FFT Convolution
    if len(stereo_dry) >= 16 and np.max(np.abs(stereo_dry)) > 0:
        sos_hp = _cached_butter_sos(2, 75.0, "highpass", sr)
        sos_mud = _peaking_eq_sos(215.0, -3.2, 1.15, sr)
        sos_sheen = _peaking_eq_sos(min(0.42 * sr, 4500.0), 1.6, 0.9, sr)
        sos_parts = [s for s in (sos_hp, sos_mud, sos_sheen) if s is not None]
        sos_chain = np.vstack(sos_parts)
        stereo_dry[:, 0] = sosfilt(sos_chain, stereo_dry[:, 0])
        stereo_dry[:, 1] = sosfilt(sos_chain, stereo_dry[:, 1])

        ir_l, ir_r = _get_studio_room_ir(sr)
        symp_ir = _get_sympathetic_ir(sr, capo=capo)
        comb_len = max(len(ir_l), len(ir_r), len(symp_ir))
        wet_gain = 10.0 ** (-18.5 / 20.0)
        hl = np.zeros(comb_len, dtype=np.float64)
        hr = np.zeros(comb_len, dtype=np.float64)
        hl[: len(ir_l)] += wet_gain * ir_l
        hr[: len(ir_r)] += wet_gain * ir_r
        # Slight natural stereo spread across sympathetic strings (bass slightly left, treble slightly right)
        hl[: len(symp_ir)] += 0.96 * symp_ir
        hr[: len(symp_ir)] += 1.04 * symp_ir

        mono_sum = 0.5 * (stereo_dry[:, 0] + stereo_dry[:, 1])
        nfft = sp_fft.next_fast_len(len(mono_sum) + comb_len - 1, real=True)
        X_mono = sp_fft.rfft(mono_sum, nfft)
        wet_l = sp_fft.irfft(X_mono * sp_fft.rfft(hl, nfft), nfft)
        wet_r = sp_fft.irfft(X_mono * sp_fft.rfft(hr, nfft), nfft)

        # Keep output length tightly aligned to room tail so sympathetic resonance blooms internally without bloating stem end
        out_len = len(stereo_dry) + len(ir_l) - 1
        stereo = np.zeros((out_len, 2), dtype=np.float64)
        stereo[: len(stereo_dry), :] = stereo_dry
        stereo[:, 0] += wet_l[:out_len]
        stereo[:, 1] += wet_r[:out_len]
        end_fade = min(out_len // 4, max(1, round(0.035 * sr)))
        if end_fade > 1:
            fade_curve = np.cos(np.linspace(0.0, math.pi / 2.0, end_fade)) ** 2
            stereo[-end_fade:, :] *= fade_curve[:, None]
    else:
        stereo = stereo_dry.copy()

    if sf2_path is not None:
        stereo += _sf2_attacks(sf2_path, scheduled, sr, len(stereo))

    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0:
        act_rms, _ = _active_rms(stereo, sr, gate_db=-46.0)
        desired_gain = min(3.5, (_TARGET_GUITAR_ACTIVE_RMS / act_rms)) if act_rms > 1e-6 else (0.55 / peak)
        min_gain = 0.28 / peak
        stereo *= max(min_gain, desired_gain)
        abs_s = np.abs(stereo)
        over = abs_s > 0.66
        if np.any(over):
            stereo[over] = np.sign(stereo[over]) * (0.66 + 0.20 * np.tanh((abs_s[over] - 0.66) / 0.20))

    if not np.isfinite(stereo).all():
        raise ValueError("Nonfinite synthesized audio")
    stereo = stereo.astype(np.float32)
    if out_wav_path is not None:
        sf.write(str(out_wav_path), stereo, sr, format="WAV", subtype="PCM_16")
    return stereo


def mix_real_vocal_and_guitar(
    vocal_wav_path: str | Path,
    project,
    arrangement: dict,
    out_wav_path: str | Path,
    sr: int = 44100,
    vocal_gain: float = 1.0,
    guitar_gain: float = 0.85,
) -> dict:
    """Time-align the simulated acoustic guitar (`beat_origin` + `bpm` timeline) with the real singer's `vocals.wav`, calibrate studio vocal-in-front balance, apply subtle room ambience, write stereo WAV, and return spectral/envelope mix summary descriptors."""
    _sample_rate(sr)
    origin_sec = _recording_origin_sec(project, arrangement)
    if not all(math.isfinite(v) for v in (vocal_gain, guitar_gain, float(getattr(project, "beat_origin", 0.0)), origin_sec)):
        raise ValueError("Nonfinite gain or recording origin")
    vocal_channels, source_sr = sf.read(str(vocal_wav_path), dtype="float64", always_2d=True)
    if not np.isfinite(vocal_channels).all():
        raise ValueError("Nonfinite vocal samples")
    vocal = vocal_channels.mean(axis=1)
    if source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        vocal = resample_poly(vocal, sr // divisor, source_sr // divisor)
    v_peak = float(np.max(np.abs(vocal))) if vocal.size else 0.0
    if v_peak > 1e-6 and len(vocal) >= 16:
        v_rms, n_act = _active_rms(vocal, sr, gate_db=-48.0)
        if n_act >= 20 and v_rms > 1e-6:
            sos_vhp = _cached_butter_sos(2, 80.0, "highpass", sr)
            if sos_vhp is not None:
                vocal = sosfilt(sos_vhp, vocal)
            v_peak = max(1e-9, float(np.max(np.abs(vocal))))
            v_scale = min(_TARGET_VOCAL_ACTIVE_RMS / v_rms, 0.90 / v_peak)
            vocal = vocal * v_scale
    guitar = synthesize_realistic_guitar(project, arrangement, sr=sr).astype(np.float64)
    offset = int(round(origin_sec * sr))
    guitar = np.pad(guitar, ((offset, 0), (0, 0))) if offset >= 0 else guitar[min(-offset, len(guitar)) :]
    vocal_room = _room(vocal, 0.021, 0.035, 0.032, 0.026, sr)
    length = max(len(vocal_room), len(guitar))
    padded_guitar = np.pad(guitar, ((0, length - len(guitar)), (0, 0)))
    padded_vocal_room = np.pad(vocal_room, ((0, length - len(vocal_room)), (0, 0)))
    mix = vocal_gain * padded_vocal_room + guitar_gain * padded_guitar
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    ceil_peak = 10.0 ** (-1.0 / 20.0)
    norm_gain = (ceil_peak / peak) if peak > ceil_peak else 1.0
    if norm_gain != 1.0:
        mix *= norm_gain
    sf.write(str(out_wav_path), mix, sr, format="WAV", subtype="PCM_16")
    wet_vocal_mono = (vocal_gain * norm_gain) * padded_vocal_room.mean(axis=1)
    scaled_guitar_mono = (guitar_gain * norm_gain) * padded_guitar.mean(axis=1)
    return evaluate_vocal_guitar_mix(wet_vocal_mono, scaled_guitar_mono, sr=sr)


_ACOUSTIC_SHARE = 0.6
_KERNEL_REFERENCE_HZ = 330.0
_KERNEL_PARTIALS = 6
_TIS_WEIGHTS = np.array([2.0, 11.0, 17.0, 16.0, 19.0, 7.0])
_JUST_RATIOS = ((1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3), (45, 32), (3, 2), (8, 5), (5, 3), (16, 9), (15, 8))
_TIS_BASIS = np.exp(-2j * np.pi * np.outer(np.arange(1, 7), np.arange(12)) / 12.0)


def _interval_class_kernels():
    """Graded interval-class kernels replacing binary consonant / clash masks."""
    if _interval_class_kernels.cache is None:
        from .reharmonizer import _plomp_levelt_roughness
        f0, n = _KERNEL_REFERENCE_HZ, _KERNEL_PARTIALS
        raw = np.array([
            0.5 * (_plomp_levelt_roughness(f0, f0 * 2 ** (k / 12), n)
                   + _plomp_levelt_roughness(f0, f0 * 2 ** ((12 - k) / 12), n))
            for k in range(12)
        ])
        span = raw.max() - raw.min()
        roughness = (raw - raw.min()) / span if span > 0 else np.zeros(12)
        tenney = np.array([math.log2(p * q) for p, q in _JUST_RATIOS])
        tenney = 0.5 * (tenney + tenney[(-np.arange(12)) % 12])
        harmonicity = 1.0 - tenney / tenney.max()
        consonance_by_class = 0.5 * (1.0 - roughness) + 0.5 * harmonicity
        intervals = (np.arange(12)[:, None] - np.arange(12)[None, :]) % 12
        unit = np.eye(12)
        tis01 = 0.5 * (1.0 + _tis_frame_cosine(unit[[0] * 12], unit))
        f_k = f0 * (2.0 ** (np.arange(12, dtype=np.float64) / 12.0))
        pl_dyad = _pl_pair_roughness(f0, f_k)
        reg_dyad = np.clip(1.0 - pl_dyad / 0.22, 0.0, 1.0)
        acoustic_dyad = 0.75 * consonance_by_class + 0.25 * reg_dyad
        blended = _ACOUSTIC_SHARE * acoustic_dyad + (1.0 - _ACOUSTIC_SHARE) * tis01
        threshold = 0.5 * (blended[2] + blended[10])
        _interval_class_kernels.cache = (consonance_by_class[intervals], float(threshold))
    return _interval_class_kernels.cache


_interval_class_kernels.cache = None


def _pl_pair_roughness(f_a: np.ndarray | float, f_b: np.ndarray | float) -> np.ndarray:
    """Register-aware Plomp-Levelt (1965) / Sethares (1993) sensory roughness curve for unit-weight spectral peak pairs."""
    f_min = np.minimum(f_a, f_b)
    df = np.abs(f_a - f_b)
    s_pl = 0.24 / (0.0207 * f_min + 18.96)
    return np.exp(-3.5 * s_pl * df) - np.exp(-5.75 * s_pl * df)


def _tis_frame_cosine(chroma_a: np.ndarray, chroma_b: np.ndarray) -> np.ndarray:
    """Per-frame cosine similarity in Tonal Interval Space (6 weighted DFT coefficients of each chroma frame)."""
    basis = _TIS_BASIS

    def tiv(chroma):
        totals = chroma.sum(axis=1, keepdims=True)
        normalised = np.divide(chroma, totals, out=np.zeros_like(chroma), where=totals > 0)
        return (normalised @ basis.T) * _TIS_WEIGHTS

    ta, tb = tiv(chroma_a), tiv(chroma_b)
    dot = np.real(np.sum(ta * np.conj(tb), axis=1))
    norms = np.linalg.norm(ta, axis=1) * np.linalg.norm(tb, axis=1)
    return np.clip(np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 1e-12), -1.0, 1.0)


def evaluate_vocal_guitar_mix(
    vocal_audio: np.ndarray,
    guitar_audio: np.ndarray,
    sr: int = 44100,
) -> dict:
    """Heuristic spectral and envelope summary descriptors for a vocal/guitar mix."""
    _sample_rate(sr)

    def mono(audio):
        array = np.asarray(audio, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] in (1, 2):
            array = array.mean(axis=1)
        elif array.ndim != 1:
            raise ValueError("Expected mono or stereo audio")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite audio samples")
        return array

    vocal, guitar = mono(vocal_audio), mono(guitar_audio)
    length = max(len(vocal), len(guitar))
    vocal = np.pad(vocal, (0, length - len(vocal)))
    guitar = np.pad(guitar, (0, length - len(guitar)))
    frequencies = np.fft.rfftfreq(4096, 1 / sr)
    mask = (frequencies >= 50) & (frequencies <= 5000)
    masked_freqs = frequencies[mask]
    pcs = np.rint(69 + 12 * np.log2(masked_freqs / 440)).astype(np.int64) % 12
    starts = range(0, max(1, length), 1024)
    window = np.hanning(4096)
    n_peaks = min(4, len(masked_freqs))

    def analyze(audio):
        chroma, rms, peak_freqs, peak_amps = [], [], [], []
        for start in starts:
            frame = np.pad(audio[start : start + 4096], (0, max(0, start + 4096 - len(audio))))
            rms.append(np.sqrt(np.mean(frame ** 2)))
            power = (np.abs(np.fft.rfft(frame * window)) ** 2)[mask]
            vector = np.bincount(pcs, weights=power, minlength=12)
            total = vector.sum()
            chroma.append(vector / total if total > 0 else vector)
            if total > 0 and n_peaks > 0:
                if len(power) >= 3:
                    loc = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
                else:
                    loc = np.arange(len(power))
                if len(loc) >= n_peaks:
                    top_local = loc[np.argpartition(power[loc], -n_peaks)[-n_peaks:]]
                else:
                    top_local = np.argpartition(power, -n_peaks)[-n_peaks:]
                amps = np.sqrt(power[top_local])
                a_sum = amps.sum()
                peak_freqs.append(masked_freqs[top_local])
                peak_amps.append(amps / a_sum if a_sum > 0 else amps)
            else:
                peak_freqs.append(np.full(n_peaks, 330.0))
                peak_amps.append(np.zeros(n_peaks))
        return np.asarray(chroma), np.asarray(rms), np.asarray(peak_freqs), np.asarray(peak_amps)

    vc, vr, vf, va = analyze(vocal)
    gc, gr, gf, ga = analyze(guitar)
    w_pair = va[:, :, None] * ga[:, None, :]
    pl_rough = np.sum(w_pair * _pl_pair_roughness(vf[:, :, None], gf[:, None, :]), axis=(1, 2))
    register_consonance = np.clip(1.0 - pl_rough / 0.22, 0.0, 1.0)
    consonance_kernel, clash_threshold = _interval_class_kernels()
    chroma_consonance = np.einsum("bi,ij,bj->b", vc, consonance_kernel, gc)
    acoustic_consonance = 0.75 * chroma_consonance + 0.25 * register_consonance
    tis_coherence = 0.5 * (1.0 + _tis_frame_cosine(vc, gc))
    consonance = _ACOUSTIC_SHARE * acoustic_consonance + (1.0 - _ACOUSTIC_SHARE) * tis_coherence
    overlap = (vr > 0) & (gr > 0)
    weights = vr[overlap] * gr[overlap]
    active_count = int(overlap.sum())
    if active_count and weights.sum() > 0:
        spectral = float(np.average(consonance[overlap], weights=weights))
    elif active_count:
        spectral = float(np.mean(consonance[overlap]))
    else:
        spectral = 0.0
    spectral = min(1.0, max(0.0, spectral))
    clash_ratio = float(np.mean(consonance[overlap] < clash_threshold)) if active_count else 0.0
    vo = np.maximum(0, np.diff(vr))
    go = np.maximum(0, np.diff(gr))
    correlation = float(np.corrcoef(vo, go)[0, 1]) if len(vo) > 1 and np.std(vo) > 0 and np.std(go) > 0 else 0.0
    summed = vocal + guitar
    peak = float(np.max(np.abs(summed))) if length else 0.0
    rms = float(np.sqrt(np.mean(summed ** 2))) if length else 0.0
    crest = float(20 * np.log10(peak / rms)) if rms > 0 else 0.0
    return dict(
        spectral_consonance=spectral,
        clash_frame_ratio=clash_ratio,
        beat_sync_corr=correlation,
        crest_factor_db=crest,
        active_overlap_frames=active_count,
    )
