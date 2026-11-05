"""Multi-Sampled 6-String Acoustic Guitar Synthesizer (375 Meta Steel Guitar OGG Samples + FSS Steel-String WAV Layer + Karplus-Strong Fallback + Per-String Polyphonic Scheduler) & Vocal/Guitar Stem Mixer."""
from __future__ import annotations
from pathlib import Path
import math
import os
import re
import threading
import numpy as np
import soundfile as sf
from scipy.signal import butter, iirpeak, lfilter, resample_poly, sosfilt
from .chord_voicing import _integer


def _resolve_samples_base() -> Path:
    env_dir = os.environ.get("MELODYDESK_GUITAR_SAMPLES")
    if env_dir:
        cand = Path(env_dir)
        if cand.is_dir():
            return cand
    return Path(__file__).resolve().parents[2] / "guitar_samples"


_SAMPLES_BASE = _resolve_samples_base()
_META_OGG_DIR = _SAMPLES_BASE / "meta-steel-guitar" / "ogg"
_FSS_WAV_DIR = _SAMPLES_BASE / "FSS-SteelStringGuitar-SFZ-20200521" / "samples"
_CACHE_LOCK = threading.RLock()

_NOTE_TO_SEMI = {
    "C": 0, "Cs": 1, "D": 2, "Ds": 3, "E": 4, "F": 5,
    "Fs": 6, "G": 7, "Gs": 8, "A": 9, "As": 10, "B": 11,
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
_WAVE_CACHE: dict[tuple[str, int], np.ndarray] = {}
_BLENDED_NOTE_CACHE: dict[tuple[int, int, int, int, bool], np.ndarray] = {}
_SLAP_CACHE: dict[int, np.ndarray] = {}
_INDEX_BUILT = False
_INDEX_DIRS_SCANNED: tuple[str, str] | None = None


def _sample_rate(sr):
    return _integer(sr, 8001, float("inf"), "sample rate")


def _ensure_sample_index():
    global _INDEX_BUILT, _INDEX_DIRS_SCANNED
    current_dirs = (str(_META_OGG_DIR), str(_FSS_WAV_DIR))
    if _INDEX_BUILT or _INDEX_DIRS_SCANNED == current_dirs:
        return
    with _CACHE_LOCK:
        if _INDEX_BUILT or _INDEX_DIRS_SCANNED == current_dirs:
            return
        if _META_OGG_DIR.is_dir():
            for p in _META_OGG_DIR.glob("*.ogg"):
                m = re.match(r"INSGuitarSteel_(String\w+)_Vel0([123])_([A-G]s?)(\d)_01\.ogg", p.name)
                if m:
                    s_num = _STRING_NAME_TO_NUM[m.group(1)]
                    vel = int(m.group(2))
                    midi = (int(m.group(4)) + 1) * 12 + _NOTE_TO_SEMI[m.group(3)]
                    _META_INDEX[(s_num, midi, vel)] = p
        if _FSS_WAV_DIR.is_dir():
            for p in _FSS_WAV_DIR.glob("*.wav"):
                m = re.match(r"(MV|HV)_(\d+)\.wav", p.name)
                if m:
                    _FSS_INDEX[(int(m.group(2)), m.group(1))] = p
        _INDEX_BUILT = bool(_META_INDEX or _FSS_INDEX)
        _INDEX_DIRS_SCANNED = current_dirs


def _read_cached_audio(path: Path, target_sr: int) -> np.ndarray:
    key = (str(path), target_sr)
    with _CACHE_LOCK:
        if key in _WAVE_CACHE:
            return _WAVE_CACHE[key]
    y, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if src_sr != target_sr:
        g = math.gcd(int(src_sr), int(target_sr))
        y = resample_poly(y, int(target_sr) // g, int(src_sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        idx = int(np.argmax(np.abs(y) > 0.025 * peak))
        trim = max(0, idx - int(0.0008 * target_sr))
        if trim > 0:
            y = y[trim:]
    with _CACHE_LOCK:
        _WAVE_CACHE[key] = y
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


def _select_vel_layer(velocity: float, technique: str) -> tuple[int, float]:
    if velocity <= 0:
        return 1, 0.0
    vel_norm = float(np.clip(velocity / 127.0, 0.0, 1.0))
    if technique == "PM":
        return 1, float(0.85 * (0.25 + 0.75 * vel_norm))
    if technique == "CM":
        return (3 if velocity >= 82 else 2), float(0.85 + 0.28 * vel_norm)
    if velocity <= 68:
        layer = 1
    elif velocity <= 94:
        layer = 2
    else:
        layer = 3
    span_pos = np.clip((velocity - 1.0) / 126.0, 0.0, 1.0)
    db_offset = -2.2 + 3.4 * span_pos
    return layer, float(10.0 ** (db_offset / 20.0))


def _get_blended_studio_base(pitch: int, string: int, layer: int, sr: int, is_pm: bool) -> np.ndarray | None:
    key = (string, int(pitch), layer, sr, is_pm)
    with _CACHE_LOCK:
        if key in _BLENDED_NOTE_CACHE:
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

    max_len = round(4.5 * sr)
    if primary_wave is not None:
        seg = primary_wave[: min(len(primary_wave), max_len)].astype(np.float64)
        seg = _pitch_shift_resample(seg, shift_semi) * 2.45
        if abs(shift_semi) > 0.5:
            seg = sosfilt(butter(2, min(0.48 * sr, 9500.0), fs=sr, output='sos'), seg)
        if fss_wave is not None:
            f_seg = fss_wave[: min(len(fss_wave), max_len)].astype(np.float64)
            f_seg = _pitch_shift_resample(f_seg, fss_shift)
            n_blend = min(len(seg), len(f_seg))
            seg[:n_blend] = 0.90 * seg[:n_blend] + 0.10 * (f_seg[:n_blend] * 0.42)
    elif fss_wave is not None:
        seg = fss_wave[: min(len(fss_wave), max_len)].astype(np.float64)
        seg = _pitch_shift_resample(seg, fss_shift) * 0.55
    else:
        return None

    t = np.arange(len(seg), dtype=np.float64) / sr
    if is_pm:
        seg *= np.exp(-t * 9.5)
        seg = sosfilt(butter(2, 1150, fs=sr, output="sos"), seg)
    else:
        decay_rate = 0.24 if string >= 4 else (0.52 if layer == 1 else 0.42)
        seg *= np.exp(-t * decay_rate)
        if layer == 1 and string <= 3:
            seg = sosfilt(butter(1, min(0.45 * sr, 5800.0), fs=sr, output='sos'), seg)
    seg.flags.writeable = False
    with _CACHE_LOCK:
        _BLENDED_NOTE_CACHE[key] = seg
    return seg


def _fetch_studio_guitar_note(
    pitch: int,
    string: int,
    duration_samples: int,
    sr: int,
    velocity: float,
    technique: str,
    seed: int,
) -> np.ndarray | None:
    _ensure_sample_index()
    if not _META_INDEX and not _FSS_INDEX:
        return None
    layer, vel_gain = _select_vel_layer(velocity, technique)
    base = _get_blended_studio_base(pitch, string, layer, sr, technique == "PM")
    if base is None:
        return None
    rng = np.random.default_rng(seed)
    amp_human = float(10.0 ** (rng.uniform(-0.6, 0.6) / 20.0))

    skip = min(len(base) // 4, round(0.011 * sr)) if technique in {"H", "P"} else 0
    avail = len(base) - skip
    if avail >= duration_samples:
        seg = base[skip : skip + duration_samples].copy()
    else:
        seg = np.pad(base[skip:], (0, duration_samples - avail))

    atk_s = 0.014 if technique in {"H", "P"} else 0.0008
    attack = min(duration_samples, max(1, round(sr * atk_s)))
    if attack > 1:
        seg[:attack] *= np.sin(np.linspace(0.0, np.pi / 2.0, attack)) ** 2

    # Smooth note-off release (32 ms on treble strings 1-3, 45 ms on bass strings 4-6)
    rel_s = 0.045 if string >= 4 else 0.032
    release = min(duration_samples // 2, max(1, round(rel_s * sr)))
    if release > 1:
        seg[-release:] *= np.cos(np.linspace(0.0, np.pi / 2.0, release)) ** 2

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
        output = sosfilt(butter(2, 700, fs=sr, output="sos"), output)
    attack = min(duration_samples, max(1, round(sr * (0.015 if technique in {"H", "P"} else 0.002))))
    output[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    release = min(duration_samples, max(1, round(0.015 * sr)))
    output[-release:] *= np.linspace(1, 0, release) ** 2
    return output * (velocity / 127.0)


def _pluck(pitch, string, duration_samples, sr, velocity, technique, seed):
    if duration_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    if sr / frequency < 2.2:
        raise ValueError("Pitch is too high for the sample rate")
    studio = _fetch_studio_guitar_note(pitch, string, duration_samples, sr, velocity, technique, seed)
    if studio is not None:
        return studio
    return _pluck_fallback(pitch, string, duration_samples, sr, velocity, technique, seed)


_SLAP_BODY_GAIN = 0.38
_SLAP_CLICK_GAIN = 0.45


def _find_slap_click_sample() -> Path | None:
    if not _META_INDEX:
        return None
    selected_key = min(_META_INDEX.keys(), key=lambda k: (-k[0], k[1], -k[2]))
    return _META_INDEX[selected_key]


def _slap(sr, velocity, seed):
    """Two-mode body-tap synthesis (95 Hz + 210 Hz decaying sinusoids) plus bandpass-filtered transient click."""
    _ensure_sample_index()
    click_sample = _find_slap_click_sample()
    with _CACHE_LOCK:
        cached = _SLAP_CACHE.get(sr)
    if cached is not None and click_sample is not None:
        return cached * (velocity / 127.0)
    n_samples = round(0.16 * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    body = (np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.045) + 0.55 * np.sin(2 * np.pi * 210 * t) * np.exp(-t / 0.028))
    if click_sample is not None:
        source = _read_cached_audio(click_sample, sr)[:n_samples].astype(np.float64)
        if len(source) < n_samples:
            source = np.pad(source, (0, n_samples - len(source)))
    else:
        source = np.random.default_rng(seed).standard_normal(n_samples)
    click = sosfilt(butter(2, [700, 2600], btype="bandpass", fs=sr, output="sos"), source) * np.exp(-t / 0.008)
    click /= max(1e-9, float(np.max(np.abs(click))))
    result = _SLAP_BODY_GAIN * (body + _SLAP_CLICK_GAIN * click)
    attack = min(n_samples, max(1, round(0.0012 * sr)))
    result[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    rel = max(1, round(0.012 * sr))
    result[-rel:] *= np.linspace(1, 0, rel)
    if click_sample is not None:
        result.flags.writeable = False
        with _CACHE_LOCK:
            _SLAP_CACHE[sr] = result
    return result * (velocity / 127.0)


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
        if event["technique"] not in {"X", "CM", "H", "P", "PB"}:
            groups.setdefault(event["start"], []).append(event["string"])

    spb, b2s = 60.0 / bpm, (beat_to_sec or (lambda b: float(b) * (60.0 / bpm)))
    max_event_end_beat = max((e["start"] + e["duration"] for e in events), default=measure_beats)
    natural_ring_beats = min(4.0, max(2.5, 2.4 / spb))

    # Identify chord / bar transition boundaries and N.C. silence boundaries
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
        if tech not in {"X", "CM", "H", "P", "PB"}:
            played = groups[event["start"]]
            unique_strings = sorted(set(played))
            if len(unique_strings) >= 2:
                is_downstroke = abs((event["start"] * 2.0) % 2.0) < 1e-6
                step_sec = 0.011 if len(unique_strings) >= 3 else 0.002
                rake = ((max(unique_strings) - event["string"]) if is_downstroke else (event["string"] - min(unique_strings))) * step_sec
        onset = max(0, round((b2s(event["start"]) + rake) * sr))
        if tech not in {"X", "CM", "H", "P", "PB"}:
            played = groups[event["start"]]
            unique_strings = sorted(set(played))
            if len(unique_strings) == 1 and (event["start"] % 1.0) > 0.1:
                micro = (((index * 7 + event["string"] * 13) % 17) / 17.0 - 0.5) * 2 * 0.0025
                onset = max(0, onset + round(micro * sr))
        if tech == "X":
            ring_beats = event["duration"]
        elif tech == "PM":
            ring_beats = min(max(event["duration"] * 1.35, 0.65), 1.4)
        elif tech == "PB":
            # Cap passing-bass pickup note ring duration
            ring_beats = min(event["duration"] + 0.08, 0.75)
        elif tech == "CM":
            ring_beats = event["duration"] + 0.12
        else:
            # Let string ring across the current chord/bar up to the next chord change boundary
            next_bound = next((b for b in sorted_boundaries if b > event["start"] + 0.05), event["start"] + natural_ring_beats)
            overlap_beats = 0.0 if (next_bound in nc_boundary_beats or beat_to_sec is not None) else (0.045 / spb)
            chord_span_beats = (next_bound - event["start"]) + overlap_beats
            remaining_song_beats = max(event["duration"], max_event_end_beat - event["start"] + 1.0)
            string_ring_cap = natural_ring_beats if event['string'] >= 4 else min(natural_ring_beats, max(1.5, 1.45 / spb))
            ring_beats = min(max(event["duration"], chord_span_beats), string_ring_cap, remaining_song_beats)
        duration = max(1, round((b2s(event["start"] + ring_beats) - b2s(event["start"]) + (0.045 if (beat_to_sec is not None and tech not in {"X", "PM", "PB", "CM"} and next_bound not in nc_boundary_beats) else 0.0)) * sr))
        ev_nc = event.get("nc_bound")
        next_nc = next((b for b in sorted_nc_boundaries if b > event["start"] + 1e-6), None)
        if ev_nc is not None and float(ev_nc) > event["start"] + 1e-6:
            next_nc = float(ev_nc) if next_nc is None else min(next_nc, float(ev_nc))
        if next_nc is not None:
            nc_sample = round(b2s(next_nc) * sr)
            if onset >= nc_sample:
                continue
            duration = min(duration, nc_sample - onset)
        scheduled.append(dict(event=event, onset=onset, duration=duration, index=index))

    # 1. Physical 6-String Choke: a new pluck on the SAME string cuts the previous note on that string
    pitched = sorted((s for s in scheduled if s["event"].get("technique", "") != "X"), key=lambda s: s["onset"])
    last_on_string = {}
    choke_overlap = round(0.032 * sr)
    for item in pitched:
        string = item["event"]["string"]
        if string in last_on_string:
            previous = last_on_string[string]
            choke_len = max(1, item["onset"] - previous["onset"] + choke_overlap)
            previous["duration"] = min(previous["duration"], choke_len)
        last_on_string[string] = item

    # 2. Monophonic Lead Melody ('CM') Choke: consecutive melody notes choke previous melody note across strings
    cm_items = [s for s in pitched if s["event"].get("technique", "") == "CM"]
    cm_overlap = round(0.024 * sr)
    for prev_cm, next_cm in zip(cm_items, cm_items[1:]):
        if next_cm["onset"] >= prev_cm["onset"]:
            prev_cm["duration"] = min(prev_cm["duration"], max(1, next_cm["onset"] - prev_cm["onset"] + cm_overlap))

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


def synthesize_realistic_guitar(
    project,
    arrangement: dict,
    out_wav_path: str | Path | None = None,
    sr: int = 44100,
    sf2_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize stereo (N, 2) float32 multi-sampled acoustic guitar audio at `sr` Hz.

    Components:
    1. 375 `Meta Steel Guitar` OGG recordings (Strings 6..1 across 3 velocity layers) blended with
       `FSS Steel-String` WAV samples, with Karplus-Strong physical-modeling fallback when samples are absent.
    2. 6-string polyphonic scheduler with same-string re-pluck truncation, chord/bar boundary damping,
       and treble truncation on `'X'` slap events.
    3. Articulation envelopes and delays:
       - Strumming rake delay (11 ms/string sweep)
       - `'X'` slap (95/210 Hz decaying sinusoids + bandpass click)
       - `'PM'` palm mute (low-velocity layer + 1150 Hz low-pass filter)
       - `'H'` / `'P'` hammer-on & pull-off (softened attack ramp)
    4. Beat-to-time piecewise linear interpolation when `bar_start_seconds` or `beat_times` are supplied.
    """
    _sample_rate(sr)
    measure_beats = float(arrangement.get("measure_beats", 4.0))
    if not math.isfinite(measure_beats) or measure_beats <= 0:
        raise ValueError("measure_beats must be positive and finite")
    bpm = float(project.bpm)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be positive and finite")
    spb = 60.0 / bpm
    beats_per_bar = measure_beats
    n_bars = int(arrangement.get("bar_count", max((e.get("bar", 0) for e in arrangement["events"]), default=0) + 1))

    bar_start_seconds = arrangement.get("bar_start_seconds")
    if bar_start_seconds is None:
        bar_start_seconds = getattr(project, "bar_start_seconds", None)
    has_warp = (bar_start_seconds is not None
                and hasattr(bar_start_seconds, '__len__')
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
                     and hasattr(beat_times_raw, '__len__')
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
    has_dynamics = (not already_scaled) and bar_dynamics is not None and hasattr(bar_dynamics, '__len__') and len(bar_dynamics) > 0

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
                dyn = float(np.clip(bar_dynamics[b_idx], 0.55, 1.15))
                orig_vel = int(event["velocity"])
                new_vel = 0 if orig_vel <= 0 else max(1, min(127, round(orig_vel * dyn)))
                item["event"] = dict(event, velocity=new_vel)








    nominal_sec = _beat_to_warped_sec(max(0.0, n_bars * beats_per_bar))
    total_sec = (max(float(bar_start_seconds[-1]), nominal_sec) + 0.5) if has_warp else nominal_sec
    length = max(
        [1, round(total_sec * sr)]
        + [s["onset"] + (min(round(0.16 * sr), s["duration"]) if s["event"]["technique"] == "X" else s["duration"]) for s in scheduled]
    )

    stereo_dry = np.zeros((length, 2), dtype=np.float64)
    for item in scheduled:
        event = item["event"]
        seed = item["index"] * 100003 + event["pitch"] * 101 + event["string"]
        signal = (
            _slap(sr, event["velocity"], seed)[: item["duration"]]
            if event["technique"] == "X"
            else _pluck(
                event["pitch"],
                event["string"],
                item["duration"],
                sr,
                event["velocity"],
                event["technique"],
                seed,
            )
            if item["duration"] > 0
            else np.zeros(0)
        )
        if len(signal) == 0:
            continue
        # Subtle natural stereo mic image: bass strings (6..4) slightly warm left, treble strings (3..1) slightly right
        pan_pos = (3.5 - event["string"]) * 0.055  # -0.14 (string 6) to +0.14 (string 1)
        gain_l = math.sqrt(0.5 * (1.0 - pan_pos))
        gain_r = math.sqrt(0.5 * (1.0 + pan_pos))
        end_idx = min(length, item["onset"] + len(signal))
        seg = signal[: end_idx - item["onset"]]
        stereo_dry[item["onset"] : end_idx, 0] += gain_l * seg
        stereo_dry[item["onset"] : end_idx, 1] += gain_r * seg

    # Warm stereo Haas body resonance: concert-hall unplugged depth (-19 dB, LP 3200 Hz)
    mono_sum = stereo_dry.mean(axis=1)
    body_lp = sosfilt(butter(2, 3200.0, fs=sr, output='sos'), mono_sum)
    haas_gain = 10.0 ** (-19.0 / 20.0)
    d_l1, d_l2 = round(0.017 * sr), round(0.041 * sr)
    d_r1, d_r2 = round(0.027 * sr), round(0.053 * sr)
    tail = max(d_l2, d_r2) + 1
    stereo = np.pad(stereo_dry, ((0, tail), (0, 0)))
    for d in (d_l1, d_l2):
        stereo[d : d + len(body_lp), 0] += haas_gain * body_lp
    for d in (d_r1, d_r2):
        stereo[d : d + len(body_lp), 1] += haas_gain * body_lp

    if sf2_path is not None:
        stereo += _sf2_attacks(sf2_path, scheduled, sr, len(stereo))
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo *= 0.88 / peak
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
    """Time-align the simulated acoustic guitar (`beat_origin` + `bpm` timeline) with the real singer's `vocals.wav`, apply subtle room ambience, write stereo WAV, and return spectral/envelope mix summary descriptors."""
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
    guitar = synthesize_realistic_guitar(project, arrangement, sr=sr).astype(np.float64)
    offset = int(round(origin_sec * sr))
    guitar = np.pad(guitar, ((offset, 0), (0, 0))) if offset >= 0 else guitar[min(-offset, len(guitar)) :]
    vocal_room = _room(vocal, 0.023, 0.039, 0.04, 0.03, sr)
    length = max(len(vocal_room), len(guitar))
    padded_guitar = np.pad(guitar, ((0, length - len(guitar)), (0, 0)))
    padded_vocal_room = np.pad(vocal_room, ((0, length - len(vocal_room)), (0, 0)))
    mix = vocal_gain * padded_vocal_room + guitar_gain * padded_guitar
    peak = np.max(np.abs(mix)) if mix.size else 0.0
    norm_gain = (10 ** (-1 / 20) / peak) if peak > 0 else 1.0
    if peak > 0:
        mix *= norm_gain
    sf.write(str(out_wav_path), mix, sr, format="WAV", subtype="PCM_16")
    # Evaluate the normalized, room-processed stems at the exact gain scale written to disk.
    wet_vocal_mono = (vocal_gain * norm_gain) * padded_vocal_room.mean(axis=1)
    scaled_guitar_mono = (guitar_gain * norm_gain) * padded_guitar.mean(axis=1)
    return evaluate_vocal_guitar_mix(wet_vocal_mono, scaled_guitar_mono, sr=sr)


_ACOUSTIC_SHARE = 0.6        # weight of acoustic (Plomp-Levelt + Helmholtz) consonance vs. TIS coherence per frame
_KERNEL_REFERENCE_HZ = 330.0  # ≈E4: centre of the shared vocal / acoustic-guitar melodic register
_KERNEL_PARTIALS = 6
_TIS_WEIGHTS = np.array([2.0, 11.0, 17.0, 16.0, 19.0, 7.0])  # Bernardes et al. TIS DFT-coefficient weights
# Just-intonation ratios per interval class (unison .. major seventh) for Helmholtz harmonicity.
_JUST_RATIOS = ((1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3), (45, 32), (3, 2), (8, 5), (5, 3), (16, 9), (15, 8))
_TIS_BASIS = np.exp(-2j * np.pi * np.outer(np.arange(1, 7), np.arange(12)) / 12.0)  # (6, 12)


def _interval_class_kernels():
    """Graded interval-class kernels replacing binary consonant / clash masks.

    For interval class k (pitch-class difference, octave unknown in chroma, so k and its inversion 12-k are
    averaged):
      * sensory roughness r_k: multi-partial Plomp-Levelt / Sethares dissonance of two harmonic tones at the
        reference register, min-max normalised to [0, 1] (unison 0, minor second 1);
      * Helmholtz harmonicity h_k: 1 - TenneyHeight(p/q) / max, with TenneyHeight = log2(p*q) of the just ratio
        (shared low partials -> smooth fusion);
      * acoustic consonance c_k = 0.5 (1 - r_k) + 0.5 h_k.
    Returns (consonance 12x12 kernel indexed [i, j] by (i - j) mod 12, clash_threshold).  The clash threshold
    is the blended (acoustic + TIS) score of an isolated whole-tone dyad: frames rougher than a bare major
    second (dominated by m2 / M7 / tritone interactions) count as clashes, while M2 / m7 colour stays neutral.
    """
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
        # Blended single-dyad scores (chroma kernel + reference-register Plomp-Levelt + TIS) for the clash threshold.
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
        return (normalised @ basis.T) * _TIS_WEIGHTS  # (frames, 6)

    ta, tb = tiv(chroma_a), tiv(chroma_b)
    dot = np.real(np.sum(ta * np.conj(tb), axis=1))
    norms = np.linalg.norm(ta, axis=1) * np.linalg.norm(tb, axis=1)
    return np.clip(np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 1e-12), -1.0, 1.0)


def evaluate_vocal_guitar_mix(
    vocal_audio: np.ndarray,
    guitar_audio: np.ndarray,
    sr: int = 44100,
) -> dict:
    """Heuristic spectral and envelope summary descriptors for a vocal/guitar mix.

    spectral_consonance: energy-weighted mean over co-active STFT frames combining (1) register-aware Plomp-Levelt /
    Sethares roughness between the dominant STFT spectral peaks in Hz, (2) a 12x12 interval-class kernel
    (Plomp-Levelt + Tenney-height harmonicity), and (3) 6-coefficient TIS chroma cosine similarity (range [0, 1]).
    clash_frame_ratio: fraction of co-active frames where the composite score falls below the whole-tone reference
    threshold (range [0, 1]).  beat_sync_corr: correlation of positive RMS onset envelopes.  crest_factor_db:
    peak-to-RMS of the summed signals.  active_overlap_frames: frames where both stems have non-zero energy.
    """
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
    # 1. Register-aware multi-peak Plomp-Levelt / Sethares sensory consonance in Hz across top STFT peaks.
    w_pair = va[:, :, None] * ga[:, None, :]
    pl_rough = np.sum(w_pair * _pl_pair_roughness(vf[:, :, None], gf[:, None, :]), axis=(1, 2))
    register_consonance = np.clip(1.0 - pl_rough / 0.22, 0.0, 1.0)
    # 2. Graded Helmholtz / Plomp-Levelt interval-class kernels on chroma.
    consonance_kernel, clash_threshold = _interval_class_kernels()
    chroma_consonance = np.einsum("bi,ij,bj->b", vc, consonance_kernel, gc)
    acoustic_consonance = 0.75 * chroma_consonance + 0.25 * register_consonance
    # 3. Tonal Interval Space frame coherence: cosine similarity of 6-coefficient weighted chroma DFTs, mapped to [0, 1].
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
