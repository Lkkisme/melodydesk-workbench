"""Studio Multi-Sampled 6-String Acoustic Guitar Engine (375 Per-String Velocity-Layered Studio Recordings + FSS Seagull Steel-String Layer + True 6-String 'Let Ring' Physical Choke) & Real-Vocal Unplugged Mixer."""
from __future__ import annotations
from pathlib import Path
import math
import re
import threading
import numpy as np
import soundfile as sf
from scipy.signal import butter, iirpeak, lfilter, resample_poly, sosfilt
from .chord_voicing import _integer

_INSTALL_ROOT = Path(__file__).resolve().parents[4]
_SAMPLES_BASE = (_INSTALL_ROOT / "work" / "guitar_samples") if (_INSTALL_ROOT / "work" / "guitar_samples").is_dir() else Path(r"E:\MelodyDesk\work\guitar_samples")
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


def _sample_rate(sr):
    _integer(sr, 8001, float("inf"), "sample rate")


def _ensure_sample_index():
    global _INDEX_BUILT
    if _INDEX_BUILT:
        return
    with _CACHE_LOCK:
        if _INDEX_BUILT:
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
    if technique == "PM":
        return 1, 0.85
    if technique == "CM":
        return (3 if velocity >= 82 else 2), 1.08
    if velocity <= 54:
        layer = 1
        span_pos = (velocity - 1.0) / 53.0
    elif velocity <= 86:
        layer = 2
        span_pos = (velocity - 55.0) / 31.0
    else:
        layer = 3
        span_pos = (velocity - 87.0) / 40.0
    db_offset = -1.8 + 3.6 * np.clip(span_pos, 0.0, 1.0)
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
                    best_s, best_m = min(any_str, key=lambda sm: (abs(sm[1] - pitch), abs(sm[0] - string)))
                    primary_wave = _read_cached_audio(_META_INDEX[(best_s, best_m, layer)], sr)
                    shift_semi = float(pitch - best_m)

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
            seg[:n_blend] = 0.82 * seg[:n_blend] + 0.18 * (f_seg[:n_blend] * 0.48)
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
        seg *= np.exp(-t * 0.18)
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

    # Smooth 45 ms note-off release (30-80 ms per Meta Steel Guitar spec so low strings never click)
    rel_s = 0.045 if string >= 4 else 0.032
    release = min(duration_samples // 2, max(1, round(rel_s * sr)))
    if release > 1:
        seg[-release:] *= np.cos(np.linspace(0.0, np.pi / 2.0, release)) ** 2

    return seg * (vel_gain * amp_human)


def _pluck_fallback(pitch, string, duration_samples, sr, velocity, technique, seed):
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
    frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    if sr / frequency < 2.2:
        raise ValueError("Pitch is too high for the sample rate")
    studio = _fetch_studio_guitar_note(pitch, string, duration_samples, sr, velocity, technique, seed)
    if studio is not None:
        return studio
    return _pluck_fallback(pitch, string, duration_samples, sr, velocity, technique, seed)


_SLAP_BODY_GAIN = 0.38
_SLAP_CLICK_GAIN = 0.45


def _slap(sr, velocity, seed):
    """Warm cedar-top thumb/body tap: 95 Hz + 210 Hz wood resonance plus a subtle string click."""
    _ensure_sample_index()
    with _CACHE_LOCK:
        cached = _SLAP_CACHE.get(sr)
    if cached is not None and (6, 40, 3) in _META_INDEX:
        return cached * (velocity / 127.0)
    n_samples = round(0.16 * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    body = (np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.045) + 0.55 * np.sin(2 * np.pi * 210 * t) * np.exp(-t / 0.028))
    if (6, 40, 3) in _META_INDEX:
        source = _read_cached_audio(_META_INDEX[(6, 40, 3)], sr)[:n_samples].astype(np.float64)
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
    if (6, 40, 3) in _META_INDEX:
        result.flags.writeable = False
        with _CACHE_LOCK:
            _SLAP_CACHE[sr] = result
    return result * (velocity / 127.0)


def _schedule(events, bpm, sr, measure_beats=4.0):
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

    spb = 60.0 / bpm
    max_event_end_beat = max((e["start"] + e["duration"] for e in events), default=measure_beats)
    natural_ring_beats = min(4.0, max(2.5, 2.4 / spb))

    # Identify chord / bar transition boundaries so fretted chords damp cleanly when the left hand changes chord shapes
    boundary_beats = set()
    sorted_by_start = sorted(events, key=lambda e: e["start"])
    prev_chord_id = None
    for e in sorted_by_start:
        bar_idx = e.get("bar", int(e["start"] // measure_beats))
        seg_idx = e.get("segment", 0)
        chord_sym = e.get("chord", "")
        cid = (bar_idx, seg_idx, chord_sym)
        if prev_chord_id is not None and cid != prev_chord_id:
            # If the chord symbol itself changed, damp at the transition beat; if same chord across bars, let ring
            if chord_sym != prev_chord_id[2] or not chord_sym:
                boundary_beats.add(float(round(e["start"] * 4.0) / 4.0))
        prev_chord_id = cid
    sorted_boundaries = sorted(boundary_beats)

    for index, event in enumerate(events):
        rake = 0.0
        tech = event.get("technique", "")
        if tech not in {"X", "CM", "H", "P", "PB"}:
            played = groups[event["start"]]
            unique_strings = sorted(set(played))
            if len(unique_strings) >= 2:
                is_downstroke = abs((event["start"] * 2.0) % 2.0) < 1e-6
                step_sec = 0.012 if len(unique_strings) >= 3 else 0.005
                rake = ((max(unique_strings) - event["string"]) if is_downstroke else (event["string"] - min(unique_strings))) * step_sec
        onset = round((event["start"] * spb + rake) * sr)
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
            # Passing bass pickup note damps right as the target downbeat bass note arrives
            ring_beats = min(event["duration"] + 0.08, 0.75)
        elif tech == "CM":
            ring_beats = min(event["duration"] + 0.12, 2.5)
        else:
            # Let string ring across the current chord/bar up to the next chord change boundary (+45ms hand transition overlap)
            next_bound = next((b for b in sorted_boundaries if b > event["start"] + 0.05), event["start"] + natural_ring_beats)
            chord_span_beats = (next_bound - event["start"]) + (0.045 / spb)
            remaining_song_beats = max(event["duration"], max_event_end_beat - event["start"] + 1.0)
            ring_beats = min(max(event["duration"], chord_span_beats), natural_ring_beats, remaining_song_beats)
        duration = max(1, round(ring_beats * spb * sr))
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
        if next_cm["onset"] > prev_cm["onset"]:
            prev_cm["duration"] = min(prev_cm["duration"], max(1, next_cm["onset"] - prev_cm["onset"] + cm_overlap))

    # 3. Percussive 'X' Slap: thumb slap lets bass strings (5-6) ring through and only gently damps treble strings (1-3)
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


def synthesize_realistic_guitar(
    project,
    arrangement: dict,
    out_wav_path: str | Path | None = None,
    sr: int = 44100,
    sf2_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize stereo (N, 2) float32 studio-sampled acoustic guitar audio at 44.1 kHz.

    Engine:
    1. 375 Per-String × 3-Velocity-Layer Studio Recordings (`Meta Steel Guitar`, E2-C6 across Strings 6..1)
       blended with close-miked `FSS Seagull Steel-String Acoustic Guitar` studio recordings.
    2. True 6-String 'Let Ring' Polyphony & Physical String Choke:
       Notes on different strings ring together freely across arpeggios and strums; re-plucking the same
       string damps the previous vibration; a percussive 'X' slap gently damps treble strings while bass rings on.
    3. Articulation physics:
       - Strumming Rake Delay (12 ms/string downstroke & upstroke sweep)
       - 'X' Thumb Slap (warm 95/210 Hz cedar-top body tap + subtle string click)
       - 'P.M.' Palm Mute (Vel01 soft layer + bridge palm damping + 1150 Hz low-pass warmth)
       - 'H' / 'P' Hammer-on & Pull-off (suppressed pick transient, smooth legato transition)
    4. Beat-Synchronous Time-Warping:
       When `arrangement['bar_start_seconds']` provides real detected bar onset times,
       events are placed at the actual rubato-following positions instead of a rigid BPM grid.
    """
    _sample_rate(sr)
    measure_beats = float(arrangement.get("measure_beats", 4.0))
    bpm = project.bpm
    spb = 60.0 / bpm
    beats_per_bar = measure_beats
    n_bars = int(arrangement.get("bar_count", max((e.get("bar", 0) for e in arrangement["events"]), default=0) + 1))

    bar_start_seconds = arrangement.get("bar_start_seconds")
    has_warp = (bar_start_seconds is not None
                and hasattr(bar_start_seconds, '__len__')
                and len(bar_start_seconds) >= 2)
    if has_warp:
        raw_bss = [float(x) for x in bar_start_seconds]
        bss_origin = raw_bss[0]
        bar_start_seconds = [max(0.0, x - bss_origin) for x in raw_bss]
        avg_bar_dur = (bar_start_seconds[-1] - bar_start_seconds[0]) / max(1, len(bar_start_seconds) - 1)

    beat_times_raw = arrangement.get("beat_times")
    has_beat_warp = (beat_times_raw is not None
                     and hasattr(beat_times_raw, '__len__')
                     and len(beat_times_raw) >= 2)
    if has_beat_warp:
        bt = [max(0.0, float(x) - float(beat_times_raw[0])) for x in beat_times_raw]
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
        bar_dur = max(0.25, bar_t1 - bar_t0)
        return bar_t0 + (local_beat / beats_per_bar) * bar_dur

    scheduled = _schedule(arrangement["events"], bpm, sr, measure_beats=measure_beats)

    if has_warp or has_beat_warp or has_dynamics:
        for item in scheduled:
            event = item["event"]
            if has_warp or has_beat_warp:
                sched_start_beat = item["onset"] / (spb * sr)
                sched_end_beat = (item["onset"] + item["duration"]) / (spb * sr)
                t0_sec = _beat_to_warped_sec(sched_start_beat)
                t1_sec = _beat_to_warped_sec(sched_end_beat)
                item["onset"] = max(0, round(t0_sec * sr))
                item["duration"] = max(1, round((t1_sec - t0_sec) * sr))
            if has_dynamics:
                b_idx = event.get("bar", int(event["start"] // beats_per_bar))
                if 0 <= b_idx < len(bar_dynamics):
                    dyn = float(np.clip(bar_dynamics[b_idx], 0.55, 1.15))
                    item["event"] = dict(event, velocity=max(1, min(127, round(event["velocity"] * dyn))))

    length = max(
        (s["onset"] + (round(0.16 * sr) if s["event"]["technique"] == "X" else s["duration"]) for s in scheduled),
        default=1,
    )
    if has_warp:
        total_sec = max(float(bar_start_seconds[-1]), n_bars * beats_per_bar * spb) + 0.5
        length = max(length, round(total_sec * sr))

    stereo_dry = np.zeros((length, 2), dtype=np.float64)
    for item in scheduled:
        event = item["event"]
        seed = item["index"] * 100003 + event["pitch"] * 101 + event["string"]
        signal = (
            _slap(sr, event["velocity"], seed)
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
    """Time-align the simulated acoustic guitar (`beat_origin` + `bpm` timeline) with the real singer's `vocals.wav`, apply subtle room ambience, write stereo WAV, and return objective acoustic evaluation metrics."""
    _sample_rate(sr)
    if not all(math.isfinite(v) for v in (vocal_gain, guitar_gain, project.beat_origin)):
        raise ValueError("Nonfinite gain or recording origin")
    vocal_channels, source_sr = sf.read(str(vocal_wav_path), dtype="float64", always_2d=True)
    if not np.isfinite(vocal_channels).all():
        raise ValueError("Nonfinite vocal samples")
    vocal = vocal_channels.mean(axis=1)
    if source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        vocal = resample_poly(vocal, sr // divisor, source_sr // divisor)
    guitar = synthesize_realistic_guitar(project, arrangement, sr=sr).astype(np.float64)
    offset = int(round(getattr(project, "beat_origin", 0.0) * sr))
    guitar = np.pad(guitar, ((offset, 0), (0, 0))) if offset >= 0 else guitar[min(-offset, len(guitar)) :]
    vocal_room = _room(vocal, 0.023, 0.039, 0.04, 0.03, sr)
    length = max(len(vocal_room), len(guitar))
    padded_guitar = np.pad(guitar, ((0, length - len(guitar)), (0, 0)))
    padded_vocal = np.pad(vocal, (0, length - len(vocal)))
    mix = vocal_gain * np.pad(vocal_room, ((0, length - len(vocal_room)), (0, 0))) + guitar_gain * padded_guitar
    peak = np.max(np.abs(mix)) if mix.size else 0.0
    if peak > 0:
        mix *= 10 ** (-1 / 20) / peak
    sf.write(str(out_wav_path), mix, sr, format="WAV", subtype="PCM_16")
    return evaluate_vocal_guitar_mix(padded_vocal, padded_guitar.mean(axis=1), sr=sr)


def evaluate_vocal_guitar_mix(
    vocal_audio: np.ndarray,
    guitar_audio: np.ndarray,
    sr: int = 44100,
) -> dict:
    """Compute objective audio-domain metrics verifying harmonic consonance and rhythmic lock between the real vocal and simulated guitar."""
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
    pcs = np.rint(69 + 12 * np.log2(frequencies[mask] / 440)).astype(np.int64) % 12
    starts = range(0, max(1, length), 1024)
    window = np.hanning(4096)

    def analyze(audio):
        chroma, rms = [], []
        for start in starts:
            frame = np.pad(audio[start : start + 4096], (0, max(0, start + 4096 - len(audio))))
            rms.append(np.sqrt(np.mean(frame ** 2)))
            power = np.abs(np.fft.rfft(frame * window)) ** 2
            vector = np.bincount(pcs, weights=power[mask], minlength=12)
            total = vector.sum()
            chroma.append(vector / total if total > 0 else vector)
        return np.asarray(chroma), np.asarray(rms)

    vc, vr = analyze(vocal)
    gc, gr = analyze(guitar)
    intervals = (np.arange(12)[:, None] - np.arange(12)[None, :]) % 12
    kernel = np.isin(intervals, [0, 3, 4, 5, 7, 8, 9]).astype(float)
    clash_kernel = np.isin(intervals, [1, 6, 11]).astype(float)
    consonance = np.einsum("bi,ij,bj->b", vc, kernel, gc)
    clashes = np.einsum("bi,ij,bj->b", vc, clash_kernel, gc)
    overlap = (vr > 0) & (gr > 0)
    weights = vr[overlap] * gr[overlap]
    active_count = int(overlap.sum())
    spectral = float(np.average(consonance[overlap], weights=weights)) if active_count else 0.0
    clash_ratio = float(np.mean(clashes[overlap] > consonance[overlap])) if active_count else 0.0
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
