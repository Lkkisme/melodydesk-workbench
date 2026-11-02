"""Stem separation engine with three tiers.

1. ``audio_separator``: Mel-Band-RoFormer-Karaoke (lead vocal only) or BS-RoFormer ViperX
   checkpoints, on CUDA when available.
2. ``demucs`` ``htdemucs`` two-stem separation.
3. ``_dsp_separate``: deterministic CPU spectral-spatial fallback (HPSS + formant-weighted
   soft-Wiener mid/side mask). It estimates harmonic, centre-panned, vocal-band energy; it is
   not a learned source separator and cannot isolate dry lead vocals or remove backing harmonies.
"""
from __future__ import annotations
from pathlib import Path
import logging
import tempfile
import numpy as np
import soundfile as sf
import subprocess
import sys


def resolve_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' when an NVIDIA GPU is available and requested, else 'cpu'."""
    if not prefer_cuda:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger(__name__)

_N_FFT = 2048
_HOP = 512
# Vocal-band spectral prior: smooth log-frequency band-pass (sung fundamental to sibilance
# region) with a near-unity plateau, gently shaped by log-Gaussian bumps on the typical
# sung-vowel formant regions F1/F2/F3 (in-band weight ranges over [1 - _FORMANT_DEPTH, 1]).
_VOCAL_BAND_LO_HZ = 80.0
_VOCAL_BAND_HI_HZ = 8000.0
_VOCAL_BAND_SLOPE_OCT = 0.25
_FORMANT_CENTRES_HZ = (500.0, 1500.0, 2800.0)
_FORMANT_WIDTH_OCT = (0.9, 0.8, 0.7)
_FORMANT_DEPTH = 0.08
_PRIOR_FLOOR = 0.1
# Temporal smoothing (frames) of the power spectral density estimates before forming the mask.
_PSD_SMOOTH_FRAMES = 3


def _read_audio(path):
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if not samples.size or sr <= 0 or not np.all(np.isfinite(samples)):
        raise ValueError(f"Invalid audio: {path}")
    return samples, sr


def _validate_pair(vocals_path, acc_path):
    vocals, vocal_sr = _read_audio(vocals_path)
    accompaniment, acc_sr = _read_audio(acc_path)
    if vocals.shape != accompaniment.shape or vocal_sr != acc_sr or vocal_sr <= 0:
        raise ValueError("Stem shape or sample rate mismatch between vocals and accompaniment")
    return vocals, accompaniment, vocal_sr


def _vocal_band_prior(freqs: np.ndarray) -> np.ndarray:
    """Smooth vocal-band spectral prior w(f) in [_PRIOR_FLOOR, 1] over STFT bin frequencies."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)
    log_f = np.log2(f)
    high_pass = 1.0 / (1.0 + np.exp(-(log_f - np.log2(_VOCAL_BAND_LO_HZ)) / _VOCAL_BAND_SLOPE_OCT))
    low_pass = 1.0 / (1.0 + np.exp((log_f - np.log2(_VOCAL_BAND_HI_HZ)) / _VOCAL_BAND_SLOPE_OCT))
    formants = np.zeros_like(f)
    for centre, width in zip(_FORMANT_CENTRES_HZ, _FORMANT_WIDTH_OCT):
        formants += np.exp(-0.5 * ((log_f - np.log2(centre)) / width) ** 2)
    peak = float(np.max(formants)) if formants.size else 0.0
    formant_shape = 1.0 - _FORMANT_DEPTH + _FORMANT_DEPTH * (formants / peak if peak > 0 else 1.0)
    band = high_pass * low_pass * formant_shape
    return _PRIOR_FLOOR + (1.0 - _PRIOR_FLOOR) * band


def _smooth_frames(power: np.ndarray, width: int = _PSD_SMOOTH_FRAMES) -> np.ndarray:
    """Moving-average of a (freq, frames) power array along time with edge replication."""
    if width <= 1 or power.shape[1] < 2:
        return power
    pad = width // 2
    padded = np.pad(power, ((0, 0), (pad, width - 1 - pad)), mode="edge")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate([np.zeros((power.shape[0], 1)), csum], axis=1)
    return ((csum[:, width:] - csum[:, :-width]) / width).astype(power.dtype, copy=False)


def _dsp_separate(samples, sr):
    """Deterministic CPU spectral-spatial fallback tier.

    For the mid channel M (and side channel S when stereo):
      1. HPSS splits M into harmonic H and percussive P components.
      2. A smooth vocal-band / formant spectral prior w(f) is evaluated on ``fft_frequencies``.
      3. Spatial posterior rho = |H|^2 / (|H|^2 + |S|^2) (1 for mono) gives the fraction of
         harmonic energy that is centre-panned.
      4. Vocal PSD  V = w(f) * rho * |H|^2  (centre-panned, vocal-band harmonic energy),
         accompaniment PSD A = |P|^2 + (1 - w(f) * rho) * |H|^2 (percussive energy plus the
         off-centre / out-of-band harmonic remainder); both are smoothed over a few frames.
      5. Soft-Wiener mask G = V / (V + A) is applied to the mid mixture STFT, then ``istft``.
    Accompaniment is the residual ``samples - vocals`` so the stems sum to the input.
    """
    import librosa

    mid = samples.mean(axis=1)
    s_mid = librosa.stft(mid, n_fft=_N_FFT, hop_length=_HOP)
    harmonic, percussive = librosa.decompose.hpss(s_mid)
    harmonic_power = np.abs(harmonic) ** 2
    percussive_power = np.abs(percussive) ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    prior = _vocal_band_prior(freqs)[:, None]

    spatial = np.ones_like(harmonic_power)
    if samples.shape[1] == 2:
        side = (samples[:, 0] - samples[:, 1]) / 2.0
        s_side = librosa.stft(side, n_fft=_N_FFT, hop_length=_HOP)
        side_power = np.abs(s_side) ** 2
        denominator = harmonic_power + side_power
        spatial = np.divide(harmonic_power, denominator, out=np.zeros_like(harmonic_power), where=denominator > 0)

    vocal_weight = prior * spatial
    vocal_psd = _smooth_frames(vocal_weight * harmonic_power)
    acc_psd = _smooth_frames(percussive_power + (1.0 - vocal_weight) * harmonic_power)
    total = vocal_psd + acc_psd
    mask = np.divide(vocal_psd, total, out=np.zeros_like(vocal_psd), where=total > 1e-20)
    vocal_spec = s_mid * mask

    vocal_mono = librosa.istft(vocal_spec, hop_length=_HOP, length=len(mid))
    vocals = np.repeat(vocal_mono[:, None], samples.shape[1], axis=1).astype(samples.dtype, copy=False)
    return vocals, samples - vocals


def separate_stems(
    audio_path: str | Path,
    out_dir: str | Path,
    prefer_cuda: bool = True,
    strip_backing_harmonies: bool = True,
) -> dict:
    """Separate vocals and accompaniment using RoFormer, Demucs, then DSP.

    The DSP tier estimates harmonic/center audio; it cannot promise isolated dry
    lead vocals or backing-harmony removal.
    """
    audio_path, out_dir = Path(audio_path).resolve(), Path(out_dir).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    samples, input_sr = _read_audio(audio_path)
    out_sr = input_sr
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals_path, acc_path = out_dir / "vocals.wav", out_dir / "no_vocals.wav"
    if audio_path in (vocals_path, acc_path):
        raise ValueError("Input cannot be a canonical output path")
    device = resolve_device(prefer_cuda)
    pair = None
    backend, actual_device = "librosa", "cpu"
    with tempfile.TemporaryDirectory(dir=out_dir) as temporary:
        tier_dir = Path(temporary)
        try:
            import torch
            from audio_separator.separator import Separator

            roformer_cache = Path(__file__).resolve().parents[4] / "work" / "model-cache" / "roformer"
            if getattr(Separator, "__module__", "").startswith("audio_separator"):
                roformer_cache.mkdir(parents=True, exist_ok=True)
                separator = Separator(output_dir=str(tier_dir), output_format="WAV", model_file_dir=str(roformer_cache))
            else:
                separator = Separator(output_dir=str(tier_dir), output_format="WAV")
            separator.torch_device = torch.device(device)
            model_filename = (
                "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
                if strip_backing_harmonies else "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            )
            separator.load_model(model_filename=model_filename)
            files = separator.separate(str(audio_path), custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"})
            returned = {Path(f).resolve() if Path(f).is_absolute() else (tier_dir / f).resolve() for f in files}
            candidates = (tier_dir / "vocals.wav", tier_dir / "no_vocals.wav")
            if not all(p.resolve() in returned for p in candidates):
                raise ValueError("Separator did not return the requested stem pair")
            pair_v, pair_a, tier_sr = _validate_pair(*candidates)
            if (
                strip_backing_harmonies
                and getattr(Separator, "__module__", "").startswith("audio_separator")
                and float(np.sqrt(np.mean(samples ** 2))) > 0.01
                and float(np.sqrt(np.mean(pair_v ** 2))) < 0.008
            ):
                logger.info(
                    "Karaoke lead-vocal separator muted multi-voice choir (rms=%.6f); retrying with BS-RoFormer ViperX SDR 12.9755",
                    float(np.sqrt(np.mean(pair_v ** 2))),
                )
                for p in candidates:
                    p.unlink(missing_ok=True)
                separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
                files = separator.separate(str(audio_path), custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"})
                returned = {Path(f).resolve() if Path(f).is_absolute() else (tier_dir / f).resolve() for f in files}
                if not all(p.resolve() in returned for p in candidates):
                    raise ValueError("BS-RoFormer fallback did not return the requested stem pair")
                pair_v, pair_a, tier_sr = _validate_pair(*candidates)
            pair = (pair_v, pair_a)
            out_sr = tier_sr
            actual_device = "cuda" if separator.torch_device.type == "cuda" else "cpu"
            backend = "audio_separator"
        except Exception as exc:
            logger.warning("audio_separator tier failed: %s", exc)
    if pair is None:
        with tempfile.TemporaryDirectory(dir=out_dir) as temporary:
            tier_dir = Path(temporary)
            try:
                command = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "--two-stems=vocals", "-d", device, "-o", str(tier_dir), str(audio_path)]
                subprocess.run(command, check=True, capture_output=True, text=True, errors="replace", timeout=300)
                model_dir = tier_dir / "htdemucs" / audio_path.stem
                pair_v, pair_a, tier_sr = _validate_pair(model_dir / "vocals.wav", model_dir / "no_vocals.wav")
                pair = (pair_v, pair_a)
                out_sr = tier_sr
                backend, actual_device = "demucs", device
            except Exception as exc:
                logger.warning("Demucs tier failed: %s; stderr: %s", exc, getattr(exc, "stderr", ""))
    if pair is None:
        pair = _dsp_separate(samples, input_sr)
        out_sr = input_sr
    sf.write(str(vocals_path), np.clip(pair[0], -1.0, 1.0), out_sr, subtype="PCM_16")
    sf.write(str(acc_path), np.clip(pair[1], -1.0, 1.0), out_sr, subtype="PCM_16")
    return {"vocals": str(vocals_path), "accompaniment": str(acc_path), "backend": backend, "device": actual_device}
