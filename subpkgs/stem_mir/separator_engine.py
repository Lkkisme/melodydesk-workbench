"""GPU Stem Separation Engine (MVSEP SOTA BS-RoFormer + Mel-Band-RoFormer-Karaoke with Demucs CUDA fallback)."""
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


def _dsp_separate(samples, sr):
    import librosa

    mid = samples.mean(axis=1)
    s_mid = librosa.stft(mid, n_fft=2048, hop_length=512)
    harmonic, _ = librosa.decompose.hpss(s_mid)
    vocal_spec = harmonic
    if samples.shape[1] == 2:
        side = (samples[:, 0] - samples[:, 1]) / 2.0
        s_side = librosa.stft(side, n_fft=2048, hop_length=512)
        center_power = np.abs(harmonic) ** 2
        side_power = np.abs(s_side) ** 2
        denominator = center_power + side_power
        mask = np.divide(center_power, denominator, out=np.zeros_like(center_power), where=denominator > 0)
        vocal_spec = harmonic * mask
    vocal_mono = librosa.istft(vocal_spec, hop_length=512, length=len(mid))
    vocals = np.repeat(vocal_mono[:, None], samples.shape[1], axis=1)
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
    samples, sr = _read_audio(audio_path)
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
            pair_v, pair_a, sr = _validate_pair(*candidates)
            pair = (pair_v, pair_a)
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
                pair_v, pair_a, sr = _validate_pair(model_dir / "vocals.wav", model_dir / "no_vocals.wav")
                pair = (pair_v, pair_a)
                backend, actual_device = "demucs", device
            except Exception as exc:
                logger.warning("Demucs tier failed: %s; stderr: %s", exc, getattr(exc, "stderr", ""))
    if pair is None:
        pair = _dsp_separate(samples, sr)
    sf.write(str(vocals_path), np.clip(pair[0], -1.0, 1.0), sr, subtype="PCM_16")
    sf.write(str(acc_path), np.clip(pair[1], -1.0, 1.0), sr, subtype="PCM_16")
    return {"vocals": str(vocals_path), "accompaniment": str(acc_path), "backend": backend, "device": actual_device}
