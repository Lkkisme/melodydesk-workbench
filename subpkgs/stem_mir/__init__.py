"""Subpackage 1: Stem Separation (BS-RoFormer/Karaoke/Demucs CUDA), Beat Tracking (beat_this), and Accompaniment Chord & Intro Melody Recognition."""
from .separator_engine import separate_stems, resolve_device
from .beat_tracker import track_beats_and_downbeats, seconds_to_beats_piecewise, beats_to_seconds_piecewise
from .chord_recognizer import recognize_accompaniment_chords, extract_instrumental_melody, synthesize_melody_conditioned_harmony

__all__ = [
    "separate_stems",
    "resolve_device",
    "track_beats_and_downbeats",
    "seconds_to_beats_piecewise",
    "beats_to_seconds_piecewise",
    "recognize_accompaniment_chords",
    "extract_instrumental_melody",
    "synthesize_melody_conditioned_harmony",
]
