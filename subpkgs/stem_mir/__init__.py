"""Subpackage 1: Pure Lead-Vocal Isolation (Mel-Band-RoFormer Karaoke / BS-RoFormer ViperX), Vocal Beat & Rubato Tracking (Beat This! + Vocal Onset PLL), and Pure Vocal-Melody-Conditioned Functional Harmony Generation."""
from .separator_engine import separate_stems, resolve_device
from .beat_tracker import track_beats_and_downbeats, seconds_to_beats_piecewise, beats_to_seconds_piecewise
from .chord_recognizer import synthesize_melody_conditioned_harmony

__all__ = [
    "separate_stems",
    "resolve_device",
    "track_beats_and_downbeats",
    "seconds_to_beats_piecewise",
    "beats_to_seconds_piecewise",
    "synthesize_melody_conditioned_harmony",
]
