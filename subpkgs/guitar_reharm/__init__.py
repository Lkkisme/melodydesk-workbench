"""Subpackage 3: High-Level Guitar Re-harmonization, CAGED Voicing & Voice-Leading DP, Advanced Grooves (Split-Bar, X/PM/HP, Chord-Melody), and Realistic Acoustic Guitar Sound Simulator + Real-Vocal Mixer."""
from .chord_voicing import (
    EXTENDED_INTERVALS,
    parse_extended_chord,
    generate_voicings,
    optimize_voice_leading,
)
from .reharmonizer import (
    REHARMONIZE_PRESETS,
    roman_numeral,
    check_melody_clash,
    reharmonize_progression,
    suggest_bar_substitutions,
)
from .arrangement_grooves import (
    ADVANCED_STYLES,
    arrange_advanced,
)
from .guitar_simulator import (
    synthesize_realistic_guitar,
    mix_real_vocal_and_guitar,
    evaluate_vocal_guitar_mix,
)

__all__ = [
    "EXTENDED_INTERVALS",
    "parse_extended_chord",
    "generate_voicings",
    "optimize_voice_leading",
    "REHARMONIZE_PRESETS",
    "roman_numeral",
    "check_melody_clash",
    "reharmonize_progression",
    "suggest_bar_substitutions",
    "ADVANCED_STYLES",
    "arrange_advanced",
    "synthesize_realistic_guitar",
    "mix_real_vocal_and_guitar",
    "evaluate_vocal_guitar_mix",
]
