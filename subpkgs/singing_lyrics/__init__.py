"""Subpackage 2: GPU Singing Voice Transcription (RMVPE + ROSVOT with word_bd) & Dual-Mode Lyric Forced Alignment (Qwen3-ASR / ForcedAligner + DP)."""
from .singing_transcriber import transcribe_vocal_notes, regulate_singing_notes_for_score
from .lyric_aligner import (
    tokenize_lyrics,
    transcribe_word_timestamps,
    align_lyrics_to_notes,
)

__all__ = [
    "transcribe_vocal_notes",
    "regulate_singing_notes_for_score",
    "tokenize_lyrics",
    "transcribe_word_timestamps",
    "align_lyrics_to_notes",
]
