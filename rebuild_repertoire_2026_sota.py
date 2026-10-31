"""Rebuild all 6 full-length vocal songs in repertoire_manifest.json using the verified 2026 SOTA 100% Vocal-Driven Pipeline:
1. BS-RoFormer isolated pure vocal stem (`vocals.wav`)
2. Meta `MMS_FA` 300M CTC Forced Alignment (chunked 30s + 1s context pad + `pypinyin` for CJK) -> `mms-fa-cuda`
3. CPJKU `Beat This!` vocal beat & downbeat tracking with bar-boundary downbeat phase re-locking & multi-beat IOI folding
4. ROSVOT + 2-Stage RMVPE vocal note transcription regulated by CTC `word_boundaries_sec` & continuous-beat `score_cleanup.organize`
5. 100% Pure Vocal-Melody-Conditioned Functional Harmony (`synthesize_melody_conditioned_harmony`)
6. Acoustic-time (`beats_to_seconds_piecewise`) DP syllable-to-note lyric alignment (`align_lyrics_to_notes`)
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
import soundfile as sf

from subpkgs.singing_lyrics import (
    align_lyrics_to_notes,
    regulate_singing_notes_for_score,
    transcribe_vocal_notes,
    transcribe_word_timestamps,
)
from subpkgs.stem_mir import (
    synthesize_melody_conditioned_harmony,
    track_beats_and_downbeats,
)

MANIFEST_PATH = Path(r"E:\MelodyDesk\work\repertoire_audio\repertoire_manifest.json")

SONG_CONFIGS = {
    "scarborough_fair": {"tonic": "G", "minor": True, "meter": "3/4", "bpm_range": (60.0, 140.0)},
    "greensleeves": {"tonic": "F", "minor": True, "meter": "3/4", "bpm_range": (68.0, 120.0)},
    "romance_amor": {"tonic": "A", "minor": False, "meter": "4/4", "bpm_range": (68.0, 110.0)},
    "canon_in_d": {"tonic": "A", "minor": False, "meter": "4/4", "bpm_range": (70.0, 120.0)},
    "mandarin_pop": {"tonic": "D", "minor": False, "meter": "4/4", "bpm_range": (60.0, 136.0)},
    "real_vocal": {"tonic": "G", "minor": False, "meter": "4/4", "bpm_range": (72.0, 118.0)},
}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key, cfg in SONG_CONFIGS.items():
        t0 = time.time()
        entry = manifest[key]
        vocals_wav = Path(entry["vocals_wav"])
        source_wav = Path(entry["source"]) if entry.get("source") and Path(entry["source"]).is_file() else vocals_wav
        info = sf.info(str(vocals_wav))
        clip_duration = round(float(info.duration), 3)
        raw_lyrics = entry["raw_lyrics"]

        # 1. Frame-level CTC Forced Alignment (Meta MMS_FA + pypinyin)
        aligned_words = transcribe_word_timestamps(
            vocals_wav, prefer_cuda=True, ground_truth_text=raw_lyrics
        )
        align_backend = aligned_words.get("backend", "unknown")
        word_ts = aligned_words.get("word_timestamps") or []
        vocal_onsets = sorted({float(w["start"]) for w in word_ts if "start" in w and float(w["start"]) >= 0.0})
        word_boundaries = sorted(
            set(vocal_onsets) | {float(w["end"]) for w in word_ts if "end" in w and float(w["end"]) >= 0.0}
        )

        # Extract acoustic vocal note onsets first so 1-bar pre-vocal origin never misses soft choir openings
        pre_tx = transcribe_vocal_notes(
            vocals_wav,
            bpm=100.0,
            beat_origin=0.0,
            word_boundaries_sec=word_boundaries or None,
            prefer_cuda=True,
        )
        note_onsets = sorted({float(n.raw_start) for n in pre_tx["notes"] if n.raw_duration >= 0.18 and n.raw_start >= 0.5})
        combined_onsets = sorted(set(vocal_onsets) | set(note_onsets))

        # 2. Dual-Mode Neural Downbeat Bar-Grid Locking + 1-Bar Pre-Vocal Pickup Origin
        beat_info = track_beats_and_downbeats(
            source_wav,
            prefer_cuda=True,
            bpm_range=cfg["bpm_range"],
            meter_hint=cfg["meter"],
            vocal_onsets_sec=combined_onsets or None,
        )
        bpm = round(float(beat_info["bpm"]), 2)
        beat_origin = round(float(beat_info["beat_origin"]), 4)
        meter = str(beat_info["meter"])
        beat_times = [round(float(t), 4) for t in beat_info["beat_times"]]
        bar_start_seconds = [round(float(t), 4) for t in beat_info["bar_start_seconds"]]

        # 3. Singing Voice Transcription (ROSVOT + RMVPE) + Word-Boundary Regulation
        tx = transcribe_vocal_notes(
            vocals_wav,
            bpm=bpm,
            beat_origin=beat_origin,
            word_boundaries_sec=word_boundaries or None,
            prefer_cuda=True,
            beat_times=beat_times,
        )
        merged_wbs = sorted(set(word_boundaries) | set(tx.get("word_boundaries_sec") or []))
        notes = regulate_singing_notes_for_score(
            tx["notes"],
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            word_boundaries_sec=merged_wbs or None,
            beat_times=beat_times,
        )

        # 4. 100% Pure Vocal-Melody-Conditioned Functional Harmony
        harmony = synthesize_melody_conditioned_harmony(
            notes,
            bpm=bpm,
            beat_origin=beat_origin,
            meter=meter,
            tonic=cfg["tonic"],
            minor=cfg["minor"],
            clip_duration=clip_duration,
            bar_start_seconds=bar_start_seconds,
        )
        base_chords = list(harmony["chords"])
        bar_dynamics = [round(float(d), 4) for d in harmony["bar_dynamics"]]

        # 5. Syllable-to-Note Lyric Alignment on Piecewise Acoustic Timing
        from music import Project
        proj = Project(
            title=entry["title"],
            tonic=cfg["tonic"],
            meter=meter,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            source=entry["source"],
            audio={"vocals": str(vocals_wav)},
            notes=notes,
            reference_notes=list(notes),
        )
        lyric_res = align_lyrics_to_notes(
            proj,
            raw_lyrics=raw_lyrics,
            word_timestamps=word_ts,
        )
        note_lyrics = list(lyric_res["note_lyrics"])

        # Remove any legacy backing_wav key
        entry.pop("backing_wav", None)
        entry.update(
            tonic=cfg["tonic"],
            preset="original",
            meter=meter,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            notes=[asdict(n) for n in notes],
            base_chords=base_chords,
            note_lyrics=note_lyrics,
            word_timestamps=word_ts,
            beat_times=beat_times,
            bar_start_seconds=bar_start_seconds,
            bar_dynamics=bar_dynamics,
        )
        non_empty_lyrics = sum(1 for lyr in note_lyrics if lyr and lyr != "-")
        print(
            f"[{key}] align={align_backend} | bpm={bpm} | origin={beat_origin}s | "
            f"notes={len(notes)} (lyric_filled={non_empty_lyrics}) | bars={len(base_chords)} | "
            f"chords[:8]={base_chords[:8]} | {time.time() - t0:.1f}s"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Successfully updated repertoire_manifest.json")


if __name__ == "__main__":
    main()
