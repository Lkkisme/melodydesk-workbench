"""Rebuild all 7 full-length real-vocal songs using the complete 2026 SOTA Leaderboard stack:
1. MVSEP #1 `Mel-Band-RoFormer-Karaoke` (`mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`) GPU Stem Separation -> ultra-clean lead vocal stems (`stems_<key>_roformer/vocals.wav`).
2. ISMIR #1 `CPJKU/Beat This!` (`beat_this-final0.ckpt`) with `bpm_range` and `meter_hint` -> true piecewise `beat_times` & `bar_start_seconds`.
3. Forced-Projected Syllable Timestamps (`transcribe_word_timestamps` with `vad_filter=True` + Needleman-Wunsch Ground-Truth Projection) -> 0 zero-duration words.
4. ACL/ISMIR `RMVPE + ROSVOT` Singing Transcription + Piecewise Beat Regulation (`regulate_singing_notes_for_score(..., beat_times=beat_times)`).
5. 100% Vocal-Melody-Conditioned Functional Harmony (`synthesize_melody_conditioned_harmony(..., bar_start_seconds=bar_start_seconds)`) + Acoustic-Interval `align_lyrics_to_notes`.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
INSTALL_ROOT = ROOT.parents[1] if len(ROOT.parents) >= 2 else ROOT.parent
MODEL_CACHE_DIR = Path(os.environ.get("MELODYDESK_MODEL_CACHE", str(INSTALL_ROOT / "work" / "model-cache")))
os.environ["TORCH_HOME"] = str(MODEL_CACHE_DIR)
os.environ["MELODYDESK_MODEL_CACHE"] = str(MODEL_CACHE_DIR)
WORK_DIR = Path(os.environ.get("MELODYDESK_REPERTOIRE_DIR", str(INSTALL_ROOT / "work" / "repertoire_audio")))
sys.path.insert(0, str(ROOT))

from music import Project
from subpkgs.singing_lyrics import (
    align_lyrics_to_notes,
    regulate_singing_notes_for_score,
    transcribe_vocal_notes,
    transcribe_word_timestamps,
)
from subpkgs.stem_mir import (
    separate_stems,
    synthesize_melody_conditioned_harmony,
    track_beats_and_downbeats,
)

FULL_LYRICS: dict[str, str] = {
    "mandarin_pop": (
        "让我掉下眼泪的 不止昨夜的酒 让我依依不舍的 不止你的温柔 "
        "余路还要走多久 你攥着我的手 让我感到为难的 是挣扎的自由 "
        "分别总是在九月 回忆是思念的愁 深秋嫩绿的垂柳 亲吻着我额头 "
        "在那座阴雨的小城里 我从未忘记你 成都 带不走的 只有你 "
        "和我在成都的街头走一走 喔哦 直到所有的灯都熄灭了也不停留 "
        "你会挽着我的衣袖 我会把手揣进裤兜 走到玉林路的尽头 坐在小酒馆的门口 "
        "分别总是在九月 回忆是思念的愁 深秋嫩绿的垂柳 亲吻着我额头 "
        "在那座阴雨的小城里 我从未忘记你 成都 带不走的 只有你 "
        "和我在成都的街头走一走 喔哦 直到所有的灯都熄灭了也不停留 "
        "你会挽着我的衣袖 我会把手揣进裤兜 走到玉林路的尽头 坐在小酒馆的门口 "
        "和我在成都的街头走一走 喔哦 直到所有的灯都熄灭了也不停留 "
        "和我在成都的街头走一走 喔哦 直到所有的灯都熄灭了也不停留 "
        "你会挽着我的衣袖 我会把手揣进裤兜 走到玉林路的尽头 走过玉林路的尽头 坐在小酒馆的门口"
    ),
    "canon_in_d": (
        "Almost heaven West Virginia Blue Ridge Mountains Shenandoah River "
        "Life is old there older than the trees younger than the mountains growing like a breeze "
        "Country roads take me home to the place I belong West Virginia mountain mama take me home country roads "
        "All my memories gather round her miner's lady stranger to blue water "
        "Dark and dusty painted on the sky misty taste of moonshine teardrop in my eye "
        "Country roads take me home to the place I belong West Virginia mountain mama take me home country roads "
        "I hear her voice in the morning hour she calls me the radio reminds me of my home far away "
        "Driving down the road I get a feeling that I should have been home yesterday yesterday "
        "Country roads take me home to the place I belong West Virginia mountain mama take me home country roads "
        "Country roads take me home to the place I belong West Virginia mountain mama take me home country roads "
        "Take me home down country roads take me home down country roads"
    ),
    "romance_amor": (
        "Would you know my name if I saw you in heaven "
        "Would it be the same if I saw you in heaven "
        "I must be strong and carry on cause I know I don't belong here in heaven "
        "Would you hold my hand if I saw you in heaven "
        "Would you help me stand if I saw you in heaven "
        "I'll find my way through night and day cause I know I just can't stay here in heaven "
        "Time can bring you down time can bend your knees "
        "Time can break your heart have you begging please begging please "
        "Beyond the door there's peace I'm sure and I know there'll be no more tears in heaven "
        "Would you know my name if I saw you in heaven "
        "Would it be the same if I saw you in heaven "
        "I must be strong and carry on cause I know I don't belong here in heaven"
    ),
    "scarborough_fair": (
        "Are you going to Scarborough Fair Parsley sage rosemary and thyme "
        "Remember me to one who lives there She once was a true love of mine "
        "Tell her to make me a cambric shirt Parsley sage rosemary and thyme "
        "Without no seams nor needle work Then she'll be a true love of mine "
        "Tell her to find me an acre of land Parsley sage rosemary and thyme "
        "Between the salt water and the sea strand Then she'll be a true love of mine "
        "Tell her to reap it with a sickle of leather Parsley sage rosemary and thyme "
        "And gather it all in a bunch of heather Then she'll be a true love of mine "
        "Are you going to Scarborough Fair Parsley sage rosemary and thyme "
        "Remember me to one who lives there She once was a true love of mine"
    ),
    "greensleeves": (
        "Alas my love you do me wrong to cast me off discourteously "
        "For I have loved you well and long delighting in your company "
        "Greensleeves was all my joy Greensleeves was my delight "
        "Greensleeves was my heart of gold and who but my lady Greensleeves "
        "I have been ready at your hand to grant whatever you would crave "
        "I have both waged life and land your love and good will for to have "
        "Greensleeves was all my joy Greensleeves was my delight "
        "Greensleeves was my heart of gold and who but my lady Greensleeves "
        "Thy smock of silk both fair and white with gold embroidered gorgeously "
        "Thy petticoat of sendal right and these I bought all gladly "
        "Greensleeves was all my joy Greensleeves was my delight "
        "Greensleeves was my heart of gold and who but my lady Greensleeves"
    ),
    "real_vocal": (
        "Woke up early in the morning sun shining through the kitchen door "
        "Grabbed my tackle box and willow pole heading down to the river shore "
        "Let's go fishin down by the creek where the cool clear waters flow "
        "Leave all your worries on the dusty bank and watch that bobber go "
        "Honeysuckle breeze in the sycamore trees crickets singing in the shade "
        "Best kind of memories that you ever keep are the ones that the river made "
        "Let's go fishin down by the creek where the cool clear waters flow "
        "Leave all your worries on the dusty bank and watch that bobber go"
    ),
    "regular_friends": (
        "等待 我随时随地在等待 做你感情上的依赖 我没有任何的疑问 这是爱 "
        "我猜 你早就想要说明白 我觉得自己好失败 从天堂掉落到深渊 多无奈 "
        "我愿意改变 what can I do 重新再来一遍 just give me a chance "
        "我无法只是普通朋友 感情已那么深 叫我怎么能放手 "
        "但你说 I only wanna be your friend 做个朋友 "
        "我在你心中只是 just a friend 不是情人 "
        "我感激你对我这样的坦白 但我给你的爱暂时收不回来 "
        "So I 我不能只是 be your friend I just can't be your friend "
        "我猜 你早就想要说明白 我觉得自己好失败 从天堂掉落到深渊 多无奈 "
        "我愿意改变 what can I do 重新再来一遍 just give me a chance "
        "我无法只是普通朋友 感情已那么深 叫我怎么能放手 "
        "但你说 I only wanna be your friend 做个朋友 "
        "我在你心中只是 just a friend 不是情人 "
        "我感激你对我这样的坦白 但我给你的爱暂时收不回来 "
        "So I 我不能只是 be your friend I just can't be your friend "
        "no no no 我不能只是做你的朋友 不能做普通朋友"
    ),
}

SONG_SPECS: dict[str, dict] = {
    "canon_in_d": {
        "tonic": "A",
        "minor": False,
        "meter": "4/4",
        "capo": 2,
        "preset": "emotional_pop",
        "style": "指弹律动",
        "bpm_range": (78.0, 92.0),
    },
    "romance_amor": {
        "tonic": "A",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "抒情琶音",
        "bpm_range": (74.0, 86.0),
    },
    "scarborough_fair": {
        "tonic": "G",
        "minor": True,
        "meter": "3/4",
        "capo": 2,
        "preset": "emotional_pop",
        "style": "分解和弦",
        "bpm_range": (80.0, 108.0),
    },
    "greensleeves": {
        "tonic": "F",
        "minor": True,
        "meter": "3/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "分解和弦",
        "bpm_range": (82.0, 106.0),
    },
    "mandarin_pop": {
        "tonic": "D",
        "minor": False,
        "meter": "4/4",
        "capo": 2,
        "preset": "emotional_pop",
        "style": "抒情琶音",
        "bpm_range": (72.0, 96.0),
    },
    "real_vocal": {
        "tonic": "G",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "指弹律动",
        "bpm_range": (84.0, 118.0),
    },
    "regular_friends": {
        "tonic": "F",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "neosoul_jazz",
        "style": "Neo-Soul 律动",
        "bpm_range": (82.0, 96.0),
        "beat_source": "mix",
    },
}

MANIFEST_PATH = WORK_DIR / "repertoire_manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    t_all = time.time()

    for key, spec in SONG_SPECS.items():
        t0 = time.time()
        entry = manifest[key]
        src_wav = Path(entry["source"])
        info = sf.info(str(src_wav))
        clip_duration = round(float(info.duration), 3)
        meter = spec["meter"]
        raw_lyrics = FULL_LYRICS[key]

        # 1. MVSEP #1 Mel-Band-RoFormer-Karaoke / BS-RoFormer ViperX GPU Lead-Vocal Isolation
        roformer_dir = WORK_DIR / f"stems_{key}_roformer"
        voc_wav = roformer_dir / "vocals.wav"
        cached_ok = False
        if voc_wav.is_file() and voc_wav.stat().st_size > 100_000:
            v_arr, _ = sf.read(str(voc_wav), dtype="float32")
            if float(np.sqrt(np.mean(v_arr ** 2))) > 0.008:
                cached_ok = True
        if cached_ok:
            print(f"\n[{key}] 1. Reusing cached RoFormer lead vocal in {roformer_dir.name}", flush=True)
        else:
            print(f"\n[{key}] 1. Running MVSEP #1 RoFormer GPU lead-vocal isolation on {src_wav.name}...", flush=True)
            sep_res = separate_stems(src_wav, roformer_dir, prefer_cuda=True, strip_backing_harmonies=True)
            print(f"  -> Separation backend: {sep_res['backend']} ({sep_res['device']})", flush=True)
            voc_wav = Path(sep_res["vocals"])

        # 2. Forced-Projected Syllable Timestamps on PURE vocals.wav
        if entry.get("word_timestamps") and len(entry["word_timestamps"]) >= 50:
            word_timestamps = entry["word_timestamps"]
            print(
                f"[{key}] 2. Reusing cached word timestamps on {voc_wav.name} "
                f"(tokens={len(word_timestamps)}, {word_timestamps[0]['start']:.2f}s..{word_timestamps[-1]['end']:.2f}s)",
                flush=True,
            )
        else:
            print(f"[{key}] 2. Extracting forced-projected word timestamps on {voc_wav.name}...", flush=True)
            wt_res = transcribe_word_timestamps(voc_wav, prefer_cuda=True, ground_truth_text=raw_lyrics)
            word_timestamps = wt_res["word_timestamps"]
            print(
                f"  -> Timestamp backend: {wt_res['backend']} | tokens={len(word_timestamps)} | "
                f"first={word_timestamps[0]['start']:.2f}s | last={word_timestamps[-1]['end']:.2f}s",
                flush=True,
            )
        all_vocal_onsets = sorted({round(float(w["start"]), 3) for w in word_timestamps})

        # 3. ISMIR #1 Beat This! + Syllable Onset Phase-Locking
        beat_wav = src_wav if spec.get("beat_source") == "mix" else voc_wav
        print(f"[{key}] 3. Tracking rhythm & beats on {beat_wav.name} ({spec['bpm_range']}, {meter})...", flush=True)
        beat_info = track_beats_and_downbeats(
            beat_wav,
            prefer_cuda=True,
            bpm_range=spec["bpm_range"],
            meter_hint=meter,
            vocal_onsets_sec=all_vocal_onsets,
        )
        bpm = round(float(beat_info["bpm"]), 3)
        beat_times = [round(float(x), 4) for x in beat_info["beat_times"]]
        bar_start_seconds = [round(float(x), 4) for x in beat_info["bar_start_seconds"]]
        if spec.get("beat_source") == "mix" and all_vocal_onsets:
            while len(bar_start_seconds) > 2 and bar_start_seconds[1] < all_vocal_onsets[0] - 1.5:
                bar_start_seconds.pop(0)
            beat_times = [t for t in beat_times if t >= bar_start_seconds[0] - 1e-4]
        beat_origin = round(float(bar_start_seconds[0]), 4)
        word_boundaries_sec = [t for t in all_vocal_onsets if t >= beat_origin]
        print(
            f"  -> Vocal Beat backend: {beat_info['backend']} | vocal_bpm={bpm} | origin={beat_origin}s | "
            f"beats={len(beat_times)} | bars={len(bar_start_seconds)-1}",
            flush=True,
        )

        # 4. RMVPE + ROSVOT Singing Transcription + Piecewise Vocal-Beat Regulation
        print(f"[{key}] 4. Transcribing & regulating vocal notes on piecewise vocal beat_times grid...", flush=True)
        tx = transcribe_vocal_notes(
            voc_wav,
            bpm=bpm,
            beat_origin=beat_origin,
            word_boundaries_sec=word_boundaries_sec,
            prefer_cuda=True,
            beat_times=beat_times,
        )
        raw_notes = tx["notes"]
        if spec.get("beat_source") == "mix" and all_vocal_onsets:
            raw_notes = [n for n in raw_notes if n.raw_start >= all_vocal_onsets[0] - 0.15]
        notes = regulate_singing_notes_for_score(
            raw_notes,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            word_boundaries_sec=word_boundaries_sec,
            beat_times=beat_times,
        )
        print(f"  -> Transcribed {len(notes)} pure human vocal notes (backend={tx['backend']})", flush=True)

        # 5. 100% Vocal-Melody-Driven Harmony & Vocal-Envelope Dynamics
        tonic = spec["tonic"]
        harm = synthesize_melody_conditioned_harmony(
            notes=notes,
            bpm=bpm,
            beat_origin=beat_origin,
            meter=meter,
            tonic=tonic,
            minor=spec["minor"],
            clip_duration=clip_duration,
            bar_start_seconds=bar_start_seconds,
            beat_times=beat_times,
        )
        base_chords = harm["chords"]
        bar_dynamics = harm["bar_dynamics"]
        print(f"  -> Synthesized {len(base_chords)} bars of 100% vocal-driven harmony: first 8 = {base_chords[:8]}", flush=True)

        # 6. Acoustic-Interval Syllable-to-Note Forced Alignment
        project = Project(
            title=entry["title"],
            tonic=tonic,
            meter=meter,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            source=str(src_wav),
            audio={"vocals": str(voc_wav)},
            notes=notes,
        )
        aligned = align_lyrics_to_notes(
            project,
            raw_lyrics=raw_lyrics,
            word_timestamps=word_timestamps,
        )
        note_lyrics = aligned["note_lyrics"]
        non_empty_lyrics = sum(1 for lyr in note_lyrics if lyr and lyr != "-")
        melisma_count = sum(1 for lyr in note_lyrics if lyr == "-")
        print(
            f"  -> Aligned {non_empty_lyrics} lyrics + {melisma_count} melismas across {len(notes)} notes "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

        entry.update(
            tonic=tonic,
            meter=meter,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            vocals_wav=str(voc_wav),
            capo=spec["capo"],
            style=spec["style"],
            preset=spec["preset"],
            base_chords=base_chords,
            bar_start_seconds=bar_start_seconds,
            beat_times=beat_times,
            bar_dynamics=bar_dynamics,
            word_timestamps=word_timestamps,
            raw_lyrics=raw_lyrics,
            note_lyrics=note_lyrics,
            notes=[asdict(n) for n in notes],
        )
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== ALL 7 SONGS REBUILT WITH 2026 SOTA STACK IN {time.time() - t_all:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
