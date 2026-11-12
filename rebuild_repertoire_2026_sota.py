"""Rebuild all 7 full-length real-vocal songs using the Plan A Hybrid 2026 SOTA Leaderboard stack:
1. MVSEP #1 `Mel-Band-RoFormer-Karaoke` (`mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`) GPU Stem Separation
   -> isolated `vocals.wav` + `no_vocals.wav` (reusing local caches without re-downloading).
2. ISMIR #1 `CPJKU/Beat This!` (`beat_this-final0.ckpt`) on the full acoustic mix (`src_wav`) when accompaniment is present
   (falling back to `vocals.wav` for a-cappella) with `bpm_range`, compound `6/8` subdivision preservation, and syllable phase-locking.
3. Forced-Projected Syllable Timestamps (`transcribe_word_timestamps` with `vad_filter=True` + Needleman-Wunsch projection).
4. ACL/ISMIR `RMVPE + ROSVOT` Singing Transcription + Piecewise Beat Regulation (`regulate_singing_notes_for_score`).
5. Hybrid Accompaniment-First Harmony (`recognize_accompaniment_chords` on `no_vocals.wav` with harmonic-percussive separation,
   dual-band Bass/Body 36-bin CQT, concert-key lock, and Viterbi decoding; automatic fallback to `synthesize_melody_conditioned_harmony`
   on unaccompanied vocals) + Acoustic-Interval `align_lyrics_to_notes`.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
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

from music import NAMES, Project
from subpkgs.guitar_reharm import recommend_artistry
from subpkgs.singing_lyrics import (
    align_lyrics_to_notes,
    detect_pre_lyric_vocalise_onsets,
    estimate_key_albrecht_shanahan,
    filter_pre_lyric_bleed_notes,
    regulate_singing_notes_for_score,
    transcribe_vocal_notes,
    transcribe_word_timestamps,
)
from subpkgs.stem_mir import (
    recognize_accompaniment_chords,
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
        "For I have courted you so long delighting in your company "
        "Greensleeves was all my joy Greensleeves was my delight "
        "Greensleeves was my heart of gold and who but my Lady Greensleeves "
        "So Greensleeves now farewell adieu may good fortune prosper thee "
        "For I am still thy lover true oh come once again and love me "
        "Greensleeves was all my joy Greensleeves was my delight "
        "Greensleeves my heart of gold and who but my Lady Greensleeves"
    ),
    "real_vocal": (
        "Hey now baby it's a beautiful day let's call in to work and we'll go out and play sweet darlin' "
        "Cause I've got nothin' but time on my hands I just wanna spend a little time with my man sweet darlin' "
        "You grab the rods and I'll get the bait we'll do a little fishin' let's make it a date "
        "with the blue skies above and the sun on our skin oh sweet darlin' let's go fishin' "
        "I'll pack us a lunch we'll eat by the water talk about the good times down at the harbor darlin' "
        "You can hold my hand like you did in high school I'll even let you sneak a kiss if you want to darlin' "
        "You grab the rods and I'll get the bait we'll do a little fishin' let's make it a date "
        "with the blue skies above and the sun on our skin oh sweet darlin' let's go fishin' "
        "Yeah let's go fishin' Come on darlin' Oh sweet darlin'"
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

# Note on `tonic` vs `harmony_tonic`:
# - `tonic` is the Jianpu `1 = X` key signature (relative major for minor keys: G for Em, F for Dm) required by Project/reharmonizer.
# - `harmony_tonic` + `minor` is the concert key root passed to chord_recognizer (E minor for Scarborough Fair, D minor for Greensleeves).
SONG_SPECS: dict[str, dict] = {
    "canon_in_d": {
        "tonic": "A",
        "harmony_tonic": "A",
        "minor": False,
        "meter": "4/4",
        "capo": 2,
        "preset": "emotional_pop",
        "style": "指弹律动",
        "bpm_range": (78.0, 92.0),
    },
    "romance_amor": {
        "tonic": "A",
        "harmony_tonic": "A",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "抒情琶音",
        "bpm_range": (74.0, 86.0),
    },
    "scarborough_fair": {
        "tonic": "G",
        "harmony_tonic": "E",
        "minor": True,
        "meter": "3/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "分解和弦",
        "bpm_range": (116.0, 142.0),
    },
    "greensleeves": {
        "tonic": "F",
        "harmony_tonic": "D",
        "minor": True,
        "meter": "3/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "分解和弦",
        "bpm_range": (84.0, 106.0),
    },
    "mandarin_pop": {
        "tonic": "D",
        "harmony_tonic": "D",
        "minor": False,
        "meter": "6/8",
        "capo": 2,
        "preset": "emotional_pop",
        "style": "抒情琶音",
        "bpm_range": (82.0, 102.0),
    },
    "real_vocal": {
        "tonic": "C",
        "harmony_tonic": "C",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "emotional_pop",
        "style": "指弹律动",
        "bpm_range": (80.0, 98.0),
    },
    "regular_friends": {
        "tonic": "F",
        "harmony_tonic": "F",
        "minor": False,
        "meter": "4/4",
        "capo": 0,
        "preset": "neosoul_jazz",
        "style": "Neo-Soul 律动",
        "bpm_range": (82.0, 96.0),
    },
}

MANIFEST_PATH = WORK_DIR / "repertoire_manifest.json"
PORTABLE_MANIFEST_PATH = ROOT / "data" / "repertoire_manifest.json"


def _from_manifest_path(raw: str) -> Path:
    norm = str(raw).replace("\\", "/")
    cand = Path(norm)
    if cand.is_file():
        return cand.resolve()
    if not cand.is_absolute():
        for base in (INSTALL_ROOT, ROOT, WORK_DIR):
            resolved = base / Path(norm.lstrip("/"))
            if resolved.is_file():
                return resolved.resolve()
    return Path(raw)


def _to_portable_path(p: Path) -> str:
    resolved = p.resolve()
    try:
        return resolved.relative_to(INSTALL_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _stem_is_audible(wav_path: Path) -> bool:
    if not wav_path.is_file() or wav_path.stat().st_size < 50_000:
        return False
    arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if arr.size == 0 or sr <= 0:
        return False
    ms = np.mean(arr.astype(np.float64) ** 2, axis=1)
    overall_rms = float(np.sqrt(np.mean(ms)))
    if overall_rms < 0.004:
        return False
    win = max(1, round(0.050 * sr))
    if len(ms) < win:
        trimmed = ms.reshape(1, -1)
    else:
        n_frames = len(ms) // win
        trimmed = ms[: n_frames * win].reshape(n_frames, win)
    f_rms = np.sqrt(np.mean(trimmed, axis=1))
    return float(np.percentile(f_rms, 90.0)) >= 0.008


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    t_all = time.time()

    for key, spec in SONG_SPECS.items():
        t0 = time.time()
        entry = manifest[key]
        src_wav = _from_manifest_path(entry["source"])
        info = sf.info(str(src_wav))
        clip_duration = round(float(info.duration), 3)
        meter = spec["meter"]
        raw_lyrics = FULL_LYRICS[key]

        # 1. MVSEP #1 Mel-Band-RoFormer-Karaoke GPU Stem Separation (reusing local cache)
        roformer_dir = WORK_DIR / f"stems_{key}_roformer"
        voc_wav = roformer_dir / "vocals.wav"
        acc_wav = roformer_dir / "no_vocals.wav"
        cached_ok = False
        if voc_wav.is_file() and voc_wav.stat().st_size > 100_000:
            v_arr, _ = sf.read(str(voc_wav), dtype="float32")
            if float(np.sqrt(np.mean(v_arr ** 2))) > 0.008:
                cached_ok = True
        if cached_ok:
            print(f"\n[{key}] 1. Reusing cached RoFormer stems in {roformer_dir.name}", flush=True)
        else:
            print(f"\n[{key}] 1. Running MVSEP #1 RoFormer GPU separation on {src_wav.name}...", flush=True)
            sep_res = separate_stems(src_wav, roformer_dir, prefer_cuda=True, strip_backing_harmonies=True)
            print(f"  -> Separation backend: {sep_res['backend']} ({sep_res['device']})", flush=True)
            voc_wav = Path(sep_res["vocals"])
            acc_wav = Path(sep_res.get("accompaniment") or (roformer_dir / "no_vocals.wav"))

        has_acc = _stem_is_audible(acc_wav)

        # 2. Forced-Projected Syllable Timestamps on isolated vocals.wav
        expected_tokens = len(re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", raw_lyrics))
        voc_stat = voc_wav.stat()
        wt_cache_key = hashlib.sha256(
            f"mms_ctc_v3|{voc_wav.name}|{voc_stat.st_size}|{int(voc_stat.st_mtime)}|{raw_lyrics}".encode("utf-8")
        ).hexdigest()[:16]
        cached_wt = entry.get("word_timestamps")
        if (
            entry.get("word_timestamps_cache_key") == wt_cache_key
            and isinstance(cached_wt, list)
            and len(cached_wt) >= max(1, int(0.8 * expected_tokens))
        ):
            word_timestamps = cached_wt
            print(
                f"[{key}] 2. Reusing cached word timestamps on {voc_wav.name} "
                f"(tokens={len(word_timestamps)}, {word_timestamps[0]['start']:.2f}s..{word_timestamps[-1]['end']:.2f}s)",
                flush=True,
            )
        else:
            print(f"[{key}] 2. Extracting forced-projected word timestamps on {voc_wav.name}...", flush=True)
            wt_res = transcribe_word_timestamps(voc_wav, prefer_cuda=True, ground_truth_text=raw_lyrics)
            word_timestamps = wt_res["word_timestamps"]
            interp_cnt = sum(1 for w in word_timestamps if w.get("interpolated"))
            unsung_cnt = sum(1 for w in word_timestamps if w.get("unsung"))
            print(
                f"  -> Timestamp backend: {wt_res['backend']} | tokens={len(word_timestamps)} | "
                f"interp={interp_cnt} | unsung={unsung_cnt} | "
                f"first={word_timestamps[0]['start']:.2f}s | last={word_timestamps[-1]['end']:.2f}s",
                flush=True,
            )
        measured_ts = [w for w in word_timestamps if not w.get("interpolated") and not w.get("unsung") and not w.get("subdivided")]
        subdiv_ts = [w for w in word_timestamps if not w.get("interpolated") and not w.get("unsung") and w.get("subdivided")]
        all_vocal_onsets = sorted(
            {round(float(w["start"]), 3) for w in measured_ts if "start" in w and float(w["start"]) >= 0.0}
            | {round(float(w["group_start"]), 3) for w in subdiv_ts if "group_start" in w and float(w["group_start"]) >= 0.0}
        )
        pre_voc_onsets = detect_pre_lyric_vocalise_onsets(voc_wav, all_vocal_onsets[0]) if all_vocal_onsets else []
        beat_vocal_onsets = sorted(set(pre_voc_onsets) | set(all_vocal_onsets))

        # 3. Zero-Hint Blind Beat & Meter Tracking via ISMIR #1 Beat This! + Multi-Cue Octave Posterior
        beat_wav = src_wav if has_acc else voc_wav
        print(f"[{key}] 3. Blind tracking rhythm & meter on {beat_wav.name}...", flush=True)
        beat_info = track_beats_and_downbeats(
            beat_wav,
            prefer_cuda=True,
            vocal_onsets_sec=beat_vocal_onsets,
            pre_vocal_bars=4 if has_acc else 1,
        )
        meter = str(beat_info["meter"])
        if meter != spec["meter"]:
            raise RuntimeError(f"[{key}] Blind meter mismatch: got {meter}, expected {spec['meter']}")
        bpm = round(float(beat_info["bpm"]), 3)
        beat_times = [round(float(x), 4) for x in beat_info["beat_times"]]
        bar_start_seconds = [round(float(x), 4) for x in beat_info["bar_start_seconds"]]
        beat_origin = round(float(bar_start_seconds[0]), 4)
        word_boundaries_sec = [t for t in all_vocal_onsets if t >= beat_origin]
        print(
            f"  -> Beat backend: {beat_info['backend']} | bpm={bpm} | blind_meter={meter} | origin={beat_origin}s | "
            f"beats={len(beat_times)} | bars={len(bar_start_seconds)-1}",
            flush=True,
        )

        # 4. RMVPE + ROSVOT Singing Transcription + Piecewise Beat Regulation
        print(f"[{key}] 4. Transcribing & regulating vocal notes on piecewise beat_times grid...", flush=True)
        tx = transcribe_vocal_notes(
            voc_wav,
            bpm=bpm,
            beat_origin=beat_origin,
            word_boundaries_sec=word_boundaries_sec,
            prefer_cuda=True,
            beat_times=beat_times,
        )
        raw_notes = tx["notes"]
        if all_vocal_onsets:
            pickup_tol_sec = min(0.5, 0.5 * (60.0 / bpm))
            raw_notes = filter_pre_lyric_bleed_notes(
                raw_notes, voc_wav, all_vocal_onsets[0], beat_origin, pickup_tol_sec
            )
        notes = regulate_singing_notes_for_score(
            raw_notes,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            word_boundaries_sec=word_boundaries_sec,
            beat_times=beat_times,
        )
        print(f"  -> Transcribed {len(notes)} human vocal notes (backend={tx['backend']})", flush=True)

        # 5. Zero-Hint Blind Accompaniment-First Harmony & Key/Mode Recognition
        harm = None
        tonic_pc = NAMES.index(spec["harmony_tonic"])
        is_minor = bool(spec["minor"])
        if has_acc:
            acc_harm = recognize_accompaniment_chords(
                accompaniment_path=acc_wav,
                bpm=bpm,
                beat_origin=beat_origin,
                meter=meter,
                tonic=None,
                minor=None,
                clip_duration=clip_duration,
                bar_start_seconds=bar_start_seconds,
                beat_times=beat_times,
                notes=notes,
                vocal_audio_path=voc_wav,
            )
            if acc_harm.get("available") and acc_harm.get("chords"):
                harm = acc_harm
                tonic_pc = int(acc_harm["tonic_pc"])
                is_minor = bool(acc_harm["is_minor"])
                inferred_ht = NAMES[tonic_pc]
                if inferred_ht != spec["harmony_tonic"] or is_minor != bool(spec["minor"]):
                    raise RuntimeError(
                        f"[{key}] Blind key/mode mismatch: got {inferred_ht}{'m' if is_minor else ''}, "
                        f"expected {spec['harmony_tonic']}{'m' if spec['minor'] else ''}"
                    )
                print(
                    f"  -> Blind recognized {len(harm['chords'])} bars of harmony "
                    f"(blind_key={inferred_ht}{'m' if is_minor else ''}): first 10 = {harm['chords'][:10]}",
                    flush=True,
                )
        if harm is None:
            if notes:
                k_est = estimate_key_albrecht_shanahan(notes)
                tonic_pc = int(k_est["tonic_pc"])
                is_minor = bool(k_est["is_minor"])
            harm = synthesize_melody_conditioned_harmony(
                notes=notes,
                bpm=bpm,
                beat_origin=beat_origin,
                meter=meter,
                tonic=NAMES[tonic_pc],
                minor=is_minor,
                clip_duration=clip_duration,
                bar_start_seconds=bar_start_seconds,
                beat_times=beat_times,
            )
            print(
                f"  -> Fallback vocal-melody harmony ({len(harm['chords'])} bars): first 10 = {harm['chords'][:10]}",
                flush=True,
            )
        tonic = NAMES[(tonic_pc + 3) % 12] if is_minor else NAMES[tonic_pc]
        base_chords = harm["chords"]
        bar_dynamics = harm["bar_dynamics"]

        # 6. Territory + 3-State Viterbi Lyric Alignment & Pure Musical Preset Recommendation
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
        rec = recommend_artistry(project, base_chords, bar_dynamics)
        rec_preset = rec["preset"]
        rec_style = rec["style"]
        non_empty_lyrics = sum(1 for lyr in note_lyrics if lyr and lyr != "-")
        melisma_count = sum(1 for lyr in note_lyrics if lyr == "-")
        print(
            f"  -> Aligned {non_empty_lyrics} lyrics + {melisma_count} melismas across {len(notes)} notes | "
            f"Recommended: {rec_preset} ({rec_style}) [{rec['reason']}] in {time.time() - t0:.1f}s",
            flush=True,
        )

        for wav_k in ("accompaniment_wav", "no_vocals_wav"):
            if wav_k in entry and entry[wav_k]:
                entry[wav_k] = _to_portable_path(Path(_from_manifest_path(entry[wav_k])))
        entry.update(
            source=_to_portable_path(src_wav),
            tonic=tonic,
            meter=meter,
            bpm=bpm,
            beat_origin=beat_origin,
            clip_duration=clip_duration,
            vocals_wav=_to_portable_path(voc_wav),
            capo=spec["capo"],
            style=rec_style,
            preset=rec_preset,
            split_bar=False,
            passing_bass=(rec_preset in ("emotional_pop", "neosoul_jazz")),
            embellish_hp=(rec_preset in ("emotional_pop", "neosoul_jazz")),
            intro_mode="chord_melody",
            recommended_preset=rec_preset,
            recommended_reason=rec["reason"],
            base_chords=base_chords,
            detected_split_chords={
                str(k): list(v)
                for k, v in sorted((harm.get("split_chords") or {}).items(), key=lambda kv: int(kv[0]))
            },
            bar_start_seconds=bar_start_seconds,
            beat_times=beat_times,
            bar_dynamics=bar_dynamics,
            word_timestamps_cache_key=wt_cache_key,
            word_timestamps=word_timestamps,
            raw_lyrics=raw_lyrics,
            note_lyrics=note_lyrics,
            notes=[asdict(n) for n in notes],
        )
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
        MANIFEST_PATH.write_text(serialized, encoding="utf-8")
        PORTABLE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PORTABLE_MANIFEST_PATH.write_text(serialized, encoding="utf-8")

    print(f"\n=== ALL 7 SONGS REBUILT WITH PLAN A 2026 SOTA STACK IN {time.time() - t_all:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
