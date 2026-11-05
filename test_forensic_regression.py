"""End-to-end regression suite for Phase 2 forensic bug fixes and cross-module invariants."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music import Note, Project, parse_jianpu
import guitar
from subpkgs.stem_mir.beat_tracker import (
    _metrical_level_score,
    _syllable_ioi_penalty,
    _tempo_range_penalty,
    track_beats_and_downbeats,
)
from subpkgs.stem_mir.chord_recognizer import (
    _degree_index,
    _gttm_note_salience,
    synthesize_melody_conditioned_harmony,
)
from subpkgs.singing_lyrics.singing_transcriber import (
    estimate_key_albrecht_shanahan,
    regulate_singing_notes_for_score,
)
from subpkgs.singing_lyrics.lyric_aligner import align_lyrics_to_notes
from subpkgs.guitar_reharm.chord_voicing import (
    _pair_strain_mm,
    _required,
    generate_voicings,
    parse_extended_chord,
)
from subpkgs.guitar_reharm.reharmonizer import (
    check_melody_clash,
    reharmonize_progression,
)
from subpkgs.guitar_reharm.arrangement_grooves import arrange_advanced
from subpkgs.guitar_reharm.guitar_simulator import (
    _pluck,
    _pluck_fallback,
    evaluate_vocal_guitar_mix,
    synthesize_realistic_guitar,
)
import subpkgs.score_web.server as srv


class ForensicRegressionTests(unittest.TestCase):
    def test_beat_tracker_narrow_bpm_range_and_compound_meter_scale(self):
        onsets = np.arange(0.0, 8.0, 60.0 / 136.0, dtype=np.float64)
        acf = np.ones(400, dtype=np.float64)
        score = _metrical_level_score(acf, 100.0, 2.5, 4, onsets, 68.0, 132.0)
        self.assertTrue(np.isfinite(score))
        self.assertTrue(np.isfinite(_tempo_range_penalty(96.0, 68.0, 132.0)))
        self.assertTrue(np.isfinite(_syllable_ioi_penalty(onsets, 0.5)))

        sr = 22050
        dur = 6.0
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        sig = np.zeros_like(t)
        for k in range(14):
            idx = int((0.2 + k * (60.0 / 136.0)) * sr)
            if idx + 400 < len(sig):
                sig[idx : idx + 400] = np.sin(2 * np.pi * 220.0 * t[:400]) * 0.8
        with tempfile.TemporaryDirectory() as tmp:
            wav_p = Path(tmp) / "compound.wav"
            sf.write(str(wav_p), sig.astype(np.float32), sr)
            with self.assertRaises(ValueError):
                track_beats_and_downbeats(wav_p, prefer_cuda=False, bpm_range=(140.0, 120.0))
            info_narrow = track_beats_and_downbeats(wav_p, prefer_cuda=False, bpm_range=(130.0, 142.0))
            self.assertGreaterEqual(info_narrow["bpm"], 130.0)
            self.assertLessEqual(info_narrow["bpm"], 142.0)
            info_68 = track_beats_and_downbeats(wav_p, prefer_cuda=False, meter_hint="6/8")
            self.assertEqual(info_68["meter"], "6/8")
            self.assertGreater(len(info_68["bar_start_seconds"]), 1)

    def test_chord_recognizer_salience_degree_index_and_scalar_triad(self):
        sal_eighth = _gttm_note_salience(0.5, 0.5, 4.0, "4/4")
        sal_offgrid = _gttm_note_salience(0.25, 0.5, 4.0, "4/4")
        self.assertGreater(sal_eighth, sal_offgrid)

        self.assertEqual(_degree_index(2, "7", 0, False), 1)
        self.assertEqual(_degree_index(4, "7", 0, False), 2)
        self.assertEqual(_degree_index(7, "7", 0, False), 4)

        notes = [
            Note(start=0.0, duration=1.5, pitch=60, raw_start=0.0, raw_duration=1.5),
            Note(start=1.5, duration=0.5, pitch=62, raw_start=0.75, raw_duration=0.25),
            Note(start=2.0, duration=2.0, pitch=64, raw_start=1.0, raw_duration=1.0),
        ]
        res = synthesize_melody_conditioned_harmony(notes, bpm=120.0, beat_origin=0.0, meter="4/4", tonic="C")
        self.assertEqual(res["chords"][0], "C")

    def test_singing_transcriber_confidence_clamp_and_validation(self):
        notes, _ = parse_jianpu("1:1 3:1 5:2 | 1':2 5:2")
        key_res = estimate_key_albrecht_shanahan(notes)
        self.assertGreaterEqual(key_res["confidence"], 0.0)
        self.assertLessEqual(key_res["confidence"], 1.0)

        with self.assertRaises(ValueError):
            regulate_singing_notes_for_score(notes, bpm=0.0)
        with self.assertRaises(ValueError):
            regulate_singing_notes_for_score(notes, bpm=120.0, min_grid_beat=0.0)

    def test_lyric_aligner_edited_notes_and_rest_melisma_guard(self):
        n1 = Note(start=0.0, duration=1.0, pitch=60, raw_start=0.0, raw_duration=0.5, edited=True)
        n2 = Note(start=1.0, duration=1.0, pitch=62, raw_start=0.5, raw_duration=0.5, edited=True)
        n3 = Note(start=3.0, duration=1.0, pitch=64, raw_start=1.5, raw_duration=0.5, edited=True)
        proj = Project(title="test", tonic="C", meter="4/4", bpm=60.0, beat_origin=0.0, notes=[n1, n2, n3])
        ts = [
            {"word": "春", "start": 0.05, "end": 0.8},
            {"word": "风", "start": 3.05, "end": 3.8},
        ]
        res = align_lyrics_to_notes(proj, raw_lyrics="春风", word_timestamps=ts)
        self.assertEqual(res["note_lyrics"][0], "春")
        self.assertEqual(res["note_lyrics"][2], "风")

    def test_chord_voicing_cm13_slash_g_preserves_minor_third_and_fingers(self):
        parsed = parse_extended_chord("Cm13/G")
        self.assertIsNotNone(parsed)
        req_set = _required(parsed)
        self.assertIn((parsed["root"] + 3) % 12, req_set)
        voicings = generate_voicings("Cmaj7", capo=0)
        self.assertTrue(voicings)
        self.assertIn("fingers", voicings[0])
        self.assertIsInstance(voicings[0]["fingers"], int)

        strain_narrow = _pair_strain_mm({1: [(1, 1), (2, 1)], 2: [(3, 2)]}, 0)
        strain_wide = _pair_strain_mm({1: [(1, 1), (5, 1)], 2: [(3, 6)]}, 0)
        self.assertGreater(strain_wide, strain_narrow)

    def test_reharmonizer_maj7_melody_and_capo_propagation(self):
        notes = [Note(start=0.0, duration=2.0, pitch=71, raw_start=0.0, raw_duration=1.0)]  # B4 over C
        rep = check_melody_clash("C", notes, 0.0, 4.0, "C")
        self.assertNotEqual(rep["status"], "clash")

        proj = Project(title="test", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        bars_c2 = reharmonize_progression(["C", "Am"], proj, preset="original", split_bar=True, capo=2)
        self.assertTrue(bars_c2)
        for seg in bars_c2[0]["split_segments"]:
            self.assertEqual(seg["capo"], 2)

    def test_arrangement_grooves_nc_slap_and_cm_sustain(self):
        notes = [
            Note(start=0.0, duration=2.0, pitch=64, raw_start=0.0, raw_duration=1.0),
        ]
        proj = Project(title="test", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        arr = arrange_advanced(
            proj,
            chords=["N.C.", "C"],
            style="指弹律动",
            slap_x=True,
            embellish_hp=True,
            intro_mode="chord_melody",
            instrumental_melody={1: [{"beat": 0.0, "duration": 4.0, "pitch": 72}]},
        )
        bar0_slaps = [e for e in arr["events"] if e["bar"] == 0 and e.get("technique") == "X"]
        self.assertEqual(len(bar0_slaps), 0)

        bar1_cm = [e for e in arr["events"] if e["bar"] == 1 and e.get("technique") == "CM" and abs(e["start"] - 4.0) < 1e-6]
        self.assertEqual(len(bar1_cm), 1)
        self.assertAlmostEqual(bar1_cm[0]["duration"], 4.0, places=3)

    def test_guitar_simulator_validation_and_register_aware_mix_eval(self):
        self.assertEqual(len(_pluck(60, 3, 0, 44100, 80, "", 1)), 0)
        self.assertEqual(len(_pluck_fallback(60, 3, 0, 44100, 80, "", 1)), 0)

        notes, _ = parse_jianpu("1:1 2:1 3:1 5:1")
        proj = Project(title="test", tonic="C", meter="4/4", bpm=120.0, notes=notes)
        arr = arrange_advanced(proj, chords=["C"], style="分解和弦")
        with self.assertRaises(ValueError):
            synthesize_realistic_guitar(proj, dict(arr, bar_start_seconds=[2.0, 1.0]))

        sr = 44100
        t = np.linspace(0.0, 0.5, sr // 2, endpoint=False)
        v_octave = np.sin(2 * np.pi * 440.0 * t)
        g_octave = np.sin(2 * np.pi * 220.0 * t)
        g_clash = np.sin(2 * np.pi * 415.3 * t)
        m_con = evaluate_vocal_guitar_mix(v_octave, g_octave, sr=sr)
        m_dis = evaluate_vocal_guitar_mix(v_octave, g_clash, sr=sr)
        self.assertGreater(m_con["spectral_consonance"], m_dis["spectral_consonance"])

    def test_server_state_payload_and_beat_origin_stem_alignment(self):
        sess = srv._initial_session("real_vocal")
        self.assertEqual(sess.demo_mode, "real_vocal")
        payload = srv.get_state_payload(sess)
        self.assertIn("bar_start_seconds", payload)
        self.assertIn("beat_times", payload)
        self.assertEqual(list(sess.project.bar_start_seconds), list(sess.bar_start_seconds))

        sr = 44100
        with tempfile.TemporaryDirectory() as tmp:
            voc_path = Path(tmp) / "voc.wav"
            # Put an impulse at exactly beat_origin = 0.5s
            voc_audio = np.zeros((sr * 2, 2), dtype=np.float32)
            voc_audio[int(0.5 * sr) : int(0.5 * sr) + 100] = 0.8
            sf.write(str(voc_path), voc_audio, sr)
            sess.project.beat_origin = 0.5
            sess.project.audio["vocals"] = str(voc_path)
            v_only = srv._vocal_only_samples(sess, sr=sr)
            v_mix = srv._vocal_mix_samples(sess, sr=sr)
            g_only = srv._guitar_samples(sess, sr=sr)
            # At t=0 (after slicing beat_origin=0.5s), v_mix - 0.85*g_only must equal 0.7/0.85 * v_only
            min_len = min(len(v_only), len(v_mix), len(g_only))
            residual = v_mix[:min_len] - g_only[:min_len] * 0.85
            expected_vocal = v_only[:min_len] * (0.7 / 0.85)
            self.assertLess(float(np.max(np.abs(residual - expected_vocal))), 1e-4)

    def test_lkktest_wave1_counterexamples(self):
        from subpkgs.singing_lyrics.lyric_aligner import _nw_score, _project_records_to_ground_truth
        from subpkgs.stem_mir.beat_tracker import (
            _acf_at,
            _onset_autocorrelation,
            beats_to_seconds_piecewise,
            seconds_to_beats_piecewise,
        )

        # Issue 2: 'station' vs 'starlight' must not get a false prefix boost, and any non-exact match is interpolated
        self.assertLess(_nw_score("starlight", "station"), 0.0)
        proj_typo = _project_records_to_ground_truth(
            [{"word": "starligh", "start": 1.0, "end": 2.0, "probability": 0.9}],
            "starlight",
        )
        self.assertEqual(len(proj_typo), 1)
        self.assertTrue(proj_typo[0].get("interpolated", False))

        # Issue 3: Rejected out-of-window anchor ('old' at [0, 0.5] vs notes at [10, 10.5] and [11, 11.5])
        n_late1 = Note(start=20.0, duration=1.0, pitch=60, raw_start=10.0, raw_duration=0.5)
        n_late2 = Note(start=22.0, duration=1.0, pitch=62, raw_start=11.0, raw_duration=0.5)
        proj_late = Project(title="late", tonic="C", meter="4/4", bpm=120.0, beat_origin=0.0, notes=[n_late1, n_late2], source="vocal.wav")
        res_late = align_lyrics_to_notes(
            proj_late,
            raw_lyrics="old middle new",
            word_timestamps=[
                {"word": "old", "start": 0.0, "end": 0.5, "probability": 0.95},
                {"word": "new", "start": 11.0, "end": 11.5, "probability": 0.95},
            ],
        )
        self.assertEqual(res_late["note_lyrics"], ["middle", "new"])

        # Issue 5 & Issue 6: Piecewise beat_times round-trip in guitar.midi and guitar.render_audio
        bt_nonunif = [0.0, 0.4, 1.0, 1.4, 2.0]
        b_coord = seconds_to_beats_piecewise(0.4, bt_nonunif, 120.0, 0.0)
        self.assertAlmostEqual(b_coord, 1.0, places=6)
        self.assertAlmostEqual(beats_to_seconds_piecewise(b_coord, bt_nonunif, 120.0, 0.0), 0.4, places=6)
        proj_bt = Project(
            title="bt",
            tonic="C",
            meter="4/4",
            bpm=120.0,
            beat_origin=0.0,
            bar_start_seconds=[0.0, 2.0],
            beat_times=bt_nonunif,
            notes=[Note(start=1.0, duration=1.0, pitch=60, raw_start=0.4, raw_duration=0.6)],
        )
        arr_bt = guitar.arrange(proj_bt, ["C"])
        midi_bytes = guitar.midi(proj_bt, arr_bt)
        # First beat tempo is 0.4s = 400,000 us -> \x06\x1a\x80
        self.assertIn(b"\xff\x51\x03" + (400_000).to_bytes(3, "big"), midi_bytes)

        # Issue 8: _acf_at at final index len(acf) - 1
        acf = _onset_autocorrelation(np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
        self.assertGreater(_acf_at(acf, 4, halfwidth=2), 0.25)

    def test_lkktest_wave2_counterexamples(self):
        from fastapi.testclient import TestClient

        # 1. _implemented probe: function catching NotImplementedError internally passes; function raising TypeError fails
        def _working_with_fallback(proj, chords):
            try:
                raise NotImplementedError("optional accelerator unavailable")
            except NotImplementedError:
                return {"events": [], "bars": []}

        def _broken_type_error(proj, chords):
            raise TypeError("internal bug")

        self.assertTrue(srv._implemented(_working_with_fallback))
        self.assertFalse(srv._implemented(_broken_type_error))

        # 2. _rebuild sort permutation of note_lyrics and note_ref_indices
        unsorted_notes = [
            Note(pitch=67, start=2.0, duration=1.0, raw_start=1.0, raw_duration=0.5),
            Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.5),
        ]
        proj_unsorted = Project(
            title="unsorted",
            tonic="C",
            meter="4/4",
            bpm=120.0,
            notes=unsorted_notes,
            reference_notes=[
                Note(pitch=67, start=2.0, duration=1.0, raw_start=1.0, raw_duration=0.5),
                Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.5),
            ],
        )
        sess_unsorted = srv.Session(
            project=proj_unsorted,
            base_chords=["C"],
            custom_voicings=[None],
            note_lyrics=["late", "early"],
            reference_note_lyrics=["late", "early"],
            note_ref_indices=[0, 1],
        )
        srv._rebuild(sess_unsorted)
        self.assertEqual([n.pitch for n in sess_unsorted.project.notes], [60, 67])
        self.assertEqual(sess_unsorted.note_lyrics, ["early", "late"])
        self.assertEqual(sess_unsorted.note_ref_indices, [1, 0])

        # 3. _rebuild bar count with bar_start_seconds=[0, 2, 4] and 6s source clip_duration (3 bars)
        proj_3bars = Project(
            title="3bars",
            tonic="C",
            meter="4/4",
            bpm=120.0,
            clip_duration=6.0,
            source="six_sec.wav",
            notes=[Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.5)],
            bar_start_seconds=[0.0, 2.0, 4.0],
        )
        suggested_3 = guitar.suggest(proj_3bars, 0)
        voicings_3 = [generate_voicings(ch, 0)[0]["frets"] for ch in suggested_3]
        sess_3bars = srv.Session(
            project=proj_3bars,
            base_chords=["C", "G"],
            custom_voicings=voicings_3,
            bar_start_seconds=[0.0, 2.0, 4.0],
        )
        srv._rebuild(sess_3bars)
        self.assertEqual(len(sess_3bars.base_chords), 3)
        self.assertEqual(len(sess_3bars.custom_voicings), 3)
        self.assertEqual(sess_3bars.custom_voicings[0], voicings_3[0])

        # 4. _load_stem_audio content-digest invalidation when file at same path is overwritten
        sr = 22050
        with tempfile.TemporaryDirectory() as tmp:
            p_wav = Path(tmp) / "regen.wav"
            a1 = np.full((sr // 4, 2), 0.25, dtype=np.float32)
            a2 = np.full((sr // 4, 2), 0.75, dtype=np.float32)
            sf.write(str(p_wav), a1, sr, subtype="FLOAT")
            out1 = srv._load_stem_audio(p_wav, sr, 0.0)
            sf.write(str(p_wav), a2, sr, subtype="FLOAT")
            out2 = srv._load_stem_audio(p_wav, sr, 0.0)
            self.assertGreater(float(np.mean(out2)), float(np.mean(out1)) + 0.4)

        # 5. FastAPI /api/reharmonize split_bar toggle after segment chord override & meter change with custom_voicings
        test_app = srv.create_app()
        client = TestClient(test_app)
        client.post("/api/load_demo", json={"demo": "real_vocal"})
        r_split_on = client.post("/api/reharmonize", json={"split_bar": True})
        self.assertEqual(r_split_on.status_code, 200)
        r_seg = client.post("/api/bar_chord", json={"bar": 0, "segment": 1, "chord": "G"})
        self.assertEqual(r_seg.status_code, 200)
        r_split_off = client.post("/api/reharmonize", json={"split_bar": False})
        self.assertEqual(r_split_off.status_code, 200)
        self.assertFalse(r_split_off.json()["split_bar"])


if __name__ == "__main__":
    unittest.main()
