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
        expected_3 = ["C", "G", suggested_3[2]]
        voicings_3 = [generate_voicings(ch, 0)[0]["frets"] for ch in expected_3]
        sess_3bars = srv.Session(
            project=proj_3bars,
            base_chords=["C", "G"],
            custom_voicings=voicings_3,
            bar_start_seconds=[0.0, 2.0, 4.0],
        )
        srv._rebuild(sess_3bars)
        self.assertEqual(sess_3bars.base_chords, expected_3)
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

    def test_plan_a_audit_counterexamples(self):
        import xml.etree.ElementTree as ET
        from subpkgs.guitar_reharm.guitar_simulator import _active_rms
        from subpkgs.score_web.yoopu_svg_renderer import _svg_text
        from subpkgs.stem_mir.beat_tracker import _bar_beat_positions
        from subpkgs.stem_mir.chord_recognizer import synthesize_melody_conditioned_harmony

        # 1. Compound 6/8 _bar_beat_positions with half-bar dotted-quarter beat at 1.2s in [0.0, 2.4]
        pos_68 = _bar_beat_positions(np.array([1.35]), 0.0, 2.4, measure_beats=3, is_compound=True)
        self.assertAlmostEqual(pos_68[0], 0.0, places=4)
        self.assertAlmostEqual(pos_68[1], 0.90, places=3)
        self.assertAlmostEqual(pos_68[2], 1.70, places=3)
        self.assertEqual(len(pos_68), 3)

        # 2. Short-buffer (< 50ms) safety in _active_rms and _has_audible_accompaniment
        short_stereo = np.full((400, 2), 0.15, dtype=np.float32)
        rms_short, _ = _active_rms(short_stereo, sr=44100)
        self.assertAlmostEqual(rms_short, 0.15, places=3)
        with tempfile.TemporaryDirectory() as tmp:
            short_wav = Path(tmp) / "short_acc.wav"
            sf.write(str(short_wav), short_stereo, 44100)
            self.assertTrue(srv._has_audible_accompaniment(short_wav))

        # 3. _svg_text emits a single valid XML text-anchor attribute when overridden
        svg_node = _svg_text(10.0, 20.0, "1 = D", 12, text_anchor="start")
        self.assertEqual(svg_node.count("text-anchor="), 1)
        parsed = ET.fromstring(svg_node)
        self.assertEqual(parsed.attrib.get("text-anchor"), "start")

        # 4. _calibrate_lead_vocal_stem content-hash cache distinguishes different waveforms
        sr = 22050
        t = np.linspace(0.0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
        v_a = np.column_stack([0.2 * np.sin(2 * np.pi * 220.0 * t), 0.2 * np.sin(2 * np.pi * 220.0 * t)])
        v_b = np.column_stack([0.2 * np.sin(2 * np.pi * 660.0 * t), 0.2 * np.sin(2 * np.pi * 660.0 * t)])
        cal_a = srv._calibrate_lead_vocal_stem(v_a, sr)
        cal_b = srv._calibrate_lead_vocal_stem(v_b, sr)
        self.assertGreater(float(np.max(np.abs(cal_a - cal_b))), 0.05)

        # 5. 7sus4 avoid-note check penalizes natural 3rd melody clash against sus4
        mel_b4 = [Note(start=0.0, duration=4.0, pitch=71, raw_start=0.0, raw_duration=2.0)]
        harm = synthesize_melody_conditioned_harmony(mel_b4, bpm=120.0, beat_origin=0.0, meter="4/4", tonic="C", minor=False, clip_duration=2.0)
        self.assertNotEqual(harm["chords"][0], "G7sus4")

        # 6. Portable manifest path resolution handles missing relative paths cleanly
        self.assertIsNone(srv._resolve_repertoire_path("work/repertoire_audio/nonexistent_file_xyz.wav"))

    def test_fable_audit_and_musicality_deliverables(self):
        from fastapi.testclient import TestClient
        from subpkgs.guitar_reharm import recommend_artistry
        from subpkgs.singing_lyrics.lyric_aligner import _project_records_to_ground_truth, _token_match
        from subpkgs.singing_lyrics.singing_transcriber import regulate_singing_notes_for_score
        from subpkgs.stem_mir.beat_tracker import _onset_autocorrelation, _tactus_octave_log_posterior

        # A2: regulate_singing_notes_for_score preserves immutable physical raw_start / raw_duration
        raw_in = [
            Note(start=0.0, duration=1.0, pitch=64, raw_start=10.275, raw_duration=0.785),
            Note(start=2.0, duration=1.0, pitch=67, raw_start=11.310, raw_duration=0.640),
        ]
        reg_out = regulate_singing_notes_for_score(raw_in, bpm=120.0, beat_origin=10.0, clip_duration=14.0)
        self.assertAlmostEqual(reg_out[0].raw_start, 10.275, places=5)
        self.assertAlmostEqual(reg_out[0].raw_duration, 0.785, places=5)

        # A1, A3, A4: /api/cleanup piecewise bounds, /api/reharmonize meter/bpm sync, tail-only base_chords resize
        bt_warped = [0.0, 0.4, 1.0, 1.4, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        test_app = srv.create_app()
        with test_app.state.session_lock:
            s_cur = test_app.state.session
            p_warped = Project(
                title="seam_test",
                tonic="C",
                meter="4/4",
                bpm=120.0,
                beat_origin=0.0,
                clip_duration=6.0,
                source="dummy_rec.wav",
                notes=[
                    Note(pitch=60, start=1.25, duration=0.5, raw_start=0.4, raw_duration=0.3, edited=False),
                    Note(pitch=64, start=2.0, duration=1.0, raw_start=1.0, raw_duration=0.4, edited=True),
                    Note(pitch=67, start=7.0, duration=1.0, raw_start=3.5, raw_duration=0.5, edited=False),
                ],
                bar_start_seconds=[0.0, 2.0, 4.0, 6.0],
                beat_times=list(bt_warped),
            )
            s_cur.project = p_warped
            s_cur.capo = 0
            s_cur.preset = "original"
            s_cur.split_bar = False
            s_cur.bar_start_seconds = [0.0, 2.0, 4.0, 6.0]
            s_cur.beat_times = list(bt_warped)
            s_cur.base_chords = ["Cmaj7", "Am7", "Dm7"]
            s_cur.custom_voicings = [[-1, 3, 2, 0, 0, 0], None, None]
            s_cur.note_lyrics = ["一", "二", "三"]
            s_cur.reference_note_lyrics = ["一", "二", "三"]
            s_cur.note_ref_indices = [0, 1, 2]
            srv._rebuild(s_cur)

        client = TestClient(test_app)
        # A1: cleanup uses piecewise beat_times (raw_start=0.4 -> beat 1.0, bounded below edited note at 2.0)
        r_clean = client.post("/api/cleanup", json={"mode": "binary"})
        self.assertEqual(r_clean.status_code, 200)
        n_tbl = r_clean.json()["notes_table"]
        self.assertAlmostEqual(n_tbl[0]["start"], 1.0, places=3)
        self.assertLessEqual(n_tbl[0]["start"] + n_tbl[0]["duration"], 2.0 + 1e-6)

        # A4: moving last note into bar 3 extends tail only without wiping bar 0's chord or custom voicing
        r_ext = client.post("/api/edit_note", json={"index": 2, "start": 12.5, "duration": 1.0})
        self.assertEqual(r_ext.status_code, 200)
        with test_app.state.session_lock:
            self.assertEqual(len(test_app.state.session.base_chords), 4)
            self.assertEqual(test_app.state.session.base_chords[:3], ["Cmaj7", "Am7", "Dm7"])
            self.assertEqual(test_app.state.session.custom_voicings[0], [-1, 3, 2, 0, 0, 0])

        # A3: meter-only change preserves beat_times and recomputes 3/4 bar_start_seconds
        r_meter = client.post("/api/reharmonize", json={"meter": "3/4"})
        self.assertEqual(r_meter.status_code, 200)
        m_json = r_meter.json()
        self.assertEqual(len(m_json["beat_times"]), len(bt_warped))
        self.assertAlmostEqual(m_json["bar_start_seconds"][1], bt_warped[3], places=3)

        # A3: bpm change on recording-backed session clears beat_times and resyncs unedited notes from raw_start
        r_bpm = client.post("/api/reharmonize", json={"bpm": 60.0})
        self.assertEqual(r_bpm.status_code, 200)
        b_json = r_bpm.json()
        self.assertEqual(b_json["beat_times"], [])
        # Note 0 had raw_start=0.4s -> at 60 BPM (1 beat/sec) from origin=0.0, start snaps near 0.5 (not 1.0)
        self.assertLess(b_json["notes_table"][0]["start"], 0.76)

        # A5: 6/8 compound strum direction follows metrical depth (beat 1.5 = downstroke, weak eighths = upstroke)
        proj_68 = Project(
            title="68_dir",
            tonic="D",
            meter="6/8",
            bpm=90.0,
            notes=[Note(pitch=62, start=0.0, duration=1.5), Note(pitch=66, start=1.5, duration=1.5)],
        )
        arr_68 = arrange_advanced(proj_68, chords=["D"], style="扫弦伴奏")
        ev_by_beat = {}
        for ev in arr_68["events"]:
            ev_by_beat.setdefault(round(ev["beat"], 2), set()).add(ev.get("direction"))
        self.assertEqual(ev_by_beat.get(0.0), {"↓"})
        self.assertEqual(ev_by_beat.get(1.5), {"↓"})
        for wk in (0.5, 1.0, 2.0, 2.5):
            if wk in ev_by_beat:
                self.assertEqual(ev_by_beat[wk], {"↑"})

        # A6 & B3 UI wiring check in static/index.html
        idx_html = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("sj.status === 'superseded'", idx_html)
        self.assertIn("function pickStyle(", idx_html)
        self.assertIn("★ 本曲推荐", idx_html)

        # Deliverable 2: Multi-cue tactus octave posterior flips double-time 164 BPM -> 82 BPM (m* = 2.0)
        sr_env = 100.0
        env_len = int(20 * sr_env)
        env_t = np.zeros(env_len, dtype=np.float64)
        # True pulse every 0.7317s (~82 BPM), tracker falsely emitted double-time every 0.36585s (~164 BPM)
        raw_164 = np.arange(0.0, 18.0, 60.0 / 164.0)
        for idx_p, bt_s in enumerate(raw_164):
            f_i = int(round(bt_s * sr_env))
            if f_i < env_len:
                env_t[f_i] = 1.0 if (idx_p % 2 == 0) else 0.75
        acf_sub = _onset_autocorrelation(env_t)
        raw_db_164 = raw_164[::4]  # Tracker grouped 4 fast pulses = 1.4634s pseudo-bar (half of true 4/4 bar)
        median_bar_164 = float(np.median(np.diff(raw_db_164)))
        vocal_82 = np.arange(0.5, 17.5, 0.40, dtype=np.float64)
        post_82 = _tactus_octave_log_posterior(
            acf_sub=acf_sub,
            frames_per_sec=sr_env,
            median_bar=median_bar_164,
            bar_q_beats=4.0,
            beat_arr=raw_164,
            cleaned_db=raw_db_164,
            syl_arr=vocal_82,
            lo_bpm=55.0,
            hi_bpm=190.0,
            is_compound_meter=False,
            d_hc_sec=2.10,
        )
        best_scale = max(post_82, key=lambda sc: post_82[sc])
        self.assertEqual(best_scale, 2.0)

        # Deliverable 3: Phonetic CJK matching + unsung Verse 3 cutoff + humming island & 16th-rest melisma
        sc_trad, tr_trad = _token_match("愛", "爱")
        self.assertGreater(sc_trad, 2.0)
        self.assertGreater(tr_trad, 0.8)
        sc_homo, _ = _token_match("他", "她")
        self.assertGreater(sc_homo, 2.0)
        sc_diff, _ = _token_match("他", "大")
        self.assertLessEqual(sc_diff, 0.0)

        asr_2v = [
            {"word": "春", "start": 4.0, "end": 4.4, "probability": 0.95},
            {"word": "风", "start": 4.5, "end": 4.9, "probability": 0.95},
        ]
        proj_3v = _project_records_to_ground_truth(asr_2v, "春风 秋雨 冬雪")
        self.assertFalse(proj_3v[0].get("unsung", False))
        self.assertFalse(proj_3v[1].get("unsung", False))
        self.assertTrue(all(r.get("unsung", False) for r in proj_3v[2:]))

        # Humming note at beat 0.0 (raw 0.2s) > 3 beats before first lyric at 4.0s, plus 16th-rest melisma on '风'
        notes_lyr = [
            Note(start=0.0, duration=1.5, pitch=60, raw_start=0.2, raw_duration=0.7),
            Note(start=8.0, duration=1.0, pitch=64, raw_start=4.0, raw_duration=0.45),
            Note(start=9.0, duration=0.75, pitch=67, raw_start=4.5, raw_duration=0.35),
            Note(start=10.0, duration=1.0, pitch=69, raw_start=4.95, raw_duration=0.45),
        ]
        proj_lyr = Project(title="lyr", tonic="C", meter="4/4", bpm=120.0, beat_origin=0.0, source="v.wav", notes=notes_lyr)
        al_out = align_lyrics_to_notes(
            proj_lyr,
            raw_lyrics="春风 秋雨 冬雪",
            word_timestamps=[
                {"word": "春", "start": 4.0, "end": 4.4, "probability": 0.95},
                {"word": "风", "start": 4.5, "end": 5.4, "probability": 0.95},
                {"word": "秋", "start": 5.4, "end": 5.4, "probability": 0.0, "interpolated": True, "unsung": True},
                {"word": "雨", "start": 5.4, "end": 5.4, "probability": 0.0, "interpolated": True, "unsung": True},
                {"word": "冬", "start": 5.4, "end": 5.4, "probability": 0.0, "interpolated": True, "unsung": True},
                {"word": "雪", "start": 5.4, "end": 5.4, "probability": 0.0, "interpolated": True, "unsung": True},
            ],
        )
        self.assertEqual(al_out["note_lyrics"], ["", "春", "风", "-"])

        # Deliverables 4 & 5: B1 final-bar sustained roll + B2 motif CM non-collision
        # 6-bar piece: Bars 0..1 silent Intro, Bar 2 sung phrase, Bar 3 single silent breath bar, Bar 4 sung, Bar 5 silent Outro/Final
        notes_b12 = [
            Note(start=8.0, duration=1.0, pitch=64, raw_start=4.0, raw_duration=0.5),
            Note(start=9.0, duration=1.0, pitch=67, raw_start=4.5, raw_duration=0.5),
            Note(start=10.0, duration=1.5, pitch=69, raw_start=5.0, raw_duration=0.75),
            Note(start=16.0, duration=1.0, pitch=67, raw_start=8.0, raw_duration=0.5),
            Note(start=17.0, duration=2.0, pitch=64, raw_start=8.5, raw_duration=1.0),
        ]
        proj_b12 = Project(title="b12", tonic="C", meter="4/4", bpm=96.0, notes=notes_b12)
        arr_b12 = arrange_advanced(
            proj_b12,
            chords=["C", "G/B", "Am7", "Fmaj7", "G7", "Cmaj7"],
            style="指弹律动",
            slap_x=True,
            embellish_hp=True,
            passing_bass=True,
            intro_mode="chord_melody",
        )
        # Intro bars 0..1 have CM notes, and NO same-string overlap between CM and accompaniment
        intro_cm = [e for e in arr_b12["events"] if e["bar"] in (0, 1) and e.get("technique") == "CM"]
        self.assertGreaterEqual(len(intro_cm), 2)
        by_str: dict[int, list[dict]] = {}
        for ev in arr_b12["events"]:
            if ev.get("technique") != "X":
                by_str.setdefault(ev["string"], []).append(ev)
        for s_num, s_evs in by_str.items():
            s_evs_sorted = sorted(s_evs, key=lambda x: x["start"])
            for i_e in range(len(s_evs_sorted) - 1):
                self.assertLessEqual(
                    s_evs_sorted[i_e]["start"] + s_evs_sorted[i_e]["duration"],
                    s_evs_sorted[i_e + 1]["start"] + 1e-6,
                )
        # Single interior silent Bar 3 is a B1 breath bar (no scale-drill CM)
        bar3_cm = [e for e in arr_b12["events"] if e["bar"] == 3 and e.get("technique") == "CM"]
        self.assertEqual(len(bar3_cm), 0)
        # Final Bar 5 is strictly one rolled chord at beat 0 of duration = measure (4.0), with roll_step_sec and no slaps
        bar5_evs = [e for e in arr_b12["events"] if e["bar"] == 5]
        self.assertTrue(bar5_evs)
        self.assertTrue(all(abs(e["beat"]) < 1e-9 and abs(e["duration"] - 4.0) < 1e-6 for e in bar5_evs))
        self.assertTrue(all("roll_step_sec" in e for e in bar5_evs))
        self.assertTrue(all(e.get("technique") != "X" for e in bar5_evs))

        # Deliverable 6: recommend_artistry pure musical discrimination
        rec_waltz = recommend_artistry(
            Project(title="x", tonic="G", meter="3/4", bpm=96.0, notes=[
                Note(start=0.0, duration=2.0, pitch=64),
                Note(start=2.0, duration=1.0, pitch=67),
                Note(start=3.0, duration=3.0, pitch=69),
            ]),
            ["Em", "D", "C", "Em", "G", "D", "Em", "Em"],
        )
        self.assertEqual(rec_waltz["preset"], "original")
        rec_rnb = recommend_artistry(
            Project(title="y", tonic="F", meter="4/4", bpm=90.0, notes=[
                Note(start=0.25, duration=0.75, pitch=65),
                Note(start=1.75, duration=0.75, pitch=67),
                Note(start=2.75, duration=0.75, pitch=69),
                Note(start=4.25, duration=0.75, pitch=65),
            ]),
            ["Fmaj7", "E7", "Am7", "Gm7", "C9", "Bbm6", "Am7", "D7"],
        )
        self.assertEqual(rec_rnb["preset"], "neosoul_jazz")
        self.assertTrue(rec_rnb["reason"])

        # Gate 4 counterexamples: R1/R2 (all-N.C. & '0/4'/'Am'), T1 (sub-tick clamp), L1 (edited unsung tail), G1 (CM ceiling & 1-bar intro)
        from types import SimpleNamespace
        from subpkgs.score_web._score_timing import build_score_timeline
        rec_nc = recommend_artistry(
            Project(title="nc_waltz", tonic="C", meter="3/4", bpm=96.0, notes=[
                Note(start=0.0, duration=2.0, pitch=69),
                Note(start=2.0, duration=1.0, pitch=72),
                Note(start=3.0, duration=3.0, pitch=76),
            ]),
            ["N.C."] * 8,
        )
        self.assertEqual(rec_nc["preset"], "original")
        self.assertGreater(max(rec_nc["scores"].values()) - min(rec_nc["scores"].values()), 0.20)
        rec_edge = recommend_artistry(
            SimpleNamespace(meter="0/4", tonic="Am", bpm=100.0, notes=[Note(start=0.0, duration=1.0, pitch=69)]),
            ["Am"],
        )
        self.assertIn(rec_edge["preset"], ("folk_easy", "original", "emotional_pop", "neosoul_jazz"))

        reg_subtick = regulate_singing_notes_for_score(
            [Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.5),
             Note(pitch=62, start=4.0, duration=1.0, raw_start=2.0, raw_duration=0.5)],
            bpm=120.0, beat_origin=0.0, clip_duration=2.004,
        )
        p_subtick = Project(title="st", tonic="C", meter="4/4", bpm=120.0, notes=reg_subtick)
        tl_subtick = build_score_timeline(p_subtick, arrange_advanced(p_subtick, chords=["C", "G"]))
        self.assertGreaterEqual(len(tl_subtick), 1)

        proj_l1 = Project(title="l1", tonic="C", meter="4/4", bpm=120.0, beat_origin=0.0, source="v.wav", notes=[
            Note(start=0.0, duration=1.0, pitch=60, raw_start=0.0, raw_duration=0.45),
            Note(start=1.0, duration=1.0, pitch=62, raw_start=0.5, raw_duration=0.45),
            Note(start=2.0, duration=1.0, pitch=64, raw_start=1.0, raw_duration=0.45),
            Note(start=3.0, duration=1.0, pitch=65, raw_start=1.5, raw_duration=0.45),
        ])
        wt_l1 = [
            {"word": "one", "start": 0.0, "end": 0.45, "probability": 0.95},
            {"word": "two", "start": 0.5, "end": 1.95, "probability": 0.95},
            {"word": "three", "start": 1.95, "end": 1.95, "probability": 0.0, "interpolated": True, "unsung": True},
            {"word": "four", "start": 1.95, "end": 1.95, "probability": 0.0, "interpolated": True, "unsung": True},
        ]
        al_l1 = align_lyrics_to_notes(proj_l1, raw_lyrics="one two three five", word_timestamps=wt_l1)
        self.assertEqual(al_l1["note_lyrics"], ["one", "two", "-", "-"])

        p_1bar = Project(title="1b", tonic="C", meter="4/4", bpm=100.0, notes=[
            Note(pitch=64, start=4.0, duration=1.0, raw_start=2.4, raw_duration=0.6),
            Note(pitch=67, start=5.0, duration=3.0, raw_start=3.0, raw_duration=1.8),
        ])
        arr_1bar = arrange_advanced(
            p_1bar, chords=["C", "G", "C"], style="抒情琶音",
            intro_mode="chord_melody", passing_bass=True, slap_x=True, embellish_hp=True,
        )
        b0_evs = [e for e in arr_1bar["events"] if e["bar"] == 0]
        self.assertTrue(any(e.get("technique") == "CM" for e in b0_evs))
        self.assertTrue(any(abs(e["beat"]) < 1e-9 and "roll_step_sec" in e for e in b0_evs))
        self.assertTrue(any(e["beat"] > 1e-6 and e.get("technique", "") == "" for e in b0_evs))
        self.assertTrue(all(e.get("technique") not in ("PB", "X", "H", "P") for e in b0_evs))
        self.assertIn("function cmIntroMode(", idx_html)

        # lkktest Issue 3: custom_note_lyrics keyed by stable ref_idx survives note reorder + BPM resync
        with test_app.state.session_lock:
            s_ref = test_app.state.session
            s_ref.project = Project(
                title="ref_lyric_bpm", tonic="C", meter="4/4", bpm=120.0, beat_origin=0.0,
                clip_duration=4.0, source="dummy_rec.wav",
                notes=[
                    Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.4, edited=False),
                    Note(pitch=64, start=1.0, duration=1.0, raw_start=0.5, raw_duration=0.4, edited=False),
                ],
                reference_notes=[
                    Note(pitch=60, start=0.0, duration=1.0, raw_start=0.0, raw_duration=0.4),
                    Note(pitch=64, start=1.0, duration=1.0, raw_start=0.5, raw_duration=0.4),
                ],
            )
            s_ref.raw_lyrics = "one two"
            s_ref.word_timestamps = [
                {"word": "one", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"word": "two", "start": 0.5, "end": 0.9, "probability": 0.9},
            ]
            s_ref.note_lyrics = ["one", "two"]
            s_ref.reference_note_lyrics = ["one", "two"]
            s_ref.note_ref_indices = [0, 1]
            s_ref.custom_note_lyrics = {}
            s_ref.base_chords = ["C"]
            s_ref.custom_voicings = [None]
            srv._rebuild(s_ref)
        # Set custom lyric 'alpha' on note 0 (ref_idx=0), then move it after note 1 (start=2.5)
        self.assertEqual(client.post("/api/edit_note", json={"index": 0, "lyric": "alpha"}).status_code, 200)
        self.assertEqual(client.post("/api/edit_note", json={"index": 0, "start": 2.5, "duration": 1.0}).status_code, 200)
        r_bpm2 = client.post("/api/reharmonize", json={"bpm": 100.0})
        self.assertEqual(r_bpm2.status_code, 200)
        tbl2 = r_bpm2.json()["notes_table"]
        self.assertEqual(tbl2[1]["lyric"], "alpha")
        self.assertNotEqual(tbl2[0]["lyric"], "alpha")

        # lkktest round-2: restore_reference after BPM change keeps reference_notes aligned to new BPM
        r_rest = client.post("/api/restore_reference")
        self.assertEqual(r_rest.status_code, 200)
        rest_tbl = r_rest.json()["notes_table"]
        # Note 1 had raw_start=0.5s -> at 100 BPM (0.833 beats), start snaps near 0.75..1.0 (not stale 120-BPM beat)
        self.assertLess(rest_tbl[1]["start"], 0.90)

        # Anti-phase stereo accompaniment preserves Parseval energy in _has_audible_accompaniment
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td_st:
            st_wav = Path(td_st) / "antiphase_acc.wav"
            t_vec = np.arange(0, 22050, dtype=np.float64) / 22050.0
            sig = (0.10 * np.sin(2.0 * np.pi * 440.0 * t_vec)).astype(np.float32)
            stereo_ap = np.stack([sig, -sig], axis=1)
            sf.write(str(st_wav), stereo_ap, 22050)
            self.assertTrue(srv._has_audible_accompaniment(st_wav))

    def test_compound_68_arpeggio_baritone_reharm_and_diatonic_pb_invariants(self):
        # 1. Baritone E3 (MIDI 52) & A3 (MIDI 57) over G major in D major must be consonant tension, NOT clash
        baritone_e3 = [Note(start=0.0, duration=2.0, pitch=52, raw_start=0.0, raw_duration=1.2)]
        rep_g_e3 = check_melody_clash("G", baritone_e3, 0.0, 3.0, "D")
        self.assertNotEqual(rep_g_e3["status"], "clash")
        rep_gadd9_e3 = check_melody_clash("Gadd9", baritone_e3, 0.0, 3.0, "D")
        self.assertNotEqual(rep_gadd9_e3["status"], "clash")

        # G6 voicing must keep treble strings open/sounding rather than muting both Strings 1 & 2
        g6_v = generate_voicings("G6", capo=0)[0]["frets"]
        self.assertFalse(g6_v[4] < 0 and g6_v[5] < 0)

        # 2. Tonic D pedal with root D3/D4 vocal notes in emotional_pop avoids 3x Dmaj7 spam and G6 fallback
        notes_d_folk = [
            Note(start=12.0, duration=2.5, pitch=50, raw_start=8.0, raw_duration=1.6),
            Note(start=15.0, duration=1.0, pitch=54, raw_start=10.0, raw_duration=0.6),
            Note(start=17.5, duration=0.5, pitch=55, raw_start=11.6, raw_duration=0.3),
            Note(start=18.0, duration=2.5, pitch=52, raw_start=12.0, raw_duration=1.6),
        ]
        proj_d = Project(title="folk68", tonic="D", meter="6/8", bpm=90.0, notes=notes_d_folk)
        reh = reharmonize_progression(
            ["D", "Bm7", "A7sus4", "D", "D", "F#m7", "G", "A"],
            proj_d,
            preset="emotional_pop",
            tonic="D",
            capo=2,
        )
        reh_syms = [b["chord"] for b in reh]
        self.assertNotIn("G6", reh_syms)
        self.assertNotIn("Dmaj7", [reh_syms[4]])  # Bar 4 sings root D3 -> D or Dadd9, not Dmaj7
        for idx_r in range(len(reh_syms) - 2):
            self.assertFalse(reh_syms[idx_r] == reh_syms[idx_r + 1] == reh_syms[idx_r + 2] == "Dmaj7")

        # 3. 6/8 抒情琶音 follows P-i-m-a-m-i (Strings 3 -> 2 -> 1 -> 2 -> 3), keeps CM second-half continuity,
        #    and never emits out-of-key C-natural (pc 0) passing bass on G -> D in D major
        arr_d = arrange_advanced(
            proj_d,
            chords=["D", "Bm7", "A7sus4", "D", "D", "F#m7", "G", "D"],
            style="抒情琶音",
            capo=2,
            passing_bass=True,
            embellish_hp=True,
            intro_mode="chord_melody",
        )
        # Bar 5 (F#m7, vocal ends at beat 1.0 -> vocal_gap_here on beats 1.0..2.5) must traverse String 3, 2, and 1
        b5_treble = [
            e["string"]
            for e in sorted(arr_d["events"], key=lambda x: x["beat"])
            if e["bar"] == 5 and e["beat"] > 0.0 and e.get("technique", "") == ""
        ]
        self.assertEqual(b5_treble[:4], [3, 2, 1, 2])

        # Multi-bar 6/8 intro Bar 1 (Bm7, CM) has accompaniment/melody activity in second half (beat >= 1.5)
        b1_late = [e for e in arr_d["events"] if e["bar"] == 1 and e["beat"] >= 1.99]
        self.assertGreaterEqual(len(b1_late), 1)

        # Passing bass in D major never plays C natural (pc == 0) and never fires on static D -> D (Bar 3 -> Bar 4)
        pb_evs = [e for e in arr_d["events"] if e.get("technique") == "PB"]
        self.assertTrue(all((e["pitch"] % 12) != 0 for e in pb_evs))
        self.assertFalse(any(e["bar"] == 3 and e.get("technique") == "PB" for e in arr_d["events"]))

        # 4. Secondary fill_beat (4/4 non-arpeggio with vnc<=1) enforces H/P refractory cooldown (no back-to-back bars)
        proj_c44 = Project(
            title="c44_fill",
            tonic="C",
            meter="4/4",
            bpm=100.0,
            notes=[Note(start=4.0 * b, duration=1.5, pitch=64, raw_start=2.4 * b, raw_duration=0.9) for b in range(5)],
        )
        arr_c44 = arrange_advanced(
            proj_c44,
            chords=["Cadd9"] * 5,
            style="指弹律动",
            passing_bass=True,
            embellish_hp=True,
        )
        hp_bars_44 = sorted({e["bar"] for e in arr_c44["events"] if e.get("technique") in ("H", "P")})
        for idx_hp in range(len(hp_bars_44) - 1):
            self.assertGreaterEqual(hp_bars_44[idx_hp + 1] - hp_bars_44[idx_hp], 2)

        # 5. Late/out-of-bounds vocal_onsets_sec with pre_vocal_bars=0 still keeps beat_origin < duration & non-empty downbeats
        with tempfile.TemporaryDirectory() as td_bt:
            bt_wav = Path(td_bt) / "click_8s.wav"
            sr_bt = 22050
            dur_bt = 8.0
            y_bt = np.zeros(int(sr_bt * dur_bt), dtype=np.float32)
            for click_t in np.arange(0.0, dur_bt, 0.5):
                idx0 = int(round(click_t * sr_bt))
                idx1 = min(len(y_bt), idx0 + int(0.02 * sr_bt))
                y_bt[idx0:idx1] = 0.4 * np.sin(2.0 * np.pi * 880.0 * (np.arange(idx1 - idx0) / sr_bt)).astype(np.float32)
            sf.write(str(bt_wav), y_bt, sr_bt)
            bt_res = track_beats_and_downbeats(
                bt_wav,
                bpm_range=(90.0, 150.0),
                prefer_cuda=False,
                meter_hint="4/4",
                vocal_onsets_sec=[7.95, 8.50, float("nan")],
                pre_vocal_bars=0,
            )
            self.assertLess(bt_res["beat_origin"], dur_bt)
            self.assertGreaterEqual(len(bt_res["downbeats"]), 1)

    def test_singing_gap_solo_turnarounds_unvoiced_trim_and_cm_vibrato(self):
        from singing_model import _trim_and_split_voiced_intervals
        from subpkgs.guitar_reharm.guitar_simulator import synthesize_realistic_guitar

        # 1. Zombie phrase-ending silence trimming & breath-gap splitting in singing_model
        sr_sm, hop_sm = 24000, 128
        n_frames = round(4.0 * sr_sm / hop_sm)
        uv = np.ones(n_frames, dtype=bool)
        # Voiced 0.0..1.2s, then 2.0s pure breath silence (1.2..3.2s), then voiced 3.2..3.8s inside one unsplit interval
        uv[0 : round(1.2 * sr_sm / hop_sm)] = False
        uv[round(3.2 * sr_sm / hop_sm) : round(3.8 * sr_sm / hop_sm)] = False
        probs = np.zeros((1, 128), dtype=np.float32)
        probs[0, 62] = 0.92
        trimmed = _trim_and_split_voiced_intervals(
            pitches=np.array([62]),
            intervals=np.array([[0, n_frames]]),
            uv=uv,
            probability=probs,
            sr=sr_sm,
            hop=hop_sm,
            wav_len=4 * sr_sm,
        )
        self.assertEqual(len(trimmed), 2)
        self.assertLessEqual(trimmed[0].raw_duration, 1.26)
        self.assertGreaterEqual(trimmed[1].raw_start, 3.19)
        self.assertLessEqual(trimmed[1].raw_duration, 0.66)

        # 2. Post-verse turnaround with partial-bar vocal tail & pickup + multi-bar 6/8 interlude
        # Verse 1 spans Bars 0..3 (12 beats in 6/8), holds 0.4b into Bar 4 (ends at 12.4),
        # Verse 2 enters with a 0.4b pickup at end of Bar 5 (starts at 17.6 -> gap = 5.2b = 1.73 bars),
        # Verse 2 sings Bars 6..9 (18.0..29.5), then 4-bar interlude Bars 10..13 (30.0..42.0), Verse 3 at Bar 14, Outro Bar 15.
        notes_gap = [
            Note(start=0.0, duration=2.5, pitch=54, raw_start=0.0, raw_duration=1.6),
            Note(start=3.0, duration=2.5, pitch=57, raw_start=2.0, raw_duration=1.6),
            Note(start=6.0, duration=2.5, pitch=59, raw_start=4.0, raw_duration=1.6),
            Note(start=9.0, duration=3.4, pitch=57, raw_start=6.0, raw_duration=2.2),  # ends at 12.4 (0.4b into Bar 4)
            Note(start=17.6, duration=2.9, pitch=54, raw_start=11.7, raw_duration=1.9),  # 0.4b pickup at end of Bar 5
            Note(start=21.0, duration=2.5, pitch=57, raw_start=14.0, raw_duration=1.6),
            Note(start=24.0, duration=2.5, pitch=62, raw_start=16.0, raw_duration=1.6),
            Note(start=27.0, duration=2.5, pitch=59, raw_start=18.0, raw_duration=1.6),
            Note(start=42.0, duration=2.5, pitch=54, raw_start=28.0, raw_duration=1.6),
        ]
        chords_gap = [
            "D", "Bm7", "G", "A7sus4",
            "D", "A/C#",  # Turnaround Bars 4..5 (partial boundary bars!)
            "Bm7", "F#m7", "G", "A",
            "G", "A", "F#m7", "Bm7",  # 4-bar Interlude Bars 10..13
            "Em7", "D",
        ]
        proj_gap = Project(title="gap_solo", tonic="D", meter="6/8", bpm=90.0, notes=notes_gap)
        arr_gap = arrange_advanced(
            proj_gap,
            chords=chords_gap,
            style="抒情琶音",
            capo=2,
            passing_bass=True,
            embellish_hp=True,
            intro_mode="chord_melody",
            bar_dynamics=[0.76] * len(chords_gap),
        )
        # Turnaround Bars 4..5 MUST emit lyrical CM notes despite boundary vocal tail/pickup & 0.76 bar_dynamics, with zero vocal overlap
        turn_cm = [e for e in arr_gap["events"] if e["bar"] in (4, 5) and e.get("technique") == "CM"]
        self.assertGreaterEqual(len(turn_cm), 4)
        for ev in turn_cm:
            ev_s, ev_e = ev["start"], ev["start"] + ev["duration"]
            self.assertGreaterEqual(ev["pitch"], 64)
            self.assertGreaterEqual(ev["velocity"], 84)
            for vn in notes_gap:
                self.assertTrue(ev_e <= vn.start + 1e-6 or ev_s >= vn.start + vn.duration - 1e-6)

        # Interlude Bars 10..13: every bar has >= 3 lyrical CM notes, upper singing register, foreground velocity, and no 3x same pitch
        inter_pitches = []
        for b_int in (10, 11, 12, 13):
            b_cm = sorted(
                [e for e in arr_gap["events"] if e["bar"] == b_int and e.get("technique") == "CM"],
                key=lambda x: x["beat"],
            )
            self.assertGreaterEqual(len(b_cm), 3)
            self.assertTrue(all(e["pitch"] >= 64 for e in b_cm))
            b_inner = [
                e for e in arr_gap["events"]
                if e["bar"] == b_int and e.get("technique", "") == "" and e["string"] <= 4 and e.get("roll_step_sec") is None
            ]
            if b_inner:
                self.assertGreater(min(e["velocity"] for e in b_cm), max(e["velocity"] for e in b_inner))
            inter_pitches.extend(e["pitch"] for e in b_cm)
        for i_p in range(len(inter_pitches) - 2):
            self.assertFalse(inter_pitches[i_p] == inter_pitches[i_p + 1] == inter_pitches[i_p + 2])

        # 3. Extracted vocal motifs (m_head & m_tail) genuinely shape interlude solo rhythm & contour
        # Song A: recurring quarter-note stepwise motif [0, +1, +1] on beats [0, 1, 2]
        notes_motif_a = [
            Note(start=0.0, duration=1.0, pitch=60), Note(start=1.0, duration=1.0, pitch=62), Note(start=2.0, duration=1.5, pitch=64),
            Note(start=4.0, duration=1.0, pitch=60), Note(start=5.0, duration=1.0, pitch=62), Note(start=6.0, duration=1.5, pitch=64),
            Note(start=20.0, duration=2.0, pitch=60),
        ]
        # Song B: recurring syncopated fourth-leap motif [0, +3, -2] on beats [0, 1.5, 2.5]
        notes_motif_b = [
            Note(start=0.0, duration=1.5, pitch=60), Note(start=1.5, duration=1.0, pitch=65), Note(start=2.5, duration=1.0, pitch=62),
            Note(start=4.0, duration=1.5, pitch=60), Note(start=5.5, duration=1.0, pitch=65), Note(start=6.5, duration=1.0, pitch=62),
            Note(start=20.0, duration=2.0, pitch=62),
        ]
        chords_ab = ["C", "G", "Am7", "F", "G", "C", "C"]
        arr_ma = arrange_advanced(Project(title="ma", tonic="C", meter="4/4", bpm=96.0, notes=notes_motif_a), chords=chords_ab, intro_mode="chord_melody")
        arr_mb = arrange_advanced(Project(title="mb", tonic="C", meter="4/4", bpm=96.0, notes=notes_motif_b), chords=chords_ab, intro_mode="chord_melody")
        cm_a_b2 = [(round(e["beat"], 2), e["pitch"]) for e in arr_ma["events"] if e["bar"] == 2 and e.get("technique") == "CM"]
        cm_b_b2 = [(round(e["beat"], 2), e["pitch"]) for e in arr_mb["events"] if e["bar"] == 2 and e.get("technique") == "CM"]
        self.assertNotEqual(cm_a_b2, cm_b_b2)

        # 4. Sustained CM solo synthesis produces finite, clean stereo audio with delayed wrist vibrato
        audio_solo = synthesize_realistic_guitar(proj_gap, arr_gap, sr=22050)
        self.assertEqual(audio_solo.ndim, 2)
        self.assertEqual(audio_solo.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(audio_solo)))
        self.assertGreater(float(np.max(np.abs(audio_solo))), 0.05)

    def test_route1_studio_rr_acoustics_and_route3_btc_melodyt5_ergonomic_solos(self):
        import os
        import sys
        from fastapi.testclient import TestClient
        from subpkgs.guitar_reharm.chord_voicing import _assign_fingers, _fingers
        from subpkgs.guitar_reharm.guitar_simulator import (
            _ensure_sample_index,
            _fetch_studio_guitar_note,
            _SHINY_PITCHED,
            synthesize_realistic_guitar,
        )
        from subpkgs.guitar_reharm.melodyt5_bridge import (
            abc_to_notes,
            generate_melodyt5_solo_cells,
            notes_to_abc,
        )
        from subpkgs.score_web import server as srv
        from subpkgs.score_web.alphatex_builder import to_alphatex
        from subpkgs.score_web.yoopu_svg_renderer import render_publication_score_html

        # 1. Route 1: Studio Multi-RR anti-machine-gun non-identical waveforms & formant-preserving slide synthesis
        _ensure_sample_index()
        if _SHINY_PITCHED:
            rr_waves = [
                _fetch_studio_guitar_note(64, 1, 11025, 22050, 92.0, "CM", seed=100 + r, rr_idx=r)
                for r in (1, 2, 3, 4)
            ]
            for w in rr_waves:
                self.assertIsNotNone(w)
                self.assertTrue(np.all(np.isfinite(w)))
            # Distinct studio Round-Robin takes must NOT be scalar multiples of one another
            c12 = float(np.corrcoef(rr_waves[0][:4096], rr_waves[1][:4096])[0, 1])
            self.assertLess(abs(c12), 0.985)

            sl_up = _fetch_studio_guitar_note(66, 1, 11025, 22050, 94.0, "CM", seed=202, rr_idx=1, slide_delta=2)
            sl_dn = _fetch_studio_guitar_note(64, 1, 11025, 22050, 94.0, "CM", seed=203, rr_idx=2, slide_delta=-2)
            self.assertIsNotNone(sl_up)
            self.assertIsNotNone(sl_dn)
            self.assertTrue(np.all(np.isfinite(sl_up)))
            self.assertTrue(np.all(np.isfinite(sl_dn)))

        # 2. Route 1: Simultaneous 'CM' + 'DS' Double-Stop dyad sustains full duration without premature self-choke
        proj_ds = Project(title="ds_sustain", tonic="D", meter="4/4", bpm=60.0, notes=[])
        arr_ds = {
            "title": "ds_sustain",
            "tonic": "D",
            "meter": "4/4",
            "bpm": 60.0,
            "beat_origin": 0.0,
            "style": "指弹律动",
            "capo": 0,
            "bar_count": 1,
            "measure_beats": 4.0,
            "chords": ["D"],
            "bars": [{
                "bar": 0,
                "chord": "D",
                "roman": "I",
                "frets": [-1, -1, 0, 2, 3, 2],
                "split_segments": [{"beat": 0.0, "duration": 4.0, "chord": "D", "roman": "I", "frets": [-1, -1, 0, 2, 3, 2], "capo": 0}],
            }],
            "events": [
                {"bar": 0, "beat": 0.0, "start": 0.0, "duration": 2.5, "string": 4, "fret": 0, "pitch": 50, "velocity": 78, "technique": "", "chord": "D", "segment": 0},
                {"bar": 0, "beat": 0.0, "start": 0.0, "duration": 2.5, "string": 1, "fret": 5, "pitch": 69, "velocity": 94, "technique": "CM", "slide_from_fret": 3, "chord": "D", "segment": 0},
                {"bar": 0, "beat": 0.0, "start": 0.0, "duration": 2.5, "string": 2, "fret": 7, "pitch": 66, "velocity": 85, "technique": "DS", "slide_from_fret": 5, "chord": "D", "segment": 0},
            ],
        }
        wav_ds = synthesize_realistic_guitar(proj_ds, arr_ds, sr=22050)
        self.assertTrue(np.all(np.isfinite(wav_ds)))
        # Verify the Double-Stop + bass cluster is still audibly ringing at t = 0.8s..1.4s (not choked at 12ms)
        late_rms = float(np.sqrt(np.mean(wav_ds[int(0.8 * 22050) : int(1.4 * 22050)] ** 2)))
        self.assertGreater(late_rms, 0.008)

        # 3. Route 1 + Score Sync: AlphaTex emits {sib}/{sia} without breaking ties, SVG renders diagonal slide 'S'
        tex_ds = to_alphatex(proj_ds, arr_ds)
        self.assertIn("5.1 {sib}", tex_ds)
        self.assertIn("7.2 {sib}", tex_ds)
        svg_ds = render_publication_score_html(proj_ds, arr_ds)
        self.assertIn(">S<", svg_ds)

        # 4. Route 3: MelodyT5 ABC round-trip & hermetic variation bridge (zero cwd/sys.path mutation)
        cwd_before = os.getcwd()
        syspath_before = list(sys.path)
        for key_rt, meter_rt in (("D", "6/8"), ("F#m", "4/4"), ("Bb", "4/4"), ("Dm", "3/4")):
            sample_rt = [
                {"start": 0.0, "duration": 1.0, "pitch": 62},
                {"start": 1.0, "duration": 0.5, "pitch": 65},
                {"start": 1.5, "duration": 1.5, "pitch": 69},
            ]
            abc_s = notes_to_abc(sample_rt, meter_rt, key_rt)
            self.assertEqual(abc_to_notes(abc_s, meter_rt, key_rt), sample_rt)

        vocal_16 = [
            Note(start=12.0 + k * 0.75, duration=0.5, pitch=p)
            for k, p in enumerate((62, 64, 66, 69, 66, 64, 62, 59, 62, 64, 66, 71, 69, 66, 64, 62))
        ]
        mt5_out = generate_melodyt5_solo_cells(vocal_16, "6/8", "D", False, num_bars=4)
        self.assertEqual(os.getcwd(), cwd_before)
        self.assertEqual(sys.path, syspath_before)
        self.assertIsInstance(mt5_out, list)

        # 5. Route 3: Joint Fingerboard Solo Viterbi emits warm Double-Stops ('DS') + Slides ('slide_from_fret')
        #    while strictly satisfying 4-finger playability (overflow == 0, span <= 4) with concurrent bass
        #    and zero same-string overlap across all 6 strings!
        proj_solo68 = Project(
            title="solo68_ergonomic",
            tonic="D",
            meter="6/8",
            bpm=92.0,
            notes=vocal_16 + [Note(start=42.0, duration=2.0, pitch=62)],
        )
        chords_solo68 = [
            "D", "Bm7", "G", "A7sus4",   # 4-bar Intro (Bars 0..3)
            "D", "F#m7", "G", "A", "D",  # Sung verse (Bars 4..8)
            "G", "A", "F#m7", "Bm7", "Em7",  # 5-bar Interlude (Bars 9..13)
            "D", "D",
        ]
        arr_solo68 = arrange_advanced(
            proj_solo68,
            chords=chords_solo68,
            style="抒情琶音",
            capo=2,
            passing_bass=True,
            embellish_hp=True,
            intro_mode="chord_melody",
        )
        evs_68 = arr_solo68["events"]
        ds_evs = [e for e in evs_68 if e.get("technique") == "DS"]
        sl_evs = [e for e in evs_68 if e.get("slide_from_fret") is not None]
        self.assertGreaterEqual(len(ds_evs), 2)
        self.assertGreaterEqual(len(sl_evs), 1)

        # Check strict same-string non-overlap across all non-X events
        for s_num in range(1, 7):
            s_evs = sorted(
                [e for e in evs_68 if e["string"] == s_num and e.get("technique") != "X"],
                key=lambda e: e["start"],
            )
            for idx_e in range(len(s_evs) - 1):
                self.assertLessEqual(
                    s_evs[idx_e]["start"] + s_evs[idx_e]["duration"],
                    s_evs[idx_e + 1]["start"] + 1e-6,
                )

        # Check 4-finger left-hand playability on every solo onset (sustained bass + CM + DS, and slide start!)
        for cm_e in [e for e in evs_68 if e.get("technique") == "CM"]:
            t_on = cm_e["start"]
            concurrent = [
                e for e in evs_68
                if e.get("technique") != "X" and e["start"] - 1e-6 <= t_on < e["start"] + e["duration"] - 1e-6
            ]
            frets6 = [-1] * 6
            for ce in concurrent:
                if ce["string"] >= 4 or ce.get("technique") in ("CM", "DS"):
                    frets6[6 - ce["string"]] = int(ce["fret"])
            pos_f = [f for f in frets6 if f > 0]
            if pos_f:
                self.assertLessEqual(max(pos_f) - min(pos_f), 4)
            self.assertLessEqual(_fingers(frets6)[0], 4)
            self.assertEqual(_assign_fingers(tuple(frets6), 2)[1], 0)

        # 6. Route 3: detected_split_chords wires cleanly into reharmonize_progression & FastAPI /api/reharmonize
        reh_det = reharmonize_progression(
            ["D", "G", "D"],
            Project(title="det_split", tonic="D", meter="6/8", bpm=90.0, notes=[]),
            preset="emotional_pop",
            tonic="D",
            split_bar=False,
            capo=2,
            detected_split_chords={"1": ["G", "A"]},
        )
        self.assertEqual(len(reh_det[0]["split_segments"]), 1)
        self.assertEqual(len(reh_det[1]["split_segments"]), 2)
        self.assertAlmostEqual(reh_det[1]["split_segments"][0]["duration"], 1.5)
        self.assertAlmostEqual(reh_det[1]["split_segments"][1]["duration"], 1.5)

        app_det = srv.create_app()
        client_det = TestClient(app_det)
        with app_det.state.session_lock:
            s_det = app_det.state.session
            s_det.detected_split_chords = {"0": ["C", "G"]}
            s_det.use_detected_splits = True
            s_det.split_bar = False
            cur_style = s_det.style
            srv._rebuild(s_det)
            self.assertEqual(len(s_det.bar_analysis[0]["split_segments"]), 2)
        # Override segment 1 of detected split bar, then re-click preset card with split_bar=False:
        # must preserve detected splits AND segment override without 422!
        r_ov = client_det.post("/api/bar_chord", json={"bar": 0, "segment": 1, "chord": "G7"})
        self.assertEqual(r_ov.status_code, 200)
        r_preset_same = client_det.post(
            "/api/reharmonize",
            json={"preset": s_det.preset, "style": cur_style, "split_bar": False},
        )
        self.assertEqual(r_preset_same.status_code, 200)
        self.assertTrue(r_preset_same.json()["use_detected_splits"])
        self.assertEqual(len(r_preset_same.json()["bar_analysis"][0]["split_segments"]), 2)
        self.assertEqual(r_preset_same.json()["bar_analysis"][0]["split_segments"][1]["chord"], "G7")
        # Standalone split_bar=False collapses detected split bars cleanly without 422 even with prior 0:1 override
        r_off = client_det.post("/api/reharmonize", json={"split_bar": False})
        self.assertEqual(r_off.status_code, 200)
        self.assertFalse(r_off.json()["use_detected_splits"])
        self.assertEqual(len(r_off.json()["bar_analysis"][0]["split_segments"]), 1)
        r_mtr = client_det.post("/api/reharmonize", json={"meter": "3/4"})
        self.assertEqual(r_mtr.status_code, 200)
        with app_det.state.session_lock:
            self.assertEqual(app_det.state.session.detected_split_chords, {})

        # 7. Enharmonic sharp tonics (A#, D#, G#, A#m), 3/4 cross-segment sustained avoid-note guard, & bare 'SL' parity
        from subpkgs.guitar_reharm.melodyt5_bridge import canonical_abc_key, phrase_cells
        from subpkgs.guitar_reharm.arrangement_grooves import _adapt_cell_viterbi
        self.assertEqual(canonical_abc_key("A#"), "Bb")
        self.assertEqual(canonical_abc_key("D#"), "Eb")
        self.assertEqual(canonical_abc_key("G#"), "Ab")
        self.assertEqual(canonical_abc_key("A#m"), "Bbm")
        self.assertEqual(canonical_abc_key("Ddor"), "Ddor")
        self.assertEqual(canonical_abc_key("A#dor"), "Bbdor")
        from subpkgs.guitar_reharm.melodyt5_bridge import key_signature
        for mode_k in ("Ddor", "Gmix", "Flyd", "Ephr", "A#dor", "F#m", "A#"):
            self.assertEqual(key_signature(canonical_abc_key(mode_k)), key_signature(mode_k))
        cells_bb_dn = phrase_cells(
            [{"start": 0.0, "duration": 1.0, "pitch": 70}, {"start": 1.0, "duration": 1.0, "pitch": 69}],
            "A#",
            cumulative=False,
        )
        cells_bb_up = phrase_cells(
            [{"start": 0.0, "duration": 1.0, "pitch": 72}, {"start": 1.0, "duration": 1.0, "pitch": 74}],
            "A#",
            cumulative=False,
        )
        self.assertEqual(cells_bb_dn[1][2], -1)
        self.assertEqual(cells_bb_up[1][2], 1)

        # 3/4 split C (beats 0..2) | G (beats 2..3): note at beat 1.0 sustaining 2.0 beats overlaps G, so pitch C (pc=0) must be banned
        proj_34 = Project(title="split34", tonic="C", meter="3/4", bpm=96.0, notes=[])
        reh_34 = reharmonize_progression(
            ["C", "Am"],
            proj_34,
            preset="original",
            tonic="C",
            split_bar=False,
            capo=0,
            detected_split_chords={"0": ["C", "G"]},
        )
        arr_34_seed = arrange_advanced(
            proj_34,
            chords=[r["chord"] for r in reh_34],
            style="抒情琶音",
            capo=0,
            split_chords=[r["split_segments"] for r in reh_34],
            intro_mode="none",
        )
        bar0_34 = arr_34_seed["bars"][0]
        vit_34 = _adapt_cell_viterbi(
            bar=bar0_34,
            cell_specs=[(0.0, 1.0, 0), (1.0, 2.0, -2)],
            capo=0,
            tonic_pc=0,
            diatonic={0, 2, 4, 5, 7, 9, 11},
            depths=[0, 2, 1, 2, 1, 2],
            measure=3.0,
            role=3,
            allow_double_stops=False,
            allow_slides=False,
        )
        sustained_1 = [ev for ev in vit_34 if abs(ev["beat"] - 1.0) < 1e-6]
        self.assertTrue(sustained_1)
        for ev in sustained_1:
            self.assertNotEqual(ev["pitch"] % 12, 0, "Pitch C must not sustain across G major (minor-9th above B)")

        # Bare technique='SL' without slide_from_fret emits {sib} (not {sl} to a distant bar)
        proj_sl = Project(
            title="bare_sl",
            tonic="A",
            meter="4/4",
            bpm=96.0,
            notes=[Note(pitch=69, start=12.0, duration=1.0, raw_start=7.5, raw_duration=0.625)],
        )
        arr_sl = arrange_advanced(
            proj_sl,
            chords=["A", "E", "F#m", "D"],
            style="抒情琶音",
            capo=0,
            intro_mode="chord_melody",
            instrumental_melody={
                0: [{"beat": 0.0, "duration": 1.0, "pitch": 69, "string": 1, "technique": "SL", "velocity": 90}],
                3: [{"beat": 0.0, "duration": 1.0, "pitch": 73, "string": 1, "technique": "CM", "velocity": 90}],
            },
        )
        tex_sl = to_alphatex(proj_sl, arr_sl, [""])
        self.assertIn("5.1 {sib}", tex_sl)
        self.assertNotIn("5.1 {sl}", tex_sl)

        # 8. Anti-Boingy ('Q弹') Discrete Fretted Grace-Slide & Anti-Noodling ('xjb乱弹') Home-Position Intro Guard
        from subpkgs.guitar_reharm.guitar_simulator import _slide_head
        synth_base = np.sin(2.0 * np.pi * 440.0 * np.arange(22050, dtype=np.float64) / 22050.0).astype(np.float32)
        sl_head, sl_d_int = _slide_head(synth_base, 0, -2, 0.060, 22050)
        cont_idx = len(sl_head) + sl_d_int
        self.assertGreaterEqual(cont_idx, 0)
        # Verify sample-exact seam between end of _slide_head and base[cont_idx] (no step discontinuity)
        expected_step = abs(float(synth_base[cont_idx]) - float(synth_base[cont_idx - 1]))
        actual_step = abs(float(synth_base[cont_idx]) - float(sl_head[-1]))
        self.assertAlmostEqual(actual_step, expected_step, places=5)

        # Multi-bar intro with mid-bar vocal pickup in bar 4 (beat 14.5 in 6/8) must:
        # (a) leave bar 4 free of CM solo licks so singer's pickup is clean,
        # (b) keep Intro CM notes in warm home position (fret <= 6),
        # (c) never attach slide_from_fret to a DS dyad, and emit <= 1 slide across the 4-bar Intro,
        # (d) never machine-gun the exact same accompaniment note 3+ times in a row under a CM bar.
        proj_intro_pickup = Project(
            title="intro_pickup_68",
            tonic="D",
            meter="6/8",
            bpm=91.0,
            notes=[
                Note(start=14.5, duration=0.5, pitch=57),
                Note(start=15.0, duration=1.5, pitch=62),
                Note(start=16.5, duration=1.5, pitch=66),
            ],
        )
        arr_ip = arrange_advanced(
            proj_intro_pickup,
            chords=["Dadd9", "Bm7", "Gadd9", "Dadd9", "Dadd9", "Bm7"],
            style="抒情琶音",
            capo=2,
            intro_mode="chord_melody",
        )
        bar4_cm = [e for e in arr_ip["events"] if e["bar"] == 4 and e.get("technique") in ("CM", "DS")]
        self.assertEqual(bar4_cm, [])
        intro_cm_frets = [e["fret"] for e in arr_ip["events"] if e["bar"] < 4 and e.get("technique") == "CM"]
        self.assertTrue(intro_cm_frets)
        self.assertLessEqual(max(intro_cm_frets), 6)
        intro_ds_slides = [
            e for e in arr_ip["events"]
            if e.get("technique") == "DS" and e.get("slide_from_fret") is not None
        ]
        self.assertEqual(intro_ds_slides, [])
        intro_slides = [e for e in arr_ip["events"] if e["bar"] < 4 and e.get("slide_from_fret") is not None]
        self.assertLessEqual(len(intro_slides), 1)
        for b_i in range(4):
            acc_pitches = [
                e["pitch"] for e in sorted(arr_ip["events"], key=lambda x: (x["start"], x["string"]))
                if e["bar"] == b_i and e.get("technique", "") == "" and e["string"] <= 3
            ]
            for k_a in range(len(acc_pitches) - 2):
                self.assertFalse(
                    acc_pitches[k_a] == acc_pitches[k_a + 1] == acc_pitches[k_a + 2],
                    f"Machine-gun repeated inner accompaniment pitch in bar {b_i}: {acc_pitches}",
                )
        # Canonical Dadd9 voicing must be monotonic without high-fret unison duplication
        dadd9_v = generate_voicings("Dadd9", capo=0)[0]["frets"]
        self.assertEqual(dadd9_v, [-1, 5, 4, 2, 3, 0])

    def test_wooden_acoustic_timbre_slash_harmony_arpeggio_continuity_and_mms_star(self):
        import json
        from types import SimpleNamespace
        from subpkgs.guitar_reharm import guitar_simulator as gsim
        from subpkgs.guitar_reharm.reharmonizer import _resolve_harmony
        from subpkgs.stem_mir.beat_tracker import _bar_beat_positions
        from subpkgs.singing_lyrics.lyric_aligner import (
            detect_pre_lyric_vocalise_onsets,
            filter_pre_lyric_bleed_notes,
            _free_end_ctc_prefix_len,
        )

        # (a) Primary wooden acoustic sample index (_META_INDEX + _FSS_INDEX) & high-fret sub-150Hz rumble guard
        gsim._ensure_sample_index()
        self.assertGreater(len(gsim._META_INDEX), 300)
        self.assertGreater(len(gsim._FSS_INDEX), 40)
        hi_note = gsim._get_blended_studio_base(74, 1, 1, 22050, False, rr=1)
        self.assertIsNotNone(hi_note)
        mono_hi = hi_note.mean(axis=-1) if hi_note.ndim == 2 else hi_note
        spec = np.abs(np.fft.rfft(mono_hi[:8192]))
        freqs = np.fft.rfftfreq(len(mono_hi[:8192]), 1.0 / 22050.0)
        rumble_e = float(np.sum(spec[freqs < 150.0] ** 2))
        fund_e = float(np.sum(spec[(freqs >= 500.0) & (freqs <= 2500.0)] ** 2))
        self.assertLess(rumble_e, 0.02 * max(1e-9, fund_e))

        # (b) Explicit slash bass preservation + capo-consistent _resolve_harmony + borrowed iv + Phrygian iii
        proj_a = Project(title="slash_walk", tonic="A", meter="4/4", bpm=78.0, notes=[])
        slash_res = reharmonize_progression(
            ["A", "E/G#", "F#m", "A/E", "D/F#"],
            proj_a,
            preset="emotional_pop",
            tonic="A",
            capo=2,
        )
        self.assertEqual(parse_extended_chord(slash_res[1]["chord"])["bass"], "G#")
        self.assertEqual(parse_extended_chord(slash_res[3]["chord"])["bass"], "E")
        self.assertEqual(parse_extended_chord(slash_res[4]["chord"])["bass"], "F#")

        clash_notes = [Note(start=0.0, duration=4.0, pitch=61)]  # C# clashes with C-root triad/sus/7ths
        res_sym, _ = _resolve_harmony("C", None, "G", "D", clash_notes, 0.0, 4.0, capo=2)
        self.assertTrue(len(generate_voicings(res_sym, capo=2)) > 0)

        proj_f = Project(title="neosoul_f", tonic="F", meter="4/4", bpm=90.0, notes=[])
        neosoul_f = reharmonize_progression(
            ["F", "Bbm", "Am", "Dm"],
            proj_f,
            preset="neosoul_jazz",
            tonic="F",
            capo=0,
        )
        bbm_pcs = set(parse_extended_chord(neosoul_f[1]["chord"])["pcs"])
        am_pcs = set(parse_extended_chord(neosoul_f[2]["chord"])["pcs"])
        self.assertNotIn(8, bbm_pcs)   # No Ab (pc 8) clashing against tonic third A (pc 9)
        self.assertNotIn(11, am_pcs)   # No B natural (pc 11, #4 tritone) on Phrygian iii in F major

        # (c) Continuous 6/8 arpeggio under active vocals + multi-bar CM inner accompaniment + no b3 over major V
        proj_68 = Project(
            title="continuous_68",
            tonic="C",
            meter="6/8",
            bpm=68.0,
            notes=[
                Note(start=6.0, duration=1.0, pitch=64),
                Note(start=7.0, duration=0.5, pitch=67),
                Note(start=7.5, duration=1.5, pitch=64),
            ],
        )
        arr_68 = arrange_advanced(
            proj_68,
            chords=["C", "Am", "C", "F"],
            style="抒情琶音",
            capo=0,
            intro_mode="chord_melody",
        )
        b2_onsets = {round(e["start"] - 6.0, 2) for e in arr_68["events"] if e["bar"] == 2}
        self.assertEqual(b2_onsets, {0.0, 0.5, 1.0, 1.5, 2.0, 2.5})
        b1_cm = [e for e in arr_68["events"] if e["bar"] == 1 and e.get("technique") in ("CM", "DS")]
        b1_acc = [e for e in arr_68["events"] if e["bar"] == 1 and e.get("technique", "") == ""]
        self.assertTrue(len(b1_cm) >= 1)
        self.assertTrue(len(b1_acc) >= 2)

        proj_hm = Project(
            title="harmonic_minor_v",
            tonic="F",
            meter="6/8",
            bpm=92.0,
            notes=[Note(start=0.0, duration=1.5, pitch=69), Note(start=1.5, duration=1.5, pitch=73)],
        )
        arr_hm = arrange_advanced(
            proj_hm,
            chords=["A", "Dm"],
            style="抒情琶音",
            capo=0,
            intro_mode="groove",
        )
        bar0_pcs = {e["pitch"] % 12 for e in arr_hm["events"] if e["bar"] == 0}
        self.assertNotIn(0, bar0_pcs)  # Forbid C natural (b3/#9) over harmonic-minor V (A major)

        # (d) Beat jitter filter, pre-lyric vocalise vs bleed filter, free-end CTC <star>, and manifest health
        jitter_beats = np.array([0.85, 0.92, 1.85], dtype=np.float64)
        pos = _bar_beat_positions(jitter_beats, 0.0, 2.4, 4, is_compound=False)
        self.assertEqual(len(pos), 4)
        self.assertTrue(all(np.diff(pos) >= 0.35))

        # Free-end CTC prefix length absorbs leading <star> vocalise and drops trailing unsung words cleanly
        emissions = np.full((24, 5), -10.0, dtype=np.float32)
        emissions[:, 0] = 5.0          # CTC blank default
        emissions[0:6, 0] = -8.0       # Non-blank pre-lyric vocalise in frames 0..5 absorbed by <star> (id 1)
        emissions[0:6, 1] = 6.0
        emissions[8:11, 2] = 7.0       # Sung word 1 (token [2])
        emissions[13:16, 3] = 7.0      # Sung word 2 (token [3])
        kept_words = _free_end_ctc_prefix_len(
            emissions,
            [[2], [3], [4, 4]],        # Word 3 ([4, 4]) is unsung (-10.0 logits -> ~ -15 log-prob per char)
            blank_id=0,
            star_id=1,
        )
        self.assertEqual(kept_words, 2)

        with tempfile.TemporaryDirectory() as td:
            wav_p = Path(td) / "voc_test.wav"
            sr_t = 16000
            sig_t = np.zeros(sr_t * 12, dtype=np.float32)
            # Low-energy bleed at 0.3..0.7s, real sustained humming at 1.5..3.5s, lyric at 5.2..6.7s
            t_vec = np.arange(len(sig_t), dtype=np.float32) / sr_t
            sig_t[int(0.3 * sr_t) : int(0.7 * sr_t)] = 0.004 * np.sin(2 * np.pi * 220.0 * t_vec[: int(0.4 * sr_t)])
            sig_t[int(1.5 * sr_t) : int(3.5 * sr_t)] = 0.12 * np.sin(2 * np.pi * 330.0 * t_vec[: int(2.0 * sr_t)])
            sig_t[int(5.2 * sr_t) : int(6.7 * sr_t)] = 0.15 * np.sin(2 * np.pi * 392.0 * t_vec[: int(1.5 * sr_t)])
            sf.write(str(wav_p), sig_t, sr_t)

            voc_onsets = detect_pre_lyric_vocalise_onsets(wav_p, first_lyric_sec=5.2)
            self.assertEqual(len(voc_onsets), 1)
            self.assertAlmostEqual(voc_onsets[0], 1.5, delta=0.15)

            raw_ns = [
                SimpleNamespace(raw_start=0.3, raw_duration=0.4, confidence=0.8),
                SimpleNamespace(raw_start=1.6, raw_duration=1.8, confidence=0.9),
                SimpleNamespace(raw_start=5.2, raw_duration=1.2, confidence=0.95),
            ]
            kept_ns = filter_pre_lyric_bleed_notes(raw_ns, wav_p, first_lyric_sec=5.2, beat_origin=0.2, pickup_tol_sec=0.5)
            self.assertEqual([n.raw_start for n in kept_ns], [1.6, 5.2])

        manifest_p = ROOT_DIR / "data" / "repertoire_manifest.json"
        self.assertTrue(manifest_p.is_file())
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 7)
        items = manifest.values() if isinstance(manifest, dict) else manifest
        for item in items:
            self.assertFalse(item.get("split_bar", False), f"split_bar should be False on {item.get('id')}")
            bt = item.get("beat_times", [])
            if len(bt) >= 2:
                self.assertGreaterEqual(min(np.diff(bt)), 0.30, f"Sub-0.30s beat glitch in {item.get('id')}")

    def test_universal_anti_fable_generality_and_upload_parity(self):
        import time
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from subpkgs.guitar_reharm import ADVANCED_STYLES, select_ergonomic_capo
        from subpkgs.score_web import server as srv

        # 1. 1-bar intro accompaniment continuity across 4/4, 3/4, and 6/8 (no 2-second dead-air hole)
        for meter_s, meas_b in (("4/4", 4.0), ("3/4", 3.0), ("6/8", 3.0)):
            p_intro1 = Project(
                title=f"intro1_{meter_s}",
                tonic="C",
                meter=meter_s,
                bpm=88.0,
                notes=[
                    Note(pitch=64, start=meas_b, duration=1.0, raw_start=2.0, raw_duration=0.6),
                    Note(pitch=67, start=meas_b + 1.0, duration=1.5, raw_start=2.6, raw_duration=0.9),
                ],
            )
            arr_i1 = arrange_advanced(
                p_intro1,
                chords=["C", "G", "C"],
                style="抒情琶音",
                intro_mode="chord_melody",
            )
            b0_beats = sorted({round(float(e["beat"]), 3) for e in arr_i1["events"] if e["bar"] == 0})
            self.assertGreaterEqual(len(b0_beats), 4, f"1-bar intro in {meter_s} must fill inner accompaniment: {b0_beats}")
            gaps = [b0_beats[k + 1] - b0_beats[k] for k in range(len(b0_beats) - 1)] + [meas_b - b0_beats[-1]]
            self.assertLessEqual(max(gaps), 1.0 + 1e-6, f"Dead-air gap in {meter_s} 1-bar intro: {b0_beats}")

        # 2. Universal mid-song breathing across ALL 5 styles: phrase-endings & 1-bar vocal breaths keep pulse
        notes_breath = [
            Note(pitch=64, start=0.0, duration=2.0),
            Note(pitch=67, start=2.0, duration=2.0),
            Note(pitch=65, start=4.0, duration=1.5),   # Phrase 1 ends early in Bar 1 (V->I cadence G7->C)
            # Bar 2 (beats 8..12) is a 1-bar inter-phrase vocal breath (silent, no CM turnaround)
            Note(pitch=64, start=12.0, duration=2.0),  # Phrase 2 resumes in Bar 3
            Note(pitch=60, start=14.0, duration=2.0),
            # Bar 4 (beats 16..20) is the final bar -> full sustained rolled chord
        ]
        p_breath = Project(title="all_styles_breath", tonic="C", meter="4/4", bpm=92.0, notes=notes_breath)
        for st_name in ADVANCED_STYLES:
            arr_st = arrange_advanced(
                p_breath,
                chords=["C", "G7", "Am7", "F", "C"],
                style=st_name,
                intro_mode="chord_melody",
            )
            for mid_bar in (1, 2):
                mid_onsets = sorted({round(float(e["beat"]), 3) for e in arr_st["events"] if e["bar"] == mid_bar})
                self.assertGreaterEqual(
                    len(mid_onsets), 5,
                    f"Style {st_name} froze or starved mid-song bar {mid_bar}: onsets={mid_onsets}",
                )
            final_evs = [e for e in arr_st["events"] if e["bar"] == 4]
            self.assertTrue(final_evs and all(abs(e["beat"]) < 1e-9 and "roll_step_sec" in e for e in final_evs))

        # 3. Universal scale-consistent secondary dominants & harmonic-minor V across presets
        proj_am = Project(title="am_sec_dom", tonic="C", meter="4/4", bpm=90.0, notes=[])
        for preset_k in ("neosoul_jazz", "emotional_pop", "folk_easy", "walking_bass", "suspend_dreamy"):
            res_am = reharmonize_progression(
                ["Am", "Dm", "E", "Am", "E7", "Am"],
                proj_am,
                preset=preset_k,
                tonic="C",
                capo=0,
            )
            for b_dom in (2, 4):
                pcs_e = set(parse_extended_chord(res_am[b_dom]["chord"])["pcs"])
                self.assertNotIn(3, pcs_e, f"{preset_k} emitted D# (Emaj7) on V/vi in Am: {res_am[b_dom]['chord']}")
                self.assertNotIn(6, pcs_e, f"{preset_k} emitted F# (E9/add9/sus2) on V/vi in Am: {res_am[b_dom]['chord']}")
                self.assertNotIn(1, pcs_e, f"{preset_k} emitted C# (E6) on V/vi in Am: {res_am[b_dom]['chord']}")

        res_sec_c = reharmonize_progression(
            ["C", "A", "Dm", "D", "G", "Bb", "Ab", "C"],
            proj_am,
            preset="neosoul_jazz",
            tonic="C",
            capo=0,
        )
        self.assertNotIn(8, set(parse_extended_chord(res_sec_c[1]["chord"])["pcs"]))  # A (V/ii) -> no G# (Amaj7)
        self.assertNotIn(1, set(parse_extended_chord(res_sec_c[3]["chord"])["pcs"]))  # D (V/V) -> no C# (Dmaj7)
        self.assertIn(9, set(parse_extended_chord(res_sec_c[5]["chord"])["pcs"]))     # Bb (bVII) -> keeps diatonic A (Bbmaj7)
        self.assertIn(7, set(parse_extended_chord(res_sec_c[6]["chord"])["pcs"]))     # Ab (bVI) -> keeps diatonic G (Abmaj7)

        # 4. Ergonomic Capo selection across open vs barre-heavy keys
        self.assertEqual(select_ergonomic_capo(["C", "G/B", "Am7", "F"] * 4), 0)
        self.assertEqual(select_ergonomic_capo(["Em", "D", "C", "G"] * 4), 0)
        self.assertEqual(select_ergonomic_capo(["Eb", "Bb/D", "Cm7", "Ab"] * 4), 3)
        self.assertEqual(select_ergonomic_capo(["Ab", "Eb/G", "Fm7", "Db"] * 4), 1)
        self.assertEqual(select_ergonomic_capo(["B", "F#", "G#m", "E"] * 4), 4)

        # 5. /api/analyze_audio session reset (no stale capo=2 / slap_x / split_bar leak) & Pro-form split_bar toggle
        app_u = srv.create_app()
        client_u = TestClient(app_u)
        with app_u.state.session_lock:
            s0 = app_u.state.session
            s0.capo = 2
            s0.slap_x = True
            s0.palm_mute = True
            s0.split_bar = True
        with tempfile.TemporaryDirectory() as td_u:
            sr_u = 16000
            t_u = np.arange(sr_u * 4, dtype=np.float32) / sr_u
            wav_u = Path(td_u) / "upload_c.wav"
            sf.write(str(wav_u), 0.15 * np.sin(2 * np.pi * 261.63 * t_u), sr_u)
            voc_p = Path(td_u) / "vocals.wav"
            acc_p = Path(td_u) / "no_vocals.wav"
            sf.write(str(voc_p), 0.15 * np.sin(2 * np.pi * 261.63 * t_u), sr_u)
            sf.write(str(acc_p), np.zeros_like(t_u), sr_u)

            fake_notes = [
                Note(pitch=60, start=4.0, duration=1.0, raw_start=2.0, raw_duration=0.5),
                Note(pitch=64, start=5.0, duration=1.0, raw_start=2.5, raw_duration=0.5),
                Note(pitch=67, start=6.0, duration=2.0, raw_start=3.0, raw_duration=0.9),
            ]
            with (
                patch("subpkgs.stem_mir.separate_stems", return_value={"vocals": str(voc_p), "accompaniment": str(acc_p), "backend": "mock", "device": "cpu"}),
                patch("subpkgs.singing_lyrics.transcribe_word_timestamps", return_value={"word_timestamps": [{"word": "hello", "start": 2.0, "end": 2.9, "probability": 0.95}], "raw_lyrics": "hello"}),
                patch("subpkgs.stem_mir.track_beats_and_downbeats", return_value={"bpm": 120.0, "beat_origin": 0.0, "meter": "4/4", "beat_times": [0.5 * k for k in range(9)], "bar_start_seconds": [0.0, 2.0, 4.0]}),
                patch("subpkgs.singing_lyrics.transcribe_vocal_notes", return_value={"notes": fake_notes, "word_boundaries_sec": [2.0], "backend": "mock"}),
                patch("subpkgs.singing_lyrics.regulate_singing_notes_for_score", return_value=fake_notes),
                patch("subpkgs.stem_mir.synthesize_melody_conditioned_harmony", return_value={"chords": ["C", "G"], "bar_start_seconds": [0.0, 2.0, 4.0], "bar_dynamics": [0.9, 0.95]}),
            ):
                resp_up = client_u.post("/api/analyze_audio", files={"file": ("upload_c.wav", wav_u.read_bytes(), "audio/wav")}, data={"lyrics": "hello"})
                self.assertEqual(resp_up.status_code, 200)
                jid = resp_up.json()["job_id"]
                for _ in range(40):
                    j_st = client_u.get(f"/api/job/{jid}").json()
                    if j_st.get("status") in ("done", "error", "superseded"):
                        break
                    time.sleep(0.05)
                self.assertEqual(j_st.get("status"), "done", j_st)

        st_after = client_u.get("/api/state").json()
        self.assertEqual(st_after["capo"], 0)
        self.assertFalse(st_after["slap_x"])
        self.assertFalse(st_after["palm_mute"])
        self.assertFalse(st_after["split_bar"])
        self.assertTrue(st_after["use_detected_splits"])

        # Switching to neosoul_jazz turns split_bar=True; unchecking split_bar via multi-field Pro-form turns use_detected_splits=False
        r_ns = client_u.post("/api/reharmonize", json={"preset": "neosoul_jazz", "style": "Neo-Soul 律动", "split_bar": True}).json()
        self.assertTrue(r_ns["split_bar"])
        self.assertTrue(r_ns["use_detected_splits"])
        r_pro_off = client_u.post("/api/reharmonize", json={"preset": "neosoul_jazz", "style": "Neo-Soul 律动", "capo": 0, "split_bar": False}).json()
        self.assertFalse(r_pro_off["split_bar"])
        self.assertFalse(r_pro_off["use_detected_splits"])

    def test_visual_score_engraving_and_60fps_follow_along_contracts(self):
        import re
        import xml.etree.ElementTree as ET
        from fastapi.testclient import TestClient
        from subpkgs.score_web import server as srv
        from subpkgs.score_web.yoopu_svg_renderer import render_publication_score_html

        svg_ns = "{http://www.w3.org/2000/svg}"

        # 1. Deterministic engraving verification: rolled chord + slide lane clearance, 16th dual beam vs 8th-triplet
        #    single beam, dotted-8th + 16th fractional stub, CM/DS lead ink, H/P slur, and intra/cross-bar Jianpu ties.
        notes_eng = [
            Note(pitch=60, start=0.0, duration=2.0),
            Note(pitch=64, start=2.0, duration=3.0),  # spans beat 2..4 of Bar 0 and ties across barline into Bar 1 beat 0..1
            Note(pitch=67, start=5.0, duration=3.0),
        ]
        proj_eng = Project(title="精雕六线与简谱测试", tonic="C", meter="4/4", bpm=96.0, notes=notes_eng)
        arr_eng = {
            "title": "精雕六线与简谱测试",
            "tonic": "C",
            "meter": "4/4",
            "bpm": 96.0,
            "style": "指弹律动",
            "capo": 2,
            "bar_count": 2,
            "measure_beats": 4.0,
            "chords": ["C", "G"],
            "bars": [
                {
                    "bar": 0,
                    "chord": "C",
                    "roman": "I",
                    "frets": [-1, 3, 2, 0, 1, 0],
                    "split_segments": [{"beat": 0.0, "duration": 4.0, "chord": "C", "roman": "I", "frets": [-1, 3, 2, 0, 1, 0]}],
                },
                {
                    "bar": 1,
                    "chord": "G",
                    "roman": "V",
                    "frets": [3, 2, 0, 0, 0, 3],
                    "split_segments": [{"beat": 0.0, "duration": 4.0, "chord": "G", "roman": "V", "frets": [3, 2, 0, 0, 0, 3]}],
                },
            ],
            "events": [
                # Beat 0.0: rolled chord stack with slide_from_fret on string 2 (lane shifts to x - 22)
                {"bar": 0, "start": 0.0, "duration": 1.0, "string": 5, "fret": 3, "pitch": 48, "velocity": 86, "technique": "", "roll_step_sec": 0.022},
                {"bar": 0, "start": 0.0, "duration": 1.0, "string": 3, "fret": 0, "pitch": 55, "velocity": 82, "technique": "", "roll_step_sec": 0.022},
                {"bar": 0, "start": 0.0, "duration": 1.0, "string": 2, "fret": 3, "pitch": 62, "velocity": 88, "technique": "", "slide_from_fret": 1, "roll_step_sec": 0.022},
                # Beat 1.0..1.75: four 16th notes -> 2 beam lines (primary + secondary)
                {"bar": 0, "start": 1.00, "duration": 0.25, "string": 1, "fret": 0, "pitch": 64, "velocity": 80, "technique": ""},
                {"bar": 0, "start": 1.25, "duration": 0.25, "string": 2, "fret": 1, "pitch": 60, "velocity": 78, "technique": ""},
                {"bar": 0, "start": 1.50, "duration": 0.25, "string": 3, "fret": 0, "pitch": 55, "velocity": 76, "technique": ""},
                {"bar": 0, "start": 1.75, "duration": 0.25, "string": 2, "fret": 1, "pitch": 60, "velocity": 78, "technique": ""},
                # Beat 2.0..2.667: three 8th triplets (16 ticks each) -> single beam only
                {"bar": 0, "start": 2.0, "duration": 1.0 / 3.0, "string": 3, "fret": 0, "pitch": 55, "velocity": 78, "technique": ""},
                {"bar": 0, "start": 2.0 + 1.0 / 3.0, "duration": 1.0 / 3.0, "string": 2, "fret": 1, "pitch": 60, "velocity": 80, "technique": ""},
                {"bar": 0, "start": 2.0 + 2.0 / 3.0, "duration": 1.0 / 3.0, "string": 1, "fret": 0, "pitch": 64, "velocity": 82, "technique": ""},
                # Beat 3.0 (dotted 8th = 0.75) + Beat 3.75 (16th = 0.25) -> duration dot + fractional secondary stub
                {"bar": 0, "start": 3.00, "duration": 0.75, "string": 4, "fret": 2, "pitch": 52, "velocity": 82, "technique": ""},
                {"bar": 0, "start": 3.75, "duration": 0.25, "string": 3, "fret": 0, "pitch": 55, "velocity": 78, "technique": ""},
                # Bar 1: CM + DS lead dyad at beat 4.0, followed by H legato on string 1 at beat 5.0
                {"bar": 1, "start": 4.0, "duration": 1.0, "string": 1, "fret": 3, "pitch": 67, "velocity": 94, "technique": "CM"},
                {"bar": 1, "start": 4.0, "duration": 1.0, "string": 2, "fret": 0, "pitch": 59, "velocity": 86, "technique": "DS"},
                {"bar": 1, "start": 5.0, "duration": 1.0, "string": 1, "fret": 5, "pitch": 69, "velocity": 88, "technique": "H"},
            ],
        }
        html_eng = render_publication_score_html(proj_eng, arr_eng, ["春", "风", "吹"], [{"bar": 0, "section": "Intro"}, {"bar": 1, "section": "Verse"}])
        self.assertIn("@media print", html_eng)
        self.assertIn(".score-playhead, .score-bar-highlight, .score-hover-guide { display: none !important; }", html_eng)

        svgs_eng = re.findall(r"<svg\b[^>]*>.*?</svg>", html_eng, flags=re.S)
        self.assertEqual(len(svgs_eng), 1)
        root_eng = ET.fromstring(svgs_eng[0])
        self.assertEqual(root_eng.attrib.get("data-bar-start"), "0")
        self.assertEqual(root_eng.attrib.get("data-bar-end"), "2")
        self.assertEqual(root_eng.attrib.get("data-measure-ticks"), "192")

        bars_eng = [el for el in root_eng.findall(f"{svg_ns}g") if "score-bar" in el.attrib.get("class", "")]
        self.assertEqual(len(bars_eng), 2)

        # Verify Bar 0 engraving details & exact linear time<->x bijection
        b0 = bars_eng[0]
        x_left0 = float(b0.attrib["data-x-left"])
        x_pad0 = float(b0.attrib["data-x-pad"])
        w0 = float(b0.attrib["data-bar-width"])
        hit0 = list(b0)[0]
        self.assertIn("score-bar-hit", hit0.attrib.get("class", ""))
        self.assertAlmostEqual(float(hit0.attrib["x"]), x_left0, places=4)
        self.assertAlmostEqual(float(hit0.attrib["width"]), w0, places=4)

        roll_arrows = [el for el in b0.iter(f"{svg_ns}path") if "tab-roll-arrow" in el.attrib.get("class", "")]
        self.assertEqual(len(roll_arrows), 1)
        # Beat 0 x is x_left0 + x_pad0; with slide present on beat 0, roll arrow starts at x0 - 22
        m_roll_x = re.match(r"M([0-9.+-]+),", roll_arrows[0].attrib["d"])
        self.assertIsNotNone(m_roll_x)
        self.assertAlmostEqual(float(m_roll_x.group(1)), (x_left0 + x_pad0) - 22.0, places=2)

        # Check tab-note bijection & slide data-tech fallback
        tab_notes_b0 = [el for el in b0.findall(f"{svg_ns}g") if "tab-note" in el.attrib.get("class", "")]
        self.assertEqual(len(tab_notes_b0), 12)
        slide_tabs = [el for el in tab_notes_b0 if el.attrib.get("data-string") == "2" and el.attrib.get("data-ev-start") == "0"]
        self.assertEqual(len(slide_tabs), 1)
        self.assertEqual(slide_tabs[0].attrib.get("data-tech"), "SL")
        for tn in tab_notes_b0:
            ev_s = float(tn.attrib["data-ev-start"])
            rect_el = tn.find(f"{svg_ns}rect")
            expected_x = x_left0 + x_pad0 + (ev_s / 4.0) * (w0 - 2.0 * x_pad0)
            self.assertAlmostEqual(float(rect_el.attrib["x"]) + 9.0, expected_x, places=2)

        # Check beams in Bar 0:
        # Beat 1 (four 16ths) -> 2 beams; Beat 2 (three 8th triplets) -> 1 beam ONLY; Beat 3 (dotted 8th + 16th) -> 2 beams + 1 dot
        beams_b0 = [el for el in b0.iter(f"{svg_ns}line") if "tab-beam" in el.attrib.get("class", "")]
        self.assertEqual(len(beams_b0), 5)  # 2 (beat 1) + 1 (beat 2 triplet) + 2 (beat 3 dotted+16th) = 5!
        dots_b0 = [el for el in b0.iter(f"{svg_ns}circle") if "tab-dot" in el.attrib.get("class", "")]
        self.assertEqual(len(dots_b0), 1)

        # Check Bar 1 lead styling, H slur, and cross-barline incoming jp-tie
        b1 = bars_eng[1]
        lead_tabs_b1 = [el for el in b1.findall(f"{svg_ns}g") if "is-lead" in el.attrib.get("class", "")]
        self.assertEqual(len(lead_tabs_b1), 2)
        slurs_b1 = [el for el in b1.iter(f"{svg_ns}path") if "tab-slur" in el.attrib.get("class", "")]
        self.assertEqual(len(slurs_b1), 1)
        ties_b1 = [el for el in b1.iter(f"{svg_ns}path") if "jp-tie" in el.attrib.get("class", "")]
        self.assertGreaterEqual(len(ties_b1), 1)

        # 2. Parse every SVG system across live repertoire songs (4/4, 6/8, 3/4) to guarantee 100% XML validity
        client_v = TestClient(srv.create_app())
        for demo_id in ("mandarin_pop", "canon_in_d", "scarborough_fair", "regular_friends"):
            r_d = client_v.post("/api/load_demo", json={"demo": demo_id})
            self.assertEqual(r_d.status_code, 200)
            pub_html = r_d.json()["publication_html"]
            svg_blocks = re.findall(r"<svg\b[^>]*>.*?</svg>", pub_html, flags=re.S)
            self.assertGreater(len(svg_blocks), 0, f"No SVG systems in {demo_id}")
            for sb in svg_blocks:
                parsed = ET.fromstring(sb)
                self.assertIsNotNone(parsed.find(f".//*[@class='score-playhead']"))
                self.assertIsNotNone(parsed.find(f".//*[@class='score-bar-highlight']"))

        # 3. Verify static/index.html DOM hooks, 60fps sync functions, and concert-pitch playbackRate invariance
        index_html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "btn-follow-toggle",
            "btn-loop-a",
            "btn-loop-b",
            "btn-loop-clear",
            "loop-readout",
            "neck-tech",
            "guitar-neck-viz",
            "score-iframe",
            "seek-bar",
            "btn-play-toggle",
            "slider-vol-vocal",
            "slider-vol-guitar",
        ):
            self.assertIn(f'id="{required_id}"', index_html)
        for fn_name in ("beatToSec", "secToBeat", "buildScoreCache", "syncVisuals", "updateNeck", "scoreHit", "effectiveDuration"):
            self.assertIn(f"function {fn_name}(", index_html)
        # Never mutate AudioBufferSourceNode.playbackRate (must stay strictly at concert pitch 1.0)
        self.assertNotIn(".playbackRate", index_html)

    def test_campaign_fable_audit_and_split_inspector_regressions(self):
        import re
        import xml.etree.ElementTree as ET
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from subpkgs.guitar_reharm import arrange_advanced, check_melody_clash, reharmonize_progression
        from subpkgs.score_web import render_publication_score_html

        # M1: Meter / BPM change resamples base_chords & bar_dynamics by dominant bar overlap instead of truncating/drifting
        app_m1 = srv.create_app()
        with app_m1.state.session_lock:
            s_m1 = srv._build_session("mandarin_pop")
            # Force synthetic 12-bar 4/4 grid so assertions are deterministic regardless of local audio files
            s_m1.project.meter = "4/4"
            s_m1.project.bpm = 92.0
            s_m1.project.beat_origin = 0.0
            s_m1.project.source = ""
            s_m1.project.audio = {}
            s_m1.beat_times = []
            s_m1.bar_start_seconds = []
            s_m1.base_chords = ["C", "G/B", "Am7", "G", "C", "G/B", "Am7", "Fmaj7", "Fmaj7", "C/E", "Dm7", "C"]
            s_m1.bar_dynamics = [0.80 + 0.02 * i for i in range(12)]
            s_m1.custom_voicings = [None] * 12
            s_m1.project.notes, _ = parse_jianpu(srv._DEMO_PIECES["mandarin_pop"]["jianpu"], tonic="C", bpm=92.0)
            s_m1.project.reference_notes = [Note(**vars(n)) for n in s_m1.project.notes]
            s_m1.note_lyrics = [""] * len(s_m1.project.notes)
            s_m1.reference_note_lyrics = list(s_m1.note_lyrics)
            s_m1.note_ref_indices = list(range(len(s_m1.project.notes)))
            srv._rebuild(s_m1)
            app_m1.state.session = s_m1

        c_m1 = TestClient(app_m1)
        r_m1 = c_m1.post("/api/reharmonize", json={"meter": "3/4"})
        self.assertEqual(r_m1.status_code, 200)
        ba_m1 = r_m1.json()["bar_analysis"]
        self.assertEqual(len(ba_m1), 16)
        # Beats 9..12 -> old bar 2 ('Am7'), beats 12..15 -> old bar 3 ('G'), beats 45..48 -> old bar 11 ('C')
        self.assertEqual(ba_m1[3]["original_chord"], "Am7")
        self.assertEqual(ba_m1[4]["original_chord"], "G")
        self.assertEqual(ba_m1[15]["original_chord"], "C")

        # M2: 3/4 and 9/8 user split_bar=True lands on metrically valid beats (2.0 in 3/4, 3.0 in 9/8, 1.5 in 6/8)
        p_34 = Project(title="m2_34", tonic="C", meter="3/4", bpm=96.0, notes=[])
        reh_34_split = reharmonize_progression(["Am", "Dm", "G7", "C"], p_34, preset="walking_bass", tonic="C", split_bar=True)
        for row in reh_34_split:
            for seg in row["split_segments"]:
                self.assertIn(seg["beat"], (0.0, 2.0))
        p_98 = Project(title="m2_98", tonic="C", meter="9/8", bpm=90.0, notes=[])
        reh_98_split = reharmonize_progression(["Am", "Dm", "G7", "C"], p_98, preset="walking_bass", tonic="C", split_bar=True)
        for row in reh_98_split:
            for seg in row["split_segments"]:
                self.assertIn(seg["beat"], (0.0, 3.0))

        # M3: 3/4 palm_mute=True uses groove['bass_pivot'] (beat 2.0), never injecting an off-beat 1.5 PM bass
        p_pm34 = Project(
            title="m3_pm34",
            tonic="C",
            meter="3/4",
            bpm=96.0,
            notes=[Note(start=0.0, duration=2.0, pitch=69), Note(start=3.0, duration=2.0, pitch=65)],
        )
        arr_pm34 = arrange_advanced(p_pm34, ["Am", "Dm"], style="分解和弦", palm_mute=True, intro_mode="none")
        self.assertFalse(any(e.get("technique") == "PM" and abs(e["beat"] - 1.5) < 1e-9 for e in arr_pm34["events"]))

        # M4: Asymmetric 3/4 split [(0, 2), (2, 1)] places the SVG dashed divider at exact x_at(2.0 * TICKS)
        arr_svg34 = {
            "title": "m4_svg34",
            "tonic": "C",
            "meter": "3/4",
            "bpm": 96.0,
            "style": "指弹律动",
            "capo": 0,
            "bar_count": 1,
            "measure_beats": 3.0,
            "chords": ["Am7"],
            "bars": [
                {
                    "bar": 0,
                    "chord": "Am7",
                    "roman": "vi7",
                    "frets": [-1, 0, 2, 0, 1, 0],
                    "split_segments": [
                        {"beat": 0.0, "duration": 2.0, "chord": "Am7", "roman": "vi7", "frets": [-1, 0, 2, 0, 1, 0]},
                        {"beat": 2.0, "duration": 1.0, "chord": "G/B", "roman": "V/7", "frets": [-1, 2, 0, 0, 0, 3]},
                    ],
                }
            ],
            "events": [
                {"bar": 0, "start": 0.0, "duration": 1.0, "string": 5, "fret": 0, "pitch": 45, "velocity": 84, "technique": ""},
                {"bar": 0, "start": 2.0, "duration": 1.0, "string": 5, "fret": 2, "pitch": 47, "velocity": 84, "technique": ""},
            ],
        }
        p_svg34 = Project(title="m4_svg34", tonic="C", meter="3/4", bpm=96.0, notes=[Note(start=0.0, duration=3.0, pitch=69)])
        svg_34_html = render_publication_score_html(p_svg34, arr_svg34, ["月"])
        svg_34_match = re.search(r"<svg\b[^>]*>.*?</svg>", svg_34_html, flags=re.S)
        self.assertIsNotNone(svg_34_match)
        root_34 = ET.fromstring(svg_34_match.group(0))
        svg_ns = "{http://www.w3.org/2000/svg}"
        bar_g = next(el for el in root_34.iter(f"{svg_ns}g") if el.attrib.get("class") == "score-bar")
        b_left = float(bar_g.attrib["data-x-left"])
        b_pad = float(bar_g.attrib["data-x-pad"])
        b_w = float(bar_g.attrib["data-bar-width"])
        dashed = [
            el for el in bar_g.findall(f"{svg_ns}line")
            if el.attrib.get("stroke-dasharray") == "4,3"
        ]
        self.assertEqual(len(dashed), 1)
        expected_div_x = b_left + b_pad + (2.0 / 3.0) * (b_w - 2.0 * b_pad)
        self.assertAlmostEqual(float(dashed[0].attrib["x1"]), expected_div_x, places=2)

        # M5: Leaped downbeat minor-2nd Db4 over C is flagged as clash ('minor_ninth'), while offbeat stepwise passing F4 stays tension
        rep_clash = check_melody_clash(
            "C",
            [Note(pitch=61, start=0.0, duration=0.5), Note(pitch=67, start=0.5, duration=3.5)],
            0.0,
            4.0,
            "C",
        )
        self.assertEqual(rep_clash["status"], "clash")
        self.assertEqual(rep_clash["clashes"][0]["reason"], "minor_ninth")
        rep_pass = check_melody_clash(
            "C",
            [
                Note(pitch=64, start=0.0, duration=1.0),
                Note(pitch=65, start=1.0, duration=0.5),
                Note(pitch=67, start=1.5, duration=2.5),
            ],
            0.0,
            4.0,
            "C",
        )
        self.assertNotEqual(rep_pass["status"], "clash")

        # M6: Fractional beat_origin strictly between beat_times grid points aligns arrangement['beat_times'][0] == 0.0 at warp_origin
        s_m6 = srv._build_session("real_vocal")
        s_m6.project.source = ""
        s_m6.project.audio = {}
        s_m6.project.bpm = 120.0
        s_m6.project.beat_origin = 0.25
        s_m6.beat_times = [0.5 * i for i in range(40)]
        s_m6.bar_start_seconds = [2.0 * i for i in range(10)]
        srv._rebuild(s_m6)
        self.assertAlmostEqual(s_m6.arrangement["beat_times"][0], 0.0, places=6)
        self.assertAlmostEqual(s_m6.arrangement["beat_times"][1], 0.5, places=6)
        self.assertAlmostEqual(s_m6.arrangement["warp_origin"], 0.25, places=6)

        # M7 + SG1 + SG3: /api/load_demo runs _rebuild once; split segments expose per-segment voicings/candidates; /evolution_atlas serves 200
        app_m7 = srv.create_app()
        c_m7 = TestClient(app_m7)
        with patch.object(srv, "_rebuild", wraps=srv._rebuild) as mock_reb:
            r_ld = c_m7.post("/api/load_demo", json={"demo": "real_vocal"})
            self.assertEqual(r_ld.status_code, 200)
            self.assertEqual(mock_reb.call_count, 1)
        r_sp = c_m7.post("/api/reharmonize", json={"preset": "walking_bass", "split_bar": True})
        self.assertEqual(r_sp.status_code, 200)
        split_bars = [b for b in r_sp.json()["bar_analysis"] if len(b.get("split_segments", [])) > 1]
        self.assertGreaterEqual(len(split_bars), 1)
        seg1_info = split_bars[0]["split_segments"][1]
        self.assertTrue(len(seg1_info.get("voicings", [])) >= 1)
        self.assertTrue(len(seg1_info.get("candidates", [])) >= 1)
        # Replacing segment 1 chord preserves segment 0 chord intact
        target_bar_idx = split_bars[0]["bar"]
        seg0_chord_before = split_bars[0]["split_segments"][0]["chord"]
        alt_chord = seg1_info["candidates"][0]["chord"]
        r_seg_edit = c_m7.post("/api/bar_chord", json={"bar": target_bar_idx, "segment": 1, "chord": alt_chord})
        self.assertEqual(r_seg_edit.status_code, 200)
        updated_bar = r_seg_edit.json()["bar_analysis"][target_bar_idx]
        self.assertEqual(len(updated_bar["split_segments"]), 2)
        self.assertEqual(updated_bar["split_segments"][0]["chord"], seg0_chord_before)
        self.assertEqual(updated_bar["split_segments"][1]["chord"], alt_chord)

        r_atlas = c_m7.get("/evolution_atlas")
        self.assertEqual(r_atlas.status_code, 200)
        self.assertIn("MelodyDesk 2.0", r_atlas.text)


if __name__ == "__main__":
    unittest.main()
