"""FastAPI Backend for MelodyDesk 2.0 Workbench (with Re-harmonization Options A/B/C & Real-Vocal + Simulated Guitar Unplugged Audio Mix Endpoints)."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
import hashlib
import inspect
import io
import logging
import math
import threading
import wave

logger = logging.getLogger(__name__)
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from music import Note, Project, NAMES, parse_jianpu, pitch_name, numbered
import json
import guitar
from score_cleanup import organize
from .alphatex_builder import to_alphatex
from .yoopu_svg_renderer import render_publication_score_html
from ._score_timing import build_score_timeline

from runtime_paths import get_app_root, get_install_root, get_work_root, get_writable_work_root

APP_ROOT = get_app_root()
INSTALL_ROOT = get_install_root()
WORK_ROOT = get_work_root()
STATIC_ROOT = APP_ROOT / 'static'
EXAMPLE_PROJECT = INSTALL_ROOT / 'examples' / 'demo-project.json'
SIMULATION_ROOT = WORK_ROOT / 'simulation_demo'
REPERTOIRE_AUDIO_ROOT = WORK_ROOT / 'repertoire_audio'
_DEFAULT_REPERTOIRE_MANIFEST = REPERTOIRE_AUDIO_ROOT / 'repertoire_manifest.json'
REPERTOIRE_MANIFEST = _DEFAULT_REPERTOIRE_MANIFEST
PORTABLE_REPERTOIRE_MANIFEST = APP_ROOT / 'data' / 'repertoire_manifest.json'
_STEM_AUDIO_CACHE: dict[tuple, np.ndarray] = {}
_VOCAL_CAL_CACHE: dict[tuple, np.ndarray] = {}

try:
    from subpkgs.guitar_reharm import (reharmonize_progression, arrange_advanced,
        roman_numeral, check_melody_clash, generate_voicings, select_ergonomic_capo, suggest_bar_substitutions,
        REHARMONIZE_PRESETS, ADVANCED_STYLES, synthesize_realistic_guitar, recommend_artistry)
except ModuleNotFoundError as exc:
    if not (exc.name == 'subpkgs.guitar_reharm' or (exc.name and exc.name.startswith('subpkgs.guitar_reharm.'))):
        raise
    reharmonize_progression = arrange_advanced = synthesize_realistic_guitar = suggest_bar_substitutions = recommend_artistry = select_ergonomic_capo = None
    REHARMONIZE_PRESETS, ADVANCED_STYLES = {}, ()
try:
    from subpkgs.singing_lyrics import align_lyrics_to_notes, estimate_key_albrecht_shanahan
except ModuleNotFoundError as exc:
    if not (exc.name == 'subpkgs.singing_lyrics' or (exc.name and exc.name.startswith('subpkgs.singing_lyrics.'))):
        raise
    align_lyrics_to_notes = None
    estimate_key_albrecht_shanahan = None
try:
    from subpkgs.stem_mir.beat_tracker import beats_to_seconds_piecewise, seconds_to_beats_piecewise
except ModuleNotFoundError as exc:
    if not (exc.name == 'subpkgs.stem_mir' or (exc.name and exc.name.startswith('subpkgs.stem_mir.'))):
        raise
    beats_to_seconds_piecewise = seconds_to_beats_piecewise = None


_PROBE_CACHE: dict[tuple, bool] = {}

def _implemented(function) -> bool:
    if not callable(function) or getattr(function, '__is_stub__', False):
        return False
    key = (id(function), getattr(function, '__code__', None))
    if key in _PROBE_CACHE: return _PROBE_CACHE[key]
    p = Project(title='probe', tonic='C', meter='4/4', bpm=120.0, notes=[Note(60, 0.0, 1.0, 0.0, 0.5)])
    name = getattr(function, '__name__', '')
    try:
        if function is reharmonize_progression or name == 'reharmonize_progression':
            ok = isinstance(function(['C'], p, preset='original', tonic='C'), list)
        elif function is arrange_advanced or name == 'arrange_advanced':
            ok = isinstance(function(p, ['C']), dict)
        elif function is suggest_bar_substitutions or name == 'suggest_bar_substitutions':
            ok = isinstance(function('C', None, None, 'C', p.notes, 0.0, 4.0), list)
        elif function is align_lyrics_to_notes or name == 'align_lyrics_to_notes':
            ok = isinstance(function(p, raw_lyrics='la'), dict)
        elif function is estimate_key_albrecht_shanahan or name == 'estimate_key_albrecht_shanahan':
            ok = isinstance(function(p.notes), dict)
        elif function is synthesize_realistic_guitar or name == 'synthesize_realistic_guitar':
            ok = function(p, {'events': [], 'bar_count': 1, 'measure_beats': 4.0}, out_wav_path=None, sr=22050) is not None
        else:
            req = [v for v in inspect.signature(function).parameters.values() if v.default is inspect.Parameter.empty and v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD)]
            ok = (function(*([p, ['C']][:len(req)])) if len(req) <= 2 else None) is not None
    except NotImplementedError: return False
    except Exception as exc:
        logger.warning("Backend probe failed for %s: %s", name or function, exc); return False
    if ok:
        _PROBE_CACHE[key] = True
    return ok

def _refresh_backends():
    global reharmonize_progression, arrange_advanced, roman_numeral, check_melody_clash, generate_voicings, select_ergonomic_capo, suggest_bar_substitutions
    global REHARMONIZE_PRESETS, ADVANCED_STYLES, synthesize_realistic_guitar, recommend_artistry, align_lyrics_to_notes, estimate_key_albrecht_shanahan
    global beats_to_seconds_piecewise, seconds_to_beats_piecewise
    global NATIVE, PRESETS, STYLES
    import importlib
    try:
        gr = importlib.import_module('subpkgs.guitar_reharm')
        reharmonize_progression, arrange_advanced = getattr(gr, 'reharmonize_progression', None), getattr(gr, 'arrange_advanced', None)
        roman_numeral, check_melody_clash = getattr(gr, 'roman_numeral', None), getattr(gr, 'check_melody_clash', None)
        generate_voicings, suggest_bar_substitutions = getattr(gr, 'generate_voicings', None), getattr(gr, 'suggest_bar_substitutions', None)
        select_ergonomic_capo = getattr(gr, 'select_ergonomic_capo', None)
        REHARMONIZE_PRESETS, ADVANCED_STYLES = getattr(gr, 'REHARMONIZE_PRESETS', {}), getattr(gr, 'ADVANCED_STYLES', ())
        synthesize_realistic_guitar = getattr(gr, 'synthesize_realistic_guitar', None)
        recommend_artistry = getattr(gr, 'recommend_artistry', None)
    except ModuleNotFoundError as exc:
        logger.debug("guitar_reharm optional module absent: %s", exc)
    except Exception as exc:
        logger.warning("guitar_reharm import failed: %s", exc)
    try:
        sl = importlib.import_module('subpkgs.singing_lyrics')
        align_lyrics_to_notes, estimate_key_albrecht_shanahan = getattr(sl, 'align_lyrics_to_notes', None), getattr(sl, 'estimate_key_albrecht_shanahan', None)
    except ModuleNotFoundError as exc:
        logger.debug("singing_lyrics optional module absent: %s", exc)
    except Exception as exc:
        logger.warning("singing_lyrics import failed: %s", exc)
    try:
        bt_mod = importlib.import_module('subpkgs.stem_mir.beat_tracker')
        beats_to_seconds_piecewise = getattr(bt_mod, 'beats_to_seconds_piecewise', None)
        seconds_to_beats_piecewise = getattr(bt_mod, 'seconds_to_beats_piecewise', None)
    except ModuleNotFoundError as exc:
        logger.debug("stem_mir.beat_tracker optional module absent: %s", exc)
    except Exception as exc:
        logger.warning("stem_mir.beat_tracker import failed: %s", exc)
    NATIVE = _implemented(reharmonize_progression) and _implemented(arrange_advanced) and bool(REHARMONIZE_PRESETS)
    PRESETS = REHARMONIZE_PRESETS if NATIVE else {'original': {'name': '原版和弦', 'description': 'Basic triad accompaniment'}}
    STYLES = list(ADVANCED_STYLES) if NATIVE else ['分解和弦', '流行扫弦']


_refresh_backends()


def _has_audible_accompaniment(acc_path: str | Path | None) -> bool:
    """Return True when the separated accompaniment stem exists and has multi-channel RMS >= 0.004 overall and >= 0.008 at the 90th-percentile 50ms frame."""
    if not acc_path:
        return False
    p = Path(acc_path)
    if not p.is_file():
        return False
    try:
        import soundfile as sf
        audio, sr = sf.read(str(p), dtype='float32', always_2d=True)
    except Exception as exc:
        logger.debug("Accompaniment stem read failed for %s: %s", p, exc)
        return False
    if audio.size == 0 or sr <= 0 or not np.isfinite(audio).all():
        return False
    ms = np.mean(audio.astype(np.float64) ** 2, axis=1)
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
    p90_rms = float(np.percentile(f_rms, 90.0)) if f_rms.size else overall_rms
    return p90_rms >= 0.008


@dataclass
class Session:
    project: Project
    base_chords: list[str]
    custom_chords: dict = field(default_factory=dict)
    custom_voicings: list = field(default_factory=list)
    custom_segment_voicings: dict = field(default_factory=dict)
    preset: str = 'original'
    style: str = STYLES[0]
    capo: int = 0
    passing_bass: bool = False
    split_bar: bool = False
    slap_x: bool = False
    palm_mute: bool = False
    embellish_hp: bool = False
    intro_mode: str = 'chord_melody'
    raw_lyrics: str = ''
    note_lyrics: list[str] = field(default_factory=list)
    reference_note_lyrics: list[str] = field(default_factory=list)
    note_ref_indices: list[int] = field(default_factory=list)
    custom_note_lyrics: dict[int, str] = field(default_factory=dict)
    bar_analysis: list[dict] = field(default_factory=list)
    arrangement: dict = field(default_factory=dict)
    undo_stack: list = field(default_factory=list)
    bar_start_seconds: list[float] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)
    bar_dynamics: list[float] = field(default_factory=list)
    detected_split_chords: dict = field(default_factory=dict)
    use_detected_splits: bool = True
    word_timestamps: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lyric_extraction_failed: bool = False
    harmony_synthesis_failed: bool = False
    demo_mode: str = 'mandarin_pop'
    _cached_guitar_samples: tuple | None = field(default=None, repr=False)


def _snapshot(session):
    # Snapshots share the read-only _cached_guitar_samples pair and omit undo_stack to avoid recursive history nesting.
    return Session(**{key: (value if key == '_cached_guitar_samples' else deepcopy(value))
                      for key, value in vars(session).items() if key != 'undo_stack'})


def _align(candidate, timestamps=None, require_backend=False):
    if timestamps is not None:
        candidate.word_timestamps = deepcopy(timestamps)
    eff_timestamps = candidate.word_timestamps or None
    if (not candidate.raw_lyrics and eff_timestamps is None) or (not _implemented(align_lyrics_to_notes) and not require_backend):
        n_notes = len(candidate.project.notes)
        if len(candidate.note_lyrics) < n_notes:
            candidate.note_lyrics = list(candidate.note_lyrics) + [''] * (n_notes - len(candidate.note_lyrics))
        elif len(candidate.note_lyrics) > n_notes:
            candidate.note_lyrics = list(candidate.note_lyrics[:n_notes])
    else:
        if not _implemented(align_lyrics_to_notes):
            raise HTTPException(503, 'Lyric alignment dependency is unavailable')
        vocal_path = candidate.project.audio.get('vocals')
        available = vocal_path if (vocal_path and Path(vocal_path).is_file() and not candidate.raw_lyrics.strip() and eff_timestamps is None) else None
        alignment = align_lyrics_to_notes(candidate.project, raw_lyrics=candidate.raw_lyrics,
                                         word_timestamps=eff_timestamps, vocal_audio_path=available)
        candidate.note_lyrics = list(alignment['note_lyrics'])
        candidate.raw_lyrics = alignment['raw_lyrics']
    if candidate.custom_note_lyrics and len(candidate.note_ref_indices) == len(candidate.note_lyrics):
        for i, ref_idx in enumerate(candidate.note_ref_indices):
            if ref_idx in candidate.custom_note_lyrics:
                candidate.note_lyrics[i] = candidate.custom_note_lyrics[ref_idx]


def _rebuild(candidate):
    _refresh_backends()
    candidate._cached_guitar_samples = None
    candidate.project.bar_start_seconds = list(candidate.bar_start_seconds)
    candidate.project.beat_times = list(candidate.beat_times)
    if len(candidate.note_ref_indices) != len(candidate.project.notes):
        candidate.note_ref_indices = list(range(len(candidate.project.notes)))
    if not candidate.reference_note_lyrics and len(candidate.project.reference_notes) == len(candidate.note_lyrics):
        candidate.reference_note_lyrics = list(candidate.note_lyrics)
    pre_order = {id(n): i for i, n in enumerate(candidate.project.notes)}
    candidate.project.validate()
    perm = [pre_order[id(n)] for n in candidate.project.notes]
    if perm != list(range(len(perm))):
        if len(candidate.note_lyrics) == len(perm):
            candidate.note_lyrics = [candidate.note_lyrics[i] for i in perm]
        if len(candidate.note_ref_indices) == len(perm):
            candidate.note_ref_indices = [candidate.note_ref_indices[i] for i in perm]
    measure = int(candidate.project.meter.split('/')[0])*4/int(candidate.project.meter.split('/')[1])
    count = math.ceil(max((n.start+n.duration for n in candidate.project.notes), default=0)/measure)
    if candidate.bar_start_seconds and len(candidate.bar_start_seconds) >= 2:
        rem_sec = max(0.0, candidate.project.clip_duration - candidate.bar_start_seconds[-1]) if candidate.project.source else 0.0
        count = max(count, len(candidate.bar_start_seconds) - 1 + math.ceil(rem_sec * candidate.project.bpm / 60 / measure))
    elif candidate.project.source:
        count = max(count, math.ceil((candidate.project.clip_duration-candidate.project.beat_origin)*candidate.project.bpm/60/measure))
    voicings_explicit, voicings_reset = bool(candidate.__dict__.pop('_voicings_explicit', False)), bool(candidate.__dict__.pop('_voicings_reset', False))
    chords_explicit = bool(candidate.__dict__.pop('_chords_explicit', False))
    if voicings_reset:
        candidate.custom_voicings = [None] * count
    old_c_len = len(candidate.base_chords)
    if old_c_len != count:
        if old_c_len > count:
            candidate.base_chords = list(candidate.base_chords[:count])
        else:
            suggested = guitar.suggest(candidate.project, NAMES.index(candidate.project.tonic))
            fallback_chord = candidate.base_chords[-1] if candidate.base_chords else candidate.project.tonic
            tail = [suggested[b] if b < len(suggested) else fallback_chord for b in range(old_c_len, count)]
            candidate.base_chords = list(candidate.base_chords) + tail
    if not voicings_explicit and not voicings_reset and len(candidate.custom_voicings) != count:
        old_v_len = len(candidate.custom_voicings)
        if old_v_len > count:
            candidate.custom_voicings = list(candidate.custom_voicings[:count])
        else:
            candidate.custom_voicings = list(candidate.custom_voicings) + [None] * (count - old_v_len)
    if not chords_explicit and candidate.custom_chords:
        candidate.custom_chords = {k: v for k, v in candidate.custom_chords.items() if 0 <= int(k.split(':')[0]) < count}
    if candidate.custom_segment_voicings:
        candidate.custom_segment_voicings = {k: v for k, v in candidate.custom_segment_voicings.items() if 0 <= int(k.split(':')[0]) < count}
    if candidate.detected_split_chords:
        candidate.detected_split_chords = {k: v for k, v in candidate.detected_split_chords.items() if str(k).isdigit() and 0 <= int(k) < count}
    if candidate.preset not in PRESETS or candidate.style not in STYLES:
        raise ValueError('Unsupported preset or style for the active backend')
    if len(candidate.custom_voicings) != count:
        raise ValueError('Custom voicings must match bar count')
    if NATIVE:
        _reharm_kwargs = dict(
            preset=candidate.preset, tonic=candidate.project.tonic, split_bar=candidate.split_bar,
            capo=candidate.capo,
        )
        if (
            candidate.use_detected_splits
            and candidate.detected_split_chords
            and 'detected_split_chords' in inspect.signature(reharmonize_progression).parameters
        ):
            _reharm_kwargs['detected_split_chords'] = candidate.detected_split_chords
        analysis = reharmonize_progression(candidate.base_chords, candidate.project, **_reharm_kwargs)
        for identity, chord in candidate.custom_chords.items():
            pieces = identity.split(':')
            bar = int(pieces[0])
            if not 0 <= bar < len(analysis):
                raise ValueError('Chord override bar is out of range')
            segments = analysis[bar]['split_segments']
            targets = range(len(segments)) if len(pieces) == 1 else [int(pieces[1])]
            for part in targets:
                if not 0 <= part < len(segments):
                    if not chords_explicit:
                        continue
                    raise ValueError('Chord override segment is out of range')
                segment = segments[part]
                choices = generate_voicings(chord, candidate.capo)
                if not choices:
                    raise ValueError('Chord has no playable voicing')
                segment.pop('frets', None)
                segment.update(chord=chord, roman=roman_numeral(chord, candidate.project.tonic), capo=candidate.capo)
        for row in analysis:
            row['chord'] = row['split_segments'][0]['chord']
            row['roman'] = row['split_segments'][0]['roman']
        for index, row in enumerate(analysis):
            override = candidate.custom_voicings[index]
            if override is not None:
                row['split_segments'][0].update(frets=list(override), capo=candidate.capo)
            reports = []
            for part, segment in enumerate(row['split_segments']):
                identity = f'{index}:{part}'
                if identity in candidate.custom_segment_voicings:
                    segment.update(frets=list(candidate.custom_segment_voicings[identity]), capo=candidate.capo)
                reports.append(check_melody_clash(segment['chord'], candidate.project.notes,
                    index*measure+segment['beat'], index*measure+segment['beat']+segment['duration'],
                    candidate.project.tonic))
            row['chord'] = row['split_segments'][0]['chord']
            row['roman'] = row['split_segments'][0]['roman']
            status = ('clash' if any(r.get('status') == 'clash' or r['clashes'] for r in reports) else
                      'tension' if any(r.get('status') == 'tension' or r['tensions'] for r in reports) else 'safe')
            reasons = [r.get('reason', '') for r in reports if r.get('reason')]
            row['clash'] = dict(status=status,
                message='⚠️ 小二度避用音冲突' if status == 'clash' else ('✨ 色彩张力音 (9/11/13)' if status == 'tension' else '✅ 与主旋律和谐'),
                clashes=[c for r in reports for c in r['clashes']],
                tensions=[t for r in reports for t in r['tensions']],
                chord_tones=sorted({pc for r in reports for pc in r['chord_tones']}),
                notes_checked=sum(r['notes_checked'] for r in reports),
                clashing_pitches=[p for r in reports for p in r.get('clashing_pitches', [])],
                reason='; '.join(dict.fromkeys(reasons)))
            row['voicings'] = generate_voicings(row['chord'], candidate.capo)
            if (candidate.custom_chords or not row.get('candidates')) and _implemented(suggest_bar_substitutions) and row['chord'] != 'N.C.':
                prev_c, next_c = (analysis[index-1]['chord'] if index > 0 else None), (analysis[index+1]['chord'] if index+1 < len(analysis) else None)
                row['candidates'] = suggest_bar_substitutions(row['chord'], prev_c, next_c, candidate.project.tonic, candidate.project.notes, index*measure, (index+1)*measure)
        _arrange_kwargs = dict(
            chords=[row['chord'] for row in analysis], style=candidate.style, capo=candidate.capo,
            custom_voicings=candidate.custom_voicings, split_chords=[row['split_segments'] for row in analysis],
            passing_bass=candidate.passing_bass, slap_x=candidate.slap_x, palm_mute=candidate.palm_mute,
            embellish_hp=candidate.embellish_hp, intro_mode=candidate.intro_mode)
        if candidate.bar_dynamics and 'bar_dynamics' in inspect.signature(arrange_advanced).parameters:
            _arrange_kwargs['bar_dynamics'] = candidate.bar_dynamics
        arrangement = arrange_advanced(candidate.project, **_arrange_kwargs)
        for row_idx, (row, bar) in enumerate(zip(analysis, arrangement['bars'])):
            segs_copy = deepcopy(bar['split_segments'])
            prev_bar_c = analysis[row_idx - 1]['split_segments'][-1]['chord'] if row_idx > 0 else None
            next_bar_c = analysis[row_idx + 1]['split_segments'][0]['chord'] if row_idx + 1 < len(analysis) else None
            for p_idx, seg in enumerate(segs_copy):
                seg['voicings'] = generate_voicings(seg['chord'], candidate.capo)
                if _implemented(suggest_bar_substitutions) and seg['chord'] != 'N.C.':
                    p_c = segs_copy[p_idx - 1]['chord'] if p_idx > 0 else prev_bar_c
                    n_c = segs_copy[p_idx + 1]['chord'] if p_idx + 1 < len(segs_copy) else next_bar_c
                    seg_s = row_idx * measure + float(seg.get('beat', 0.0))
                    seg_e = seg_s + float(seg.get('duration', measure))
                    seg['candidates'] = suggest_bar_substitutions(
                        seg['chord'], p_c, n_c, candidate.project.tonic, candidate.project.notes, seg_s, seg_e
                    )[:8]
                else:
                    seg['candidates'] = []
            row['split_segments'] = segs_copy
            row['frets'] = list(bar['frets'])
            if segs_copy:
                row['voicings'] = list(segs_copy[0]['voicings'])
    else:
        if candidate.preset != 'original' or any((candidate.passing_bass, candidate.split_bar, candidate.slap_x,
                                                candidate.palm_mute, candidate.embellish_hp, candidate.intro_mode != 'none')):
            raise HTTPException(503, 'Advanced arrangement dependency is unavailable; basic triads only')
        chords = list(candidate.base_chords)
        for identity, chord in candidate.custom_chords.items():
            if ':' in identity:
                raise ValueError('Basic backend supports full-bar chords only')
            bar_idx = int(identity)
            if not 0 <= bar_idx < len(chords):
                raise ValueError('Chord override bar is out of range')
            chords[bar_idx] = chord
        if any(v is not None for v in candidate.custom_voicings) or candidate.custom_segment_voicings:
            raise HTTPException(503, 'Custom voicings require the advanced arranger')
        arrangement = guitar.arrange(candidate.project, chords, candidate.style, candidate.capo)
        arrangement.update(measure_beats=measure, bar_count=len(arrangement['bars']), chords=chords)
        for event in arrangement['events']:
            # Convert guitar.arrange 0..5 (low E..high E) to standard guitar string numbers 6..1 (low E..high E).
            event['string'] = 6 - event['string']
            event['technique'] = ''
        analysis = []
        for index, bar in enumerate(arrangement['bars']):
            segment = dict(beat=0.0, duration=measure, chord=bar['chord'], roman='', frets=list(bar['frets']))
            bar.update(bar=index, roman='', split_segments=[segment])
            analysis.append(dict(bar=index, chord=bar['chord'], roman='', split_segments=[deepcopy(segment)]))
    build_score_timeline(candidate.project, arrangement)
    origin_sec = float(candidate.project.beat_origin)
    raw_bt_all = [float(t) for t in candidate.beat_times]
    raw_bss_all = [float(t) for t in candidate.bar_start_seconds]
    if (
        origin_sec != 0.0
        and len(raw_bt_all) >= 2
        and callable(seconds_to_beats_piecewise)
        and callable(beats_to_seconds_piecewise)
        and not any(abs(t - origin_sec) <= 1e-6 for t in raw_bt_all)
    ):
        cur_bpm_w = float(candidate.project.bpm)
        ob = float(seconds_to_beats_piecewise(origin_sec, raw_bt_all, cur_bpm_w, origin_sec))
        n_rem_bt = max(2, int(math.ceil(max(0.0, (len(raw_bt_all) - 1) - ob))) + 1)
        warp_origin = origin_sec
        arrangement['beat_times'] = [
            float(beats_to_seconds_piecewise(ob + k, raw_bt_all, cur_bpm_w, origin_sec)) - warp_origin
            for k in range(n_rem_bt)
        ]
        n_bars_w = max(1, count)
        arrangement['bar_start_seconds'] = [
            float(beats_to_seconds_piecewise(ob + b * measure, raw_bt_all, cur_bpm_w, origin_sec)) - warp_origin
            for b in range(n_bars_w + 1)
        ]
    else:
        active_bt = [t for t in raw_bt_all if t >= origin_sec - 1e-9]
        if len(active_bt) < 2 and len(raw_bt_all) >= 2: active_bt = list(raw_bt_all)
        active_bss = [t for t in raw_bss_all if t >= origin_sec - 1e-9]
        if len(active_bss) < 2 and len(raw_bss_all) >= 2: active_bss = list(raw_bss_all)
        warp_origin = (origin_sec if origin_sec != 0.0 else min(active_bt[0], active_bss[0]) if len(active_bt) >= 2 and len(active_bss) >= 2
                       else active_bt[0] if len(active_bt) >= 2 else active_bss[0] if len(active_bss) >= 2 else 0.0)
        if active_bss: arrangement['bar_start_seconds'] = [t - warp_origin for t in active_bss]
        if active_bt: arrangement['beat_times'] = [t - warp_origin for t in active_bt]
    arrangement.update(time_origin_shifted=True, warp_origin=warp_origin, warp_beat_origin=origin_sec)
    if candidate.bar_dynamics: arrangement['bar_dynamics'] = list(candidate.bar_dynamics)
    if len(candidate.note_lyrics) != len(candidate.project.notes): _align(candidate)
    if not candidate.reference_note_lyrics and len(candidate.project.reference_notes) == len(candidate.note_lyrics):
        candidate.reference_note_lyrics = list(candidate.note_lyrics)
    candidate.bar_analysis, candidate.arrangement = analysis, arrangement


_MANIFEST_CACHE: tuple[str, int, int, dict] | None = None


def _resolve_repertoire_path(raw: str | Path | None) -> Path | None:
    """Resolve a portable relative or local absolute repertoire audio path if present on disk."""
    if not raw:
        return None
    norm = str(raw).replace('\\', '/')
    direct = Path(norm)
    if direct.is_file():
        return direct.resolve()
    if not direct.is_absolute():
        rel = Path(norm.lstrip('/'))
        for base in (INSTALL_ROOT, WORK_ROOT.parent, APP_ROOT, REPERTOIRE_AUDIO_ROOT):
            cand = base / rel
            if cand.is_file():
                return cand.resolve()
    return None


def _get_repertoire_manifest() -> dict:
    global _MANIFEST_CACHE
    manifest_path = REPERTOIRE_MANIFEST
    if (not manifest_path.is_file()
            and manifest_path == _DEFAULT_REPERTOIRE_MANIFEST
            and PORTABLE_REPERTOIRE_MANIFEST.is_file()):
        manifest_path = PORTABLE_REPERTOIRE_MANIFEST
    if manifest_path.is_file():
        try:
            st = manifest_path.stat()
            cache_sig = (str(manifest_path), st.st_mtime_ns, st.st_size)
            if _MANIFEST_CACHE is None or _MANIFEST_CACHE[:3] != cache_sig:
                _MANIFEST_CACHE = (*cache_sig, json.loads(manifest_path.read_text(encoding='utf-8')))
            return _MANIFEST_CACHE[3]
        except Exception as exc:
            logger.warning("Failed to parse repertoire manifest %s: %s", manifest_path, exc)
    return {}


# Built-in synthesized jianpu/chord practice pieces (used for synthetic demo modes and as fallback when a repertoire manifest recording is unavailable).
_DEMO_PIECES = {
    'canon_in_d': dict(title='合成演示 · 卡农主旋律分解和弦练习', tonic='C', meter='4/4', capo=2, bpm=78.0, style=-1, clip_duration=50.0, passing_bass=True, embellish_hp=True,
        jianpu="3':2 2':2 | 1':2 7:2 | 6:2 5:2 | 6:2 7:2 | 1':1 3':1 2':1 7:1 | 1':1 5:1 6:1 4:1 | 3:1 6:1 5:1 4:1 | "
               "5:1 2':1 7:1 5:1 | 1':1 7:0.5 1':0.5 7:1 5:1 | 6:1 1':1 5:1 3:1 | 4:1 6:1 5:0.5 4:0.5 3:1 | "
               "2:1 4:1 3:1 2:1 | 3:1 1':1 7:1 6:1 | 5:1 3:1 4:1 5:1 | 6:1 4:1 5:1 7:1 | 1':4",
        chords=['C', 'G/B', 'Am7', 'Em7', 'Fmaj7', 'C/E', 'Fmaj7', 'G7sus4',
                'C', 'G/B', 'Am7', 'Em7', 'Fmaj7', 'C/E', 'Dm7', 'C'], lyrics=''),
    'romance_amor': dict(title='合成演示 · 爱的罗曼史三拍琶音练习', tonic='C', meter='3/4', capo=0, bpm=86.0, style=-1, clip_duration=36.0, passing_bass=False, embellish_hp=False,
        jianpu="3':1 3':1 3':1 | 3':1 2':1 1':1 | 1':1 7:1 6:1 | 6:1 1':1 3':1 | 7:1 7:1 7:1 | 7:1 1':1 7:1 | "
               "6:1 3:1 6:1 | 6:3 | 3':1 3':1 3':1 | 4':1 3':1 2':1 | 2':1 1':1 7:1 | 1':1 3':1 5':1 | "
               "3':1 2':1 1':1 | 2':1 1':1 7:1 | 6:1 7:1 5:1 | 6:3",
        chords=['Am', 'Am', 'Dm', 'Am', 'E7', 'E7', 'Am', 'Am',
                'Am', 'Dm', 'G7', 'C', 'Am', 'Dm', 'E7', 'Am'], lyrics=''),
    'scarborough_fair': dict(title='合成演示 · 斯卡布罗集市民谣指弹练习', tonic='C', meter='3/4', capo=2, bpm=88.0, style=0, clip_duration=35.0, passing_bass=True, embellish_hp=True,
        jianpu="6,:2 6,:1 | 3:2 3:1 | 7,:1.5 1:0.5 7,:1 | 6,:3 | 3:1 3:1 5:1 | 6:2 5:1 | 3:1 4:1 2:1 | 3:3 | "
               "6:1 6:1 6:1 | 5:2 3:1 | 3:1 2:1 1:1 | 7,:2 5,:1 | 6,:2 3:1 | 2:2 1:1 | 7,:1 6,:1 5,:1 | 6,:3",
        chords=['Am', 'Am', 'G', 'Am', 'C', 'Am', 'C', 'G',
                'Am', 'Am', 'C', 'G', 'Am', 'G', 'Em7', 'Am'],
        lyrics='Are you going to Scar-bo-rough Fair Pars-ley sage rose-ma-ry and thyme '
               'Re-mem-ber me to one who lives there She once was a true love of mine'),
    'greensleeves': dict(title='合成演示 · 绿袖子古典三拍指弹练习', tonic='C', meter='3/4', capo=0, bpm=90.0, style=-1, clip_duration=34.0, passing_bass=False, embellish_hp=True,
        jianpu="6,:1 1:1 2:1 | 3:1.5 4:0.5 3:1 | 2:2 7,:1 | 5,:1.5 6,:0.5 7,:1 | 1:2 6,:1 | 6,:1.5 5,:0.5 6,:1 | "
               "7,:2 5,:1 | 3,:3 | 5:3 | 5:1.5 4:0.5 3:1 | 2:2 7,:1 | 5,:1.5 6,:0.5 7,:1 | 1:2 6,:1 | "
               "6,:1.5 5,:0.5 6,:1 | 7,:1.5 6,:0.5 5,:1 | 6,:3",
        chords=['Am', 'G', 'Am', 'E7', 'Am', 'G', 'Em7', 'Am', 'C', 'G/B', 'Am', 'E7', 'C', 'G/B', 'Em7', 'Am'],
        lyrics='A-las my love you do me wrong to cast me off dis-cour-teous-ly Green-sleeves was all my joy Green-sleeves was my de-light and my heart of gold'),
    'mandarin_pop': dict(title='合成演示 · 华语流行分解和弦弹唱练习', tonic='C', meter='4/4', capo=0, bpm=92.0, style=0, clip_duration=32.0, passing_bass=True, embellish_hp=False,
        jianpu='3:1 5:1 5:1.5 6:0.5 | 5:1 3:1 2:1 1:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 2:2 | '
               '3:1 5:1 5:1.5 6:0.5 | 5:1 3:1 2:1 1:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 1:2 | '
               "1':1 7:1 6:1 5:1 | 6:1 5:1 3:1 2:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 1:2",
        chords=['C', 'G/B', 'Am7', 'G', 'C', 'G/B', 'Am7', 'Fmaj7', 'Fmaj7', 'C/E', 'Dm7', 'C'],
        lyrics='故事的小黄花飘着童年歌谣守候着窗外的微风吹过仲夏的晴空直到永远在心间停留着不散去'),
    'real_vocal': dict(title='合成演示 · 田园民谣四拍弹唱练习', tonic='C', meter='4/4', capo=0, bpm=96.0, style=0, clip_duration=32.0, passing_bass=True, embellish_hp=True,
        jianpu='1:1 2:1 3:1 5:1 | 6:1.5 5:0.5 3:2 | 2:1 3:1 5:1 3:1 | 2:2 1:2 | 1:1 2:1 3:1 5:1 | 6:1.5 5:0.5 3:2 | 2:1 5:1 3:1 2:1 | 1:4',
        chords=['C', 'Fmaj7', 'C', 'G7', 'C', 'Fmaj7', 'G7sus4', 'C'],
        lyrics='Take the old cane pole down by the riv-er side let us go fish-in in the morn-ing sun'),
    'regular_friends': dict(title='合成演示 · 陶喆《普通朋友》R&B Neo-Soul 弹唱练习', tonic='F', meter='4/4', capo=0, bpm=82.0, style=0, clip_duration=35.0, passing_bass=True, embellish_hp=True,
        jianpu='3:1 5:1 6:1.5 5:0.5 | 3:1 2:1 1:2 | 6,:1 1:1 2:1.5 3:0.5 | 2:2 5,:2 | 3:1 5:1 6:1.5 5:0.5 | 3:1 2:1 1:2 | 2:1 3:1 2:1 1:1 | 1:4',
        chords=['Fmaj7', 'E7', 'Am7', 'Dm7', 'Gm7', 'C7', 'Fmaj7', 'Fmaj7'], lyrics='等待我随时随地在等待做你感情上的依赖我没有任何的疑问这是爱我无法只是普通朋友'),
}


def _build_session(demo_mode: str = 'mandarin_pop') -> Session:
    _refresh_backends()
    if demo_mode == 'manual':
        demo_mode = 'mandarin_pop'
    manifest = _get_repertoire_manifest()
    if demo_mode not in manifest and demo_mode not in _DEMO_PIECES:
        raise ValueError(f"Unknown demo or repertoire key: '{demo_mode}'")
    init_warnings: list[str] = []
    if demo_mode in manifest:
        entry = manifest[demo_mode]
        vocals_wav = _resolve_repertoire_path(entry.get('vocals_wav')) if isinstance(entry, dict) else None
        if vocals_wav is not None and vocals_wav.is_file():
            notes = [Note(**n) for n in entry['notes']]
            req_style = entry.get('style')
            if req_style in STYLES:
                style_name = req_style
            else:
                style_name = STYLES[0]
                if req_style:
                    init_warnings.append(f"Requested manifest style '{req_style}' is unavailable in current backend; substituted '{style_name}'")
            req_preset = entry.get('preset')
            if req_preset in PRESETS:
                preset_name = req_preset
            else:
                preset_name = 'original'
                if req_preset and req_preset != 'original':
                    init_warnings.append(f"Requested manifest preset '{req_preset}' is unavailable in current backend; substituted '{preset_name}'")
            req_intro = str(entry.get('intro_mode', 'none'))
            if req_intro not in ('none', 'nc', 'groove', 'chord_melody'):
                init_warnings.append(f"Requested manifest intro_mode '{req_intro}' is invalid; substituted 'none'")
                intro_mode = 'none'
            elif not NATIVE and req_intro != 'none':
                init_warnings.append(f"Requested manifest intro_mode '{req_intro}' requires advanced arranger; substituted 'none'")
                intro_mode = 'none'
            else:
                intro_mode = req_intro
            req_passing_bass = bool(entry.get('passing_bass', True))
            if not NATIVE and req_passing_bass:
                init_warnings.append("Requested manifest passing_bass=True requires advanced arranger; disabled in basic backend")
                passing_bass = False
            else:
                passing_bass = req_passing_bass
            req_embellish_hp = bool(entry.get('embellish_hp', style_name == '抒情琶音'))
            if not NATIVE and req_embellish_hp:
                init_warnings.append("Requested manifest embellish_hp=True requires advanced arranger; disabled in basic backend")
                embellish_hp = False
            else:
                embellish_hp = req_embellish_hp
            req_split_bar = bool(entry.get('split_bar', False))
            split_bar = bool(req_split_bar and NATIVE)
            for w in init_warnings:
                logger.warning("%s", w)
            resolved_source = _resolve_repertoire_path(entry.get('source'))
            project = Project(
                title=entry['title'],
                tonic=entry['tonic'],
                meter=entry['meter'],
                bpm=float(entry['bpm']),
                beat_origin=float(entry['beat_origin']),
                clip_duration=float(entry['clip_duration']),
                source=str(resolved_source) if resolved_source is not None else str(entry.get('source', '')),
                audio={'vocals': str(vocals_wav)},
                notes=notes,
                reference_notes=deepcopy(notes),
                bar_start_seconds=list(entry.get('bar_start_seconds', [])),
                beat_times=list(entry.get('beat_times', [])),
                warnings=list(init_warnings),
            )
            chords = list(entry['base_chords'])
            note_lyrics = list(entry['note_lyrics'])[:len(notes)]
            if len(note_lyrics) < len(notes):
                note_lyrics.extend([''] * (len(notes) - len(note_lyrics)))
            session = Session(
                project=project,
                base_chords=chords,
                custom_voicings=[None] * len(chords),
                preset=preset_name,
                style=style_name,
                capo=int(entry.get('capo', 0)),
                passing_bass=passing_bass,
                split_bar=split_bar,
                embellish_hp=embellish_hp,
                intro_mode=intro_mode,
                raw_lyrics=entry['raw_lyrics'],
                note_lyrics=note_lyrics,
                reference_note_lyrics=list(note_lyrics),
                note_ref_indices=list(range(len(notes))),
                bar_start_seconds=list(entry.get('bar_start_seconds', [])),
                beat_times=list(entry.get('beat_times', [])),
                bar_dynamics=list(entry.get('bar_dynamics', [])),
                detected_split_chords=dict(entry.get('detected_split_chords') or {}),
                use_detected_splits=True,
                word_timestamps=list(entry.get('word_timestamps', [])),
                warnings=list(init_warnings),
                demo_mode=demo_mode,
            )
            return session
        if demo_mode not in _DEMO_PIECES:
            raise ValueError(f"Manifest vocal recording for '{demo_mode}' is missing and no synthetic fallback exists")
        init_warnings.append(f"Manifest vocal recording for '{demo_mode}' is missing; using synthetic demo fallback")

    piece = _DEMO_PIECES[demo_mode]
    piece_tonic = str(piece.get('tonic', 'C'))
    notes, _ = parse_jianpu(piece['jianpu'], tonic=piece_tonic, bpm=piece['bpm'])
    raw_lyrics = str(piece.get('lyrics', ''))
    if not NATIVE:
        if piece.get('passing_bass'):
            init_warnings.append("Demo passing_bass=True requires advanced arranger; disabled in basic backend")
        if piece.get('embellish_hp'):
            init_warnings.append("Demo embellish_hp=True requires advanced arranger; disabled in basic backend")
        init_warnings.append("Demo intro_mode='chord_melody' requires advanced arranger; substituted 'none'")
    if 'note_lyrics' not in piece and raw_lyrics and not _implemented(align_lyrics_to_notes):
        init_warnings.append("Lyric alignment dependency is unavailable; per-note demo lyrics left unaligned")
    for w in init_warnings:
        logger.warning("%s", w)
    project = Project(
        title=str(piece['title']),
        tonic=piece_tonic, meter=piece['meter'], bpm=piece['bpm'], beat_origin=0.0, clip_duration=piece['clip_duration'],
        audio={},
        notes=notes, reference_notes=deepcopy(notes),
        warnings=list(init_warnings),
    )
    if 'note_lyrics' in piece:
        note_lyrics = list(piece['note_lyrics'])[:len(notes)]
        if len(note_lyrics) < len(notes):
            note_lyrics.extend([''] * (len(notes) - len(note_lyrics)))
    elif raw_lyrics and _implemented(align_lyrics_to_notes):
        note_lyrics = list(align_lyrics_to_notes(project, raw_lyrics=raw_lyrics)['note_lyrics'])
    else:
        note_lyrics = [''] * len(notes)
    chords = list(piece['chords'])
    session = Session(
        project=project,
        base_chords=chords,
        custom_voicings=[None] * len(chords),
        raw_lyrics=raw_lyrics,
        note_lyrics=note_lyrics,
        reference_note_lyrics=list(note_lyrics),
        note_ref_indices=list(range(len(notes))),
        style=STYLES[piece['style']],
        capo=piece['capo'],
        passing_bass=bool(piece['passing_bass'] and NATIVE),
        embellish_hp=bool(piece['embellish_hp'] and NATIVE),
        intro_mode='chord_melody' if NATIVE else 'none',
        warnings=list(init_warnings),
        demo_mode=demo_mode,
    )
    return session


def _initial_session(demo_mode: str = 'mandarin_pop'):
    session = _build_session(demo_mode)
    _rebuild(session)
    return session


def _active_repertoire_catalog() -> dict[str, dict]:
    manifest = _get_repertoire_manifest()
    catalog: dict[str, dict] = {}
    all_keys = list(dict.fromkeys(list(manifest.keys()) + list(_DEMO_PIECES.keys())))
    for key in all_keys:
        entry = manifest.get(key)
        vocals_wav = _resolve_repertoire_path(entry.get('vocals_wav')) if isinstance(entry, dict) else None
        if vocals_wav is not None and vocals_wav.is_file():
            title = str(entry.get('title') or key)
            dur_val = entry.get('clip_duration')
            dur_tag = f'{int(round(float(dur_val))) // 60}:{int(round(float(dur_val))) % 60:02d} ' if dur_val else ''
            subtitle = str(entry.get('subtitle') or f'{title} · {dur_tag}真人原唱全曲')
            category = str(entry.get('category') or '真人原声实录全曲')
            catalog[key] = dict(title=title, subtitle=subtitle,
                category=category,
                has_real_vocal=True,
            )
        elif key in _DEMO_PIECES:
            piece = _DEMO_PIECES[key]
            dur_s = int(round(float(piece['clip_duration'])))
            dur_str = f'{dur_s // 60}:{dur_s % 60:02d}'
            catalog[key] = dict(
                title=f"{piece['title']} ({dur_str})",
                subtitle=f'合成演示 · {dur_str} 简谱与吉他伴奏练习',
                category='合成演示练习曲',
                has_real_vocal=False,
            )
        else:
            logger.warning("Skipping repertoire key '%s': vocal recording missing and no synthetic fallback configured", key)
    return catalog


def _simulation_demos():
    if not SIMULATION_ROOT.exists():
        return dict(available=False, demos=[], reason='Pre-rendered simulation files are not installed')
    demos = []
    from urllib.parse import quote
    for path in sorted(SIMULATION_ROOT.glob('*.wav')):
        import soundfile as sf
        info = sf.info(str(path))
        if info.channels < 1 or info.samplerate <= 0 or info.frames <= 0:
            raise ValueError(f'Invalid simulation WAV: {path.name}')
        demos.append(dict(filename=path.name, label=path.stem, url='/api/simulation_audio/'+quote(path.name)))
    return dict(available=bool(demos), demos=demos, reason=None if demos else 'No simulation WAV files are installed')


def _gpu_available():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        return False
    return bool(torch.cuda.is_available())


def get_state_payload(session):
    _refresh_backends()
    project = session.project
    sim_info = _simulation_demos()
    if NATIVE and callable(recommend_artistry):
        rec = dict(recommend_artistry(project, session.base_chords, session.bar_dynamics))
        if rec.get('preset') not in PRESETS:
            rec = dict(preset=session.preset if session.preset in PRESETS else 'original', secondary_preset=None, style=session.style if session.style in STYLES else (STYLES[0] if STYLES else '分解和弦'), reason='', scores={})
        elif rec.get('style') not in STYLES:
            rec['style'] = session.style if session.style in STYLES else (STYLES[0] if STYLES else '分解和弦')
    else:
        rec = dict(preset=session.preset if session.preset in PRESETS else 'original', secondary_preset=None, style=session.style if session.style in STYLES else (STYLES[0] if STYLES else '分解和弦'), reason='', scores={})
    payload = {key: getattr(session, key) for key in ('preset', 'style', 'capo', 'passing_bass', 'split_bar',
        'use_detected_splits', 'slap_x', 'palm_mute', 'embellish_hp', 'intro_mode', 'raw_lyrics', 'bar_analysis', 'arrangement',
        'warnings', 'lyric_extraction_failed', 'harmony_synthesis_failed')}
    payload.update(title=project.title, tonic=project.tonic, meter=project.meter, bpm=float(project.bpm),
        presets=PRESETS, styles=STYLES, chords=session.arrangement['chords'],
        recommended_preset=rec['preset'], recommended_secondary_preset=rec['secondary_preset'],
        recommended_style=rec['style'], recommended_reason=rec['reason'],
        recommendation_scores=rec.get('scores', {}),
        bar_start_seconds=session.arrangement.get('bar_start_seconds', session.bar_start_seconds),
        beat_times=session.arrangement.get('beat_times', session.beat_times),
        alphatex=to_alphatex(project, session.arrangement, session.note_lyrics),
        publication_html=render_publication_score_html(project, session.arrangement, session.note_lyrics, session.bar_analysis),
        gpu_available=_gpu_available(), simulation_demos=sim_info['demos'],
        simulation_demos_available=sim_info['available'], simulation_demos_reason=sim_info['reason'],
        has_real_vocal=bool(project.audio.get('vocals') and Path(project.audio['vocals']).is_file()),
        repertoire=_active_repertoire_catalog(), active_demo=session.demo_mode,
        accompaniment_backend='advanced' if NATIVE else 'basic triads',
        synthesis_backend='native' if _implemented(synthesize_realistic_guitar) else 'fallback plucked-string synthesis')
    payload['notes_table'] = [dict(index=i, pitch=round(float(n.pitch)), pitch_name=pitch_name(n.pitch),
        numbered=numbered(n.pitch, project.tonic), jianpu=numbered(n.pitch, project.tonic),
        start=float(n.start), duration=float(n.duration),
        raw_start=float(n.raw_start), raw_duration=float(n.raw_duration), confidence=float(n.confidence),
        edited=bool(n.edited), lyric=session.note_lyrics[i]) for i, n in enumerate(project.notes)]
    return payload


def _validate_frets(frets):
    if len(frets) != 6 or any(type(f) is not int or not -1 <= f <= 24 for f in frets):
        raise ValueError('Frets must be six integers between -1 and 24')


class RequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class ReharmonizeRequest(RequestModel):
    title: str | None = None
    preset: str | None = None
    style: str | None = None
    capo: int | None = Field(default=None, ge=0, le=12, strict=True)
    passing_bass: bool | None = None
    split_bar: bool | None = None
    slap_x: bool | None = None
    palm_mute: bool | None = None
    embellish_hp: bool | None = None
    intro_mode: Literal['none', 'nc', 'groove', 'chord_melody'] | None = None
    custom_chords: dict[str, str] | None = None
    custom_voicings: list[list[Annotated[int, Field(strict=True)]] | None] | None = None
    tonic: str | None = None
    bpm: float | None = Field(default=None, ge=30, le=300)
    meter: str | None = None

    @field_validator('custom_voicings')
    @classmethod
    def _check_custom_voicings(cls, value):
        if value is not None:
            for item in value:
                if item is not None:
                    _validate_frets(item)
        return value


class BarChordRequest(RequestModel):
    bar: int | None = Field(default=None, ge=0, strict=True)
    bar_index: int | None = Field(default=None, ge=0, strict=True)
    chord: str | None = None
    segment: int | None = Field(default=None, ge=0, strict=True)
    frets: list[Annotated[int, Field(strict=True)]] | None = None

    @field_validator('frets')
    @classmethod
    def _check_frets(cls, v):
        if v is not None:
            _validate_frets(v)
        return v


class LyricRequest(RequestModel):
    raw_lyrics: str
    word_timestamps: list[dict] | None = None
    mode: str | None = None


class NoteRequest(RequestModel):
    index: int = Field(ge=0, strict=True)
    pitch: int | None = Field(default=None, ge=21, le=108, strict=True)
    start: float | None = Field(default=None, ge=0)
    duration: float | None = Field(default=None, gt=0)
    lyric: str | None = None


class CleanupRequest(RequestModel):
    mode: Literal['auto', 'binary', 'triplet'] = 'auto'


class DemoRequest(RequestModel):
    demo: str = 'manual'

    @field_validator('demo')
    @classmethod
    def _check_demo(cls, v: str) -> str:
        if v == 'manual' or v in _DEMO_PIECES or v in _get_repertoire_manifest():
            return v
        raise ValueError(f"Unknown demo or repertoire key: '{v}'")


def _synthesize_event_list(project, arrangement, events, sr=44100):
    _refresh_backends()
    arr = dict(arrangement, events=events)
    if _implemented(synthesize_realistic_guitar):
        return np.asarray(synthesize_realistic_guitar(project, arr, out_wav_path=None, sr=sr), dtype=np.float32)
    spb = 60.0 / float(project.bpm)
    measure_beats = float(arrangement.get('measure_beats', 4.0))
    raw_bss = arrangement.get('bar_start_seconds') or getattr(project, 'bar_start_seconds', None) or []
    raw_bt = arrangement.get('beat_times') or getattr(project, 'beat_times', None) or []
    has_warp = len(raw_bss) >= 2
    has_beat_warp = len(raw_bt) >= 2
    if arrangement.get('time_origin_shifted'):
        warp_origin = 0.0
    elif has_beat_warp and has_warp:
        warp_origin = min(float(raw_bt[0]), float(raw_bss[0]))
    elif has_beat_warp:
        warp_origin = float(raw_bt[0])
    elif has_warp:
        warp_origin = float(raw_bss[0])
    else:
        warp_origin = 0.0
    bss = [float(x) - warp_origin for x in raw_bss] if has_warp else []
    avg_bar_dur = ((bss[-1] - bss[0]) / max(1, len(bss) - 1)) if has_warp else measure_beats * spb
    bt = [float(x) - warp_origin for x in raw_bt] if has_beat_warp else []
    bt_avg_dur = ((bt[-1] - bt[0]) / max(1, len(bt) - 1)) if has_beat_warp else spb

    def _beat_to_sec(beat_pos: float) -> float:
        if has_beat_warp:
            max_idx = len(bt) - 1
            if beat_pos <= max_idx:
                return float(np.interp(beat_pos, np.arange(len(bt), dtype=np.float64), bt))
            return bt[-1] + (beat_pos - max_idx) * bt_avg_dur
        if not has_warp:
            return beat_pos * spb
        b_idx = max(0, int(beat_pos // measure_beats))
        local_beat = beat_pos - b_idx * measure_beats
        if b_idx + 1 < len(bss):
            bar_t0, bar_t1 = bss[b_idx], bss[b_idx + 1]
        elif b_idx < len(bss):
            bar_t0, bar_t1 = bss[b_idx], bss[b_idx] + avg_bar_dur
        else:
            bar_t0 = bss[-1] + (b_idx - len(bss) + 1) * avg_bar_dur
            bar_t1 = bar_t0 + avg_bar_dur
        bar_dur = max(0.0, bar_t1 - bar_t0)
        return bar_t0 + (local_beat / measure_beats) * bar_dur

    end = max((_beat_to_sec(float(e['start'] + e['duration'])) for e in events), default=0.0)
    samples = np.zeros((math.ceil((end + 0.25) * sr), 2), dtype=np.float32)
    for event in events:
        start_sec = _beat_to_sec(float(event['start']))
        end_sec = _beat_to_sec(float(event['start'] + event['duration']))
        duration = max(1e-3, end_sec - start_sec)
        technique = event.get('technique', '')
        count = max(1, round((min(duration, 0.09) if technique == 'X' else duration + 0.08) * sr))
        t = np.arange(count, dtype=np.float64) / sr
        signal = np.zeros(count, dtype=np.float64)
        seed = (int(round(float(event['start']) * 1000)) * 31 + int(event['pitch']) * 7 + int(event['string'])) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        if technique == 'X':
            noise = rng.standard_normal(count)
            signal = np.convolve(noise, np.ones(5) / 5, mode='same') * np.exp(-t * 65) * 0.2
        else:
            frequency = 440 * 2 ** ((event['pitch'] - 69) / 12)
            for harmonic in range(1, 13):
                if harmonic * frequency >= sr / 2:
                    break
                signal += np.sin(2 * np.pi * frequency * harmonic * t) * np.exp(-t * harmonic * 0.8) / harmonic ** 1.5
            signal *= np.exp(-t * (16 if technique == 'PM' else 3.2)) * 0.17
            if technique not in ('H', 'P'):
                signal += rng.standard_normal(count) * np.exp(-t * 220) * 0.035
        attack = min(count, max(1, round((0.012 if technique in ('H', 'P') else 0.002) * sr)))
        signal[:attack] *= np.linspace(0, 1, attack)
        release = min(count, max(1, round(0.04 * sr)))
        signal[-release:] *= np.linspace(1, 0, release)
        signal *= event['velocity'] / 127
        offset = max(0, round(start_sec * sr))
        if offset + count > len(samples):
            samples = np.pad(samples, ((0, offset + count - len(samples)), (0, 0)))
        # Standard guitar string numbering: string 6 is low E (bass, left-leaning), string 1 is high E (treble, right-leaning).
        treble_pan = (6 - event['string']) / 5.0
        samples[offset:offset + count, 0] += (signal * math.sqrt(0.8 - 0.6 * treble_pan)).astype(np.float32)
        samples[offset:offset + count, 1] += (signal * math.sqrt(0.2 + 0.6 * treble_pan)).astype(np.float32)
    return samples


def _guitar_samples(session, sr=44100):
    cached = session._cached_guitar_samples
    if cached is not None and cached[0] == sr:
        return cached[1]
    samples = _synthesize_event_list(session.project, session.arrangement, session.arrangement['events'], sr=sr)
    if samples.ndim != 2 or samples.shape[1] != 2 or not np.isfinite(samples).all():
        raise ValueError('Expected finite stereo guitar audio')
    samples.flags.writeable = False
    session._cached_guitar_samples = (sr, samples)
    return samples


def _wav_bytes(samples, sr=44100):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError('Expected stereo audio')
    if not np.isfinite(samples).all():
        raise ValueError('Audio contains non-finite values')
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > .98:
        samples = samples*(.98/peak)
    pcm = np.rint(np.clip(samples, -1, 1)*32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _load_stem_audio(path: str | Path, sr: int, origin_sec: float) -> np.ndarray:
    import hashlib, soundfile as sf
    from scipy.signal import resample_poly
    origin, raw_bytes = round(float(origin_sec) * sr), Path(path).read_bytes()
    key = (str(path), sr, origin, len(raw_bytes), hashlib.blake2b(raw_bytes, digest_size=16).digest())
    cached = _STEM_AUDIO_CACHE.get(key)
    if cached is not None:
        return cached
    audio, source_sr = sf.read(str(path), always_2d=True, dtype='float32')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        n_ch = audio.shape[1]
        left = np.mean(audio[:, 0::2], axis=1)
        right = np.mean(audio[:, 1::2], axis=1)
        mono_mean = np.mean(audio, axis=1)
        audio = np.column_stack([0.5 * (left + mono_mean), 0.5 * (right + mono_mean)]).astype('float32')
        logger.warning("Downmixed %d-channel stem audio '%s' to stereo", n_ch, path)
    if source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        audio = resample_poly(audio, sr // divisor, source_sr // divisor).astype('float32')
    if origin >= 0:
        audio = np.ascontiguousarray(audio[origin:], dtype=np.float32)
    else:
        audio = np.ascontiguousarray(np.pad(audio, ((-origin, 0), (0, 0))), dtype=np.float32)
    audio.flags.writeable = False
    if len(_STEM_AUDIO_CACHE) >= 24:
        _STEM_AUDIO_CACHE.pop(next(iter(_STEM_AUDIO_CACHE)), None)
    _STEM_AUDIO_CACHE[key] = audio
    return audio


def _calibrate_lead_vocal_stem(vocal: np.ndarray, sr: int) -> np.ndarray:
    """Shape real singing stems (>= 20 active 50ms post-HPF frames) with an 80 Hz rumble cut, bounded makeup gain (<= 4.0x toward -16.2 dBFS active RMS), subtle early room air, and soft-knee peak limiting within [-0.94, 0.94]; scale shorter clips linearly by 0.85x."""
    vocal_contig = np.ascontiguousarray(vocal, dtype=np.float32)
    key = (vocal_contig.shape, int(sr), hashlib.blake2b(vocal_contig.tobytes(), digest_size=16).digest())
    cached = _VOCAL_CAL_CACHE.get(key)
    if cached is not None:
        return cached
    win = max(1, round(0.050 * sr))
    n_frames = len(vocal) // win
    if n_frames >= 20:
        from scipy.signal import butter, sosfilt
        hp = butter(2, min(0.45 * sr, 80.0), btype='highpass', fs=sr, output='sos')
        shaped = np.empty_like(vocal, dtype=np.float64)
        shaped[:, 0] = sosfilt(hp, vocal[:, 0].astype(np.float64))
        shaped[:, 1] = sosfilt(hp, vocal[:, 1].astype(np.float64))
        ms_frames = np.mean(shaped[: n_frames * win] ** 2, axis=1).reshape(n_frames, win)
        f_rms = np.sqrt(np.mean(ms_frames, axis=1))
        max_r = float(np.max(f_rms)) if f_rms.size else 0.0
        thresh = max(10.0 ** (-48.0 / 20.0), 0.10 * max_r)
        active = f_rms[f_rms >= thresh]
        if active.size >= 20:
            act_rms = float(np.sqrt(np.mean(active ** 2)))
            target_rms = 10.0 ** (-16.2 / 20.0)
            gain = min(4.0, target_rms / max(1e-6, act_rms))
            shaped *= gain
            # Subtle studio vocal room air (-27 dB at 21ms L / 34ms R) so separated vocals sit naturally with the wooden guitar
            d_l, d_r = round(0.021 * sr), round(0.034 * sr)
            air_gain = 10.0 ** (-27.0 / 20.0)
            mono_wet = shaped.mean(axis=1)
            if d_l < len(shaped):
                shaped[d_l:, 0] += air_gain * mono_wet[: len(shaped) - d_l]
            if d_r < len(shaped):
                shaped[d_r:, 1] += air_gain * mono_wet[: len(shaped) - d_r]
            # Smooth soft-knee peak ceiling above 0.70 preventing hard clipping within [-0.94, 0.94]
            abs_v = np.abs(shaped)
            over = abs_v > 0.70
            if np.any(over):
                shaped[over] = np.sign(shaped[over]) * (0.70 + 0.22 * np.tanh((abs_v[over] - 0.70) / 0.22))
            out = np.clip(shaped, -0.94, 0.94).astype(np.float32)
            out.flags.writeable = False
            if len(_VOCAL_CAL_CACHE) >= 24:
                _VOCAL_CAL_CACHE.pop(next(iter(_VOCAL_CAL_CACHE)), None)
            _VOCAL_CAL_CACHE[key] = out
            return out
    out = (vocal * np.float32(0.85)).astype(np.float32)
    out.flags.writeable = False
    return out


def _build_lead_events(session) -> list[dict]:
    notes = session.project.notes
    if not notes:
        return []
    capo = int(session.capo)
    open_tuning = ((1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40))
    min_playable = 40 + capo
    max_playable = 64 + capo + 24
    measure = float(session.arrangement.get('measure_beats', 4.0))
    lead_events: list[dict] = []
    for n in notes:
        # Fold out-of-range notes by octaves into the capoed fretboard range [40 + capo, 64 + capo + 24]
        # so every event satisfies pitch == open_midi + capo + fret with 0 <= fret <= 24.
        pitch = round(float(n.pitch))
        while pitch < min_playable:
            pitch += 12
        while pitch > max_playable:
            pitch -= 12
        string, open_midi = 6, 40
        for s_num, s_midi in open_tuning:
            if s_midi + capo <= pitch <= s_midi + capo + 24:
                string, open_midi = s_num, s_midi
                break
        fret = pitch - (open_midi + capo)
        beat_in_bar = float(n.start % measure)
        vel = round(88 + (6 if abs(beat_in_bar) < 0.1 else 0) + (4 if n.duration >= 1.5 else -3))
        start = float(n.start) + (0.015 if abs(beat_in_bar) < 0.05 else 0.0)
        lead_events.append(dict(
            bar=int(n.start // measure), beat=beat_in_bar, start=start,
            duration=float(n.duration), string=string, fret=fret, pitch=pitch,
            velocity=vel, technique='CM',
        ))
    return lead_events


def _stem_origin_sec(session) -> float:
    arr = session.arrangement if isinstance(session.arrangement, dict) else {}
    if 'warp_origin' in arr and arr.get('warp_beat_origin') == float(session.project.beat_origin):
        return float(arr['warp_origin'])
    return float(session.project.beat_origin)


def _vocal_only_samples(session, sr=44100):
    path = session.project.audio.get('vocals')
    guitar_len = len(_guitar_samples(session, sr))
    if not path or not Path(path).is_file():
        lead_events = _build_lead_events(session)
        if not lead_events:
            raise HTTPException(404, 'No vocal recording or melody notes available for vocal-only audio')
        lead_arr = dict(session.arrangement, bars=[], chords=[])
        lead_samples = _synthesize_event_list(session.project, lead_arr, lead_events, sr=sr)
        length = max(len(lead_samples), guitar_len)
        return np.pad(lead_samples, ((0, length - len(lead_samples)), (0, 0))) * 0.65
    vocal = _load_stem_audio(path, sr, _stem_origin_sec(session))
    v_cal = _calibrate_lead_vocal_stem(vocal, sr)
    length = max(len(v_cal), guitar_len)
    return np.pad(v_cal[:length], ((0, max(0, length - len(v_cal))), (0, 0)))


def _vocal_mix_samples(session, sr=44100):
    path = session.project.audio.get('vocals')
    guitar_samples = _guitar_samples(session, sr)
    if not path or not Path(path).is_file():
        lead_events = _build_lead_events(session)
        if not lead_events:
            raise HTTPException(409, 'A vocal recording or melody notes are required for the vocal/guitar mix')
        lead_arr = dict(session.arrangement, bars=[], chords=[])
        lead_samples = _synthesize_event_list(session.project, lead_arr, lead_events, sr=sr)
        length = max(len(lead_samples), len(guitar_samples))
        return (np.pad(lead_samples, ((0, length - len(lead_samples)), (0, 0))) * 0.65 +
                np.pad(guitar_samples, ((0, length - len(guitar_samples)), (0, 0))) * 0.82)
    v_only = _vocal_only_samples(session, sr=sr)
    length = max(len(v_only), len(guitar_samples))
    return (np.pad(v_only[:length], ((0, max(0, length - len(v_only))), (0, 0))) * np.float32(0.7 / 0.85) +
            np.pad(guitar_samples[:length], ((0, max(0, length - len(guitar_samples))), (0, 0))) * np.float32(0.85))


def create_app() -> FastAPI:
    application = FastAPI(title='MelodyDesk 2.0 · AI 吉他弹唱与高阶改谱工作台')
    application.state.session = _initial_session()
    application.state.session_lock = threading.RLock()
    application.state.project_generation = 0
    application.state.upload_generation = 0

    def snapshot():
        with application.state.session_lock:
            return _snapshot(application.state.session)

    def audio_snapshot():
        # Ensure _cached_guitar_samples is populated on the live session under the lock, then return a snapshot.
        with application.state.session_lock:
            _guitar_samples(application.state.session)
            return _snapshot(application.state.session)

    def mutate(operation, bump_generation: bool = False, bump_upload_generation: bool = False):
        with application.state.session_lock:
            previous = application.state.session
            candidate = _snapshot(previous)
            try:
                operation(candidate)
                _rebuild(candidate)
                payload = get_state_payload(candidate)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if bump_generation:
                application.state.project_generation += 1
            if bump_upload_generation:
                application.state.upload_generation += 1
            candidate.undo_stack = (previous.undo_stack+[_snapshot(previous)])[-30:]
            application.state.session = candidate
            return payload

    @application.get('/')
    def index():
        if (STATIC_ROOT/'index.html').is_file():
            return FileResponse(STATIC_ROOT/'index.html')
        return HTMLResponse(_fallback_workbench())

    @application.get('/evolution_atlas')
    def evolution_atlas():
        atlas_path = STATIC_ROOT / 'evolution_atlas.html'
        if atlas_path.is_file():
            return FileResponse(atlas_path)
        raise HTTPException(404, 'Evolution atlas page is not installed')

    @application.get('/api/state')
    def state():
        return get_state_payload(snapshot())

    @application.post('/api/reharmonize')
    def reharmonize(request: ReharmonizeRequest):
        def operation(candidate):
            old_m_parts = str(candidate.project.meter).strip().split('/')
            old_meas = (
                int(old_m_parts[0]) * 4.0 / int(old_m_parts[1])
                if len(old_m_parts) == 2 and old_m_parts[0].isdigit() and old_m_parts[1].isdigit() and int(old_m_parts[1]) > 0
                else 4.0
            )
            old_bpm = float(candidate.project.bpm)
            preset_changed = 'preset' in request.model_fields_set and request.preset != candidate.preset
            capo_changed = 'capo' in request.model_fields_set and request.capo != candidate.capo
            tonic_changed = 'tonic' in request.model_fields_set and request.tonic != candidate.project.tonic
            split_changed = 'split_bar' in request.model_fields_set and request.split_bar != candidate.split_bar
            bpm_changed = 'bpm' in request.model_fields_set and request.bpm != candidate.project.bpm
            meter_changed = 'meter' in request.model_fields_set and request.meter != candidate.project.meter
            prev_use_det = bool(candidate.use_detected_splits)
            if request.model_fields_set == {'split_bar'} or (split_changed and not preset_changed):
                new_use_det = bool(request.split_bar)
            elif ('split_bar' in request.model_fields_set and bool(request.split_bar)) or preset_changed:
                new_use_det = True
            else:
                new_use_det = prev_use_det
            det_split_changed = (new_use_det != prev_use_det)
            prior_bt = list(candidate.beat_times)
            if (preset_changed or capo_changed or tonic_changed or split_changed or det_split_changed or bpm_changed or meter_changed) and 'custom_voicings' not in request.model_fields_set:
                candidate.custom_voicings = [None]*len(candidate.base_chords)
                candidate.custom_segment_voicings.clear()
            if (preset_changed or tonic_changed or split_changed or det_split_changed or bpm_changed or meter_changed) and 'custom_chords' not in request.model_fields_set:
                candidate.custom_chords.clear()
            candidate.use_detected_splits = new_use_det
            if bpm_changed or meter_changed:
                candidate.detected_split_chords = {}
            for key in request.model_fields_set:
                value = getattr(request, key)
                if value is None:
                    raise ValueError(f'{key} cannot be null')
                if key == 'custom_voicings':
                    candidate._voicings_explicit = bool(value)
                    candidate._voicings_reset = (value == [])
                    for frets in value:
                        if frets is not None:
                            _validate_frets(frets)
                if key == 'custom_chords':
                    import re; candidate._chords_explicit = bool(value)
                    if any(re.fullmatch(r'\d+(?::\d+)?', identity) is None for identity in value):
                        raise ValueError('Chord override keys must be bar or bar:segment indices')
                    if 'custom_voicings' not in request.model_fields_set:
                        for identity in value:
                            b_idx = int(identity.split(':')[0])
                            if 0 <= b_idx < len(candidate.custom_voicings):
                                candidate.custom_voicings[b_idx] = None
                            for seg_key in [k for k in candidate.custom_segment_voicings if k == str(b_idx) or k.startswith(f'{b_idx}:')]:
                                candidate.custom_segment_voicings.pop(seg_key, None)
                setattr(candidate.project if key in ('title', 'tonic', 'bpm', 'meter') else candidate, key, value)

            has_recording_grid = bool(candidate.project.source) or bool(candidate.project.audio.get('vocals'))
            if meter_changed or (bpm_changed and has_recording_grid):
                new_m_parts = str(candidate.project.meter).strip().split('/')
                if (
                    len(new_m_parts) == 2
                    and new_m_parts[0].isdigit()
                    and new_m_parts[1].isdigit()
                    and int(new_m_parts[0]) > 0
                    and int(new_m_parts[1]) in (2, 4, 8)
                ):
                    new_meas_g = int(new_m_parts[0]) * 4.0 / int(new_m_parts[1])
                    bpm_ratio = (
                        (old_bpm / float(candidate.project.bpm))
                        if (bpm_changed and has_recording_grid and float(candidate.project.bpm) > 0)
                        else 1.0
                    )
                    bar_scale = (new_meas_g * bpm_ratio) / max(1e-6, old_meas)
                    if abs(bar_scale - 1.0) > 1e-9:
                        def _resample_bar_seq(seq: list) -> list:
                            if not seq:
                                return []
                            n_out = max(1, math.ceil(len(seq) / bar_scale - 1e-9))
                            out = []
                            for b_i in range(n_out):
                                lo_u = b_i * bar_scale
                                hi_u = (b_i + 1) * bar_scale
                                k0 = max(0, min(len(seq) - 1, int(math.floor(lo_u + 1e-9))))
                                k1 = max(k0, min(len(seq) - 1, int(math.ceil(hi_u - 1e-9)) - 1))
                                best_k = max(
                                    range(k0, k1 + 1),
                                    key=lambda k: (round(min(hi_u, k + 1.0) - max(lo_u, float(k)), 9), -k),
                                )
                                out.append(seq[best_k])
                            return out

                        candidate.base_chords = _resample_bar_seq(candidate.base_chords)
                        if candidate.bar_dynamics:
                            candidate.bar_dynamics = _resample_bar_seq(candidate.bar_dynamics)
                        if 'custom_voicings' not in request.model_fields_set:
                            candidate.custom_voicings = [None] * len(candidate.base_chords)

            if bpm_changed:
                candidate.bar_start_seconds = []
                candidate.beat_times = []
                has_recording = bool(candidate.project.source) or bool(candidate.project.audio.get('vocals'))
                if has_recording and (candidate.project.notes or candidate.project.reference_notes):
                    new_bpm = float(candidate.project.bpm)
                    origin = float(candidate.project.beat_origin)
                    tick = 1.0 / 48.0

                    def _resync_run(indexed_run, min_b: float, max_b: float | None):
                        if not indexed_run:
                            return [], []
                        ordered_pairs = sorted(indexed_run, key=lambda pair: float(pair[1].raw_start))
                        perm_indices = [pair[0] for pair in ordered_pairs]
                        run_notes = [pair[1] for pair in ordered_pairs]
                        try:
                            return organize(
                                run_notes, new_bpm, origin, 'auto',
                                beat_times=None, min_beat=min_b, max_beat=max_b,
                            ), perm_indices
                        except ValueError:
                            out_run = deepcopy(run_notes)
                            cur_min = max(0.0, float(min_b))
                            n_r = len(out_run)
                            for r_i, rn in enumerate(out_run):
                                rem = n_r - 1 - r_i
                                raw_b = max(cur_min, (float(rn.raw_start) - origin) * new_bpm / 60.0)
                                q_b = round(raw_b * 4.0) / 4.0
                                if q_b < cur_min - 1e-9 or (max_b is not None and q_b + (rem + 1) * 0.25 > max_b + 1e-9):
                                    q_b = max(cur_min, round(raw_b / tick) * tick)
                                if max_b is not None:
                                    q_b = min(q_b, max(cur_min, max_b - (rem + 1) * tick))
                                rn.start = round(q_b, 6)
                                cur_min = rn.start + tick
                            for r_i, rn in enumerate(out_run):
                                lim = (out_run[r_i + 1].start - rn.start) if r_i + 1 < n_r else (
                                    (max_b - rn.start) if max_b is not None else max(0.25, float(rn.raw_duration) * new_bpm / 60.0)
                                )
                                raw_d = max(tick, float(rn.raw_duration) * new_bpm / 60.0)
                                q_d = round(raw_d * 4.0) / 4.0
                                if q_d <= 0 or q_d > lim + 1e-9:
                                    q_d = min(lim, max(tick, round(raw_d / tick) * tick))
                                rn.duration = round(max(tick, min(lim, q_d)), 6)
                            return out_run, perm_indices

                    if candidate.project.reference_notes:
                        ref_resynced, ref_perm = _resync_run(
                            [(i_r, deepcopy(rn)) for i_r, rn in enumerate(candidate.project.reference_notes)],
                            0.0, None,
                        )
                        for rn_item in ref_resynced:
                            rn_item.edited = False
                        candidate.project.reference_notes = ref_resynced
                        if ref_perm != list(range(len(ref_perm))):
                            if len(candidate.reference_note_lyrics) == len(ref_perm):
                                candidate.reference_note_lyrics = [candidate.reference_note_lyrics[i] for i in ref_perm]
                            inv_ref = {old_r: new_r for new_r, old_r in enumerate(ref_perm)}
                            if candidate.custom_note_lyrics:
                                candidate.custom_note_lyrics = {
                                    inv_ref[k]: v for k, v in candidate.custom_note_lyrics.items() if k in inv_ref
                                }
                            if candidate.note_ref_indices:
                                candidate.note_ref_indices = [inv_ref.get(r_i, r_i) for r_i in candidate.note_ref_indices]

                    if candidate.project.notes:
                        resynced, resync_perm, run_buf, min_bound = [], [], [], 0.0
                        for idx_cur, n_cur in enumerate(candidate.project.notes):
                            if n_cur.edited:
                                r_notes, r_perm = _resync_run(run_buf, min_bound, float(n_cur.start))
                                resynced.extend(r_notes)
                                resync_perm.extend(r_perm)
                                run_buf = []
                                resynced.append(n_cur)
                                resync_perm.append(idx_cur)
                                min_bound = float(n_cur.start + n_cur.duration)
                            else:
                                run_buf.append((idx_cur, n_cur))
                        r_notes, r_perm = _resync_run(run_buf, min_bound, None)
                        resynced.extend(r_notes)
                        resync_perm.extend(r_perm)
                        if len(candidate.note_lyrics) == len(resync_perm):
                            candidate.note_lyrics = [candidate.note_lyrics[i] for i in resync_perm]
                        if len(candidate.note_ref_indices) == len(resync_perm):
                            candidate.note_ref_indices = [candidate.note_ref_indices[i] for i in resync_perm]
                        candidate.project.notes = resynced
                        if (candidate.raw_lyrics.strip() or candidate.word_timestamps) and any(n.edited for n in resynced):
                            _align(candidate)
                        elif candidate.custom_note_lyrics and len(candidate.note_ref_indices) == len(candidate.note_lyrics):
                            for i_c, ref_idx in enumerate(candidate.note_ref_indices):
                                if ref_idx in candidate.custom_note_lyrics:
                                    candidate.note_lyrics[i_c] = candidate.custom_note_lyrics[ref_idx]
            elif meter_changed:
                m_parts = str(candidate.project.meter).strip().split('/')
                valid_m = (
                    len(m_parts) == 2
                    and m_parts[0].isdigit()
                    and m_parts[1].isdigit()
                    and int(m_parts[0]) > 0
                    and int(m_parts[1]) in (2, 4, 8)
                )
                if valid_m and len(prior_bt) >= 2 and callable(beats_to_seconds_piecewise):
                    new_meas = int(m_parts[0]) * 4.0 / int(m_parts[1])
                    origin = float(candidate.project.beat_origin)
                    cur_bpm = float(candidate.project.bpm)
                    o_b = float(seconds_to_beats_piecewise(origin, prior_bt, cur_bpm, origin)) if callable(seconds_to_beats_piecewise) else 0.0
                    max_end_b = max((n.start + n.duration for n in candidate.project.notes), default=0.0)
                    if candidate.project.source and candidate.project.clip_duration > origin and callable(seconds_to_beats_piecewise):
                        c_b = float(seconds_to_beats_piecewise(candidate.project.clip_duration, prior_bt, cur_bpm, origin)) - o_b
                        max_end_b = max(max_end_b, c_b)
                    n_bars_m = max(1, math.ceil(max_end_b / new_meas))
                    candidate.beat_times = prior_bt
                    raw_bars = [
                        round(float(beats_to_seconds_piecewise(o_b + b * new_meas, prior_bt, cur_bpm, origin)), 4)
                        for b in range(n_bars_m + 1)
                    ]
                    for b_i in range(1, len(raw_bars)):
                        if raw_bars[b_i] <= raw_bars[b_i - 1]:
                            raw_bars[b_i] = round(raw_bars[b_i - 1] + 1e-4, 4)
                    candidate.bar_start_seconds = raw_bars
                else:
                    candidate.bar_start_seconds = []
                    candidate.beat_times = []
        return mutate(operation)

    @application.post('/api/bar_chord')
    def bar_chord(request: BarChordRequest):
        def operation(candidate):
            if request.bar is not None and request.bar_index is not None and request.bar != request.bar_index:
                raise ValueError('Conflicting bar and bar_index values')
            bar_idx = request.bar if request.bar is not None else request.bar_index
            if bar_idx is None:
                raise ValueError('Bar index is required')
            if bar_idx >= len(candidate.arrangement['bars']):
                raise ValueError('Bar index is out of range')
            if request.segment is not None and request.segment >= len(candidate.arrangement['bars'][bar_idx]['split_segments']):
                raise ValueError('Segment index is out of range')
            identity = str(bar_idx)+(f':{request.segment}' if request.segment is not None else '')
            if request.chord is not None:
                if request.segment is None:
                    for seg_key in [k for k in candidate.custom_chords if k.startswith(f'{bar_idx}:')]:
                        candidate.custom_chords.pop(seg_key, None)
                candidate.custom_chords[identity] = request.chord
                if request.frets is None:
                    if request.segment is None:
                        if bar_idx < len(candidate.custom_voicings):
                            candidate.custom_voicings[bar_idx] = None
                        for seg_key in [k for k in candidate.custom_segment_voicings if k == str(bar_idx) or k.startswith(f'{bar_idx}:')]:
                            candidate.custom_segment_voicings.pop(seg_key, None)
                    else:
                        if request.segment == 0 and bar_idx < len(candidate.custom_voicings):
                            candidate.custom_voicings[bar_idx] = None
                        candidate.custom_segment_voicings.pop(identity, None)
            if request.frets is not None:
                _validate_frets(request.frets)
                if request.segment is None:
                    candidate.custom_voicings[bar_idx] = list(request.frets)
                    for seg_key in [k for k in candidate.custom_segment_voicings if k == str(bar_idx) or k.startswith(f'{bar_idx}:')]:
                        candidate.custom_segment_voicings.pop(seg_key, None)
                else:
                    candidate.custom_segment_voicings[identity] = list(request.frets)
        return mutate(operation)

    @application.post('/api/align_lyrics')
    def lyrics(request: LyricRequest):
        def operation(candidate):
            candidate.custom_note_lyrics.clear()
            candidate.raw_lyrics = request.raw_lyrics
            if request.word_timestamps is not None:
                candidate.word_timestamps = deepcopy(request.word_timestamps)
            elif not request.raw_lyrics:
                candidate.word_timestamps = []
            if not request.raw_lyrics and not candidate.word_timestamps:
                candidate.note_lyrics = [''] * len(candidate.project.notes)
                candidate.reference_note_lyrics = [''] * len(candidate.project.reference_notes)
            else:
                _align(candidate, request.word_timestamps, require_backend=True)
                if len(candidate.note_ref_indices) == len(candidate.project.notes):
                    if len(candidate.reference_note_lyrics) != len(candidate.project.reference_notes):
                        candidate.reference_note_lyrics = [''] * len(candidate.project.reference_notes)
                    for i, ref_idx in enumerate(candidate.note_ref_indices):
                        if 0 <= ref_idx < len(candidate.reference_note_lyrics) and i < len(candidate.note_lyrics):
                            candidate.reference_note_lyrics[ref_idx] = candidate.note_lyrics[i]
        return mutate(operation)

    @application.post('/api/edit_note')
    def edit_note(request: NoteRequest):
        def operation(candidate):
            if request.index >= len(candidate.project.notes):
                raise ValueError('Note index is out of range')
            if len(candidate.note_ref_indices) != len(candidate.project.notes):
                candidate.note_ref_indices = list(range(len(candidate.project.notes)))
            if len(candidate.reference_note_lyrics) != len(candidate.project.reference_notes):
                candidate.reference_note_lyrics = [''] * len(candidate.project.reference_notes)
            for i, ref_idx in enumerate(candidate.note_ref_indices):
                if 0 <= ref_idx < len(candidate.reference_note_lyrics) and i < len(candidate.note_lyrics):
                    candidate.reference_note_lyrics[ref_idx] = candidate.note_lyrics[i]
            note = candidate.project.notes[request.index]
            prior_by_note = {
                id(n): (
                    candidate.note_lyrics[i] if i < len(candidate.note_lyrics) else '',
                    candidate.note_ref_indices[i] if i < len(candidate.note_ref_indices) else i,
                )
                for i, n in enumerate(candidate.project.notes)
            }
            note_changed = False
            for key in request.model_fields_set - {'index', 'lyric'}:
                value = getattr(request, key)
                if value is None:
                    if 'lyric' in request.model_fields_set and request.lyric is not None:
                        continue
                    raise ValueError(f'{key} cannot be null')
                setattr(note, key, value)
                note_changed = True
            target_idx = request.index
            if note_changed:
                note.edited = True
                candidate.project.validate()
                target_idx = next(
                    (i for i, n in enumerate(candidate.project.notes) if n is note),
                    request.index,
                )
                candidate.note_ref_indices = [
                    prior_by_note.get(id(n), ('', i))[1]
                    for i, n in enumerate(candidate.project.notes)
                ]
                if ('start' in request.model_fields_set or 'duration' in request.model_fields_set) and (candidate.raw_lyrics or candidate.word_timestamps) and _implemented(align_lyrics_to_notes):
                    _align(candidate)
                else:
                    candidate.note_lyrics = [
                        candidate.custom_note_lyrics.get(candidate.note_ref_indices[i], prior_by_note.get(id(n), ('', i))[0])
                        for i, n in enumerate(candidate.project.notes)
                    ]
            if 'lyric' in request.model_fields_set and request.lyric is not None:
                if len(candidate.note_lyrics) != len(candidate.project.notes):
                    candidate.note_lyrics = [''] * len(candidate.project.notes)
                candidate.note_lyrics[target_idx] = str(request.lyric)
                if target_idx < len(candidate.note_ref_indices):
                    ref_idx = candidate.note_ref_indices[target_idx]
                    candidate.custom_note_lyrics[ref_idx] = str(request.lyric)
                    if 0 <= ref_idx < len(candidate.reference_note_lyrics):
                        candidate.reference_note_lyrics[ref_idx] = str(request.lyric)
        return mutate(operation)

    @application.get('/api/export/score.html')
    def export_score_html():
        cur = snapshot()
        html = render_publication_score_html(cur.project, cur.arrangement, cur.note_lyrics, cur.bar_analysis)
        return HTMLResponse(html)

    @application.get('/api/export/alphatex.txt')
    def export_alphatex_txt():
        cur = snapshot()
        tex = to_alphatex(cur.project, cur.arrangement, cur.note_lyrics)
        return Response(content=tex.encode('utf-8'), media_type='text/plain; charset=utf-8')

    @application.post('/api/cleanup')
    def cleanup(request: CleanupRequest = Body(default=CleanupRequest())):
        def operation(candidate):
            if len(candidate.note_ref_indices) != len(candidate.project.notes):
                candidate.note_ref_indices = list(range(len(candidate.project.notes)))
            if len(candidate.reference_note_lyrics) != len(candidate.project.reference_notes):
                candidate.reference_note_lyrics = [''] * len(candidate.project.reference_notes)
            for i, ref_idx in enumerate(candidate.note_ref_indices):
                if 0 <= ref_idx < len(candidate.reference_note_lyrics) and i < len(candidate.note_lyrics):
                    candidate.reference_note_lyrics[ref_idx] = candidate.note_lyrics[i]
            prior_by_id = {
                id(n): (
                    candidate.note_lyrics[i] if i < len(candidate.note_lyrics) else '',
                    candidate.note_ref_indices[i] if i < len(candidate.note_ref_indices) else i,
                )
                for i, n in enumerate(candidate.project.notes)
            }
            prior_by_raw = [
                (
                    float(n.raw_start),
                    candidate.note_lyrics[i] if i < len(candidate.note_lyrics) else '',
                    candidate.note_ref_indices[i] if i < len(candidate.note_ref_indices) else i,
                )
                for i, n in enumerate(candidate.project.notes)
            ]
            bt_map = candidate.beat_times if len(candidate.beat_times) >= 2 else None
            notes, run = [], []
            min_b = 0.0
            for note in candidate.project.notes:
                if note.edited:
                    if run:
                        notes.extend(organize(
                            run, candidate.project.bpm, candidate.project.beat_origin,
                            request.mode, beat_times=bt_map, min_beat=min_b, max_beat=float(note.start),
                        ))
                        run = []
                    notes.append(note)
                    min_b = float(note.start + note.duration)
                else:
                    run.append(note)
            if run:
                notes.extend(organize(
                    run, candidate.project.bpm, candidate.project.beat_origin,
                    request.mode, beat_times=bt_map, min_beat=min_b, max_beat=None,
                ))
            candidate.project.notes = notes
            candidate.project.validate()
            max_match_sec = max(0.25, 60.0 / float(candidate.project.bpm))
            preserved: list[str] = []
            preserved_refs: list[int] = []
            used_raw: set[int] = set()
            for idx, n in enumerate(candidate.project.notes):
                if id(n) in prior_by_id:
                    lyr, r_idx = prior_by_id[id(n)]
                    preserved.append(lyr)
                    preserved_refs.append(r_idx)
                else:
                    best_j, best_dist = None, max_match_sec
                    for j, (r_start, _lyr, _r_idx) in enumerate(prior_by_raw):
                        if j not in used_raw and abs(float(n.raw_start) - r_start) <= best_dist:
                            best_dist = abs(float(n.raw_start) - r_start)
                            best_j = j
                    if best_j is not None:
                        used_raw.add(best_j)
                        preserved.append(prior_by_raw[best_j][1])
                        preserved_refs.append(prior_by_raw[best_j][2])
                    else:
                        preserved.append('')
                        preserved_refs.append(idx)
            candidate.note_ref_indices = preserved_refs
            if not ((candidate.raw_lyrics or candidate.word_timestamps) and _implemented(align_lyrics_to_notes)):
                candidate.note_lyrics = preserved
            _align(candidate)
        return mutate(operation)

    @application.post('/api/undo')
    def undo():
        with application.state.session_lock:
            current = application.state.session
            if not current.undo_stack:
                return get_state_payload(_snapshot(current))
            candidate = _snapshot(current.undo_stack[-1])
            payload = get_state_payload(candidate)
            application.state.project_generation += 1
            candidate.undo_stack = current.undo_stack[:-1]
            application.state.session = candidate
            return payload

    @application.post('/api/restore_reference')
    def restore():
        def operation(candidate):
            if len(candidate.note_ref_indices) == len(candidate.project.notes):
                if len(candidate.reference_note_lyrics) != len(candidate.project.reference_notes):
                    candidate.reference_note_lyrics = [''] * len(candidate.project.reference_notes)
                for i, ref_idx in enumerate(candidate.note_ref_indices):
                    if 0 <= ref_idx < len(candidate.reference_note_lyrics) and i < len(candidate.note_lyrics):
                        candidate.reference_note_lyrics[ref_idx] = candidate.note_lyrics[i]
            candidate.project.notes = deepcopy(candidate.project.reference_notes)
            candidate.note_ref_indices = list(range(len(candidate.project.notes)))
            if not ((candidate.raw_lyrics or candidate.word_timestamps) and _implemented(align_lyrics_to_notes)):
                if len(candidate.reference_note_lyrics) == len(candidate.project.notes):
                    candidate.note_lyrics = list(candidate.reference_note_lyrics)
            _align(candidate)
        return mutate(operation, bump_generation=True)

    @application.post('/api/load_demo')
    def load_demo(request: DemoRequest = Body(default=DemoRequest())):
        def operation(candidate):
            fresh = _build_session(request.demo)
            candidate.__dict__.update(vars(fresh))
        return mutate(operation, bump_generation=True, bump_upload_generation=True)

    @application.get('/api/audio/guitar-only.wav')
    def guitar_audio():
        return Response(content=_wav_bytes(_guitar_samples(audio_snapshot())), media_type='audio/wav')

    @application.get('/api/audio/vocal-only.wav')
    def vocal_only_audio():
        return Response(content=_wav_bytes(_vocal_only_samples(audio_snapshot())), media_type='audio/wav')

    @application.get('/api/audio/vocal-guitar-mix.wav')
    def vocal_audio():
        return Response(content=_wav_bytes(_vocal_mix_samples(audio_snapshot())), media_type='audio/wav')

    @application.get('/api/simulation_demos')
    def simulations():
        return _simulation_demos()

    @application.get('/api/simulation_audio/{filename}')
    def simulation_audio(filename: str):
        registry = {demo['filename'] for demo in _simulation_demos()['demos']}
        if filename not in registry or '/' in filename or '\\' in filename:
            raise HTTPException(404, 'Unknown simulation recording')
        path = (SIMULATION_ROOT/filename).resolve()
        if path.parent != SIMULATION_ROOT.resolve():
            raise HTTPException(404, 'Unknown simulation recording')
        return FileResponse(path, media_type='audio/wav')

    application.state.jobs = {}

    @application.post('/api/analyze_audio')
    async def analyze_audio(file: UploadFile = File(...), lyrics: str = Form('')):
        import shutil
        import subprocess
        import time
        import uuid
        job_id = uuid.uuid4().hex
        upload_dir = get_writable_work_root() / 'uploads' / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        raw_filename = Path(file.filename or 'song.wav').name
        audio_path = upload_dir / raw_filename
        audio_path.write_bytes(await file.read())
        song_title = Path(raw_filename).stem or 'Uploaded Song'
        t_job_start = time.monotonic()

        def _update_job(progress_val: float, msg: str, status: str = 'running', **extra) -> None:
            payload = {
                'status': status,
                'progress': round(float(progress_val), 3),
                'message': msg,
                'elapsed_sec': round(time.monotonic() - t_job_start, 1),
            }
            payload.update(extra)
            application.state.jobs[job_id] = payload

        class _UploadSuperseded(Exception):
            pass

        with application.state.session_lock:
            application.state.project_generation += 1
            application.state.upload_generation += 1
            my_upload_gen = application.state.upload_generation

        def _raise_if_superseded() -> None:
            if my_upload_gen != application.state.upload_generation:
                raise _UploadSuperseded()

        _update_job(0.08, '[1/6] 正在分离人声与伴奏音轨…')

        def _worker():
            try:
                from subpkgs.stem_mir import (
                    separate_stems,
                    track_beats_and_downbeats,
                    recognize_accompaniment_chords,
                    synthesize_melody_conditioned_harmony,
                )
                from subpkgs.singing_lyrics import transcribe_vocal_notes, regulate_singing_notes_for_score
                import soundfile as sf

                work_audio_path = audio_path
                try:
                    sf.info(str(work_audio_path))
                except Exception:
                    ffmpeg_bin = shutil.which('ffmpeg')
                    if ffmpeg_bin:
                        decoded_wav = upload_dir / '_decoded_source.wav'
                        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                        subprocess.run(
                            [ffmpeg_bin, '-nostdin', '-y', '-i', str(audio_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', str(decoded_wav)],
                            check=True,
                            capture_output=True,
                            timeout=120,
                            creationflags=creationflags,
                        )
                        work_audio_path = decoded_wav

                # Step 1 — Stem Separation
                _raise_if_superseded()
                stems = separate_stems(work_audio_path, upload_dir / 'stems', prefer_cuda=True)
                _raise_if_superseded()
                _update_job(0.28, '[2/6] 正在识别歌词与词级时间戳…')

                # Step 2 — Early Vocal Word/Syllable Timestamp Extraction
                stage_warnings: list[str] = []
                lyric_extraction_failed = False
                harmony_synthesis_failed = False
                word_ts: list[dict] = []
                vocal_onsets: list[float] = []
                word_boundaries: list[float] = []
                inferred_lyrics = lyrics or ''
                try:
                    from subpkgs.singing_lyrics import (
                        transcribe_word_timestamps,
                        detect_pre_lyric_vocalise_onsets,
                    )
                    aligned = transcribe_word_timestamps(stems['vocals'], prefer_cuda=True, ground_truth_text=inferred_lyrics)
                    word_ts = aligned.get('word_timestamps') or []
                    if not inferred_lyrics.strip() and aligned.get('raw_lyrics'):
                        inferred_lyrics = str(aligned['raw_lyrics']).strip()
                    measured_ts = [w for w in word_ts if not w.get('interpolated') and not w.get('unsung') and not w.get('subdivided')]
                    subdiv_ts = [w for w in word_ts if not w.get('interpolated') and not w.get('unsung') and w.get('subdivided')]
                    vocal_onsets = sorted(
                        {float(w['start']) for w in measured_ts if 'start' in w and float(w['start']) >= 0}
                        | {float(w['group_start']) for w in subdiv_ts if 'group_start' in w and float(w['group_start']) >= 0}
                    )
                    word_boundaries = list(vocal_onsets)
                    if vocal_onsets:
                        pre_voc = detect_pre_lyric_vocalise_onsets(stems['vocals'], vocal_onsets[0])
                        if pre_voc:
                            vocal_onsets = sorted(set(pre_voc) | set(vocal_onsets))
                except Exception as exc:
                    lyric_extraction_failed = True
                    warn_msg = f"lyric_extraction_failed: {exc}"
                    stage_warnings.append(warn_msg)
                    logger.warning("Word timestamp extraction failed: %s", exc)

                _raise_if_superseded()
                _update_job(0.46, '[3/6] 正在追踪全曲节拍与小节强拍…')

                # Step 3 — Hybrid Beat & Downbeat Tracking:
                # Lock onto full mix when real accompaniment exists; fall back to vocals on pure a-cappella uploads.
                acc_wav = stems.get('accompaniment')
                has_accompaniment = _has_audible_accompaniment(acc_wav)
                beat_source = work_audio_path if has_accompaniment else stems['vocals']
                beat_info = track_beats_and_downbeats(
                    beat_source,
                    prefer_cuda=True,
                    vocal_onsets_sec=vocal_onsets or None,
                    pre_vocal_bars=4 if has_accompaniment else 1,
                )
                bpm = float(beat_info['bpm'])
                beat_origin = float(beat_info['beat_origin'])
                meter = str(beat_info['meter'])
                beat_times_secs: list[float] = [float(t) for t in (beat_info.get('beat_times') or [])]
                bar_start_secs: list[float] = [float(t) for t in (beat_info.get('bar_start_seconds') or [])]

                info = sf.info(str(work_audio_path))
                clip_duration = float(info.duration)

                # Step 4 — Vocal Note Transcription & Regulation
                _raise_if_superseded()
                _update_job(0.64, '[4/6] 正在转写人声主旋律音符…')
                from subpkgs.singing_lyrics import filter_pre_lyric_bleed_notes
                first_lyric_sec = word_boundaries[0] if word_boundaries else None
                word_boundaries = [t for t in word_boundaries if t >= beat_origin]
                tx = transcribe_vocal_notes(stems['vocals'], bpm=bpm, beat_origin=beat_origin,
                                            word_boundaries_sec=word_boundaries or None, prefer_cuda=True,
                                            beat_times=beat_times_secs or None)
                raw_notes = sorted(tx['notes'], key=lambda n: n.raw_start)
                if first_lyric_sec is not None:
                    pickup_tol_s = min(0.5, 0.5 * (60.0 / max(20.0, bpm)))
                    raw_notes = filter_pre_lyric_bleed_notes(
                        raw_notes, stems['vocals'], first_lyric_sec, beat_origin, pickup_tol_s
                    )
                word_boundaries = sorted(set(word_boundaries) | set(tx.get('word_boundaries_sec') or []))
                try:
                    notes = regulate_singing_notes_for_score(
                        raw_notes, bpm=bpm, beat_origin=beat_origin,
                        clip_duration=clip_duration, word_boundaries_sec=word_boundaries or None,
                        beat_times=beat_times_secs or None)
                    if not notes and raw_notes:
                        raise ValueError('regulate returned empty')
                except Exception as exc:
                    warn_msg = f"note_regulation_degraded: {exc}"
                    stage_warnings.append(warn_msg)
                    logger.warning("regulate_singing_notes_for_score degraded to fallback: %s", exc)
                    try:
                        notes = organize(
                            raw_notes, bpm, beat_origin, 'auto',
                            beat_times=beat_times_secs or None,
                        ) if raw_notes else []
                    except Exception as org_exc:
                        logger.warning("organize fallback to raw_notes: %s", org_exc)
                        notes = raw_notes

                # Step 5 — Hybrid Key & Base Harmony Extraction:
                _raise_if_superseded()
                _update_job(0.82, '[5/6] 正在识别伴奏和弦与调性…')
                # Extract true accompaniment chords & concert key when accompaniment is audible;
                # fall back automatically to vocal key + melody-conditioned harmony on pure a-cappella.
                tonic_pc, is_minor = 0, False
                if notes and _implemented(estimate_key_albrecht_shanahan):
                    key_info = estimate_key_albrecht_shanahan(notes)
                    tonic_pc = key_info['tonic_pc']
                    is_minor = key_info['is_minor']
                elif notes:
                    kc = guitar.key_candidates(Project(notes=notes, bpm=bpm, beat_origin=beat_origin))
                    if kc:
                        _, tonic_pc, is_minor = kc[0]
                    else:
                        tonic_pc = int(notes[-1].pitch) % 12
                        warn_msg = f"key_estimation_unresolved: no key candidates found; defaulted tonic to {NAMES[tonic_pc]} major from final note"
                        stage_warnings.append(warn_msg)
                        logger.warning("%s", warn_msg)

                base_chords: list[str] = []
                bar_dyn: list[float] = []
                det_splits: dict = {}
                used_acc_harmony = False
                if has_accompaniment and callable(recognize_accompaniment_chords):
                    try:
                        acc_res = recognize_accompaniment_chords(
                            acc_wav,
                            bpm=bpm,
                            beat_origin=beat_origin,
                            meter=meter,
                            tonic=None,
                            minor=None,
                            clip_duration=clip_duration,
                            bar_start_seconds=bar_start_secs or None,
                            beat_times=beat_times_secs or None,
                            notes=notes,
                            vocal_audio_path=stems['vocals'],
                        )
                        if isinstance(acc_res, dict) and acc_res.get('available') and acc_res.get('chords'):
                            used_acc_harmony = True
                            raw_tpc = acc_res.get('tonic_pc')
                            if isinstance(raw_tpc, int) and not isinstance(raw_tpc, bool) and 0 <= raw_tpc < 12:
                                tonic_pc = int(raw_tpc)
                                is_minor = bool(acc_res.get('is_minor', False))
                            elif acc_res.get('tonic') in NAMES:
                                tonic_pc = NAMES.index(acc_res['tonic'])
                                is_minor = bool(acc_res.get('is_minor', False))
                            base_chords = list(acc_res['chords'])
                            if isinstance(acc_res.get('split_chords'), dict):
                                det_splits = {str(k): list(v) for k, v in acc_res['split_chords'].items()}
                            if acc_res.get('bar_start_seconds'):
                                bar_start_secs = [float(t) for t in acc_res['bar_start_seconds']]
                            if acc_res.get('bar_dynamics'):
                                bar_dyn = [float(d) for d in acc_res['bar_dynamics']]
                    except Exception as exc:
                        stage_warnings.append(f"accompaniment_harmony_degraded: {exc}")
                        logger.warning("Accompaniment chord recognition degraded to vocal harmony fallback: %s", exc)

                rel_major_pc = (tonic_pc + 3) % 12 if is_minor else tonic_pc
                rel_tonic = NAMES[rel_major_pc]
                project = Project(
                    title=song_title, tonic=rel_tonic, meter=meter, bpm=bpm, beat_origin=beat_origin,
                    clip_duration=clip_duration, source=str(audio_path),
                    audio={'vocals': stems['vocals']},
                    notes=notes, reference_notes=deepcopy(notes),
                    bar_start_seconds=list(bar_start_secs),
                    beat_times=list(beat_times_secs),
                    warnings=list(stage_warnings),
                )

                if not used_acc_harmony:
                    vocal_chords = guitar.suggest(project, tonic_pc, minor=is_minor) if notes else []
                    if vocal_chords:
                        base_chords = [ch if ch else 'N.C.' for ch in vocal_chords]
                    else:
                        bar_count = max(1, len(bar_start_secs) - 1) if len(bar_start_secs) >= 2 else 1
                        base_chords = ['N.C.'] * bar_count
                    try:
                        if callable(synthesize_melody_conditioned_harmony) and notes:
                            harmony = synthesize_melody_conditioned_harmony(
                                notes, bpm=bpm, beat_origin=beat_origin, meter=meter, tonic=NAMES[tonic_pc], minor=is_minor,
                                clip_duration=clip_duration, bar_start_seconds=bar_start_secs or None,
                                beat_times=beat_times_secs or None)
                            if isinstance(harmony, dict):
                                if harmony.get('chords'):
                                    base_chords = list(harmony['chords'])
                                if harmony.get('bar_start_seconds'):
                                    bar_start_secs = [float(t) for t in harmony['bar_start_seconds']]
                                if harmony.get('bar_dynamics'):
                                    bar_dyn = [float(d) for d in harmony['bar_dynamics']]
                    except Exception as exc:
                        harmony_synthesis_failed = True
                        warn_msg = f"harmony_synthesis_failed: {exc}"
                        stage_warnings.append(warn_msg)
                        logger.warning("Harmony synthesis failed: %s", exc)

                if not bar_start_secs and beat_info.get('downbeats'):
                    bar_start_secs = [float(t) for t in beat_info['downbeats']]
                if not bar_dyn and bar_start_secs:
                    bar_dyn = [1.0] * (max(1, len(bar_start_secs) - 1) if len(bar_start_secs) >= 2 else 1)
                project.bar_start_seconds = list(bar_start_secs)
                project.warnings = list(stage_warnings)

                # Step 6 — Apply session state and align lyrics when raw_lyrics or word_timestamps are available
                _raise_if_superseded()
                _update_job(0.94, '[6/6] 正在生成六线吉他编配与出版级总谱…')
                rec_init = (
                    recommend_artistry(project, base_chords, bar_dyn)
                    if callable(recommend_artistry)
                    else None
                )
                auto_capo = (
                    int(select_ergonomic_capo(base_chords))
                    if callable(select_ergonomic_capo)
                    else 0
                )

                def _apply(candidate):
                    candidate.demo_mode = 'upload'
                    candidate.project = project
                    candidate.base_chords = base_chords
                    candidate.custom_chords = {}
                    candidate.custom_segment_voicings = {}
                    candidate.custom_voicings = [None] * len(base_chords)
                    preset_init = 'original'
                    style_init = STYLES[0] if STYLES else '分解和弦'
                    if rec_init is not None:
                        if rec_init.get('preset') in PRESETS:
                            preset_init = rec_init['preset']
                        if rec_init.get('style') in STYLES:
                            style_init = rec_init['style']
                    candidate.preset = preset_init
                    candidate.style = style_init
                    candidate.capo = auto_capo
                    candidate.split_bar = False
                    candidate.slap_x = False
                    candidate.palm_mute = False
                    candidate.passing_bass = bool(NATIVE and preset_init in ('emotional_pop', 'neosoul_jazz', 'walking_bass'))
                    candidate.embellish_hp = bool(NATIVE and (preset_init in ('emotional_pop', 'neosoul_jazz') or style_init == '抒情琶音'))
                    candidate.intro_mode = 'chord_melody' if NATIVE else 'none'
                    candidate.bar_start_seconds = bar_start_secs
                    candidate.beat_times = beat_times_secs
                    candidate.bar_dynamics = bar_dyn
                    candidate.detected_split_chords = det_splits
                    candidate.use_detected_splits = True
                    candidate.raw_lyrics = inferred_lyrics
                    candidate.note_lyrics = [''] * len(notes)
                    candidate.note_ref_indices = list(range(len(notes)))
                    candidate.custom_note_lyrics = {}
                    candidate.word_timestamps = word_ts
                    candidate.warnings = list(stage_warnings)
                    candidate.lyric_extraction_failed = lyric_extraction_failed
                    candidate.harmony_synthesis_failed = harmony_synthesis_failed
                    if inferred_lyrics or word_ts:
                        try:
                            _align(candidate, require_backend=True)
                        except Exception as align_exc:
                            candidate.note_lyrics = (candidate.note_lyrics + [''] * len(notes))[:len(notes)]
                            stage_warnings.append(f'Lyric alignment degraded: {align_exc}')
                            candidate.warnings = list(stage_warnings)
                            project.warnings = list(stage_warnings)
                    candidate.reference_note_lyrics = list(candidate.note_lyrics)
                with application.state.session_lock:
                    if my_upload_gen != application.state.upload_generation:
                        raise _UploadSuperseded()
                    mutate(_apply)
                    _update_job(
                        1.0,
                        '已完成分析' if not stage_warnings else f'已完成分析（部分环节降级：{"; ".join(stage_warnings)}）',
                        status='done',
                        degraded=bool(stage_warnings),
                        warnings=list(stage_warnings),
                        lyric_extraction_failed=lyric_extraction_failed,
                        harmony_synthesis_failed=harmony_synthesis_failed,
                    )
                if work_audio_path != audio_path:
                    Path(work_audio_path).unlink(missing_ok=True)
                if acc_wav and Path(acc_wav) != Path(stems['vocals']) and Path(acc_wav).parent == (upload_dir / 'stems'):
                    Path(acc_wav).unlink(missing_ok=True)
            except _UploadSuperseded:
                _update_job(1.0, '分析结果已被更新的曲目或上传请求取代', status='superseded')
                shutil.rmtree(upload_dir, ignore_errors=True)
            except Exception as exc:
                _update_job(1.0, str(exc), status='error')
                shutil.rmtree(upload_dir, ignore_errors=True)

        threading.Thread(target=_worker, daemon=True).start()
        return {'job_id': job_id}

    @application.get('/api/job/{job_id}')
    def job_status(job_id: str):
        if job_id not in application.state.jobs:
            raise HTTPException(404, 'Unknown job')
        return application.state.jobs[job_id]

    if STATIC_ROOT.is_dir():
        application.mount('/static', StaticFiles(directory=STATIC_ROOT), name='static')
    return application


def _fallback_workbench():
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MelodyDesk</title>
<style>body{font-family:system-ui;margin:24px}iframe{width:100%;height:650px;border:0}td input{width:80px}#error{color:#a00;white-space:pre-wrap}</style>
<h1>MelodyDesk 吉他弹唱工作台</h1><pre id="error" role="alert"></pre>
<form id="options"><select name="preset"></select><select name="style"></select>
<label>Capo <input name="capo" type="number" min="0" max="12" value="0"></label>
<label><input type="checkbox" name="passing_bass">Passing bass</label>
<label><input type="checkbox" name="split_bar">Split bars</label><label><input type="checkbox" name="slap_x">X slap</label>
<label><input type="checkbox" name="palm_mute">Palm mute</label><label><input type="checkbox" name="embellish_hp">H/P</label>
<select name="intro_mode"><option value="none">No intro melody</option><option value="nc">Intro N.C.</option><option value="groove">Intro groove</option><option value="chord_melody">Chord melody</option></select><button>编配</button></form>
<textarea id="lyrics" placeholder="歌词"></textarea><button id="align">对齐歌词</button>
<button data-route="cleanup">整理节奏</button><button data-route="undo">撤销</button>
<button data-route="restore_reference">恢复参考</button><button data-route="load_demo">手动示例</button>
<iframe title="Publication score" id="score" sandbox></iframe><table id="notes"></table>
<p>吉他 <audio controls src="/api/audio/guitar-only.wav"></audio></p>
<p><span id="mix-label">合成旋律与吉他</span> <audio controls src="/api/audio/vocal-guitar-mix.wav"></audio></p><ul id="simulations"></ul>
<script>
const form=document.querySelector('#options');
async function call(route,body){try{document.querySelector('#error').textContent='';const r=await fetch('/api/'+route,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const s=await r.json();if(!r.ok)throw Error(JSON.stringify(s));render(s);}catch(e){document.querySelector('#error').textContent=e.message}}
function render(s){document.querySelector('#score').srcdoc=s.publication_html;document.querySelector('#lyrics').value=s.raw_lyrics;document.querySelector('#mix-label').textContent=s.has_real_vocal?'真实人声与吉他':'合成旋律与吉他';
for(const key of ['preset','style']){const el=form.elements[key];el.replaceChildren();for(const v of key==='preset'?Object.keys(s.presets):s.styles){const o=new Option(v,v);el.add(o)}el.value=s[key]}
for(const el of form.elements){if(el.name in s){if(el.type==='checkbox')el.checked=s[el.name];else el.value=s[el.name]}}
const table=document.querySelector('#notes');table.replaceChildren();for(const n of s.notes_table){const row=table.insertRow();row.insertCell().textContent=n.index;for(const k of ['pitch','start','duration']){const input=document.createElement('input');input.type='number';input.step='any';input.value=n[k];input.onchange=()=>call('edit_note',{index:n.index,[k]:Number(input.value)});row.insertCell().append(input)}row.insertCell().textContent=n.lyric}
const list=document.querySelector('#simulations');list.replaceChildren();for(const d of s.simulation_demos){const li=document.createElement('li'),a=document.createElement('audio');li.textContent=d.label;a.controls=true;a.src=d.url;li.append(a);list.append(li)}
for(const audio of document.querySelectorAll('body>p audio')){audio.src=audio.src.split('?')[0]+'?v='+Date.now()}}
form.onsubmit=e=>{e.preventDefault();const data={};for(const el of form.elements){if(el.name)data[el.name]=el.type==='checkbox'?el.checked:el.name==='capo'?Number(el.value):el.value}call('reharmonize',data)};
document.querySelector('#align').onclick=()=>call('align_lyrics',{raw_lyrics:document.querySelector('#lyrics').value});
for(const b of document.querySelectorAll('[data-route]'))b.onclick=()=>call(b.dataset.route,{});call('state');
</script></html>'''


app = create_app()
