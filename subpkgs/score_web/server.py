"""FastAPI Backend for MelodyDesk 2.0 Workbench (with Re-harmonization Options A/B/C & Real-Vocal + Simulated Guitar Unplugged Audio Mix Endpoints)."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
import inspect
import io
import math
import threading
import wave
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from music import Note, Project, NAMES, parse_jianpu, pitch_name, numbered
import json
import guitar
from score_cleanup import organize
from .alphatex_builder import to_alphatex
from .yoopu_svg_renderer import render_publication_score_html
from ._score_timing import build_score_timeline

APP_ROOT = Path(__file__).resolve().parents[2]
INSTALL_ROOT = APP_ROOT.parents[1]
STATIC_ROOT = APP_ROOT / 'static'
EXAMPLE_PROJECT = INSTALL_ROOT / 'examples' / 'demo-project.json'
SIMULATION_ROOT = INSTALL_ROOT / 'work' / 'simulation_demo'
REPERTOIRE_AUDIO_ROOT = INSTALL_ROOT / 'work' / 'repertoire_audio'
REPERTOIRE_MANIFEST = REPERTOIRE_AUDIO_ROOT / 'repertoire_manifest.json'
_STEM_AUDIO_CACHE: dict[tuple[str, int, int], np.ndarray] = {}

try:
    from subpkgs.guitar_reharm import (reharmonize_progression, arrange_advanced,
        roman_numeral, check_melody_clash, generate_voicings, suggest_bar_substitutions,
        REHARMONIZE_PRESETS, ADVANCED_STYLES, synthesize_realistic_guitar)
except ModuleNotFoundError as exc:
    if exc.name not in {'subpkgs.guitar_reharm', 'subpkgs.guitar_reharm.reharmonizer',
                         'subpkgs.guitar_reharm.arrangement_grooves'}:
        raise
    reharmonize_progression = arrange_advanced = synthesize_realistic_guitar = suggest_bar_substitutions = None
    REHARMONIZE_PRESETS, ADVANCED_STYLES = {}, ()
try:
    from subpkgs.singing_lyrics import align_lyrics_to_notes, estimate_key_albrecht_shanahan
except ModuleNotFoundError as exc:
    if exc.name not in {'subpkgs.singing_lyrics', 'subpkgs.singing_lyrics.singing_transcriber'}:
        raise
    align_lyrics_to_notes = None
    estimate_key_albrecht_shanahan = None


def _implemented(function):
    return function is not None and 'NotImplementedError' not in function.__code__.co_names


def _refresh_backends():
    global reharmonize_progression, arrange_advanced, roman_numeral, check_melody_clash, generate_voicings, suggest_bar_substitutions
    global REHARMONIZE_PRESETS, ADVANCED_STYLES, synthesize_realistic_guitar, align_lyrics_to_notes, estimate_key_albrecht_shanahan
    global NATIVE, PRESETS, STYLES
    try:
        import importlib
        gr = importlib.import_module('subpkgs.guitar_reharm')
        reharmonize_progression = getattr(gr, 'reharmonize_progression', None)
        arrange_advanced = getattr(gr, 'arrange_advanced', None)
        roman_numeral = getattr(gr, 'roman_numeral', None)
        check_melody_clash = getattr(gr, 'check_melody_clash', None)
        generate_voicings = getattr(gr, 'generate_voicings', None)
        suggest_bar_substitutions = getattr(gr, 'suggest_bar_substitutions', None)
        REHARMONIZE_PRESETS = getattr(gr, 'REHARMONIZE_PRESETS', {})
        ADVANCED_STYLES = getattr(gr, 'ADVANCED_STYLES', ())
        synthesize_realistic_guitar = getattr(gr, 'synthesize_realistic_guitar', None)
    except Exception:
        pass
    try:
        import importlib
        sl = importlib.import_module('subpkgs.singing_lyrics')
        align_lyrics_to_notes = getattr(sl, 'align_lyrics_to_notes', None)
        estimate_key_albrecht_shanahan = getattr(sl, 'estimate_key_albrecht_shanahan', None)
    except Exception:
        pass
    NATIVE = _implemented(reharmonize_progression) and _implemented(arrange_advanced) and bool(REHARMONIZE_PRESETS)
    PRESETS = REHARMONIZE_PRESETS if NATIVE else {'original': {'name': '原版和弦', 'description': 'Basic triad accompaniment'}}
    STYLES = list(ADVANCED_STYLES) if NATIVE else ['分解和弦', '流行扫弦']


_refresh_backends()


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
    passing_bass: bool = True
    split_bar: bool = False
    slap_x: bool = False
    palm_mute: bool = False
    embellish_hp: bool = False
    intro_mode: str = 'chord_melody'
    raw_lyrics: str = ''
    note_lyrics: list[str] = field(default_factory=list)
    bar_analysis: list[dict] = field(default_factory=list)
    arrangement: dict = field(default_factory=dict)
    undo_stack: list = field(default_factory=list)
    bar_start_seconds: list[float] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)
    bar_dynamics: list[float] = field(default_factory=list)
    word_timestamps: list = field(default_factory=list)
    demo_mode: str = 'mandarin_pop'
    _cached_guitar_samples: tuple | None = field(default=None, repr=False)


def _snapshot(session):
    # The cached (sr, samples) pair is read-only, so snapshots share it instead of deep-copying audio.
    return Session(**{key: (value if key == '_cached_guitar_samples' else deepcopy(value))
                      for key, value in vars(session).items() if key != 'undo_stack'})


def _align(candidate, timestamps=None):
    eff_timestamps = timestamps if timestamps is not None else (candidate.word_timestamps or None)
    if not candidate.raw_lyrics and eff_timestamps is None:
        candidate.note_lyrics = [''] * len(candidate.project.notes)
        return
    if not _implemented(align_lyrics_to_notes):
        raise HTTPException(503, 'Lyric alignment dependency is unavailable')
    vocal_path = candidate.project.audio.get('vocals')
    available = vocal_path if (vocal_path and Path(vocal_path).is_file() and not candidate.raw_lyrics.strip() and eff_timestamps is None) else None
    alignment = align_lyrics_to_notes(candidate.project, raw_lyrics=candidate.raw_lyrics,
                                     word_timestamps=eff_timestamps, vocal_audio_path=available)
    candidate.note_lyrics = alignment['note_lyrics']
    candidate.raw_lyrics = alignment['raw_lyrics']


def _rebuild(candidate):
    _refresh_backends()
    candidate._cached_guitar_samples = None
    candidate.project.validate()
    measure = int(candidate.project.meter.split('/')[0])*4/int(candidate.project.meter.split('/')[1])
    count = math.ceil(max((n.start+n.duration for n in candidate.project.notes), default=0)/measure)
    if candidate.bar_start_seconds and len(candidate.bar_start_seconds) >= 2:
        count = max(count, len(candidate.bar_start_seconds) - 1)
    elif candidate.project.source:
        count = max(count, math.ceil((candidate.project.clip_duration-candidate.project.beat_origin)*candidate.project.bpm/60/measure))
    if len(candidate.base_chords) != count:
        candidate.base_chords = guitar.suggest(candidate.project, NAMES.index(candidate.project.tonic))
        candidate.custom_voicings = [None]*count
    if candidate.preset not in PRESETS or candidate.style not in STYLES:
        raise ValueError('Unsupported preset or style for the active backend')
    if NATIVE:
        analysis = reharmonize_progression(candidate.base_chords, candidate.project,
            preset=candidate.preset, tonic=candidate.project.tonic, split_bar=candidate.split_bar)
        for identity, chord in candidate.custom_chords.items():
            pieces = identity.split(':')
            bar = int(pieces[0])
            if not 0 <= bar < len(analysis):
                raise ValueError('Chord override bar is out of range')
            segments = analysis[bar]['split_segments']
            targets = range(len(segments)) if len(pieces) == 1 else [int(pieces[1])]
            for part in targets:
                if not 0 <= part < len(segments):
                    raise ValueError('Chord override segment is out of range')
                segment = segments[part]
                choices = generate_voicings(chord, candidate.capo)
                if not choices:
                    raise ValueError('Chord has no playable voicing')
                segment.update(chord=chord, roman=roman_numeral(chord, candidate.project.tonic),
                               frets=list(choices[0]['frets']), capo=candidate.capo)
        for index, row in enumerate(analysis):
            if len(candidate.custom_voicings) != count:
                raise ValueError('Custom voicings must match bar count')
            override = candidate.custom_voicings[index]
            if override is not None:
                row['split_segments'][0].update(frets=list(override), capo=candidate.capo)
            reports = []
            for part, segment in enumerate(row['split_segments']):
                identity = f'{index}:{part}'
                if identity in candidate.custom_segment_voicings:
                    segment.update(frets=list(candidate.custom_segment_voicings[identity]), capo=candidate.capo)
                reports.append(check_melody_clash(segment['chord'], candidate.project.notes,
                    index*measure+segment['beat'], index*measure+segment['beat']+segment['duration']))
            row['chord'] = row['split_segments'][0]['chord']
            row['roman'] = row['split_segments'][0]['roman']
            status = ('clash' if any(r['clashes'] for r in reports) else
                      'tension' if any(r['tensions'] for r in reports) else 'safe')
            row['clash'] = dict(status=status,
                message='⚠️ 小二度避用音冲突' if status == 'clash' else ('✨ 色彩张力音 (9/11/13)' if status == 'tension' else '✅ 与主旋律和谐'),
                clashes=[c for r in reports for c in r['clashes']],
                tensions=[t for r in reports for t in r['tensions']],
                chord_tones=sorted({pc for r in reports for pc in r['chord_tones']}),
                notes_checked=sum(r['notes_checked'] for r in reports))
            row['voicings'] = generate_voicings(row['chord'], candidate.capo)
            if _implemented(suggest_bar_substitutions) and row['chord'] != 'N.C.':
                prev_c = analysis[index-1]['chord'] if index > 0 else None
                next_c = analysis[index+1]['chord'] if index+1 < len(analysis) else None
                row['candidates'] = suggest_bar_substitutions(
                    row['chord'], prev_c, next_c, candidate.project.tonic,
                    candidate.project.notes, index*measure, (index+1)*measure
                )[:8]
        _arrange_kwargs = dict(
            chords=[row['chord'] for row in analysis],
            style=candidate.style, capo=candidate.capo, custom_voicings=candidate.custom_voicings,
            split_chords=[row['split_segments'] for row in analysis], passing_bass=candidate.passing_bass,
            slap_x=candidate.slap_x, palm_mute=candidate.palm_mute,
            embellish_hp=candidate.embellish_hp, intro_mode=candidate.intro_mode)
        if candidate.bar_dynamics and 'bar_dynamics' in inspect.signature(arrange_advanced).parameters:
            _arrange_kwargs['bar_dynamics'] = candidate.bar_dynamics
        arrangement = arrange_advanced(candidate.project, **_arrange_kwargs)
        for row, bar in zip(analysis, arrangement['bars']):
            row['split_segments'] = deepcopy(bar['split_segments'])
            row['frets'] = list(bar['frets'])
    else:
        if candidate.preset != 'original' or any((candidate.split_bar, candidate.slap_x, candidate.palm_mute,
                                                candidate.embellish_hp, candidate.intro_mode != 'none')):
            raise HTTPException(503, 'Advanced arrangement dependency is unavailable; basic triads only')
        chords = list(candidate.base_chords)
        for identity, chord in candidate.custom_chords.items():
            if ':' in identity:
                raise ValueError('Basic backend supports full-bar chords only')
            chords[int(identity)] = chord
        if any(v is not None for v in candidate.custom_voicings):
            raise HTTPException(503, 'Custom voicings require the advanced arranger')
        arrangement = guitar.arrange(candidate.project, chords, candidate.style, candidate.capo)
        arrangement.update(measure_beats=measure, bar_count=len(arrangement['bars']), chords=chords)
        for event in arrangement['events']:
            event['string'] = 6-event['string']
            event['technique'] = ''
        analysis = []
        for index, bar in enumerate(arrangement['bars']):
            segment = dict(beat=0.0, duration=measure, chord=bar['chord'], roman='', frets=list(bar['frets']))
            bar.update(bar=index, roman='', split_segments=[segment])
            analysis.append(dict(bar=index, chord=bar['chord'], roman='', split_segments=[deepcopy(segment)]))
    build_score_timeline(candidate.project, arrangement)
    if candidate.bar_start_seconds:
        offset0 = float(candidate.bar_start_seconds[0])
        arrangement['bar_start_seconds'] = [max(0.0, float(t) - offset0) for t in candidate.bar_start_seconds]
    if candidate.beat_times:
        bt0 = float(candidate.beat_times[0])
        arrangement['beat_times'] = [max(0.0, float(t) - bt0) for t in candidate.beat_times]
    if candidate.bar_dynamics:
        arrangement['bar_dynamics'] = list(candidate.bar_dynamics)
    if len(candidate.note_lyrics) != len(candidate.project.notes):
        _align(candidate)
    candidate.bar_analysis, candidate.arrangement = analysis, arrangement


REPERTOIRE_CATALOG = {
    'canon_in_d': dict(title='John Denver《Take Me Home, Country Roads》· 世界民谣吉他真人原唱全曲 (3:16)',
        subtitle='John Denver · 3:16 民谣木吉他传世名曲真人原唱全曲', category='世界名曲真人原唱全曲'),
    'romance_amor': dict(title='Eric Clapton《Tears in Heaven (Unplugged)》· 殿堂级木吉他真人原唱全曲 (4:36)',
        subtitle='Eric Clapton Unplugged · 4:36 艺术木吉他真人弹唱全曲', category='世界名曲真人原唱全曲'),
    'scarborough_fair': dict(title='Simon & Garfunkel《Scarborough Fair》· 英伦指弹民谣真人原唱全曲 (3:12)',
        subtitle='Simon & Garfunkel · 3:12 真人双声部原唱全曲', category='世界名曲真人原唱全曲'),
    'greensleeves': dict(title='The Brothers Four《Lady Greensleeves》· 文艺复兴古典吉他真人原唱全曲 (3:05)',
        subtitle='The Brothers Four · 3:05 真人合唱与古典吉他全曲', category='世界名曲真人原唱全曲'),
    'mandarin_pop': dict(title='赵雷《成都》· 华语民谣木吉他真人原唱完整全曲 (5:28)',
        subtitle='赵雷 · 5:28 华语诗意木吉他弹唱真人原唱全曲', category='华语名曲真人原唱全曲'),
    'real_vocal': dict(title="Karissa Hobbs《Let's Go Fishin'》· 乡村木吉他女声实录完整全曲 (2:13)",
        subtitle='Karissa Hobbs · 2:13 录音室乡村女声与木吉他全曲', category='真人原声实录全曲'),
}

_MANIFEST_CACHE: dict | None = None


def _get_repertoire_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    if REPERTOIRE_MANIFEST.is_file():
        try:
            _MANIFEST_CACHE = json.loads(REPERTOIRE_MANIFEST.read_text(encoding='utf-8'))
            return _MANIFEST_CACHE
        except Exception:
            pass
    return {}


# Fallback synthesized pieces used only when the full-length vocal repertoire manifest is absent.
_DEMO_PIECES = {
    'canon_in_d': dict(meter='4/4', capo=2, bpm=78.0, style=-1, clip_duration=50.0, passing_bass=True, embellish_hp=True,
        jianpu="3':2 2':2 | 1':2 7:2 | 6:2 5:2 | 6:2 7:2 | 1':1 3':1 2':1 7:1 | 1':1 5:1 6:1 4:1 | 3:1 6:1 5:1 4:1 | "
               "5:1 2':1 7:1 5:1 | 1':1 7:0.5 1':0.5 7:1 5:1 | 6:1 1':1 5:1 3:1 | 4:1 6:1 5:0.5 4:0.5 3:1 | "
               "2:1 4:1 3:1 2:1 | 3:1 1':1 7:1 6:1 | 5:1 3:1 4:1 5:1 | 6:1 4:1 5:1 7:1 | 1':4",
        chords=['C', 'G/B', 'Am7', 'Em7', 'Fmaj7', 'C/E', 'Fmaj7', 'G7sus4',
                'C', 'G/B', 'Am7', 'Em7', 'Fmaj7', 'C/E', 'Dm7', 'C'], lyrics=''),
    'romance_amor': dict(meter='3/4', capo=0, bpm=86.0, style=-1, clip_duration=36.0, passing_bass=False, embellish_hp=False,
        jianpu="3':1 3':1 3':1 | 3':1 2':1 1':1 | 1':1 7:1 6:1 | 6:1 1':1 3':1 | 7:1 7:1 7:1 | 7:1 1':1 7:1 | "
               "6:1 3:1 6:1 | 6:3 | 3':1 3':1 3':1 | 4':1 3':1 2':1 | 2':1 1':1 7:1 | 1':1 3':1 5':1 | "
               "3':1 2':1 1':1 | 2':1 1':1 7:1 | 6:1 7:1 5:1 | 6:3",
        chords=['Am', 'Am', 'Dm', 'Am', 'E7', 'E7', 'Am', 'Am',
                'Am', 'Dm', 'G7', 'C', 'Am', 'Dm', 'E7', 'Am'], lyrics=''),
    'scarborough_fair': dict(meter='3/4', capo=2, bpm=88.0, style=0, clip_duration=35.0, passing_bass=True, embellish_hp=True,
        jianpu="6,:2 6,:1 | 3:2 3:1 | 7,:1.5 1:0.5 7,:1 | 6,:3 | 3:1 3:1 5:1 | 6:2 5:1 | 3:1 4:1 2:1 | 3:3 | "
               "6:1 6:1 6:1 | 5:2 3:1 | 3:1 2:1 1:1 | 7,:2 5,:1 | 6,:2 3:1 | 2:2 1:1 | 7,:1 6,:1 5,:1 | 6,:3",
        chords=['Am', 'Am', 'G', 'Am', 'C', 'Am', 'C', 'G',
                'Am', 'Am', 'C', 'G', 'Am', 'G', 'Em7', 'Am'],
        lyrics='Are you going to Scar-bo-rough Fair Pars-ley sage rose-ma-ry and thyme '
               'Re-mem-ber me to one who lives there She once was a true love of mine'),
    'greensleeves': dict(meter='3/4', capo=0, bpm=90.0, style=-1, clip_duration=34.0, passing_bass=False, embellish_hp=True,
        jianpu="6,:1 1:1 2:1 | 3:1.5 4:0.5 3:1 | 2:2 7,:1 | 5,:1.5 6,:0.5 7,:1 | 1:2 6,:1 | 6,:1.5 5,:0.5 6,:1 | "
               "7,:2 5,:1 | 3,:3 | 5:3 | 5:1.5 4:0.5 3:1 | 2:2 7,:1 | 5,:1.5 6,:0.5 7,:1 | 1:2 6,:1 | "
               "6,:1.5 5,:0.5 6,:1 | 7,:1.5 6,:0.5 5,:1 | 6,:3",
        chords=['Am', 'G', 'Am', 'E7', 'Am', 'G', 'Em7', 'Am',
                'C', 'G/B', 'Am', 'E7', 'C', 'G/B', 'Em7', 'Am'],
        lyrics='A-las my love you do me wrong to cast me off dis-cour-teous-ly '
               'Green-sleeves was all my joy Green-sleeves was my de-light and my heart of gold'),
    'mandarin_pop': dict(meter='4/4', capo=0, bpm=92.0, style=0, clip_duration=32.0, passing_bass=True, embellish_hp=False,
        jianpu='3:1 5:1 5:1.5 6:0.5 | 5:1 3:1 2:1 1:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 2:2 | '
               '3:1 5:1 5:1.5 6:0.5 | 5:1 3:1 2:1 1:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 1:2 | '
               "1':1 7:1 6:1 5:1 | 6:1 5:1 3:1 2:1 | 6,:1 1:1 2:1 3:1 | 2:1.5 1:0.5 1:2",
        chords=['C', 'G/B', 'Am7', 'G', 'C', 'G/B', 'Am7', 'Fmaj7', 'Fmaj7', 'C/E', 'Dm7', 'C'],
        lyrics='故事的小黄花飘着童年歌谣守候着窗外的微风吹过仲夏的晴空直到永远在心间停留着不散去'),
}


def _initial_session(demo_mode: str = 'mandarin_pop'):
    _refresh_backends()
    if demo_mode == 'manual':
        demo_mode = 'mandarin_pop'
    manifest = _get_repertoire_manifest()
    if demo_mode in manifest:
        entry = manifest[demo_mode]
        vocals_wav = Path(entry['vocals_wav'])
        if vocals_wav.is_file():
            notes = [Note(**n) for n in entry['notes']]
            project = Project(
                title=entry['title'],
                tonic=entry['tonic'],
                meter=entry['meter'],
                bpm=float(entry['bpm']),
                beat_origin=float(entry['beat_origin']),
                clip_duration=float(entry['clip_duration']),
                source=entry['source'],
                audio={'vocals': str(vocals_wav)},
                notes=notes,
                reference_notes=deepcopy(notes),
            )
            chords = list(entry['base_chords'])
            style_name = entry.get('style') if entry.get('style') in STYLES else STYLES[0]
            preset_name = entry.get('preset') if entry.get('preset') in PRESETS else 'original'
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
                passing_bass=True,
                embellish_hp=(style_name == '抒情琶音'),
                intro_mode='none',
                raw_lyrics=entry['raw_lyrics'],
                note_lyrics=note_lyrics,
                bar_start_seconds=list(entry.get('bar_start_seconds', [])),
                beat_times=list(entry.get('beat_times', [])),
                bar_dynamics=list(entry.get('bar_dynamics', [])),
                word_timestamps=list(entry.get('word_timestamps', [])),
                demo_mode=demo_mode,
            )
            _rebuild(session)
            return session

    if demo_mode not in _DEMO_PIECES:
        demo_mode = 'mandarin_pop'
    piece = _DEMO_PIECES[demo_mode]
    notes, _ = parse_jianpu(piece['jianpu'], tonic='C', bpm=piece['bpm'])
    raw_lyrics = piece['lyrics']
    # Chinese lyrics map one character per note; hyphenated English maps one syllable per note.
    tokens = raw_lyrics.replace('-', ' ').split() if ' ' in raw_lyrics else list(raw_lyrics)
    note_lyrics = tokens[:len(notes)] + [''] * max(0, len(notes) - len(tokens))
    project = Project(
        title=REPERTOIRE_CATALOG[demo_mode]['title'],
        tonic='C', meter=piece['meter'], bpm=piece['bpm'], beat_origin=0.0, clip_duration=piece['clip_duration'],
        audio={},
        notes=notes, reference_notes=deepcopy(notes),
    )
    chords = list(piece['chords'])
    session = Session(project=project, base_chords=chords, custom_voicings=[None]*len(chords),
                      raw_lyrics=raw_lyrics, note_lyrics=note_lyrics, style=STYLES[piece['style']],
                      capo=piece['capo'], passing_bass=piece['passing_bass'], embellish_hp=piece['embellish_hp'],
                      demo_mode=demo_mode)
    _rebuild(session)
    return session


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
    payload = {key: getattr(session, key) for key in ('preset', 'style', 'capo', 'passing_bass', 'split_bar',
        'slap_x', 'palm_mute', 'embellish_hp', 'intro_mode', 'raw_lyrics', 'bar_analysis', 'arrangement')}
    payload.update(title=project.title, tonic=project.tonic, meter=project.meter, bpm=float(project.bpm),
        presets=PRESETS, styles=STYLES, chords=session.arrangement['chords'], bar_start_seconds=session.arrangement.get('bar_start_seconds', session.bar_start_seconds),
        alphatex=to_alphatex(project, session.arrangement, session.note_lyrics),
        publication_html=render_publication_score_html(project, session.arrangement, session.note_lyrics, session.bar_analysis),
        gpu_available=_gpu_available(), simulation_demos=_simulation_demos()['demos'],
        has_real_vocal=bool(project.audio.get('vocals') and Path(project.audio['vocals']).is_file()),
        repertoire=REPERTOIRE_CATALOG, active_demo=session.demo_mode,
        accompaniment_backend='advanced' if NATIVE else 'basic triads',
        synthesis_backend='native' if _implemented(synthesize_realistic_guitar) else 'fallback plucked-string synthesis')
    payload['notes_table'] = [dict(index=i, pitch=int(n.pitch), pitch_name=pitch_name(n.pitch),
        numbered=numbered(n.pitch, project.tonic), jianpu=numbered(n.pitch, project.tonic),
        start=float(n.start), duration=float(n.duration),
        raw_start=float(n.raw_start), raw_duration=float(n.raw_duration), confidence=float(n.confidence),
        edited=bool(n.edited), lyric=session.note_lyrics[i]) for i, n in enumerate(project.notes)]
    return payload


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
    meter: Literal['2/4', '3/4', '4/4', '6/8'] | None = None


class BarChordRequest(RequestModel):
    bar: int | None = Field(default=None, ge=0, strict=True)
    bar_index: int | None = Field(default=None, ge=0, strict=True)
    chord: str | None = None
    segment: int | None = Field(default=None, ge=0, strict=True)
    frets: list[Annotated[int, Field(strict=True)]] | None = None


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
    demo: Literal['manual', 'canon_in_d', 'romance_amor', 'scarborough_fair', 'greensleeves', 'mandarin_pop', 'real_vocal'] = 'manual'


def _validate_frets(frets):
    if len(frets) != 6 or any(type(f) is not int or not -1 <= f <= 24 for f in frets):
        raise ValueError('Frets must be six integers between -1 and 24')


def _guitar_samples(session, sr=44100):
    cached = session._cached_guitar_samples
    if cached is not None and cached[0] == sr:
        return cached[1]
    _refresh_backends()
    if _implemented(synthesize_realistic_guitar):
        samples = np.asarray(synthesize_realistic_guitar(session.project, session.arrangement, out_wav_path=None, sr=sr), dtype=np.float32)
    else:
        events = session.arrangement['events']
        spb = 60/session.project.bpm
        end = max((e['start']+e['duration'] for e in events), default=0)*spb
        samples = np.zeros((math.ceil((end+.25)*sr), 2), dtype=np.float32)
        rng = np.random.default_rng(2026)
        for event in events:
            duration = event['duration']*spb
            technique = event.get('technique', '')
            count = max(1, round((min(duration, .09) if technique == 'X' else duration+.08)*sr))
            t = np.arange(count, dtype=np.float64)/sr
            signal = np.zeros(count, dtype=np.float64)
            if technique == 'X':
                noise = rng.standard_normal(count)
                signal = np.convolve(noise, np.ones(5)/5, mode='same')*np.exp(-t*65)*.2
            else:
                frequency = 440*2**((event['pitch']-69)/12)
                for harmonic in range(1, 13):
                    if harmonic*frequency >= sr/2:
                        break
                    signal += np.sin(2*np.pi*frequency*harmonic*t)*np.exp(-t*harmonic*.8)/harmonic**1.5
                signal *= np.exp(-t*(16 if technique == 'PM' else 3.2))*.17
                if technique not in ('H', 'P'):
                    signal += rng.standard_normal(count)*np.exp(-t*220)*.035
            attack = min(count, max(1, round((.012 if technique in ('H', 'P') else .002)*sr)))
            signal[:attack] *= np.linspace(0, 1, attack)
            release = min(count, max(1, round(.04*sr)))
            signal[-release:] *= np.linspace(1, 0, release)
            signal *= event['velocity']/127
            offset = round(event['start']*spb*sr)
            pan = (event['string']-1)/5
            samples[offset:offset+count, 0] += (signal*math.sqrt(.2+.6*pan)).astype(np.float32)
            samples[offset:offset+count, 1] += (signal*math.sqrt(.8-.6*pan)).astype(np.float32)
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
    origin = max(0, round(origin_sec * sr))
    key = (str(path), sr, origin)
    cached = _STEM_AUDIO_CACHE.get(key)
    if cached is not None:
        return cached
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, source_sr = sf.read(str(path), always_2d=True, dtype='float32')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] != 2:
        audio = audio[:, :2]
    if source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        audio = resample_poly(audio, sr // divisor, source_sr // divisor).astype('float32')
    audio = np.ascontiguousarray(audio[origin:], dtype=np.float32)
    audio.flags.writeable = False
    if len(_STEM_AUDIO_CACHE) < 24:
        _STEM_AUDIO_CACHE[key] = audio
    return audio


def _vocal_only_samples(session, sr=44100):
    path = session.project.audio.get('vocals')
    guitar_len = len(_guitar_samples(session, sr))
    if not path or not Path(path).is_file():
        lead_events = []
        measure = session.arrangement.get('measure_beats', 4.0)
        for n in session.project.notes:
            # Lift the lead an octave above the C3–E4 chord voicings onto the bell-register strings 1/2,
            # then transpose by the capo so it sounds in the same key as the capoed accompaniment.
            pitch = (int(n.pitch) + 12 if int(n.pitch) < 67 else int(n.pitch)) + int(session.capo)
            string = 1 if pitch >= 64 else 2
            tuning = {1: 64, 2: 59}[string]
            fret = max(0, min(24, pitch - tuning))
            # Concert phrasing: accent downbeats, sustain long notes, and roll the downbeat slightly behind the chord.
            beat_in_bar = float(n.start % measure)
            vel = min(102, max(72, round(88 + (6 if abs(beat_in_bar) < 0.1 else 0) + (4 if n.duration >= 1.5 else -3))))
            start = float(n.start) + (0.015 if abs(beat_in_bar) < 0.05 and n.start > 0 else 0.0)
            lead_events.append(dict(
                bar=int(n.start // measure), beat=beat_in_bar, start=start,
                duration=float(n.duration), string=string, fret=fret, pitch=pitch,
                velocity=vel, technique='CM',
            ))
        if lead_events and _implemented(synthesize_realistic_guitar):
            lead_arr = dict(session.arrangement, events=lead_events)
            lead_samples = np.asarray(synthesize_realistic_guitar(session.project, lead_arr, out_wav_path=None, sr=sr), dtype=np.float32)
            length = max(len(lead_samples), guitar_len)
            return np.pad(lead_samples, ((0, length-len(lead_samples)), (0, 0))) * 0.58
        return np.zeros((max(1, guitar_len), 2), dtype=np.float32)
    vocal = _load_stem_audio(path, sr, session.project.beat_origin)
    length = max(len(vocal), guitar_len)
    return np.pad(vocal[:length], ((0, max(0, length-len(vocal))), (0, 0))) * 0.85


def _vocal_mix_samples(session, sr=44100):
    path = session.project.audio.get('vocals')
    guitar_samples = _guitar_samples(session, sr)
    if not path or not Path(path).is_file():
        # Synthesize the song's own lead melody on a studio-sampled lead acoustic guitar so key & BPM match 100%
        lead_events = []
        measure = session.arrangement.get('measure_beats', 4.0)
        for n in session.project.notes:
            pitch = (int(n.pitch) + 12 if int(n.pitch) < 67 else int(n.pitch)) + int(session.capo)
            string = 1 if pitch >= 64 else 2
            tuning = {1: 64, 2: 59}[string]
            fret = max(0, min(24, pitch - tuning))
            beat_in_bar = float(n.start % measure)
            vel = min(102, max(72, round(88 + (6 if abs(beat_in_bar) < 0.1 else 0) + (4 if n.duration >= 1.5 else -3))))
            start = float(n.start) + (0.015 if abs(beat_in_bar) < 0.05 and n.start > 0 else 0.0)
            lead_events.append(dict(
                bar=int(n.start // measure), beat=beat_in_bar, start=start,
                duration=float(n.duration), string=string, fret=fret, pitch=pitch,
                velocity=vel, technique='CM',
            ))
        if lead_events and _implemented(synthesize_realistic_guitar):
            lead_arr = dict(session.arrangement, events=lead_events)
            lead_samples = np.asarray(synthesize_realistic_guitar(session.project, lead_arr, out_wav_path=None, sr=sr), dtype=np.float32)
            length = max(len(lead_samples), len(guitar_samples))
            return (np.pad(lead_samples, ((0, length-len(lead_samples)), (0, 0))) * 0.58 +
                    np.pad(guitar_samples, ((0, length-len(guitar_samples)), (0, 0))) * 0.82)
        for fallback in [
            SIMULATION_ROOT / '02_extracted_lead_vocal.wav',
            SIMULATION_ROOT / 'real_singer_excerpt_20s.wav',
        ]:
            if fallback.is_file():
                path = str(fallback)
                break
    if not path or not Path(path).is_file():
        raise HTTPException(409, 'A real vocal recording is required for the vocal/guitar mix')
    import soundfile as sf
    from scipy.signal import resample_poly
    vocal, source_sr = sf.read(path, always_2d=True, dtype='float32')
    if vocal.shape[1] == 1:
        vocal = np.repeat(vocal, 2, axis=1)
    elif vocal.shape[1] != 2:
        raise ValueError('Vocal recording must be mono or stereo')
    if source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        vocal = resample_poly(vocal, sr//divisor, source_sr//divisor)
    origin = round(session.project.beat_origin*sr)
    if origin >= 0:
        guitar_samples = np.pad(guitar_samples, ((origin, 0), (0, 0)))
    else:
        guitar_samples = guitar_samples[min(-origin, len(guitar_samples)):]
    length = max(len(vocal), len(guitar_samples))
    return (np.pad(vocal, ((0, length-len(vocal)), (0, 0)))*.7 +
            np.pad(guitar_samples, ((0, length-len(guitar_samples)), (0, 0)))*.85)


def create_app() -> FastAPI:
    application = FastAPI(title='MelodyDesk 2.0 · AI 吉他弹唱与高阶改谱工作台')
    application.state.session = _initial_session()
    application.state.session_lock = threading.RLock()

    def snapshot():
        with application.state.session_lock:
            return _snapshot(application.state.session)

    def audio_snapshot():
        # Synthesize guitar once per arrangement on the live session; concurrent stem requests wait and reuse it.
        with application.state.session_lock:
            _guitar_samples(application.state.session)
            return _snapshot(application.state.session)

    def mutate(operation):
        with application.state.session_lock:
            previous = application.state.session
            candidate = _snapshot(previous)
            try:
                operation(candidate)
                _rebuild(candidate)
                payload = get_state_payload(candidate)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            candidate.undo_stack = (previous.undo_stack+[_snapshot(previous)])[-30:]
            application.state.session = candidate
            return payload

    @application.get('/')
    def index():
        if (STATIC_ROOT/'index.html').is_file():
            return FileResponse(STATIC_ROOT/'index.html')
        return HTMLResponse(_fallback_workbench())

    @application.get('/api/state')
    def state():
        return get_state_payload(snapshot())

    @application.post('/api/reharmonize')
    def reharmonize(request: ReharmonizeRequest):
        def operation(candidate):
            if any(k in request.model_fields_set for k in ('preset', 'capo', 'tonic', 'split_bar')) and 'custom_voicings' not in request.model_fields_set:
                candidate.custom_voicings = [None]*len(candidate.base_chords)
                candidate.custom_segment_voicings.clear()
            if 'preset' in request.model_fields_set and 'custom_chords' not in request.model_fields_set:
                candidate.custom_chords.clear()
            if any(k in request.model_fields_set for k in ('bpm', 'meter')):
                candidate.bar_start_seconds = []
                candidate.beat_times = []
            for key in request.model_fields_set:
                value = getattr(request, key)
                if value is None:
                    raise ValueError(f'{key} cannot be null')
                if key == 'custom_voicings':
                    if value == []:
                        value = [None]*len(candidate.base_chords)
                    for frets in value:
                        if frets is not None:
                            _validate_frets(frets)
                if key == 'custom_chords':
                    import re
                    if any(re.fullmatch(r'\d+(?::\d+)?', identity) is None for identity in value):
                        raise ValueError('Chord override keys must be bar or bar:segment indices')
                setattr(candidate.project if key in ('title', 'tonic', 'bpm', 'meter') else candidate, key, value)
        return mutate(operation)

    @application.post('/api/bar_chord')
    def bar_chord(request: BarChordRequest):
        def operation(candidate):
            bar_idx = request.bar if request.bar is not None else request.bar_index
            if bar_idx is None:
                raise ValueError('Bar index is required')
            if bar_idx >= len(candidate.arrangement['bars']):
                raise ValueError('Bar index is out of range')
            if request.segment is not None and request.segment >= len(candidate.arrangement['bars'][bar_idx]['split_segments']):
                raise ValueError('Segment index is out of range')
            identity = str(bar_idx)+(f':{request.segment}' if request.segment is not None else '')
            if request.chord is not None:
                candidate.custom_chords[identity] = request.chord
                if request.frets is None:
                    if request.segment is None and bar_idx < len(candidate.custom_voicings):
                        candidate.custom_voicings[bar_idx] = None
                    candidate.custom_segment_voicings.pop(identity, None)
            if request.frets is not None:
                _validate_frets(request.frets)
                if request.segment is None:
                    candidate.custom_voicings[bar_idx] = list(request.frets)
                else:
                    candidate.custom_segment_voicings[identity] = list(request.frets)
        return mutate(operation)

    @application.post('/api/align_lyrics')
    def lyrics(request: LyricRequest):
        def operation(candidate):
            candidate.raw_lyrics = request.raw_lyrics
            if not request.raw_lyrics:
                candidate.note_lyrics = ['']*len(candidate.project.notes)
            else:
                _align(candidate, request.word_timestamps)
        return mutate(operation)

    @application.post('/api/edit_note')
    def edit_note(request: NoteRequest):
        def operation(candidate):
            if request.index >= len(candidate.project.notes):
                raise ValueError('Note index is out of range')
            note = candidate.project.notes[request.index]
            note_changed = False
            for key in request.model_fields_set-{'index', 'lyric'}:
                value = getattr(request, key)
                if value is None:
                    if 'lyric' in request.model_fields_set and request.lyric is not None:
                        continue
                    raise ValueError(f'{key} cannot be null')
                setattr(note, key, value)
                note_changed = True
            if note_changed:
                note.edited = True
                candidate.project.validate()
                _align(candidate)
            if 'lyric' in request.model_fields_set and request.lyric is not None:
                if len(candidate.note_lyrics) != len(candidate.project.notes):
                    candidate.note_lyrics = ['']*len(candidate.project.notes)
                candidate.note_lyrics[request.index] = str(request.lyric)
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
            notes, run = [], []
            for note in candidate.project.notes:
                if note.edited:
                    notes.extend(organize(run, candidate.project.bpm, candidate.project.beat_origin, request.mode))
                    run = []
                    notes.append(note)
                else:
                    run.append(note)
            notes.extend(organize(run, candidate.project.bpm, candidate.project.beat_origin, request.mode))
            candidate.project.notes = notes
            candidate.project.validate()
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
            candidate.undo_stack = current.undo_stack[:-1]
            application.state.session = candidate
            return payload

    @application.post('/api/restore_reference')
    def restore():
        def operation(candidate):
            candidate.project.notes = deepcopy(candidate.project.reference_notes)
            _align(candidate)
        return mutate(operation)

    @application.post('/api/load_demo')
    def load_demo(request: DemoRequest = Body(default=DemoRequest())):
        def operation(candidate):
            fresh = _initial_session(request.demo)
            candidate.__dict__.update(vars(fresh))
        return mutate(operation)

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
        import uuid
        job_id = uuid.uuid4().hex[:10]
        upload_dir = INSTALL_ROOT / 'work' / 'uploads' / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / (Path(file.filename or 'song.wav').name)
        audio_path.write_bytes(await file.read())
        application.state.jobs[job_id] = {'status': 'running', 'progress': 0.1, 'message': '正在分离人声与伴奏...'}

        def _worker():
            try:
                from subpkgs.stem_mir import separate_stems, track_beats_and_downbeats, synthesize_melody_conditioned_harmony
                from subpkgs.singing_lyrics import transcribe_vocal_notes, regulate_singing_notes_for_score
                import soundfile as sf

                # Step 1 — Stem Separation
                stems = separate_stems(audio_path, upload_dir / 'stems', prefer_cuda=True)
                application.state.jobs[job_id] = {'status': 'running', 'progress': 0.3, 'message': '正在提取人声词语时间戳...'}

                # Step 2 — Early Vocal Word/Syllable Timestamp Extraction
                word_ts: list[dict] = []
                vocal_onsets: list[float] = []
                word_boundaries: list[float] = []
                try:
                    from subpkgs.singing_lyrics import transcribe_word_timestamps
                    aligned = transcribe_word_timestamps(stems['vocals'], prefer_cuda=True, ground_truth_text=lyrics or '')
                    word_ts = aligned.get('word_timestamps') or []
                    vocal_onsets = sorted({float(w['start']) for w in word_ts if 'start' in w and float(w['start']) >= 0})
                    word_boundaries = sorted(set(vocal_onsets) | {float(w['end']) for w in word_ts if 'end' in w and float(w['end']) >= 0})
                except Exception:
                    pass

                application.state.jobs[job_id] = {'status': 'running', 'progress': 0.4, 'message': '正在提取人声节奏、主旋律与生成吉他和弦...'}

                # Step 3 — Vocal Beat & Downbeat Tracking
                beat_info = track_beats_and_downbeats(stems['vocals'], prefer_cuda=True, vocal_onsets_sec=vocal_onsets or None)
                bpm = float(beat_info['bpm'])
                beat_origin = float(beat_info['beat_origin'])
                meter = str(beat_info['meter'])
                beat_times_secs: list[float] = [float(t) for t in (beat_info.get('beat_times') or [])]
                bar_start_secs: list[float] = [float(t) for t in (beat_info.get('bar_start_seconds') or [])]
                if not beat_times_secs:
                    while bpm > 132.0:
                        bpm *= 0.5
                    while bpm < 68.0:
                        bpm *= 2.0

                info = sf.info(str(audio_path))
                clip_duration = float(info.duration)

                # Step 4 — Vocal Note Transcription & Regulation
                tx = transcribe_vocal_notes(stems['vocals'], bpm=bpm, beat_origin=beat_origin,
                                            word_boundaries_sec=word_boundaries or None, prefer_cuda=True,
                                            beat_times=beat_times_secs or None)
                raw_notes = sorted(tx['notes'], key=lambda n: n.raw_start)
                word_boundaries = sorted(set(word_boundaries) | set(tx.get('word_boundaries_sec') or []))
                try:
                    notes = regulate_singing_notes_for_score(
                        raw_notes, bpm=bpm, beat_origin=beat_origin,
                        clip_duration=clip_duration, word_boundaries_sec=word_boundaries or None,
                        beat_times=beat_times_secs or None)
                    if not notes and raw_notes:
                        raise ValueError('regulate returned empty')
                except Exception:
                    try:
                        notes = organize(raw_notes, bpm, beat_origin, 'auto') if raw_notes else []
                    except Exception:
                        notes = raw_notes

                # Step 5 — Albrecht-Shanahan (2013) Empirical Key Estimation
                tonic_pc, is_minor = 0, False
                if notes and estimate_key_albrecht_shanahan is not None:
                    key_info = estimate_key_albrecht_shanahan(notes)
                    tonic_pc = key_info['tonic_pc']
                    is_minor = key_info['is_minor']
                elif notes:
                    tonic_pc = int(notes[-1].pitch) % 12
                rel_major_pc = (tonic_pc + 3) % 12 if is_minor else tonic_pc
                rel_tonic = NAMES[rel_major_pc]
                project = Project(
                    title=audio_path.stem, tonic=rel_tonic, meter=meter, bpm=bpm, beat_origin=beat_origin,
                    clip_duration=clip_duration, source=str(audio_path),
                    audio={'vocals': stems['vocals']},
                    notes=notes, reference_notes=deepcopy(notes),
                )
                turnaround = [
                    guitar.chord_name(tonic_pc, is_minor),
                    guitar.chord_name(rel_major_pc + 7, False),
                    guitar.chord_name(rel_major_pc + 9, True),
                    guitar.chord_name(rel_major_pc + 5, False),
                ]
                vocal_chords = guitar.suggest(project, tonic_pc, minor=is_minor) if notes else turnaround
                base_chords = [
                    (turnaround[i % 4] if (not ch or ch == 'N.C.') else ch)
                    for i, ch in enumerate(vocal_chords)
                ]
                bar_dyn: list[float] = []
                try:
                    if callable(synthesize_melody_conditioned_harmony) and notes:
                        harmony = synthesize_melody_conditioned_harmony(
                            notes, bpm=bpm, beat_origin=beat_origin, meter=meter, tonic=rel_tonic, minor=is_minor,
                            clip_duration=clip_duration, bar_start_seconds=bar_start_secs or None)
                        if isinstance(harmony, dict):
                            if harmony.get('chords'):
                                base_chords = list(harmony['chords'])
                            if harmony.get('bar_start_seconds'):
                                bar_start_secs = [float(t) for t in harmony['bar_start_seconds']]
                            if harmony.get('bar_dynamics'):
                                bar_dyn = [float(d) for d in harmony['bar_dynamics']]
                except (ImportError, AttributeError):
                    pass
                except Exception:
                    pass
                if not bar_start_secs and beat_info.get('downbeats'):
                    bar_start_secs = [float(t) for t in beat_info['downbeats']]
                if not bar_dyn and bar_start_secs:
                    bar_dyn = [1.0] * len(bar_start_secs)

                # Step 6 — Preserve word_timestamps when aligning lyrics
                def _apply(candidate):
                    candidate.demo_mode = 'upload'
                    candidate.project = project
                    candidate.base_chords = base_chords
                    candidate.custom_voicings = [None]*len(base_chords)
                    candidate.bar_start_seconds = bar_start_secs
                    candidate.beat_times = beat_times_secs
                    candidate.bar_dynamics = bar_dyn
                    candidate.raw_lyrics = lyrics
                    candidate.note_lyrics = ['']*len(notes)
                    candidate.word_timestamps = word_ts
                    if lyrics:
                        _align(candidate)
                mutate(_apply)
                application.state.jobs[job_id] = {'status': 'done', 'progress': 1.0, 'message': '已完成分析'}
            except Exception as exc:
                application.state.jobs[job_id] = {'status': 'error', 'progress': 1.0, 'message': str(exc)}

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
<select name="intro_mode"><option value="none">No intro melody</option><option value="chord_melody">Chord melody</option></select><button>编配</button></form>
<textarea id="lyrics" placeholder="歌词"></textarea><button id="align">对齐歌词</button>
<button data-route="cleanup">整理节奏</button><button data-route="undo">撤销</button>
<button data-route="restore_reference">恢复参考</button><button data-route="load_demo">手动示例</button>
<iframe title="Publication score" id="score" sandbox></iframe><table id="notes"></table>
<p>吉他 <audio controls src="/api/audio/guitar-only.wav"></audio></p>
<p>真实人声与吉他 <audio controls src="/api/audio/vocal-guitar-mix.wav"></audio></p><ul id="simulations"></ul>
<script>
const form=document.querySelector('#options');
async function call(route,body){try{document.querySelector('#error').textContent='';const r=await fetch('/api/'+route,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const s=await r.json();if(!r.ok)throw Error(JSON.stringify(s));render(s);}catch(e){document.querySelector('#error').textContent=e.message}}
function render(s){document.querySelector('#score').srcdoc=s.publication_html;document.querySelector('#lyrics').value=s.raw_lyrics;
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
