"""Reproducible smoke test. Synthetic tones are NOT a singing accuracy benchmark."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import soundfile as sf
import engine
from music import Project, parse_jianpu, midi_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--separation', action='store_true')
    args = parser.parse_args()
    folder = engine.WORK / 'verification'
    folder.mkdir(parents=True, exist_ok=True)
    notes, _ = parse_jianpu('1:1 2:1 3:1 5:1 6:1 5:1 3:1 1:1', bpm=100)
    expected = Project(title='合成音阶测试（非真实歌曲）', bpm=100, notes=notes, clip_duration=5)
    voice = folder / 'synthetic-melody.wav'
    engine.synthesize(expected, voice)
    y, sr = sf.read(voice)
    # Weak low bass and percussion produce a reproducible mixed fixture.
    t = np.arange(len(y)) / sr
    mix = y + .04 * np.sin(2 * np.pi * 65.406 * t)
    for onset in np.arange(0, len(y) / sr, .6):
        a = int(onset * sr)
        n = min(int(.05 * sr), len(mix) - a)
        mix[a:a+n] += .12 * np.sin(2 * np.pi * 100 * np.arange(n) / sr) * np.exp(-np.arange(n) / (sr * .01))
    mixture = folder / 'synthetic-mix.wav'
    sf.write(mixture, mix, sr, subtype='PCM_16')
    began = time.monotonic()
    detected, bpm, origin, warnings = engine.transcribe(voice, mixture, progress=lambda p, s: print(round(p), s, flush=True))
    pitches = []
    for n in detected:
        if not pitches or pitches[-1] != n.pitch:
            pitches.append(n.pitch)
    target = [n.pitch for n in notes]
    assert pitches == target, (pitches, target)
    project = Project(title=expected.title, bpm=bpm, notes=detected, beat_origin=origin,
                      audio={'mix': str(mixture), 'vocals': str(voice)}, warnings=warnings)
    project.save(folder / 'synthetic-result.json')
    (folder / 'synthetic-result.mid').write_bytes(midi_bytes(project))
    engine.compare_audio(project, folder / 'compare.wav')
    report = {'synthetic_pitch_sequence': pitches, 'estimated_bpm': bpm, 'elapsed_seconds': time.monotonic() - began,
              'real_song_accuracy': 'not measured; no user song supplied'}
    if args.separation:
        began = time.monotonic()
        separated = engine.analyze(mixture, 0, 4.8, progress=lambda p, s: print(round(p), s, flush=True))
        separated.save(folder / 'full-pipeline-result.json')
        report['full_pipeline_note_count'] = len(separated.notes)
        report['full_pipeline_elapsed_seconds'] = time.monotonic() - began
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
