"""Local full-mix -> Demucs vocals -> pYIN monophonic transcription baseline."""
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import uuid
import wave
from music import Note, Project, performance_timing

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / 'work'
CACHE = WORK / 'model-cache'
JOBS = WORK / 'audio-jobs'
os.environ.setdefault('TORCH_HOME', str(CACHE))
os.environ.setdefault('NUMBA_CACHE_DIR', str(WORK / 'numba-cache'))
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled('已取消')


def dependencies():
    missing = [name for name in ('numpy', 'scipy', 'librosa', 'soundfile', 'torch', 'torchaudio', 'demucs')
               if importlib.util.find_spec(name) is None]
    if not shutil.which('ffmpeg'):
        missing.append('ffmpeg')
    return missing


def run(command, cancel=None, log=None):
    """No shell; output drained to a file to avoid pipe deadlocks."""
    JOBS.mkdir(parents=True, exist_ok=True)
    log = Path(log or JOBS / ('command-' + uuid.uuid4().hex + '.log'))
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=HIDDEN)
        try:
            while process.poll() is None:
                check_cancel(cancel)
                time.sleep(0.15)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        tail = log.read_text(encoding='utf-8', errors='replace')[-1800:]
        raise RuntimeError('音频处理失败，详细日志：' + str(log) + '\n' + tail)


def probe(path):
    import json
    tool = shutil.which('ffprobe')
    if not tool:
        raise RuntimeError('找不到 ffprobe，请安装 FFmpeg 并加入 PATH')
    result = subprocess.run([tool, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                            capture_output=True, creationflags=HIDDEN, timeout=30, check=True)
    duration = float(json.loads(result.stdout)['format']['duration'])
    if not 0 < duration <= 3600:
        raise ValueError('首版支持最长 60 分钟的音频')
    return duration


def decode(source, destination, start, duration, cancel=None):
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start), '-i', str(source),
         '-t', str(duration), '-vn', '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(destination)], cancel)


def transcribe(vocal_path, mix_path, cancel=None, progress=lambda p, s: None, backend='pyin', vocal_only=True):
    import numpy as np
    import librosa
    progress(50, '分析节拍与速度…')
    y, sr = librosa.load(str(vocal_path), sr=22050, mono=True)
    mix, _ = librosa.load(str(mix_path), sr=sr, mono=True)
    if len(y) < sr * 0.25 or float(np.max(np.abs(y))) < 0.0001:
        raise ValueError('片段中没有足够的可识别人声，请更换片段')
    tempo, frames = librosa.beat.beat_track(y=mix, sr=sr, hop_length=256, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    warnings = ['拍号默认 4/4、简谱 1=C；均需人工确认。', '速度可能存在半速／倍速歧义；伴唱、滑音与同音重复需检查。']
    if not 30 <= bpm <= 300 or not len(frames):
        bpm = 120.0
        warnings.append('未找到稳定节拍，暂用 120 BPM，请手动设置。')
    times = librosa.frames_to_time(frames, sr=sr, hop_length=256)
    spb = 60 / bpm
    origin = float(times[0] - np.ceil(times[0] / spb) * spb) if len(times) else 0.0
    check_cancel(cancel)
    progress(58, '跟踪主唱音高（本地处理，可能需要数分钟）…')
    if backend == 'pyin':
        from pitch_notes import track_melody
    elif backend == 'rosvot':
        from singing_model import track_singing as track_melody
    else:
        raise ValueError('未知旋律模型')
    joined = track_melody(y, sr, lambda: check_cancel(cancel), progress)
    if vocal_only:
        from vocal_activity import singing_scores, filter_notes
        progress(92, '筛选人声活动，排除器乐残留…')
        centers, scores = singing_scores(y, sr, lambda: check_cancel(cancel))
        before = len(joined)
        joined = filter_notes(joined, centers, scores)
        warnings.append(f'人声活动筛选移除 {before-len(joined)} 个音符；轻声、伴唱和器乐残留仍需试听检查。')
    if not joined:
        raise ValueError('没有得到稳定旋律，请换一个主唱更清楚的片段')
    return performance_timing(joined, bpm, origin), bpm, origin, warnings


def analyze(source, start, duration, cancel=None, progress=lambda p, s: None, backend='pyin', vocal_only=True):
    if backend not in ('pyin', 'rosvot'):
        raise ValueError('未知旋律模型')
    if backend == 'rosvot':
        from singing_model import ready
        if not ready():
            raise RuntimeError('实验模型尚未安装，请选择默认方案，或运行 install_singing_model.py --all。')
    if vocal_only:
        from vocal_activity import ready
        if not ready():
            raise RuntimeError('人声筛选模型尚未安装，请运行 install_singing_model.py。')
    missing = dependencies()
    if missing:
        raise RuntimeError('缺少运行组件：' + ', '.join(missing) + '。请运行安装脚本。')
    length = probe(source)
    if start < 0 or duration < 2 or duration > 3600 or start + duration > length + 0.05:
        raise ValueError('请选择音频范围内的 2 秒至 60 分钟音频')
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    clip = job / 'clip.wav'
    progress(5, '准备歌曲片段…')
    decode(source, clip, start, duration, cancel)
    check_cancel(cancel)
    progress(12, '分离人声；首次使用可能需要下载模型…')
    python = Path(sys.executable).with_name('python.exe') if os.name == 'nt' else Path(sys.executable)
    run([str(python), '-m', 'demucs', '--two-stems=vocals', '-n', 'htdemucs', '-d', 'cpu',
         '--shifts', '0', '-j', '0', '-o', str(job / 'stems'), str(clip)], cancel, job / 'separation.log')
    vocals = job / 'stems' / 'htdemucs' / 'clip' / 'vocals.wav'
    if not vocals.exists():
        raise RuntimeError('人声分离未生成音轨，请查看处理日志')
    notes, bpm, origin, warnings = transcribe(vocals, clip, cancel, progress, backend, vocal_only)
    check_cancel(cancel)
    project = Project(title=Path(source).stem, source=str(Path(source).resolve()), clip_start=start,
                      clip_duration=duration, bpm=bpm, beat_origin=origin, notes=notes,
                      audio={'mix': str(clip), 'vocals': str(vocals)}, warnings=warnings,
                      engine=f'demucs-htdemucs + {"pyin-vibrato-v3" if backend == "pyin" else "rosvot-v2"} + vocal-gate={vocal_only} / experimental')
    project.validate()
    project.save(job / 'result.json')
    progress(100, f'提取完成：{len(notes)} 个音符，请试听校对')
    return project


def synthesize(project, path, raw=False, tail=.25):
    import numpy as np
    project.validate()
    sr = 22050
    events = [(n.pitch, n.raw_start if raw else n.start * 60 / project.bpm,
               n.raw_duration if raw else n.duration * 60 / project.bpm) for n in project.notes]
    end = max((start + duration for _, start, duration in events), default=0) + tail
    if end > 3600:
        raise ValueError('试听长度超过一小时')
    audio = np.zeros(max(1, int(end * sr)), dtype=np.float32)
    for pitch, start, duration in events:
        n = max(1, int(duration * sr))
        t = np.arange(n, dtype=np.float32) / sr
        f = 440 * 2 ** ((pitch - 69) / 12)
        signal = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * 0.28
        ramp = min(int(sr * 0.01), n // 2)
        if ramp:
            signal[:ramp] *= np.linspace(0, 1, ramp)
            signal[-ramp:] *= np.linspace(1, 0, ramp)
        offset = int(start * sr)
        audio[offset:offset + n] += signal[:len(audio[offset:offset + n])]
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())


def compare_audio(project, destination):
    reference = Path(project.audio.get('mix', ''))
    if not reference.is_file():
        raise ValueError('找不到原曲片段，请重新识别原音频')
    synth = Path(destination).with_name('aligned-melody.wav')
    synthesize(project, synth, raw=True)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(reference), '-i', str(synth),
         '-filter_complex', '[0:a]volume=0.65[a];[1:a]volume=0.8[b];[a][b]amix=inputs=2:duration=longest:normalize=0',
         '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(destination)])
