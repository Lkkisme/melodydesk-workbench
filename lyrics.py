"""Local, provisional singing transcription. Segment timing is not note alignment."""
from pathlib import Path
import threading
import os
from runtime_paths import resolve_model_path
CACHE=resolve_model_path('whisper', require_writable_parent=True)
os.environ.setdefault('HF_HOME',str(CACHE))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
_MODEL=None
_LOCK=threading.Lock()

def filter_vocal_lines(lines,path,cancel_check=lambda:None):
    import librosa
    import numpy as np
    from vocal_activity import singing_scores
    y,sr=librosa.load(str(path),sr=32000,mono=True)
    centers,scores=singing_scores(y,sr,cancel_check)
    return [line for line in lines if np.median(np.interp(np.linspace(line['start'],line['end'],5),centers,scores))>=.08]

def recognize(path,cancel_check=lambda:None,progress=lambda p,s:None):
    global _MODEL
    from faster_whisper import WhisperModel
    cancel_check()
    progress(3,'加载歌词模型；首次使用需下载 small 多语言模型…')
    with _LOCK:
        if _MODEL is None:
            _MODEL=WhisperModel('small',device='cpu',compute_type='int8',cpu_threads=4,download_root=str(CACHE))
    segments,info=_MODEL.transcribe(str(path),language='zh',beam_size=3,word_timestamps=True,
        condition_on_previous_text=False,vad_filter=False,temperature=0,
        no_speech_threshold=.6,compression_ratio_threshold=2.4)
    result=[]
    for segment in segments:
        cancel_check()
        progress(min(99,100*segment.end/max(info.duration,1)),f'识别歌词 {segment.end:.0f} / {info.duration:.0f} 秒…')
        if segment.no_speech_prob>.9 and segment.avg_logprob < -1.5:
            continue
        text=segment.text.strip()
        if text:
            result.append({'start':float(segment.start),'end':min(float(segment.end),info.duration),'text':text,
                           'reviewed':False,'avg_logprob':float(segment.avg_logprob),'no_speech_prob':float(segment.no_speech_prob),'words':[{'start':float(w.start),'end':float(w.end),'text':w.word,'probability':float(w.probability)} for w in segment.words or []]})
    cancel_check()
    progress(99,'检查歌词所在区间是否有人声，排除前奏误识别…')
    return filter_vocal_lines(result,path,cancel_check)

def lyric_text(lines):
    return '\n'.join(f"{line['start']:.2f}\t{line['end']:.2f}\t{line['text']}" for line in lines)

def parse_text(text,duration):
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        fields=line.split('\t',2)
        if len(fields)!=3:raise ValueError('每行应为：开始秒数、结束秒数、歌词，以 Tab 分隔')
        a,b=map(float,fields[:2])
        import math
        if not math.isfinite(a+b) or not 0<=a<b<=duration+1:raise ValueError('歌词时间无效')
        result.append({'start':a,'end':b,'text':fields[2],'reviewed':True,'words':[]})
    return sorted(result,key=lambda x:x['start'])
