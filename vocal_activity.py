"""AudioSet vocal-event gate (speech and singing), not a pitch detector.

PANNs CNN6 uses multi-second context; this is a conservative segment gate, not
an exact vocal onset detector. False positives and false negatives are possible.
"""
from pathlib import Path
import csv
import numpy as np

from runtime_paths import get_app_root, resolve_model_path

ROOT=get_app_root()
_DEFAULT_WEIGHTS=resolve_model_path('panns-cnn6.pth')
WEIGHTS=_DEFAULT_WEIGHTS
_MODEL=None
_LABELS=None
SINGING={'Speech','Male speech, man speaking','Female speech, woman speaking','Child speech, kid speaking','Conversation','Narration, monologue','Singing','Male singing','Female singing','Choir','Yodeling','Chant','Mantra','Child singing','Synthetic singing','Humming','Vocal music','A capella','Rapping'}


def _active_weights():
    return WEIGHTS if WEIGHTS != _DEFAULT_WEIGHTS else resolve_model_path('panns-cnn6.pth')


def ready():
    return _active_weights().is_file()


def load_model():
    global _MODEL,_LABELS
    if _MODEL is None:
        import torch
        from third_party.panns.cnn6 import Cnn6
        if not ready():
            raise RuntimeError('人声筛选模型未安装，请运行 install_singing_model.py。')
        if torch.get_num_threads() > 16:
            torch.set_num_threads(16)
        model=Cnn6(32000,1024,320,64,50,14000,527).eval()
        state=torch.load(_active_weights(),map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True)
        with (ROOT/'third_party/panns/labels.csv').open(encoding='utf-8',newline='') as f:
            _LABELS=[row['display_name'] for row in csv.DictReader(f)]
        _MODEL=model
    return _MODEL,_LABELS


def singing_scores(y,sr,cancel_check=lambda:None):
    import librosa
    import torch
    model,labels=load_model()
    indices=[i for i,label in enumerate(labels) if label in SINGING]
    if not indices:
        raise RuntimeError('人声分类标签缺失')
    audio=librosa.resample(np.asarray(y,dtype=np.float32),orig_sr=sr,target_sr=32000) if sr!=32000 else y
    duration=len(audio)/32000
    centers=np.arange(0,max(duration,.01),1.0)
    scores=[]
    for center in centers:
        cancel_check()
        left=round((center-2)*32000);right=left+128000
        segment=audio[max(0,left):min(len(audio),right)]
        segment=np.pad(segment,(max(0,-left),max(0,right-len(audio))))
        with torch.inference_mode():
            output=model(torch.tensor(segment).float().unsqueeze(0))['clipwise_output'][0].numpy()
        scores.append(float(np.max(output[indices])))
    return centers,np.asarray(scores)


def filter_notes(notes,centers,scores,threshold=.08):
    """Retain a note when at least half its interior has vocal evidence."""
    kept=[]
    for note in notes:
        probes=np.linspace(note.raw_start+.1*note.raw_duration,note.raw_start+.9*note.raw_duration,5)
        values=np.interp(probes,centers,scores)
        if np.median(values)>=threshold:
            kept.append(note)
    return kept
