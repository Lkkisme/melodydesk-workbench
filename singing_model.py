"""CPU adapter for official ROSVOT + RWBD + RMVPE checkpoints.

Upstream MIT implementation is retained under third_party/rosvot.
Only separated vocal audio is supplied by the application. This is not a
general-purpose instrument transcription model or a lead/backing separator.
"""
from pathlib import Path
import sys
import threading
import numpy as np
from music import Note

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT.parents[1] / 'work' / 'model-cache' / 'rosvot' / 'checkpoints'
_MODELS = {}
_LOCK = threading.Lock()


def ready():
    return all((CHECKPOINTS/name/'model.pt').is_file() for name in ('rosvot','rwbd','rmvpe')) and all((CHECKPOINTS/name/'config.yaml').is_file() for name in ('rosvot','rwbd'))


def load_models(device='cpu'):
    global _MODELS
    with _LOCK:
        if isinstance(_MODELS, dict) and device in _MODELS:
            return _MODELS[device]
        if not ready():
            raise RuntimeError('演唱识别模型尚未安装，请运行 install_singing_model.py；可先选择基础模式。')
        sys.path.insert(0,str(ROOT/'third_party'/'rosvot'))
        import torch
        import yaml
        from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor
        from modules.pe.rmvpe import RMVPE
        from utils.audio.mel import MelNet
        torch.set_num_threads(min(4,torch.get_num_threads()))
        hp=yaml.safe_load((CHECKPOINTS/'rosvot'/'config.yaml').read_text(encoding='utf-8'))
        wp=yaml.safe_load((CHECKPOINTS/'rwbd'/'config.yaml').read_text(encoding='utf-8'))
        hp['note_bd_threshold']=.5
        wp['word_bd_threshold']=.5
        model=MidiExtractor(hp).eval().to(device)
        words=WordbdExtractor(wp).eval().to(device)
        for name,network in [('rosvot',model),('rwbd',words)]:
            state=torch.load(CHECKPOINTS/name/'model.pt',map_location='cpu',weights_only=True)['state_dict']
            state=state['model'] if 'model' in state else {k[6:]:v for k,v in state.items() if k.startswith('model.')}
            network.load_state_dict(state,strict=True)
        pitch=RMVPE(str(CHECKPOINTS/'rmvpe'/'model.pt'),device=device)
        melnet=MelNet(hp,device=device)
        loaded=(hp,wp,model,words,pitch,melnet)
        if not isinstance(_MODELS, dict):
            _MODELS={}
        _MODELS[device]=loaded
        return loaded


def _trim_and_split_voiced_intervals(pitches, intervals, uv, probability, sr, hop, wav_len):
    """Trim unvoiced margins and split intervals across breath silences (~0.26s unvoiced gap)."""
    max_uv_bridge = max(1, round(0.26 * sr / hop))
    tail_pad = max(1, round(0.025 * sr / hop))
    notes = []
    for index, (pitch, (a, b)) in enumerate(zip(pitches, intervals)):
        if not 21 <= pitch <= 108 or b <= a:
            continue
        seg_voiced = np.flatnonzero(~uv[a:b])
        if seg_voiced.size == 0:
            continue
        uv_gaps = np.where(np.diff(seg_voiced) > max_uv_bridge)[0]
        v_starts = np.r_[0, uv_gaps + 1]
        v_ends = np.r_[uv_gaps, len(seg_voiced) - 1]
        for sv, ev in zip(v_starts, v_ends):
            sub_a = a + int(seg_voiced[sv])
            next_bound = b if ev == len(seg_voiced) - 1 else (a + int(seg_voiced[ev + 1]))
            sub_b = min(next_bound, a + int(seg_voiced[ev]) + 1 + tail_pad)
            start = sub_a * hop / sr
            end = min(wav_len / sr, sub_b * hop / sr)
            if end - start < 0.045:
                continue
            voiced_fraction = float(np.mean(~uv[sub_a:sub_b])) if sub_b > sub_a else 0.0
            if voiced_fraction < 0.15:
                continue
            notes.append(Note(int(pitch), 0, 1, float(start), float(end - start), float(probability[index, pitch])))
    return notes


def predict_chunk(wav, models, word_boundaries_sec=None, return_dict=False):
    import torch
    from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval
    hp,wp,model,words,pe,melnet=models
    sr=hp['audio_sample_rate'];hop=hp['hop_size']
    device=next(model.parameters()).device
    with torch.inference_mode():
        mel=melnet(wav)
        real_length=mel.shape[1]
        # Upstream data loader pads frames to the U-Net downsampling multiple.
        length=((real_length+15)//16)*16
        mel=torch.nn.functional.pad(mel,(0,0,0,length-real_length))
        # Use the same postprocessing as upstream batch inference.
        f0=pe.infer_from_audio(wav,sample_rate=sr)
        f0=pe.postprocess(f0,hp['f0_min'],hp['f0_max'],min_gap=6)
        from utils.audio.pitch_utils import resample_align_curve
        unvoiced=f0==0
        aligned=resample_align_curve(f0,.01,hop/sr,length)
        uv=resample_align_curve(unvoiced.astype(np.float32),.01,hop/sr,length)>.5
        uv[real_length:]=True
        aligned[uv]=0
        normalized,uv=norm_interp_f0(aligned)
        uv_t=torch.from_numpy(uv).long().unsqueeze(0).to(device)
        f0_t=torch.from_numpy(normalized).float().unsqueeze(0).to(device)
        coarse=f0_to_coarse(denorm_f0(f0_t,uv_t))
        nonpadding=mel.abs().sum(-1)>0
        if word_boundaries_sec is None:
            w=words(mel=mel[:,:,:wp.get('use_mel_bins',80)],pitch=coarse,uv=uv_t,non_padding=nonpadding,train=False)
            word_bd=w['word_bd_pred']
        else:
            word_bd=torch.zeros((1,length),dtype=torch.long,device=mel.device)
            for seconds in word_boundaries_sec:
                frame=round(float(seconds)*sr/hop)
                if 0<frame<real_length-1:
                    word_bd[0,frame]=1
        output=model(mel=mel[:,:,:hp.get('use_mel_bins',80)],word_bd=word_bd,pitch=coarse,
                     uv=uv_t,non_padding=nonpadding,train=False)
        size=int(output['note_lengths'][0])
        pitches=output['note_pred'][0,:size].detach().cpu().numpy()
        intervals=boundary2Interval(output['note_bd_pred'][0,:real_length].detach().cpu().numpy())
        probability=torch.softmax(output['note_logits'][0,:size],dim=-1).detach().cpu().numpy()
        boundary_frames=torch.nonzero(word_bd[0,:real_length],as_tuple=False).flatten().detach().cpu().tolist()
        detected_boundaries=[float(frame*hop/sr) for frame in boundary_frames]
    notes=_trim_and_split_voiced_intervals(pitches,intervals,uv,probability,sr,hop,len(wav))
    if return_dict:
        return {'notes': notes, 'word_boundaries_sec': detected_boundaries}
    return notes


def track_singing(y, sr, cancel_check=lambda:None, progress=lambda p,s:None, *, device='cpu', word_boundaries_sec=None, return_dict=False):
    import librosa
    cancel_check()
    progress(55,'加载演唱音符模型…')
    models=load_models(device=device)
    target=models[0]['audio_sample_rate']
    if sr != target:
        y=librosa.resample(y,orig_sr=sr,target_sr=target)
    duration=len(y)/target
    result=[]
    detected_boundaries=[]
    # Context on both sides reduces artificial notes at chunk boundaries.
    core_seconds=12
    for core_start in np.arange(0,duration,core_seconds):
        cancel_check()
        core_end=min(duration,core_start+core_seconds)
        left=max(0,core_start-.75);right=min(duration,core_end+.75)
        segment=y[round(left*target):round(right*target)]
        if len(segment)<512:
            continue
        local_boundaries=None if word_boundaries_sec is None else [b-left for b in word_boundaries_sec if left<=b<right]
        pred=predict_chunk(segment,models,word_boundaries_sec=local_boundaries,return_dict=True)
        local=pred['notes']
        for b_loc in pred['word_boundaries_sec']:
            b_abs=b_loc+left
            if core_start<=b_abs<core_end:
                detected_boundaries.append(float(b_abs))
        for note in local:
            absolute_start=note.raw_start+left
            absolute_end=absolute_start+note.raw_duration
            start=max(core_start,absolute_start);end=min(core_end,absolute_end)
            if end-start<.025:
                continue
            note.raw_start=float(start);note.raw_duration=float(end-start)
            all_bds=word_boundaries_sec if word_boundaries_sec is not None else detected_boundaries
            has_word_split=any(abs(b-start)<.02 for b in all_bds)
            if result and not has_word_split and abs(start-core_start)<.015 and result[-1].pitch==note.pitch and abs(result[-1].raw_start+result[-1].raw_duration-start)<.02:
                result[-1].raw_duration=end-result[-1].raw_start
            else:
                result.append(note)
        progress(58+32*core_end/duration,f'演唱旋律已分析 {core_end:.0f} / {duration:.0f} 秒')
    cancel_check()
    if return_dict:
        return {
            'notes': result,
            'word_boundaries_sec': list(word_boundaries_sec) if word_boundaries_sec is not None else sorted(set(detected_boundaries)),
            'backend': 'rosvot-rmvpe',
            'device': device,
        }
    return result
