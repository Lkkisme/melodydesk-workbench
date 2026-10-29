"""Fixed first-eight evaluation against both human annotations; no best-of picking."""
from pathlib import Path
import json
import hashlib
import shutil
from dataclasses import asdict
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from music import Note, Project, quantize, numbered
import engine


def baseline(audio, bpm, origin):
    """Original v1 pitch segmentation, including grid-collision filtering."""
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    notes = []
    for sample_start in range(0, len(y), 8 * sr):
        segment = y[sample_start:sample_start + 8 * sr]
        if len(segment) < 2048:
            segment = np.pad(segment, (0, 2048 - len(segment)))
        f0, voiced, probability = librosa.pyin(segment, sr=sr, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=2048, hop_length=256, fill_na=np.nan)
        pitches = np.where(voiced & np.isfinite(f0), np.rint(librosa.hz_to_midi(np.nan_to_num(f0, nan=440))), -1).astype(int)
        borders = np.flatnonzero(np.r_[True, np.diff(pitches >= 0) != 0, True])
        for a, b in zip(borders, borders[1:]):
            if pitches[a] >= 0 and b - a >= 3:
                pitches[a:b] = median_filter(pitches[a:b], size=3, mode='nearest')
        boundaries = set(np.flatnonzero(np.r_[True, np.diff(pitches) != 0, True]).tolist())
        boundaries.update(int(v) for v in librosa.onset.onset_detect(y=segment, sr=sr, hop_length=256) if 0 < v < len(pitches))
        boundaries = sorted(boundaries)
        for a, b in zip(boundaries, boundaries[1:]):
            pitch = int(pitches[a]); start = sample_start/sr+a*256/sr
            end = min(len(y)/sr, sample_start/sr+b*256/sr)
            if pitch >= 0 and end-start >= .085:
                notes.append(Note(pitch,0,1,start,end-start,float(np.nan_to_num(np.mean(probability[a:b])))))
    joined=[]
    for n in notes:
        if joined and abs(n.raw_start/8-round(n.raw_start/8))<.004 and n.pitch==joined[-1].pitch and abs(n.raw_start-joined[-1].raw_start-joined[-1].raw_duration)<.04:
            joined[-1].raw_duration=n.raw_start+n.raw_duration-joined[-1].raw_start
        else:
            joined.append(n)
    return quantize(joined,bpm,origin)


def score(ref, notes, tolerance=.05, offsets=False):
    if not notes:
        return dict(precision=0,recall=0,f1=0,matches=0)
    pred=np.array([[n.raw_start,n.pitch,n.raw_duration] for n in notes])
    eligible=(np.abs(ref[:,0,None]-pred[None,:,0])<=tolerance) & (np.abs(ref[:,1,None]-pred[None,:,1])<=.5)
    if offsets:
        eligible &= np.abs((ref[:,0]+ref[:,2])[:,None]-(pred[:,0]+pred[:,2])[None,:])<=np.maximum(.05,ref[:,2]*.2)[:,None]
    matches=int(np.sum(maximum_bipartite_matching(csr_matrix(eligible),perm_type='column')>=0))
    return dict(precision=matches/len(notes),recall=matches/len(ref),f1=2*matches/(len(notes)+len(ref)),matches=matches)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(engine.ROOT.parent/'公开样本测试'))
    args=parser.parse_args()
    root=engine.WORK/'benchmarks'; out=Path(args.output);out.mkdir(exist_ok=True)
    cache=root/'predictions';cache.mkdir(exist_ok=True)
    digest=hashlib.sha256((engine.ROOT/'pitch_notes.py').read_bytes()+(engine.ROOT/'engine.py').read_bytes()).hexdigest()[:10]
    results=[]
    for i in range(1,9):
        name=f'vocadito_{i}';audio=root/'Audio'/(name+'.wav')
        print(name,'start',flush=True)
        cached=cache/(name+'-'+digest+'.json')
        if cached.exists():
            data=json.loads(cached.read_text());old=[Note(**n) for n in data['old']];new=[Note(**n) for n in data['new']]
        else:
            new,bpm,origin,_=engine.transcribe(audio,audio,vocal_only=False)
            old=baseline(audio,bpm,origin)
            cached.write_text(json.dumps({'old':[asdict(n) for n in old],'new':[asdict(n) for n in new]}))
        sample=out/name;sample.mkdir(exist_ok=True);shutil.copy2(audio,sample/audio.name)
        for annotator in ('A1','A2'):
            annotation=root/'Annotations'/'Notes'/(name+'_notes'+annotator+'.csv')
            shutil.copy2(annotation,sample/annotation.name)
            ref=np.loadtxt(annotation,delimiter=',');ref[:,1]=librosa.hz_to_midi(ref[:,1])
            text=['# '+name+' / '+annotator+' 人工标注转事件简谱',
                '来源：Bittner 等，vocadito v3，https://zenodo.org/records/5578807，CC BY 4.0。',
                '以 1=C 显示音级，不代表原曲调性；单引号／逗号表示高／低八度。',
                '时值按标注真实秒数列出，不伪造固定速度或拍号；非出版简谱。评测使用原始频率，不使用取整音高。','',
                '| 起音 / 秒 | 音级 | 持续 / 秒 |','|---:|:---:|---:|']
            text += [f'| {a:.3f} | {numbered(round(p))} | {d:.3f} |' for a,p,d in ref]
            (sample/(annotator+'-参考简谱.md')).write_text('\n'.join(text),encoding='utf-8')
            row=dict(track=name,annotator=annotator,reference_notes=len(ref),old_notes=len(old),new_notes=len(new))
            for label,notes in [('old',old),('new',new)]:
                row[label+'_onset50']=score(ref,notes);row[label+'_onset100']=score(ref,notes,.1)
                row[label+'_offset']=score(ref,notes,offsets=True)
            results.append(row)
        for label,notes in [('旧版',old),('修正版',new)]:
            engine.synthesize(Project(title=name,notes=notes),sample/(label+'旋律.wav'),raw=True)
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,'old',len(old),'new',len(new),'F1',round(results[-2]['old_onset50']['f1'],3),round(results[-2]['new_onset50']['f1'],3),flush=True)
    summary={model+'_'+metric:float(np.mean([row[model+'_'+metric]['f1'] for row in results])) for model in ('old','new') for metric in ('onset50','onset100','offset')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
