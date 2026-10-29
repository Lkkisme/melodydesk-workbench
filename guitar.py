"""Editable melody-conditioned accompaniment drafts, not chord recognition."""
from pathlib import Path
import copy
import html
import json
import math
import re
import struct
import wave
from music import NAMES, vlq

TUNING=(40,45,50,55,59,64)  # low E -> high e
OPEN={'C':[-1,3,2,0,1,0],'D':[-1,-1,0,2,3,2],'E':[0,2,2,1,0,0],
      'F':[1,3,3,2,1,1],'G':[3,2,0,0,0,3],'A':[-1,0,2,2,2,0],
      'Am':[-1,0,2,2,1,0],'Dm':[-1,-1,0,2,3,1],'Em':[0,2,2,0,0,0]}


def parse_chord(name):
    if name=='N.C.':
        return None
    match=re.fullmatch(r'([A-G](?:#|b)?)(m?)',name.strip())
    if not match:
        raise ValueError('和弦格式：C、F#m、Bb，或 N.C.；当前支持大／小三和弦。')
    natural={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
    root=natural[match[1][0]]+('#' in match[1])-('b' in match[1])
    return root%12,match[2]=='m'


def chord_name(root,minor=False):
    return NAMES[root%12]+('m' if minor else '')


def suggest_capo(chords):
    def effort(capo):
        score=capo*.12
        for chord in chords:
            parsed=parse_chord(chord)
            if parsed is None:continue
            root,minor=parsed
            score+=0 if chord_name(root-capo,minor) in OPEN else 2
            score+=max(voicing(chord,capo))*.05
        return score
    return min(range(8),key=effort)


def voicing(name,capo=0):
    """Frets relative to capo; sounding chord remains unchanged."""
    parsed=parse_chord(name)
    if parsed is None:
        return [-1]*6
    root,minor=parsed
    shape=chord_name(root-capo,minor)
    if shape in OPEN:
        return OPEN[shape][:]
    root=(root-capo)%12
    ebase=(root-4)%12
    abase=(root-9)%12
    e=[ebase+x for x in ([0,2,2,0,0,0] if minor else [0,2,2,1,0,0])]
    a=[-1]+[abase+x for x in ([0,2,2,1,0] if minor else [0,2,2,2,0])]
    return min((e,a),key=lambda v:(max(v),sum(f>0 for f in v)))


def key_candidates(project):
    # Deliberately simple melody-only prior, not a calibrated key detector.
    weights=[0.0]*12
    for n in project.notes:
        weights[n.pitch%12]+=n.duration
    total=sum(weights) or 1
    options=[]
    for root in range(12):
        for minor in (False,True):
            scale={0,2,3,5,7,8,10} if minor else {0,2,4,5,7,9,11}
            inside=sum(w for pc,w in enumerate(weights) if (pc-root)%12 in scale)/total
            tonic=weights[root]/total
            fifth=weights[(root+7)%12]/total
            final=.04 if project.notes and project.notes[-1].pitch%12==root else 0
            options.append((inside+.12*tonic+.04*fifth+final,root,minor))
    return sorted(options,reverse=True)


def suggest(project,root,minor=False):
    project.validate()
    measure=int(project.meter.split('/')[0])*4/int(project.meter.split('/')[1])
    count=math.ceil(max((n.start+n.duration for n in project.notes),default=0)/measure)
    if project.source:
        count=max(count,math.ceil((project.clip_duration-project.beat_origin)*project.bpm/60/measure))
    if count>1000:
        raise ValueError('当前编配最多支持 1000 小节')
    degrees=[(0,True),(3,False),(5,True),(7,False),(8,False),(10,False)] if minor else [(0,False),(2,True),(4,True),(5,False),(7,False),(9,True)]
    names=[chord_name(root+d,m) for d,m in degrees]
    rows=[]
    for bar in range(count):
        weights=[0.0]*12
        for n in project.notes:
            overlap=max(0,min(n.start+n.duration,(bar+1)*measure)-max(n.start,bar*measure))
            # Longer notes and notes entering on the bar line carry more weight.
            weights[n.pitch%12]+=overlap*(1.25 if abs(n.start-bar*measure)<.13 else 1)
        if not sum(weights):
            rows.append({'N.C.':0});continue
        emission={}
        for name in names:
            r,m=parse_chord(name); tones={r,(r+(3 if m else 4))%12,(r+7)%12}
            emission[name]=sum(w*(1 if pc in tones else -.45) for pc,w in enumerate(weights))/sum(weights)
        rows.append(emission)
    states={None:(0,[])}
    for row in rows:
        updated={}
        for name,fit in row.items():
            choices=[]
            for previous,(cost,path) in states.items():
                transition=0
                if previous and previous!='N.C.' and name!='N.C.':
                    distance=(parse_chord(name)[0]-parse_chord(previous)[0])%12
                    transition=.06 if distance in (5,7) else -.02 if previous==name else -.05
                choices.append((cost+fit+transition,path+[name]))
            updated[name]=max(choices,key=lambda x:x[0])
        states=updated
    return max(states.values(),key=lambda x:x[0])[1] if rows else []


def arrange(project,chords,style='分解和弦',capo=0):
    project.validate()
    if style not in ('分解和弦','流行扫弦') or not isinstance(capo,int) or not 0<=capo<=7:
        raise ValueError('编配风格或变调夹位置无效')
    measure=int(project.meter.split('/')[0])*4/int(project.meter.split('/')[1])
    count=math.ceil(max((n.start+n.duration for n in project.notes),default=0)/measure)
    if project.source:
        count=max(count,math.ceil((project.clip_duration-project.beat_origin)*project.bpm/60/measure))
    if len(chords)!=count or not 0<count<=1000:
        raise ValueError(f'需要按顺序填写 {count} 个和弦，每小节一个')
    bars=[];events=[]
    for bar,name in enumerate(chords):
        frets=voicing(name,capo)
        active=[i for i,f in enumerate(frets) if f>=0]
        slots=[]
        for step in range(round(measure*2)):
            if not active:
                slots.append([]);continue
            if style=='分解和弦':
                order=[active[0],active[-3],active[-2],active[-1],active[-2],active[-3],active[-2],active[-1]]
                strings=[order[step%len(order)]]
                direction=''
            else:
                # A sparse 4/4 pattern; other meters use alternating eighths.
                if measure==4 and step in (1,4):
                    slots.append([]);continue
                direction='↓' if step%2==0 else '↑'
                strings=active[:] if direction=='↓' else list(reversed(active[-4:]))
            slot=[]
            for order_index,string in enumerate(strings):
                at=bar*measure+step*.5+(order_index*.012*project.bpm/60 if style=='流行扫弦' else 0)
                # Short plucks make the displayed eighth-note timing unambiguous.
                duration=min(.48,bar*measure+min(measure,step*.5+.5)-at)
                event={'start':at,'duration':duration,'pitch':TUNING[string]+frets[string]+capo,
                       'string':string,'fret':frets[string],'velocity':82 if step==0 else 67,'bar':bar}
                events.append(event);slot.append({'string':string,'fret':frets[string],'direction':direction})
            slots.append(slot)
        bars.append({'chord':name,'frets':frets,'slots':slots})
    from dataclasses import asdict
    return {'title':project.title,'bpm':project.bpm,'meter':project.meter,'capo':capo,'style':style,'bars':bars,'events':events,
            'measure':measure,'tonic':project.tonic,'beat_origin':project.beat_origin,'melody':[asdict(n) for n in project.notes],'lyrics':copy.deepcopy(project.lyrics),
            'warning':'旋律条件下的编配草稿，不是原曲和弦识别。每小节一个和弦；请确认调性、拍点和和弦。'}


def midi(project,arrangement):
    ticks=480
    tracks=[]
    numerator,denominator=map(int,project.meter.split('/'))
    header=[(0,0,b'\xff\x51\x03'+round(60_000_000/project.bpm).to_bytes(3,'big')),
            (0,0,b'\xff\x58\x04'+bytes([numerator,int(math.log2(denominator)),24,8]))]
    for channel,program,notes in [(0,0,[{'start':n.start,'duration':n.duration,'pitch':n.pitch,'velocity':83} for n in project.notes]),(1,24,arrangement['events'])]:
        events=(header[:] if channel==0 else [])+[(0,0,bytes([0xc0+channel,program]))]
        for n in notes:
            a=round(n['start']*ticks);b=max(a+1,round((n['start']+n['duration'])*ticks))
            events.extend([(a,2,bytes([0x90+channel,n['pitch'],n['velocity']])),(b,1,bytes([0x80+channel,n['pitch'],0]))])
        data=bytearray();previous=0
        for tick,_,event in sorted(events,key=lambda e:(e[0],e[1])):
            data.extend(vlq(tick-previous));data.extend(event);previous=tick
        data.extend(b'\x00\xff\x2f\x00')
        tracks.append(b'MTrk'+struct.pack('>I',len(data))+data)
    return b'MThd'+struct.pack('>IHHH',6,1,2,ticks)+b''.join(tracks)


def render_audio(project,arrangement,path,with_melody=True):
    import numpy as np
    sr=22050;spb=60/project.bpm
    end=max([n.start+n.duration for n in project.notes]+[n['start']+n['duration'] for n in arrangement['events']])+1
    if project.source:
        end=max(end,project.clip_duration/spb)
    if end*spb>3600:
        raise ValueError('试听时长超过一小时')
    sound=np.zeros(math.ceil(end*spb*sr),dtype=np.float32)
    notes=[(e['pitch'],e['start'],e['duration'],e['velocity']/127,True) for e in arrangement['events']]
    if with_melody:
        notes += [(n.pitch,n.start,n.duration,.65,False) for n in project.notes]
    for pitch,start,duration,volume,guitar in notes:
        count=max(1,round(duration*spb*sr));t=np.arange(count)/sr
        frequency=440*2**((pitch-69)/12)
        signal=np.zeros(count)
        for harmonic in range(1,7 if guitar else 3):
            if frequency*harmonic>=sr/2:break
            signal+=np.sin(2*np.pi*frequency*harmonic*t)/harmonic**(1.4 if guitar else 2)
        signal*=np.exp(-t*(5 if guitar else 1))*volume*.12
        ramp=min(count//2,round(.008*sr))
        if ramp:
            signal[:ramp]*=np.linspace(0,1,ramp);signal[-ramp:]*=np.linspace(1,0,ramp)
        offset=round(start*spb*sr);sound[offset:offset+count]+=signal.astype(np.float32)
    peak=float(np.max(np.abs(sound)))
    if peak>.95:sound*=.95/peak
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes((sound*32767).astype('<i2').tobytes())


def export_bundle(project,chords,directory,style='分解和弦',capo=0):
    arrangement=arrange(project,chords,style,capo)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'arrangement.json').write_text(json.dumps(arrangement,ensure_ascii=False,indent=2),encoding='utf-8')
    project.save(directory/'melody-project.json')
    from lyrics import lyric_text
    (directory/'lyrics.txt').write_text(lyric_text(project.lyrics),encoding='utf-8-sig')
    (directory/'accompaniment.mid').write_bytes(midi(project,arrangement))
    render_audio(project,arrangement,directory/'melody-guitar.wav')
    render_audio(project,arrangement,directory/'guitar-only.wav',False)
    path=directory/'guitar-score.html'
    path.write_text(score_html(arrangement),encoding='utf-8')
    from score_pdf import export_pdf
    export_pdf(arrangement,directory/'guitar-score.pdf')
    return path


# Current combined renderer; old standalone exports remain readable files.
from score_layout import score_html
