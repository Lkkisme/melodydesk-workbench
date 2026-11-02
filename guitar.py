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

ALBRECHT_MAJOR=(0.238,0.006,0.111,0.006,0.137,0.094,0.016,0.214,0.009,0.080,0.008,0.081)
ALBRECHT_MINOR=(0.220,0.006,0.104,0.123,0.019,0.103,0.012,0.214,0.062,0.022,0.061,0.052)

# Functional-degree transition model. The pseudo-counts are hand-specified
# priors (not corpus-fitted); rows are Dirichlet-smoothed (alpha) and
# row-normalized into stochastic matrices (sum_j P_ij = 1, P_ij > 0). The
# additive Viterbi weight is the scaled log-likelihood ratio c*log(P_ij/(1/K)).
# Major degree order: I ii iii IV V vi.  Minor: i III iv v VI VII.
_TRANS_ALPHA=1.0
_TRANS_LOG_SCALE=0.06
_GUITAR_MAJ_COUNTS=(
    ( 2,12, 7,25,25,18),
    ( 9, 2, 7,10,42, 9),
    ( 7,12, 2,21, 9,30),
    (25,10, 6, 2,25, 9),
    (59, 7, 7,10, 2,30),
    ( 9,21,10,30,15, 2),
)
_GUITAR_MIN_COUNTS=(
    ( 2,12,25, 9, 9,25),
    (12, 2,15, 9,12,15),
    (12,10, 2,18,10,12),
    (21,12,12, 2,15,12),
    (10,12,12,10, 2,18),
    (25,12,10, 7,10, 2),
)


def _row_stochastic(counts,alpha=_TRANS_ALPHA):
    """Dirichlet posterior-mean rows: P_ij=(n_ij+alpha)/sum_k(n_ik+alpha)."""
    return tuple(tuple((c+alpha)/sum(x+alpha for x in row) for c in row) for row in counts)


def _log_ratio_matrix(probs,scale=_TRANS_LOG_SCALE):
    return tuple(tuple(scale*math.log(p*len(row)) for p in row) for row in probs)


_GUITAR_MAJ_PROBS=_row_stochastic(_GUITAR_MAJ_COUNTS)
_GUITAR_MIN_PROBS=_row_stochastic(_GUITAR_MIN_COUNTS)
_GUITAR_MAJ_TRANS=_log_ratio_matrix(_GUITAR_MAJ_PROBS)
_GUITAR_MIN_TRANS=_log_ratio_matrix(_GUITAR_MIN_PROBS)

# Tonal Interval Space DFT-coefficient weights (Bernardes et al. 2016).
_TIS_WEIGHTS=(2.0,11.0,17.0,16.0,19.0,7.0)


def _tiv_vec(chroma):
    """6-harmonic Tonal Interval Vector (real/imag parts) of a 12-bin chroma."""
    total=sum(chroma)
    if total<=1e-12:
        return [0.0]*12
    re_part=[];im_part=[]
    for k,w in enumerate(_TIS_WEIGHTS,start=1):
        re_k=sum(c*math.cos(2*math.pi*k*pc/12) for pc,c in enumerate(chroma))/total
        im_k=-sum(c*math.sin(2*math.pi*k*pc/12) for pc,c in enumerate(chroma))/total
        re_part.append(w*re_k);im_part.append(w*im_k)
    return re_part+im_part


def _cosine(a,b):
    na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b))
    if na<1e-12 or nb<1e-12:
        return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)


def _pearson(x,y):
    """Mean-centered, variance-normalized Pearson correlation; 0 for flat inputs."""
    mx=sum(x)/len(x);my=sum(y)/len(y)
    dx=[v-mx for v in x];dy=[v-my for v in y]
    den=math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
    if den<1e-12:
        return 0.0
    return sum(a*b for a,b in zip(dx,dy))/den
OPEN={'C':[-1,3,2,0,1,0],'D':[-1,-1,0,2,3,2],'E':[0,2,2,1,0,0],
      'F':[1,3,3,2,1,1],'G':[3,2,0,0,0,3],'A':[-1,0,2,2,2,0],
      'Am':[-1,0,2,2,1,0],'Dm':[-1,-1,0,2,3,1],'Em':[0,2,2,0,0,0]}


def parse_chord(name):
    if not isinstance(name,str):
        raise ValueError('和弦格式：C、F#m、Bb，或 N.C.；当前支持大／小三和弦。')
    s=name.strip()
    if s=='N.C.':
        return None
    match=re.fullmatch(r'([A-G](?:#|b)?)(m7|maj7|dim|aug|7|m?)',s)
    if match:
        natural={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
        root=natural[match[1][0]]+('#' in match[1])-('b' in match[1])
        q=match[2]
        return root%12,(True if q=='m' else False if q=='' else q)
    from subpkgs.guitar_reharm.chord_voicing import parse_extended_chord
    ext=parse_extended_chord(s)
    if ext is None:
        raise ValueError('和弦格式：C、F#m、Bb，或 N.C.；当前支持大／小三和弦。')
    q_full=ext['quality']+('/'+ext['bass'] if ext['bass'] else '')
    return ext['root'],(True if q_full=='m' else False if q_full=='' else q_full)


def chord_name(root,minor=False):
    suf=minor if isinstance(minor,str) else ('m' if minor else '')
    return NAMES[root%12]+suf


def suggest_capo(chords):
    def effort(capo):
        score=capo*.12
        for chord in chords:
            parsed=parse_chord(chord)
            if parsed is None:continue
            root,minor=parsed
            score+=0 if (not isinstance(minor,str) and chord_name(root-capo,minor) in OPEN) else 2
            score+=max(0,max(voicing(chord,capo)))*.05
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
    if isinstance(minor,str):
        from subpkgs.guitar_reharm.chord_voicing import generate_voicings
        vs=generate_voicings(name,capo=capo) if '/' in minor else generate_voicings(shape,capo=0)
        if vs:return list(vs[0]['frets'])
        raise ValueError(f'无法在当前把位生成可弹奏指型: {name}')
    root=(root-capo)%12
    ebase=(root-4)%12
    abase=(root-9)%12
    is_m=(minor is True) or (isinstance(minor,str) and minor.startswith(('m','dim')) and not minor.startswith('maj'))
    e=[ebase+x for x in ([0,2,2,0,0,0] if is_m else [0,2,2,1,0,0])]
    a=[-1]+[abase+x for x in ([0,2,2,1,0] if is_m else [0,2,2,2,0])]
    return min((e,a),key=lambda v:(max(v),sum(f>0 for f in v)))


def key_candidates(project):
    weights=[0.0]*12
    for n in project.notes:
        weights[n.pitch%12]+=n.duration
    total=sum(weights) or 1
    options=[]
    for root in range(12):
        for minor in (False,True):
            profile=ALBRECHT_MINOR if minor else ALBRECHT_MAJOR
            rotated=[profile[(pc-root)%12] for pc in range(12)]
            corr=_pearson(weights,rotated)
            triad={root,(root+(3 if minor else 4))%12,(root+7)%12}
            triad_w=sum(weights[pc] for pc in triad)/total
            fifth=weights[(root+7)%12]/total
            final=.06 if project.notes and project.notes[-1].pitch%12==root else 0
            options.append((corr+.15*triad_w+.05*fifth+final,root,minor))
    return sorted(options,reverse=True)


def _gttm_salience(rel_onset,overlap,measure,meter='4/4'):
    sigma=0.14
    num_m,den_m=4,4
    if isinstance(meter,str) and '/' in meter:
        try:num_m,den_m=map(int,meter.strip().split('/',1))
        except ValueError:num_m,den_m=4,4
    sub_u=4.0/den_m if den_m>0 else 1.0
    if den_m>=8 and num_m%3==0 and num_m>=6:
        grid=[(0.0,1.45)]
        n_cb=num_m//3
        c_step=3.0*sub_u
        for cb in range(1,n_cb):
            grid.append((cb*c_step,1.30 if (n_cb%2==0 and cb==n_cb//2) else 1.25))
        for s_i in range(num_m):
            if s_i%3!=0:grid.append((s_i*sub_u,0.88))
    else:
        grid=[(0.0,1.45)]
        if num_m%2==0 and num_m>=2:
            grid.append((measure*0.5,1.22))
        for q in range(1,num_m):grid.append((q*sub_u,1.05))
        for q in range(num_m*2):grid.append((q*sub_u*0.5,0.85))
    max_salience=0.85
    for pos,w in grid:
        d=abs(rel_onset-pos)
        if d>measure/2:d=measure-d
        g=math.exp(-0.5*(d/sigma)**2)
        v=w*g
        if v>max_salience:max_salience=v
    dur_f=min(1.45,max(0.35,0.40+0.65*math.log(1+2.2*overlap)))
    return max_salience*dur_f


def suggest(project,root=None,minor=False):
    project.validate()
    if root is None:
        root=NAMES.index(project.tonic)
    elif isinstance(root,str):
        parsed=parse_chord(root)
        if parsed is None or (isinstance(parsed[1],str) and '/' in parsed[1]):
            raise ValueError('无效的主音或调号')
        root=parsed[0]
        minor=bool(minor or (parsed[1] is True) or (isinstance(parsed[1],str) and parsed[1].startswith(('m','dim')) and not parsed[1].startswith('maj')))
    elif isinstance(root,bool) or not isinstance(root,int) or not 0<=root<12:
        raise ValueError('无效的主音或调号')
    measure=int(project.meter.split('/')[0])*4/int(project.meter.split('/')[1])
    note_count=math.ceil(max((n.start+n.duration for n in project.notes),default=0)/measure)
    bs=getattr(project,'bar_start_seconds',None)
    if bs is not None and len(bs)>=2:
        clip_extra=math.ceil(max(0.0,project.clip_duration-bs[-1])*project.bpm/60/measure) if project.source else 0
        count=max(note_count,(len(bs)-1)+clip_extra)
    elif project.source:
        count=max(note_count,math.ceil((project.clip_duration-project.beat_origin)*project.bpm/60/measure))
    else:
        count=note_count
    if count>1000:
        raise ValueError('当前编配最多支持 1000 小节')
    degrees=[(0,True),(3,False),(5,True),(7,False),(8,False),(10,False)] if minor else [(0,False),(2,True),(4,True),(5,False),(7,False),(9,True)]
    diatonic_names=[chord_name(root+d,m) for d,m in degrees]
    name_to_deg={n:i for i,n in enumerate(diatonic_names)}
    from subpkgs.guitar_reharm.chord_voicing import EXTENDED_INTERVALS, parse_extended_chord
    root_to_deg={(root+d)%12:i for i,(d,_) in enumerate(degrees)}
    q_ivs={False:(0,4,7),True:(0,3,7)}
    ext_quals=[]
    seen_q_pcs={(0,4,7),(0,3,7)}
    for q_k,ivs_k in EXTENDED_INTERVALS.items():
        norm_pcs=tuple(sorted({iv%12 for iv in ivs_k}))
        if len(norm_pcs)<3 or norm_pcs in seen_q_pcs:
            continue
        seen_q_pcs.add(norm_pcs)
        q_ivs[q_k]=ivs_k
        ext_quals.append(q_k)
    for bar_i in range(count):
        b_pcs={n.pitch%12 for n in project.notes if min(n.start+n.duration,(bar_i+1)*measure)-max(n.start,bar_i*measure)>0}
        if len(b_pcs)<3:continue
        for r_c in range(12):
            if r_c not in b_pcs:continue
            rel=tuple(sorted((p-r_c)%12 for p in b_pcs))
            if rel in seen_q_pcs:continue
            rel_s=set(rel)
            if 4 in rel_s:base_q='7' if 10 in rel_s else ('maj7' if 11 in rel_s else '')
            elif 3 in rel_s:base_q='m7b5' if (10 in rel_s and 6 in rel_s and 7 not in rel_s) else ('m7' if 10 in rel_s else ('mmaj7' if 11 in rel_s else ('dim7' if (6 in rel_s and 7 not in rel_s and 9 in rel_s) else ('dim' if (6 in rel_s and 7 not in rel_s) else 'm'))))
            elif 5 in rel_s:base_q='7sus4' if 10 in rel_s else 'sus4'
            elif 2 in rel_s:base_q='7sus2' if 10 in rel_s else 'sus2'
            else:continue
            bpcs={x%12 for x in EXTENDED_INTERVALS[base_q]}
            mods=[]
            if 7 in bpcs and 7 not in rel_s:
                if 6 in rel_s and 6 not in bpcs:mods.append('b5');bpcs.discard(7);bpcs.add(6)
                elif 8 in rel_s and 8 not in bpcs:mods.append('#5');bpcs.discard(7);bpcs.add(8)
                else:mods.append('no5');bpcs.discard(7)
            has_7=(10 in rel_s or 11 in rel_s)
            for p_rel in sorted(rel_s-bpcs):
                if p_rel==1:mods.append('b9' if has_7 else 'addb9')
                elif p_rel==2:mods.append('add9')
                elif p_rel==3 and 4 in rel_s:mods.append('#9' if has_7 else 'add#9')
                elif p_rel==5:mods.append('add11')
                elif p_rel==6:mods.append('#11' if has_7 else 'add#11')
                elif p_rel==8:mods.append('b13' if has_7 else 'addb13')
                elif p_rel==9:mods.append('add13' if has_7 else '6')
                elif p_rel==11:mods.append('add#7')
            q_try=base_q+''.join(mods)
            p_res=parse_extended_chord('C'+q_try)
            if p_res is not None and tuple(sorted({x%12 for x in p_res['intervals']}))==rel:
                q_can=p_res['quality']
                if q_can not in q_ivs:
                    seen_q_pcs.add(rel);q_ivs[q_can]=p_res['intervals'];ext_quals.append(q_can)
    all_names=list(dict.fromkeys(diatonic_names+[chord_name((root+off)%12,m) for m in (False,True,*ext_quals) for off in range(12)]))
    trans_mat=_GUITAR_MIN_TRANS if minor else _GUITAR_MAJ_TRANS
    rows=[]
    for bar in range(count):
        weights=[0.0]*12
        for n in project.notes:
            overlap=max(0,min(n.start+n.duration,(bar+1)*measure)-max(n.start,bar*measure))
            if overlap<=0:continue
            rel_onset=(n.start-bar*measure)%measure
            weights[n.pitch%12]+=_gttm_salience(rel_onset,overlap,measure,project.meter)
        tot_w=sum(weights)
        if not tot_w:
            rows.append({'N.C.':0});continue
        emission={}
        bar_tiv=_tiv_vec(weights)
        for name in all_names:
            r,m=parse_chord(name); ivs=q_ivs[m]; tones={(r+iv)%12 for iv in ivs}
            is_dia=name in name_to_deg
            distinct=sum(1 for t in tones if weights[t]>0.05*tot_w)
            cov=sum(weights[t] for t in tones)/tot_w
            min_req=len(tones) if isinstance(m,str) else 2
            if not is_dia and (cov<(0.92 if len(tones)>=4 else 0.85) or distinct<min_req):
                continue
            missing=sum(1 for t in tones if weights[t]<=0.05*tot_w)
            prior=(0.0 if is_dia else (-0.04 - 0.02*max(0,len(tones)-3))) - 0.03*missing
            coverage_score=sum(w*(1 if pc in tones else -.45) for pc,w in enumerate(weights))/tot_w
            chord_tiv=_tiv_vec([1.0 if pc in tones else 0.0 for pc in range(12)])
            emission[name]=.8*coverage_score+.2*_cosine(bar_tiv,chord_tiv)+prior
        rows.append(emission)
    states={None:(0,[])}
    for row in rows:
        updated={}
        for name,fit in row.items():
            choices=[]
            for previous,(cost,path) in states.items():
                transition=0
                if previous and previous!='N.C.' and name!='N.C.':
                    pi=name_to_deg.get(previous,root_to_deg.get(parse_chord(previous)[0],-1))
                    ci=name_to_deg.get(name,root_to_deg.get(parse_chord(name)[0],-1))
                    if pi>=0 and ci>=0:
                        transition=trans_mat[pi][ci]
                    else:
                        transition=-0.05
                choices.append((cost+fit+transition,path+[name]))
            updated[name]=max(choices,key=lambda x:x[0])
        states=updated
    return max(states.values(),key=lambda x:x[0])[1] if rows else []


def arrange(project,chords,style='分解和弦',capo=0):
    project.validate()
    _arpeggio_styles=('分解和弦','抒情琶音','arpeggio','ballad')
    _groove_styles=('指弹律动','Neo-Soul 律动','fingerstyle','bossa','neosoul','groove')
    _strum_styles=('流行扫弦','扫弦伴奏','strum','strumming','pop')
    if (not isinstance(style,str) or style not in (*_arpeggio_styles,*_groove_styles,*_strum_styles)
            or isinstance(capo,bool) or not isinstance(capo,int) or not 0<=capo<=7):
        raise ValueError('编配风格或变调夹位置无效')
    measure=int(project.meter.split('/')[0])*4/int(project.meter.split('/')[1])
    note_count=math.ceil(max((n.start+n.duration for n in project.notes),default=0)/measure)
    bs=getattr(project,'bar_start_seconds',None)
    if bs is not None and len(bs)>=2:
        clip_extra=math.ceil(max(0.0,project.clip_duration-bs[-1])*project.bpm/60/measure)
        min_count=max(note_count,(len(bs)-1)+(clip_extra if project.source else 0))
        max_count=max(min_count,(len(bs)-1)+clip_extra)
    elif project.source:
        min_count=max_count=max(note_count,math.ceil((project.clip_duration-project.beat_origin)*project.bpm/60/measure))
    else:
        min_count=max_count=note_count
    if not (min_count<=len(chords)<=max_count) or not 0<len(chords)<=1000:
        raise ValueError(f'需要按顺序填写 {min_count} 个和弦，每小节一个')
    bars=[];events=[]
    for bar,name in enumerate(chords):
        frets=voicing(name,capo)
        active=[i for i,f in enumerate(frets) if f>=0]
        slots=[]
        n_steps=max(1,round(measure*2))
        step_beats=measure/n_steps
        for step in range(n_steps):
            if not active:
                slots.append([]);continue
            if style in _arpeggio_styles:
                order=[active[0],active[-3],active[-2],active[-1],active[-2],active[-3],active[-2],active[-1]]
                strings=[order[step%len(order)]]
                direction=''
            elif style in _groove_styles:
                if step==0:
                    strings=[active[0],active[-1]] if len(active)>=2 else [active[0]]
                elif n_steps>=4 and step==n_steps//2:
                    strings=[active[min(1,len(active)-1)] if len(active)>=4 else active[0]]
                elif step%2==1:
                    strings=sorted(set(active[-3:-1] or [active[-1]]))
                else:
                    strings=[active[min(2,len(active)-1)]]
                direction=''
            else:
                # A sparse 4/4 pattern; other meters use alternating eighths.
                if measure==4 and step in (1,4):
                    slots.append([]);continue
                direction='↓' if step%2==0 else '↑'
                strings=active[:] if direction=='↓' else list(reversed(active[-4:]))
            slot=[]
            strum_step=min(.012*project.bpm/60,step_beats*.5/max(1,len(strings))) if style in _strum_styles else 0.0
            for order_index,string in enumerate(strings):
                at=bar*measure+step*step_beats+order_index*strum_step
                # Short plucks make the displayed eighth-note timing unambiguous.
                duration=max(1e-4,min(step_beats*.96,bar*measure+min(measure,(step+1)*step_beats)-at))
                event={'start':at,'duration':duration,'pitch':TUNING[string]+frets[string]+capo,
                       'string':string,'fret':frets[string],'velocity':82 if step==0 else 67,'bar':bar}
                events.append(event);slot.append({'string':string,'fret':frets[string],'direction':direction})
            slots.append(slot)
        bars.append({'chord':name,'frets':frets,'slots':slots})
    from dataclasses import asdict
    return {'title':project.title,'bpm':project.bpm,'meter':project.meter,'capo':capo,'style':style,'bars':bars,'events':events,
            'measure':measure,'tonic':project.tonic,'beat_origin':project.beat_origin,
            'bar_start_seconds':([float(x) for x in bs]+[float(bs[-1])+(k+1)*measure*60/project.bpm for k in range(max(0,len(chords)-(len(bs)-1)))]) if (bs is not None and len(bs)>=2) else None,
            'melody':[asdict(n) for n in project.notes],'lyrics':copy.deepcopy(project.lyrics),
            'warning':'旋律条件下的编配草稿，不是原曲和弦识别。每小节一个和弦；请确认调性、拍点和和弦。'}


def midi(project,arrangement):
    ticks=480
    tracks=[]
    numerator,denominator=map(int,project.meter.split('/'))
    measure=numerator*4/denominator
    spb=60/project.bpm
    raw_bs=arrangement.get('bar_start_seconds') if isinstance(arrangement,dict) else None
    if not raw_bs:
        raw_bs=getattr(project,'bar_start_seconds',None)
    bs=[float(x) for x in raw_bs] if (raw_bs is not None and len(raw_bs)>=2) else None
    eff_origin=float(project.beat_origin)
    lead_sec=max(0.0,bs[0]) if bs else max(0.0,eff_origin)
    def _sec_to_beat(sec):
        if not bs:
            return (sec-eff_origin)/spb
        if sec<=bs[0]:
            return (sec-bs[0])/spb
        for k in range(len(bs)-1):
            if sec<=bs[k+1]:
                return (k+(sec-bs[k])/max(1e-6,bs[k+1]-bs[k]))*measure
        return (len(bs)-1)*measure+(sec-bs[-1])/spb
    beat_shift=(-_sec_to_beat(0.0)) if (bs and bs[0]<0) else ((eff_origin/spb) if (not bs and eff_origin<0) else 0.0)
    end_beats=max([n.start+n.duration for n in project.notes]+[n['start']+n['duration'] for n in arrangement['events']]+[len(arrangement.get('bars',[]))*measure],default=0.0)
    if project.source:
        end_beats=max(end_beats,max(0.0,_sec_to_beat(project.clip_duration)))
    if bs:
        for b_idx in range(len(bs)-1):
            if round((bs[b_idx+1]-bs[b_idx])*1_000_000/measure)>16_777_215:
                raise ValueError('小节速度超出有效范围')
    header=[]
    if lead_sec>0:
        lead_bars=max(1,math.ceil(lead_sec/(16.0*measure)))
        lead_beats=lead_bars*measure
        lead_us=max(1,min(16_777_215,round(lead_sec*1_000_000/lead_beats)))
        header.append((0,0,b'\xff\x51\x03'+lead_us.to_bytes(3,'big')))
        tick_offset=round(lead_beats*ticks)
    else:
        tick_offset=0
    if bs:
        for b_idx in range(len(bs)-1):
            if (b_idx+1)*measure+beat_shift<=1e-6:
                continue
            bar_sec=max(1e-4,bs[b_idx+1]-bs[b_idx])
            us_per_beat=max(1,min(16_777_215,round(bar_sec*1_000_000/measure)))
            t_pos=tick_offset+round(max(0.0,b_idx*measure+beat_shift)*ticks)
            header.append((t_pos,0,b'\xff\x51\x03'+us_per_beat.to_bytes(3,'big')))
        if end_beats>(len(bs)-1)*measure:
            nom_us=max(1,min(16_777_215,round(60_000_000/project.bpm)))
            t_pos=tick_offset+round(max(0.0,(len(bs)-1)*measure+beat_shift)*ticks)
            header.append((t_pos,0,b'\xff\x51\x03'+nom_us.to_bytes(3,'big')))
    else:
        nom_us=max(1,min(16_777_215,round(60_000_000/project.bpm)))
        header.append((tick_offset,0,b'\xff\x51\x03'+nom_us.to_bytes(3,'big')))
    header.append((0,0,b'\xff\x58\x04'+bytes([numerator,int(math.log2(denominator)),24,8])))
    end_tick=tick_offset+round(max(0.0,end_beats+beat_shift)*ticks)
    for channel,program,notes in [(0,0,[{'start':n.start,'duration':n.duration,'pitch':n.pitch,'velocity':83} for n in project.notes]),(1,24,arrangement['events'])]:
        events=(header[:] if channel==0 else [])+[(0,0,bytes([0xc0+channel,program]))]
        for n in notes:
            s_b=n['start']+beat_shift;e_b=n['start']+n['duration']+beat_shift
            if e_b<=1e-6:
                continue
            s_b=max(0.0,s_b)
            a=tick_offset+round(s_b*ticks);b=max(a+1,tick_offset+round(e_b*ticks))
            events.extend([(a,2,bytes([0x90+channel,n['pitch'],n['velocity']])),(b,1,bytes([0x80+channel,n['pitch'],0]))])
        data=bytearray();previous=0
        for tick,_,event in sorted(events,key=lambda e:(e[0],e[1])):
            data.extend(vlq(tick-previous));data.extend(event);previous=tick
        data.extend(vlq(max(0,end_tick-previous))+b'\xff\x2f\x00')
        tracks.append(b'MTrk'+struct.pack('>I',len(data))+data)
    return b'MThd'+struct.pack('>IHHH',6,1,2,ticks)+b''.join(tracks)


def render_audio(project,arrangement,path,with_melody=True):
    import numpy as np
    sr=22050;spb=60/project.bpm
    measure=float(arrangement.get('measure',int(project.meter.split('/')[0])*4/int(project.meter.split('/')[1])))
    raw_bs=arrangement.get('bar_start_seconds') if isinstance(arrangement,dict) else None
    if not raw_bs:
        raw_bs=getattr(project,'bar_start_seconds',None)
    bs=[float(x) for x in raw_bs] if (raw_bs is not None and len(raw_bs)>=2) else None
    eff_origin=float(project.beat_origin)
    def _beat_to_sec(b_pos):
        if not bs:
            return eff_origin+b_pos*spb
        n_bars=len(bs)-1
        if b_pos<=0:
            return bs[0]+b_pos*spb
        if b_pos<=n_bars*measure:
            b_idx=min(n_bars-1,int(b_pos//measure))
            frac=(b_pos-b_idx*measure)/measure
            return bs[b_idx]+frac*(bs[b_idx+1]-bs[b_idx])
        return bs[-1]+(b_pos-n_bars*measure)*spb
    bar_end=len(arrangement.get('bars',[]))*measure
    end_beats=max([n.start+n.duration for n in project.notes]+[n['start']+n['duration'] for n in arrangement['events']]+[bar_end],default=0.0)
    end_sec=max(spb,_beat_to_sec(end_beats)+spb)
    if project.source:
        end_sec=max(end_sec,float(project.clip_duration))
    if end_sec>3600:
        raise ValueError('试听时长超过一小时')
    sound=np.zeros(math.ceil(end_sec*sr),dtype=np.float32)
    notes=[(e['pitch'],e['start'],e['duration'],e['velocity']/127,True) for e in arrangement['events']]
    if with_melody:
        notes += [(n.pitch,n.start,n.duration,.65,False) for n in project.notes]
    for pitch,start,duration,volume,guitar in notes:
        raw_s=_beat_to_sec(start);raw_e=_beat_to_sec(start+duration)
        if raw_e<=1e-6:
            continue
        start_sec=max(0.0,raw_s)
        dur_sec=max(1e-4,raw_e-start_sec)
        count=max(1,round(dur_sec*sr));t=np.arange(count)/sr
        frequency=440*2**((pitch-69)/12)
        signal=np.zeros(count)
        for harmonic in range(1,7 if guitar else 3):
            if frequency*harmonic>=sr/2:break
            signal+=np.sin(2*np.pi*frequency*harmonic*t)/harmonic**(1.4 if guitar else 2)
        signal*=np.exp(-t*(5 if guitar else 1))*volume*.12
        ramp=min(count//2,round(.008*sr))
        if ramp:
            signal[:ramp]*=np.linspace(0,1,ramp);signal[-ramp:]*=np.linspace(1,0,ramp)
        offset=round(start_sec*sr)
        if offset+count>len(sound):
            sound=np.pad(sound,(0,offset+count-len(sound)))
        sound[offset:offset+count]+=signal.astype(np.float32)
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
