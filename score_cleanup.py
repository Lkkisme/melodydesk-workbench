"""Phrase-level rhythm inference; never changes pitch or source timing.

Small dynamic program balances onset fit, interval fit and simple durations.
The user supplies tempo/beat origin. This is not key or harmony inference.
"""
import copy
import math


def _phrase(notes, bpm, origin, division):
    unit = 1 / division
    starts = [(n.raw_start-origin)*bpm/60 for n in notes]
    lengths = [n.raw_duration*bpm/60 for n in notes]
    candidates = [sorted({max(0, k*unit) for k in range(round(t/unit)-2, round(t/unit)+3)}) for t in starts]
    states = {s: ((s-starts[0])**2, [s]) for s in candidates[0]}
    common = [unit,2*unit,3*unit,1,1.5,2,3,4,6,8]
    for i in range(1,len(notes)):
        next_states = {}
        for s in candidates[i]:
            options=[]
            for previous,(cost,path) in states.items():
                interval=s-previous
                if interval < unit-1e-8:
                    continue
                observed=starts[i]-starts[i-1]
                complexity=min(abs(interval-d) for d in common)
                options.append((cost+(s-starts[i])**2+.5*(interval-observed)**2+.05*complexity,path+[s]))
            if options:
                next_states[s]=min(options,key=lambda x:x[0])
        if not next_states:
            raise ValueError('音符过密，无法在所选节奏网格中保留全部音符；请先合并误切音符，或选择更细节奏。')
        states=next_states
    cost,path=min(states.values(),key=lambda x:x[0])
    result=copy.deepcopy(notes)
    for i,n in enumerate(result):
        n.start=path[i]
        limit=path[i+1]-path[i] if i+1<len(path) else max(8,math.ceil(lengths[i])+1)
        durations=sorted({round(k*unit,10) for k in range(1,max(2,math.ceil(limit/unit)+1)) if k*unit<=limit+1e-8})
        gap=(starts[i+1]-starts[i]-lengths[i]) if i+1<len(path) else 1
        # Small articulation gaps belong inside a notated note; real rests do not.
        target=limit if i+1<len(path) and gap<.22 else lengths[i]
        n.duration=min(durations,key=lambda d:(d-target)**2+.025*(0 if any(abs(d-c)<1e-6 for c in common) else 1))
        cost+=(n.duration-lengths[i])**2*.15
    return result,cost/len(notes)


def organize(notes,bpm,origin=0,mode='auto'):
    if not notes:
        return []
    if mode not in ('auto','binary','triplet'):
        raise ValueError('未知节奏类型')
    ordered=sorted(copy.deepcopy(notes),key=lambda n:n.raw_start)
    phrases=[]
    for n in ordered:
        if not phrases or n.raw_start-(phrases[-1][-1].raw_start+phrases[-1][-1].raw_duration)>.45:
            phrases.append([])
        phrases[-1].append(n)
    result=[]
    for phrase in phrases:
        options=[]
        last_exc=None
        for division in ([2,3,4,6] if mode=='auto' else [2,4] if mode=='binary' else [3,6]):
            try:
                arranged,cost=_phrase(phrase,bpm,origin,division)
                # Prefer simple binary rhythm unless finer/triplet fit is better.
                options.append((cost+{2:0,4:.006,3:.009,6:.015}[division],arranged))
            except ValueError as exc:
                # Candidate grid division is too coarse for this phrase density; try finer divisions.
                last_exc=exc
        if not options:
            raise ValueError('此乐句音符太密，请先合并误切音符再整理。') from last_exc
        result.extend(min(options,key=lambda p:p[0])[1])
    for a,b in zip(result,result[1:]):
        if a.start+a.duration>b.start+1e-6:
            raise ValueError('整理后乐句边界冲突，请调整拍点或缩小选择范围。')
    return result
