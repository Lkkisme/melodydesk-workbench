"""Advanced Guitar Accompaniment & Fingerstyle Chord-Melody Arrangement Engine (Supports Split-Bar Chords, X Slap, P.M. Palm Mute, H/P Licks, Passing Bass, and Intro/Interlude Chord-Melody)."""
from __future__ import annotations

ADVANCED_STYLES = (
    "分解和弦",
    "扫弦伴奏",
    "指弹律动",
    "Neo-Soul 律动",
    "抒情琶音",
)


import math
from .chord_voicing import (STANDARD_TUNING, _integer, _make_voicing, generate_voicings,
                            _optimize_candidates, parse_extended_chord, _note_pc)
from .reharmonizer import _measure_beats, _overlapping_notes, roman_numeral


def arrange_advanced(
    project,
    chords: list[str] | None = None,
    style: str = "指弹律动",
    capo: int | None = None,
    custom_voicings: list[list[int] | None] | None = None,
    split_chords: list[list[dict]] | None = None,
    passing_bass: bool = True,
    slap_x: bool = False,
    palm_mute: bool = False,
    embellish_hp: bool = False,
    intro_mode: str = "chord_melody",
    instrumental_melody: dict[int, list[dict]] | None = None,
    bar_dynamics: list[float] | None = None,
) -> dict:
    """Generate rich 6-string guitar tab events with selectable Split-Bar harmony, X/PM/HP/PB articulations, and Intro/Interlude Chord-Melody soloing."""
    if style not in ADVANCED_STYLES:
        raise ValueError(f'Unknown style: {style!r}')
    if intro_mode not in {'none', 'nc', 'groove', 'chord_melody'}:
        raise ValueError(f'Unknown intro mode: {intro_mode!r}')
    capo = 0 if capo is None else capo
    _integer(capo,0,12,'capo')
    measure = _measure_beats(project)
    if not math.isfinite(project.bpm) or project.bpm <= 0:
        raise ValueError('BPM must be positive and finite')
    if chords is None:
        end = max((n.start+n.duration for n in project.notes),default=0.0)
        if getattr(project, 'source', None):
            end = max(end, getattr(project, 'clip_duration', 0.0)*project.bpm/60)
        chords = [project.tonic]*math.ceil(end/measure)
    else:
        chords = list(chords)
    count = len(chords)
    for name, supplied in [('custom_voicings',custom_voicings),('split_chords',split_chords)]:
        if supplied is not None and len(supplied) != count:
            raise ValueError(f'{name} must match bar count')
    bars, layers, flat = [], [], []
    for index, chord in enumerate(chords):
        # Validate all custom input independently of segment precedence and rests.
        custom = None
        if custom_voicings is not None and custom_voicings[index] is not None:
            custom = _make_voicing(chord,custom_voicings[index],capo,chord)
        supplied = ([dict(beat=0.0,duration=measure,chord=chord)] if split_chords is None
                    else [dict(s) for s in split_chords[index]])
        segments = sorted(supplied,key=lambda s:s['beat'])
        end = 0.0
        for j, segment in enumerate(segments):
            beat, duration = segment['beat'], segment['duration']
            if (not math.isfinite(beat) or not math.isfinite(duration) or duration <= 0
                    or abs(beat-end) > 1e-9):
                raise ValueError(f'Invalid segment coverage in bar {index}')
            end = beat+duration
            symbol = segment['chord']
            fixed = None
            if 'frets' in segment:
                provenance = segment['capo'] if 'capo' in segment else capo
                _integer(provenance,0,12,'segment capo')
                supplied_voicing = _make_voicing(symbol,segment['frets'],provenance,symbol)
                if provenance == capo:
                    fixed = supplied_voicing
            elif j == 0 and custom is not None:
                fixed = _make_voicing(symbol,custom['frets'],capo,symbol)
            candidates = [fixed] if fixed is not None else generate_voicings(symbol,capo)
            if not candidates:
                raise ValueError(f'No voicing for {symbol!r} in bar {index}')
            segment.update(beat=float(beat),duration=float(duration),roman=roman_numeral(symbol,project.tonic),capo=capo)
            flat.append(segment)
            layers.append(candidates)
        if segments and abs(end-measure) > 1e-9:
            raise ValueError(f'Incomplete segment coverage in bar {index}')
        bars.append(dict(bar=index,chord=segments[0]['chord'] if segments else 'N.C.',
                         roman=segments[0]['roman'] if segments else 'N.C.',frets=[-1]*6,split_segments=segments))
    for segment, voicing in zip(flat,_optimize_candidates(layers)):
        segment['frets'] = list(voicing['frets'])
        segment['base_fret'] = voicing['base_fret']
    for bar in bars:
        if bar['split_segments']:
            bar['frets'] = list(bar['split_segments'][0]['frets'])
        if bar_dynamics is not None and bar['bar'] < len(bar_dynamics):
            dyn = max(0.55, min(1.15, float(bar_dynamics[bar['bar']])))
            bar['dynamic_scale'] = round(dyn, 3)

    _GENTLE_STYLES = {'民谣扫弦', '流行扫弦', '拍弦律动', '扫弦伴奏', '指弹律动'}

    events = {}
    slaps = []
    def active(bar, beat):
        return next((s for s in bar['split_segments'] if s['beat'] <= beat < s['beat']+s['duration']),None)

    def strings(segment):
        return [(6-i,f,STANDARD_TUNING[i]+capo+f) for i,f in enumerate(segment['frets']) if f >= 0]

    def bass_of(segment):
        candidates = [x for x in strings(segment) if x[0] >= 4]
        return min(candidates,key=lambda x:x[2]) if candidates else None

    def insert(bar, beat, duration, played, velocity, technique='', replace=False):
        segment = active(bar,beat)
        if segment is None:
            return
        string, fret, pitch = played
        duration = min(duration,segment['beat']+segment['duration']-beat,measure-beat)
        if duration <= 0:
            return
        dyn_scale = bar.get('dynamic_scale')
        if dyn_scale is not None:
            velocity = max(1, min(127, round(velocity * dyn_scale)))
        start = bar['bar']*measure+beat
        event = dict(bar=bar['bar'],beat=beat,start=start,duration=duration,string=string,fret=fret,
                     pitch=pitch,velocity=velocity,technique=technique,chord=segment['chord'],
                     segment=bar['split_segments'].index(segment))
        key = (start,string)
        if replace or key not in events:
            events[key] = event

    tonic_pc = _note_pc(project.tonic)
    pentatonic = {(tonic_pc+x)%12 for x in (0,2,4,7,9)}
    diatonic = {(tonic_pc+x)%12 for x in (0,2,4,5,7,9,11)}
    _FINGERPICK_STYLES = {'分解和弦', '抒情琶音', '指弹律动'}
    _PHRASE_VEL_CURVE = [0.97, 1.00, 1.04, 0.98]

    for bar in bars:
        index = bar['bar']
        bar_index = index
        silent = not _overlapping_notes(project.notes,index*measure,(index+1)*measure)
        if intro_mode == 'nc' and silent:
            continue
        cm = intro_mode == 'chord_melody' and silent
        bar_dyn = bar.get('dynamic_scale')
        use_gentle = bar_dyn is not None and bar_dyn < 0.68 and style in _GENTLE_STYLES
        effective_style_idx = 0 if use_gentle else ADVANCED_STYLES.index(style)
        phrase_vel_scale = _PHRASE_VEL_CURVE[bar_index % 4]
        dyn_scale = bar.get('dynamic_scale')
        is_cadence_bar = (bar_index + 1) % 4 == 0
        is_fingerpick = style in _FINGERPICK_STYLES
        is_4_4 = abs(measure - 4.0) < 1e-9
        segments = bar['split_segments']
        grid = sorted(set([i*0.5 for i in range(math.ceil(measure*2)) if i*0.5 < measure]
                          +[s['beat'] for s in segments]))
        for beat in grid:
            segment = active(bar,beat)
            if segment is None:
                continue
            played = strings(segment)
            bass = bass_of(segment)
            treble = sorted((x for x in played if x[0] <= 3),key=lambda x:x[2])
            mid_treble = sorted((x for x in played if x[0] <= 4),key=lambda x:x[2]) or treble
            low = treble[0:1]
            middle = treble[min(1,len(treble)-1):min(1,len(treble)-1)+1] if treble else []
            high = treble[-1:]
            bass_list = [bass] if bass is not None else []
            slot = int(round(beat*2))%8
            use_alt_bass = False
            alt_bass_item = None
            if (is_fingerpick and is_4_4 and len(segments) == 1
                    and (is_cadence_bar or bar_index % 2 == 0)
                    and abs(beat - 2.0) < 1e-9 and bass is not None):
                bass_s = bass[0]
                for candidate in sorted(played, key=lambda x: -x[0]):
                    if candidate[0] in (4, 5) and candidate[1] >= 0 and candidate[0] != bass_s:
                        alt_bass_item = candidate
                        use_alt_bass = True
                        break
            patterns = (
                [bass_list,low,middle,low,high,low,middle,low],
                [played,[],mid_treble,treble,bass_list,treble,mid_treble,treble],
                [bass_list+high,low,middle,low,bass_list+middle,high,low,middle],
                [bass_list+treble,[],mid_treble,high,bass_list,treble,[],treble],
                [bass_list,low,middle,high,middle,low,middle,low],
            )
            slot_vels = (90, 45, 68, 44, 84, 46, 72, 44)
            selection = list(patterns[effective_style_idx][slot])
            if use_alt_bass and alt_bass_item is not None:
                if bass in selection:
                    selection = [alt_bass_item if (x == bass) else x for x in selection]
                elif is_cadence_bar and selection:
                    selection = [alt_bass_item]
            is_strum = len(selection) >= 3
            if any(abs(beat-s['beat']) < 1e-9 for s in segments) and bass is not None and bass not in selection:
                if not use_alt_bass:
                    selection.append(bass)
            for item in selection:
                if cm and item[0] <= 3:
                    continue
                is_bass = bass is not None and (item == bass or (use_alt_bass and item == alt_bass_item))
                technique = 'PM' if palm_mute and is_bass and any(abs(beat-b) < 1e-9 for b in (0,measure/2)) else ''
                velocity = (92 if slot == 0 else slot_vels[slot]) if is_strum else (92 if (is_bass and slot == 0) else 86 if is_bass else slot_vels[slot])
                velocity = max(1, min(127, round(velocity * phrase_vel_scale)))
                insert(bar,beat,0.5,item,velocity,technique)
        if cm:
            explicit = instrumental_melody is not None and index in instrumental_melody
            lead = list(instrumental_melody[index]) if explicit else []
            if not explicit:
                for cycle, beat in enumerate(range(math.ceil(measure))):
                    if beat >= measure:
                        continue
                    segment = active(bar,beat)
                    if segment is None or segment['chord'] == 'N.C.':
                        continue
                    pcs = set(parse_extended_chord(segment['chord'])['pcs'])
                    preferred = pcs & pentatonic or pcs
                    pitches = [p for p in range(60,77) if p%12 in preferred and any(0 <= p-t-capo <= 24-capo for t in STANDARD_TUNING[3:])]
                    if pitches:
                        route = pitches+pitches[-2:0:-1]
                        lead.append(dict(beat=float(beat),duration=1.0,pitch=route[cycle%len(route)]))
            for note in lead:
                beat = note['beat'] if 'beat' in note else note['beat_offset']
                duration, pitch = note['duration'], note['pitch']
                _integer(pitch,0,127,'melody pitch')
                if not math.isfinite(beat) or not math.isfinite(duration) or beat < 0 or beat >= measure or duration <= 0 or beat+duration > measure+1e-9:
                    raise ValueError(f'Invalid instrumental melody timing in bar {index}')
                segment = active(bar,beat)
                if segment is None:
                    raise ValueError('Instrumental melody falls in an accompaniment rest')
                choices = [(6-i,pitch-t-capo,pitch) for i,t in enumerate(STANDARD_TUNING) if i >= 3 and 0 <= pitch-t-capo <= 24-capo]
                if not choices:
                    raise ValueError(f'Unplayable treble melody pitch {pitch}')
                item = min(choices,key=lambda x:(abs(x[1]-segment['base_fret']),x[1],x[0]))
                insert(bar,beat,duration,item,88,'CM',True)
        tail = measure-0.5
        segment = active(bar,tail) if tail >= 0 else None
        ornament_candidates = []
        if embellish_hp and segment is not None:
            parsed = parse_extended_chord(segment['chord'])
            tones = set(parsed['pcs']) | {parsed['bass_pc']} if parsed else set()
            for string in (1,2,3):
                previous = [e for e in events.values() if e['bar'] == index and e['string'] == string and e['beat'] < tail]
                if not previous:
                    continue
                previous_event = max(previous,key=lambda e:e['beat'])
                for fret in range(max(0,previous_event['fret']-2),min(24-capo,previous_event['fret']+2)+1):
                    pitch = STANDARD_TUNING[6-string]+capo+fret
                    if fret != previous_event['fret'] and pitch%12 in tones:
                        ornament_candidates.append((abs(fret-previous_event['fret']),string,fret,pitch,previous_event['fret']))
            if ornament_candidates:
                _, string, fret, pitch, old = min(ornament_candidates)
                existing = events.get((index*measure+tail,string))
                if existing is None or existing['technique'] not in {'CM','H','P','PB'}:
                    insert(bar,tail,0.5,(string,fret,pitch),78,'H' if fret > old else 'P',True)
        if passing_bass and segment is not None and index+1 < len(bars):
            next_segment = active(bars[index+1],0)
            target = bass_of(next_segment) if next_segment is not None else None
            current = bass_of(segment)
            candidates = []
            if target is not None and current is not None:
                for string in (4,5,6):
                    if (index*measure+tail,string) in events:
                        continue
                    for fret in range(25-capo):
                        pitch = STANDARD_TUNING[6-string]+capo+fret
                        if abs(pitch-target[2]) in (1,2) and pitch%12 in diatonic:
                            candidates.append((abs(pitch-current[2]),0,fret,string,pitch))
            if candidates:
                _, _, fret, string, pitch = min(candidates)
                insert(bar,tail,0.5,(string,fret,pitch),88,'PB')
        if slap_x and bar['split_segments']:
            for beat in (1.0,3.0):
                segment = active(bar,beat) if beat < measure else None
                if segment is not None:
                    slap_vel = max(1, min(127, round(82 * phrase_vel_scale * (float(dyn_scale) if dyn_scale is not None else 1.0))))
                    slaps.append(dict(bar=index,beat=beat,start=index*measure+beat,duration=min(0.5,measure-beat),
                                      string=6,fret=0,pitch=40,velocity=slap_vel,technique='X',chord=segment['chord'],
                                      segment=bar['split_segments'].index(segment)))
    ordered = sorted(events.values(),key=lambda e:(e['start'],e['string'],e['technique']))
    previous_by_string = {}
    for event in ordered:
        previous = previous_by_string.get(event['string'])
        if previous is not None:
            previous['duration'] = min(previous['duration'],event['start']-previous['start'])
        previous_by_string[event['string']] = event
    return dict(title=project.title,tonic=project.tonic,meter=project.meter,bpm=project.bpm,
                beat_origin=project.beat_origin,style=style,capo=capo,bar_count=count,measure_beats=measure,
                chords=[b['chord'] for b in bars],bars=bars,
                events=sorted(ordered+slaps,key=lambda e:(e['start'],e['string'],e['technique'])))
