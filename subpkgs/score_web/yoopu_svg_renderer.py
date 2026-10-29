"""Chinese Publication-Grade ('Yoopu / 有谱么' Style) Combined Guitar Tab + Jianpu + Syllabic Lyric SVG/HTML Renderer."""
from __future__ import annotations
from html import escape
from music import NAMES, DEGREES
from ._score_timing import build_score_timeline, duration_parts, TICKS


def _svg_text(x, y, text, size=16, **attributes):
    attrs = ' '.join(f'{escape(k.replace("_", "-"), quote=True)}="{escape(str(v), quote=True)}"' for k, v in attributes.items())
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="middle" {attrs}>{escape(str(text), quote=True)}</text>'


def _line(x1, y1, x2, y2, width=1):
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="black" stroke-width="{width}"/>'


def _chord_diagram(x, segment):
    frets = segment['frets']
    if len(frets) != 6:
        raise ValueError('A chord diagram requires six frets')
    positive = [f for f in frets if f > 0]
    first = min(positive) if positive and max(positive) > 5 else 1
    rows = max(4, max(positive, default=first) - first + 1)
    gap = 54 / rows
    result = []
    for i in range(6):
        result.append(_line(x-25+i*10, 60, x-25+i*10, 114))
    for i in range(rows+1):
        result.append(_line(x-25, 60+i*gap, x+25, 60+i*gap, 2 if i == 0 and first == 1 else 1))
    if first > 1:
        result.append(_svg_text(x-42, 72, f'{first}fr', 11))
    for i, fret in enumerate(frets):
        xx = x-25+i*10
        if fret <= 0:
            result.append(_svg_text(xx, 54, '×' if fret < 0 else '○', 12))
        else:
            yy = 60 + (fret-first+.5)*gap
            result.append(f'<circle cx="{xx:g}" cy="{yy:g}" r="3.4"/>')
    for fret in sorted(set(positive)):
        indices = [i for i, f in enumerate(frets) if f == fret]
        if len(indices) > 1 and all(f >= fret for f in frets[indices[0]:indices[-1]+1]):
            yy = 60+(fret-first+.5)*gap
            result.append(_line(x-25+indices[0]*10, yy, x-25+indices[-1]*10, yy, 5))
    return ''.join(result)


def _render_tab_bar(arrangement, index, x_at, width, left):
    result = []
    for string, label in enumerate('EBGDAE', 1):
        y = 155+(string-1)*15
        result += [_line(left+20, y, left+width, y), _svg_text(left+10, y+4, label, 12)]
    result += [_line(left+width, 155, left+width, 230)]
    for event in arrangement['events']:
        relative = event['start'] - index*arrangement['measure_beats']
        if not 0 <= relative < arrangement['measure_beats']:
            continue
        x, y = x_at(relative*TICKS), 155+(event['string']-1)*15
        technique = event.get('technique', '')
        label = 'X' if technique == 'X' else str(event['fret'])
        result += [f'<rect x="{x-9:g}" y="{y-10:g}" width="18" height="16" fill="white"/>',
                   _svg_text(x, y+5, label, 15)]
        if technique == 'PM':
            result.append(_svg_text(x, 143, 'PM', 11))
        elif technique in ('H', 'P', 'CM'):
            result.append(_svg_text(x+14, y-5, technique, 10))
    return ''.join(result)


def _render_jianpu_bar(project, spans, lyrics, x_at, lyric_y, baseline):
    result = []
    shift = baseline-272
    for span in spans:
        tick = span['start_tick']
        for i, part in enumerate(duration_parts(span['duration_ticks'])):
            x = x_at(tick)
            pitch, octave, degree = span['pitch'], 0, '0'
            continuation = span['continuation'] or i > 0
            if pitch is not None:
                octave, semitone = divmod(pitch-(60+NAMES.index(project.tonic)), 12)
                degree = DEGREES[semitone]
            digit = '⌒' if continuation and pitch is not None else degree[-1]
            result.append(_svg_text(x, baseline, digit, 22, **{'data-note-index': span['note_index']}))
            if len(degree) > 1 and not continuation:
                result.append(_svg_text(x-13, 265+shift, degree[:-1], 13))
            for dot in range(abs(octave)):
                yy = 249+shift-dot*7 if octave > 0 else 292+shift+dot*7
                result.append(f'<circle cx="{x:g}" cy="{yy:g}" r="1.8"/>')
            beams = max(0, part['denominator'].bit_length()-3)
            for beam in range(beams):
                result.append(_line(x-7, 278+shift+beam*4, x+7, 278+shift+beam*4, 1.5))
            for dot in range(part['dots']):
                result.append(f'<circle cx="{x+12+dot*5:g}" cy="{266+shift:g}" r="1.8"/>')
            if part['tuplet']:
                result.append(_svg_text(x, 238+shift, '3', 11))
            for offset in range(TICKS, part['ticks'], TICKS):
                result.append(_svg_text(x_at(tick+offset), baseline, '—', 18))
            if pitch is not None and not continuation and lyrics:
                result.append(_svg_text(x, lyric_y, lyrics[span['note_index']], 16,
                                        **{'data-lyric-index': span['note_index']}))
            tick += part['ticks']
    return ''.join(result)


def render_publication_score_html(
    project,
    arrangement: dict,
    note_lyrics: list[str] | None = None,
    bar_analysis: list[dict] | None = None,
) -> str:
    """Render standard Chinese Guitar Songbook combined systems (supports Split-Bar dual chord diagrams, X/PM/HP/CM tab markers, stacked Jianpu, and vertically aligned lyrics)."""
    if note_lyrics and len(note_lyrics) != len(project.notes):
        raise ValueError('Lyrics must contain one entry per source note')
    timeline = build_score_timeline(project, arrangement)
    measure = timeline['measure_ticks']
    analysis = {} if bar_analysis is None else {row['bar']: row for row in bar_analysis}
    widths, paddings = [], []
    for spans in timeline['bars']:
        width, anchors = 380.0, []
        for span in spans:
            lyric = note_lyrics[span['note_index']] if note_lyrics and span['pitch'] is not None and not span['continuation'] else ''
            estimated = sum(16 if ord(c) > 255 else 9 for c in lyric)
            anchors.append((span['start_tick'], max(22, estimated)))
        padding = max(36, max((w for _, w in anchors), default=22)/2+10)
        for (a, wa), (b, wb) in zip(anchors, anchors[1:]):
            if b > a:
                width = max(width, ((wa+wb)/2+14)*measure/(b-a)+2*padding)
        if anchors:
            width = max(width, anchors[0][1]+80, anchors[-1][1]+80)
        widths.append(width)
        paddings.append(padding)
    max_system_width = max((48+sum(widths[i:i+3]) for i in range(0, len(widths), 3)), default=1)
    systems = []
    for first in range(0, len(timeline['bars']), 3):
        end = min(first+3, len(timeline['bars']))
        low = max((max(0, -((s['pitch']-(60+NAMES.index(project.tonic)))//12))
                   for spans in timeline['bars'][first:end] for s in spans if s['pitch'] is not None), default=0)
        high = max((max(0, (s['pitch']-(60+NAMES.index(project.tonic)))//12)
                    for spans in timeline['bars'][first:end] for s in spans if s['pitch'] is not None), default=0)
        baseline = max(272, 250+high*7)
        lyric_y = max(baseline+50, baseline+31+low*7)
        height = lyric_y+25
        total = 48+sum(widths[first:end])
        content, left = [], 28.0
        for index in range(first, end):
            width = widths[index]
            spans = timeline['bars'][index]
            padding = paddings[index]
            def x_at(tick):
                return left+padding+tick/measure*(width-2*padding)
            content.append(f'<g role="group" aria-label="Bar {index+1}">')
            content.append(_svg_text(left+8, 20, str(index+1), 11, fill='#666'))
            if index < len(arrangement['bars']):
                bar = arrangement['bars'][index]
                segments = bar['split_segments'] if 'split_segments' in bar else [bar]
                for part, supplied in enumerate(segments):
                    segment = dict(supplied)
                    if not segment.get('roman') and index in analysis:
                        source = analysis[index]['split_segments']
                        segment['roman'] = source[part]['roman']
                    center = left+(part+.5)*width/len(segments)
                    content += [_svg_text(center, 20, segment['chord'], 18),
                                _svg_text(center, 39, segment.get('roman', ''), 13, fill='#555'),
                                _chord_diagram(center, segment)]
            content += [_render_tab_bar(arrangement, index, x_at, width, left),
                        _render_jianpu_bar(project, spans, note_lyrics, x_at, lyric_y, baseline), '</g>']
            left += width
        systems.append(f'<svg xmlns="http://www.w3.org/2000/svg" class="score-system" role="img" aria-label="Score bars {first+1}–{end}" viewBox="0 0 {total:g} {height:g}" width="100%" style="width:{total/max_system_width*100:g}%">'+''.join(content)+'</svg>')
    title = escape(str(project.title), quote=True)
    metadata = escape(f'1={project.tonic}　{project.meter}　♩={project.bpm:g}　Capo {arrangement["capo"]}', quote=True)
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>'+title+
            '</title><style>body{background:white;color:black;margin:28px;font-family:"Noto Serif CJK SC","SimSun",serif}'
            'h1{text-align:center;font-size:26px}.metadata{text-align:center}.score-system{display:block;margin:28px 0;overflow:visible}'
            '@media print{body{margin:0}.score-system{break-inside:avoid}h1{margin-top:0}}</style></head><body><h1>'+title+
            '</h1><p class="metadata">'+metadata+'</p>'+(''.join(systems) if systems else '<p>暂无乐谱 · Empty score</p>')+'</body></html>')
