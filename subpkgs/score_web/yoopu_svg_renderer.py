"""Chinese Publication-Grade ('Yoopu / 有谱么' Style) Combined Guitar Tab + Jianpu + Syllabic Lyric SVG/HTML Renderer."""
from __future__ import annotations
from html import escape
from music import NAMES, DEGREES
from ._score_timing import build_score_timeline, duration_parts, TICKS


def _svg_text(x, y, text, size=16, **attributes):
    attrs = ' '.join(f'{escape(k.replace("_", "-"), quote=True)}="{escape(str(v), quote=True)}"' for k, v in attributes.items())
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="middle" {attrs}>{escape(str(text), quote=True)}</text>'


def _line(x1, y1, x2, y2, width=1, color="#0f172a"):
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width}"/>'


def _chord_diagram(x, y_top, segment):
    frets = segment['frets']
    if len(frets) != 6:
        raise ValueError('A chord diagram requires six frets')
    positive = [f for f in frets if f > 0]
    first = min(positive) if positive and max(positive) > 5 else 1
    rows = max(4, max(positive, default=first) - first + 1)
    gw, gh = 50, 52
    sx = x - gw / 2
    string_gap = gw / 5
    fret_gap = gh / rows
    result = []
    for i in range(6):
        lx = sx + i * string_gap
        result.append(_line(lx, y_top, lx, y_top + gh, 0.8, "#475569"))
    for i in range(rows + 1):
        fy = y_top + i * fret_gap
        w = 2.5 if i == 0 and first == 1 else 0.8
        c = "#0f172a" if i == 0 and first == 1 else "#475569"
        result.append(_line(sx, fy, sx + gw, fy, w, c))
    if first > 1:
        result.append(_svg_text(sx - 14, y_top + fret_gap * 0.7, f'{first}fr', 10, fill="#64748b", font_weight="500"))
    for i, fret in enumerate(frets):
        lx = sx + i * string_gap
        if fret < 0:
            result.append(_svg_text(lx, y_top - 4, '×', 11, fill="#dc2626", font_weight="700"))
        elif fret == 0:
            result.append(_svg_text(lx, y_top - 4, '○', 10, fill="#0f172a"))
        else:
            cy = y_top + (fret - first + 0.5) * fret_gap
            result.append(f'<circle cx="{lx:g}" cy="{cy:g}" r="3.6" fill="#0f172a"/>')
    for fret in sorted(set(positive)):
        indices = [i for i, f in enumerate(frets) if f == fret]
        if len(indices) > 1 and all(f >= fret for f in frets[indices[0]:indices[-1] + 1]):
            cy = y_top + (fret - first + 0.5) * fret_gap
            bx1 = sx + indices[0] * string_gap
            bx2 = sx + indices[-1] * string_gap
            result.append(f'<line x1="{bx1:g}" y1="{cy:g}" x2="{bx2:g}" y2="{cy:g}" stroke="#0f172a" stroke-width="5" stroke-linecap="round"/>')
    return ''.join(result)


def _render_tab_bar(arrangement, index, x_at, bar_width, left, string_y_base):
    result = []
    tab_right = left + bar_width
    for string in range(1, 7):
        y = string_y_base + (string - 1) * 15
        result.append(_line(left, y, tab_right, y, 1, "#64748b"))
    result.append(_line(tab_right, string_y_base, tab_right, string_y_base + 75, 1.5, "#0f172a"))
    beat_xs = set()
    for event in arrangement['events']:
        relative = event['start'] - index * arrangement['measure_beats']
        if not 0 <= relative < arrangement['measure_beats']:
            continue
        x = x_at(relative * TICKS)
        y = string_y_base + (event['string'] - 1) * 15
        technique = event.get('technique', '')
        label = 'X' if technique == 'X' else str(event['fret'])
        result.append(f'<rect x="{x - 9:g}" y="{y - 10:g}" width="18" height="16" fill="white"/>')
        if technique == 'X':
            result.append(_svg_text(x, y + 5, label, 15, fill="#dc2626", font_weight="700"))
        else:
            result.append(_svg_text(x, y + 5, label, 15, fill="#0f172a"))
        if technique == 'PM':
            result.append(_svg_text(x, string_y_base - 14, 'PM', 10, fill="#b45309", font_weight="600"))
        elif technique in ('H', 'P'):
            tc = "#2563eb"
            result.append(_svg_text(x + 14, y - 6, technique, 9, fill=tc, font_weight="600"))
            result.append(f'<path d="M{x + 2:g},{y - 12:g} Q{x + 10:g},{y - 20:g} {x + 18:g},{y - 12:g}" fill="none" stroke="{tc}" stroke-width="1"/>')
        beat_xs.add(x)
    stem_top = string_y_base + 80
    stem_bottom = stem_top + 14
    sorted_xs = sorted(beat_xs)
    for bx in sorted_xs:
        result.append(_line(bx, stem_top, bx, stem_bottom, 1, "#475569"))
    for i in range(0, len(sorted_xs) - 1, 2):
        result.append(_line(sorted_xs[i], stem_bottom, sorted_xs[i + 1], stem_bottom, 1.5, "#475569"))
    return ''.join(result)


def _render_jianpu_bar(project, spans, lyrics, x_at, lyric_y, baseline):
    result = []
    shift = baseline - 246
    for span in spans:
        tick = span['start_tick']
        for i, part in enumerate(duration_parts(span['duration_ticks'])):
            x = x_at(tick)
            pitch, octave, degree = span['pitch'], 0, '0'
            continuation = span['continuation'] or i > 0
            if pitch is not None:
                octave, semitone = divmod(pitch - (60 + NAMES.index(project.tonic)), 12)
                degree = DEGREES[semitone]
            digit = '⌒' if continuation and pitch is not None else degree[-1]
            result.append(_svg_text(x, baseline, digit, 17, fill="#1E1612", **{'data-note-index': span['note_index']}))
            if len(degree) > 1 and not continuation:
                result.append(_svg_text(x - 11, 240 + shift, degree[:-1], 11, fill="#1E1612"))
            for dot in range(abs(octave)):
                yy = 227 + shift - dot * 6 if octave > 0 else 263 + shift + dot * 6
                result.append(f'<circle cx="{x:g}" cy="{yy:g}" r="1.5" fill="#1E1612"/>')
            beams = max(0, part['denominator'].bit_length() - 3)
            for beam in range(beams):
                result.append(_line(x - 6, 251 + shift + beam * 3.5, x + 6, 251 + shift + beam * 3.5, 1.3, "#1E1612"))
            for dot in range(part['dots']):
                result.append(f'<circle cx="{x + 10 + dot * 4:g}" cy="{241 + shift:g}" r="1.5" fill="#1E1612"/>')
            if part['tuplet']:
                result.append(_svg_text(x, 218 + shift, '3', 10, fill="#1E1612"))
            for offset in range(TICKS, part['ticks'], TICKS):
                result.append(_svg_text(x_at(tick + offset), baseline, '—', 15, fill="#5C4938"))
            if pitch is not None and not continuation and lyrics:
                result.append(_svg_text(x, lyric_y, lyrics[span['note_index']], 13.5,
                                        font_weight="500", fill="#2C221E",
                                        **{'data-lyric-index': span['note_index']}))
            tick += part['ticks']
    return ''.join(result)


ZONE1_TOP = 10
CHORD_NAME_Y = 28
ROMAN_Y = 40
DIAGRAM_Y_TOP = 42
DIAGRAM_Y_BOTTOM = DIAGRAM_Y_TOP + 48
TAB_Y_BASE = 122
TAB_Y_BOTTOM = TAB_Y_BASE + 75
STEM_BOTTOM = TAB_Y_BASE + 92


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
    bars_per_row = 4
    widths, paddings = [], []
    for spans in timeline['bars']:
        width, anchors = 330.0, []
        for span in spans:
            lyric = note_lyrics[span['note_index']] if note_lyrics and span['pitch'] is not None and not span['continuation'] else ''
            estimated = sum(16 if ord(c) > 255 else 9 for c in lyric)
            anchors.append((span['start_tick'], max(22, estimated)))
        padding = min(40.0, max(30.0, max((w for _, w in anchors), default=22) / 2 + 10))
        for (a, wa), (b, wb) in zip(anchors, anchors[1:]):
            if b > a:
                dt = max(b - a, measure / 8.0)
                width = max(width, ((wa + wb) / 2 + 14) * measure / dt + 2 * padding)
        if anchors:
            width = max(width, anchors[0][1] + 70, anchors[-1][1] + 70)
        # Keep every bar width uniform and bounded so 4 bars per row display at crisp publication scale
        widths.append(min(360.0, max(320.0, width)))
        paddings.append(padding)

    clef_margin = 44

    systems = []
    for first in range(0, len(timeline['bars']), bars_per_row):
        end = min(first + bars_per_row, len(timeline['bars']))
        row_count = end - first
        low = max((max(0, -((s['pitch'] - (60 + NAMES.index(project.tonic))) // 12))
                   for spans in timeline['bars'][first:end] for s in spans if s['pitch'] is not None), default=0)
        high = max((max(0, (s['pitch'] - (60 + NAMES.index(project.tonic))) // 12)
                    for spans in timeline['bars'][first:end] for s in spans if s['pitch'] is not None), default=0)
        baseline = max(246, 232 + high * 6)
        lyric_y = max(baseline + 30, baseline + 22 + low * 6)
        height = lyric_y + 18
        total = clef_margin + sum(widths[first:end])
        # If the final row has fewer than bars_per_row bars, pad viewBox width to a full row so scale stays identical
        view_width = total if row_count == bars_per_row else clef_margin + sum(widths[first:end]) * (bars_per_row / max(1, row_count))
        content = []

        content.append(_svg_text(14, CHORD_NAME_Y, '和弦', 10, fill="#785A46", font_weight="600", text_anchor="start"))
        content.append(_svg_text(10, TAB_Y_BASE + 12, 'T', 12, fill="#785A46", font_weight="700", text_anchor="start"))
        content.append(_svg_text(10, TAB_Y_BASE + 34, 'A', 12, fill="#785A46", font_weight="700", text_anchor="start"))
        content.append(_svg_text(10, TAB_Y_BASE + 56, 'B', 12, fill="#785A46", font_weight="700", text_anchor="start"))
        string_labels = ['1', '2', '3', '4', '5', '6']
        for si, sl in enumerate(string_labels):
            sy = TAB_Y_BASE + si * 15
            content.append(_svg_text(28, sy + 4, sl, 8.5, fill="#8C7A6B", text_anchor="start"))
        content.append(_svg_text(10, baseline, '简谱', 10, fill="#785A46", font_weight="600", text_anchor="start"))
        content.append(_svg_text(10, lyric_y, '唱词', 10, fill="#785A46", font_weight="600", text_anchor="start"))

        content.append(_line(clef_margin, TAB_Y_BASE, clef_margin, TAB_Y_BASE + 75, 1.6, "#1E1612"))

        left = float(clef_margin)
        for index in range(first, end):
            width = widths[index]
            spans = timeline['bars'][index]
            padding = paddings[index]

            def x_at(tick, _left=left, _padding=padding, _width=width):
                return _left + _padding + tick / measure * (_width - 2 * _padding)

            content.append(f'<g role="group" aria-label="Bar {index + 1}">')

            bar_info = analysis.get(index, {})
            section = bar_info.get('section', '')
            if section:
                content.append(f'<rect x="{left + 2:g}" y="2" width="{len(section) * 8 + 12}" height="14" rx="3" fill="#FEF3C7" stroke="#D97706" stroke-width="0.6"/>')
                content.append(_svg_text(left + 8 + len(section) * 4, 11, section, 8.5, fill="#92400E", font_weight="700", text_anchor="middle"))

            content.append(f'<rect x="{left + 4:g}" y="{TAB_Y_BASE - 18:g}" width="22" height="13" rx="3" fill="#F5EFE6"/>')
            content.append(_svg_text(left + 15, TAB_Y_BASE - 8, f'#{index + 1}', 9, fill="#785A46", font_weight="700"))

            if index < len(arrangement['bars']):
                bar = arrangement['bars'][index]
                segments = bar['split_segments'] if 'split_segments' in bar else [bar]
                for part_idx, supplied in enumerate(segments):
                    segment = dict(supplied)
                    center = left + (part_idx + 0.5) * width / len(segments)
                    content.append(_svg_text(center, CHORD_NAME_Y, segment['chord'], 16, fill="#1E1612", font_weight="700"))
                    content.append(_chord_diagram(center, DIAGRAM_Y_TOP, segment))

                    if len(segments) > 1 and part_idx < len(segments) - 1:
                        div_x = left + (part_idx + 1) * width / len(segments)
                        content.append(f'<line x1="{div_x:g}" y1="{ZONE1_TOP:g}" x2="{div_x:g}" y2="{DIAGRAM_Y_BOTTOM + 8:g}" stroke="#D6C7B2" stroke-width="1" stroke-dasharray="4,3"/>')

            content.append(_render_tab_bar(arrangement, index, x_at, width, left, TAB_Y_BASE))
            content.append(_render_jianpu_bar(project, spans, note_lyrics, x_at, lyric_y, baseline))
            content.append('</g>')
            left += width

        systems.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" class="score-system" role="img" '
            f'aria-label="Score bars {first + 1}–{end}" '
            f'viewBox="0 0 {view_width:g} {height:g}" width="100%" '
            f'style="width:100%">'
            + ''.join(content) + '</svg>')

    title = escape(str(project.title), quote=True)

    capo = arrangement.get('capo', 0)
    capo_label = f'变调夹夹第 {capo} 品' if capo > 0 else '不夹变调夹 (原调)'
    style_label = arrangement.get('style', '')

    meta_pills = [
        f'1 = {project.tonic}',
        project.meter,
        f'速度 {project.bpm:g}',
        capo_label,
    ]
    if style_label:
        meta_pills.append(style_label)

    meta_html = ' '.join(
        f'<span class="pill">{escape(p, quote=True)}</span>' for p in meta_pills
    )

    legend = (
        '依据人声主旋律生成 · 上方：左手和弦按法图 · 中间：艺术木吉他六线谱 · 下方：人声主唱简谱与唱词'
    )

    css = (
        'body{background:#FFFDF9;color:#1E1612;margin:0 auto;max-width:1040px;padding:14px 18px;'
        'font-family:"Noto Serif SC","Songti SC","Noto Sans SC","Microsoft YaHei",sans-serif;'
        'line-height:1.45}'
        'h1{text-align:center;font-size:18px;font-weight:700;color:#1E1612;margin:0 0 6px}'
        '.meta-bar{text-align:center;margin:0 0 6px;display:flex;flex-wrap:wrap;justify-content:center;gap:6px}'
        '.pill{display:inline-block;padding:2px 10px;background:#F5EFE6;border:1px solid #D6C7B2;'
        'border-radius:16px;font-size:11.5px;color:#5C4938;font-weight:600}'
        '.legend{text-align:center;font-size:11px;color:#8C7A6B;margin:0 0 10px;'
        'border-bottom:1px solid #E6DEC8;padding-bottom:8px}'
        '.score-system{display:block;width:100%;max-width:1000px;margin:6px auto;overflow:visible}'
        '@media print{body{margin:0;padding:12px;max-width:none}'
        '.score-system{break-inside:avoid;max-width:none}h1{margin-top:0}}'
    )

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>' + title
        + '</title><style>' + css + '</style></head><body>'
        + '<h1>' + title + '</h1>'
        + '<div class="meta-bar">' + meta_html + '</div>'
        + '<p class="legend">' + escape(legend, quote=True) + '</p>'
        + (''.join(systems) if systems else '<p>暂无乐谱 · Empty score</p>')
        + '</body></html>'
    )
