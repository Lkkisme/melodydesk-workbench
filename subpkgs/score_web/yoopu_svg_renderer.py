"""Chinese Publication-Grade ('Yoopu / 有谱么' Style) Combined Guitar Tab + Jianpu + Syllabic Lyric SVG/HTML Renderer.

SVG <-> JS interface contract (locked):

* ``svg.score-system`` carries ``data-bar-start`` / ``data-bar-end`` / ``data-measure-ticks`` and owns the
  interactive overlays ``rect.score-bar-highlight`` (background), ``line.score-hover-guide`` and
  ``g.score-playhead`` (foreground; moved purely through ``transform="translate(x,0)"``).
* ``g.score-bar`` carries ``data-bar-index`` / ``data-x-left`` / ``data-x-pad`` / ``data-bar-width`` and its first
  child is ``rect.score-bar-hit``.  Inside a bar the time <-> x bijection is
  ``x = left + pad + (localBeat / measureBeats) * (barWidth - 2 * pad)``.
* ``g.tab-note`` wraps every guitar event (knockout rect + fret text) with ``data-ev-start`` / ``data-ev-dur`` /
  ``data-string`` / ``data-fret`` / ``data-tech``; JS toggles ``.tab-active``.
* Jianpu digits carry ``data-note-index`` (``.jp-active``), lyrics carry ``data-lyric-index`` (``.lyric-active``).
"""
from __future__ import annotations
from html import escape
from music import NAMES, DEGREES
from ._score_timing import build_score_timeline, duration_parts, TICKS


RHYTHM_INK = "#475569"
LEAD_INK = "#9A3412"
LEAD_KNOCKOUT = "#FFFBEB"
ROLL_INK = "#785A46"
TIE_INK = "#6B5340"
LEAD_TECHNIQUES = ('CM', 'DS')
# A sixteenth is 12 ticks at 48 ticks per quarter; the +2 tolerance absorbs grid rounding while keeping
# eighth triplets (16 ticks) on a single beam.  Dotted sixteenths (18 ticks) are matched explicitly.
_SIXTEENTH_MAX_TICKS = TICKS // 4 + 2
_DOTTED_TICKS = (TICKS * 3 // 8, TICKS * 3 // 4, TICKS * 3 // 2, TICKS * 3)


def _svg_text(x, y, text, size=16, **attributes):
    merged = {"text-anchor": "middle"}
    for k, v in attributes.items():
        merged[k.replace("_", "-")] = v
    attrs = ' '.join(f'{escape(k, quote=True)}="{escape(str(v), quote=True)}"' for k, v in merged.items())
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" {attrs}>{escape(str(text), quote=True)}</text>'


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


def _has_slide(event):
    slide_from = event.get('slide_from_fret')
    return event.get('technique', '') == 'SL' or (slide_from is not None and int(slide_from) != int(event['fret']))


def _roll_arrow(x, y_bottom, y_top):
    """Vertical wavy arpeggio-roll line from the low string (bottom) up to the high string, with an upward arrowhead."""
    span = y_bottom - y_top
    waves = max(2, round(span / 5))
    step = span / waves
    d = [f'M{x:g},{y_bottom:g}']
    for i in range(waves):
        side = 2.3 if i % 2 == 0 else -2.3
        d.append(f'Q{x + side:g},{y_bottom - (i + 0.5) * step:g} {x:g},{y_bottom - (i + 1) * step:g}')
    d.append(f'M{x - 2.7:g},{y_top + 4:g} L{x:g},{y_top:g} L{x + 2.7:g},{y_top + 4:g}')
    return (f'<path class="tab-roll-arrow" d="{" ".join(d)}" fill="none" stroke="{ROLL_INK}" '
            f'stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round"/>')


def _rhythm_flag(x, y):
    """Curved flag attached to the tip (bottom) of a TAB rhythm stem."""
    return (f'<path class="tab-flag" d="M{x:g},{y:g} c0.5,-3.4 6.2,-4.2 5.4,-9.4" fill="none" '
            f'stroke="{RHYTHM_INK}" stroke-width="1.3" stroke-linecap="round"/>')


def _is_sixteenth(eff_ticks):
    return eff_ticks <= _SIXTEENTH_MAX_TICKS or eff_ticks == _DOTTED_TICKS[0]


def _render_tab_rhythm(onsets, measure_ticks, group_span, string_y_base):
    """Stems, beams, flags and duration dots below the TAB staff.

    ``onsets`` maps onset tick -> ``[x, min_event_duration_ticks]``.
    """
    result = []
    stem_top = string_y_base + 80
    stem_bottom = stem_top + 14
    ticks_sorted = sorted(onsets)
    groups: dict[int, list[tuple[float, int]]] = {}
    for k, tick in enumerate(ticks_sorted):
        x, min_ev_ticks = onsets[tick]
        next_tick = ticks_sorted[k + 1] if k + 1 < len(ticks_sorted) else measure_ticks
        eff_ticks = min(next_tick - tick, max(12, min_ev_ticks))
        g_idx = int((tick / TICKS + 1e-6) // group_span)
        groups.setdefault(g_idx, []).append((x, eff_ticks))

    for g_idx in sorted(groups):
        members = groups[g_idx]
        if len(members) == 1:
            x, eff = members[0]
            if eff >= 2 * TICKS:
                # Half / whole notes: short open stem
                result.append(_line(x, stem_top, x, stem_top + 7, 1, RHYTHM_INK))
            else:
                result.append(_line(x, stem_top, x, stem_bottom, 1, RHYTHM_INK))
                if eff <= TICKS * 0.6 or eff == _DOTTED_TICKS[1]:
                    result.append(_rhythm_flag(x, stem_bottom))
                    if _is_sixteenth(eff):
                        result.append(_rhythm_flag(x, stem_bottom - 4.2))
        else:
            for x, _eff in members:
                result.append(_line(x, stem_top, x, stem_bottom, 1, RHYTHM_INK))
            xs = [x for x, _ in members]
            result.append(f'<line class="tab-beam" x1="{xs[0] - 0.5:g}" y1="{stem_bottom:g}" x2="{xs[-1] + 0.5:g}" '
                          f'y2="{stem_bottom:g}" stroke="{RHYTHM_INK}" stroke-width="2.0"/>')
            sixteenth = [_is_sixteenth(eff) for _, eff in members]
            y2 = stem_bottom - 4.2
            i, count = 0, len(members)
            while i < count:
                if not sixteenth[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < count and sixteenth[j + 1]:
                    j += 1
                if j > i:
                    xa, xb = xs[i] - 0.5, xs[j] + 0.5
                else:
                    # Lone sixteenth against a dotted eighth: fractional beam stub pointing into the group
                    if i == 0:
                        xa, xb = xs[i] - 0.5, xs[i] + min(7.0, (xs[i + 1] - xs[i]) * 0.6)
                    else:
                        xa, xb = xs[i] - min(7.0, (xs[i] - xs[i - 1]) * 0.6), xs[i] + 0.5
                result.append(f'<line class="tab-beam" x1="{xa:g}" y1="{y2:g}" x2="{xb:g}" y2="{y2:g}" '
                              f'stroke="{RHYTHM_INK}" stroke-width="1.8"/>')
                i = j + 1
        for x, eff in members:
            if eff in _DOTTED_TICKS:
                result.append(f'<circle class="tab-dot" cx="{x + 4.5:g}" cy="{stem_top + 3.5:g}" r="1.3" fill="{RHYTHM_INK}"/>')
    return ''.join(result)


def _render_tab_bar(arrangement, index, x_at, bar_width, left, string_y_base):
    result = []
    tab_right = left + bar_width
    for string in range(1, 7):
        y = string_y_base + (string - 1) * 15
        result.append(_line(left, y, tab_right, y, 1, "#64748b"))
    result.append(_line(tab_right, string_y_base, tab_right, string_y_base + 75, 1.5, "#0f172a"))
    measure_beats = float(arrangement['measure_beats'])
    measure_ticks = round(measure_beats * TICKS)
    meter_str = str(arrangement.get('meter', '4/4')).strip()
    is_compound = meter_str.endswith('/8') and round(measure_beats * 2) >= 6 and round(measure_beats * 2) % 3 == 0
    group_span = 1.5 if is_compound else 1.0

    # Events may carry only start/duration (no 'beat' key): always derive the bar-relative position.
    in_bar = []
    stacks: dict[int, list[tuple[float, float, dict]]] = {}
    for event in arrangement['events']:
        relative = event['start'] - index * measure_beats
        if not 0 <= relative < measure_beats:
            continue
        tick = round(relative * TICKS)
        x = x_at(relative * TICKS)
        y = string_y_base + (event['string'] - 1) * 15
        in_bar.append((tick, x, y, event))
        stacks.setdefault(tick, []).append((x, y, event))

    # Arpeggio roll arrows sit behind the fret numbers, left of the chord stack
    for tick in sorted(stacks):
        stack = stacks[tick]
        if len({ev['string'] for _, _, ev in stack}) >= 2 and any(ev.get('roll_step_sec') for _, _, ev in stack):
            ys = [y for _, y, _ in stack]
            # Step further left when a slide marker (x-16 .. x-7) already occupies the default lane
            lane = 22 if any(_has_slide(ev) for _, _, ev in stack) else 12
            result.append(_roll_arrow(stack[0][0] - lane, max(ys) + 3, min(ys) - 4))

    onsets: dict[int, list] = {}
    pm_ticks = set()
    for tick, x, y, event in in_bar:
        technique = event.get('technique', '')
        has_slide = _has_slide(event)
        tech_attr = technique or ('SL' if has_slide else '')
        is_lead = technique in LEAD_TECHNIQUES
        label = 'X' if technique == 'X' else str(event['fret'])
        knockout = LEAD_KNOCKOUT if is_lead else 'white'
        result.append(
            f'<g class="tab-note{" is-lead" if is_lead else ""}" data-ev-start="{event["start"]:g}" '
            f'data-ev-dur="{event["duration"]:g}" data-string="{event["string"]}" data-fret="{event["fret"]}" '
            f'data-tech="{escape(str(tech_attr), quote=True)}">'
            f'<rect x="{x - 9:g}" y="{y - 10:g}" width="18" height="16" fill="{knockout}"/>')
        if technique == 'X':
            result.append(_svg_text(x, y + 5, label, 15, fill="#dc2626", font_weight="700"))
        elif is_lead:
            result.append(_svg_text(x, y + 5, label, 15, fill=LEAD_INK, font_weight="700"))
        else:
            result.append(_svg_text(x, y + 5, label, 15, fill="#0f172a"))
        result.append('</g>')

        if technique == 'PM':
            if tick not in pm_ticks:
                pm_ticks.add(tick)
                result.append(_svg_text(x, string_y_base - 14, 'PM', 10, fill="#b45309", font_weight="600"))
        elif technique in ('H', 'P'):
            tc = "#2563eb"
            # Slur from the previous note on the same string (legato source) into this note
            earlier = [(t, px) for t, px, _, ev in in_bar if ev['string'] == event['string'] and t < tick]
            x_from = max(earlier)[1] + 6 if earlier else x - 20
            x_to = x - 6
            if x_to - x_from < 6:
                x_from = x_to - 6
            span = x_to - x_from
            rise = min(9.0, 4.5 + span * 0.1)
            ctrl = span * 0.28
            y_slur = y - 10
            result.append(
                f'<path class="tab-slur" d="M{x_from:g},{y_slur:g} C{x_from + ctrl:g},{y_slur - rise:g} '
                f'{x_to - ctrl:g},{y_slur - rise:g} {x_to:g},{y_slur:g}" fill="none" stroke="{tc}" '
                f'stroke-width="1.1" stroke-linecap="round"/>')
            result.append(_svg_text((x_from + x_to) / 2, y_slur - rise * 0.75 - 2, technique, 8.5,
                                    fill=tc, font_weight="700"))
        if has_slide:
            slide_from = event.get('slide_from_fret')
            sc = "#0f766e"
            up_slide = True if slide_from is None else (int(slide_from) < int(event['fret']))
            y0, y1 = (y + 5, y - 4) if up_slide else (y - 4, y + 5)
            result.append(f'<line class="tab-slide" x1="{x - 16:g}" y1="{y0:g}" x2="{x - 7:g}" y2="{y1:g}" '
                          f'stroke="{sc}" stroke-width="1.4" stroke-linecap="round"/>')
            result.append(_svg_text(x - 12, y - 7, 'S', 8.5, fill=sc, font_weight="700"))

        if 0 <= tick < measure_ticks:
            ev_ticks = max(1, round(float(event['duration']) * TICKS))
            if tick in onsets:
                onsets[tick][1] = min(onsets[tick][1], ev_ticks)
            else:
                onsets[tick] = [x, ev_ticks]

    result.append(_render_tab_rhythm(onsets, measure_ticks, group_span, string_y_base))
    return ''.join(result)


def _render_jianpu_bar(project, spans, lyrics, x_at, lyric_y, baseline):
    result = []
    shift = baseline - 246
    previous = None  # (x, note_index) of the last rendered pitched part in this bar
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
            if pitch is not None and continuation:
                if previous is not None and previous[1] == span['note_index']:
                    gap = x - previous[0]
                    if gap > 4:
                        # The arc starts/ends beside the digits (never over their centre column), so it stays
                        # clear of the octave dots stacked above each digit.
                        inset = min(6.0, gap / 4)
                        xa, xb, ya = previous[0] + inset, x - inset, baseline - 14
                        rise = min(8.0, 3.5 + (xb - xa) * 0.07)
                        ctrl = (xb - xa) * 0.28
                        result.append(
                            f'<path class="jp-tie" d="M{xa:g},{ya:g} C{xa + ctrl:g},{ya - rise:g} '
                            f'{xb - ctrl:g},{ya - rise:g} {xb:g},{ya:g}" fill="none" stroke="{TIE_INK}" '
                            f'stroke-width="1.15" stroke-linecap="round"/>')
                elif previous is None and i == 0:
                    xa, xb, ya = x - 16.0, x - 5.0, baseline - 14
                    result.append(
                        f'<path class="jp-tie" d="M{xa:g},{ya:g} C{xa + 3.2:g},{ya - 4.5:g} '
                        f'{xb - 3.2:g},{ya - 4.5:g} {xb:g},{ya:g}" fill="none" stroke="{TIE_INK}" '
                        f'stroke-width="1.15" stroke-linecap="round"/>')
            previous = (x, span['note_index']) if pitch is not None else None
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


SCORE_CSS = (
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
    # Interactive layer
    '.score-bar{cursor:pointer}'
    '.score-bar-hit{transition:fill .12s ease}'
    '.score-bar:hover .score-bar-hit{fill:rgba(180, 83, 9, 0.03)}'
    '.score-bar-highlight{fill:rgba(217, 119, 6, 0.085);stroke:rgba(180, 83, 9, 0.28);stroke-width:1;'
    'pointer-events:none}'
    '.score-hover-guide{stroke:rgba(120, 90, 70, 0.35);stroke-width:1;stroke-dasharray:3,3;pointer-events:none}'
    '.score-playhead{pointer-events:none}'
    '.score-playhead-line{stroke:#C2410C;stroke-width:2.1;stroke-linecap:round}'
    '.score-playhead-cap{fill:#C2410C;stroke:#FFFDF9;stroke-width:1}'
    '.jp-active{fill:#B45309 !important;font-weight:800}'
    '.lyric-active{fill:#9A3412 !important;font-weight:700}'
    '.tab-active rect{fill:#FEF3C7 !important}'
    '.tab-active text{fill:#C2410C !important;font-weight:800}'
    # Print: ink only, no interactive chrome
    '@media print{body{margin:0;padding:12px;max-width:none}'
    '.score-system{break-inside:avoid;max-width:none}h1{margin-top:0}'
    '.score-playhead, .score-bar-highlight, .score-hover-guide { display: none !important; }'
    '.score-bar{cursor:auto}'
    '.score-bar-hit,.score-bar:hover .score-bar-hit{fill:transparent !important}'
    '.jp-active,.lyric-active,.tab-active text{fill:#0f172a !important;font-weight:inherit !important}'
    '.tab-active rect{fill:#ffffff !important}}'
)


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
        # Keep every bar width uniform and bounded so 4 bars per row display at crisp publication scale.
        # Rounded to 2 decimals so the data-x-left / data-x-pad / data-bar-width attributes (":g") reproduce the
        # exact geometry used for drawing — the JS time<->x mapping must be a true bijection.
        widths.append(round(min(360.0, max(320.0, width)), 2))
        paddings.append(round(padding, 2))

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

        # Background interactive layer (behind every bar group)
        content.append(f'<rect class="score-bar-highlight" x="0" y="4" width="0" height="{height - 8:g}" rx="6" style="display:none"/>')

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
        arr_for_tab = dict(arrangement, meter=project.meter)
        for index in range(first, end):
            width = widths[index]
            spans = timeline['bars'][index]
            padding = paddings[index]

            def x_at(tick, _left=left, _padding=padding, _width=width):
                return _left + _padding + tick / measure * (_width - 2 * _padding)

            content.append(
                f'<g class="score-bar" role="group" aria-label="Bar {index + 1}" data-bar-index="{index}" '
                f'data-x-left="{left:g}" data-x-pad="{padding:g}" data-bar-width="{width:g}">')
            content.append(f'<rect class="score-bar-hit" x="{left:g}" y="0" width="{width:g}" height="{height:g}" fill="transparent"/>')

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

            content.append(_render_tab_bar(arr_for_tab, index, x_at, width, left, TAB_Y_BASE))
            content.append(_render_jianpu_bar(project, spans, note_lyrics, x_at, lyric_y, baseline))
            content.append('</g>')
            left += width

        # Foreground interactive layer (above every bar group)
        content.append(f'<line class="score-hover-guide" x1="0" y1="8" x2="0" y2="{height - 8:g}" style="display:none"/>')
        content.append(
            f'<g class="score-playhead" style="display:none">'
            f'<line class="score-playhead-line" x1="0" y1="6" x2="0" y2="{height - 6:g}"/>'
            f'<circle class="score-playhead-cap" cx="0" cy="8" r="3.8"/></g>')

        systems.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" class="score-system" role="img" '
            f'aria-label="Score bars {first + 1}–{end}" '
            f'data-bar-start="{first}" data-bar-end="{end}" data-measure-ticks="{measure}" '
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
        '原曲和声基底与人声主旋律联合编配 · 上方：左手和弦按法图 · 中间：艺术木吉他六线谱 · 下方：人声主唱简谱与唱词'
    )

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>' + title
        + '</title><style>' + SCORE_CSS + '</style></head><body>'
        + '<h1>' + title + '</h1>'
        + '<div class="meta-bar">' + meta_html + '</div>'
        + '<p class="legend">' + escape(legend, quote=True) + '</p>'
        + (''.join(systems) if systems else '<p>暂无乐谱 · Empty score</p>')
        + '</body></html>'
    )
