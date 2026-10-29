"""Self-contained SVG chord diagrams and combined accompaniment score."""
import html
import math
from music import NAMES,DEGREES

def diagram(frets,x=42,y=42):
    positive=[f for f in frets if f>0]
    first=1 if max(positive,default=0)<=5 else min(positive)
    dx=11;dy=12;parts=[]
    for i in range(6):
        parts.append(f'<line x1="{x+i*dx}" y1="{y}" x2="{x+i*dx}" y2="{y+5*dy}" stroke="black"/>')
    for i in range(6):
        parts.append(f'<line x1="{x}" y1="{y+i*dy}" x2="{x+5*dx}" y2="{y+i*dy}" stroke="black" stroke-width="{3 if i==0 and first==1 else 1}"/>')
    parts.append(f'<text x="{x-24}" y="{y+10}" font-size="10">{first}品</text>')
    for string,fret in enumerate(frets):
        xx=x+string*dx
        if fret<=0:
            parts.append(f'<text x="{xx}" y="{y-5}" text-anchor="middle" font-size="11">{"×" if fret<0 else "○"}</text>')
        else:
            yy=y+(fret-first+.5)*dy
            parts.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="black"/>')
    # Show a barre only when every intervening string is fretted at or above it.
    if positive:
        base=min(positive);points=[i for i,f in enumerate(frets) if f==base]
        if len(points)>1 and all(f>=base for f in frets[min(points):max(points)+1]):
            yy=y+(base-first+.5)*dy
            parts.append(f'<line x1="{x+min(points)*dx}" y1="{yy}" x2="{x+max(points)*dx}" y2="{yy}" stroke="black" stroke-width="6" stroke-linecap="round"/>')
    return '<g class="chord-diagram">'+''.join(parts)+'</g>'

def lyric_bars(arrangement):
    measure=arrangement['measure'];notes=arrangement.get('melody',[])
    result={}
    for line in arrangement.get('lyrics',[]):
        start=line['start']
        nearest=min(notes,key=lambda n:abs(n['raw_start']-start),default=None)
        beat=(nearest['start'] if nearest and abs(nearest['raw_start']-start)<1.5 else (start-arrangement.get('beat_origin',0))*arrangement['bpm']/60)
        bar=max(0,min(len(arrangement['bars'])-1,int(beat//measure)))
        result.setdefault(bar,[]).append(line)
    return result

def melody_svg(notes,start,measure,tonic):
    parts=[];end=start+measure;cursor=start
    visible=[n for n in notes if n['start']<end-1e-7 and n['start']+n['duration']>start+1e-7]
    def glyph(at,length,pitch=None,continuation=False):
        x=38+(at-start)/measure*540;y=256
        relative=(pitch-(60+NAMES.index(tonic))) if pitch is not None else 0
        octave,semitone=divmod(relative,12)
        label='0' if pitch is None else '—' if continuation else DEGREES[semitone]
        parts.append(f'<text x="{x}" y="{y}" font-size="18" text-anchor="middle">{label}</text>')
        if pitch is not None and not continuation:
            for d in range(abs(octave)):
                dot_y=y-23-d*5 if octave>0 else y+12+d*5
                parts.append(f'<circle cx="{x}" cy="{dot_y}" r="1.5"/>')
        if length<.99:
            bars=1 if length>=.49 else 2
            if abs(length-2/3)<.015:bars=0
            if abs(length-1/3)<.015:bars=1
            if abs(length-1/6)<.015:bars=2
            for b in range(bars):
                yy=y+3+b*4
                parts.append(f'<line x1="{x-7}" y1="{yy}" x2="{x+7}" y2="{yy}" stroke="black"/>')
            if abs(length-.75)<.015 or abs(length-.375)<.015:
                parts.append(f'<circle cx="{x+12}" cy="{y-5}" r="1.5"/>')
            if any(abs(length-d)<.015 for d in (1/3,2/3,1/6)):
                parts.append(f'<text x="{x}" y="{y-32}" text-anchor="middle" font-size="9">3连</text>')
        elif abs(length-1.5)<.015:
            parts.append(f'<circle cx="{x+12}" cy="{y-5}" r="1.5"/>')
        for beat in range(1,int(length+1e-6)):
            xx=38+(at+beat-start)/measure*540
            parts.append(f'<text x="{xx}" y="{y}" font-size="16">—</text>')
        # Exact duration prevents the simplified notation from implying precision
        # beyond its supported engraving vocabulary.
        parts.append(f'<text x="{x}" y="292" text-anchor="middle" font-size="9" fill="#64748b">{length:.2g}拍</text>')
    for n in visible:
        a=max(start,n['start']);b=min(end,n['start']+n['duration'])
        if a>cursor+.02:glyph(cursor,a-cursor)
        glyph(a,b-a,n['pitch'],n['start']<start-1e-7)
        cursor=b
    if cursor<end-.02:glyph(cursor,end-cursor)
    return ''.join(parts)

def score_html(a):
    esc=html.escape;panels=[];sentences=lyric_bars(a)
    for index,bar in enumerate(a['bars']):
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 308" role="img" aria-label="第{index+1}小节 {esc(bar["chord"])}">',
               f'<text x="22" y="18" font-size="15">{index+1}. {esc(bar["chord"])}</text>',diagram(bar['frets']),
               '<text x="130" y="50" font-size="12">按弦图：左侧为六弦低音 E</text>',
               '<text x="130" y="70" font-size="12">○ 空弦　× 不弹　粗线为横按</text>']
        for string,label in enumerate(['e','B','G','D','A','E']):
            y=112+string*17
            parts.append(f'<text x="12" y="{y+5}" font-size="12">{label}</text><line class="tab-string" x1="32" y1="{y}" x2="580" y2="{y}" stroke="#64748b"/>')
        for step,slot in enumerate(bar['slots']):
            x=38+step/len(bar['slots'])*540
            beat=str(step+1) if a['meter']=='6/8' else str(step//2+1) if step%2==0 else '&'
            parts.append(f'<text x="{x}" y="218" font-size="10" text-anchor="middle">{esc(beat)}</text>')
            if slot and slot[0]['direction']:
                parts.append(f'<text x="{x}" y="103" text-anchor="middle">{slot[0]["direction"]}</text>')
            for event in slot:
                y=112+(5-event['string'])*17
                parts.append(f'<rect x="{x-8}" y="{y-8}" width="16" height="16" fill="white"/><text x="{x}" y="{y+5}" text-anchor="middle" font-size="14">{event["fret"]}</text>')
        parts.append(melody_svg(a.get('melody',[]),index*a['measure'],a['measure'],a.get('tonic','C')))
        parts.append('</svg>')
        lines=sentences.get(index,[])
        text='<br>'.join(f"<span>{line['start']:.1f}s {'✓' if line.get('reviewed') else '待校对'}　{esc(line['text'])}</span>" for line in lines)
        panels.append('<section>'+''.join(parts)+'<div class="lyrics">'+(text or '　')+'</div></section>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{esc(a['title'])} · 弹唱谱</title>
<style>body{{font:15px "Microsoft YaHei",system-ui;background:#eef1f4;color:#152131;padding:24px}}main{{max-width:1200px;margin:auto}}header,section{{background:white;padding:16px}}h1{{font-size:25px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:16px}}section{{break-inside:avoid;border:1px solid #ddd}}svg{{display:block;width:100%;color:black}}.lyrics{{min-height:42px;line-height:1.7;font-size:13px;padding:0 16px;overflow-wrap:anywhere}}audio{{width:100%;max-width:450px}}button{{padding:9px 18px}}@media(max-width:750px){{.grid{{grid-template-columns:1fr}}}}@media print{{body{{background:white;padding:0}}.controls{{display:none}}header{{font-size:11px}}section{{padding:6px}}.grid{{gap:6px}}}}</style>
<main><header><h1>{esc(a['title'])} · 吉他弹唱草稿</h1><p>1={esc(a.get('tonic','C'))}（需确认）　{a['meter']}　♩={a['bpm']:.1f}　{esc(a['style'])}　变调夹 {a['capo']} 品</p><p>{esc(a['warning'])}</p><p>六线谱上方是一弦高音 e，指型图左侧是六弦低音 E。品位相对变调夹，和弦名为实际音高。下方是主唱简谱草稿，标有精确拍数；非标准时值和复杂连音仍需校对。歌词按句定位，不是逐字音符对齐。</p><div class="controls"><p>旋律＋吉他 <audio controls src="melody-guitar.wav"></audio></p><p>吉他伴奏 <audio controls src="guitar-only.wav"></audio></p><a href="guitar-score.pdf">下载 PDF</a> · <a href="accompaniment.mid" download>双轨 MIDI</a> · <a href="arrangement.json" download>编配数据</a> · <a href="lyrics.txt" download>歌词与时间</a> <button onclick="print()">打印／另存 PDF</button></div></header><div class="grid">{''.join(panels)}</div></main></html>'''
