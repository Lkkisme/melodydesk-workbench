"""Print the same SVG score offline; no browser or network required."""
import io
import re
import html
from pathlib import Path

def export_pdf(arrangement,path):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics import renderPDF
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from svglib.svglib import svg2rlg
    from score_layout import score_html,lyric_bars
    font=Path('C:/Windows/Fonts/simhei.ttf')
    if not font.is_file():raise RuntimeError('PDF 导出需要 Windows 黑体 simhei.ttf')
    if 'ScoreCN' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('ScoreCN',str(font)))
    svgs=re.findall(r'<svg.*?</svg>',score_html(arrangement),re.S)
    lyrics=lyric_bars(arrangement)
    page=canvas.Canvas(str(path),pagesize=(595.28,841.89))
    page.setTitle(arrangement['title']+' - 吉他弹唱草稿')
    style=ParagraphStyle('lyrics',fontName='ScoreCN',fontSize=7,leading=10)
    total=(len(svgs)+7)//8
    for sheet in range(total):
        page.setFont('ScoreCN',14);page.drawString(30,815,arrangement['title']+' · 吉他弹唱草稿')
        page.setFont('ScoreCN',8)
        page.drawString(30,798,f"1={arrangement['tonic']} (候选)   {arrangement['meter']}   BPM {arrangement['bpm']:.1f}   变调夹 {arrangement['capo']} 品")
        page.drawString(30,784,'和弦与歌词待校对；歌词按句定位。品位相对变调夹，简谱下方为拍数。')
        for position,svg in enumerate(svgs[sheet*8:sheet*8+8]):
            index=sheet*8+position;col=position%2;row=position//2
            x=30+col*272;y=768-row*178
            svg=svg.replace('<text ','<text font-family="ScoreCN" ')
            drawing=svg2rlg(io.BytesIO(svg.encode('utf-8')))
            scale=258/drawing.width
            drawing.scale(scale,scale)
            renderPDF.draw(drawing,page,x,y-133)
            text='<br/>'.join(f"{line['start']:.1f}s {'已校对' if line.get('reviewed') else '待校对'} {html.escape(line['text'])}" for line in lyrics.get(index,[]))
            if text:
                paragraph=Paragraph(text,style);w,h=paragraph.wrap(252,1000)
                if h>37:
                    # Keep overflow in the bundled editable lyric file rather
                    # than silently overlap the following staff.
                    paragraph=Paragraph(html.escape('歌词较长，请查阅随附歌词文件。'),style);w,h=paragraph.wrap(252,37)
                paragraph.drawOn(page,x+4,y-140-h)
        page.setFont('ScoreCN',8);page.drawCentredString(297,22,f'{sheet+1} / {total}')
        page.showPage()
    page.save()
