"""Deterministic scan fixtures for OCR acceptance; no text layer on raster pages."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

CHINESE_LINES = [
    "科研记录与数据管理",
    "研究数据应当保留原始记录与完整的实验日志。",
    "数据备份至少保留两份，并分别存放在不同设备中。",
    "样本编号采用统一编码，记录采集时间和处理方法。",
    "实验报告必须说明研究方法、结果以及存在的限制。",
    "跨学科研究需要核对术语定义，并保留文献来源。",
]
LEFT = ["Research methods", "Collect the original observations.",
        "Keep complete experiment logs.", "Record the sampling procedure.",
        "Archive the instrument settings.", "Document each processing step."]
RIGHT = ["Results and limitations", "Report uncertainty estimates.",
         "Validate the independent samples.", "Compare the control groups.",
         "Explain the study limitations.", "Preserve the source references."]


def default_font() -> Path:
    for name in [
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]:
        if Path(name).is_file():
            return Path(name)
    raise FileNotFoundError("Specify a Chinese font with --font (TTF/TTC/OTF)")


def create_samples(directory: Path, font_path: Path | None = None) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    font_path = font_path or default_font()
    zh = ImageFont.truetype(str(font_path), 36)
    en = ImageFont.truetype(str(font_path), 26)
    cn = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(cn)
    for i, line in enumerate(CHINESE_LINES):
        draw.text((70, 130 + i * 100), line, font=zh, fill="black")
    columns = Image.new("RGB", cn.size, "white")
    draw = ImageDraw.Draw(columns)
    for col, lines in enumerate([LEFT, RIGHT]):
        for i, line in enumerate(lines):
            draw.text((65 + col * 620, 150 + i * 100), line, font=en, fill="black")
    cn.save(directory / "scan-preview.png")
    columns.save(directory / "columns-preview.png")
    scanned = directory / "scanned.pdf"
    cn.save(scanned, "PDF", resolution=200, save_all=True, append_images=[columns])

    native_io = io.BytesIO()
    page = canvas.Canvas(native_io, pagesize=(446.4, 631.44))
    page.setFont("Helvetica", 12)
    page.drawString(35, 550, "Native page: Research records retain original page numbers.")
    page.drawString(35, 520, "The next page is a scanned research record.")
    page.save()
    writer = PdfWriter()
    writer.add_page(PdfReader(native_io).pages[0])
    # Slightly skewed scanned content tests tolerance to physical scanning.
    skew = io.BytesIO()
    cn.rotate(1.5, fillcolor="white").save(skew, "PDF", resolution=200)
    writer.add_page(PdfReader(skew).pages[0])
    mixed = directory / "mixed.pdf"
    with mixed.open("wb") as stream:
        writer.write(stream)
    blank = directory / "blank.pdf"
    Image.new("RGB", (1240, 1754), "white").save(blank, "PDF", resolution=200)
    return {"scanned": scanned, "mixed": mixed, "blank": blank}
