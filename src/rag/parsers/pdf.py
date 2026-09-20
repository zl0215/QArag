"""PDF 解析（pdfplumber，MIT）。

三件在中文 PDF 上必须做的事：
1. **丢弃 furniture**（页眉/页脚/页码）—— 它们在每页重复，是检索噪声的最大来源
2. **按字号推断标题层级** —— PDF 没有语义结构，只有字号和坐标
3. **表格区域从正文中剔除** —— 否则同一段文字会被索引两遍（正文一次、表格一次）

局限：扫描件（无文本层）会被标记出来而不是静默产出空 chunk。
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from html import unescape
from pathlib import Path

import pdfplumber

from rag.core.logging import get_logger
from rag.parsers.base import finalize
from rag.parsers.tables import table_to_markdown
from rag.schemas.document import DocMeta, DocNode, NodeType, ParsedDocument, make_node_id

logger = get_logger(__name__)

PARSER_NAME = "pdfplumber-layout-v3"
# 字号超过正文中位数的这个倍数才算标题
_HEADING_SIZE_RATIO = 1.12
# 家具判定：出现在超过这个比例页面上，且位于页面上下边缘区
_FURNITURE_PAGE_RATIO = 0.35
_EDGE_ZONE = 0.12
# 一页可抽取字符数低于此值，视为无文本层
_MIN_CHARS_PER_PAGE = 30
# pdfplumber 默认 x_tolerance=3 会把很多论文中的英文词直接粘成一串。
# IEEE/Elsevier PDF 常用字符定位而不是真空格，实测 1 可以恢复绝大多数词界。
_X_TOLERANCE = 1
_Y_TOLERANCE = 3
_MIN_TWO_COLUMN_ROWS = 6

_CID = re.compile(r"\(cid:\d+\)", re.I)
_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")
_NUMBERED_HEADING = re.compile(
    r"^(?:[IVXLC]+\.|[A-Z]\.|\d+(?:\.\d+)*\.)\s+[A-Z\u4e00-\u9fff]"
)
_REFERENCE_MARKERS = re.compile(r"\b(vol\.|no\.|pp\.|doi:|arxiv:)", re.I)
_REFERENCE_START = re.compile(r"^(?:references|bibliography|参考文献)\b", re.I)
_ABSTRACT_START = re.compile(r"\babstract\s*[—:-]", re.I)


def _extract_pdf_metadata(path: Path) -> tuple[str, str]:
    """提取稳定的书目信息和首页作者区，作为独立可检索证据。

    PDF 侧栏中的会议名、页眉中的作者常被版面抽取器拆散。pypdf 按内容流读取
    首页时不负责正文分栏，却很适合读取 Abstract 之前的题名/作者/单位；标准
    metadata 的 Subject 则通常直接包含期刊、会议和 DOI。
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        raw = reader.metadata or {}
        title = unescape(str(raw.get("/Title") or "")).strip()
        author = unescape(str(raw.get("/Author") or "")).strip()
        subject = unescape(str(raw.get("/Subject") or "")).strip()
        first_page = reader.pages[0].extract_text() if reader.pages else ""
    except Exception as exc:  # noqa: BLE001 - metadata 失败不应阻断正文解析
        logger.debug("pdf.metadata_extract_failed", error=type(exc).__name__)
        return "", ""

    first_page = _SPACE.sub(" ", unescape(first_page or "")).strip()
    match = _ABSTRACT_START.search(first_page)
    front_matter = first_page[:match.start()].strip() if match else first_page[:1600].strip()
    parts = []
    if title:
        parts.append(f"Document title: {title}")
    if author:
        parts.append(f"Author metadata: {author}")
    if front_matter:
        parts.append(f"Title-page authors and affiliations: {front_matter}")
    if subject:
        parts.append(f"Publication: {subject}")
    return title, "\n".join(parts)


def _classify_heading_levels(size_counts: Counter[float]) -> dict[float, int]:
    """按字号从大到小映射到 1..6 级标题。"""
    if not size_counts:
        return {}
    body_size = size_counts.most_common(1)[0][0]
    heading_sizes = sorted(
        (s for s in size_counts if s >= body_size * _HEADING_SIZE_RATIO),
        reverse=True,
    )
    return {size: min(idx + 1, 6) for idx, size in enumerate(heading_sizes)}


def _furniture_key(text: str) -> str:
    """把每页变化的页码归一化后再比较页眉页脚。"""
    text = _DIGITS.sub("#", text.casefold())
    return _SPACE.sub("", text).strip("-—|·•")


def _collect_furniture(pages_lines: list[list[dict]], heights: list[float]) -> set[str]:
    """找出跨页重复出现的页眉/页脚文本。"""
    counter: Counter[str] = Counter()
    for lines, height in zip(pages_lines, heights, strict=False):
        if not height:
            continue
        seen_on_page: set[str] = set()
        for line in lines:
            top_ratio = line["top"] / height
            if top_ratio < _EDGE_ZONE or top_ratio > 1 - _EDGE_ZONE:
                text = line["text"].strip()
                if text:
                    key = _furniture_key(text)
                    if key and key not in seen_on_page:
                        counter[key] += 1
                        seen_on_page.add(key)

    threshold = max(2, int(len(pages_lines) * _FURNITURE_PAGE_RATIO))
    furniture = {text for text, n in counter.items() if n >= threshold}
    if furniture:
        logger.debug("pdf.furniture_dropped", count=len(furniture), sample=list(furniture)[:5])
    return furniture


def _line_rows(words: list[dict]) -> list[list[dict]]:
    """按视觉 y 坐标把词归到同一行，供双栏检测使用。"""
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        top = float(word["top"])
        row = next(
            (candidate for candidate in reversed(rows[-4:])
             if abs(float(candidate[0]["top"]) - top) <= _Y_TOLERANCE),
            None,
        )
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    return rows


def _detect_two_columns(page) -> float | None:  # noqa: ANN001
    """返回双栏分隔线；单栏页返回 None。

    只看“左右都有词”会把普通单栏误判成双栏。这里还要求多行在页面中部有
    稳定的宽 gutter：单栏的一行在中点附近只是普通词间距，通常只有 2--6pt；
    双栏论文的栏间距通常在 14pt 以上。
    """
    try:
        words = page.extract_words(
            x_tolerance=_X_TOLERANCE,
            y_tolerance=_Y_TOLERANCE,
            keep_blank_chars=False,
        ) or []
    except Exception:
        return None

    width, height = float(page.width), float(page.height)
    center = width / 2
    body = [w for w in words if height * _EDGE_ZONE <= float(w["top"]) <= height * (1 - _EDGE_ZONE)]
    gaps: list[tuple[float, float, float]] = []
    for row in _line_rows(body):
        left = [w for w in row if float(w["x1"]) < center]
        right = [w for w in row if float(w["x0"]) > center]
        if not left or not right:
            continue
        left_edge = max(float(w["x1"]) for w in left)
        right_edge = min(float(w["x0"]) for w in right)
        gap = right_edge - left_edge
        if gap >= 10 and width * 0.38 <= (left_edge + right_edge) / 2 <= width * 0.62:
            gaps.append((gap, left_edge, right_edge))

    if len(gaps) < _MIN_TWO_COLUMN_ROWS:
        return None
    median_gap = statistics.median(g[0] for g in gaps)
    if median_gap < 9.5:
        return None
    split = statistics.median((g[1] + g[2]) / 2 for g in gaps)
    return min(max(split, width * 0.42), width * 0.58)


def _extract_layout_lines(page) -> tuple[list[dict], float | None]:  # noqa: ANN001
    """按真实阅读顺序抽行：双栏页先左栏，再右栏。"""
    split = _detect_two_columns(page)
    regions = [(0.0, 0.0, float(page.width), float(page.height))]
    if split is not None:
        regions = [
            (0.0, 0.0, split, float(page.height)),
            (split, 0.0, float(page.width), float(page.height)),
        ]

    out: list[dict] = []
    for column, bbox in enumerate(regions):
        crop = page.crop(bbox) if split is not None else page
        lines = crop.extract_text_lines(
            x_tolerance=_X_TOLERANCE,
            y_tolerance=_Y_TOLERANCE,
            strip=True,
            return_chars=True,
        ) or []
        for line in lines:
            item = dict(line)
            item["column"] = column
            out.append(item)
    return out, split


def _looks_like_heading(text: str, size: float, body_size: float) -> bool:
    clean = _CID.sub("", text).strip()
    if not clean or len(clean) > 180 or len(clean.split()) > 24:
        return False
    if len(_REFERENCE_MARKERS.findall(clean)) >= 2:
        return False
    alpha_count = sum(ch.isalpha() for ch in clean)
    if len(clean) <= 2 or alpha_count < 4 or not any(
        len(word.strip(".,:;()[]")) >= 3 for word in clean.split()
    ):
        return False
    if _NUMBERED_HEADING.match(clean):
        return True
    letters = [ch for ch in clean if ch.isalpha()]
    upper_ratio = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
    return (len(clean.split()) <= 14 and size >= body_size * _HEADING_SIZE_RATIO) and (
        not clean.endswith((".", "。", ";", "；")) or upper_ratio > 0.75
    )


def _is_running_header(text: str, top: float, height: float) -> bool:
    """过滤页码变化导致“跨页重复”规则漏掉的期刊页眉。"""
    if top / max(height, 1) > 0.08:
        return False
    upper = text.upper()
    return (
        "TRANSACTIONS ON" in upper
        or (" VOL. " in f" {upper} " and " NO. " in f" {upper} ")
        or bool(re.match(r"^\d{2,}\s+[A-Z]", upper))
        or bool(re.search(r"\bet\s+al\.?:.*\d{2,}\s*$", text, re.I))
    )


def _heading_level(text: str, size: float, heading_levels: dict[float, int], body_size: float) -> int | None:
    if not _looks_like_heading(text, size, body_size):
        return None
    numbered = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
    if numbered:
        return min(numbered.group(1).count(".") + 1, 6)
    return heading_levels.get(size, 2)


def _join_wrapped_lines(lines: list[str]) -> str:
    """恢复论文排版换行，并处理跨行断词。"""
    out = ""
    for raw in lines:
        text = _SPACE.sub(" ", raw).strip()
        if not text:
            continue
        if not out:
            out = text
            continue
        if out.endswith("-") and text[0].islower():
            out = out[:-1] + text
        else:
            out += " " + text
    return out


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 16:
        return False
    # 纯数字、罗马数字，或 "第 3 页 / 共 10 页" 这类
    return stripped.isdigit() or stripped.strip("-— 第页共/").isdigit()


def parse_pdf(path: Path, meta: DocMeta) -> ParsedDocument:
    nodes: list[DocNode] = []
    warnings: list[str] = []
    reading_order = 0
    cursor = 0  # 在重建全文中的字符偏移

    metadata_title, metadata_text = _extract_pdf_metadata(path)
    if metadata_title and not meta.title:
        meta.title = metadata_title
    if metadata_text:
        nodes.append(DocNode(
            node_id=make_node_id(meta.doc_id, reading_order),
            doc_id=meta.doc_id,
            type=NodeType.METADATA,
            reading_order=reading_order,
            text=metadata_text,
            page_start=1,
            page_end=1,
            heading_path="Document metadata",
            char_start=cursor,
            char_end=cursor + len(metadata_text),
        ))
        cursor += len(metadata_text) + 2
        reading_order += 1

    with pdfplumber.open(path) as pdf:
        meta.page_count = len(pdf.pages)
        raw_pages: list[dict] = []

        # ---- 第一遍：抽取所有行与表格，统计字号分布 ----
        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                lines, column_split = _extract_layout_lines(page)
            except Exception as exc:  # pdfplumber 对畸形 PDF 会抛各种异常
                warnings.append(f"第 {page_no} 页文本抽取失败：{type(exc).__name__}")
                lines = []
                column_split = None

            tables = []
            try:
                for table in page.find_tables():
                    data = table.extract()
                    if data and any(any(c for c in row) for row in data):
                        tables.append({"bbox": table.bbox, "rows": data})
            except Exception as exc:
                warnings.append(f"第 {page_no} 页表格抽取失败：{type(exc).__name__}")

            raw_pages.append({
                "page_no": page_no,
                "width": page.width,
                "height": page.height,
                "lines": lines,
                "tables": tables,
                "column_split": column_split,
            })

        # ---- 统计 ----
        size_counts: Counter[float] = Counter()
        total_chars = 0
        for pg in raw_pages:
            for line in pg["lines"]:
                text = line["text"].strip()
                if not text:
                    continue
                total_chars += len(text)
                for ch in line.get("chars", []):
                    size_counts[round(float(ch.get("size", 0)), 1)] += 1

        body_size = size_counts.most_common(1)[0][0] if size_counts else 0.0
        heading_levels = _classify_heading_levels(size_counts)
        furniture = _collect_furniture(
            [pg["lines"] for pg in raw_pages], [pg["height"] for pg in raw_pages]
        )

        if total_chars < _MIN_CHARS_PER_PAGE * max(len(raw_pages), 1):
            warnings.append("整份文档可抽取文本极少，可能是扫描件（需要 OCR）")

        # ---- 第二遍：按阅读顺序产出节点 ----
        heading_stack: list[tuple[int, str]] = []
        in_references = False

        for pg in raw_pages:
            page_no, width = pg["page_no"], pg["width"]
            column_split = pg["column_split"]
            table_boxes = [t["bbox"] for t in pg["tables"]]
            items: list[tuple[tuple[int, float], str, dict]] = []

            for line in pg["lines"]:
                text = line["text"].strip()
                if not text:
                    continue
                center_y = (line["top"] + line["bottom"]) / 2
                center_x = (line.get("x0", 0) + line.get("x1", width)) / 2
                # 落在表格区域内的行由表格节点负责，不重复产出
                if any(box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
                       for box in table_boxes):
                    continue
                if (_furniture_key(text) in furniture or _is_page_number(text)
                        or _is_running_header(text, float(line["top"]), float(pg["height"]))):
                    continue
                chars = line.get("chars", [])
                size = (
                    statistics.median(float(c.get("size", 0)) for c in chars)
                    if chars else 0.0
                )
                column = int(line.get("column", 0))
                items.append(((column, float(line["top"])), "line", {
                    "text": text,
                    "size": round(size, 1),
                    "column": column,
                    "top": float(line["top"]),
                    "bottom": float(line["bottom"]),
                    "x0": float(line.get("x0", 0)),
                    "x1": float(line.get("x1", width)),
                }))

            for tbl in pg["tables"]:
                table_center = (tbl["bbox"][0] + tbl["bbox"][2]) / 2
                column = 1 if column_split is not None and table_center > column_split else 0
                items.append(((column, float(tbl["bbox"][1])), "table", tbl))

            items.sort(key=lambda it: it[0])

            paragraph: list[dict] = []

            def append_node(text: str, node_type: NodeType, node_page: int, *, level: int | None = None,
                            table: dict | None = None, bbox=None) -> None:  # noqa: ANN001
                nonlocal cursor, reading_order
                node = DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=node_type,
                    level=level,
                    reading_order=reading_order,
                    text=text,
                    table_markdown=text if table is not None else None,
                    table_cells=([[c or "" for c in row] for row in table["rows"]]
                                 if table is not None else None),
                    page_start=node_page,
                    page_end=node_page,
                    heading_path=_render_heading_path(heading_stack),
                    char_start=cursor,
                    char_end=cursor + len(text),
                    bbox=bbox,
                )
                cursor += len(text) + 2
                nodes.append(node)
                reading_order += 1

            def flush_paragraph(node_page: int = page_no) -> None:
                nonlocal paragraph, in_references
                if not paragraph:
                    return
                text = _join_wrapped_lines([line["text"] for line in paragraph])
                if text:
                    if _REFERENCE_START.match(text):
                        in_references = True
                    append_node(
                        text,
                        NodeType.REFERENCE if in_references else NodeType.PARAGRAPH,
                        node_page,
                        bbox=(
                            min(line["x0"] for line in paragraph),
                            min(line["top"] for line in paragraph),
                            max(line["x1"] for line in paragraph),
                            max(line["bottom"] for line in paragraph),
                        ),
                    )
                paragraph = []

            for _, kind, payload in items:
                if kind == "table":
                    flush_paragraph()
                    md = table_to_markdown(payload["rows"])
                    if not md:
                        continue
                    append_node(md, NodeType.TABLE, page_no, table=payload, bbox=payload["bbox"])
                    continue

                text = payload["text"]
                level = _heading_level(text, payload["size"], heading_levels, body_size)
                if level is not None:
                    flush_paragraph()
                    if _REFERENCE_START.match(text):
                        in_references = True
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, text))
                    append_node(
                        text,
                        NodeType.TITLE,
                        page_no,
                        level=level,
                        bbox=(payload["x0"], payload["top"], payload["x1"], payload["bottom"]),
                    )
                    continue

                # 栏切换和明显的大纵向间隔意味着新的段落。普通排版换行在这里合并，
                # 避免每一视觉行都变成带双换行的“伪段落”。
                if paragraph:
                    prev = paragraph[-1]
                    gap = payload["top"] - prev["bottom"]
                    if (payload["column"] != prev["column"]
                            or gap > max(payload["size"], prev["size"], body_size) * 1.35):
                        flush_paragraph()
                paragraph.append(payload)

            flush_paragraph()

    meta.parse_warnings.extend(warnings)
    return finalize(meta, nodes, parser_name=f"{PARSER_NAME}:pdf")


def _render_heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)
