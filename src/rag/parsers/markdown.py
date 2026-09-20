"""Markdown 解析（markdown-it-py，MIT）。

★ 绝对不要用正则解析 Markdown。代码块里的 `#` 会被误判成标题，
表格里的 `|` 会被误判成列分隔 —— 这是最常见的"Markdown 解析事故"。
必须走真正的 CommonMark 解析器，并且记得 **`.enable("table")`**（表格不在 CommonMark 核心规范里）。

用 token 的 `.map`（源码行号区间）反推字符偏移，因此 char_start/char_end 指向原文真实位置。
"""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from rag.core.logging import get_logger
from rag.parsers.base import finalize
from rag.schemas.document import DocMeta, DocNode, NodeType, ParsedDocument, make_node_id

logger = get_logger(__name__)

PARSER_NAME = "markdown-it-py"

# 块级容器 token：开/闭配对，整体作为一个节点
_CONTAINER_TYPES = {
    "bullet_list_open": NodeType.LIST,
    "ordered_list_open": NodeType.LIST,
    "blockquote_open": NodeType.PARAGRAPH,
    "table_open": NodeType.TABLE,
}


def _line_offsets(text: str) -> list[int]:
    """第 i 行首字符在全文中的偏移。最后一项等于全文长度。"""
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _slice(text: str, offsets: list[int], token_map: list[int] | None) -> tuple[str, int, int]:
    if not token_map:
        return "", 0, 0
    start_line, end_line = token_map
    if start_line >= len(offsets):
        return "", 0, 0
    start = offsets[start_line]
    end = offsets[min(end_line, len(offsets) - 1)]
    return text[start:end].strip(), start, end


def _inline_text(token) -> str:  # noqa: ANN001
    """把 inline token 的 children 拍平成纯文本，保留代码与换行。"""
    parts: list[str] = []
    for child in token.children or []:
        if child.type == "text":
            parts.append(child.content)
        elif child.type in ("code_inline",):
            parts.append(f"`{child.content}`")
        elif child.type in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif child.type == "image":
            alt = child.attrGet("alt") or ""
            parts.append(f"[图片: {alt}]" if alt else "[图片]")
        elif child.type in ("link_open", "link_close"):
            continue
        elif child.content:
            parts.append(child.content)
    return "".join(parts).strip()


def parse_markdown(path: Path, meta: DocMeta, *, plain: bool = False) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    offsets = _line_offsets(raw)

    md = MarkdownIt("commonmark")
    if not plain:
        md = md.enable("table")
    tokens = md.parse(raw)

    nodes: list[DocNode] = []
    reading_order = 0
    heading_stack: list[tuple[int, str]] = []

    idx = 0
    total = len(tokens)
    while idx < total:
        token = tokens[idx]

        # ---- 标题 ----
        if token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[idx + 1] if idx + 1 < total else None
            text = _inline_text(inline) if inline else ""
            _, char_start, char_end = _slice(raw, offsets, token.map)

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

            if text:
                nodes.append(DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=NodeType.TITLE,
                    level=level,
                    reading_order=reading_order,
                    text=text,
                    heading_path=" > ".join(t for _, t in heading_stack),
                    char_start=char_start,
                    char_end=char_end,
                ))
                reading_order += 1
            idx += 3  # heading_open / inline / heading_close
            continue

        # ---- 围栏代码块（原子，绝不切碎）----
        if token.type == "fence":
            lang = (token.info or "").strip().split()[0] if token.info else ""
            body = token.content.rstrip("\n")
            _, char_start, char_end = _slice(raw, offsets, token.map)
            if body:
                nodes.append(DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=NodeType.CODE,
                    reading_order=reading_order,
                    text=body,
                    code_lang=lang or None,
                    heading_path=" > ".join(t for _, t in heading_stack),
                    char_start=char_start,
                    char_end=char_end,
                ))
                reading_order += 1
            idx += 1
            continue

        # ---- 缩进代码块 ----
        if token.type == "code_block":
            body = token.content.rstrip("\n")
            _, char_start, char_end = _slice(raw, offsets, token.map)
            if body:
                nodes.append(DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=NodeType.CODE,
                    reading_order=reading_order,
                    text=body,
                    heading_path=" > ".join(t for _, t in heading_stack),
                    char_start=char_start,
                    char_end=char_end,
                ))
                reading_order += 1
            idx += 1
            continue

        # ---- 表格 / 列表 / 引用：整体取源码切片 ----
        if token.type in _CONTAINER_TYPES:
            close_type = token.type.replace("_open", "_close")
            depth, end_idx = 1, idx + 1
            while end_idx < total and depth > 0:
                if tokens[end_idx].type == token.type:
                    depth += 1
                elif tokens[end_idx].type == close_type:
                    depth -= 1
                end_idx += 1

            close_map = tokens[end_idx - 1].map if end_idx - 1 < total else None
            start_map = token.map
            merged = None
            if start_map and close_map:
                merged = [start_map[0], close_map[1]]
            elif start_map:
                merged = start_map

            snippet, char_start, char_end = _slice(raw, offsets, merged)
            node_type = _CONTAINER_TYPES[token.type]

            if node_type is NodeType.TABLE:
                # 表格保留源码形态（已是 Markdown 表），另存二维数组便于展示
                cells = _parse_md_table(snippet)
                text = snippet
            else:
                cells = None
                text = _flatten_block(tokens[idx:end_idx], raw, offsets)

            if text:
                nodes.append(DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=node_type,
                    reading_order=reading_order,
                    text=text,
                    table_cells=cells,
                    table_markdown=text if node_type is NodeType.TABLE else None,
                    heading_path=" > ".join(t for _, t in heading_stack),
                    char_start=char_start,
                    char_end=char_end,
                ))
                reading_order += 1
            idx = end_idx
            continue

        # ---- 普通段落 ----
        if token.type == "paragraph_open":
            inline = tokens[idx + 1] if idx + 1 < total else None
            text = _inline_text(inline) if inline else ""
            _, char_start, char_end = _slice(raw, offsets, token.map)
            if text:
                nodes.append(DocNode(
                    node_id=make_node_id(meta.doc_id, reading_order),
                    doc_id=meta.doc_id,
                    type=NodeType.PARAGRAPH,
                    reading_order=reading_order,
                    text=text,
                    heading_path=" > ".join(t for _, t in heading_stack),
                    char_start=char_start,
                    char_end=char_end,
                ))
                reading_order += 1
            idx += 3
            continue

        idx += 1

    return finalize(meta, nodes, parser_name=f"{PARSER_NAME}:{'txt' if plain else 'md'}")


def _flatten_block(tokens: list, raw: str, offsets: list[int]) -> str:  # noqa: ANN001
    """列表/引用：把内部每个 inline 的文本取出来，列表项用 - 前缀。"""
    lines: list[str] = []
    for token in tokens:
        if token.type == "inline":
            text = _inline_text(token)
            if text:
                lines.append(text)
        elif token.type == "list_item_open":
            lines.append("- ")
    # 把 "- " 与紧随其后的文本拼起来
    merged: list[str] = []
    for line in lines:
        if line == "- " and merged and not merged[-1].startswith("- "):
            merged[-1] = "- " + merged[-1]
        else:
            merged.append(line)
    return "\n".join(merged).strip()


def _parse_md_table(snippet: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in snippet.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} and c for c in cells):
            continue  # 分隔行
        rows.append(cells)
    return rows
