"""表格 → Markdown（PDF / DOCX 共用）。

为什么用 Markdown 而不是 HTML 作为主格式：
索引与喂给 LLM 时省 token。HTML 版本（保留 rowspan/colspan）按需在展示层生成。
"""

from __future__ import annotations


def _clean(cell: str | None) -> str:
    return (cell or "").replace("\n", " ").replace("|", "\\|").strip()


def table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    grid = [[_clean(c) for c in row] + [""] * (width - len(row)) for row in rows]

    # 整表为空则视为无效
    if not any(any(cell for cell in row) for row in grid):
        return ""

    header, *body = grid
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
