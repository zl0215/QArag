"""把知识库里的全部 chunk 导出成 JSON，给建 golden set 时人工读用。

★ 为什么不是一次性的临时脚本：
  gold chunk_id 会随语料变化而失效（重灌、换 chunk_size、删文档）。留一个
  能随时重导的入口，改完语料才能重新对齐标注 —— 否则评测集就成了一次性
  消耗品，第二次改语料时只能凭记忆重标。

★ 为什么读 Postgres 而不是 Milvus：
  Postgres 是正文的**权威存储**，Milvus 只是可重建的派生索引（见 SPEC）。
  标注要对着权威的那一份做。

用法：
    python eval/dump_corpus.py                 # 全部文档
    python eval/dump_corpus.py --doc 12        # 只导某篇
    python eval/dump_corpus.py --digest        # 只要"结构摘要"，用来快速看主题
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from rag.core.config import get_settings  # noqa: E402
from rag.infra.db import Database  # noqa: E402
from rag.infra.models import EPOCH_ZERO  # noqa: E402

# chunk 表在 models.py 里的真实表名。写死在这里是有意的：这个脚本是**取证**工具，
# 依赖 ORM 模型意味着模型一改它就悄悄跟着变，导出结果和上一次不可比。
SQL = """
SELECT c.id, c.document_id, c.chunk_index, c.content, c.section_path,
       c.page_start, c.page_end, c.node_type, c.token_count,
       d.title AS doc_title
FROM chunks c
JOIN documents d ON d.id = c.document_id
  -- ★ 软删的判据是 `deleted_at = EPOCH_ZERO`（1970-01-01），**不是 IS NULL**。
  --   这一列是 NOT NULL，用 NULL 判"未删除"会一行都查不出来 —— 而且不报错，
  --   只是安静地返回空集，正是最费时间的那种错。常量取自 models.EPOCH_ZERO，
  --   不在这里写死字面量，否则哨兵一改这个脚本就悄悄失准。
  AND d.deleted_at = :epoch
  AND c.deleted_at = :epoch
  -- ★ 必须显式 CAST：`(:doc IS NULL OR ...)` 在 asyncpg 下会报
  --   AmbiguousParameterError —— 传 None 时它推不出 $1 的类型，
  --   而 `IS NULL` 本身不给类型提示。（psycopg 能猜，asyncpg 不猜。）
  AND (CAST(:doc AS INTEGER) IS NULL OR c.document_id = :doc)
ORDER BY c.document_id, c.chunk_index
"""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=int, default=None, help="只导某个 document_id")
    ap.add_argument("--digest", action="store_true",
                    help="只输出结构摘要（章节路径 + 每节块数 + 首句），不输出全文")
    ap.add_argument("--out", default=None, help="输出路径，默认 eval/corpus.json")
    args = ap.parse_args()

    settings = get_settings()
    db = Database(settings.database_url)
    try:
        async with db.session() as s:
            rows = (
                await s.execute(text(SQL), {"doc": args.doc, "epoch": EPOCH_ZERO})
            ).mappings().all()
    finally:
        await db.dispose()

    if args.digest:
        payload = _digest(rows)
        out = Path(args.out) if args.out else ROOT / "eval" / "corpus_digest.txt"
        out.write_text(payload, encoding="utf-8")
    else:
        records = [dict(r) for r in rows]
        out = Path(args.out) if args.out else ROOT / "eval" / "corpus.json"
        out.write_text(
            json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    print(f"{len(rows)} chunks -> {out}")
    return 0


def _digest(rows) -> str:  # noqa: ANN001
    """按文档 → 章节聚合，每节给块数、页码范围、每块首句。

    用来**一眼看出语料讲了什么**，好出题。全文导出是给标注 gold chunk_id 用的，
    读它来想题目会淹死在 676 段正文里。
    """
    lines: list[str] = []
    cur_doc = None
    cur_sec = None
    for r in rows:
        if r["document_id"] != cur_doc:
            cur_doc = r["document_id"]
            cur_sec = None
            lines.append(
                f"\n{'=' * 70}\n"
                f"[doc {cur_doc}] {r['doc_title']}  "
                f"(chunk_id {r['id']} 起)\n{'=' * 70}"
            )
        sec = r["section_path"] or "(无章节)"
        if sec != cur_sec:
            cur_sec = sec
            lines.append(f"\n  ── {sec}")
        head = " ".join((r["content"] or "").split())[:110]
        lines.append(
            f"    #{r['id']:<5} idx={r['chunk_index']:<4} "
            f"p{r['page_start']}-{r['page_end']:<3} {r['token_count']:>4}t  {head}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
