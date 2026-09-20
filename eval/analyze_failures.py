"""失败归因：把 top-10 里的每一条按"文本质量"分档，看召回到底丢在哪一档。

★ 为什么不用 diagnose.py 里的 is_junk：
  那一版判据是"长度<40 或 引用标记>=3"。放到真实语料上它把 **16.9% 的 gold 块**
  也判成了垃圾 —— 因为学术正文本身就满篇 `[12]`，而 `[0.7,1]` 这种数学区间
  也会被正则当成引用。判据本身错了，基于它的"污染率"就是伪影。
  这里换成**参考文献条目**的专有特征：同时出现 vol./no./pp./doi 中的两个以上。
  正文几乎不可能这么写。

三条相互独立的线索，各自可以直接读出结论：
  1. 抽取质量 —— PDF 双栏抽取把空格吞了（`puritythresholdisoptimized`）。
     按 gold 块自身是否"粘连"给题目分两组，比召回率。
  2. 参考文献污染 —— 用严格判据重新数一遍，并统计真 gold 被误判的比例。
  3. 跨语言 —— 复用 diagnose.json 的 A 组结果。

用法：python eval/analyze_failures.py
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
API = "http://127.0.0.1:8000"
TOP_K = 10
CONCURRENCY = 6

# 参考文献条目的专有特征：这四个标记同时出现两个以上，基本只有文献列表会这样。
# 单看 `[12]` 不行 —— 学术正文里到处都是。
REF_ENTRY = re.compile(r"\b(vol\.|no\.|pp\.|doi:|arXiv:)", re.I)
# 空格被吞的证据：20 个以上连续字母。正常英文单词不会这么长。
GLUED = re.compile(r"[A-Za-z]{20,}")
CJK = re.compile(r"[一-鿿]")


def is_ref_entry(t: str) -> bool:
    return len(REF_ENTRY.findall(t)) >= 2


def is_glued(t: str) -> bool:
    return bool(GLUED.search(t))


def hit_rank(g: dict, chunks: list[dict]) -> int | None:
    gc = set(g["gold_chunk_ids"])
    return next((i + 1 for i, c in enumerate(chunks) if c["chunk_id"] in gc), None)


async def main() -> int:
    golden = [json.loads(line) for line in
              (EVAL / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = {r["id"]: r for r in
              json.loads((EVAL / "corpus.json").read_text(encoding="utf-8"))}
    diag = json.loads((EVAL / "results" / "diagnose.json").read_text(encoding="utf-8"))

    ans = [g for g in golden if g["type"] != "unanswerable"]
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as cli:
        async def one(g: dict) -> tuple[dict, list[dict]]:
            async with sem:
                d = (await cli.post(f"{API}/api/v1/retrieve",
                                    json={"query": g["question"], "top_k": TOP_K},
                                    timeout=120)).json()
            return g, d["chunks"]

        pairs = await asyncio.gather(*(one(g) for g in ans))

    # ---- 1. 抽取质量：按 gold 块自身是否粘连分组 ----
    rows = []
    for g, cs in pairs:
        gc = g["gold_chunk_ids"]
        gold_texts = [corpus[c]["content"] for c in gc if c in corpus]
        rows.append({
            "id": g["id"], "type": g["type"], "rank": hit_rank(g, cs),
            "gold_glued": any(is_glued(t) for t in gold_texts),
            "gold_ref": any(is_ref_entry(t) for t in gold_texts),
        })

    def rate(sub: list[dict]) -> str:
        n = len(sub) or 1
        return (f"n={len(sub):>2}  R@5={sum(1 for r in sub if r['rank'] and r['rank']<=5)/n:.3f}  "
                f"R@10={sum(1 for r in sub if r['rank'] and r['rank']<=10)/n:.3f}")

    glued = [r for r in rows if r["gold_glued"]]
    clean = [r for r in rows if not r["gold_glued"]]

    print("=" * 72)
    print("1. 抽取质量（gold 块是否被 PDF 抽取粘连）")
    print(f"   gold 粘连   {rate(glued)}")
    print(f"   gold 干净   {rate(clean)}")
    print("   -> 粘连组召回" + ("更低，抽取质量是瓶颈之一" if
          (sum(1 for r in glued if r['rank'] and r['rank']<=10)/max(len(glued),1)) <
          (sum(1 for r in clean if r['rank'] and r['rank']<=10)/max(len(clean),1))
          else "不低，抽取质量不是主因"))

    # ---- 2. 污染：严格判据 ----
    print()
    print("=" * 72)
    print("2. 参考文献条目污染（严格判据：vol./no./pp./doi 出现 >=2 个）")
    all_gold = [c for g in ans for c in g["gold_chunk_ids"]]
    false_pos = [c for c in all_gold if is_ref_entry(corpus[c]["content"])]
    print(f"   gold 块被误判为文献条目：{len(false_pos)}/{len(all_gold)} "
          f"({len(false_pos)/len(all_gold):.1%})  <- 判据有多干净")

    miss_top, hit_top, corpus_ref = [], [], 0
    for r, (_g, cs) in zip(rows, pairs, strict=True):
        for c in cs:
            ref = is_ref_entry(corpus[c["chunk_id"]]["content"])
            (miss_top if r["rank"] is None else hit_top).append(ref)
    for r in corpus.values():
        corpus_ref += is_ref_entry(r["content"])
    print(f"   top-10 里文献条目的比例：未命中题 "
          f"{sum(miss_top)/max(len(miss_top),1):.1%}，命中题 "
          f"{sum(hit_top)/max(len(hit_top),1):.1%}")
    print(f"   全库基线 {corpus_ref/len(corpus):.1%}  "
          f"(过采样倍数 {sum(miss_top)/max(len(miss_top),1) / max(corpus_ref/len(corpus),1e-9):.2f}x)")

    # ---- 3. 跨语言 ----
    print()
    print("=" * 72)
    print("3. 跨语言（来自 diagnose.py，只改查询语言）")
    if "A" in diag:
        a = diag["A"]
        for k, label in (("zh", "中文查询"), ("en", "英文查询")):
            print(f"   {label}   R@5={a[k]['r@5']:.3f}  R@10={a[k]['r@10']:.3f}  MRR={a[k]['mrr']:.3f}")
        print(f"   -> 同一套 gold、同一套检索代码，只把问题换成英文，"
              f"R@5 提升 {a['en']['r@5']-a['zh']['r@5']:+.3f}")
    else:
        print("   （缺 A 组数据，先跑 python eval/diagnose.py a）")

    out = {"quality": {"glued": glued, "clean": clean},
           "ref_entry": {"gold_false_pos": len(false_pos), "n_gold": len(all_gold),
                         "miss_share": sum(miss_top)/max(len(miss_top),1),
                         "hit_share": sum(hit_top)/max(len(hit_top),1),
                         "corpus_share": corpus_ref/len(corpus)},
           "per_q": rows}
    (EVAL / "results" / "failures.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n-> eval/results/failures.json")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
