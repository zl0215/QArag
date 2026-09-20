"""漏斗分级测量：gold 块到底是在哪一级丢的。

★ 为什么这个测量最重要：
  "召回率低"有两种完全不同的病，药也完全不同 ——
    (a) 候选池里压根没有 gold  → 嵌入/切块的锅，重排再强也救不回来
    (b) 候选池里有 gold，但重排没把它排上来 → 排序的锅
  只看最终的 recall@5 是分不出来的。把每一级的 recall@50 量出来，
  (a) 和 (b) 立刻分开。

同时跑中文和英文两套查询，这样"跨语言到底伤在哪一级"也能看到。

用法：python eval/funnel.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import EN  # noqa: E402  复用中英对照，别再抄一份

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
API = "http://127.0.0.1:8000"
POOL = 50          # 候选池深度，跟 RETRIEVE_DENSE_TOP_K 对齐
CONCURRENCY = 6

STAGES = {
    "dense 池@50":   {"use_dense": True,  "use_sparse": False, "use_rerank": False},
    "sparse 池@50":  {"use_dense": False, "use_sparse": True,  "use_rerank": False},
    "融合池@50":      {"use_dense": True,  "use_sparse": True,  "use_rerank": False},
    "融合池+重排@50":  {"use_dense": True,  "use_sparse": True,  "use_rerank": True},
}


def rank_of(g: dict, chunks: list[dict]) -> int | None:
    gc = set(g["gold_chunk_ids"])
    return next((i + 1 for i, c in enumerate(chunks) if c["chunk_id"] in gc), None)


async def run(cli: httpx.AsyncClient, ans: list[dict], qmap, cfg: dict) -> list[int | None]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(g: dict) -> int | None:
        async with sem:
            d = (await cli.post(f"{API}/api/v1/retrieve",
                                json={"query": qmap(g), "top_k": POOL, **cfg},
                                timeout=180)).json()
        return rank_of(g, d["chunks"])
    return list(await asyncio.gather(*(one(g) for g in ans)))


def summ(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {f"r@{k}": sum(1 for r in ranks if r and r <= k) / n
            for k in (1, 5, 10, 30, 50)} | {"mrr": sum(
                (1 / r) if r else 0.0 for r in ranks) / n}


async def main() -> int:
    golden = [json.loads(line) for line in
              (EVAL / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ans = [g for g in golden if g["type"] != "unanswerable"]

    out: dict = {}
    # 英文查询不会触发翻译（语言一致），所以只对中文跑"翻译开/关"两档。
    ARMS = (
        ("zh+翻译", lambda g: g["question"], True),
        ("zh原样", lambda g: g["question"], False),
        ("en", lambda g: EN[g["id"]], None),
    )
    async with httpx.AsyncClient() as cli:
        for lang, qmap, xlate in ARMS:
            print(f"── {lang}")
            out[lang] = {}
            for name, cfg in STAGES.items():
                if xlate is not None:
                    cfg = {**cfg, "use_translate": xlate}
                ranks = await run(cli, ans, qmap, cfg)
                s = summ(ranks)
                out[lang][name] = {"summary": s, "ranks": ranks}
                print(f"   {name:16} R@5={s['r@5']:.3f}  R@10={s['r@10']:.3f}  "
                      f"R@30={s['r@30']:.3f}  R@50={s['r@50']:.3f}  MRR={s['mrr']:.3f}")
            print()

    (EVAL / "results" / "funnel.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 结论行：把 (a) 候选缺失 和 (b) 排序丢失 分开 ----
    print("=" * 72)
    for arm in (a[0] for a in ARMS):
        pool = out[arm]["融合池@50"]["summary"]["r@50"]
        final = out[arm]["融合池+重排@50"]["summary"]["r@10"]
        dense = out[arm]["dense 池@50"]["summary"]["r@50"]
        sparse = out[arm]["sparse 池@50"]["summary"]["r@50"]
        print(f"[{arm:8}] 稠密池 R@50={dense:.3f}  稀疏池 R@50={sparse:.3f}  "
              f"融合池 R@50={pool:.3f}")
        print(f"           候选生成丢掉 {1-pool:.1%}（池里就没有），"
              f"重排+截断再丢 {pool-final:.1%}（池里有但没排上来）")
    print("\n-> eval/results/funnel.json")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
