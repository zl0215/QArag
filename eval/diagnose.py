"""召回率归因诊断：把"为什么低"拆成可分别测量的几个假设。

★ 为什么不把英文查询直接写进 golden.jsonl：
  golden 是**冻结的基准**，改它就没法和之前的数字比。诊断用的中英对照放在这里，
  只服务于"定位原因"，不进正式评测。

四个假设，各自可独立证伪：
  A 跨语言鸿沟 —— 查询是中文、语料是英文论文，而嵌入模型是 bge-large-zh。
  B 参考文献污染 —— 参考文献列表会提到所有方法名，对任何查询都"像"。
  C 分词/稀疏通道 —— BM25 用的 analyzer 对英文论文是否可用。
  D 抽取质量 —— 双栏 PDF 抽出来的文本有多少是残的。

用法：python eval/diagnose.py [a|b|c|d|all]
"""

from __future__ import annotations

import argparse
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

# 40 道可回答题的中→英对照。英文侧刻意用论文里的原词
# （granular ball / slack factor / purity / location parameter），
# 因为这才是"用论文的语言问论文"的样子。
EN: dict[str, str] = {
    "s01": "What is the time complexity of the granular-ball classifier in GBSVM?",
    "s02": "How many particles (pop) are used in the PSO algorithm of GBSVM?",
    "s03": "Over what interval is the purity threshold optimized in GBSVM?",
    "s04": "Which model is ISFFSVM built upon as an improvement?",
    "s05": "What is the computational complexity of ISFFSVM?",
    "s06": "What value of the location parameter a is used in the ISFFSVM illustration?",
    "s07": "Which two diseases is Pin-GBTSVM applied to diagnose?",
    "s08": "What is the computational complexity of generating granular balls in Pin-GBTSVM?",
    "s09": "Which Alzheimer's disease dataset is used in the Pin-GBTSVM experiments?",
    "s10": "Which statistical learning principle does Pin-GBTSVM lack?",
    "s11": "How many imbalanced benchmark datasets is RD-IFTSVM evaluated on?",
    "s12": "How is the number of nearest neighbors k chosen in RD-IFTSVM?",
    "s13": "What is the time complexity of RD-IFTSVM?",
    "s14": "Which kind of fuzzy set does RD-IFTSVM use for the nonmembership degree?",
    "s15": "What is the definition of the slack factor in SFFSVM?",
    "s16": "What is the computational complexity of SFFSVM?",
    "s17": "What is the range of the smoothing parameter beta in SFFSVM?",
    "s18": "In which journal, year and volume was CKA-FSVM published?",
    "s19": "What does centered kernel alignment (CKA) measure?",
    "s20": "How many folds does the protein fold prediction dataset in CKA-FSVM have?",
    "s21": "What is the core idea of the curriculum learning based fuzzy support vector machine?",
    "s22": "What is the time complexity of DBSCAN in the curriculum learning FSVM?",
    "s23": "What value is the parameter beta set to in the curriculum learning FSVM experiments?",
    "s24": "How is the purity of a fuzzy granular ball defined in GBFSVM?",
    "s25": "Under what condition does GBFSVM become equivalent to FSVM?",
    "s26": "At which conference was GB-CLFSVM published?",
    "s27": "What clustering method is the dynamic granular-ball splitting strategy in GB-CLFSVM based on?",
    "s28": "How much did the accuracy of GB-CLFSVM and CLFSVM drop under noise interference?",
    "s29": "What are the symptoms when the SSH tunnel goes down?",
    "s30": "Which two configurations does the worker preflight refuse?",
    "m01": "Which papers use granular ball methods?",
    "m02": "Compare the time complexity of GBSVM, ISFFSVM and SFFSVM, which is lower?",
    "m03": "Which papers specifically address class imbalance problems?",
    "m04": "Which papers apply curriculum learning to fuzzy support vector machines?",
    "m05": "Which papers improve upon FSVM?",
    "m06": "How to distinguish an SSH tunnel failure from a worker startup failure?",
    "m07": "Which papers use DBSCAN for sample selection?",
    "m08": "How are granular balls and fuzzy sets combined?",
    "m09": "Do any of these papers come from the same research group?",
    "m10": "Which papers redefine the usage of the slack variable?",
}

REF_PAT = re.compile(r"(vol\.|pp\.|et al\.|IEEE Trans|arXiv|Proceedings|doi:|\[\d+\])", re.I)


def load() -> tuple[list[dict], dict[int, dict]]:
    golden = [json.loads(line) for line in
              (EVAL / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = {r["id"]: r for r in
              json.loads((EVAL / "corpus.json").read_text(encoding="utf-8"))}
    return golden, corpus


def is_junk(text: str) -> bool:
    """参考文献块 / 表格碎片 / 版面噪声。

    判据是"这看起来像不像连续的自然语言正文"：短、且引用标记密集的，
    基本都是从参考文献或表格里切出来的。
    """
    t = " ".join((text or "").split())
    if len(t) < 40:
        return True
    return len(REF_PAT.findall(t)) >= 3


def score(g: dict, chunks: list[dict]) -> dict:
    gc, gd = set(g["gold_chunk_ids"]), set(g["gold_doc_ids"])
    cr = next((i + 1 for i, c in enumerate(chunks) if c["chunk_id"] in gc), None)
    dr = next((i + 1 for i, c in enumerate(chunks) if c["document_id"] in gd), None)
    return {"id": g["id"], "cr": cr, "dr": dr}


def agg(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "r@1": sum(1 for r in rows if r["cr"] and r["cr"] <= 1) / n,
        "r@5": sum(1 for r in rows if r["cr"] and r["cr"] <= 5) / n,
        "r@10": sum(1 for r in rows if r["cr"] and r["cr"] <= 10) / n,
        "mrr": sum((1 / r["cr"]) if r["cr"] else 0.0 for r in rows) / n,
    }


CONCURRENCY = 6   # 和 run_eval 一致：检索要打嵌入/重排服务，别一次全推过去


async def run_queries(cli: httpx.AsyncClient, pairs: list[tuple[dict, str]],
                      **kw) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(g: dict, q: str) -> dict:
        async with sem:
            d = (await cli.post(f"{API}/api/v1/retrieve",
                                json={"query": q, "top_k": TOP_K, **kw}, timeout=120)).json()
        return score(g, d["chunks"])
    return list(await asyncio.gather(*(one(g, q) for g, q in pairs)))


# ----------------------------------------------------------------------
async def diag_a(cli: httpx.AsyncClient, golden: list[dict]) -> dict:
    """假设 A：中文查询 vs 英文查询，同一套 gold。

    这是**唯一一个只改查询语言、不改任何检索代码**的对照。
    如果英文查询的召回显著更高，那低召回的主因就是跨语言，
    而不是切块、融合或重排 —— 后面那些都是次要项。
    """
    ans = [g for g in golden if g["type"] != "unanswerable"]
    zh = await run_queries(cli, [(g, g["question"]) for g in ans])
    print("  [A] 英文查询中…（40 次检索）")
    en = await run_queries(cli, [(g, EN[g["id"]]) for g in ans])
    return {"zh": agg(zh), "en": agg(en),
            "per_q": {r["id"]: {"zh": r["cr"], "en": e["cr"]}
                      for r, e in zip(zh, en, strict=True)}}


async def diag_b(cli: httpx.AsyncClient, golden: list[dict],
                 corpus: dict[int, dict]) -> dict:
    """假设 B：参考文献块占了多少 top-10，滤掉后召回变多少。

    ★ 这里的"滤掉"是**事后统计**，不等于真实修复 —— 真实修复要在重排前过滤，
      才能把名额让给正文块。所以这个数字是**下界**，不是收益上限。
    """
    ans = [g for g in golden if g["type"] != "unanswerable"]
    raw, filt = [], []
    junk_share: list[int] = []
    for g in ans:
        d = (await cli.post(f"{API}/api/v1/retrieve",
                            json={"query": g["question"], "top_k": TOP_K},
                            timeout=60)).json()
        cs = d["chunks"]
        junk_share.append(sum(1 for c in cs if is_junk(corpus[c["chunk_id"]]["content"])))
        raw.append(score(g, cs))
        kept = [c for c in cs if not is_junk(corpus[c["chunk_id"]]["content"])]
        filt.append(score(g, kept))
    return {"before": agg(raw), "after": agg(filt),
            "junk_per_query": sum(junk_share) / len(junk_share),
            "junk_share": junk_share}


async def diag_c(cli: httpx.AsyncClient, golden: list[dict]) -> dict:
    """假设 C：稀疏（BM25）通道到底有没有在工作。

    单跑 sparse，看它召回的块数和命中情况。语料是英文，而
    `MILVUS_ANALYZER=chinese` —— 中文分词器切英文论文，很可能切得一塌糊涂。
    """
    ans = [g for g in golden if g["type"] != "unanswerable"]
    sparse = await run_queries(cli, [(g, g["question"]) for g in ans],
                               use_dense=False, use_rerank=False)
    dense = await run_queries(cli, [(g, g["question"]) for g in ans],
                              use_sparse=False, use_rerank=False)
    # 稀疏通道的实际召回条数（diagnostics.recall.sparse）
    counts = []
    for g in ans[:10]:
        d = (await cli.post(f"{API}/api/v1/retrieve",
                            json={"query": g["question"], "top_k": TOP_K,
                                  "use_dense": False}, timeout=60)).json()
        counts.append(d["diagnostics"]["recall"].get("sparse", 0))
    return {"sparse_only": agg(sparse), "dense_only": agg(dense),
            "sparse_recall_counts": counts}


def diag_d(corpus: dict[int, dict]) -> dict:
    """假设 D：抽取出来的正文有多少是残的。

    不需要调接口 —— 直接看语料本身。空格被吞掉的长串
    （`IEEETRANSACTIONSON...`）会让子词切分几乎失效，向量和 BM25 一起受害。
    """
    docs: dict[int, dict] = {}
    for r in corpus.values():
        t = r["content"]
        has_long = bool(re.search(r"[A-Za-z]{25,}", t))
        stripped = bool(re.search(r"[a-z][A-Z]{3,}", t))   # 空格被吞的痕迹
        d = docs.setdefault(r["document_id"], {"n": 0, "junk": 0, "long": 0, "stripped": 0})
        d["n"] += 1
        d["junk"] += is_junk(t)
        d["long"] += has_long
        d["stripped"] += stripped
    return docs


# ----------------------------------------------------------------------
def _fmt(name: str, a: dict) -> str:
    return (f"  {name:12} R@1={a['r@1']:.3f}  R@5={a['r@5']:.3f}  "
            f"R@10={a['r@10']:.3f}  MRR={a['mrr']:.3f}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["a", "b", "c", "d", "all"])
    args = ap.parse_args()
    golden, corpus = load()
    # ★ 合并而非覆盖：单跑 `diagnose.py d` 会把之前 a/b/c 的结果冲掉，
    #   而下游 analyze_failures.py 要读 A 组。分次跑就不该互相踩。
    res = EVAL / "results" / "diagnose.json"
    out: dict = json.loads(res.read_text(encoding="utf-8")) if res.exists() else {}

    async with httpx.AsyncClient() as cli:
        if args.which in ("a", "all"):
            print("── 假设 A：跨语言（中文查询 vs 英文查询）")
            out["A"] = await diag_a(cli, golden)
            print(_fmt("中文查询", out["A"]["zh"]))
            print(_fmt("英文查询", out["A"]["en"]))
            worse = [k for k, v in out["A"]["per_q"].items()
                     if (v["zh"] or 99) > (v["en"] or 99)]
            print(f"  英文更优的题数：{len(worse)}/40\n")

        if args.which in ("b", "all"):
            print("── 假设 B：参考文献/碎片块污染")
            out["B"] = await diag_b(cli, golden, corpus)
            print(_fmt("滤除前", out["B"]["before"]))
            print(_fmt("滤除后(事后)", out["B"]["after"]))
            print(f"  top-10 里平均有 {out['B']['junk_per_query']:.1f} 条是垃圾块\n")

        if args.which in ("c", "all"):
            print("── 假设 C：稀疏通道是否有效")
            out["C"] = await diag_c(cli, golden)
            print(_fmt("仅稀疏", out["C"]["sparse_only"]))
            print(_fmt("仅稠密", out["C"]["dense_only"]))
            print(f"  稀疏通道召回条数(前10题)：{out['C']['sparse_recall_counts']}\n")

        if args.which in ("d", "all"):
            print("── 假设 D：PDF 抽取质量")
            out["D"] = diag_d(corpus)
            print(f"  {'doc':>4} {'块数':>5} {'垃圾块':>7} {'含超长串':>9} {'空格被吞':>9}")
            for did, d in sorted(out["D"].items()):
                print(f"  {did:>4} {d['n']:>5} {d['junk']:>7} "
                      f"{d['long']:>9} {d['stripped']:>9}")
            print()

    res.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {res}（含本次跑过的 {', '.join(k for k in ('A','B','C','D') if k in out)}）")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
