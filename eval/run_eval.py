"""RAG 评测：检索指标 + 消融对照 + 拒答准确率 + 引用真实率。

★ 为什么分两条链路（retrieval / chat）而不是全走 /chat：
  检索指标**不需要 LLM**，一次 50 题 × 6 组消融只要几百毫秒；走 /chat 的话
  每题都要等一次生成，六组消融就是 300 次 LLM 调用。把"检索好不好"和
  "生成/拒答对不对"分开测，前者可以随便重跑，后者才需要省着用。

★ 为什么指标分 chunk 级和 doc 级两套：
  这是个**科研助手**：用户真正问的是"哪篇论文里有 X"，doc 级命中就算有用；
  但 chunk 级 recall 才能反映切块与排序的质量。只看 doc 级会掩盖"找对了论文
  但没找对段落"，只看 chunk 级又会让数字低到看不出改进。两套一起报。

用法：
    python eval/run_eval.py check                # 校验 golden 与语料是否对得上
    python eval/run_eval.py retrieval            # 检索指标 + 六组消融
    python eval/run_eval.py chat                 # 拒答准确率 + 引用真实率
    python eval/run_eval.py all
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
API = os.getenv("RAG_EVAL_API", "http://127.0.0.1:8000").rstrip("/")
TOP_K = 10                      # 检索深度：recall@k 的 k 最大取到这个值
CONCURRENCY = 2                 # 本机单卡还同时跑 reranker；过高并发会制造假失败

# 上一条的消融设计：每一组只相对 full 改**一个**开关。
# 这样任何一行的差异都能唯一归因到一个组件，不会出现"同时去掉两样、
# 说不清是谁的功劳"。dense_only / sparse_only 是单通道下限，用来定位瓶颈。
CONFIGS: dict[str, dict[str, bool]] = {
    "full":        {"use_dense": True,  "use_sparse": True,  "use_rerank": True,
                    "use_translate": True},
    # ★ `-translate` 就是**上一版的 full**。中文提问、英文语料、纯中文嵌入模型，
    #   所以这一行是"改之前"的基线，和 `full` 的差就是跨语言查询扩展的净收益。
    "-translate":  {"use_dense": True,  "use_sparse": True,  "use_rerank": True,
                    "use_translate": False},
    "-rerank":     {"use_dense": True,  "use_sparse": True,  "use_rerank": False,
                    "use_translate": True},
    "-dense":      {"use_dense": False, "use_sparse": True,  "use_rerank": True,
                    "use_translate": True},
    "-sparse":     {"use_dense": True,  "use_sparse": False, "use_rerank": True,
                    "use_translate": True},
    "dense_only":  {"use_dense": True,  "use_sparse": False, "use_rerank": False,
                    "use_translate": True},
    "sparse_only": {"use_dense": False, "use_sparse": True,  "use_rerank": False,
                    "use_translate": True},
}


# ======================================================================
# 数据加载
# ======================================================================
def load_golden() -> list[dict]:
    rows = []
    for line in (EVAL / "golden.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_evidence(golden: list[dict], corpus: dict[int, dict]) -> list[dict]:
    """把稳定证据短语解析为当前索引的 chunk_id。

    数据库自增 ID 会在每次重新分块后变化，直接把 ID 当长期 gold 会让“改进
    分块”这件事本身破坏评测。短语只描述答案证据，按文档范围映射到当前块。
    """
    evidence_path = EVAL / "evidence.json"
    patterns = json.loads(evidence_path.read_text(encoding="utf-8"))

    def normalized(text: str) -> str:
        table = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        return "".join(ch for ch in text.translate(table).casefold() if ch.isalnum())

    resolved: list[dict] = []
    for original in golden:
        row = dict(original)
        phrases = [normalized(value) for value in patterns.get(row["id"], [])]
        if phrases:
            allowed_docs = set(row["gold_doc_ids"])
            matches = []
            for chunk_id, chunk in corpus.items():
                if chunk["document_id"] not in allowed_docs:
                    continue
                content = normalized(chunk["content"])
                if any(phrase and phrase in content for phrase in phrases):
                    matches.append(chunk_id)
            row["gold_chunk_ids"] = matches
            row["gold_evidence"] = patterns[row["id"]]
        resolved.append(row)
    return resolved


def load_corpus() -> dict[int, dict]:
    """chunk_id -> chunk。引用真实率要靠它做逐字比对。"""
    data = json.loads((EVAL / "corpus.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in data}


# ======================================================================
# check：先证明标注是对的，再谈指标
# ======================================================================
def cmd_check(golden: list[dict], corpus: dict[int, dict]) -> int:
    """校验 golden.jsonl 自洽。

    ★ 这一步不能省。gold_chunk_id 打错一个数字，评测不会报错，只会安静地
      把 recall 拉低几个点 —— 然后你会去调检索参数，而真正的问题在标注里。
      这里把"标了不存在的块"和"块不属于它声称的文档"都变成硬失败。
    """
    errs: list[str] = []
    seen: set[str] = set()
    counts: dict[str, int] = defaultdict(int)

    for g in golden:
        gid, typ = g["id"], g["type"]
        counts[typ] += 1
        if gid in seen:
            errs.append(f"{gid}: id 重复")
        seen.add(gid)

        for cid in g["gold_chunk_ids"]:
            if cid not in corpus:
                errs.append(f"{gid}: gold chunk {cid} 不在语料里")
            elif corpus[cid]["document_id"] not in g["gold_doc_ids"]:
                errs.append(
                    f"{gid}: chunk {cid} 属于 doc {corpus[cid]['document_id']}，"
                    f"但 gold_doc_ids={g['gold_doc_ids']}"
                )
        if typ != "unanswerable" and not g["gold_chunk_ids"]:
            errs.append(f"{gid}: 可回答题却没有 gold chunk")
        if typ == "unanswerable" and g["gold_chunk_ids"]:
            errs.append(f"{gid}: 不可回答题不该有 gold chunk")

    out = [f"共 {len(golden)} 题：" + "，".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    if errs:
        out.append(f"\n✗ {len(errs)} 处标注错误：")
        out += [f"  - {e}" for e in errs]
    else:
        out.append("✓ 标注自洽（gold chunk 全部存在，且与 gold doc 一致）")
    print("\n".join(out))
    return 1 if errs else 0


# ======================================================================
# retrieval：recall@k / MRR + 消融
# ======================================================================
async def _retrieve(client: httpx.AsyncClient, q: str, cfg: dict[str, bool]) -> dict:
    r = await client.post(f"{API}/api/v1/retrieve",
                          json={"query": q, "top_k": TOP_K, **cfg}, timeout=60)
    r.raise_for_status()
    return r.json()


def _score_one(g: dict, chunks: list[dict]) -> dict:
    """单题的 chunk 级与 doc 级指标。

    MRR 取的是**第一个**命中 gold 的位置。chunk 级用 chunk_id 命中，
    doc 级用 document_id 命中 —— 后者更宽松，正是"找对论文就行"的语义。
    """
    gold_c = set(g["gold_chunk_ids"])
    gold_d = set(g["gold_doc_ids"])

    cids = [c["chunk_id"] for c in chunks]
    dids = [c["document_id"] for c in chunks]

    c_rank = next((i + 1 for i, c in enumerate(cids) if c in gold_c), None)
    d_rank = next((i + 1 for i, d in enumerate(dids) if d in gold_d), None)
    return {
        "id": g["id"],
        "type": g["type"],
        "chunk_rank": c_rank,
        "doc_rank": d_rank,
        **{f"chunk_hit@{k}": bool(c_rank and c_rank <= k) for k in (1, 3, 5, 10)},
        **{f"doc_hit@{k}": bool(d_rank and d_rank <= k) for k in (1, 3, 5, 10)},
    }


def _agg(rows: list[dict]) -> dict:
    """聚合单题结果。

    ★ MRR 的分母是**全部题目**，没命中的记 0，不是"只在命中题上取平均"。
      后者会把低召回的配置抬高：一个只召回了 5 道题但都排第一的配置，
      和召回了 20 道题、名次参差的配置，算出来的"MRR"前者反而更高 ——
      那个数就不再表示"整体上用户要往下翻几条"，完全没法比。
      上一版就是这么写的，`-dense` 因此显示 MRR=0.690 > full 的 0.512，
      而它的 recall@5 只有 0.125。指标必须和 recall 同向才有意义。
    """
    n = len(rows) or 1
    out = {}
    for prefix in ("chunk", "doc"):
        for k in (1, 3, 5, 10):
            out[f"{prefix}_recall@{k}"] = sum(r[f"{prefix}_hit@{k}"] for r in rows) / n
        out[f"{prefix}_mrr"] = sum(
            (1 / r[f"{prefix}_rank"]) if r[f"{prefix}_rank"] else 0.0 for r in rows
        ) / n
    return out


async def cmd_retrieval(golden: list[dict]) -> dict:
    answerable = [g for g in golden if g["type"] != "unanswerable"]
    results: dict[str, dict] = {}

    async def one(client: httpx.AsyncClient, g: dict, cfg: dict[str, bool],
                  sem: asyncio.Semaphore) -> dict:
        async with sem:
            data = await _retrieve(client, g["question"], cfg)
        return _score_one(g, data["chunks"])

    async with httpx.AsyncClient() as client:
        for name, cfg in CONFIGS.items():
            sem = asyncio.Semaphore(CONCURRENCY)
            rows = await asyncio.gather(
                *(one(client, g, cfg, sem) for g in answerable)
            )
            results[name] = {"config": cfg, "rows": list(rows), **_agg(list(rows))}
            print(f"  {name:12} chunk_r@5={results[name]['chunk_recall@5']:.3f} "
                  f"doc_r@5={results[name]['doc_recall@5']:.3f} "
                  f"chunk_mrr={results[name]['chunk_mrr']:.3f}")

    return results


# ======================================================================
# chat：拒答准确率 + 引用真实率
# ======================================================================
async def cmd_chat(golden: list[dict], corpus: dict[int, dict]) -> dict:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(g: dict) -> dict:
            async with sem:
                r = await client.post(
                    f"{API}/api/v1/chat",
                    json={"question": g["question"], "top_k": 5}, timeout=180,
                )
                r.raise_for_status()
                d = r.json()
            # 引用真实率：quote 必须能在**它自己声称的源块**里逐字找到。
            # 只判断"引用非空"是不够的 —— 那等于不测。
            checked = grounded = 0
            for c in d.get("citations") or []:
                src = corpus.get(c["chunk_id"])
                if not src:
                    continue
                checked += 1
                q = " ".join((c.get("quote") or "").split())
                if q and q in " ".join(src["content"].split()):
                    grounded += 1
            return {
                "id": g["id"], "type": g["type"], "route": d.get("route") or "",
                "n_citations": len(d.get("citations") or []),
                "verified": bool(d.get("verified")),
                "retries": int(d.get("retries") or 0),
                "cites_checked": checked, "cites_grounded": grounded,
                "answer_head": (d.get("answer") or "")[:80],
            }

        rows = await asyncio.gather(*(one(g) for g in golden))

    unans = [r for r in rows if r["type"] == "unanswerable"]
    ans = [r for r in rows if r["type"] != "unanswerable"]

    refused_ok = sum(1 for r in unans if r["route"] == "refuse"
                     and r["n_citations"] == 0)
    false_refuse = sum(1 for r in ans if r["route"] == "refuse")
    checked = sum(r["cites_checked"] for r in rows)
    grounded = sum(r["cites_grounded"] for r in rows)

    summary = {
        # ★ 分母是"不可回答题数"：问的是"该拒的有没有拒"。
        #   和"答对率"分开报 —— 一个只会拒答的系统 refuse 准确率是 100%，
        #   必须配上 false_refuse 一起看才不会被刷分。
        "refuse_accuracy": refused_ok / (len(unans) or 1),
        "refuse_n": len(unans),
        "false_refuse_rate": false_refuse / (len(ans) or 1),
        "false_refuse_n": false_refuse,
        "citation_groundedness": (grounded / checked) if checked else 0.0,
        "citations_checked": checked,
        "rows": rows,
    }
    print(f"  弃答准确率      {summary['refuse_accuracy']:.3f} "
          f"({refused_ok}/{len(unans)})")
    print(f"  误拒率          {summary['false_refuse_rate']:.3f} "
          f"({false_refuse}/{len(ans)})")
    print(f"  引用逐字真实率  {summary['citation_groundedness']:.3f} "
          f"({grounded}/{checked})")
    return summary


# ======================================================================
def _report(ret: dict, chat: dict, outdir: Path) -> Path:
    lines = ["# 评测结果（自动生成，勿手改）", "",
             f"检索深度 top_k={TOP_K}，消融每组只改一个开关。", ""]

    if ret:
        lines += ["## 检索指标（50 题中 40 道可回答题）", "",
                  "| 配置 | chunk R@1 | chunk R@5 | chunk R@10 | chunk MRR | doc R@5 | doc MRR |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in ret.items():
            lines.append(
                f"| `{name}` | {r['chunk_recall@1']:.3f} | {r['chunk_recall@5']:.3f} | "
                f"{r['chunk_recall@10']:.3f} | {r['chunk_mrr']:.3f} | "
                f"{r['doc_recall@5']:.3f} | {r['doc_mrr']:.3f} |"
            )
        lines += ["", "> chunk 级 = 命中标注的那个块；doc 级 = 只要求命中那篇论文。", ""]

    if chat:
        lines += ["## 生成与拒答", "",
                  "| 指标 | 值 | 说明 |", "|---|---|---|",
                  f"| 弃答准确率 | {chat['refuse_accuracy']:.3f} | "
                  f"{chat['refuse_n']} 道不可回答题中，route=refuse 且引用为空的比例 |",
                  f"| 误拒率 | {chat['false_refuse_rate']:.3f} | "
                  f"40 道可回答题中被拒的比例（与上一行一起看才有意义）|",
                  f"| 引用逐字真实率 | {chat['citation_groundedness']:.3f} | "
                  f"{chat['citations_checked']} 条引用中，quote 能在其声称的源块里逐字找到的比例 |",
                  ""]

        # ★ 误拒率必须和检索命中率交叉看，否则会被读成"系统太保守"。
        #   如果被拒的题本来就没检索到答案，拒答是**正确**行为 ——
        #   它对着一堆不相关片段硬编一段话才是错的。分开报，才不会被误读。
        if ret and "full" in ret:
            hit5 = {r["id"]: r["chunk_hit@5"] for r in ret["full"]["rows"]}
            fr = [r for r in chat["rows"]
                  if r["type"] != "unanswerable" and r["route"] == "refuse"]
            justified = sum(1 for r in fr if not hit5.get(r["id"]))
            lines += [
                f"其中 **{justified}/{len(fr)}** 道误拒题，在 `full` 配置下检索",
                "`chunk_hit@5` 本来就是 False —— 也就是**答案压根没被召回**，",
                f"此时拒答是正确行为。真正「有证据却拒答」的有 {len(fr) - justified} 道。",
                "",
            ]

    p = outdir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "retrieval", "chat", "all"])
    args = ap.parse_args()

    corpus = load_corpus()
    golden = resolve_evidence(load_golden(), corpus)

    if args.cmd in ("check", "all"):
        rc = cmd_check(golden, corpus)
        if rc and args.cmd == "check":
            return rc

    outdir = EVAL / "results"
    outdir.mkdir(exist_ok=True)

    ret = await cmd_retrieval(golden) if args.cmd in ("retrieval", "all") else {}
    chat = await cmd_chat(golden, corpus) if args.cmd in ("chat", "all") else {}

    if ret:
        (outdir / "retrieval.json").write_text(
            json.dumps(ret, ensure_ascii=False, indent=1), encoding="utf-8")
    if chat:
        (outdir / "chat.json").write_text(
            json.dumps(chat, ensure_ascii=False, indent=1), encoding="utf-8")

    # ★ 只跑一个阶段时，把**上一次**另一阶段的结果读回来一起出报告。
    #   否则 `retrieval` 和 `chat` 会互相覆盖 report.md —— 跑完 chat 之后
    #   报告里就只剩拒答那三行，检索那张表凭空消失，看起来像"没测"。
    if args.cmd != "all":
        if not ret and (outdir / "retrieval.json").exists():
            ret = json.loads((outdir / "retrieval.json").read_text(encoding="utf-8"))
            print("  （检索结果沿用上次的 eval/results/retrieval.json）")
        if not chat and (outdir / "chat.json").exists():
            chat = json.loads((outdir / "chat.json").read_text(encoding="utf-8"))
            print("  （生成结果沿用上次的 eval/results/chat.json）")
    if ret or chat:
        print(f"\n报告 -> {_report(ret, chat, outdir)}")
    return 0


if __name__ == "__main__":
    # Windows 控制台默认 GBK，直接 print 中文与 ✓/✗ 会 UnicodeEncodeError。
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
