"""多轮追问探针：找出"基于上一轮回答继续问"时会在哪里崩。

★ 为什么要单独探这一组：
  单轮评测（run_eval.py）把每道题当**独立**问题问，所以它测不出上下文问题。
  而真实用法是连着问的 —— `run_eval.py` 的 0.500 覆盖率在这个场景下不成立。

五种追问形态，各自会打在不同的环节上：
  rephrase  —— 换个说法问同一件事（考验改写）
  detail    —— 追问上一轮**答案正文里**的细节（考验 history 截断）
  combine   —— 把前两轮的两件事放在一起比（考验多轮累积）
  why       —— 极省略的追问，只有"为什么"（考验改写下限）
  number    —— 追问数值/参数（语料里常是表格碎片，最容易丢）

★ 线程卫生：thread_id 带一个运行标签（argv[1]），默认 "a"。
  **同一标签跑第二次会读到上一次的历史** —— 上一轮的拒答会被当成上下文，
  于是"第二轮"其实是在一个有污染的多轮会话里问的，结果不可复现。
  换标签就是干净的一轮。要重复验证就换标签，不要原地重跑。

用法：python eval/probe_multiturn.py [标签]
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
API = "http://127.0.0.1:8000"

# 每个场景：先问 seed，再连着追问。最后一题才是判据。
SCENARIOS: list[dict] = [
    {
        "id": "rephrase",
        "turns": [
            "GBSVM 里粒度球分类器的时间复杂度能降到多少？",
            "那它大概要多久算完？",
        ],
    },
    {
        "id": "detail",
        "turns": [
            "RD-IFTSVM 的时间和空间复杂度分别是什么？",
            "刚才说的 n1 和 n2 分别代表什么？",
        ],
    },
    {
        # ★ 场景必须用**语料真的支持**的比较（对应 golden 的 m02）。
        #   最初这里写的是"GBSVM 和 CKA-FSVM 哪个更快"——两篇论文各说各的，
        #   语料里没有任何可比的数字，系统拒答是**正确**的。而当时它"答出来了"，
        #   是因为 grader 拿 rewrite 后的 A-only 关键词 query 去判，问什么答什么，
        #   于是生成了一段只讲 A 的答案冒充比较。换成 m02 这种两边都有明确
        #   复杂度结论的题，才测得出"多轮比较"这件事本身成不成立。
        "id": "combine",
        "turns": [
            "GBSVM 的时间复杂度是多少？",
            "SFFSVM 的时间复杂度又是多少？",
            "这两个哪个更低？",
        ],
    },
    {
        "id": "why",
        "turns": [
            "Pin-GBTSVM 缺少什么统计学习理论的基础？",
            "为什么这是个问题？",
        ],
    },
    {
        # ★ 同样换成语料真的有的数值参数。最初写的是"ISFFSVM 里 a 取多少 / 那 β 呢"——
        #   实测 ISFFSVM 全文 54138 字符里 `β` 出现 **0 次**（β 是 SFFSVM 的参数，
        #   见 golden s17）。问一个语料里不存在的参数，拒答是**正确**的，
        #   把它当成 bug 去"修"只会把 grader 越改越松。
        "id": "number",
        "turns": [
            "SFFSVM 里的平滑参数 β 的取值范围是什么？",
            "那它在实验里取的是哪个值？",
        ],
    },
]


async def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "a"
    out = []
    async with httpx.AsyncClient(timeout=240) as cli:
        for sc in SCENARIOS:
            tid = f"probe-{tag}-{sc['id']}"
            print(f"── {sc['id']}")
            for i, q in enumerate(sc["turns"], 1):
                r = await cli.post(f"{API}/api/v1/chat",
                                   json={"question": q, "thread_id": tid})
                d = r.json()
                dg = d.get("diagnostics") or {}
                queries = dg.get("queries") or {}
                rec = {
                    "scenario": sc["id"], "turn": i, "q": q,
                    "route": d.get("route"),
                    "n_citations": len(d.get("citations") or []),
                    "answer_len": len(d.get("answer") or ""),
                    "answer_head": (d.get("answer") or "")[:120],
                    # ★ 改写后实际拿去检索的问题。这一列是判断"上下文有没有接上"的
                    #   唯一直接证据 —— 如果它还是"那它呢"，后面必然全崩。
                    "retrieval_query": queries.get("orig"),
                    "n_chunks": len(dg.get("recall") or {}),
                    "fresh_returned": dg.get("fresh_returned"),
                    "grade_reason": (d.get("grade_reason") or "")[:150],
                }
                out.append(rec)
                flag = "拒答" if rec["route"] == "refuse" else "    "
                print(f"   [{flag}] 轮{i} route={rec['route']:<9} "
                      f"引用={rec['n_citations']:<2} 检索用问={rec['retrieval_query']}")
            print()

    (EVAL / "results" / "multiturn.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 只有最后一轮才算"这个追问成没成"——前面几轮是铺垫
    last = [r for r in out if r["turn"] == len(
        next(s["turns"] for s in SCENARIOS if s["id"] == r["scenario"]))]
    refused = [r for r in last if r["route"] == "refuse"]
    print("=" * 72)
    print(f"追问轮共 {len(last)} 个，拒答 {len(refused)} 个")
    for r in refused:
        print(f"   ✗ [{r['scenario']}] {r['q']}  ->  检索用问：{r['retrieval_query']}")
    print("\n-> eval/results/multiturn.json")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
