"""改写提示词对照台：不动图、不动检索，只比不同提示词下改写出来的问题。

★ 为什么要单独测提示词：
  改写节点在图的最上游，它错了后面全错 —— 但这个错误在最终指标里只表现为
  "拒答"，看不出是改写干的。把改写单独拎出来，才能确定是提示词的问题
  还是检索的问题。

★ 判据不是"像不像人话"，而是**有没有偷换技术概念**：
  实测里 "时间复杂度" 被改写成 "训练时间/运行耗时/秒/分钟"，
  意思变了、语料里也没有，于是必然检索失败 -> 拒答。

用法：python eval/probe_rewrite.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.core.config import get_settings  # noqa: E402
from rag.providers import build_llm  # noqa: E402

# 当前线上用的提示词（和 src/rag/agent/prompts.py 保持一致）
CURRENT = """把用户的最新提问改写成一个**不依赖对话历史**的独立问题。

对话历史：
{history}

最新提问：{question}

要求：
1. 补全代词和省略的主语。例如历史里在聊"Milvus 的索引"，提问"它占多少内存"→
   改写为"Milvus 的索引占多少内存"。
2. 如果最新提问本身已经完整，原样返回。
3. 只输出改写后的问题，不要解释。
"""

# 候选：显式禁止"关键词堆"和"偷换概念"，并给出正反例。
CANDIDATE = """把用户的最新提问改写成一个**不依赖对话历史**的独立问题。

对话历史：
{history}

最新提问：{question}

要求：
1. 只做一件事：把代词和省略的部分补全，指向历史里已经出现过的那个对象。
2. **保持最新提问原本所问的技术概念不变**。历史里在聊"时间复杂度"，
   提问"那它要多久算完"指的是"时间复杂度是多少"，**不是**"训练耗时多少秒"。
   不要把"时间复杂度"换成"运行时间""耗时""秒""分钟"这类意思不同的词。
3. **必须输出一个完整的问句**，不要输出关键词列表、不要用空格堆词。
4. 不要引入历史里没出现过的新概念、新词。
5. 如果最新提问本身已经完整，原样返回。
6. 只输出改写后的问题，不要解释。

反例（不要这样写）：
  历史在聊 GBSVM 的时间复杂度，提问"那它大概要多久算完？"
  ✗ 错误：GBSVM 训练时间 实测 运行耗时 秒 分钟      ← 关键词堆，且偷换了概念
  ✓ 正确：GBSVM 里粒度球分类器的时间复杂度大概是多少？
"""

HISTORY = """user: GBSVM 里粒度球分类器的时间复杂度能降到多少？
assistant: 根据现有资料，GBSVM 中粒度球分类器的时间复杂度可以降到接近 O(N) [1]。具体来说，资料给出的推导逻辑是：粒度球的数量几乎可以看作一个小常数，因此粒度球生成的时间复杂度等于生成最大粒度球的时间复杂度；二均值聚类通常收敛速度快，因此可以视为近似线性算法。"""

# 覆盖几种真实追问：换说法、省略、指代细节、跨轮对比
CASES = [
    "那它大概要多久算完？",
    "它的空间复杂度呢？",
    "为什么能降到这么低？",
    "这个方法用在什么数据上？",
]


async def main() -> int:
    llm = build_llm(get_settings())
    rows = []
    for name, tpl in (("current", CURRENT), ("candidate", CANDIDATE)):
        print(f"── {name}")
        for q in CASES:
            try:
                out = await llm.acomplete(
                    [{"role": "user", "content": tpl.format(history=HISTORY, question=q)}],
                    temperature=0.0, max_tokens=256,
                )
            except Exception as exc:  # noqa: BLE001
                out = f"<失败 {type(exc).__name__}: {exc}>"
            out = " ".join((out or "").split())
            rows.append({"prompt": name, "q": q, "rewritten": out})
            print(f"   问：{q}")
            print(f"   改：{out}")
        print()

    (ROOT / "eval" / "results" / "rewrite_ab.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> eval/results/rewrite_ab.json")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
