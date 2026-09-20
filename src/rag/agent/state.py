"""Agent 状态定义。

★ 状态设计的两条原则：

1. **只放跨节点必须传递的数据**。检索到的正文（chunks）放进去，因为 generate
   和 verify 都要用；而"当前时间"这种每个节点自己取的东西不放。
   状态越大，checkpoint 越大 —— 每次节点跳转都要序列化一次写 Postgres。

2. **列表字段用 `operator.add` 归约的要谨慎**。这里 `citations` 用累加，
   因为 verify 阶段会往里追加校验结果；其余字段一律**覆盖**，
   `chunks` 字段本身仍采用覆盖语义；retrieve 节点会显式地对多轮证据去重、
   限长后再整体覆盖。这样累积策略可审计，也不会由 reducer 无限增长。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Route = Literal["retrieve", "generate", "rewrite", "refuse", "end"]


class AgentState(TypedDict, total=False):
    # ---- 输入 ----
    question: str                    # 用户原始问题，永不改写（引用与日志都用它）
    query: str                       # 当前用于检索的查询（可能被 rewrite 改写）
    # ★ 消解了上下文、但不带检索改写的"用户到底在问什么"。
    #   prepare 写好后再也不动（rewrite 只改 query）。
    #
    #   为什么必须单独存一个：grader 要判"现有证据能不能回答用户的问题"，
    #   而它是唯一正确的判据 —— 用 question 会读到"那它大概要多久算完？"
    #   这种脱离上下文读不懂的话；用 query 又会被 rewrite 换成关键词表，
    #   把"A 和 B 哪个快"的比较意图丢掉，于是只答一半还判 sufficient。
    standalone: str
    thread_id: str
    tenant_id: int
    history: list[dict[str, str]]    # 裁剪过的对话历史

    # ---- 检索 ----
    chunks: list[dict[str, Any]]     # RetrievedChunk 的序列化形式（可进 checkpoint）
    retrieval_diagnostics: dict[str, Any]

    # ---- 控制流 ----
    retries: int                     # 已重写次数
    max_retries: int
    route: Route
    grade_reason: str                # 判定"证据不足"的理由，便于排查
    # ★ grader 的原始结论（"sufficient" / "insufficient" / ""）。
    #   重试次数用尽时必须靠它来决定是"硬着头皮生成"还是"拒答" ——
    #   曾经用的是 `route = "generate" if chunks else "refuse"`，但稠密通道
    #   一次召回 50 条，chunks 永远非空，于是"证据不足"被静默改写成了生成。
    grade_verdict: str
    error: str | None

    # ---- 输出 ----
    answer: str
    citations: Annotated[list[dict[str, Any]], operator.add]
    verified: bool                   # 引用是否全部通过逐字校验
    usage: dict[str, Any]


def initial_state(
    question: str,
    *,
    thread_id: str,
    tenant_id: int = 1,
    history: list[dict[str, str]] | None = None,
    max_retries: int = 2,
) -> AgentState:
    return AgentState(
        question=question,
        query=question,
        standalone=question,
        thread_id=thread_id,
        tenant_id=tenant_id,
        history=history or [],
        chunks=[],
        retrieval_diagnostics={},
        retries=0,
        max_retries=max_retries,
        route="retrieve",
        grade_reason="",
        grade_verdict="",
        error=None,
        answer="",
        citations=[],
        verified=False,
        usage={},
    )
