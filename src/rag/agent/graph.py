"""LangGraph 图装配。

拓扑：

    START → prepare → retrieve → grade ─┬─ sufficient ──────────────→ generate → verify → END
                 ↑                      └─ insufficient → rewrite ─┬─ 还有重试 → retrieve
                 └───────────────────────────────────────────────┘
                                                                    ├─ 用尽且证据不足 → refuse → END
                                                                    └─ 用尽但判过 sufficient → generate

★ 为什么用图而不是一条直线 if-else：
   这条链路有**环**（rewrite → retrieve）。用 if-else 写会变成嵌套循环 +
   手工维护的重试计数，而且中间状态无处存放 —— 一旦要加"人工审核""工具调用"
   就得推倒重来。图把控制流显式化，checkpoint 还能免费拿到断点续跑。

★ checkpointer 的作用：thread_id 相同时，LangGraph 会自动把该会话的历史状态
   载入。多轮对话因此不需要我们自己拼接 history —— 但**上下文改写仍需要 history**，
   所以 conversation 历史还是单独存了一份（见 repository 的 messages 表）。
"""

from __future__ import annotations

from typing import Any

from rag.agent.nodes import (
    make_generate_node,
    make_grade_node,
    make_prepare_node,
    make_refuse_node,
    make_retrieve_node,
    make_rewrite_node,
    verify_node,
)
from rag.agent.state import AgentState
from rag.core.logging import get_logger
from rag.services.retrieval import RetrievalService

logger = get_logger(__name__)


def build_graph(
    *,
    retrieval: RetrievalService,
    llm: Any = None,
    checkpointer: Any = None,
    top_k: int = 5,
    max_retries: int = 2,
):  # noqa: ANN201
    """编译并返回可执行的图。"""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)

    builder.add_node("prepare", make_prepare_node(llm))
    builder.add_node("retrieve", make_retrieve_node(retrieval, top_k=top_k))
    builder.add_node("grade", make_grade_node(llm))
    builder.add_node("rewrite", make_rewrite_node(llm))
    builder.add_node("generate", make_generate_node(llm))
    builder.add_node("refuse", make_refuse_node())
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "retrieve")
    builder.add_edge("retrieve", "grade")

    # grade 决定"直接答"还是"改写后重查"
    builder.add_conditional_edges(
        "grade",
        _route_of,
        {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"},
    )
    # rewrite 决定"再查一轮"还是"用手上的料作答 / 拒答"
    builder.add_conditional_edges(
        "rewrite",
        _route_of,
        {"retrieve": "retrieve", "generate": "generate", "refuse": "refuse"},
    )

    builder.add_edge("generate", "verify")
    builder.add_edge("verify", END)
    # refuse 直达 END：不进 generate（会被无关 chunk 诱惑着硬编），
    # 也不进 verify（没有引用可校验，拒答本身就是可信的）
    builder.add_edge("refuse", END)

    return builder.compile(checkpointer=checkpointer)


def _route_of(state: AgentState) -> str:
    """路由函数。

    ★ 必须返回**已声明的**字面量之一。返回未声明的值 LangGraph 会抛
      `InvalidUpdateError`，而且错误信息只说"未知分支"，不告诉你是哪个节点发出的。
      所以这里对未知值兜底成 "generate"，让流程能走完并留下 error 记录。
    """
    route = state.get("route", "generate")
    if route not in ("retrieve", "generate", "rewrite", "refuse"):
        logger.warning("agent.unknown_route", route=route)
        return "generate"
    return route


async def run_agent(
    graph,  # noqa: ANN001
    question: str,
    *,
    thread_id: str,
    tenant_id: int = 1,
    history: list[dict[str, str]] | None = None,
    max_retries: int = 2,
    recursion_limit: int = 50,
) -> dict[str, Any]:
    """执行一次问答。

    ★ recursion_limit 不是可有可无的：图里有环，一旦路由写错就会无限循环，
      表现为请求挂死 + Postgres 里 checkpoint 疯狂增长。这个参数是硬保险。
    """
    from rag.agent.state import initial_state

    state = initial_state(
        question, thread_id=thread_id, tenant_id=tenant_id,
        history=history, max_retries=max_retries,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    return await graph.ainvoke(state, config=config)
