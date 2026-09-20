"""Agent 图的节点实现。

★ 依赖注入而不是模块级单例：每个节点做成闭包工厂（make_xxx_node），
   把 service 传进去。这样测试里可以塞 fake，不需要 monkeypatch 全局变量。

★ 所有节点都是 async 且**不抛异常**：失败时写 state["error"] 并给出降级路由。
   LangGraph 里某个节点抛异常会中断整张图，而 checkpoint 停在中间状态，
   排查起来非常痛苦。宁可降级也不要中断。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from rag.agent.prompts import (
    CITATION_PATTERN,
    CONTEXTUALIZE_PROMPT,
    GRADE_PROMPT,
    RAG_SYSTEM_PROMPT,
    REFUSE_ANSWER,
    REWRITE_PROMPT,
)
from rag.agent.state import AgentState
from rag.core.logging import get_logger
from rag.providers.base import is_enabled
from rag.services.retrieval import RetrievalService, RetrievedChunk

logger = get_logger(__name__)

_CITATION_RE = re.compile(CITATION_PATTERN)


# ======================================================================
# 序列化：状态要进 checkpoint，必须是可 JSON 序列化的
# ======================================================================
def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "score": chunk.score,
        "document_id": chunk.document_id,
        "doc_title": chunk.doc_title,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "node_type": chunk.node_type,
        "rank": chunk.rank,
        "label": chunk.label,
        "dense_rank": chunk.dense_rank,
        "sparse_rank": chunk.sparse_rank,
        "rerank_score": chunk.rerank_score,
        "dense_score": chunk.dense_score,
    }


def dict_to_chunk(payload: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        content=payload["content"],
        score=payload.get("score", 0.0),
        document_id=payload.get("document_id", 0),
        doc_title=payload.get("doc_title", ""),
        section_path=payload.get("section_path", ""),
        page_start=payload.get("page_start", 0),
        page_end=payload.get("page_end", 0),
        node_type=payload.get("node_type", "paragraph"),
        rank=payload.get("rank", 0),
        dense_rank=payload.get("dense_rank"),
        sparse_rank=payload.get("sparse_rank"),
        rerank_score=payload.get("rerank_score"),
        dense_score=payload.get("dense_score"),
    )


# ======================================================================
# 上下文改写的两个文本工具
# ======================================================================
_SENT_END = "。！？!?；;\n"
# 问句标记：中文疑问词/语气词 + 疑问标点。改写结果里带任意一个就不算词表。
_INTERROGATIVE = re.compile(
    r"[?？]|多少|是什么|什么是|为什么|如何|怎么|怎样|哪个|哪些|哪一|"
    r"是否|能否|可以吗|吗|呢|分别|区别|差别"
)
# 空格分隔的、含中文的片段
_CJK_CHUNK = re.compile(r"[^\s]*[一-鿿][^\s]*")


def _clip(text: str, limit: int) -> str:
    """截断到 limit 字符，尽量切在句子边界上。

    ★ 为什么不能直接 `text[:limit]`：实测里 392 字的回答被从
      "补充一点相关但不同的复杂度信息：对于 GBSVM 的最大时间复杂" 中间切断，
      半句话进了提示词。模型看到的是个断句，容易被带偏；而完整的句子
      既更短又更可用。
    """
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(ch) for ch in _SENT_END)
    # 找不到句末标点（或它太靠前）时按原样截断，不要丢掉几乎全部内容
    return head[:cut + 1] if cut >= limit // 2 else head


def _looks_like_keyword_salad(rewritten: str) -> bool:
    """改写结果是不是一坨空格分隔的关键词，而不是一句话。

    判据保守到只打词表：**既没有疑问标记、又有 ≥4 个空格分隔的中文片段**。
    正常的改写（"GBSVM 的粒度球分类器大概要多久算完？"）只有 2 段且有"？"，
    不会被误伤。宁可漏放，不可误杀 —— 误杀会把一个好改写换成原问题。
    """
    if _INTERROGATIVE.search(rewritten):
        return False
    return len(_CJK_CHUNK.findall(rewritten)) >= 4


def make_prepare_node(llm) -> Any:  # noqa: ANN001
    """把"它占多少内存"这类依赖上下文的问题改写成独立问题。

    ★ 只在**有历史**时才调用 LLM。首轮提问本来就完整，白花一次调用。
    """

    async def prepare(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history") or []
        # 首轮提问本来就完整；未配置 LLM 时空实现会抛异常，也要跳过
        if not history or not is_enabled(llm):
            return {"query": question, "standalone": question}

        history_text = "\n".join(
            f"{m.get('role', 'user')}: {_clip(m.get('content', ''), 300)}"
            for m in history[-6:]
        )
        try:
            rewritten = await llm.acomplete(
                [{"role": "user", "content": CONTEXTUALIZE_PROMPT.format(
                    history=history_text, question=question)}],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:
            # 改写失败不影响主流程 —— 用原问题检索，最多是召回差一点
            logger.warning("agent.contextualize_failed", error=str(exc))
            return {"query": question, "standalone": question}

        rewritten = rewritten.strip().strip('"').strip("「」")
        # ★ 长度防线：改写失败时模型可能返回一整段解释，直接拿去检索会更差
        if not rewritten or len(rewritten) > len(question) * 4 + 50:
            return {"query": question, "standalone": question}
        # ★ 关键词堆防线：模型偶尔会把问题压成空格分隔的词表
        #   （实测出现过 "GBSVM 训练时间 实测 运行耗时 秒 分钟"）。
        #   词表既不是问句、又和用户实际问的东西对不上，拿去检索必然更差，
        #   此时退回原问题至少保留了用户的语言。判据故意保守：只打词表。
        if _looks_like_keyword_salad(rewritten):
            logger.warning("agent.contextualize_keyword_salad", rewritten=rewritten[:80])
            return {"query": question, "standalone": question}
        return {"query": rewritten, "standalone": rewritten}

    return prepare


def make_retrieve_node(retrieval: RetrievalService, *, top_k: int) -> Any:  # noqa: ANN001
    async def retrieve(state: AgentState) -> dict[str, Any]:
        try:
            result = await retrieval.retrieve(
                state["query"], tenant_id=state.get("tenant_id", 1), top_k=top_k
            )
        except Exception as exc:
            logger.exception("agent.retrieve_failed")
            return {"chunks": [], "error": f"检索失败：{type(exc).__name__}",
                    "retrieval_diagnostics": {}}

        fresh = [chunk_to_dict(c) for c in result.chunks]
        chunks = fresh
        if state.get("retries", 0) > 0 and state.get("chunks"):
            # 改写通常只聚焦 grader 指出的一个缺口。多跳/比较问题若直接覆盖
            # 第一轮结果，会出现“补到了 B、却把已经找到的 A 丢掉”。保留首轮
            # 高相关证据，再追加新证据；限制为 2×top_k，避免无关重试无限膨胀。
            chunks = _merge_retrieval_rounds(
                state["chunks"], fresh, limit=top_k * 2
            )
        diagnostics = dict(result.diagnostics)
        diagnostics["fresh_returned"] = len(fresh)
        diagnostics["accumulated_returned"] = len(chunks)
        return {"chunks": chunks, "retrieval_diagnostics": diagnostics}

    return retrieve


def _merge_retrieval_rounds(
    previous: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """合并改写前后的证据，按 chunk_id 去重并重编引用序号。"""
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for chunk in [*previous, *fresh]:
        chunk_id = int(chunk.get("chunk_id") or 0)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(dict(chunk))
        if len(merged) >= limit:
            break
    for rank, chunk in enumerate(merged, start=1):
        chunk["rank"] = rank
    return merged


def make_grade_node(llm) -> Any:  # noqa: ANN001
    """判断证据是否充分。没有 LLM 时直接判 sufficient（跳过该节点）。"""

    async def grade(state: AgentState) -> dict[str, Any]:
        chunks = state.get("chunks") or []
        if not chunks:
            return {"route": "rewrite", "grade_reason": "检索结果为空",
                    "grade_verdict": "insufficient"}

        if not is_enabled(llm):
            # 无 LLM 的降级路径：有召回就生成（此时生成也会走抽取式兜底）
            return {"route": "generate", "grade_reason": "未配置 LLM，跳过判定",
                    "grade_verdict": ""}

        context = _render_context(chunks)
        # ★ 判定用 standalone（"用户到底在问什么"），既不是 question 也不是 query。
        #
        #   - 不能用 question：追问的原话是"那它大概要多久算完？"，"它"是什么
        #     只有历史知道，而 GRADE_PROMPT 里**不含历史**（故意的：检索到的
        #     资料才是判据）。拿原话去判，模型读不懂问题，只能判 insufficient。
        #   - 不能用 query：rewrite 节点会把它换成关键词表，比较题的
        #     "A 和 B 哪个快"就被压成"A 的指标词汇"，于是只答一半还判 sufficient。
        #
        #   实测：改写正确、检索命中（top-1 rerank 0.994），却在 grader 这里被判
        #   不足 -> 进 rewrite 把好 query 改坏 -> 拒答。判据必须锚在用户意图上。
        grade_question = state.get("standalone") or state.get("query") or state["question"]
        try:
            verdict = await llm.astructured(
                [{"role": "user", "content": GRADE_PROMPT.format(
                    question=grade_question, context=context)}],
                _GradeVerdict,
            )
            route = "generate" if verdict.verdict == "sufficient" else "rewrite"
            return {"route": route, "grade_reason": verdict.reason,
                    "grade_verdict": verdict.verdict}
        except Exception as exc:
            # ★ 判定失败时倾向于"生成"而不是"重试"：
            #   重试要再花一轮检索 + LLM 调用，而生成至少能给出有引用的答案，
            #   由用户判断是否可信。
            logger.warning("agent.grade_failed", error=str(exc))
            return {"route": "generate", "grade_reason": f"判定失败，降级生成：{exc}",
                    "grade_verdict": ""}

    return grade


def _exhausted_route(state: AgentState) -> str:
    """重试用尽后的去向：grader 判过 insufficient 就拒答，否则带着现有证据生成。

    ★ 判据必须是 **grade_verdict**，不能是 `state["chunks"]`。
      稠密通道一次召回 top_k*10 条，chunks 几乎永远非空 —— 用它做判断等于
      永远走 generate，grader 的"证据不足"被静默吞掉，用户看到的就是
      "重试了两轮还是硬生成一段拒答文字，并且挂着 5 条无关引用"。
    """
    if state.get("grade_verdict") == "insufficient":
        return "refuse"
    return "generate"


def make_rewrite_node(llm) -> Any:  # noqa: ANN001
    async def rewrite(state: AgentState) -> dict[str, Any]:
        retries = state.get("retries", 0)
        max_retries = state.get("max_retries", 2)

        if retries >= max_retries:
            return {"route": _exhausted_route(state), "retries": retries,
                    "grade_reason": f"已达最大重试次数 {max_retries}"}

        if not is_enabled(llm):
            return {"route": _exhausted_route(state), "retries": retries + 1}

        try:
            new_query = await llm.acomplete(
                # ★ question 传 standalone：追问的原话（"这两个方法哪个更快？"）
                #   脱离上下文读不懂，改写模型会猜错对象。
                [{"role": "user", "content": REWRITE_PROMPT.format(
                    question=state.get("standalone") or state["question"],
                    query=state["query"],
                    reason=state.get("grade_reason", ""))}],
                temperature=0.3,
                max_tokens=128,
            )
        except Exception as exc:
            logger.warning("agent.rewrite_failed", error=str(exc))
            return {"retries": retries + 1, "route": _exhausted_route(state)}

        new_query = new_query.strip().strip('"').strip("「」").splitlines()[0] if new_query.strip() else ""
        if not new_query or new_query == state["query"]:
            # 改写没有产生新查询，再检索一次也是同样结果 —— 直接结束
            return {"retries": max_retries, "route": _exhausted_route(state)}
        return {"query": new_query, "retries": retries + 1, "route": "retrieve"}

    return rewrite


def make_refuse_node() -> Any:  # noqa: ANN001
    """拒答节点：**不调用 LLM**，直接给出确定性回答。

    ★ 为什么不能让 generate 兼职拒答：
      generate 会把 top-5 无关 chunk 拼进 prompt，模型面对"必须基于资料回答"的
      系统提示，倾向于硬凑一段话并标上 [1][2] —— 那正是"挂着 5 条无关引用"的来源。
      拒答必须是**一条独立的路**：不进 prompt、不产生引用、不经 verify。
    """

    async def refuse(state: AgentState) -> dict[str, Any]:
        reason = state.get("grade_reason") or "检索到的内容与问题不相关"
        logger.info("agent.refused", question=state.get("question"),
                    reason=reason, retries=state.get("retries", 0))
        return {
            "answer": REFUSE_ANSWER,
            "citations": [],
            "verified": True,     # 拒答本身是可信的：它没有断言任何事实
            "route": "refuse",
            "usage": {**state.get("usage", {}), "refused": True,
                      "refuse_reason": reason,
                      "cited": 0, "invalid_citations": []},
        }

    return refuse


def make_generate_node(llm) -> Any:  # noqa: ANN001
    async def generate(state: AgentState) -> dict[str, Any]:
        # ★ 这里**不写 citations** —— 引用由 verify 节点统一产出。
        #   state["citations"] 的归约器是 operator.add，两处都写会变成重复累加。
        chunks = state.get("chunks") or []
        if not chunks:
            return {"answer": REFUSE_ANSWER}

        context = _render_context(chunks)

        if not is_enabled(llm):
            # ★ 无 LLM 的抽取式兜底：把 top-1 原文附上引用编号返回。
            #   这不是"假装有答案"，而是明确标注为检索结果 ——
            #   没有配 LLM 时系统依然可用，这对本地开发和演示很重要。
            top = chunks[0]
            answer = (
                "（未配置 LLM，以下为检索到的原文片段）\n\n"
                f"[1] {top.get('content', '')}"
            )
            return {"answer": answer}

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
            *[{"role": m["role"], "content": m["content"]} for m in (state.get("history") or [])[-6:]],
            {"role": "user", "content": state["question"]},
        ]
        try:
            answer = await llm.acomplete(messages, temperature=0.0)
        except Exception as exc:
            logger.exception("agent.generate_failed")
            return {"answer": REFUSE_ANSWER, "error": f"生成失败：{type(exc).__name__}"}
        return {"answer": answer}

    return generate


def verify_node(state: AgentState) -> dict[str, Any]:
    """★ 确定性校验，不调用 LLM。

    做三件事：
      ① 剔除指向不存在编号的引用（模型凭空写 [7]，但上下文只有 5 条）
      ② 检查被引用的块是否真的与答案有内容重叠（防"引用了但内容无关"）
      ③ 没有任何有效引用时标记 verified=False

    ②用的是 CJK 字符二元组重合度：不需要分词、不需要模型、完全确定性。

    ★ 拒答**不经过这里** —— 走独立的 refuse 节点。以前是在这里用
      `answer.startswith("根据现有资料无法回答")` 反推"模型是不是在拒答"：
      靠措辞判断语义，模型换个说法（"资料中没有提到"）就漏判，
      而漏判的后果是 verified=False 被当成"答案不可信"，与事实正好相反。
    """
    answer = state.get("answer") or ""
    chunks = state.get("chunks") or []
    by_rank = {i: c for i, c in enumerate(chunks, start=1)}

    cited_ranks: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        rank = int(match.group(1))
        if rank in by_rank and rank not in seen:
            seen.add(rank)
            cited_ranks.append(rank)

    invalid = sorted({
        int(m.group(1)) for m in _CITATION_RE.finditer(answer)
    } - set(by_rank))

    citations = [
        {"chunk_id": by_rank[r]["chunk_id"], "rank": r,
         "quote": _best_quote(answer, by_rank[r].get("content", ""))}
        for r in cited_ranks
    ]

    # 有答案但一条有效引用都没有 → 不可信
    verified = bool(citations)

    if invalid:
        logger.warning("agent.invalid_citations", ranks=invalid, n_chunks=len(chunks))

    return {
        "citations": citations,
        "verified": verified,
        "usage": {**state.get("usage", {}), "invalid_citations": invalid,
                  "cited": len(citations)},
    }


def _render_context(chunks: list[dict[str, Any]]) -> str:
    """与 RetrievalService.build_context 保持同一种编号格式。"""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.get("label") or f"doc-{chunk.get('document_id')}"
        blocks.append(f"[{i}] {label}\n{chunk.get('content', '')}")
    return "\n\n---\n\n".join(blocks)


def _best_quote(answer: str, content: str, *, window: int = 60) -> str:
    """从原文里找出与答案重叠度最高的一段，作为引用摘录。

    返回的必须是**原文的连续子串**（逐字摘录），不是答案里的句子 ——
    否则"引用校验"就变成了自己证明自己。
    """
    if not content:
        return ""
    answer_grams = _bigrams(answer)
    if not answer_grams:
        return content[:window]

    best_pos, best_score = 0, -1.0
    step = max(window // 3, 1)
    for pos in range(0, max(len(content) - window, 0) + 1, step):
        piece = content[pos:pos + window]
        grams = _bigrams(piece)
        if not grams:
            continue
        score = len(grams & answer_grams) / len(grams)
        if score > best_score:
            best_pos, best_score = pos, score
    return content[best_pos:best_pos + window]


def _bigrams(text: str) -> set[str]:
    cleaned = [ch for ch in text if not ch.isspace()]
    return {"".join(cleaned[i:i + 2]) for i in range(max(len(cleaned) - 1, 0))}


class _GradeVerdict(BaseModel):
    verdict: str = Field(description="sufficient 或 insufficient")
    reason: str = Field(default="", description="一句话理由")
