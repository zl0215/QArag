"""Agent V1 的轻量 Harness：注册工具、执行 loop、收敛结果。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from rag.agent.agent import PiAgent
from rag.agent.nodes import verify_node
from rag.agent.prompts import REFUSE_ANSWER
from rag.agent.tools import SearchKnowledgeTool, ToolContext, ToolExecution
from rag.core.logging import get_logger
from rag.providers.base import LLMChatResponse, LLMToolCall

logger = get_logger(__name__)


@dataclass
class AgentRunResult:
    answer: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    verified: bool = False
    route: str = ""
    grade_reason: str = ""
    iterations: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: list[dict[str, str]] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)

    def to_state(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "chunks": self.chunks,
            "citations": self.citations,
            "verified": self.verified,
            "route": self.route,
            "grade_reason": self.grade_reason,
            "retries": 0,
            "retrieval_diagnostics": {
                "agent_iterations": self.iterations,
                "tool_calls": self.tool_calls,
                "tool_errors": self.tool_errors,
                "searches": self.searches,
            },
        }


class AgentHarness:
    """统一的 Agent 入口；V1 保持一个工具和一个短循环。"""

    def __init__(
        self,
        *,
        agent: PiAgent,
        tools: list[SearchKnowledgeTool],
        max_iterations: int = 4,
        tool_timeout_seconds: float = 120,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于 0")
        self.agent = agent
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.tool_timeout_seconds = tool_timeout_seconds

    async def run(
        self,
        question: str,
        *,
        thread_id: str = "",
        tenant_id: int = 1,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> AgentRunResult:
        logger.info("[Agent] user request", thread_id=thread_id, question_len=len(question))
        messages = self.agent.initial_messages(question, history)
        context = ToolContext(tenant_id=tenant_id, top_k=top_k)
        chunks_by_id: dict[int, dict[str, Any]] = {}
        call_log: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        searches: list[dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            response = await self.agent.step(messages)
            if (
                not response.tool_calls
                and not call_log
                and self.agent.requires_knowledge(question)
            ):
                # 这是 Prompt 规则的确定性护栏：模型偶发直答学校规定时，
                # 强制先拿证据，再让同一个 Agent 继续判断和生成。
                logger.info("[Agent] tool policy enforced", tool="search_knowledge")
                response = LLMChatResponse(
                    content=response.content,
                    tool_calls=[LLMToolCall(
                        id=f"policy-search-{iteration}",
                        name="search_knowledge",
                        arguments={"query": question},
                    )],
                    finish_reason=response.finish_reason,
                )
            if not response.tool_calls:
                result = self._finalize(
                    response.content,
                    chunks=list(chunks_by_id.values()),
                    iterations=iteration,
                    tool_calls=call_log,
                    tool_errors=errors,
                    searches=searches,
                )
                logger.info(
                    "[Agent] final response",
                    route=result.route,
                    iterations=iteration,
                    tool_call_count=len(call_log),
                )
                return result

            messages.append(self.agent.assistant_message(response))
            for call in response.tool_calls:
                logger.info("[Agent] tool selected: search_knowledge" if call.name == "search_knowledge"
                            else "[Agent] unknown tool selected", tool=call.name)
                call_log.append({"name": call.name, "iteration": iteration})
                execution = await self._execute_tool(call.name, call.arguments, context, errors)
                if execution is not None:
                    self._merge_chunks(chunks_by_id, execution.chunks)
                    payload = self._number_payload(execution.payload, chunks_by_id)
                    searches.append({
                        "query": payload.get("query", ""),
                        "returned": len(payload.get("results") or []),
                        "diagnostics": payload.get("diagnostics") or {},
                    })
                else:
                    payload = {"error": "工具执行失败，不能据此回答事实问题。"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False),
                })

        reason = f"Agent 达到最大迭代次数 {self.max_iterations}，未生成最终回答"
        result = AgentRunResult(
            answer=REFUSE_ANSWER,
            chunks=list(chunks_by_id.values()),
            citations=[],
            verified=True,
            route="refuse",
            grade_reason=reason,
            iterations=self.max_iterations,
            tool_calls=call_log,
            tool_errors=errors,
            searches=searches,
        )
        logger.warning("[Agent] final response", route="refuse", reason="max_iterations")
        return result

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        errors: list[dict[str, str]],
    ) -> ToolExecution | None:
        tool = self.tools.get(name)
        if tool is None:
            errors.append({"tool": name, "error": "unknown_tool"})
            return None
        try:
            return await asyncio.wait_for(
                tool.execute(arguments, context),
                timeout=self.tool_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — tool 失败要回填消息，不能破坏对话链
            errors.append({"tool": name, "error": type(exc).__name__})
            logger.exception("agent.tool_failed", tool=name)
            return None

    @staticmethod
    def _merge_chunks(
        chunks_by_id: dict[int, dict[str, Any]],
        new_chunks: list[dict[str, Any]],
    ) -> None:
        for chunk in new_chunks:
            chunk_id = int(chunk["chunk_id"])
            if chunk_id in chunks_by_id:
                continue
            normalized = dict(chunk)
            normalized["rank"] = len(chunks_by_id) + 1
            chunks_by_id[chunk_id] = normalized

    @staticmethod
    def _number_payload(
        payload: dict[str, Any],
        chunks_by_id: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        numbered = dict(payload)
        results = []
        for item in payload.get("results") or []:
            row = dict(item)
            chunk = chunks_by_id.get(int(row.get("chunk_id") or 0))
            row["citation_index"] = int(chunk["rank"]) if chunk else 0
            results.append(row)
        numbered["results"] = results
        return numbered

    @staticmethod
    def _finalize(
        answer: str,
        *,
        chunks: list[dict[str, Any]],
        iterations: int,
        tool_calls: list[dict[str, Any]],
        tool_errors: list[dict[str, str]],
        searches: list[dict[str, Any]],
    ) -> AgentRunResult:
        answer = (answer or "").strip()
        if not tool_calls:
            return AgentRunResult(
                answer=answer or "抱歉，我暂时无法生成回答。",
                chunks=[],
                citations=[],
                verified=True,
                route="direct",
                iterations=iterations,
                tool_calls=tool_calls,
                tool_errors=tool_errors,
                searches=searches,
            )

        checked = verify_node({"answer": answer, "chunks": chunks, "usage": {}})
        citations = checked.get("citations") or []
        if not chunks or not citations:
            reason = (
                "知识库检索失败" if tool_errors and not chunks
                else "检索证据不足，或最终回答没有可核验引用"
            )
            return AgentRunResult(
                answer=REFUSE_ANSWER,
                chunks=chunks,
                citations=[],
                verified=True,
                route="refuse",
                grade_reason=reason,
                iterations=iterations,
                tool_calls=tool_calls,
                tool_errors=tool_errors,
                searches=searches,
            )

        return AgentRunResult(
            answer=answer,
            chunks=chunks,
            citations=citations,
            verified=bool(checked.get("verified")),
            route="generate",
            iterations=iterations,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            searches=searches,
        )


__all__ = ["AgentHarness", "AgentRunResult"]
