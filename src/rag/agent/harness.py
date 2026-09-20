"""Agent V2 Harness：Task State、统一工具、Observe/Decide/Evaluate Loop。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from rag.agent.agent import PiAgent
from rag.agent.nodes import verify_node
from rag.agent.prompts import REFUSE_ANSWER
from rag.agent.task_state import TaskState, prepare_task_state
from rag.agent.tools import (
    AgentTool,
    ToolContext,
    ToolExecution,
    ToolExecutionError,
    tool_failure,
)
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
    task_state: dict[str, Any] = field(default_factory=dict)

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
                "task_state": self.task_state,
            },
        }


class AgentHarness:
    def __init__(
        self,
        *,
        agent: PiAgent,
        tools: list[AgentTool],
        task_store: Any = None,
        max_iterations: int = 8,
        tool_timeout_seconds: float = 120,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于 0")
        self.agent = agent
        self.tools = {tool.name: tool for tool in tools}
        self.task_store = task_store
        self.max_iterations = max_iterations
        self.tool_timeout_seconds = tool_timeout_seconds

    async def run(
        self,
        question: str,
        *,
        thread_id: str = "",
        tenant_id: int = 1,
        student_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> AgentRunResult:
        logger.info("[Agent] user request", thread_id=thread_id, question_len=len(question))
        task = prepare_task_state(question, await self._load_task(thread_id, tenant_id))
        logger.info("[Task] " + task.task_type, status=task.status, turn=task.turn_count)
        await self._save_task(thread_id, tenant_id, student_id, task)

        messages = self.agent.initial_messages(question, history, task.prompt_view())
        context = ToolContext(
            tenant_id=tenant_id,
            top_k=top_k,
            student_id=student_id,
            thread_id=thread_id,
        )
        chunks_by_id: dict[int, dict[str, Any]] = {}
        call_log: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        searches: list[dict[str, Any]] = []
        successful_tools: set[str] = set()

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = await self.agent.step(messages)
            except Exception as exc:  # noqa: BLE001 — 模型失败也要落 Task State
                logger.exception("agent.step_failed", iteration=iteration)
                result = self._failed_result(
                    f"Agent 模型调用失败：{type(exc).__name__}",
                    task=task,
                    chunks=list(chunks_by_id.values()),
                    iterations=iteration,
                    calls=call_log,
                    errors=errors,
                    searches=searches,
                )
                await self._finish_task(result, task, thread_id, tenant_id, student_id, failed=True)
                return result

            if not response.tool_calls and not call_log and self.agent.requires_knowledge(question):
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
                answer = (response.content or "").strip()
                asking_user = answer.endswith(("?", "？"))
                missing = self._missing_required_tools(task, successful_tools)
                if missing and not asking_user and iteration < self.max_iterations:
                    logger.info("[Agent] task incomplete", missing_tools=sorted(missing))
                    messages.append(self.agent.assistant_message(response))
                    messages.append({
                        "role": "system",
                        "content": (
                            "任务尚未完成。缺少以下事实来源："
                            f"{', '.join(sorted(missing))}。继续选择合适工具；"
                            "不要凭模型知识补全，也不要向用户声称已经完成。"
                        ),
                    })
                    continue
                if missing and not asking_user:
                    result = self._failed_result(
                        "达到迭代上限时仍缺少工具事实：" + ", ".join(sorted(missing)),
                        task=task,
                        chunks=list(chunks_by_id.values()),
                        iterations=iteration,
                        calls=call_log,
                        errors=errors,
                        searches=searches,
                    )
                    await self._finish_task(
                        result, task, thread_id, tenant_id, student_id, failed=True
                    )
                    return result

                result = self._finalize(
                    answer,
                    task=task,
                    chunks=list(chunks_by_id.values()),
                    iterations=iteration,
                    tool_calls=call_log,
                    tool_errors=errors,
                    searches=searches,
                    successful_tools=successful_tools,
                )
                await self._finish_task(
                    result, task, thread_id, tenant_id, student_id,
                    failed=result.route in {"error", "refuse"},
                )
                return result

            messages.append(self.agent.assistant_message(response))
            for call in response.tool_calls:
                logger.info("[Agent] selecting tool: " + call.name, iteration=iteration)
                execution = await self._execute_tool(call.name, call.arguments, context)
                if execution.chunks:
                    self._merge_chunks(chunks_by_id, execution.chunks)
                payload = self._number_payload(execution.payload, chunks_by_id)
                success = bool(payload.get("success"))
                if success:
                    successful_tools.add(call.name)
                else:
                    error = payload.get("error") or {}
                    errors.append({
                        "tool": call.name,
                        "error": str(error.get("code") or "tool_error"),
                    })
                call_log.append({
                    "name": call.name,
                    "iteration": iteration,
                    "success": success,
                })
                task.record_tool(call.name, call.arguments, payload)
                await self._save_task(thread_id, tenant_id, student_id, task)
                if call.name == "search_knowledge" and success:
                    data = payload.get("data") or {}
                    searches.append({
                        "query": data.get("query", ""),
                        "returned": len(data.get("results") or []),
                        "diagnostics": data.get("diagnostics") or {},
                    })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                })

        result = self._failed_result(
            f"Agent 达到最大迭代次数 {self.max_iterations}，任务仍未完成",
            task=task,
            chunks=list(chunks_by_id.values()),
            iterations=self.max_iterations,
            calls=call_log,
            errors=errors,
            searches=searches,
        )
        await self._finish_task(result, task, thread_id, tenant_id, student_id, failed=True)
        logger.warning("[Agent] final response", route="error", reason="max_iterations")
        return result

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolExecution:
        tool = self.tools.get(name)
        if tool is None:
            return ToolExecution(payload=tool_failure(
                "unknown_tool", f"未注册工具：{name}", retryable=False
            ))
        try:
            return await asyncio.wait_for(
                tool.execute(arguments, context),
                timeout=self.tool_timeout_seconds,
            )
        except ToolExecutionError as exc:
            logger.warning("agent.tool_rejected", tool=name, code=exc.code)
            return ToolExecution(payload=tool_failure(
                exc.code, exc.public_message, retryable=exc.retryable
            ))
        except ValidationError:
            logger.warning("agent.tool_invalid_arguments", tool=name)
            return ToolExecution(payload=tool_failure(
                "invalid_arguments",
                "工具参数校验失败，请根据工具 schema 修正参数后重试。",
                retryable=True,
            ))
        except TimeoutError:
            logger.warning("agent.tool_timeout", tool=name)
            return ToolExecution(payload=tool_failure(
                "timeout", "工具执行超时，可以缩小查询范围后重试。", retryable=True
            ))
        except Exception as exc:  # noqa: BLE001 — 必须回填完整 tool message 链
            logger.exception("agent.tool_failed", tool=name)
            return ToolExecution(payload=tool_failure(
                "internal_error", f"工具暂时不可用（{type(exc).__name__}）。", retryable=True
            ))

    async def _load_task(self, thread_id: str, tenant_id: int) -> dict | None:
        if not thread_id or self.task_store is None:
            return None
        getter = getattr(self.task_store, "get_task_state", None)
        if getter is None:
            return None
        try:
            return await getter(thread_id, tenant_id=tenant_id)
        except Exception:
            logger.warning("task.load_failed", thread_id=thread_id, exc_info=True)
            return None

    async def _save_task(
        self,
        thread_id: str,
        tenant_id: int,
        student_id: str | None,
        task: TaskState,
    ) -> None:
        if not thread_id or self.task_store is None:
            return
        saver = getattr(self.task_store, "save_task_state", None)
        if saver is None:
            return
        try:
            await saver(
                thread_id,
                task.model_dump(mode="json"),
                tenant_id=tenant_id,
                student_id=student_id,
            )
        except Exception:
            logger.warning("task.save_failed", thread_id=thread_id, exc_info=True)

    async def _finish_task(
        self,
        result: AgentRunResult,
        task: TaskState,
        thread_id: str,
        tenant_id: int,
        student_id: str | None,
        *,
        failed: bool,
    ) -> None:
        task.finish(result.answer, failed=failed)
        result.task_state = task.model_dump(mode="json")
        await self._save_task(thread_id, tenant_id, student_id, task)
        logger.info("[Task] " + task.status, task_type=task.task_type)
        logger.info(
            "[Agent] final response",
            route=result.route,
            iterations=result.iterations,
            tool_call_count=len(result.tool_calls),
        )

    @staticmethod
    def _missing_required_tools(task: TaskState, successful: set[str]) -> set[str]:
        required: set[str] = set()
        if task.task_type == "graduation_progress":
            required = {"search_knowledge", "query_grades"}
        elif task.task_type == "course_selection":
            required = {"search_courses"}
            time_sensitive = any(
                value in task.constraints
                for value in ("preference:课程时间", "requirement:无课表冲突")
            )
            if time_sensitive:
                required |= {"query_schedule", "check_schedule_conflict"}
        elif task.task_type == "grade_query":
            required = {"query_grades"}
        elif task.task_type == "schedule_query":
            required = {"query_schedule"}
        elif task.task_type == "exam_query":
            required = {"query_exam"}
        elif task.task_type == "knowledge_qa":
            required = {"search_knowledge"}
        return required - successful

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
        if not payload.get("success") or not isinstance(payload.get("data"), dict):
            return payload
        numbered = dict(payload)
        data = dict(payload["data"])
        results = []
        for item in data.get("results") or []:
            row = dict(item)
            chunk = chunks_by_id.get(int(row.get("chunk_id") or 0))
            row["citation_index"] = int(chunk["rank"]) if chunk else 0
            results.append(row)
        if "results" in data:
            data["results"] = results
        numbered["data"] = data
        return numbered

    @staticmethod
    def _finalize(
        answer: str,
        *,
        task: TaskState,
        chunks: list[dict[str, Any]],
        iterations: int,
        tool_calls: list[dict[str, Any]],
        tool_errors: list[dict[str, str]],
        searches: list[dict[str, Any]],
        successful_tools: set[str],
    ) -> AgentRunResult:
        answer = answer or "抱歉，我暂时无法生成回答。"
        common = dict(
            iterations=iterations,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            searches=searches,
            task_state=task.model_dump(mode="json"),
        )
        if tool_calls and not successful_tools:
            return AgentRunResult(
                answer="当前任务所需的工具均未成功执行，无法提供可靠结果。请稍后重试。",
                chunks=chunks,
                citations=[],
                verified=True,
                route="error",
                grade_reason="all_tools_failed",
                **common,
            )
        if "search_knowledge" in successful_tools:
            checked = verify_node({"answer": answer, "chunks": chunks, "usage": {}})
            citations = checked.get("citations") or []
            if not chunks or not citations:
                return AgentRunResult(
                    answer=REFUSE_ANSWER,
                    chunks=chunks,
                    citations=[],
                    verified=True,
                    route="refuse",
                    grade_reason="知识库证据不足，或最终回答没有可核验引用",
                    **common,
                )
            return AgentRunResult(
                answer=answer,
                chunks=chunks,
                citations=citations,
                verified=bool(checked.get("verified")),
                route="task" if len(successful_tools) > 1 else "generate",
                **common,
            )
        return AgentRunResult(
            answer=answer,
            chunks=[],
            citations=[],
            verified=True,
            route="task" if tool_calls else "direct",
            **common,
        )

    @staticmethod
    def _failed_result(
        reason: str,
        *,
        task: TaskState,
        chunks: list[dict[str, Any]],
        iterations: int,
        calls: list[dict[str, Any]],
        errors: list[dict[str, str]],
        searches: list[dict[str, Any]],
    ) -> AgentRunResult:
        return AgentRunResult(
            answer=f"任务未能完成：{reason}。",
            chunks=chunks,
            citations=[],
            verified=True,
            route="error",
            grade_reason=reason,
            iterations=iterations,
            tool_calls=calls,
            tool_errors=errors,
            searches=searches,
            task_state=task.model_dump(mode="json"),
        )


__all__ = ["AgentHarness", "AgentRunResult"]
