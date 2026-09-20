"""跨轮次任务状态。它与自然语言 conversation history 分开持久化。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal[
    "general",
    "knowledge_qa",
    "graduation_progress",
    "course_selection",
    "grade_query",
    "schedule_query",
    "exam_query",
]
TaskStatus = Literal["running", "waiting_input", "completed", "failed"]


class ToolResultSummary(BaseModel):
    tool: str
    success: bool
    item_count: int = 0
    arguments: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class TaskState(BaseModel):
    version: int = 2
    task_type: TaskType = "general"
    goal: str
    constraints: list[str] = Field(default_factory=list)
    slots: dict[str, Any] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    tool_results: list[ToolResultSummary] = Field(default_factory=list)
    status: TaskStatus = "running"
    last_user_input: str = ""
    turn_count: int = 1

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self.status = "running"
        self.task_type = _type_for_tool(name, self.task_type)
        step = f"{name}#{sum(1 for item in self.completed_steps if item.startswith(name)) + 1}"
        self.completed_steps.append(step)
        success = bool(payload.get("success"))
        data = payload.get("data") or {}
        count = int(data.get("count") or 0) if isinstance(data, dict) else 0
        error = payload.get("error") or {}
        self.tool_results.append(ToolResultSummary(
            tool=name,
            success=success,
            item_count=count,
            arguments=_compact_arguments(arguments),
            error_code=error.get("code") if isinstance(error, dict) else None,
        ))
        self.tool_results = self.tool_results[-24:]
        for key in ("term", "keyword", "department", "category", "course_code"):
            value = arguments.get(key)
            if value not in (None, ""):
                self.slots[key] = value

    def finish(self, answer: str, *, failed: bool = False) -> None:
        if failed:
            self.status = "failed"
        elif (answer or "").rstrip().endswith(("?", "？")):
            self.status = "waiting_input"
        else:
            self.status = "completed"

    def prompt_view(self) -> dict[str, Any]:
        """给模型的状态不含冗长工具正文；权威事实需要时应重新查询。"""
        return {
            "task_type": self.task_type,
            "goal": self.goal,
            "constraints": self.constraints,
            "slots": self.slots,
            "completed_steps": self.completed_steps[-12:],
            "tool_results": [item.model_dump() for item in self.tool_results[-8:]],
            "status": self.status,
            "last_user_input": self.last_user_input,
            "turn_count": self.turn_count,
        }


def prepare_task_state(question: str, existing: dict[str, Any] | None = None) -> TaskState:
    detected = classify_task(question)
    previous = TaskState.model_validate(existing) if existing else None
    keep_previous = bool(
        previous
        and (
            detected == previous.task_type
            or (detected == "general" and _looks_like_followup(question, previous))
        )
    )
    if keep_previous and previous is not None:
        state = previous.model_copy(deep=True)
        state.turn_count += 1
        state.status = "running"
    else:
        state = TaskState(task_type=detected, goal=question)
    state.last_user_input = question
    for constraint in extract_constraints(question):
        if constraint not in state.constraints:
            state.constraints.append(constraint)
        key, _, value = constraint.partition(":")
        if key and value:
            state.slots[key] = value
    return state


def classify_task(question: str) -> TaskType:
    text = question.casefold()
    if re.search(r"距离毕业|毕业进度|还差.{0,8}学分|已修.{0,12}学分", text):
        return "graduation_progress"
    if re.search(r"选课|找.{0,10}课程|推荐.{0,10}课|方向的课|那.{0,8}(?:课程|课)呢", text):
        return "course_selection"
    if re.search(r"成绩|绩点|挂科|不及格", text):
        return "grade_query"
    if re.search(r"课表|上课时间|课程冲突|时间冲突", text):
        return "schedule_query"
    if re.search(r"考试|考场|座位号|期末时间", text):
        return "exam_query"
    if re.search(
        r"知识库|已上传|文档|论文|学校.{0,8}(?:规定|要求)|重修规定|毕业.{0,8}(?:要求|需要).{0,8}学分",
        text,
    ):
        return "knowledge_qa"
    return "general"


def extract_constraints(question: str) -> list[str]:
    text = question.casefold()
    constraints: list[str] = []
    if "人工智能" in text or re.search(r"\bai\b", text):
        constraints.append("direction:人工智能")
    if "机器学习" in text or "machine learning" in text:
        constraints.append("course_interest:机器学习")
    if "时间" in text:
        constraints.append("preference:课程时间")
    if "下学期" in text:
        constraints.append("term:下学期")
    if "不冲突" in text or "不会" in text and "冲突" in text:
        constraints.append("requirement:无课表冲突")
    return constraints


def _looks_like_followup(question: str, previous: TaskState) -> bool:
    text = question.strip().casefold()
    if len(text) <= 40 and re.search(r"那|呢|这个|课程|时间|机器学习|人工智能|学分", text):
        return True
    return previous.status == "waiting_input" and len(text) <= 80


def _type_for_tool(name: str, current: TaskType) -> TaskType:
    mapping: dict[str, TaskType] = {
        "search_knowledge": "knowledge_qa",
        "query_grades": "grade_query",
        "query_schedule": "schedule_query",
        "query_exam": "exam_query",
        "search_courses": "course_selection",
        "check_schedule_conflict": "course_selection",
    }
    inferred = mapping.get(name, current)
    # 组合任务不能被其中一个工具降级成单步查询。
    if current in {"graduation_progress", "course_selection"}:
        return current
    return inferred


def _compact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in arguments.items():
        if isinstance(value, list):
            compact[key] = value[:20]
        elif isinstance(value, str):
            compact[key] = value[:200]
        else:
            compact[key] = value
    return compact


__all__ = ["TaskState", "ToolResultSummary", "classify_task", "prepare_task_state"]
