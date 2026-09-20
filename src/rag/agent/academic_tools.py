"""Agent V2 教务工具：统一校验输入并返回 success/data/error 信封。"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from rag.agent.tools import (
    ToolContext,
    ToolExecution,
    ToolExecutionError,
    tool_success,
)
from rag.core.logging import get_logger
from rag.services.academic import AcademicService

logger = get_logger(__name__)


class SearchCoursesArgs(BaseModel):
    keyword: str | None = Field(default=None, max_length=200)
    term: str | None = Field(default=None, max_length=32)
    department: str | None = Field(default=None, max_length=128)
    category: str | None = Field(default=None, max_length=64)
    min_credits: float | None = Field(default=None, ge=0, le=50)
    max_credits: float | None = Field(default=None, ge=0, le=50)
    limit: int = Field(default=20, ge=1, le=50)


class QueryGradesArgs(BaseModel):
    term: str | None = Field(default=None, max_length=32)
    status: Literal["passed", "failed", "in_progress", "withdrawn"] | None = None
    course_code: str | None = Field(default=None, max_length=64)


class QueryScheduleArgs(BaseModel):
    term: str | None = Field(default=None, max_length=32)


class QueryExamArgs(BaseModel):
    term: str | None = Field(default=None, max_length=32)
    course_code: str | None = Field(default=None, max_length=64)
    from_at: dt.datetime | None = None


class CheckScheduleConflictArgs(BaseModel):
    course_ids: list[str] = Field(min_length=1, max_length=20)
    term: str | None = Field(default=None, max_length=32)


class _AcademicTool:
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    def __init__(self, service: AcademicService) -> None:
        self.service = service

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    @staticmethod
    def student_id(context: ToolContext) -> str:
        if not context.student_id:
            raise ToolExecutionError(
                "student_context_missing",
                "当前会话没有可信学生身份，无法查询个人教务数据。",
                retryable=False,
            )
        return context.student_id

    @staticmethod
    def completed(name: str, count: int) -> None:
        logger.info(f"[Tool] {name} completed", result_count=count)


class SearchCoursesTool(_AcademicTool):
    name = "search_courses"
    description = "查询课程目录，可按学期、关键词、院系、类别和学分筛选；返回课程及上课时间。"
    args_model = SearchCoursesArgs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = SearchCoursesArgs.model_validate(arguments)
        logger.info("[Tool] search_courses started")
        rows = await self.service.search_courses(tenant_id=context.tenant_id, **args.model_dump())
        self.completed(self.name, len(rows))
        return ToolExecution(payload=tool_success({"courses": rows, "count": len(rows)}))


class QueryGradesTool(_AcademicTool):
    name = "query_grades"
    description = "查询当前学生的成绩事实；可按学期、状态或课程代码过滤，不替用户判断毕业结论。"
    args_model = QueryGradesArgs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = QueryGradesArgs.model_validate(arguments)
        logger.info("[Tool] query_grades started")
        rows = await self.service.query_grades(
            self.student_id(context), tenant_id=context.tenant_id, **args.model_dump()
        )
        self.completed(self.name, len(rows))
        return ToolExecution(payload=tool_success({"grades": rows, "count": len(rows)}))


class QueryScheduleTool(_AcademicTool):
    name = "query_schedule"
    description = "查询当前学生的已选课表和每门课的时间地点。"
    args_model = QueryScheduleArgs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = QueryScheduleArgs.model_validate(arguments)
        logger.info("[Tool] query_schedule started")
        rows = await self.service.query_schedule(
            self.student_id(context), tenant_id=context.tenant_id, **args.model_dump()
        )
        self.completed(self.name, len(rows))
        return ToolExecution(payload=tool_success({"schedule": rows, "count": len(rows)}))


class QueryExamTool(_AcademicTool):
    name = "query_exam"
    description = "查询当前学生的考试安排，包括课程、时间、地点、座位和状态。"
    args_model = QueryExamArgs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = QueryExamArgs.model_validate(arguments)
        logger.info("[Tool] query_exam started")
        rows = await self.service.query_exams(
            self.student_id(context), tenant_id=context.tenant_id, **args.model_dump()
        )
        self.completed(self.name, len(rows))
        return ToolExecution(payload=tool_success({"exams": rows, "count": len(rows)}))


class CheckScheduleConflictTool(_AcademicTool):
    name = "check_schedule_conflict"
    description = (
        "确定性检查候选 course_ids 与当前学生已有课表是否时间冲突。"
        "应先用 search_courses 获得真实 course_id。"
    )
    args_model = CheckScheduleConflictArgs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = CheckScheduleConflictArgs.model_validate(arguments)
        logger.info("[Tool] check_schedule_conflict started")
        rows = await self.service.check_schedule_conflict(
            self.student_id(context),
            args.course_ids,
            tenant_id=context.tenant_id,
            term=args.term,
        )
        self.completed(self.name, len(rows))
        return ToolExecution(payload=tool_success({"checks": rows, "count": len(rows)}))


__all__ = [
    "CheckScheduleConflictTool",
    "QueryExamTool",
    "QueryGradesTool",
    "QueryScheduleTool",
    "SearchCoursesTool",
]
