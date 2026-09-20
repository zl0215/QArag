from __future__ import annotations

import datetime as dt
import json
from collections import deque

import pytest

from rag.agent.academic_tools import (
    CheckScheduleConflictTool,
    QueryExamTool,
    QueryGradesTool,
    QueryScheduleTool,
    SearchCoursesTool,
)
from rag.agent.agent import PiAgent
from rag.agent.harness import AgentHarness
from rag.agent.tools import SearchKnowledgeTool, ToolContext
from rag.infra.repository import (
    CourseRecord,
    ExamRecord,
    GradeRecord,
    MemoryRepository,
    ScheduleRecord,
)
from rag.providers.base import LLMChatResponse, LLMToolCall
from rag.services.academic import AcademicService
from rag.services.retrieval import RetrievalResult, RetrievedChunk


class ScriptedLLM:
    model_id = "fake-tool-model"

    def __init__(self, *responses: LLMChatResponse) -> None:
        self.responses = deque(responses)
        self.requests: list[dict] = []

    async def achat(self, messages: list[dict], **kwargs) -> LLMChatResponse:  # noqa: ANN003
        self.requests.append({"messages": [dict(m) for m in messages], **kwargs})
        return self.responses.popleft()


class FakeRetrieval:
    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:  # noqa: ANN003
        if "毕业" not in query:
            return RetrievalResult(chunks=[], diagnostics={"fake": True})
        return RetrievalResult(chunks=[RetrievedChunk(
            chunk_id=901,
            content="本科培养方案要求学生修满 160 学分方可毕业。",
            score=0.9,
            document_id=9,
            doc_title="本科培养方案",
            section_path="毕业要求",
            page_start=12,
            page_end=12,
            rank=1,
            rerank_score=0.99,
        )], diagnostics={"fake": True, "top_k": kwargs.get("top_k")})


def academic_repo() -> MemoryRepository:
    repo = MemoryRepository()
    repo.seed_academic_data(
        courses=[
            CourseRecord(
                course_id="AI-101-A",
                course_code="AI101",
                name="人工智能导论",
                credits=3,
                term="2027-spring",
                department="计算机学院",
                category="人工智能",
                meeting_times=[{
                    "weekday": 1, "start_time": "09:00", "end_time": "10:40",
                    "weeks": list(range(1, 17)), "location": "A201",
                }],
            ),
            CourseRecord(
                course_id="ML-201-B",
                course_code="ML201",
                name="机器学习",
                credits=3,
                term="2027-spring",
                department="计算机学院",
                category="人工智能",
                meeting_times=[{
                    "weekday": 2, "start_time": "14:00", "end_time": "15:40",
                    "weeks": list(range(1, 17)), "location": "B301",
                }],
            ),
        ],
        grades=[
            GradeRecord(
                student_id="S001", course_code="MATH101", course_name="高等数学",
                credits=60, score=85, grade_point=3.7, status="passed", term="2025-fall",
            ),
            GradeRecord(
                student_id="S001", course_code="CS101", course_name="程序设计",
                credits=60, score=90, grade_point=4.0, status="passed", term="2026-spring",
            ),
            GradeRecord(
                student_id="S001", course_code="PHY101", course_name="大学物理",
                credits=3, score=52, grade_point=0, status="failed", term="2026-spring",
            ),
        ],
        schedules=[ScheduleRecord(
            student_id="S001",
            course_id="MATH-ADV-A",
            course_code="MATH301",
            course_name="高等数学进阶",
            term="2027-spring",
            meeting_times=[{
                "weekday": 1, "start_time": "08:50", "end_time": "10:25",
                "weeks": list(range(1, 17)), "location": "A101",
            }],
        )],
        exams=[ExamRecord(
            student_id="S001",
            course_id="CS101-A",
            course_code="CS101",
            course_name="程序设计",
            term="2026-spring",
            exam_type="期末考试",
            start_at=dt.datetime(2027, 1, 8, 9, 0, tzinfo=dt.UTC),
            end_at=dt.datetime(2027, 1, 8, 11, 0, tzinfo=dt.UTC),
            location="教学楼 C302",
            seat="18",
        )],
    )
    return repo


def call(call_id: str, name: str, **arguments) -> LLMChatResponse:  # noqa: ANN003
    return LLMChatResponse(tool_calls=[
        LLMToolCall(id=call_id, name=name, arguments=arguments)
    ])


def build_harness(llm: ScriptedLLM, repo: MemoryRepository, *, max_iterations: int = 8):
    academic = AcademicService(repo)
    tools = [
        SearchKnowledgeTool(FakeRetrieval()),  # type: ignore[arg-type]
        SearchCoursesTool(academic),
        QueryGradesTool(academic),
        QueryScheduleTool(academic),
        QueryExamTool(academic),
        CheckScheduleConflictTool(academic),
    ]
    return AgentHarness(
        agent=PiAgent(llm=llm, tool_definitions=[tool.definition for tool in tools]),  # type: ignore[arg-type]
        tools=tools,
        task_store=repo,
        max_iterations=max_iterations,
        tool_timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_graduation_progress_orchestrates_knowledge_and_grades() -> None:
    llm = ScriptedLLM(
        call("k1", "search_knowledge", query="本科毕业学分要求"),
        call("g1", "query_grades", status="passed"),
        LLMChatResponse(content="毕业要求是 160 学分。[1] 已通过课程共 120 学分，因此还差 40 学分。"),
    )
    repo = academic_repo()

    result = await build_harness(llm, repo).run(
        "我距离毕业还差多少学分？", thread_id="grad-1", student_id="S001"
    )

    assert result.route == "task"
    assert [item["name"] for item in result.tool_calls] == ["search_knowledge", "query_grades"]
    assert result.citations[0]["chunk_id"] == 901
    assert result.task_state["task_type"] == "graduation_progress"
    assert result.task_state["status"] == "completed"
    saved = await repo.get_task_state("grad-1")
    assert saved and saved["completed_steps"] == ["search_knowledge#1", "query_grades#1"]


@pytest.mark.asyncio
async def test_course_selection_uses_schedule_courses_and_conflict_tool() -> None:
    llm = ScriptedLLM(
        call("s1", "query_schedule", term="2027-spring"),
        call("c1", "search_courses", keyword="人工智能", term="2027-spring"),
        call(
            "x1", "check_schedule_conflict",
            course_ids=["AI-101-A", "ML-201-B"], term="2027-spring",
        ),
        LLMChatResponse(content="机器学习（ML201）与现有课表不冲突；人工智能导论有冲突。"),
    )
    repo = academic_repo()

    result = await build_harness(llm, repo).run(
        "帮我找下学期不会和现有课程冲突的人工智能课程。",
        thread_id="course-1",
        student_id="S001",
    )

    assert result.route == "task"
    assert [item["name"] for item in result.tool_calls] == [
        "query_schedule", "search_courses", "check_schedule_conflict",
    ]
    conflict_message = json.loads(llm.requests[3]["messages"][-1]["content"])
    checks = conflict_message["data"]["checks"]
    assert checks[0]["has_conflict"] is True
    assert checks[1]["has_conflict"] is False


@pytest.mark.asyncio
async def test_grade_schedule_exam_tools_return_structured_envelopes() -> None:
    repo = academic_repo()
    service = AcademicService(repo)
    context = ToolContext(student_id="S001")

    grades = await QueryGradesTool(service).execute({"status": "failed"}, context)
    schedule = await QueryScheduleTool(service).execute({"term": "2027-spring"}, context)
    exams = await QueryExamTool(service).execute({"term": "2026-spring"}, context)

    assert grades.payload["success"] is True
    assert grades.payload["data"]["grades"][0]["course_code"] == "PHY101"
    assert schedule.payload["data"]["schedule"][0]["course_code"] == "MATH301"
    assert exams.payload["data"]["exams"][0]["seat"] == "18"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "tool_name", "arguments", "task_type"),
    [
        ("查询我的成绩", "query_grades", {}, "grade_query"),
        ("查看我下学期的课表", "query_schedule", {"term": "2027-spring"}, "schedule_query"),
        ("我的期末考试在哪里？", "query_exam", {"term": "2026-spring"}, "exam_query"),
    ],
)
async def test_agent_executes_single_academic_queries(
    question: str,
    tool_name: str,
    arguments: dict,
    task_type: str,
) -> None:
    llm = ScriptedLLM(
        call("one", tool_name, **arguments),
        LLMChatResponse(content="已根据教务数据返回查询结果。"),
    )

    result = await build_harness(llm, academic_repo()).run(question, student_id="S001")

    assert result.route == "task"
    assert result.tool_calls == [{"name": tool_name, "iteration": 1, "success": True}]
    assert result.task_state["task_type"] == task_type


@pytest.mark.asyncio
async def test_multi_turn_task_state_keeps_direction_and_time_preference() -> None:
    repo = academic_repo()
    first_llm = ScriptedLLM(LLMChatResponse(content="你更关注课程时间还是培养方案要求？"))
    first = await build_harness(first_llm, repo).run(
        "我想选人工智能方向的课。", thread_id="multi-1", student_id="S001"
    )
    assert first.task_state["status"] == "waiting_input"

    second_llm = ScriptedLLM(
        call("s1", "query_schedule", term="2027-spring"),
        call("c1", "search_courses", keyword="人工智能", term="2027-spring"),
        call("x1", "check_schedule_conflict", course_ids=["ML-201-B"], term="2027-spring"),
        LLMChatResponse(content="按课程时间筛选，机器学习当前不冲突。"),
    )
    second = await build_harness(second_llm, repo).run(
        "课程时间。", thread_id="multi-1", student_id="S001"
    )
    assert second.task_state["task_type"] == "course_selection"
    assert "direction:人工智能" in second.task_state["constraints"]
    assert "preference:课程时间" in second.task_state["constraints"]

    third_llm = ScriptedLLM(
        call("s2", "query_schedule", term="2027-spring"),
        call("c2", "search_courses", keyword="机器学习", term="2027-spring"),
        call("x2", "check_schedule_conflict", course_ids=["ML-201-B"], term="2027-spring"),
        LLMChatResponse(content="机器学习周二下午上课，与现有课程不冲突。"),
    )
    third = await build_harness(third_llm, repo).run(
        "那机器学习呢？", thread_id="multi-1", student_id="S001"
    )
    assert third.task_state["turn_count"] == 3
    assert "course_interest:机器学习" in third.task_state["constraints"]
    task_prompt = third_llm.requests[0]["messages"][1]["content"]
    assert "preference:课程时间" in task_prompt
    assert "direction:人工智能" in task_prompt


@pytest.mark.asyncio
async def test_tool_error_is_observed_and_agent_can_retry() -> None:
    llm = ScriptedLLM(
        call("bad", "search_courses", keyword="人工智能", limit=0),
        call("fixed", "search_courses", keyword="人工智能", limit=10),
        LLMChatResponse(content="找到人工智能导论和机器学习两门课程。"),
    )

    result = await build_harness(llm, academic_repo()).run("帮我查人工智能方向的课")

    assert result.route == "task"
    assert result.tool_calls[0]["success"] is False
    assert result.tool_calls[1]["success"] is True
    assert result.tool_errors == [{"tool": "search_courses", "error": "invalid_arguments"}]
    failed_message = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert failed_message["error"]["retryable"] is True


@pytest.mark.asyncio
async def test_loop_stops_at_eight_iterations() -> None:
    llm = ScriptedLLM(*[
        call(f"c{i}", "search_courses", keyword="人工智能") for i in range(8)
    ])

    result = await build_harness(llm, academic_repo()).run("帮我查人工智能方向的课")

    assert result.route == "error"
    assert result.iterations == 8
    assert result.task_state["status"] == "failed"
    assert "最大迭代次数 8" in result.grade_reason


@pytest.mark.asyncio
async def test_personal_tool_rejects_missing_student_context() -> None:
    repo = academic_repo()
    llm = ScriptedLLM(
        call("g1", "query_grades"),
        LLMChatResponse(content="当前会话没有学生身份，请先完成身份认证？"),
    )

    result = await build_harness(llm, repo).run("查询我的成绩")

    assert result.tool_calls[0]["success"] is False
    assert result.tool_errors[0]["error"] == "student_context_missing"
    assert result.route == "error"
