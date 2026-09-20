from __future__ import annotations

from collections import deque

import pytest

from rag.agent.agent import PiAgent
from rag.agent.harness import AgentHarness
from rag.agent.prompts import REFUSE_ANSWER
from rag.agent.tools import SearchKnowledgeTool
from rag.providers.base import LLMChatResponse, LLMToolCall
from rag.services.retrieval import RetrievalResult, RetrievedChunk


class ScriptedLLM:
    model_id = "fake-tool-model"

    def __init__(self, *responses: LLMChatResponse) -> None:
        self.responses = deque(responses)
        self.requests: list[dict] = []

    async def achat(self, messages: list[dict], **kwargs) -> LLMChatResponse:  # noqa: ANN003
        self.requests.append({"messages": messages, **kwargs})
        return self.responses.popleft()


class FakeRetrieval:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.queries: list[str] = []

    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:  # noqa: ANN003
        self.queries.append(query)
        return RetrievalResult(
            chunks=self.results.get(query, []),
            diagnostics={"fake": True, "top_k": kwargs.get("top_k")},
        )


def chunk(chunk_id: int, content: str, *, title: str = "学生手册") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.87,
        document_id=1,
        doc_title=title,
        section_path="学籍管理",
        page_start=23,
        page_end=23,
        rank=1,
        rerank_score=0.98,
    )


def build_harness(llm: ScriptedLLM, retrieval: FakeRetrieval) -> AgentHarness:
    tool = SearchKnowledgeTool(retrieval)  # type: ignore[arg-type]
    return AgentHarness(
        agent=PiAgent(llm=llm, tool_definitions=[tool.definition]),  # type: ignore[arg-type]
        tools=[tool],
        max_iterations=4,
        tool_timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_greeting_does_not_call_rag() -> None:
    llm = ScriptedLLM(LLMChatResponse(content="你好！有什么可以帮你？"))
    retrieval = FakeRetrieval({})

    result = await build_harness(llm, retrieval).run("你好")

    assert result.route == "direct"
    assert result.tool_calls == []
    assert retrieval.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "query", "evidence", "answer"),
    [
        (
            "学校重修有什么规定？",
            "学校 重修 规定",
            "必修课程考核不及格的，学生应当参加重修。",
            "必修课程不及格需要重修。[1]",
        ),
        (
            "毕业需要多少学分？",
            "毕业 学分 要求",
            "本科生修满培养方案规定的 160 学分方可毕业。",
            "培养方案要求修满 160 学分。[1]",
        ),
    ],
)
async def test_school_policy_questions_call_search_knowledge(
    question: str,
    query: str,
    evidence: str,
    answer: str,
) -> None:
    llm = ScriptedLLM(
        LLMChatResponse(tool_calls=[
            LLMToolCall(id="call-1", name="search_knowledge", arguments={"query": query})
        ]),
        LLMChatResponse(content=answer),
    )
    retrieval = FakeRetrieval({query: [chunk(1, evidence)]})

    result = await build_harness(llm, retrieval).run(question, top_k=5)

    assert result.route == "generate"
    assert result.verified is True
    assert retrieval.queries == [query]
    assert result.tool_calls == [
        {"name": "search_knowledge", "iteration": 1, "success": True}
    ]
    assert result.citations[0]["chunk_id"] == 1
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert '"citation_index": 1' in tool_message["content"]


@pytest.mark.asyncio
async def test_missing_knowledge_is_not_fabricated() -> None:
    llm = ScriptedLLM(
        LLMChatResponse(tool_calls=[
            LLMToolCall(
                id="call-1",
                name="search_knowledge",
                arguments={"query": "学校 火星交换生 住宿补贴"},
            )
        ]),
        # 即使模型试图给出无来源答案，Harness 也会因为没有有效引用而拒答。
        LLMChatResponse(content="学校会提供每月 5000 元补贴。"),
    )
    retrieval = FakeRetrieval({})

    result = await build_harness(llm, retrieval).run("火星交换生有多少住宿补贴？")

    assert result.route == "refuse"
    assert result.answer == REFUSE_ANSWER
    assert result.citations == []


@pytest.mark.asyncio
async def test_school_policy_guard_prevents_direct_unsupported_answer() -> None:
    llm = ScriptedLLM(
        # 模型第一轮错误地想直接回答，Harness 应强制补一次知识库检索。
        LLMChatResponse(content="可以无限次重修。"),
        LLMChatResponse(content="重修次数以培养方案规定为准。[1]"),
    )
    retrieval = FakeRetrieval({
        "学校重修有什么规定？": [chunk(1, "重修次数按各专业培养方案执行。")]
    })

    result = await build_harness(llm, retrieval).run("学校重修有什么规定？")

    assert result.route == "generate"
    assert retrieval.queries == ["学校重修有什么规定？"]
    assert result.tool_calls == [
        {"name": "search_knowledge", "iteration": 1, "success": True}
    ]


@pytest.mark.asyncio
async def test_loop_supports_multiple_tool_calls() -> None:
    llm = ScriptedLLM(
        LLMChatResponse(tool_calls=[
            LLMToolCall(id="call-1", name="search_knowledge", arguments={"query": "制度 A"})
        ]),
        LLMChatResponse(tool_calls=[
            LLMToolCall(id="call-2", name="search_knowledge", arguments={"query": "制度 B"})
        ]),
        LLMChatResponse(content="制度 A 要求甲。[1]；制度 B 要求乙。[2]"),
    )
    retrieval = FakeRetrieval({
        "制度 A": [chunk(1, "制度 A 要求甲。", title="文件 A")],
        "制度 B": [chunk(2, "制度 B 要求乙。", title="文件 B")],
    })

    result = await build_harness(llm, retrieval).run("比较制度 A 和制度 B")

    assert result.route == "generate"
    assert result.iterations == 3
    assert retrieval.queries == ["制度 A", "制度 B"]
    assert [c["rank"] for c in result.chunks] == [1, 2]
    assert [c["chunk_id"] for c in result.citations] == [1, 2]
