"""Agent 工具公共协议与知识库工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.services.retrieval import RetrievalService

logger = get_logger(__name__)


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="用于知识库检索的完整查询")


class KnowledgeResult(BaseModel):
    content: str
    source: str
    page: int = 0
    page_end: int = 0
    section: str = ""
    score: float
    chunk_id: int
    document_id: int
    rank: int
    dense_score: float | None = None
    rerank_score: float | None = None
    node_type: str = "paragraph"

    def to_agent_chunk(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "document_id": self.document_id,
            "doc_title": self.source.split(" · ", 1)[0],
            "section_path": self.section,
            "page_start": self.page,
            "page_end": self.page_end,
            "node_type": self.node_type,
            "rank": self.rank,
            "dense_score": self.dense_score,
            "rerank_score": self.rerank_score,
            "label": self.source,
        }


class SearchKnowledgeResult(BaseModel):
    query: str
    results: list[KnowledgeResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    tenant_id: int = 1
    top_k: int | None = None
    student_id: str | None = None
    thread_id: str = ""


@dataclass
class ToolExecution:
    payload: dict[str, Any]
    chunks: list[dict[str, Any]] = field(default_factory=list)


class AgentTool(Protocol):
    name: str
    description: str

    @property
    def definition(self) -> dict[str, Any]: ...

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution: ...


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.retryable = retryable


def tool_success(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def tool_failure(code: str, message: str, *, retryable: bool) -> dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


async def search_knowledge(
    query: str,
    *,
    retrieval: RetrievalService,
    tenant_id: int = 1,
    top_k: int | None = None,
) -> SearchKnowledgeResult:
    """调用现有混合检索链路并返回结构化证据，不生成最终回答。"""
    logger.info("[Tool] search_knowledge started", query_len=len(query))
    result = await retrieval.retrieve(query, tenant_id=tenant_id, top_k=top_k)
    items = [
        KnowledgeResult(
            content=chunk.content,
            source=chunk.label,
            page=chunk.page_start,
            page_end=chunk.page_end,
            section=chunk.section_path,
            score=chunk.score,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            rank=chunk.rank,
            dense_score=chunk.dense_score,
            rerank_score=chunk.rerank_score,
            node_type=chunk.node_type,
        )
        for chunk in result.chunks
    ]
    logger.info("[Tool] search_knowledge completed", result_count=len(items))
    return SearchKnowledgeResult(query=query, results=items, diagnostics=result.diagnostics)


class SearchKnowledgeTool:
    name = "search_knowledge"
    description = (
        "检索已上传知识库中的中英文资料，返回原文片段、来源、页码、章节和相关性分数。"
        "该工具只提供证据，不生成最终答案。"
    )

    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": SearchKnowledgeArgs.model_json_schema(),
            },
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        args = SearchKnowledgeArgs.model_validate(arguments)
        result = await search_knowledge(
            args.query,
            retrieval=self.retrieval,
            tenant_id=context.tenant_id,
            top_k=context.top_k,
        )
        return ToolExecution(
            payload=tool_success(result.model_dump()),
            chunks=[item.to_agent_chunk() for item in result.results],
        )


__all__ = [
    "AgentTool",
    "KnowledgeResult",
    "SearchKnowledgeResult",
    "SearchKnowledgeTool",
    "ToolContext",
    "ToolExecution",
    "ToolExecutionError",
    "search_knowledge",
    "tool_failure",
    "tool_success",
]
