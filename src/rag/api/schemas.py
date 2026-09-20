"""HTTP 请求/响应模型。

★ 为什么不直接复用 services 层的 dataclass：
   dataclass 是**内部**结构，改字段不该影响 API 契约；反过来 API 需要
   pydantic 的校验与 OpenAPI 生成。两者中间隔一层，各自演化的自由度才在。
   代价是每次加字段要写两遍 —— 这个成本比"内部重构直接破坏线上契约"低得多。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ======================================================================
# 检索
# ======================================================================
class RetrievedChunkOut(BaseModel):
    chunk_id: int
    content: str
    score: float
    document_id: int
    doc_title: str = ""
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    node_type: str = "paragraph"
    rank: int = 0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    # 稠密余弦分。`score` 是 RRF 分（只看名次，rank1/k=60 恒为 0.016393），
    # 界面上要展示"像不像"必须用这个。
    dense_score: float | None = None

    @property
    def label(self) -> str:
        parts = [self.doc_title or f"doc-{self.document_id}"]
        if self.section_path:
            parts.append(self.section_path)
        if self.page_start:
            parts.append(
                f"p{self.page_start}" if self.page_start == self.page_end
                else f"p{self.page_start}-{self.page_end}"
            )
        return " · ".join(parts)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    # ★ 三个开关暴露给 API 是为了做消融实验。
    #   如果只在配置文件里，跑一组对比就要改 .env + 重启，
    #   而重启一次要重载模型 —— 评测循环会慢到没法用。
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    # None = 用服务端默认（RETRIEVE_TRANSLATE）。显式传 False 可关掉跨语言扩展，
    # 用来量它到底贡献了多少。
    use_translate: bool | None = None

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query 不能为空白")
        return v


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunkOut]
    diagnostics: dict = Field(default_factory=dict)
    context: str = ""


# ======================================================================
# 问答
# ======================================================================
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(
        default=None, max_length=128,
        description="同一 thread_id 共享对话历史与 agent checkpoint；不传则每次独立",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question 不能为空白")
        return v


class CitationOut(BaseModel):
    index: int
    chunk_id: int
    label: str
    quote: str
    verified: bool = True


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    citations: list[CitationOut]
    verified: bool
    route: str = ""
    grade_reason: str = ""
    retries: int = 0
    diagnostics: dict = Field(default_factory=dict)


# ======================================================================
# 文档
# ======================================================================
class DocumentOut(BaseModel):
    id: int
    title: str
    status: str
    mime_type: str = ""
    lang: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    size_bytes: int = 0
    error_message: str | None = None
    created_at: str = ""
    updated_at: str = ""


class IngestResponse(BaseModel):
    document_id: int
    job_id: int | None = None
    status: str
    deduplicated: bool = False
    chunk_count: int = 0
    vectors_written: int = 0
    page_count: int = 0
    duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentOut]


class JobOut(BaseModel):
    id: int
    status: str
    document_id: int | None = None
    attempts: int = 0
    error: str | None = None


# ======================================================================
# 运维
# ======================================================================
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    env: str


class ComponentStatus(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    ready: bool
    components: list[ComponentStatus]
