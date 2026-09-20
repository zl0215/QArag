"""SQLAlchemy 2.0 ORM 模型。

★ 全局约定（都是踩过坑才定下来的）：

- **软删除用纪元零值而不是 NULL**：MySQL/Postgres 的唯一索引里 NULL 不参与去重，
  用 NULL 会出现"同一个业务键可以存在多条未删除记录"的漏洞。
- **时间统一用 `DateTime(timezone=True)`**：不用 TIMESTAMP（2038 问题 + 隐式时区转换）。
- **枚举用 VARCHAR + CHECK**，Python 侧 StrEnum 单一来源 —— 不用数据库原生 ENUM，
  因为加值要 ALTER TYPE，迁移很痛。
- **每张业务表第一列都是 tenant_id**，所有索引以它为前导列。
  一期单租户恒为 1，但结构留着，"扩展路径已预留"才不是空话。
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 软删除哨兵值
EPOCH_ZERO = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)

DEFAULT_TENANT_ID = 1


class Base(DeclarativeBase):
    pass


class DocStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"
    NEEDS_OCR = "needs_ocr"
    DELETED = "deleted"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EmbedStatus(StrEnum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    STALE = "stale"
    FAILED = "failed"


def _created_at() -> Mapped[dt.datetime]:
    return mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def _updated_at() -> Mapped[dt.datetime]:
    return mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )


def _deleted_at() -> Mapped[dt.datetime]:
    # ★ server_default 只接受 SQL 片段，不能直接塞 datetime 对象 ——
    #   必须写成 SQL 字面量。这里用 timestamptz 显式带时区，
    #   否则 Postgres 会按服务器时区解释，跨时区部署时"未删除"的判断会错。
    return mapped_column(
        DateTime(timezone=True), nullable=False,
        default=EPOCH_ZERO,
        server_default=text("'1970-01-01 00:00:00+00'::timestamptz"),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=DEFAULT_TENANT_ID)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    object_key: Mapped[str | None] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    # sha256(原始字节) —— L1 哈希，命中可跳过解析（最贵的一步）
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # sha256(归一化文本) —— L2 哈希，命中可跳过分块
    text_hash: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=DocStatus.PENDING)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    parser: Mapped[str | None] = mapped_column(String(64))
    lang: Mapped[str | None] = mapped_column(String(16))
    page_count: Mapped[int | None] = mapped_column(Integer)
    extra: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = _created_at()
    updated_at: Mapped[dt.datetime] = _updated_at()
    deleted_at: Mapped[dt.datetime] = _deleted_at()

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "content_hash", "version", "deleted_at",
                         name="uq_documents_hash_version"),
        Index("ix_documents_tenant_status", "tenant_id", "status", "id"),
        Index("ix_documents_tenant_created", "tenant_id", "deleted_at", "id"),
        CheckConstraint(
            "status IN ('pending','parsing','chunking','embedding','ready','failed','needs_ocr','deleted')",
            name="ck_documents_status",
        ),
    )


class Chunk(Base):
    """★ chunk 正文的权威存储。Milvus 里的 content 只是派生副本（供 BM25 分词用）。"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=DEFAULT_TENANT_ID)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # small-to-big：同 section 的连续子块共享一个 parent_index
    parent_index: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(String(512))
    node_type: Mapped[str] = mapped_column(String(32), nullable=False, default="paragraph")
    lang: Mapped[str | None] = mapped_column(String(8))
    embed_model: Mapped[str] = mapped_column(String(64), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embed_status: Mapped[str] = mapped_column(String(12), nullable=False, default=EmbedStatus.PENDING)
    embedded_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = _created_at()
    updated_at: Mapped[dt.datetime] = _updated_at()
    deleted_at: Mapped[dt.datetime] = _deleted_at()

    document: Mapped[Document] = relationship(back_populates="chunks", lazy="noload")

    __table_args__ = (
        UniqueConstraint("document_id", "version", "chunk_index", "deleted_at",
                         name="uq_chunks_doc_version_index"),
        Index("ix_chunks_tenant_doc", "tenant_id", "document_id", "version", "chunk_index"),
        Index("ix_chunks_parent", "document_id", "version", "parent_index"),
        Index("ix_chunks_embed_backlog", "embed_status", "updated_at"),
        CheckConstraint(
            "embed_status IN ('pending','embedded','stale','failed')",
            name="ck_chunks_embed_status",
        ),
    )


class IngestionJob(Base):
    """任务表。

    ★ 用 PostgreSQL 的 FOR UPDATE SKIP LOCKED 做原子领取 ——
      这一张表就替代了 Celery + Redis 的组合（见 README 的设计取舍）。
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=DEFAULT_TENANT_ID)
    document_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE")
    )
    job_type: Mapped[str] = mapped_column(String(24), nullable=False, default="ingest")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.QUEUED)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    # ★ 幂等键：同一文件重复上传不会产生第二次解析
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    result: Mapped[dict | None] = mapped_column(JSONB)
    last_error: Mapped[str | None] = mapped_column(Text)
    next_run_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    locked_by: Mapped[str | None] = mapped_column(String(64))
    locked_until: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = _created_at()
    updated_at: Mapped[dt.datetime] = _updated_at()

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_jobs_idempotency"),
        Index("ix_jobs_claim", "status", "next_run_at", "priority", "id"),
        Index("ix_jobs_tenant_doc", "tenant_id", "document_id", "id"),
        CheckConstraint(
            "status IN ('queued','running','succeeded','failed')", name="ck_jobs_status"
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=DEFAULT_TENANT_ID)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(String(2048))
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_message_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = _created_at()
    updated_at: Mapped[dt.datetime] = _updated_at()
    deleted_at: Mapped[dt.datetime] = _deleted_at()

    __table_args__ = (
        UniqueConstraint("thread_id", name="uq_conversations_thread"),
        Index("ix_conversations_recent", "tenant_id", "deleted_at", "last_message_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=DEFAULT_TENANT_ID)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(12), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    route: Mapped[str | None] = mapped_column(String(16))
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trace_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = _created_at()

    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uq_messages_conv_seq"),
        Index("ix_messages_tenant_created", "tenant_id", "created_at", "id"),
        CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_messages_role"),
    )


class MessageCitation(Base):
    """引用溯源。用独立表而不是 JSON —— 方便做聚合统计与评估。"""

    __tablename__ = "message_citations"

    message_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    rank_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # 逐字摘录 —— 用于校验引用真实性（模型可能编造引用）
    quote: Mapped[str | None] = mapped_column(String(512))

    __table_args__ = (Index("ix_citations_chunk", "chunk_id"),)
