"""元数据仓储：文档 / 分块 / 任务。

两个实现：
  MemoryRepository   —— 进程内，Windows 本地开发与单测用，无需 Docker
  PostgresRepository —— 生产实现，任务领取用 FOR UPDATE SKIP LOCKED

★ 为什么任务表能替代 Celery：
   Postgres 的 `SELECT ... FOR UPDATE SKIP LOCKED` 提供了原子领取、
   崩溃后可重领（锁超时）、幂等重试，而且没有额外中间件。
   代价是没有优先级队列/定时任务/跨机广播 —— 那些场景才需要上 Celery。
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag.core.logging import get_logger
from rag.infra.models import DocStatus, JobStatus
from rag.schemas.chunk import Chunk

logger = get_logger(__name__)

DEFAULT_TENANT_ID = 1

# 领取任务时的租约时长。★ 它的作用是**故障恢复**，不是超时取消：
# worker 崩溃（kill -9 / 容器被 OOM）时来不及把任务标失败，这条任务会永远停在
# running。有了租约，另一个 worker 过了这个时间就能重新领走它。
# 定 15 分钟是因为"解析 + 嵌入一个大 PDF"可能真的要几分钟，
# 太短会让正常跑着的任务被第二个 worker 抢走 —— 那就变成重复摄取了。
DEFAULT_LOCK_SECONDS = 900


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class DocumentRecord:
    id: int
    tenant_id: int
    title: str
    source_uri: str
    mime_type: str
    content_hash: str
    text_hash: str | None = None
    object_key: str | None = None
    size_bytes: int = 0
    version: int = 1
    is_active: bool = True
    status: str = DocStatus.PENDING
    progress: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    parser: str | None = None
    lang: str | None = None
    page_count: int | None = None
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


@dataclass
class ChunkRecord:
    """检索命中后回表取回的权威数据。"""

    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    parent_index: int | None = None
    page_start: int = 0
    page_end: int = 0
    section_path: str = ""
    node_type: str = "paragraph"
    token_count: int = 0
    doc_title: str = ""
    version: int = 1


@dataclass
class MessageRecord:
    """一轮对话里的一条消息。

    ★ 它和 LangGraph 的 checkpointer 是**两件事**，不要合并：
      checkpointer 存的是图状态（带着 chunks、diagnostics 这些不该回灌给
      LLM 的东西），这里存的是裁剪过的对话记录，只用于拼上下文。
      两者的生命周期也不同 —— checkpointer 可以清，对话记录要留着做审计。
    """

    role: str
    content: str
    seq: int = 0


@dataclass
class JobRecord:
    id: int
    tenant_id: int
    document_id: int | None
    job_type: str = "ingest"
    status: str = JobStatus.QUEUED
    attempt: int = 0
    max_attempts: int = 3
    idempotency_key: str = ""
    payload: dict | None = None
    result: dict | None = None
    last_error: str | None = None
    # 租约：worker 领走任务时打上，崩溃后靠它过期重领（见 DEFAULT_LOCK_SECONDS）
    locked_by: str | None = None
    locked_until: dt.datetime | None = None
    next_run_at: dt.datetime | None = None


@dataclass
class CourseRecord:
    course_id: str
    course_code: str
    name: str
    credits: float
    term: str
    tenant_id: int = DEFAULT_TENANT_ID
    category: str = ""
    department: str = ""
    instructor: str = ""
    campus: str = ""
    capacity: int | None = None
    available_seats: int | None = None
    description: str = ""
    meeting_times: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GradeRecord:
    student_id: str
    course_code: str
    course_name: str
    credits: float
    status: str
    term: str
    tenant_id: int = DEFAULT_TENANT_ID
    course_id: str = ""
    score: float | None = None
    grade_point: float | None = None


@dataclass
class ScheduleRecord:
    student_id: str
    course_id: str
    course_code: str
    course_name: str
    term: str
    meeting_times: list[dict[str, Any]]
    tenant_id: int = DEFAULT_TENANT_ID


@dataclass
class ExamRecord:
    student_id: str
    course_code: str
    course_name: str
    term: str
    exam_type: str
    start_at: dt.datetime
    end_at: dt.datetime
    tenant_id: int = DEFAULT_TENANT_ID
    course_id: str = ""
    location: str = ""
    seat: str = ""
    status: str = "scheduled"


@runtime_checkable
class Repository(Protocol):
    # with_checkpointer：只有 API 需要 LangGraph checkpointer。
    # worker 传 False —— 它不跑 agent 图，建 checkpointer 表纯属白搭，
    # 而且会白占一条 psycopg 连接、多一个"启动时可能失败"的点。
    async def ensure_ready(self, *, with_checkpointer: bool = True) -> None: ...

    # ---- documents ----
    async def create_document(self, **fields: Any) -> DocumentRecord: ...
    async def get_document(self, document_id: int, tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None: ...
    async def find_document_by_hash(self, content_hash: str, tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None: ...
    async def list_documents(self, tenant_id: int = DEFAULT_TENANT_ID, *, limit: int = 50, offset: int = 0) -> list[DocumentRecord]: ...
    async def update_document(self, document_id: int, **fields: Any) -> None: ...
    async def soft_delete_document(self, document_id: int, tenant_id: int = DEFAULT_TENANT_ID) -> None: ...

    # ---- chunks ----
    async def replace_chunks(self, document_id: int, version: int, chunks: list[Chunk],
                             *, embed_model: str, embed_dim: int) -> list[int]: ...
    async def get_chunks(self, chunk_ids: list[int]) -> list[ChunkRecord]: ...
    async def get_section_window(self, document_id: int, version: int,
                                 parent_index: int) -> list[ChunkRecord]: ...
    async def count_chunks(self, tenant_id: int = DEFAULT_TENANT_ID) -> int: ...

    # ---- conversations / messages ----
    # 会话历史。★ 没有这一对方法时，多轮对话是**静默失效**的：
    #   _history() 用 getattr(..., None) 取方法，取不到就当空历史，
    #   一个字都不会报错，表现只是"模型不记得上一轮"。
    async def get_messages(self, thread_id: str, *, limit: int = 40,
                           tenant_id: int = DEFAULT_TENANT_ID) -> list[MessageRecord]: ...
    async def append_turn(self, thread_id: str, *, question: str, answer: str,
                          tenant_id: int = DEFAULT_TENANT_ID, **meta: Any) -> None: ...

    # ---- Agent task state ----
    async def get_task_state(self, thread_id: str,
                             tenant_id: int = DEFAULT_TENANT_ID) -> dict | None: ...
    async def save_task_state(self, thread_id: str, state: dict, *,
                              tenant_id: int = DEFAULT_TENANT_ID,
                              student_id: str | None = None) -> None: ...

    # ---- academic read model ----
    async def list_courses(self, *, tenant_id: int = DEFAULT_TENANT_ID,
                           term: str | None = None, keyword: str | None = None,
                           department: str | None = None, category: str | None = None,
                           course_ids: list[str] | None = None,
                           min_credits: float | None = None,
                           max_credits: float | None = None,
                           limit: int = 20) -> list[CourseRecord]: ...
    async def list_grades(self, student_id: str, *,
                          tenant_id: int = DEFAULT_TENANT_ID,
                          term: str | None = None, status: str | None = None,
                          course_code: str | None = None) -> list[GradeRecord]: ...
    async def list_schedule(self, student_id: str, *,
                            tenant_id: int = DEFAULT_TENANT_ID,
                            term: str | None = None) -> list[ScheduleRecord]: ...
    async def list_exams(self, student_id: str, *,
                         tenant_id: int = DEFAULT_TENANT_ID,
                         term: str | None = None,
                         course_code: str | None = None,
                         from_at: dt.datetime | None = None) -> list[ExamRecord]: ...

    # ---- jobs ----
    async def enqueue_job(self, **fields: Any) -> JobRecord: ...
    async def claim_job(self, worker_id: str) -> JobRecord | None: ...
    async def finish_job(self, job_id: int, *, status: str,
                         result: dict | None = None, error: str | None = None,
                         document_id: int | None = None) -> None: ...
    async def schedule_retry(self, job_id: int, *, error: str,
                             delay_seconds: float) -> None: ...
    async def requeue_stale_jobs(self, *, lock_timeout_seconds: int) -> int: ...
    async def get_job(self, job_id: int) -> JobRecord | None: ...

    async def aclose(self) -> None: ...


# ======================================================================
# 内存实现
# ======================================================================
class MemoryRepository:
    def __init__(self) -> None:
        import asyncio

        self._documents: dict[int, DocumentRecord] = {}
        self._chunks: dict[int, ChunkRecord] = {}
        self._jobs: dict[int, JobRecord] = {}
        self._messages: dict[str, list[MessageRecord]] = {}
        self._task_states: dict[tuple[int, str], dict] = {}
        self._courses: list[CourseRecord] = []
        self._grades: list[GradeRecord] = []
        self._schedules: list[ScheduleRecord] = []
        self._exams: list[ExamRecord] = []
        self._doc_seq = 0
        self._chunk_seq = 0
        self._job_seq = 0
        self._lock = asyncio.Lock()

    async def ensure_ready(self, *, with_checkpointer: bool = True) -> None:
        # 内存后端没有 checkpointer 这回事，签名收下参数只是为了满足 Protocol
        return None

    # ---- documents ----
    async def create_document(self, **fields: Any) -> DocumentRecord:
        async with self._lock:
            self._doc_seq += 1
            record = DocumentRecord(id=self._doc_seq, **fields)
            self._documents[record.id] = record
            return record

    async def get_document(self, document_id: int,
                           tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None:
        doc = self._documents.get(document_id)
        return doc if doc and doc.tenant_id == tenant_id else None

    async def find_document_by_hash(self, content_hash: str,
                                    tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None:
        for doc in self._documents.values():
            if (doc.content_hash == content_hash and doc.tenant_id == tenant_id
                    and doc.status != DocStatus.DELETED):
                return doc
        return None

    async def list_documents(self, tenant_id: int = DEFAULT_TENANT_ID, *,
                             limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        docs = [d for d in self._documents.values()
                if d.tenant_id == tenant_id and d.status != DocStatus.DELETED]
        docs.sort(key=lambda d: d.id, reverse=True)
        return docs[offset:offset + limit]

    async def update_document(self, document_id: int, **fields: Any) -> None:
        doc = self._documents.get(document_id)
        if not doc:
            return
        for key, value in fields.items():
            if hasattr(doc, key):
                setattr(doc, key, value)

    async def soft_delete_document(self, document_id: int,
                                   tenant_id: int = DEFAULT_TENANT_ID) -> None:
        doc = self._documents.get(document_id)
        if doc and doc.tenant_id == tenant_id:
            doc.status = DocStatus.DELETED
            doc.is_active = False
        for cid in [c for c, r in self._chunks.items() if r.document_id == document_id]:
            self._chunks.pop(cid, None)

    # ---- chunks ----
    async def replace_chunks(self, document_id: int, version: int, chunks: list[Chunk],
                             *, embed_model: str, embed_dim: int) -> list[int]:
        async with self._lock:
            # 先按文档删除旧块 —— "按 doc 重建"比逐块 update 幂等得多，
            # 因为文档一改，分块数量和边界几乎必然变化
            for cid in [c for c, r in self._chunks.items() if r.document_id == document_id]:
                self._chunks.pop(cid, None)

            doc = self._documents.get(document_id)
            title = doc.title if doc else ""
            ids: list[int] = []
            for chunk in chunks:
                self._chunk_seq += 1
                chunk.chunk_id = self._chunk_seq
                chunk.document_id = document_id
                self._chunks[self._chunk_seq] = ChunkRecord(
                    chunk_id=self._chunk_seq,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    parent_index=chunk.parent_index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    node_type=str(chunk.node_type),
                    token_count=chunk.token_count,
                    doc_title=title,
                    version=version,
                )
                ids.append(self._chunk_seq)
            return ids

    async def get_chunks(self, chunk_ids: list[int]) -> list[ChunkRecord]:
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    async def get_section_window(self, document_id: int, version: int,
                                 parent_index: int) -> list[ChunkRecord]:
        rows = [
            r for r in self._chunks.values()
            if r.document_id == document_id and r.version == version
            and r.parent_index == parent_index
        ]
        rows.sort(key=lambda r: r.chunk_index)
        return rows

    async def count_chunks(self, tenant_id: int = DEFAULT_TENANT_ID) -> int:
        doc_ids = {d.id for d in self._documents.values() if d.tenant_id == tenant_id}
        return sum(1 for r in self._chunks.values() if r.document_id in doc_ids)

    # ---- conversations / messages ----
    async def get_messages(self, thread_id: str, *, limit: int = 40,
                           tenant_id: int = DEFAULT_TENANT_ID) -> list[MessageRecord]:
        box = self._messages.get(thread_id) or []
        return list(box[-limit:])   # 取最近 limit 条，顺序仍是时间正序

    async def append_turn(self, thread_id: str, *, question: str, answer: str,
                          tenant_id: int = DEFAULT_TENANT_ID, **meta: Any) -> None:
        async with self._lock:
            box = self._messages.setdefault(thread_id, [])
            base = len(box)
            box.append(MessageRecord(role="user", content=question, seq=base + 1))
            box.append(MessageRecord(role="assistant", content=answer, seq=base + 2))

    # ---- Agent task state / academic read model ----
    async def get_task_state(self, thread_id: str,
                             tenant_id: int = DEFAULT_TENANT_ID) -> dict | None:
        state = self._task_states.get((tenant_id, thread_id))
        return dict(state) if state else None

    async def save_task_state(self, thread_id: str, state: dict, *,
                              tenant_id: int = DEFAULT_TENANT_ID,
                              student_id: str | None = None) -> None:
        self._task_states[(tenant_id, thread_id)] = dict(state)

    def seed_academic_data(
        self,
        *,
        courses: list[CourseRecord] | None = None,
        grades: list[GradeRecord] | None = None,
        schedules: list[ScheduleRecord] | None = None,
        exams: list[ExamRecord] | None = None,
    ) -> None:
        """测试和本地 Demo 的显式种子入口；生产数据仍从 PostgreSQL 读取。"""
        self._courses.extend(courses or [])
        self._grades.extend(grades or [])
        self._schedules.extend(schedules or [])
        self._exams.extend(exams or [])

    async def list_courses(self, *, tenant_id: int = DEFAULT_TENANT_ID,
                           term: str | None = None, keyword: str | None = None,
                           department: str | None = None, category: str | None = None,
                           course_ids: list[str] | None = None,
                           min_credits: float | None = None,
                           max_credits: float | None = None,
                           limit: int = 20) -> list[CourseRecord]:
        keyword_cf = (keyword or "").casefold()
        wanted_ids = set(course_ids or [])
        rows = [row for row in self._courses if row.tenant_id == tenant_id]
        if term:
            rows = [row for row in rows if row.term == term]
        if keyword_cf:
            rows = [row for row in rows if keyword_cf in " ".join((
                row.course_code, row.name, row.description, row.department, row.category,
            )).casefold()]
        if department:
            rows = [row for row in rows if row.department == department]
        if category:
            rows = [row for row in rows if row.category == category]
        if wanted_ids:
            rows = [row for row in rows if row.course_id in wanted_ids]
        if min_credits is not None:
            rows = [row for row in rows if row.credits >= min_credits]
        if max_credits is not None:
            rows = [row for row in rows if row.credits <= max_credits]
        return rows[:limit]

    async def list_grades(self, student_id: str, *,
                          tenant_id: int = DEFAULT_TENANT_ID,
                          term: str | None = None, status: str | None = None,
                          course_code: str | None = None) -> list[GradeRecord]:
        rows = [r for r in self._grades
                if r.tenant_id == tenant_id and r.student_id == student_id]
        if term:
            rows = [r for r in rows if r.term == term]
        if status:
            rows = [r for r in rows if r.status == status]
        if course_code:
            rows = [r for r in rows if r.course_code == course_code]
        return rows

    async def list_schedule(self, student_id: str, *,
                            tenant_id: int = DEFAULT_TENANT_ID,
                            term: str | None = None) -> list[ScheduleRecord]:
        rows = [r for r in self._schedules
                if r.tenant_id == tenant_id and r.student_id == student_id]
        return [r for r in rows if not term or r.term == term]

    async def list_exams(self, student_id: str, *,
                         tenant_id: int = DEFAULT_TENANT_ID,
                         term: str | None = None,
                         course_code: str | None = None,
                         from_at: dt.datetime | None = None) -> list[ExamRecord]:
        rows = [r for r in self._exams
                if r.tenant_id == tenant_id and r.student_id == student_id]
        if term:
            rows = [r for r in rows if r.term == term]
        if course_code:
            rows = [r for r in rows if r.course_code == course_code]
        if from_at:
            rows = [r for r in rows if r.start_at >= from_at]
        return sorted(rows, key=lambda r: r.start_at)

    # ---- jobs ----
    async def enqueue_job(self, **fields: Any) -> JobRecord:
        async with self._lock:
            key = fields.get("idempotency_key", "")
            if key:
                # 幂等：同 key 的排队/进行中任务不重复入队。
                # ★ 已成功的要再确认产物还在 —— 否则"删掉文档再传同一个文件"
                #   会返回一条陈旧的 succeeded，而实际什么都没做。与 Postgres 实现一致。
                for job in self._jobs.values():
                    if job.idempotency_key != key:
                        continue
                    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                        return job
                    if job.status == JobStatus.SUCCEEDED and self._outcome_alive(job):
                        return job
                    job.status = JobStatus.QUEUED
                    job.attempt = 0
                    job.result = None
                    job.last_error = None
                    job.next_run_at = _now()
                    job.locked_by = None
                    job.locked_until = None
                    if fields.get("payload"):
                        job.payload = fields["payload"]
                    return job
            self._job_seq += 1
            record = JobRecord(id=self._job_seq, **fields)
            self._jobs[record.id] = record
            return record

    def _outcome_alive(self, job: JobRecord) -> bool:
        """已成功的任务，它产出的文档是否还在（未被软删除）。"""
        doc = self._documents.get(job.document_id or 0)
        return doc is not None and doc.status != DocStatus.DELETED

    async def claim_job(self, worker_id: str) -> JobRecord | None:
        async with self._lock:
            now = _now()
            for job in sorted(self._jobs.values(), key=lambda j: j.id):
                # next_run_at：重试退避时会被推到未来，没到点就不该领
                if job.status == JobStatus.QUEUED and (job.next_run_at or now) <= now:
                    job.status = JobStatus.RUNNING
                    job.attempt += 1
                    job.locked_by = worker_id
                    job.locked_until = now + dt.timedelta(seconds=DEFAULT_LOCK_SECONDS)
                    return job
        return None

    async def finish_job(self, job_id: int, *, status: str,
                         result: dict | None = None, error: str | None = None,
                         document_id: int | None = None) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = status
        job.result = result
        job.last_error = error
        job.locked_by = None
        job.locked_until = None
        # ★ 只在真正处理成功时回填 —— 不要把 None 覆盖掉已有的 document_id。
        #   异步上传时 job 是 document_id=None 建的，处理完才知道是哪个文档；
        #   而重试路径上它可能已经有值了。
        if document_id is not None:
            job.document_id = document_id

    async def schedule_retry(self, job_id: int, *, error: str,
                             delay_seconds: float) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.QUEUED
        job.last_error = error
        job.next_run_at = _now() + dt.timedelta(seconds=delay_seconds)
        job.locked_by = None
        job.locked_until = None

    async def requeue_stale_jobs(self, *, lock_timeout_seconds: int) -> int:
        async with self._lock:
            now = _now()
            n = 0
            for job in self._jobs.values():
                # 只捞"租约已过期"的 —— 还在租期内的 running 任务属于别的活着的 worker
                if (job.status == JobStatus.RUNNING and job.locked_until is not None
                        and job.locked_until < now):
                    logger.warning("job.lease_expired", job_id=job.id,
                                   locked_by=job.locked_by)
                    job.status = JobStatus.QUEUED
                    job.locked_by = None
                    job.locked_until = None
                    n += 1
            return n

    async def get_job(self, job_id: int) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def aclose(self) -> None:
        return None


# ======================================================================
# PostgreSQL 实现
# ======================================================================
class PostgresRepository:
    def __init__(self, database) -> None:  # noqa: ANN001
        self._db = database

    async def ensure_ready(self, *, with_checkpointer: bool = True) -> None:
        await self._db.create_all()
        # worker 不跑 agent 图，不需要 checkpointer 表：白建一遍，
        # 还要多开一条 psycopg 连接，并且多一个"启动时可能失败"的点。
        if with_checkpointer:
            await self._db.setup_checkpointer()

    # ---- documents ----
    async def create_document(self, **fields: Any) -> DocumentRecord:
        from rag.infra.models import Document

        async with self._db.session() as session:
            doc = Document(**fields)
            session.add(doc)
            await session.flush()
            record = _to_document_record(doc)
            await session.commit()
            return record

    async def get_document(self, document_id: int,
                           tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None:
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document

        async with self._db.session() as session:
            doc = await session.scalar(
                select(Document).where(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.deleted_at == EPOCH_ZERO,
                )
            )
            return _to_document_record(doc) if doc else None

    async def find_document_by_hash(self, content_hash: str,
                                    tenant_id: int = DEFAULT_TENANT_ID) -> DocumentRecord | None:
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document

        async with self._db.session() as session:
            doc = await session.scalar(
                select(Document).where(
                    Document.tenant_id == tenant_id,
                    Document.content_hash == content_hash,
                    Document.deleted_at == EPOCH_ZERO,
                ).order_by(Document.id.desc()).limit(1)
            )
            return _to_document_record(doc) if doc else None

    async def list_documents(self, tenant_id: int = DEFAULT_TENANT_ID, *,
                             limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document

        async with self._db.session() as session:
            rows = (await session.scalars(
                select(Document).where(
                    Document.tenant_id == tenant_id,
                    Document.deleted_at == EPOCH_ZERO,
                ).order_by(Document.id.desc()).limit(limit).offset(offset)
            )).all()
            return [_to_document_record(r) for r in rows]

    async def update_document(self, document_id: int, **fields: Any) -> None:
        from sqlalchemy import update

        from rag.infra.models import Document

        if not fields:
            return
        async with self._db.session() as session:
            await session.execute(
                update(Document).where(Document.id == document_id).values(**fields)
            )
            await session.commit()

    async def soft_delete_document(self, document_id: int,
                                   tenant_id: int = DEFAULT_TENANT_ID) -> None:
        import datetime as _dt

        from sqlalchemy import update

        from rag.infra.models import Chunk, Document

        now = _dt.datetime.now(_dt.UTC)
        async with self._db.session() as session:
            await session.execute(
                update(Document)
                .where(Document.id == document_id, Document.tenant_id == tenant_id)
                .values(status=DocStatus.DELETED, is_active=False, deleted_at=now)
            )
            await session.execute(
                update(Chunk)
                .where(Chunk.document_id == document_id)
                .values(deleted_at=now, embed_status="stale")
            )
            await session.commit()

    # ---- chunks ----
    async def replace_chunks(self, document_id: int, version: int, chunks: list[Chunk],
                             *, embed_model: str, embed_dim: int) -> list[int]:
        import datetime as _dt

        from sqlalchemy import update

        from rag.infra.models import Chunk as ChunkModel
        from rag.infra.models import Document

        now = _dt.datetime.now(_dt.UTC)
        async with self._db.session() as session:
            # 软删旧版本，而不是硬删 —— 便于回滚与审计
            await session.execute(
                update(ChunkModel)
                .where(ChunkModel.document_id == document_id,
                       ChunkModel.deleted_at == _EPOCH)
                .values(deleted_at=now, embed_status="stale")
            )
            rows = [
                ChunkModel(
                    tenant_id=DEFAULT_TENANT_ID,
                    document_id=document_id,
                    version=version,
                    chunk_index=c.chunk_index,
                    parent_index=c.parent_index,
                    content=c.content,
                    content_hash=c.content_hash,
                    token_count=c.token_count,
                    char_count=c.char_count,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    section_path=c.section_path[:512] if c.section_path else None,
                    node_type=str(c.node_type),
                    embed_model=embed_model,
                    embed_dim=embed_dim,
                    embed_status="embedded",
                    embedded_at=now,
                )
                for c in chunks
            ]
            session.add_all(rows)
            await session.flush()
            ids = [r.id for r in rows]
            await session.execute(
                update(Document).where(Document.id == document_id)
                .values(chunk_count=len(ids), version=version)
            )
            await session.commit()
            return ids

    async def get_chunks(self, chunk_ids: list[int]) -> list[ChunkRecord]:
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document
        from rag.infra.models import Chunk as ChunkModel

        if not chunk_ids:
            return []
        async with self._db.session() as session:
            rows = (await session.execute(
                select(ChunkModel, Document.title)
                .join(Document, Document.id == ChunkModel.document_id)
                .where(
                    ChunkModel.id.in_(chunk_ids),
                    ChunkModel.deleted_at == EPOCH_ZERO,
                    Document.deleted_at == EPOCH_ZERO,
                )
            )).all()
            by_id = {
                chunk.id: _to_chunk_record(chunk, title) for chunk, title in rows
            }
            # 保持传入顺序（即相关性顺序）
            return [by_id[c] for c in chunk_ids if c in by_id]

    async def get_section_window(self, document_id: int, version: int,
                                 parent_index: int) -> list[ChunkRecord]:
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document
        from rag.infra.models import Chunk as ChunkModel

        async with self._db.session() as session:
            rows = (await session.execute(
                select(ChunkModel, Document.title)
                .join(Document, Document.id == ChunkModel.document_id)
                .where(
                    ChunkModel.document_id == document_id,
                    ChunkModel.version == version,
                    ChunkModel.parent_index == parent_index,
                    ChunkModel.deleted_at == EPOCH_ZERO,
                ).order_by(ChunkModel.chunk_index)
            )).all()
            return [_to_chunk_record(chunk, title) for chunk, title in rows]

    async def count_chunks(self, tenant_id: int = DEFAULT_TENANT_ID) -> int:
        from sqlalchemy import func, select

        from rag.infra.models import EPOCH_ZERO
        from rag.infra.models import Chunk as ChunkModel

        async with self._db.session() as session:
            return int(await session.scalar(
                select(func.count()).select_from(ChunkModel).where(
                    ChunkModel.tenant_id == tenant_id,
                    ChunkModel.deleted_at == EPOCH_ZERO,
                )
            ) or 0)

    # ---- conversations / messages ----
    async def get_messages(self, thread_id: str, *, limit: int = 40,
                           tenant_id: int = DEFAULT_TENANT_ID) -> list[MessageRecord]:
        from sqlalchemy import select

        from rag.infra.models import Conversation, Message

        async with self._db.session() as session:
            conv_id = await session.scalar(
                select(Conversation.id).where(
                    Conversation.thread_id == thread_id,
                    Conversation.tenant_id == tenant_id,
                )
            )
            if conv_id is None:
                return []
            # ★ 先按 seq 倒序取最近 limit 条，再反转回时间正序 ——
            #   正序 + LIMIT 取到的是**最早**的几轮，恰好是上下文里最不该留的。
            rows = (await session.scalars(
                select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.seq.desc())
                .limit(limit)
            )).all()
        return [MessageRecord(role=m.role, content=m.content, seq=m.seq)
                for m in reversed(rows)]

    async def append_turn(self, thread_id: str, *, question: str, answer: str,
                          tenant_id: int = DEFAULT_TENANT_ID, **meta: Any) -> None:
        import datetime as _dt

        from sqlalchemy import select

        from rag.infra.models import Conversation, Message

        now = _dt.datetime.now(_dt.UTC)
        async with self._db.session() as session:
            # 行锁串行化同一个 thread 的并发写入，seq 才不会撞唯一约束。
            # （两个请求同时开一个新会话仍可能一起走到 INSERT；
            #   这一条不需要额外兜底 —— 调用方是 best-effort，冲突只是丢一轮历史。）
            conv = await session.scalar(
                select(Conversation)
                .where(Conversation.thread_id == thread_id,
                       Conversation.tenant_id == tenant_id)
                .with_for_update()
            )
            if conv is None:
                conv = Conversation(tenant_id=tenant_id, thread_id=thread_id,
                                    message_count=0, last_message_at=now)
                session.add(conv)
                await session.flush()

            base = conv.message_count
            session.add(Message(conversation_id=conv.id, tenant_id=tenant_id,
                                seq=base + 1, role="user", content=question))
            session.add(Message(conversation_id=conv.id, tenant_id=tenant_id,
                                seq=base + 2, role="assistant", content=answer,
                                model=meta.get("model"), route=meta.get("route"),
                                retries=int(meta.get("retries") or 0),
                                latency_ms=meta.get("latency_ms")))
            conv.message_count = base + 2
            conv.last_message_at = now
            await session.commit()

    # ---- Agent task state ----
    async def get_task_state(self, thread_id: str,
                             tenant_id: int = DEFAULT_TENANT_ID) -> dict | None:
        from sqlalchemy import select

        from rag.infra.models import AgentTask

        async with self._db.session() as session:
            task = await session.scalar(select(AgentTask).where(
                AgentTask.tenant_id == tenant_id,
                AgentTask.thread_id == thread_id,
            ))
            return dict(task.state) if task else None

    async def save_task_state(self, thread_id: str, state: dict, *,
                              tenant_id: int = DEFAULT_TENANT_ID,
                              student_id: str | None = None) -> None:
        from sqlalchemy import select

        from rag.infra.models import AgentTask

        async with self._db.session() as session:
            task = await session.scalar(
                select(AgentTask).where(
                    AgentTask.tenant_id == tenant_id,
                    AgentTask.thread_id == thread_id,
                ).with_for_update()
            )
            if task is None:
                task = AgentTask(
                    tenant_id=tenant_id,
                    thread_id=thread_id,
                    student_id=student_id,
                    state=state,
                )
                session.add(task)
            else:
                task.state = state
                if student_id:
                    task.student_id = student_id
            await session.commit()

    # ---- academic read model ----
    async def list_courses(self, *, tenant_id: int = DEFAULT_TENANT_ID,
                           term: str | None = None, keyword: str | None = None,
                           department: str | None = None, category: str | None = None,
                           course_ids: list[str] | None = None,
                           min_credits: float | None = None,
                           max_credits: float | None = None,
                           limit: int = 20) -> list[CourseRecord]:
        from sqlalchemy import or_, select

        from rag.infra.models import AcademicCourse

        stmt = select(AcademicCourse).where(AcademicCourse.tenant_id == tenant_id)
        if term:
            stmt = stmt.where(AcademicCourse.term == term)
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(or_(
                AcademicCourse.course_code.ilike(like),
                AcademicCourse.name.ilike(like),
                AcademicCourse.description.ilike(like),
                AcademicCourse.department.ilike(like),
                AcademicCourse.category.ilike(like),
            ))
        if department:
            stmt = stmt.where(AcademicCourse.department == department)
        if category:
            stmt = stmt.where(AcademicCourse.category == category)
        if course_ids:
            stmt = stmt.where(AcademicCourse.course_id.in_(course_ids))
        if min_credits is not None:
            stmt = stmt.where(AcademicCourse.credits >= min_credits)
        if max_credits is not None:
            stmt = stmt.where(AcademicCourse.credits <= max_credits)
        stmt = stmt.order_by(AcademicCourse.course_code, AcademicCourse.course_id).limit(limit)
        async with self._db.session() as session:
            rows = (await session.scalars(stmt)).all()
            return [_to_course_record(row) for row in rows]

    async def list_grades(self, student_id: str, *,
                          tenant_id: int = DEFAULT_TENANT_ID,
                          term: str | None = None, status: str | None = None,
                          course_code: str | None = None) -> list[GradeRecord]:
        from sqlalchemy import select

        from rag.infra.models import StudentGrade

        stmt = select(StudentGrade).where(
            StudentGrade.tenant_id == tenant_id,
            StudentGrade.student_id == student_id,
        )
        if term:
            stmt = stmt.where(StudentGrade.term == term)
        if status:
            stmt = stmt.where(StudentGrade.status == status)
        if course_code:
            stmt = stmt.where(StudentGrade.course_code == course_code)
        stmt = stmt.order_by(StudentGrade.term, StudentGrade.course_code)
        async with self._db.session() as session:
            rows = (await session.scalars(stmt)).all()
            return [_to_grade_record(row) for row in rows]

    async def list_schedule(self, student_id: str, *,
                            tenant_id: int = DEFAULT_TENANT_ID,
                            term: str | None = None) -> list[ScheduleRecord]:
        from sqlalchemy import select

        from rag.infra.models import StudentSchedule

        stmt = select(StudentSchedule).where(
            StudentSchedule.tenant_id == tenant_id,
            StudentSchedule.student_id == student_id,
        )
        if term:
            stmt = stmt.where(StudentSchedule.term == term)
        stmt = stmt.order_by(StudentSchedule.course_code)
        async with self._db.session() as session:
            rows = (await session.scalars(stmt)).all()
            return [_to_schedule_record(row) for row in rows]

    async def list_exams(self, student_id: str, *,
                         tenant_id: int = DEFAULT_TENANT_ID,
                         term: str | None = None,
                         course_code: str | None = None,
                         from_at: dt.datetime | None = None) -> list[ExamRecord]:
        from sqlalchemy import select

        from rag.infra.models import StudentExam

        stmt = select(StudentExam).where(
            StudentExam.tenant_id == tenant_id,
            StudentExam.student_id == student_id,
        )
        if term:
            stmt = stmt.where(StudentExam.term == term)
        if course_code:
            stmt = stmt.where(StudentExam.course_code == course_code)
        if from_at:
            stmt = stmt.where(StudentExam.start_at >= from_at)
        stmt = stmt.order_by(StudentExam.start_at)
        async with self._db.session() as session:
            rows = (await session.scalars(stmt)).all()
            return [_to_exam_record(row) for row in rows]

    # ---- jobs ----
    async def enqueue_job(self, **fields: Any) -> JobRecord:
        from sqlalchemy import select

        from rag.infra.models import IngestionJob
        from rag.infra.models import JobStatus as JS

        key = fields.get("idempotency_key", "")
        async with self._db.session() as session:
            if key:
                existing = await session.scalar(
                    select(IngestionJob).where(IngestionJob.idempotency_key == key)
                )
                if existing is not None:
                    # ① 还在跑：直接返回，别重复投递
                    if existing.status in (JS.QUEUED, JS.RUNNING):
                        return _to_job_record(existing)
                    # ② 成功且**产物还在**：真的做完了，返回它
                    if existing.status == JS.SUCCEEDED and await self._outcome_alive(
                        session, existing
                    ):
                        return _to_job_record(existing)
                    # ③ 失败过，或产物已经被删掉 —— 复用这一行重新排队。
                    #
                    # ★ 这里必须"复用"而不是"再插一行"，因为 idempotency_key 上有
                    #   唯一约束（uq_jobs_idempotency），再插必然 IntegrityError。
                    #   曾经的问题正是如此：旧任务状态是 failed 时，代码会落到下面的
                    #   session.add()，于是"上传一个曾经解析失败的文件"返回 500。
                    #
                    # ★ 判断 ② 里的"产物还在"同样必要：文档被删除后再上传同一个文件，
                    #   键还是一样的。只看状态就会返回那条陈旧的 succeeded，
                    #   界面上显示"已完成"，而知识库里空空如也 —— 什么都没发生。
                    return _to_job_record(
                        await self._requeue(session, existing, fields.get("payload"))
                    )
            job = IngestionJob(**fields)
            session.add(job)
            await session.flush()
            record = _to_job_record(job)
            await session.commit()
            return record

    async def _outcome_alive(self, session, job) -> bool:  # noqa: ANN001
        """已成功的任务，它产出的文档是否还在（未被软删除）。"""
        from sqlalchemy import select

        from rag.infra.models import EPOCH_ZERO, Document

        if job.document_id is None:
            return False
        alive = await session.scalar(
            select(Document.id).where(
                Document.id == job.document_id,
                Document.deleted_at == EPOCH_ZERO,
            )
        )
        return alive is not None

    async def _requeue(self, session, job, payload):  # noqa: ANN001
        """把一条已结束的任务重置回 queued，等待 worker 重新领取。

        ★ 必须把所有"上一次运行留下的痕迹"一起清掉：attempt 不清零的话，
          重试预算会凭空少一次；result 不清的话，前端会先看到上一次的旧结果；
          locked_* 不清的话，崩溃恢复逻辑会以为它还被人占着。
        """
        from rag.infra.models import JobStatus as JS

        job.status = JS.QUEUED
        job.attempt = 0
        job.result = None
        job.last_error = None
        job.next_run_at = _now()
        job.locked_by = None
        job.locked_until = None
        job.started_at = None
        job.finished_at = None
        if payload:
            job.payload = payload
        await session.commit()
        await session.refresh(job)
        return job

    async def claim_job(self, worker_id: str) -> JobRecord | None:
        """★ FOR UPDATE SKIP LOCKED —— 多 worker 并发领取不会拿到同一条。

        这是 Postgres 任务表能替代 Celery 的核心原因：原子领取、崩溃可重领、
        不需要额外中间件。（SQLite 不支持这个语法 —— 这也是集成测试不能用它的原因。）
        """
        from sqlalchemy import select, text

        from rag.infra.models import IngestionJob
        from rag.infra.models import JobStatus as JS

        now = _now()
        async with self._db.session() as session:
            # ★ 这三行必须在**同一个事务**里：SELECT ... FOR UPDATE SKIP LOCKED
            #   拿到行锁之后，随后的 UPDATE 才在锁的保护下。
            #   拆成两个事务会出现"两个 worker 都选中同一行"。
            row = await session.execute(
                text(
                    "SELECT id FROM ingestion_jobs "
                    "WHERE status = 'queued' AND next_run_at <= now() "
                    "ORDER BY priority DESC, id ASC "
                    "LIMIT 1 FOR UPDATE SKIP LOCKED"
                )
            )
            job_id = row.scalar()
            if job_id is None:
                return None

            job = await session.scalar(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job.status = JS.RUNNING
            job.attempt += 1
            job.locked_by = worker_id
            job.locked_until = now + dt.timedelta(seconds=DEFAULT_LOCK_SECONDS)
            job.started_at = job.started_at or now
            record = _to_job_record(job)
            await session.commit()
            return record

    async def finish_job(self, job_id: int, *, status: str,
                         result: dict | None = None, error: str | None = None,
                         document_id: int | None = None) -> None:
        from sqlalchemy import update

        from rag.infra.models import IngestionJob

        values: dict[str, Any] = {
            "status": status, "result": result, "last_error": error,
            "finished_at": _now(),
            # 释放租约，否则这条记录看起来还"被某个 worker 拿着"
            "locked_by": None, "locked_until": None,
        }
        # ★ 只在有值时回填，不要用 None 覆盖已有的 document_id
        if document_id is not None:
            values["document_id"] = document_id

        async with self._db.session() as session:
            await session.execute(
                update(IngestionJob).where(IngestionJob.id == job_id).values(**values)
            )
            await session.commit()

    async def schedule_retry(self, job_id: int, *, error: str,
                             delay_seconds: float) -> None:
        """把任务放回队列，并推迟到 `delay_seconds` 之后才可被领取（指数退避）。"""
        from sqlalchemy import update

        from rag.infra.models import IngestionJob
        from rag.infra.models import JobStatus as JS

        async with self._db.session() as session:
            await session.execute(
                update(IngestionJob).where(IngestionJob.id == job_id).values(
                    status=JS.QUEUED,
                    last_error=error,
                    next_run_at=_now() + dt.timedelta(seconds=delay_seconds),
                    locked_by=None, locked_until=None,
                )
            )
            await session.commit()

    async def requeue_stale_jobs(self, *, lock_timeout_seconds: int) -> int:
        """回收租约过期的任务 —— worker 崩溃后的兜底。

        ★ 这条是"任务表替代 Celery"能不能成立的关键：
          没有它，worker 被 kill -9 之后任务永远停在 running，
          既不会重试也不会报错，表现是"上传之后一直转圈，日志里什么都没有"。
          有了它，最多等一个租约周期就会自动重跑。

        ⚠️ 只在**没有别的 worker 正在处理**时才安全 —— 判据是租约过期。
          所以租约时长必须大于最慢一次摄取的时间（见 DEFAULT_LOCK_SECONDS）。
        """
        from sqlalchemy import text

        cutoff = _now() - dt.timedelta(seconds=lock_timeout_seconds)
        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "UPDATE ingestion_jobs "
                    "SET status = 'queued', locked_by = NULL, locked_until = NULL "
                    "WHERE status = 'running' "
                    "  AND locked_until IS NOT NULL AND locked_until < :cutoff"
                ),
                {"cutoff": cutoff},
            )
            n = result.rowcount or 0
            await session.commit()
        if n:
            logger.warning("job.lease_expired", count=n, cutoff=cutoff.isoformat())
        return n

    async def get_job(self, job_id: int) -> JobRecord | None:
        from sqlalchemy import select

        from rag.infra.models import IngestionJob

        async with self._db.session() as session:
            job = await session.scalar(select(IngestionJob).where(IngestionJob.id == job_id))
            return _to_job_record(job) if job else None

    async def aclose(self) -> None:
        await self._db.dispose()


# ----------------------------------------------------------------------
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def _to_document_record(doc) -> DocumentRecord:  # noqa: ANN001
    return DocumentRecord(
        id=doc.id, tenant_id=doc.tenant_id, title=doc.title, source_uri=doc.source_uri,
        mime_type=doc.mime_type, content_hash=doc.content_hash, text_hash=doc.text_hash,
        object_key=doc.object_key, size_bytes=doc.size_bytes, version=doc.version,
        is_active=doc.is_active, status=doc.status, progress=doc.progress,
        chunk_count=doc.chunk_count, error_message=doc.error_message, parser=doc.parser,
        lang=doc.lang, page_count=doc.page_count,
        created_at=doc.created_at or _EPOCH,
    )


def _to_chunk_record(chunk, title: str) -> ChunkRecord:  # noqa: ANN001
    return ChunkRecord(
        chunk_id=chunk.id, document_id=chunk.document_id, chunk_index=chunk.chunk_index,
        content=chunk.content, parent_index=chunk.parent_index,
        page_start=chunk.page_start or 0, page_end=chunk.page_end or 0,
        section_path=chunk.section_path or "", node_type=chunk.node_type,
        token_count=chunk.token_count, doc_title=title or "", version=chunk.version,
    )


def _to_job_record(job) -> JobRecord:  # noqa: ANN001
    return JobRecord(
        id=job.id, tenant_id=job.tenant_id, document_id=job.document_id,
        job_type=job.job_type, status=job.status, attempt=job.attempt,
        max_attempts=job.max_attempts, idempotency_key=job.idempotency_key,
        payload=job.payload, result=job.result, last_error=job.last_error,
        locked_by=job.locked_by, locked_until=job.locked_until,
        next_run_at=job.next_run_at,
    )


def _to_course_record(row) -> CourseRecord:  # noqa: ANN001
    return CourseRecord(
        tenant_id=row.tenant_id,
        course_id=row.course_id,
        course_code=row.course_code,
        name=row.name,
        credits=float(row.credits),
        category=row.category or "",
        department=row.department or "",
        term=row.term,
        instructor=row.instructor or "",
        campus=row.campus or "",
        capacity=row.capacity,
        available_seats=row.available_seats,
        description=row.description or "",
        meeting_times=list(row.meeting_times or []),
    )


def _to_grade_record(row) -> GradeRecord:  # noqa: ANN001
    return GradeRecord(
        tenant_id=row.tenant_id,
        student_id=row.student_id,
        course_id=row.course_id or "",
        course_code=row.course_code,
        course_name=row.course_name,
        credits=float(row.credits),
        score=float(row.score) if row.score is not None else None,
        grade_point=float(row.grade_point) if row.grade_point is not None else None,
        status=row.status,
        term=row.term,
    )


def _to_schedule_record(row) -> ScheduleRecord:  # noqa: ANN001
    return ScheduleRecord(
        tenant_id=row.tenant_id,
        student_id=row.student_id,
        course_id=row.course_id,
        course_code=row.course_code,
        course_name=row.course_name,
        term=row.term,
        meeting_times=list(row.meeting_times or []),
    )


def _to_exam_record(row) -> ExamRecord:  # noqa: ANN001
    return ExamRecord(
        tenant_id=row.tenant_id,
        student_id=row.student_id,
        course_id=row.course_id or "",
        course_code=row.course_code,
        course_name=row.course_name,
        term=row.term,
        exam_type=row.exam_type,
        start_at=row.start_at,
        end_at=row.end_at,
        location=row.location or "",
        seat=row.seat or "",
        status=row.status,
    )


def new_worker_id() -> str:
    return f"worker-{uuid.uuid4().hex[:8]}"
