"""HTTP 路由。

★ 每个 handler 只做三件事：取依赖 → 调 service → 序列化。
   任何一行 if/else 的业务判断都说明它写错地方了。

★ 错误映射集中在这里（见 main.py 的 exception_handler），handler 里不写 try：
   抛 IngestionError → 422，ServiceUnavailableError → 503，其他 → 500。
   这样"什么错返回什么码"只有一处定义，不会出现每个接口各写一套的情况。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from rag.api.deps import AppContext, require_llm
from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ComponentStatus,
    DocumentListResponse,
    DocumentOut,
    HealthResponse,
    IngestResponse,
    ReadyResponse,
    RetrievedChunkOut,
    RetrieveRequest,
    RetrieveResponse,
)
from rag.core.config import get_settings
from rag.core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
)
from rag.core.logging import get_logger
from rag.parsers.base import _sha256_file

logger = get_logger(__name__)
router = APIRouter()

# 未就绪也不影响服务可用的组件。它们的 ok=False 只作为**能力声明**给前端看
# （例如 reranker 为 False 时界面禁用"重排"开关），不参与 /readyz 的 ready 汇总。
OPTIONAL_COMPONENTS = {"reranker"}


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


# ======================================================================
# 运维
# ======================================================================
@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """存活探针。

    ★ 故意不检查任何下游依赖。它的语义是"进程还活着，别重启我"。
      把数据库检查放进来会导致：Postgres 抖一下 → 所有 API 容器被判定死亡 →
      集体重启 → 雪崩。下游健康是 /readyz 的事。
    """
    s = get_settings()
    return HealthResponse(status="ok", version="0.1.0", env=s.app_env)


@router.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(request: Request) -> ReadyResponse:
    """就绪探针：逐个组件报状态，任何一项不就绪就 ready=False。

    ★ 返回 200 而不是 503：body 里的 ready 字段才是判据。
      返回非 200 的话，反向代理/网关会直接把响应体吞掉换成自己的错误页，
      排查时看不到"到底是哪个组件没就绪"。
    """
    rows = await _ctx(request).readyz()
    return ReadyResponse(
        # ★ OPTIONAL 里的组件不参与 ready 汇总。
        #   reranker 在 RERANK_PROVIDER=none 时 ok=False（前端靠这个禁用"重排"开关），
        #   但"没配重排器"完全不影响服务可用 —— 检索会退回 RRF 名次排序。
        #   把它算进 ready 会让一个正常工作的栈在 /readyz 上永远显示未就绪。
        ready=all(ok for n, ok, _ in rows if n not in OPTIONAL_COMPONENTS),
        components=[ComponentStatus(name=n, ok=ok, detail=d) for n, ok, d in rows],
    )


# ======================================================================
# 检索
# ======================================================================
@router.post("/api/v1/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    """只检索，不生成。

    ★ 这个接口是消融实验的入口：use_* 开关让"去掉稠密 / 去掉稀疏 / 去掉重排 /
      去掉跨语言扩展"几组对比能在同一个进程里跑完，不用重启、不用重载模型。
    """
    ctx = _ctx(request)
    result = await ctx.retrieval.retrieve(
        payload.query,
        top_k=payload.top_k,
        use_dense=payload.use_dense,
        use_sparse=payload.use_sparse,
        use_rerank=payload.use_rerank,
        use_translate=payload.use_translate,
    )
    return RetrieveResponse(
        query=payload.query,
        chunks=[RetrievedChunkOut(**vars(c)) for c in result.chunks],
        diagnostics=result.diagnostics,
        context=ctx.retrieval.build_context(result.chunks),
    )


# ======================================================================
# 问答
# ======================================================================
@router.post("/api/v1/chat", tags=["chat"])
async def chat(payload: ChatRequest, request: Request):
    """问答。`stream=true` 时返回 SSE，否则返回 JSON。

    ★ 两种模式共用一个 handler 而不是拆成两个接口：
      它们的前置校验、thread_id 生成、错误映射完全一样，拆开就是复制粘贴。
    """
    ctx = _ctx(request)
    require_llm(ctx)

    thread_id = payload.thread_id or f"t-{uuid.uuid4().hex[:16]}"
    if payload.stream:
        return StreamingResponse(
            _sse(ctx, payload, thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # ★ 关掉 nginx 的缓冲，否则 SSE 会被攒成一大块再吐，
                #   "流式"就变成了"等半天然后一次性出现"。
                "X-Accel-Buffering": "no",
            },
        )

    state = await _run(ctx, payload, thread_id)
    return _to_chat_response(thread_id, state)


async def _run(ctx: AppContext, payload: ChatRequest, thread_id: str) -> dict:
    started = time.monotonic()
    history = await _history(ctx, thread_id)
    result = await ctx.agent_harness.run(
        payload.question,
        thread_id=thread_id,
        history=history,
        top_k=payload.top_k,
    )
    state = result.to_state()
    await _remember(ctx, thread_id, payload.question, state, started)
    return state


async def _remember(ctx: AppContext, thread_id: str, question: str,
                    state: dict, started: float) -> None:
    """把这一轮写进 messages 表 —— 下一轮的 `_history()` 就是从这里读的。

    ★ best-effort：历史写失败**不能**把已经生成好的答案弄丢，
      所以这里吞异常只记日志，和 `_history()` 的降级策略对称。

    ★ 空答案不落库：拒答/出错时 answer 可能是空的，写进去只会污染下一轮的
      上下文（模型会看到一条空的 assistant 消息）。
    """
    answer = (state.get("answer") or "").strip()
    appender = getattr(ctx.repository, "append_turn", None)
    if appender is None or not answer:
        return
    try:
        await appender(
            thread_id,
            question=question,
            answer=answer,
            model=getattr(ctx.llm, "model_id", None),
            route=state.get("route"),
            retries=int(state.get("retries") or 0),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception:
        logger.warning("chat.remember_failed", thread_id=thread_id, exc_info=True)


async def _history(ctx: AppContext, thread_id: str) -> list[dict[str, str]]:
    """取该会话的历史消息。

    ★ 图里的 checkpointer 保存的是**状态**，不是干净的对话记录 ——
      它带着 chunks、diagnostics 这些不该回灌给 LLM 的东西。
      所以上下文改写要用单独一份裁剪过的 history（见 repository 的 messages 表）。
      仓储没实现这个方法时降级为空历史，不影响单轮问答。
    """
    getter = getattr(ctx.repository, "get_messages", None)
    if getter is None:
        return []
    try:
        rows = await getter(thread_id, limit=ctx.settings.agent_summary_after_turns * 2)
    except Exception:
        logger.warning("chat.history_failed", thread_id=thread_id, exc_info=True)
        return []
    return [{"role": r.role, "content": r.content} for r in rows]


def _to_chat_response(thread_id: str, state: dict) -> ChatResponse:
    chunks = {c.get("rank"): c for c in (state.get("chunks") or [])}
    # ★ 拒答时**必须清空引用**。检索一定召回了东西（稠密通道一次 top-50），
    #   但这些块和问题不相关 —— 挂出去等于告诉用户"答案出自这里"，
    #   而答案恰恰是"知识库里没有"。前端也靠 citations 为空来渲染"无相关内容"。
    refused = (state.get("route") or "") == "refuse"
    citations = []
    for c in ([] if refused else state.get("citations") or []):
        source = chunks.get(c.get("rank"), {})
        citations.append(CitationOut(
            index=int(c.get("rank") or 0),
            chunk_id=int(c.get("chunk_id") or 0),
            label=source.get("label") or _label(source),
            quote=c.get("quote") or "",
        ))
    return ChatResponse(
        thread_id=thread_id,
        answer=state.get("answer") or "",
        citations=citations,
        verified=bool(state.get("verified")),
        route=state.get("route") or "",
        grade_reason=state.get("grade_reason") or "",
        retries=int(state.get("retries") or 0),
        diagnostics=state.get("retrieval_diagnostics") or {},
    )


def _label(chunk: dict) -> str:
    parts = [chunk.get("doc_title") or f"doc-{chunk.get('document_id', '?')}"]
    if chunk.get("section_path"):
        parts.append(chunk["section_path"])
    if chunk.get("page_start"):
        parts.append(f"p{chunk['page_start']}")
    return " · ".join(parts)


async def _sse(ctx: AppContext, payload: ChatRequest, thread_id: str):
    """SSE 事件流。

    ★ 事件类型分开而不是只推 token：
      前端需要在答案之前先拿到引用列表（用来渲染角标），
      也需要在最后拿到 verified 决定要不要打"未核验"的提示。
      全塞进一个 text 事件里，前端就得自己解析半成品 JSON。

    ★ 生成节点目前是**一次性**产出 answer 的（LangGraph 的节点是原子的，
      要真流式得用 astream_events 或把 generate 拆成子图）。
      这里的做法是先跑完图，再把答案切片推出去 —— 前端拿到的是同样的
      逐字效果，但首字延迟等于整个图的耗时。见文件末尾 TODO。
    """
    def event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield event("start", {"thread_id": thread_id})

    try:
        state = await _run(ctx, payload, thread_id)
    except asyncio.CancelledError:
        # 客户端断开。★ 必须显式处理：不捕获的话会往上冒泡，
        # 在 uvicorn 日志里刷一堆无意义的 traceback。
        logger.info("chat.sse_client_gone", thread_id=thread_id)
        raise
    except Exception as exc:
        logger.exception("chat.sse_failed", thread_id=thread_id)
        yield event("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    response = _to_chat_response(thread_id, state)
    yield event("citations", {"citations": [c.model_dump() for c in response.citations]})

    # 切片推送：按标点切，避免把一个 UTF-8 字符切成两半（SSE 是文本协议，
    # 切错会出现乱码方块）。中文按字推，每 8 个字一批。
    answer = response.answer
    for i in range(0, len(answer), 8):
        yield event("delta", {"text": answer[i:i + 8]})
        await asyncio.sleep(0)      # 让出事件循环，否则整段会一次性 flush

    yield event("done", {
        "verified": response.verified, "route": response.route,
        "grade_reason": response.grade_reason, "retries": response.retries,
        "diagnostics": response.diagnostics,
    })


# ======================================================================
# 文档
# ======================================================================
@router.post("/api/v1/documents", response_model=IngestResponse, tags=["documents"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008 — FastAPI parameter declaration
    title: str | None = Form(default=None),
    sync: bool = Query(
        default=True,
        description="true=请求内同步解析（小文件方便）；false=只入库并交给 worker 异步处理",
    ),
) -> IngestResponse:
    ctx = _ctx(request)
    if ctx.ingestion is None:
        # 503 而不是 500：这是"依赖没就绪"，等 tokenizer 修好后重试有意义
        raise ServiceUnavailableError(
            "摄取服务不可用：分块器未初始化（通常是 tokenizer 加载失败，见启动日志）"
        )

    settings = ctx.settings
    settings.ensure_dirs()

    # ★ 文件名来自客户端，必须消毒：`../../etc/passwd` 会让上传写到仓库外面。
    #   Path(name).name 只取最后一段，把目录成分全部丢掉。
    safe_name = Path(file.filename or "upload.bin").name or "upload.bin"
    dest = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    size = 0
    limit = settings.upload_max_bytes
    try:
        with dest.open("wb") as fh:
            # 分块写并累计大小 —— 不能先 read() 全读进来再判断，
            # 那样一个 2GB 的文件会先把内存吃光。
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    # 413 而不是 422：客户端该在**发之前**就知道太大，
                    # 分开的状态码让前端能给出"文件过大，请压缩"而不是"格式错误"
                    raise PayloadTooLargeError(
                        f"文件超过上限 {settings.upload_max_mb}MB（已读 {size // 1048576}MB）"
                    )
                fh.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        if not sync:
            job = await ctx.repository.enqueue_job(
                tenant_id=1, document_id=None, job_type="ingest",
                # ★ idempotency_key 用文件内容哈希而不是文件名：
                #   同名不同内容会被误判为重复，不同名同内容会重复解析 ——
                #   两种都是错，而哈希恰好都对。
                idempotency_key=_sha256_file(dest),
                payload={"path": str(dest), "title": title, "source_uri": safe_name},
            )
            return IngestResponse(
                document_id=0, job_id=job.id, status="queued",
            )

        result = await ctx.ingestion.ingest_file(dest, title=title, source_uri=safe_name)
    except Exception:
        # 同步路径失败时保留文件，方便复现；异步路径失败由 worker 记错误。
        # 不删的理由：解析失败的 PDF 是排查 OCR / 编码问题的唯一线索。
        logger.exception("upload.failed", file=safe_name, size=size)
        raise

    return IngestResponse(
        document_id=result.document_id,
        status=result.status,
        deduplicated=result.deduplicated,
        chunk_count=result.chunk_count,
        vectors_written=result.vectors_written,
        page_count=result.page_count,
        duration_ms=result.duration_ms,
        warnings=result.warnings,
    )


@router.get("/api/v1/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    repo = _ctx(request).repository
    docs = await repo.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        total=len(docs), items=[_doc_out(d) for d in docs],
    )


@router.get("/api/v1/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
async def get_document(document_id: int, request: Request) -> DocumentOut:
    doc = await _ctx(request).repository.get_document(document_id)
    if doc is None:
        raise NotFoundError(f"文档不存在：{document_id}")
    return _doc_out(doc)


@router.delete("/api/v1/documents/{document_id}", tags=["documents"])
async def delete_document(document_id: int, request: Request) -> dict:
    ctx = _ctx(request)
    if ctx.ingestion is None:
        raise ServiceUnavailableError("摄取服务不可用")
    await ctx.ingestion.delete_document(document_id)
    return {"document_id": document_id, "status": "deleted"}


@router.get("/api/v1/jobs/{job_id}", tags=["documents"])
async def get_job(job_id: int, request: Request) -> dict:
    job = await _ctx(request).repository.get_job(job_id)
    if job is None:
        raise NotFoundError(f"任务不存在：{job_id}")
    # ★ 字段名用 JobRecord 的原名（attempt / last_error），不要在这里改名 ——
    #   改名的代价是 OpenAPI 文档和实际模型对不上，排查时要来回翻代码。
    return {
        "id": job.id, "status": job.status, "document_id": job.document_id,
        "job_type": job.job_type, "attempt": job.attempt,
        "max_attempts": job.max_attempts, "error": job.last_error,
        "result": job.result,
    }


def _doc_out(d) -> DocumentOut:  # noqa: ANN001
    return DocumentOut(
        id=d.id, title=d.title, status=d.status,
        mime_type=d.mime_type or "", lang=d.lang,
        page_count=d.page_count or 0, chunk_count=d.chunk_count or 0,
        size_bytes=d.size_bytes or 0, error_message=d.error_message,
        created_at=str(getattr(d, "created_at", "") or ""),
        updated_at=str(getattr(d, "updated_at", "") or ""),
    )
