"""FastAPI 应用入口。

    uvicorn rag.main:app --host 0.0.0.0 --port 8000     # 或 AutoDL 的 6006

★ lifespan 是唯一做装配的地方（见 api/deps.py）。模块导入期什么都不做 ——
  在 import 时加载模型会让 `alembic`、`pytest --collect-only` 这类
  只想读一下代码的工具也卡几十秒，而且报错栈完全指不到问题。

★ 错误 → HTTP 状态码的映射只在这里定义一次。handler 里不写 try/except，
  这样"什么错返回什么码"不会出现每个接口各一套的情况。
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.api.deps import build_context
from rag.api.routes import router
from rag.core.config import get_settings
from rag.core.errors import DomainError
from rag.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level, json_output=not settings.is_dev)
    settings.ensure_dirs()

    logger.info("app.starting", env=settings.app_env, backend=settings.vector_backend)
    ctx = await build_context(settings)
    app.state.ctx = ctx
    try:
        yield
    finally:
        # ★ 放 finally：即使 yield 期间抛异常（比如 uvicorn 被 Ctrl-C），
        #   也要走完关闭流程。否则 LangGraph checkpointer 的 psycopg 连接
        #   和向量库客户端不会被释放 —— 用例是本地反复起停时连接数一直涨，
        #   最后 Postgres 报 "too many clients"。
        await ctx.shutdown()
        logger.info("app.stopped")


app = FastAPI(
    title="RAG-Agent",
    version="0.1.0",
    description=(
        "面向 PDF / Word / Markdown 的知识库问答服务。\n\n"
        "**混合检索**（BGE 稠密 + BM25 稀疏 → RRF 融合 → cross-encoder 重排）"
        " + **LangGraph Agent**（检索 → 证据判定 → 改写重试 → 生成 → 引用校验）。"
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def _request_id(request: Request, call_next):
    """给每个请求打一个 id，回写响应头。

    ★ 作用是把"用户看到的 500"和"日志里的那一条 traceback"对上。
      没有它的话，用户报"刚才失败了"，你只能在几千行日志里猜是哪次。
      客户端传来的 X-Request-Id 优先复用 —— 上游网关通常已经生成了一个，
      再生成一个会让跨服务追踪断链。
    """
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.exception_handler(DomainError)
async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
    """领域异常 → RFC 9457 Problem Details。

    ★ 状态码取自**异常类自己的 `status` 属性**，不在这里维护第二张映射表。
      维护两张表的必然结果是它们会漂移 —— 改了一处忘了另一处，
      表现是"明明抛的是 404，返回的却是 500"，而且没有任何报错提示。
      想改状态码就去改 errors.py 里的类定义，那里是唯一事实源。

    ★ detail 是否外泄由异常类自己的 `expose_detail` 决定，不在这里按状态码拍板。
      理由见 errors.py：用 `status >= 500` 判断会把 503 里"未配置 LLM_PROVIDER"
      这类**专门写给用户看**的提示也一起抹掉，前端只能显示一句
      "Service unavailable"，等于没给任何线索。
    """
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "http.domain_error", status=exc.status, path=request.url.path,
        error_type=type(exc).__name__, detail=exc.detail, request_id=request_id,
    )
    problem = exc.to_problem(instance=request.url.path, request_id=request_id)
    if not exc.expose_detail:
        problem["detail"] = exc.title
    return JSONResponse(
        status_code=exc.status,
        content=problem,
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预期的异常一律 500，且**绝不**把 `str(exc)` 返回给客户端。

    ★ 这是最常见的泄漏路径：数据库连接串（含密码）、模型文件的绝对路径、
      甚至 API key 都可能出现在异常消息里。详细信息进日志（已脱敏），
      客户端只拿到异常类型名，方便对着日志查。
    """
    logger.exception("http.unhandled_error", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://rag-agent.local/problems/internal-error",
            "title": "Internal error",
            "status": 500,
            "detail": f"服务内部错误（{type(exc).__name__}），详情见服务端日志",
        },
        media_type="application/problem+json",
    )


app.include_router(router)

# ★ 前端挂载必须放在 include_router **之后**。
#   Starlette 按注册顺序匹配路由，mount("/") 会吃掉所有未匹配的路径；
#   放到前面的话 /api/v1/* 和 /readyz 全都会被它截走，变成 404。
#
#   用 StaticFiles 而不是单独起一个前端服务：单进程、单端口、无 Node 构建，
#   和"只在本地跑"的定位一致，演示时只需要开一个口子。
_STATIC_DIR = Path(__file__).parent / "api" / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    _s = get_settings()
    uvicorn.run("rag.main:app", host="0.0.0.0", port=_s.api_port, reload=_s.is_dev)
