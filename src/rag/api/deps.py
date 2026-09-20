"""应用级依赖装配。

★ 这里做的事情只有一件：**把配置翻译成对象，并且只翻译一次。**

   provider / 向量库 / 仓储 / 分块器 / 两个 service / agent 图，全部在 lifespan
   里构造好挂在 app.state 上。请求处理时直接取，不重新构造。

   为什么不能在每个请求里构造：
   ① 本地 embedding 模型加载一次要几十秒、占 1.3GB —— 每请求一次直接不可用
   ② 数据库连接池、httpx client 都是长生命周期对象，重建会泄漏连接
   ③ agent 图编译一次有成本（虽然不高），但 checkpointer 必须复用

★ 关闭顺序与构造顺序**相反**，而且是必须的：
   先关 checkpointer（psycopg 连接）→ 再关向量库 → 最后关仓储的连接池。
   顺序错了会看到"连接已被关闭"这类报错，而且是在**下一次请求**里才炸，
   排查时会往业务代码上找，其实是关闭顺序的问题。

   注：worker 有自己的装配（见 rag/worker/runner.py），不走这里 ——
   它不需要 agent 图、checkpointer 和 reranker，构造它们纯属浪费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.core.config import Settings
from rag.core.errors import ServiceUnavailableError
from rag.core.logging import get_logger
from rag.infra import build_repository, build_vector_store
from rag.providers import build_embedding_provider, build_llm, build_reranker
from rag.providers.base import is_enabled
from rag.services.academic import AcademicService
from rag.services.ingestion import IngestionService
from rag.services.retrieval import RetrievalService
from rag.services.translate import build_translator

logger = get_logger(__name__)


@dataclass
class AppContext:
    """所有长生命周期对象的容器。挂在 `app.state.ctx` 上。"""

    settings: Settings
    embedder: Any = None
    reranker: Any = None
    llm: Any = None
    translator: Any = None
    store: Any = None
    repository: Any = None
    retrieval: RetrievalService | None = None
    ingestion: IngestionService | None = None
    academic: AcademicService | None = None
    graph: Any = None
    agent_harness: Any = None
    checkpointer: Any = None
    _saver_cm: Any = None          # AsyncExitStack，必须留住引用否则会被 GC 关掉
    ready_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    async def startup(self) -> None:
        settings = self.settings

        # ---- 1. 模型 provider ----
        # ★ 顺序有讲究：embedding 放最前，因为它的错误信息最可能是配置问题
        #   （路径写错 / 维度不符），早失败早报错。
        self.embedder = build_embedding_provider(settings)
        self.reranker = build_reranker(settings)
        self.llm = build_llm(settings)
        # 翻译复用同一个 LLM 客户端，只是指定便宜模型。
        # ★ 单独存成 translator 而不是每次现建：它带翻译缓存，
        #   建成临时对象的话缓存每请求清零，等于没有。
        self.translator = build_translator(
            self.llm,
            target=settings.retrieve_translate_target,
            model=settings.retrieve_translate_model or settings.llm_model_cheap,
        )

        # ---- 2. 存储 ----
        self.store = build_vector_store(settings)
        self.repository = build_repository(settings)

        # ★ 存储的就绪检查各自 try：向量库连不上不该阻止 API 起来 ——
        #   /healthz 要能返回（容器编排靠它判断存活），具体哪个组件坏由 /readyz 报。
        #   反过来，起不来就退出进程的话，k8s 会无限重启而你看不到日志里的原因。
        try:
            await self.store.ensure_ready()
        except Exception as exc:
            self.ready_errors.append(f"vectorstore: {type(exc).__name__}: {exc}")
            logger.exception("startup.vectorstore_failed")
        try:
            await self.repository.ensure_ready()
        except Exception as exc:
            self.ready_errors.append(f"repository: {type(exc).__name__}: {exc}")
            logger.exception("startup.repository_failed")

        # ---- 3. 分块器 ----
        # ★ tokenizer 从模型目录读，所以它依赖 embedder 就绪。
        #   API 路径用不到 chunker（摄取在 worker 里），但 worker 复用同一套装配，
        #   所以这里允许失败并降级 —— 让 API 在没装 tokenizer 的镜像里也能起。
        chunker = None
        try:
            chunker = self._build_chunker()
        except Exception as exc:
            self.ready_errors.append(f"chunker: {type(exc).__name__}: {exc}")
            logger.warning("startup.chunker_unavailable", error=str(exc))

        # ---- 4. service ----
        self.retrieval = RetrievalService(
            embedder=self.embedder,
            store=self.store,
            repository=self.repository,
            reranker=self.reranker,
            translator=self.translator,
            translate_default=settings.retrieve_translate,
            translate_target=settings.retrieve_translate_target,
            rrf_k=settings.rrf_k,
            dense_top_k=settings.retrieve_dense_top_k,
            sparse_top_k=settings.retrieve_sparse_top_k,
            fused_top_k=settings.retrieve_fused_top_k,
            final_top_k=settings.retrieve_final_top_k,
        )
        self.academic = AcademicService(self.repository)
        if chunker is not None:
            self.ingestion = IngestionService(
                repository=self.repository,
                store=self.store,
                embedder=self.embedder,
                chunker=chunker,
                chunker_version=settings.chunker_version,
            )

        # ---- 5. checkpointer ----
        await self._setup_checkpointer()

        # ---- 6. agent 图 ----
        from rag.agent.graph import build_graph

        self.graph = build_graph(
            retrieval=self.retrieval, llm=self.llm, checkpointer=self.checkpointer,
            top_k=settings.retrieve_final_top_k, max_retries=settings.agent_max_retries,
        )

        # ---- 7. V2 Agent Harness ----
        # 旧 LangGraph 保留；/chat 由轻量 Pi loop 组合 RAG 与教务只读工具。
        from rag.agent.academic_tools import (
            CheckScheduleConflictTool,
            QueryExamTool,
            QueryGradesTool,
            QueryScheduleTool,
            SearchCoursesTool,
        )
        from rag.agent.agent import PiAgent
        from rag.agent.harness import AgentHarness
        from rag.agent.tools import SearchKnowledgeTool

        tools = [
            SearchKnowledgeTool(self.retrieval),
            SearchCoursesTool(self.academic),
            QueryGradesTool(self.academic),
            QueryScheduleTool(self.academic),
            QueryExamTool(self.academic),
            CheckScheduleConflictTool(self.academic),
        ]
        pi_agent = PiAgent(llm=self.llm, tool_definitions=[tool.definition for tool in tools])
        self.agent_harness = AgentHarness(
            agent=pi_agent,
            tools=tools,
            task_store=self.repository,
            max_iterations=settings.agent_max_iterations,
            tool_timeout_seconds=settings.agent_tool_timeout_seconds,
        )

        # ---- 8. 预热 ----
        await self._warmup()

        logger.info(
            "startup.done",
            vector_backend=settings.vector_backend,
            repository_backend=settings.repository_backend,
            embed=settings.embed_provider, rerank=settings.rerank_provider,
            llm=settings.llm_provider, checkpointer=self.checkpointer is not None,
            ready_errors=len(self.ready_errors),
        )

    # ------------------------------------------------------------------
    def _build_chunker(self):  # noqa: ANN202
        # ★ 构造逻辑在 chunking 包里 —— worker 用的是同一份，不能各写一套
        #   （分块参数漂移会让 API 同步路径和 worker 异步路径产出不同的块）
        from rag.chunking import build_chunker

        return build_chunker(self.settings)

    async def _setup_checkpointer(self) -> None:
        """LangGraph 的 Postgres checkpointer。

        ★ 它**必须**和 repository 用同一个库，但连接方式完全不同：
          repository 走 asyncpg（SQLAlchemy），checkpointer 走 psycopg。
          两套驱动连同一个 Postgres 是完全正常的，不要试图统一 ——
          langgraph-checkpoint-postgres 只认 psycopg。

        ★ 失败时降级为 None（无 checkpoint），而不是让服务起不来：
          代价是多轮对话失去断点续跑，但问答本身仍然可用。
        """
        if self.settings.repository_backend != "postgres":
            logger.warning("checkpointer.skipped", reason="repository_backend != postgres")
            return
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # from_conn_string 返回的是 async context manager，
            # ★ 必须手动 __aenter__ 并保存引用 —— 写成 `async with` 的话
            #   退出块时连接就被关了，图执行时才报错。
            cm = AsyncPostgresSaver.from_conn_string(self._psycopg_dsn())
            saver = await cm.__aenter__()
            await saver.setup()
            self.checkpointer = saver
            self._saver_cm = cm
        except Exception as exc:
            self.ready_errors.append(f"checkpointer: {type(exc).__name__}: {exc}")
            logger.exception("checkpointer.setup_failed")

    def _psycopg_dsn(self) -> str:
        """psycopg 用的是 libpq DSN，不是 SQLAlchemy URL —— 驱动前缀必须去掉。"""
        from urllib.parse import quote_plus

        s = self.settings
        if s.database_url_override:
            # postgresql+asyncpg://... → postgresql://...
            return s.database_url_override.replace("+asyncpg", "").replace("+psycopg", "")
        pwd = quote_plus(s.postgres_password.get_secret_value())
        return (
            f"postgresql://{s.postgres_user}:{pwd}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
        )

    async def _warmup(self) -> None:
        """预热 embedding 模型。

        ★ 冷启动加载本地模型要几十秒。不预热的话，第一个用户请求会挂在那里，
          而且大概率触发客户端超时 —— 表现为"服务刚起来时第一次查询必失败"。
        """
        if not self.settings.embed_warmup_on_startup:
            return
        if not hasattr(self.embedder, "ensure_loaded"):
            return
        try:
            await self.embedder.ensure_loaded()
            logger.info("warmup.embedding_ready")
        except Exception as exc:
            self.ready_errors.append(f"embedding_warmup: {type(exc).__name__}: {exc}")
            logger.exception("warmup.failed")

    # ------------------------------------------------------------------
    async def readyz(self) -> list[tuple[str, bool, str]]:
        """逐组件健康检查。返回 (名字, 是否就绪, 说明)。"""
        out: list[tuple[str, bool, str]] = []

        # 向量库
        try:
            n = await self.store.count()
            out.append(("vectorstore", True, f"{n} vectors"))
        except Exception as exc:
            out.append(("vectorstore", False, f"{type(exc).__name__}: {exc}"))

        # 关系库
        try:
            await self.repository.count_chunks()
            out.append(("repository", True, self.settings.repository_backend))
        except Exception as exc:
            out.append(("repository", False, f"{type(exc).__name__}: {exc}"))

        # embedding
        ready = getattr(self.embedder, "is_ready", None)
        if ready is None:
            # API provider 没有 is_ready —— 探一下真的能不能编码
            try:
                await self.embedder.aembed_query("ping")
                out.append(("embedding", True, self.settings.embed_provider))
            except Exception as exc:
                out.append(("embedding", False, f"{type(exc).__name__}: {exc}"))
        else:
            out.append(("embedding", bool(ready), self.settings.embed_model_id))

        # reranker
        # ★ 这一项**必须存在**：前端靠它决定"重排"勾选框是否可用
        #   （见 static/app.js 里 find(c => c.name === 'reranker')）。
        #   少了它，前端拿到 undefined 就落到 else 分支，把一个**已经配好、
        #   正在生效**的重排器报成"未配置重排器：RERANK_PROVIDER=none"，
        #   还会把勾选框禁用掉 —— 功能是好的，界面在说谎。
        #   与 embedding 同理：api provider 没有 is_ready，真探一次。
        if not is_enabled(self.reranker):
            out.append(("reranker", False, "未配置（RERANK_PROVIDER=none，排序用 RRF 名次）"))
        else:
            ready = getattr(self.reranker, "is_ready", None)
            if ready is None:
                try:
                    await self.reranker.arerank("ping", ["ping"])
                    out.append(("reranker", True, self.settings.rerank_provider))
                except Exception as exc:
                    out.append(("reranker", False, f"{type(exc).__name__}: {exc}"))
            else:
                out.append(("reranker", bool(ready), self.settings.rerank_provider))

        # LLM 只报状态，不真调 —— /readyz 会被探针高频调用，
        # 每次打一次 LLM 既花钱又慢，还会让就绪状态依赖第三方可用性。
        out.append(("llm", True if is_enabled(self.llm) else False,
                    self.settings.llm_provider if is_enabled(self.llm) else "未配置（/chat 将返回 503）"))
        out.append(("checkpointer", self.checkpointer is not None,
                    "on" if self.checkpointer is not None else "off（多轮对话无断点续跑）"))
        return out

    # ------------------------------------------------------------------
    async def shutdown(self) -> None:
        # 顺序与 startup 相反
        if self._saver_cm is not None:
            try:
                await self._saver_cm.__aexit__(None, None, None)
            except Exception:
                logger.warning("shutdown.checkpointer_failed", exc_info=True)
            self._saver_cm = None

        for name, closer in (("store", self.store), ("repository", self.repository),
                             ("embedder", self.embedder)):
            if closer is None:
                continue
            aclose = getattr(closer, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:
                logger.warning("shutdown.close_failed", component=name, exc_info=True)

        logger.info("shutdown.done")


async def build_context(settings: Settings) -> AppContext:
    ctx = AppContext(settings=settings)
    await ctx.startup()
    return ctx


def require_llm(ctx: AppContext) -> None:
    """需要 LLM 的接口入口处调用。

    ★ 返回 503 而不是 500：这是配置缺失（用户没填 LLM_API_KEY），
      不是服务端 bug。503 让调用方知道"重试或改配置"，500 会让人去翻日志找异常。
    """
    from rag.providers.base import is_enabled

    if not is_enabled(ctx.llm):
        raise ServiceUnavailableError(
            "未配置 LLM。设置 LLM_PROVIDER=openai 与 LLM_API_KEY 后重试。"
            "（检索接口 /retrieve 不受影响）"
        )


__all__ = ["AppContext", "build_context", "require_llm"]
