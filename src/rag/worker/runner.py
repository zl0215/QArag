"""后台 worker：从任务表领取任务并执行。

    python -m rag.worker

★ 为什么要有独立进程：
  摄取（解析 → 分块 → 嵌入）是**分钟级**的重活。放在 API 请求里做的话，
  要么客户端等到超时，要么 uvicorn 的并发位被长时间占死。
  `POST /api/v1/documents?sync=false` 就是走这条路：请求只入库并返回 job_id，
  客户端拿 job_id 去轮询 `GET /api/v1/jobs/{id}`。

★ 为什么不用 Celery：
  Postgres 的 `FOR UPDATE SKIP LOCKED` 已经提供了原子领取 + 崩溃重领，
  不需要再引入 Redis/RabbitMQ 那一套中间件。取舍的完整论证见
  `infra/repository.py` 顶部。

★ 本模块**故意不复用** `api/deps.py` 的 AppContext：
  worker 不需要 agent 图、LangGraph checkpointer、reranker、LLM。
  构造它们纯属浪费 —— checkpointer 还会白占一条 psycopg 连接。
  真正复用的边界是 providers / infra / services / chunking 四层的工厂函数，
  这些才是"两边必须一致"的部分（尤其是分块参数）。

★ 两条启动就该拒绝的配置（见 `preflight_problems`）：
  1. `REPOSITORY_BACKEND=memory` —— 跨进程看不见的内存队列，
     worker 会一直空转。表现是"任务一直排队、日志里什么都没有"。
  2. Milvus Lite + API 同时跑 —— 一个 data_dir 只能被一个进程打开。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rag.core.config import Settings, get_settings
from rag.core.errors import IngestionError, UnprocessableDocumentError
from rag.core.logging import get_logger, setup_logging
from rag.infra import build_repository, build_vector_store
from rag.infra.repository import (
    DEFAULT_LOCK_SECONDS,
    JobRecord,
    Repository,
    new_worker_id,
)
from rag.providers import build_embedding_provider
from rag.services.ingestion import IngestionService

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 重试策略
# ----------------------------------------------------------------------
# ★ 不是所有错误都值得重试。区分标准是"再跑一次结果会不会不一样"：
#   文件不存在、解析出来是空的 —— 再跑一百次还是同一个结果，
#   重试只会把一次失败拖成三次失败，还占着队列。
#   ProviderError（向量库/模型服务抖动）和未知异常则值得重试 ——
#   未知异常偏保守：一个偶发的 bug 不该静默吞掉用户上传的文档，
#   而 max_attempts 会兜住无限重试。
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    IngestionError,
    UnprocessableDocumentError,
)

RETRY_BASE_SECONDS = 5.0
RETRY_MAX_SECONDS = 300.0


def retry_delay_seconds(attempt: int) -> float:
    """指数退避：5s → 10s → 20s → … 封顶 5 分钟。

    ★ 封顶是必须的。不封顶的话第 10 次重试要等 85 分钟，
      而且 max_attempts 一旦被调大就会失控。
    """
    return min(RETRY_BASE_SECONDS * (2 ** max(attempt - 1, 0)), RETRY_MAX_SECONDS)


def is_retryable(exc: BaseException) -> bool:
    return not isinstance(exc, PERMANENT_ERRORS)


# ----------------------------------------------------------------------
# 启动前检查
# ----------------------------------------------------------------------
def preflight_problems(settings: Settings, *, allow_lite: bool = False) -> list[str]:
    """返回启动前就已知的致命配置问题。空列表 = 可以启动。

    ★ 单独抽成纯函数是为了能直接测 —— 这些错误的共同特征是
      **运行起来之后完全看不出原因**，所以宁可在这里拦死。

    ★ allow_lite：Milvus Lite 的拒绝是可以显式关掉的（`--allow-lite`）。
      默认拦死是因为最常见的用法（worker 常驻 + API 常驻）必然抢文件锁，
      而那个报错发生在**后启动的那个进程**上，症状是"刚开始好好的，
      重启一次就起不来了"，很难往文件锁上想。
      但"停掉 API → 批量补数据 → 再起 API"是个完全正当的用法：
      比拿几百个 HTTP 请求去打一个持锁的 API 好得多。所以留一个明说的出口。
    """
    problems: list[str] = []

    if settings.repository_backend != "postgres":
        problems.append(
            f"REPOSITORY_BACKEND={settings.repository_backend}。"
            "worker 是**独立进程**，看不见 API 进程里的内存队列 —— "
            "任务会永远停在排队状态，而 worker 日志里一片安静。"
            "请设 REPOSITORY_BACKEND=postgres。"
        )

    if settings.milvus_is_lite and not allow_lite:
        problems.append(
            f"MILVUS_URI={settings.milvus_uri} 是 Milvus Lite 的本地目录。"
            "Lite 一个 data_dir 同时只能被**一个进程**打开 —— "
            "worker 和 API 抢同一个文件，后启动的那个会起不来。"
            "三选一：① 只跑 API，用 ?sync=true 同步摄取（小文件够用）；"
            "② 停掉 API，用 `--once --allow-lite` 批量补完数据再起 API；"
            "③ 换 standalone（MILVUS_URI=http://...）再开 worker。"
        )

    return problems


# ----------------------------------------------------------------------
class Worker:
    """轮询任务表并执行。一个进程一个实例，多开几个就是水平扩容。"""

    def __init__(
        self,
        settings: Settings,
        *,
        worker_id: str | None = None,
        once: bool = False,
        lock_timeout_seconds: int = DEFAULT_LOCK_SECONDS,
    ) -> None:
        self.settings = settings
        # worker_id 会写进 ingestion_jobs.locked_by —— 排查"这条任务谁在处理"时
        # 全靠它。同名的话多副本之间就分不出来了，所以默认带随机后缀。
        self.worker_id = worker_id or new_worker_id()
        # once=True：把当前排队的跑完就退出。给 cron / 一次性补数据用，
        # 也让冒烟脚本不必去 kill 一个死循环。
        self.once = once
        self.lock_timeout_seconds = lock_timeout_seconds

        self.repository: Repository | None = None
        self.store: Any = None
        self.embedder: Any = None
        self.ingestion: IngestionService | None = None

        self._stop = asyncio.Event()
        self._handlers: dict[str, Callable[[JobRecord], Awaitable[dict]]] = {
            "ingest": self._handle_ingest,
        }

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    async def startup(self, *, attempts: int = 3, delay_seconds: float = 3.0) -> None:
        """构造长生命周期对象并等待依赖就绪。

        ★ 这里和 API 的 startup **刻意不同**：API 连不上向量库也要起来
          （/healthz 得能返回，否则容器编排会无限重启，日志都看不到）；
          但 worker 连不上依赖就是个废物进程 —— 重试几次后直接退出，
          交给容器的 restart 策略做退避，比"假装活着然后空转"诚实。
        """
        settings = self.settings

        self.embedder = build_embedding_provider(settings)
        self.store = build_vector_store(settings)
        self.repository = build_repository(settings)

        # 分块器：worker 的核心依赖，缺了就什么都干不了（不像 API 可以只服务检索）
        from rag.chunking import build_chunker

        chunker = build_chunker(settings)
        self.ingestion = IngestionService(
            repository=self.repository,
            store=self.store,
            embedder=self.embedder,
            chunker=chunker,
            chunker_version=settings.chunker_version,
        )

        await self._wait_for_dependencies(attempts=attempts, delay_seconds=delay_seconds)

        # ★ 预热 embedding：不预热的话第一个任务要多等几十秒模型加载，
        #   而这几十秒会被算进"某一条任务特别慢"，很容易误判成那条文档有问题。
        if settings.embed_warmup_on_startup and hasattr(self.embedder, "ensure_loaded"):
            try:
                await self.embedder.ensure_loaded()
                logger.info("worker.embedding_ready")
            except Exception:
                logger.exception("worker.warmup_failed")

        logger.info(
            "worker.started", worker_id=self.worker_id,
            repository=settings.repository_backend,
            vector=settings.milvus_uri,
            embed=settings.embed_provider,
            once=self.once,
        )

    async def _wait_for_dependencies(self, *, attempts: int, delay_seconds: float) -> None:
        last: Exception | None = None
        for i in range(1, attempts + 1):
            try:
                await self.store.ensure_ready()
                # with_checkpointer=False：worker 不跑 agent 图
                await self.repository.ensure_ready(with_checkpointer=False)
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning(
                    "worker.dependency_not_ready", attempt=i, of=attempts,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if i < attempts:
                    # 用 wait_for(_stop) 而不是裸 sleep：Ctrl-C 时能立刻退出，
                    # 不然关一个正在重试的 worker 要等满一个 sleep 周期
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=delay_seconds)
                    if self._stop.is_set():
                        break
        raise SystemExit(
            f"worker 启动失败：依赖在 {attempts} 次重试后仍不可用 —— "
            f"{type(last).__name__}: {last}"
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self) -> None:
        assert self.repository is not None, "先调用 startup()"

        # 启动先扫一次：上一个 worker 如果是被 kill -9 的，它手上的任务
        # 还挂在 running 上，不捞回来的话要等满一个租约周期才会重跑。
        await self._reap_stale(force=True)

        idle_rounds = 0
        while not self._stop.is_set():
            claimed = 0
            for _ in range(max(self.settings.worker_batch_size, 1)):
                if self._stop.is_set():
                    break
                job = await self.repository.claim_job(self.worker_id)
                if job is None:
                    break
                claimed += 1
                await self._process(job)

            if self.once and claimed == 0:
                logger.info("worker.once_drained")
                return

            if claimed == 0:
                idle_rounds += 1
                # 周期性回收过期租约。不必每轮都做 —— 它是个全表 UPDATE，
                # 闲时每轮跑一次纯属浪费。忙的时候（claimed>0）不触发，
                # 因为那说明有活人在干活，没有需要救的任务。
                if idle_rounds % self._reap_every(idle_rounds) == 0:
                    await self._reap_stale()
                await self._sleep(self.settings.worker_poll_interval_seconds)
            else:
                idle_rounds = 0

    def _reap_every(self, idle_rounds: int) -> int:
        """多少轮空闲回收一次过期租约。

        目标是"约每分钟一次"，但不能写死轮数 —— 轮询间隔是可配的，
        写死 30 轮的话间隔设成 10 秒就变成 5 分钟才回收一次。
        """
        interval = max(self.settings.worker_poll_interval_seconds, 0.1)
        return max(int(60 / interval), 1)

    async def _reap_stale(self, *, force: bool = False) -> None:
        assert self.repository is not None
        try:
            n = await self.repository.requeue_stale_jobs(
                lock_timeout_seconds=self.lock_timeout_seconds
            )
            if n and not force:
                logger.info("worker.requeued_stale", count=n)
        except Exception:
            # 回收失败不该让 worker 退出 —— 它是**恢复**手段，不是主链路
            logger.exception("worker.requeue_stale_failed")

    async def _sleep(self, seconds: float) -> None:
        """可被打断的 sleep —— 关服务时不用等满一个轮询周期。"""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)

    # ------------------------------------------------------------------
    # 任务处理
    # ------------------------------------------------------------------
    async def _process(self, job: JobRecord) -> None:
        assert self.repository is not None
        handler = self._handlers.get(job.job_type)
        if handler is None:
            # 未知 job_type 直接判失败，不重试 —— 重试一万次也还是没有这个处理器
            msg = (
                f"未知的任务类型 {job.job_type!r}，本 worker 支持："
                f"{sorted(self._handlers)}"
            )
            logger.error("worker.unknown_job_type", job_id=job.id, job_type=job.job_type)
            await self.repository.finish_job(job.id, status="failed", error=msg)
            return

        logger.info("worker.job_start", job_id=job.id, job_type=job.job_type,
                    attempt=job.attempt, of=job.max_attempts)
        try:
            result, document_id = await handler(job)
        except Exception as exc:  # noqa: BLE001
            await self._on_failure(job, exc)
            return

        await self.repository.finish_job(
            job.id, status="succeeded", result=result, document_id=document_id,
        )
        logger.info("worker.job_done", job_id=job.id, document_id=document_id, **result)

    async def _on_failure(self, job: JobRecord, exc: BaseException) -> None:
        """失败处理：能重试就退避重排，不能重试（或次数用尽）就判死。

        ★ `job.attempt` 已经在 claim_job 里 +1 过了，所以这里直接用它判断。
          语义是"这是第 attempt 次尝试，还有 max_attempts - attempt 次机会"。
        """
        assert self.repository is not None
        error = f"{type(exc).__name__}: {exc}"[:2000]

        if not is_retryable(exc):
            logger.error("worker.job_permanent_failure", job_id=job.id, error=error)
            await self.repository.finish_job(job.id, status="failed", error=error)
            return

        if job.attempt >= job.max_attempts:
            logger.error("worker.job_exhausted", job_id=job.id,
                         attempt=job.attempt, of=job.max_attempts, error=error)
            await self.repository.finish_job(job.id, status="failed", error=error)
            return

        delay = retry_delay_seconds(job.attempt)
        logger.warning("worker.job_retry", job_id=job.id, attempt=job.attempt,
                       of=job.max_attempts, delay_seconds=delay, error=error,
                       exc_info=True)
        await self.repository.schedule_retry(job.id, error=error, delay_seconds=delay)

    # ------------------------------------------------------------------
    async def _handle_ingest(self, job: JobRecord) -> tuple[dict, int]:
        """`job_type="ingest"`：把上传的文件跑完整条摄取管道。

        返回 (写入 result 字段的摘要, document_id)。
        """
        assert self.ingestion is not None
        payload = job.payload or {}

        raw_path = payload.get("path")
        if not raw_path:
            raise IngestionError(f"任务 {job.id} 的 payload 里没有 path：{payload!r}")

        path = self._safe_upload_path(str(raw_path))

        result = await self.ingestion.ingest_file(
            path,
            title=payload.get("title"),
            source_uri=payload.get("source_uri"),
        )
        return (
            {
                "chunk_count": result.chunk_count,
                "vectors_written": result.vectors_written,
                "deduplicated": result.deduplicated,
                "duration_ms": result.duration_ms,
            },
            result.document_id,
        )

    def _safe_upload_path(self, raw_path: str) -> Path:
        """把 payload 里的路径限制在上传目录内。

        ★ payload 存在数据库列里，不是可信输入：任何能写 ingestion_jobs 的人
          （SQL 注入、误操作、将来的管理接口）都能让 worker 去解析任意路径的文件。
          worker 是跑在服务器上的高权限进程，这里必须收口。

        ★ 用 resolve() 之后再比较 —— 只做字符串前缀判断的话，
          `/data/uploads/../../etc/shadow` 这种能直接绕过。
        """
        upload_dir = self.settings.upload_dir.resolve()
        path = Path(raw_path).resolve()

        if not path.is_relative_to(upload_dir):
            raise IngestionError(
                f"拒绝处理上传目录之外的文件：{path}（上传目录 {upload_dir}）"
            )
        if not path.is_file():
            # 永久性错误：文件被清理掉了，重试也不会长回来
            raise IngestionError(f"待摄取的文件不存在：{path}")
        return path

    # ------------------------------------------------------------------
    async def shutdown(self) -> None:
        """关停。顺序与构造相反，且**务必先取消轮询再关依赖** ——
        反过来会出现"关到一半又有任务被领走"。
        """
        self._stop.set()
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
                logger.warning("worker.close_failed", component=name, exc_info=True)
        logger.info("worker.stopped", worker_id=self.worker_id)

    # ------------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """SIGTERM/SIGINT → 置停止位，让手头这条任务跑完再退。

        ★ 处理中的任务不强行中断：**半途而废的任务比慢一点更糟** ——
          Postgres 里已经写了 chunks、向量库写了半批，
          留下的是需要人工对账的中间状态。让它跑完，代价只是关得慢一点。
          （真的等不及就 kill -9，租约过期后别的 worker 会重领。）

        ★ Windows 的 ProactorEventLoop 不支持 add_signal_handler，
          这里吞掉 NotImplementedError —— 本地调试用 Ctrl-C 也能退，
          反正生产只跑 Linux。
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, AttributeError):  # pragma: no cover
                logger.debug("worker.signal_handler_unsupported", signal=str(sig))

    def _on_signal(self, sig: signal.Signals) -> None:
        if self._stop.is_set():
            logger.warning("worker.force_exit", signal=str(sig))
            raise SystemExit(1)
        logger.info("worker.stop_requested", signal=str(sig))
        self._stop.set()


async def run_worker(
    settings: Settings | None = None, *, once: bool = False, allow_lite: bool = False
) -> None:
    """进程入口：装配 → 跑 → 关。异常路径也保证走 shutdown。"""
    settings = settings or get_settings()
    setup_logging(settings.log_level, json_output=not settings.is_dev)

    problems = preflight_problems(settings, allow_lite=allow_lite)
    if problems:
        for p in problems:
            logger.error("worker.preflight_failed", problem=p)
        # 打印到 stderr 而不是只进日志：起容器失败时人先看的是终端输出
        print("\n".join(f"[X] {p}" for p in problems), flush=True)
        raise SystemExit(2)

    if allow_lite and settings.milvus_is_lite:
        # 用户已经明说知道风险了，但这句话还是得留 —— 它的作用是**以后**
        # 有人翻日志时能看到"当时是故意这么干的"，而不是怀疑配置写错了。
        logger.warning(
            "worker.lite_lock_overridden",
            milvus_uri=settings.milvus_uri,
            hint="已放行 Milvus Lite；此期间**不能**同时启动 API（同一个 data_dir 只能一个进程）",
        )

    worker = Worker(settings, once=once)
    worker.install_signal_handlers()
    try:
        await worker.startup()
        await worker.run()
    finally:
        await worker.shutdown()
