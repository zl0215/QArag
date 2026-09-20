"""Worker 冒烟 —— 验证任务生命周期，不需要 Postgres / Milvus / 模型。

    python scripts/smoke_worker.py

★ 为什么这个脚本值得存在：
  worker 是**唯一一个没有 HTTP 表面的组件**。API 出问题你能 curl，
  worker 出问题你只有一个"任务一直排队"的现象和一堆安静的日志。
  所以它的每一条状态迁移都得在本地先打一遍：

    排队 → 领取 → 成功 / 失败 → （重试 → 排队）| （判死）
    崩溃（租约过期）→ 回收 → 重新排队

★ 它用 MemoryRepository + MemoryVectorStore + hash 假向量：
  这里验的是**任务调度逻辑**，不是解析质量也不是召回质量。
  真链路（真解析 + 真分块 + 真嵌入）由 scripts/smoke_local.py 负责。
  两者刻意分开：混在一起的话，一个分块 bug 会把调度逻辑的回归也带红。

★ 它**绕过** run_worker() 的 preflight 直接构造 Worker：
  preflight 会拒绝 memory 后端（独立进程看不见内存队列，这个拒绝是对的），
  但同一个进程内直接构造 Worker 时内存队列完全可用 —— 这正是测试要的。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.core.config import Settings  # noqa: E402
from rag.core.errors import ProviderError  # noqa: E402
from rag.infra.models import JobStatus  # noqa: E402
from rag.infra.repository import MemoryRepository  # noqa: E402
from rag.services.ingestion import IngestionService  # noqa: E402
from rag.worker import Worker, preflight_problems  # noqa: E402

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def eq(name: str, got, want) -> None:  # noqa: ANN001
    check(name, got == want, f"期望 {want!r}，实际 {got!r}")


async def build_worker(tmp: Path) -> tuple[Worker, MemoryRepository]:
    """构造一个用内存后端的 Worker，并注入可控的分块器/嵌入器。

    ★ settings 先读真实的 .env（拿到本机 tokenizer 路径 —— 分块器需要它，
      而 Chunker 是 worker 的硬依赖，缺了会直接抛 FileNotFoundError），
      再覆盖掉存储与嵌入这两层，让测试不碰网络也不碰真模型。
    """
    base = Settings()
    settings = base.model_copy(update={
        "app_env": "test",
        "repository_backend": "memory",
        "vector_backend": "memory",
        "embed_provider": "hash",     # blake2b 假向量：不加载模型，毫秒级
        "embed_dim": 64,
        "embed_warmup_on_startup": False,
        "upload_dir": tmp,
    })

    worker = Worker(settings, worker_id="smoke-worker", once=True)

    from rag.chunking import build_chunker
    from rag.infra.vectorstore import MemoryVectorStore
    from rag.providers import build_embedding_provider

    worker.embedder = build_embedding_provider(settings)
    worker.store = MemoryVectorStore(dim=settings.embed_dim)
    worker.repository = MemoryRepository()
    await worker.store.ensure_ready()
    await worker.repository.ensure_ready()
    worker.ingestion = IngestionService(
        repository=worker.repository,
        store=worker.store,
        embedder=worker.embedder,
        chunker=build_chunker(settings),
        chunker_version=settings.chunker_version,
    )
    return worker, worker.repository


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_worker_smoke_"))
    upload_dir = tmp / "uploads"
    upload_dir.mkdir(parents=True)

    sample = ROOT / "data" / "samples" / "deploy_guide.md"
    if not sample.exists():
        print(f"[FAIL] 缺少样例文件 {sample}")
        return 2
    doc = upload_dir / "guide.md"
    doc.write_bytes(sample.read_bytes())

    print("=" * 68)
    print(" Worker 冒烟（内存后端，不碰网络与真模型）")
    print("=" * 68)

    worker, repo = await build_worker(upload_dir)

    # ------------------------------------------------------------------
    print("\n[1] preflight：该拒绝的配置有没有拒")
    problems = preflight_problems(worker.settings)
    eq("memory 后端被拒绝", len(problems), 1)
    check("拒绝理由提到了 postgres", "postgres" in problems[0])

    lite_settings = worker.settings.model_copy(update={"milvus_uri": "./data/milvus.db"})
    lite_problems = preflight_problems(lite_settings)
    check("Milvus Lite 被拒绝（单进程文件锁）",
          any("Lite" in p or "lite" in p for p in lite_problems),
          f"{len(lite_problems)} 条")
    # ★ 这个出口必须验：AutoDL 上 Lite 是**默认**配置，没有这个口子
    #   就等于 worker 在那条路上完全用不了，整个模块变成死代码。
    #   注意这里**不能**断言"问题清零" —— 这份 settings 的
    #   repository_backend 还是 memory，那条拒绝是对的、也该继续留着。
    #   要验的是"Lite 那一条没了"。
    still = preflight_problems(lite_settings, allow_lite=True)
    check("--allow-lite 放行了 Lite（要显式开）",
          not any("Lite" in p for p in still),
          f"剩余 {len(still)} 条：{still}")
    check("放行 Lite 不影响其它检查（memory 仍被拒）",
          any("REPOSITORY_BACKEND" in p for p in still))

    # ------------------------------------------------------------------
    print("\n[2] 正常路径：领取 → 摄取 → 成功")
    job = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="hash-1",
        payload={"path": str(doc), "title": "部署手册", "source_uri": "guide.md"},
    )
    eq("任务初始状态", job.status, JobStatus.QUEUED)

    await worker.run()

    done = await repo.get_job(job.id)
    eq("任务状态", done.status, JobStatus.SUCCEEDED)
    check("document_id 被回填（客户端的 job 轮询才拿得到）",
          done.document_id not in (None, 0), f"document_id={done.document_id}")
    check("result 里有 chunk_count",
          (done.result or {}).get("chunk_count", 0) > 0,
          f"result={done.result}")
    eq("租约已释放", done.locked_until, None)

    # 同一份文件再传一次（换个幂等键，绕开 enqueue 层的去重）：
    # 应该命中 content_hash 复用，而不是重新解析一遍
    job2 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="hash-2",
        payload={"path": str(doc), "title": "部署手册", "source_uri": "guide.md"},
    )
    await worker.run()
    done2 = await repo.get_job(job2.id)
    eq("重复上传命中哈希去重", (done2.result or {}).get("deduplicated"), True)
    eq("去重后指向同一个文档", done2.document_id, done.document_id)

    # ------------------------------------------------------------------
    print("\n[3] 永久性错误：不重试，直接判死")
    missing = upload_dir / "not-there.md"
    job3 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="missing", payload={"path": str(missing)},
    )
    await worker.run()
    done3 = await repo.get_job(job3.id)
    eq("文件不存在 → failed", done3.status, JobStatus.FAILED)
    eq("只尝试了一次（没有白白重试三遍）", done3.attempt, 1)
    check("错误信息说明了原因", "不存在" in (done3.last_error or ""),
          done3.last_error)

    # ------------------------------------------------------------------
    print("\n[4] 路径越界：payload 里的路径不能跑出上传目录")
    outside = tmp / "secret.md"          # 在 uploads/ 之外，但在 tmp 之内
    outside.write_text("不该被解析", encoding="utf-8")
    job4 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="escape", payload={"path": str(outside)},
    )
    await worker.run()
    done4 = await repo.get_job(job4.id)
    eq("越界路径 → failed", done4.status, JobStatus.FAILED)
    check("错误信息点明了越界", "上传目录之外" in (done4.last_error or ""),
          done4.last_error)

    # ------------------------------------------------------------------
    print("\n[5] 可重试错误：退避重排，不是直接判死")
    async def _boom(*_a, **_kw):  # noqa: ANN202
        raise ProviderError("向量库连接被重置（模拟抖动）")

    real_ingest = worker.ingestion.ingest_file
    worker.ingestion.ingest_file = _boom  # type: ignore[method-assign]
    job5 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="flaky", payload={"path": str(doc)},
    )
    await worker.run()
    done5 = await repo.get_job(job5.id)
    eq("ProviderError → 回到 queued", done5.status, JobStatus.QUEUED)
    eq("attempt 已 +1", done5.attempt, 1)
    check("next_run_at 被推到未来（退避）",
          done5.next_run_at is not None and done5.next_run_at > __import__("datetime").datetime.now(__import__("datetime").UTC),
          f"next_run_at={done5.next_run_at}")
    check("last_error 记下了原因", "抖动" in (done5.last_error or ""))
    worker.ingestion.ingest_file = real_ingest  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    print("\n[6] 重试耗尽：max_attempts 用完后判死")
    worker.ingestion.ingest_file = _boom  # type: ignore[method-assign]
    job6 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest", max_attempts=1,
        idempotency_key="exhaust", payload={"path": str(doc)},
    )
    await worker.run()
    done6 = await repo.get_job(job6.id)
    eq("次数用尽 → failed", done6.status, JobStatus.FAILED)
    eq("不再重排", done6.attempt, 1)
    worker.ingestion.ingest_file = real_ingest  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    print("\n[7] 未知 job_type：判死而不是无限重试")
    job7 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="reindex",
        idempotency_key="unknown", payload={},
    )
    await worker.run()
    done7 = await repo.get_job(job7.id)
    eq("未知类型 → failed", done7.status, JobStatus.FAILED)
    check("错误信息列出了支持的类型", "ingest" in (done7.last_error or ""),
          done7.last_error)

    # ------------------------------------------------------------------
    print("\n[8] 租约过期回收：模拟 worker 被 kill -9")
    import datetime as dt

    job8 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="killed", payload={"path": str(doc)},
    )
    # 手工把它伪造成"某个已崩溃的 worker 领走后失联"的状态
    stale = repo._jobs[job8.id]  # noqa: SLF001
    stale.status = JobStatus.RUNNING
    stale.locked_by = "dead-worker"
    stale.locked_until = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)

    # 还没到期的不能被抢走
    fresh = repo._jobs[done.id]  # noqa: SLF001
    fresh.status = JobStatus.RUNNING
    fresh.locked_by = "alive-worker"
    fresh.locked_until = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=600)

    n = await repo.requeue_stale_jobs(lock_timeout_seconds=900)
    eq("只回收了过期的那一条", n, 1)
    eq("崩溃的任务回到 queued", (await repo.get_job(job8.id)).status, JobStatus.QUEUED)
    eq("活着的 worker 手上的任务没被动", (await repo.get_job(done.id)).status,
       JobStatus.RUNNING)

    await worker.run()
    done8 = await repo.get_job(job8.id)
    eq("回收后能被重新处理并成功", done8.status, JobStatus.SUCCEEDED)

    # ------------------------------------------------------------------
    print("\n[9] 关停：置停止位后 shutdown 不报错")
    await worker.shutdown()
    check("停止位已置起", worker._stop.is_set())  # noqa: SLF001

    print("\n" + "=" * 68)
    if _failures:
        print(f" X {len(_failures)} 项未通过：")
        for f in _failures:
            print(f"     - {f}")
    else:
        print(" OK 全部通过 —— 任务生命周期可用")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
