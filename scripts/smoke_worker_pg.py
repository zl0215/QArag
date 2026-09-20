"""Postgres 任务队列冒烟 —— 验内存后端**验不了**的那部分。

    SMOKE_DATABASE_URL='postgresql+asyncpg://rag:pw@localhost:5432/rag_smoke' \\
        python scripts/smoke_worker_pg.py

★ 为什么必须单独有这个脚本：
  scripts/smoke_worker.py 用 MemoryRepository，验的是**调度逻辑**
  （什么时候重试、什么时候判死、租约过期怎么回收）。
  但真正承载"任务表替代 Celery"这个决定的，是 Postgres 那三条裸 SQL：

      claim_job          —— SELECT … FOR UPDATE SKIP LOCKED
      schedule_retry     —— 退避重排
      requeue_stale_jobs —— 崩溃恢复

  这些**一行都没在内存后端上执行过**。内存实现是单进程里的一个 dict 加锁，
  它天然不会出现"两个 worker 领到同一条"；Postgres 下会不会，只能真连上去试。
  尤其是 FOR UPDATE SKIP LOCKED —— 它是最容易被写错、
  又最不会立刻暴露（只在并发时才偶发）的一处。

★ 为什么必须先有数据库：
  它**不会**自己建库 —— 建库要连到 postgres 库、要 CREATEDB 权限，
  那些是部署的事（见 docs/DEPLOY.md），不该藏在冒烟脚本里。
  只建**表**（ensure_ready → create_all），而且只建在测试库里。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from rag.infra.db import Database  # noqa: E402
from rag.infra.models import JobStatus  # noqa: E402
from rag.infra.repository import PostgresRepository  # noqa: E402

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def eq(name: str, got, want) -> None:  # noqa: ANN001
    check(name, got == want, f"期望 {want!r}，实际 {got!r}")


# ----------------------------------------------------------------------
# ★ 安全闸：这个脚本会 create_all + 删数据。库名不对就拒绝跑。
#   判据是"库名里必须带 smoke 或 test" —— 代价是库名得规规矩矩，
#   收益是永远不可能因为复制粘贴错一行 URL 就把生产库清了。
# ----------------------------------------------------------------------
def guard(url: str) -> str:
    name = url.rsplit("/", 1)[-1].split("?")[0]
    if not any(tag in name.lower() for tag in ("smoke", "test")):
        raise SystemExit(
            f"拒绝执行：库名 {name!r} 里没有 smoke/test。\n"
            "这个脚本会建表并清空 ingestion_jobs —— 只能用测试库。"
        )
    return name


async def main() -> int:
    url = os.environ.get("SMOKE_DATABASE_URL") or ""
    if not url:
        print("需要 SMOKE_DATABASE_URL，例如：")
        print("  SMOKE_DATABASE_URL='postgresql+asyncpg://rag:pw@localhost:5432/rag_smoke' \\")
        print("      python scripts/smoke_worker_pg.py")
        return 2
    dbname = guard(url)

    print("=" * 70)
    print(f" Postgres 任务队列冒烟（库：{dbname}）")
    print("=" * 70)

    repo = PostgresRepository(Database(url))

    # ------------------------------------------------------------------
    print("\n[1] ensure_ready：建表（with_checkpointer=False，worker 的用法）")
    # ★ 传 False 是在验 worker 的真实路径。API 走的是 True（它要 checkpointer），
    #   worker 不走 —— 这个参数如果没生效，worker 会白建一遍 checkpointer 表，
    #   还要多占一条 psycopg 连接。
    await repo.ensure_ready(with_checkpointer=False)
    check("建表成功（不建 checkpointer 表）", True)

    # ------------------------------------------------------------------
    print("\n[0] 清空测试库 —— 让每次运行都从同一个已知状态开始")
    # ★ 这一步是**必须**的，不是洁癖。第一版只在结尾清理，
    #   结果上一轮跑到一半挂了，残留的 pg-basic 任务停在 running；
    #   这一轮的 enqueue_job 命中幂等键，把那条**旧任务**返回了，
    #   于是"[2] 初始状态应该是 queued"报成了 'running' ——
    #   看着像代码 bug，其实是脏数据。
    #   开头清一次，这个脚本才真正可重复运行。
    async with repo._db.session() as s:  # noqa: SLF001
        await s.execute(text("DELETE FROM ingestion_jobs"))
        await s.execute(text("DELETE FROM chunks"))
        await s.execute(text("DELETE FROM documents"))
        await s.commit()
    check("残留数据已清空", True)

    # ------------------------------------------------------------------
    print("\n[2] enqueue：next_run_at 必须非 NULL")
    # ★ 这一条专门盯一个坑：claim_job 的 SQL 是 `next_run_at <= now()`，
    #   而 SQL 里 NULL <= now() 的结果是 NULL（不是 true）——
    #   所以 next_run_at 一旦是 NULL，这条任务就**永远领不到**，
    #   且不会有任何报错。现在靠模型上的 server_default=func.now() 兜住，
    #   但那是数据库行为，不实测就只能靠信念。
    job = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-basic", payload={"path": "/tmp/x.md"},
    )
    check("next_run_at 被数据库默认值填上了（不是 NULL）",
          job.next_run_at is not None, f"next_run_at={job.next_run_at}")
    eq("初始状态 queued", job.status, JobStatus.QUEUED)
    eq("初始 attempt=0", job.attempt, 0)

    # ------------------------------------------------------------------
    print("\n[3] claim：领取后的字段")
    claimed = await repo.claim_job("worker-A")
    check("领到了任务", claimed is not None and claimed.id == job.id)
    eq("状态 → running", claimed.status, JobStatus.RUNNING)
    eq("attempt 已 +1", claimed.attempt, 1)
    eq("locked_by 记上了", claimed.locked_by, "worker-A")
    check("locked_until 在未来（租约已建立）",
          claimed.locked_until is not None and claimed.locked_until > claimed.next_run_at,
          f"locked_until={claimed.locked_until}")

    print("\n[4] 队列空时 claim 返回 None（不能一直领到同一批）")
    eq("再领一次是 None", await repo.claim_job("worker-A"), None)

    # ------------------------------------------------------------------
    print("\n[5] finish_job：成功收尾 + 释放租约 + 回填 document_id")
    # ★ 必须先建一个真文档：ingestion_jobs.document_id 上**有外键**指向
    #   documents(id)，随便编一个 id 会被数据库直接拒掉
    #   （第一次跑这个脚本就是这么撞上的）。
    #   这不是测试的麻烦 —— 它是条有用的约束：任务不可能关联到一个
    #   不存在的文档，孤儿 job 在 schema 层面就被排除了。
    doc = await repo.create_document(
        tenant_id=1, title="冒烟文档", source_uri="smoke.md",
        mime_type="text/markdown", content_hash="smoke-hash-1",
    )
    await repo.finish_job(job.id, status="succeeded",
                          result={"chunk_count": 3}, document_id=doc.id)
    done = await repo.get_job(job.id)
    eq("状态 → succeeded", done.status, JobStatus.SUCCEEDED)
    eq("document_id 回填了", done.document_id, doc.id)
    eq("租约释放：locked_by", done.locked_by, None)
    eq("租约释放：locked_until", done.locked_until, None)
    check("result 存下来了", (done.result or {}).get("chunk_count") == 3, f"{done.result}")

    # ------------------------------------------------------------------
    print("\n[6] finish_job 不传 document_id 时，不能把已有的覆盖成 NULL")
    # ★ 这条是回归测试：values 字典里无条件塞 document_id 的话，
    #   一次不带 document_id 的收尾就会把之前回填的 id 抹掉 ——
    #   表现是"任务成功了，但文档永远关联不上"。
    await repo.finish_job(job.id, status="succeeded", result={"chunk_count": 3})
    again = await repo.get_job(job.id)
    eq("document_id 还在", again.document_id, doc.id)

    # ------------------------------------------------------------------
    print("\n[7] 幂等键：同一个 key 重复 enqueue 返回同一条")
    dup = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-basic", payload={"path": "/tmp/x.md"},
    )
    eq("没有新建，拿回原来那条", dup.id, job.id)

    # ------------------------------------------------------------------
    print("\n[8] 幂等键：产物被删掉之后，同一个 key 必须重新排队")
    # ★ 这是"删掉文档、再上传同一个文件"走的路。
    #   如果只看任务状态就返回，用户会拿到那条陈旧的 succeeded ——
    #   接口报"已完成"，而知识库里空空如也。这个 bug 在界面上表现为
    #   "上传成功但列表里没有东西"，排查时会一直怀疑 worker。
    j8 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-reupload", payload={"path": "/tmp/y.md"},
    )
    await repo.finish_job(j8.id, status=JobStatus.SUCCEEDED,
                          result={"chunk_count": 1}, document_id=doc.id)
    hit = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-reupload", payload={},
    )
    eq("产物还在时走幂等，拿回原任务", hit.id, j8.id)
    eq("状态仍是 succeeded", hit.status, JobStatus.SUCCEEDED)

    await repo.soft_delete_document(doc.id)
    again8 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-reupload", payload={"path": "/tmp/y.md"},
    )
    eq("产物被删后复用同一条（唯一约束不允许新建）", again8.id, j8.id)
    eq("状态被重置回 queued", again8.status, JobStatus.QUEUED)
    eq("attempt 清零，重试预算不缩水", again8.attempt, 0)
    eq("旧的 result 被清掉（否则前端先看到上次的结果）", again8.result, None)
    eq("租约已释放", again8.locked_until, None)

    # ------------------------------------------------------------------
    print("\n[9] 幂等键：旧任务 failed 时重传，不能撞唯一约束")
    # ★ 曾经的 bug：failed 不在短路条件里，代码于是落到 INSERT，
    #   而 idempotency_key 上有唯一约束（uq_jobs_idempotency）
    #   → IntegrityError → "上传一个曾经解析失败的文件"直接返回 500。
    j9 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-failed-retry", payload={"path": "/tmp/z.md"},
    )
    await repo.finish_job(j9.id, status=JobStatus.FAILED, error="模拟一次失败")
    again9 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-failed-retry", payload={"path": "/tmp/z.md"},
    )
    eq("复用同一条，而不是抛 IntegrityError", again9.id, j9.id)
    eq("状态回到 queued", again9.status, JobStatus.QUEUED)
    eq("last_error 清空", again9.last_error, None)
    eq("attempt 清零", again9.attempt, 0)

    # ★ 收尾：上面两节故意把任务重置成了 queued（那正是被测行为），
    #   但留着它们会污染后面的小节 —— [11] 断言"退避期内 claim 不到"，
    #   而 claim 会优先领走这两条真正可领的任务，于是断言失败。
    #   测试之间的这种串扰排查起来很费劲（失败点离原因隔了三节），
    #   所以在这里显式置回终态。
    await repo.finish_job(j8.id, status=JobStatus.SUCCEEDED, result={})
    await repo.finish_job(j9.id, status=JobStatus.FAILED, error="本节收尾")

    # ------------------------------------------------------------------
    print("\n[10] schedule_retry：退避重排")
    j8 = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-retry", payload={},
    )
    await repo.claim_job("worker-A")
    import datetime as dt

    before = dt.datetime.now(dt.UTC)
    await repo.schedule_retry(j8.id, error="ProviderError: 抖动", delay_seconds=60)
    r8 = await repo.get_job(j8.id)
    eq("状态 → queued", r8.status, JobStatus.QUEUED)
    eq("租约已释放", r8.locked_until, None)
    check("next_run_at 被推到未来 60 秒左右",
          r8.next_run_at is not None
          and (r8.next_run_at - before).total_seconds() > 50,
          f"next_run_at={r8.next_run_at}")

    print("\n[11] 退避中的任务领不到（这是退避能生效的前提）")
    # ★ 如果忘了在 claim 的 WHERE 里加 next_run_at <= now()，
    #   退避就形同虚设：任务刚被推迟就立刻又被领走，
    #   变成忙等重试，把 CPU 和数据库连接全烧掉。
    eq("退避期内 claim 不到它", await repo.claim_job("worker-A"), None)

    # ------------------------------------------------------------------
    print("\n[12] 并发领取：FOR UPDATE SKIP LOCKED 的核心验证")
    # ★ 这是整个脚本最重要的一条。
    #   20 个任务、8 个 worker 同时抢，每个抢若干轮 ——
    #   如果 FOR UPDATE SKIP LOCKED 写漏了（比如把 SELECT 和 UPDATE
    #   拆成两个事务），就会出现两个 worker 领到同一条，
    #   表现是同一份文档被摄取两遍（重复向量 + 重复正文）。
    total = 20
    for i in range(total):
        await repo.enqueue_job(
            tenant_id=1, document_id=None, job_type="ingest",
            idempotency_key=f"pg-conc-{i}", payload={"i": i},
        )

    claimed_ids: list[int] = []
    lock = asyncio.Lock()

    async def racer(worker: str) -> None:
        while True:
            got = await repo.claim_job(worker)
            if got is None:
                return
            async with lock:
                claimed_ids.append(got.id)

    await asyncio.gather(*(racer(f"worker-{i}") for i in range(8)))

    eq(f"{total} 个任务全部被领走", len(claimed_ids), total)
    eq("没有一条被重复领取", len(set(claimed_ids)), total)
    check("队列已空", await repo.claim_job("worker-Z") is None)

    # ------------------------------------------------------------------
    print("\n[13] requeue_stale_jobs：崩溃恢复")
    # 伪造"某个 worker 领走后被 kill -9"：状态 running + 租约已过期
    stale = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-stale", payload={},
    )
    await repo.claim_job("dead-worker")

    async with repo._db.session() as s:  # noqa: SLF001
        await s.execute(
            text("UPDATE ingestion_jobs SET locked_until = now() - interval '1 hour' "
                 "WHERE id = :i"),
            {"i": stale.id},
        )
        await s.commit()

    # 一个"还活着"的 worker 手上的任务（租约未过期），不能被误伤
    live = await repo.enqueue_job(
        tenant_id=1, document_id=None, job_type="ingest",
        idempotency_key="pg-live", payload={},
    )
    await repo.claim_job("alive-worker")

    n = await repo.requeue_stale_jobs(lock_timeout_seconds=900)
    eq("只回收了过期的那一条", n, 1)
    eq("崩溃的任务回到 queued", (await repo.get_job(stale.id)).status, JobStatus.QUEUED)
    eq("活着的 worker 手上的没被动",
       (await repo.get_job(live.id)).status, JobStatus.RUNNING)

    print("\n[14] 回收后能被重新领取")
    again2 = await repo.claim_job("worker-B")
    check("重新领到了那条崩溃的任务",
          again2 is not None and again2.id == stale.id,
          f"领到 id={again2.id if again2 else None}")

    # ------------------------------------------------------------------
    print("\n[15] 清理")
    # 顺序不能反：jobs 和 chunks 都有外键指向 documents，先删子表。
    # ★ 这一步必须做干净 —— documents 上有
    #   uq_documents_hash_version(tenant_id, content_hash, version, deleted_at)，
    #   残留一条就会让**下一次**运行在 create_document 那步撞唯一约束，
    #   而报错信息完全看不出是上一轮没清干净。
    async with repo._db.session() as s:  # noqa: SLF001
        await s.execute(text("DELETE FROM ingestion_jobs"))
        await s.execute(text("DELETE FROM chunks"))
        await s.execute(text("DELETE FROM documents"))
        await s.commit()
    check("测试数据已清空（只删测试库的表）", True)

    await repo.aclose()

    print("\n" + "=" * 70)
    if _failures:
        print(f" X {len(_failures)} 项未通过：")
        for f in _failures:
            print(f"     - {f}")
    else:
        print(" OK 全部通过 —— Postgres 任务队列可用")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
