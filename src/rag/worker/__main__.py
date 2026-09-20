"""`python -m rag.worker` 的入口。

★ 用 `python -m` 而不是 uvicorn 那种 `module:app`：
  worker 没有 ASGI 应用，它就是个进程。compose.yaml 里的
  `command: ["python", "-m", "rag.worker"]` 对应的就是这里。
"""

from __future__ import annotations

import argparse
import asyncio

from rag.worker.runner import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rag.worker",
        description="RAG-Agent 摄取 worker（从 Postgres 任务表领取任务）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="把当前排队的任务跑完后退出，而不是常驻轮询。"
             "用于补数据、CI 冒烟，或配合 cron 定时跑。",
    )
    parser.add_argument(
        "--allow-lite",
        action="store_true",
        help="放行 Milvus Lite（默认拒绝：一个 data_dir 只能被一个进程打开）。"
             "只在**API 已停**的前提下用，典型是 `--once --allow-lite` 批量补数据。",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_worker(once=args.once, allow_lite=args.allow_lite))
    except KeyboardInterrupt:
        # 正常关停路径已经打完了 worker.stopped，这里不再重复报错。
        # 会走到这里只有一种情况：Ctrl-C 落在了信号处理器安装之前。
        pass


if __name__ == "__main__":
    main()
