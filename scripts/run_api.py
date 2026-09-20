"""启动 API —— 比裸 `uvicorn rag.main:app` 多传一个事件循环工厂。

## 为什么需要这个入口

LangGraph 的 Postgres checkpointer（多轮对话的断点续跑）走 psycopg，
而 psycopg 的 async 模式**跑不了** Windows 默认的 ProactorEventLoop。
`deps._setup_checkpointer()` 的设计是"失败就降级、别让服务起不来"，
于是异常被吞掉，`/readyz` 只报一句 `checkpointer: off（多轮对话无断点续跑）`，
看起来像"没配"，实际是"装了但起不来"。表现是同一个 thread_id 追问，
模型回答"这是我本次对话中收到的第一条消息"。

★ 设 `asyncio.set_event_loop_policy()` **没用** —— uvicorn ≥0.36 不用策略了，
  改用循环工厂，且 win32 分支硬编码 ProactorEventLoop。
  详见 `rag/core/loop.py` 的说明。

Linux 上没有 Proactor，Docker / AutoDL 用裸 uvicorn 就行。

用法：
    python scripts/run_api.py                # 等价于 uvicorn rag.main:app
    python scripts/run_api.py --port 8001
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

LOOP_FACTORY = "rag.core.loop:platform_loop_factory"


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 RAG API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    uvicorn.run(
        "rag.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        loop=LOOP_FACTORY,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
