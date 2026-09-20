"""uvicorn 的事件循环工厂。

## 为什么需要这个文件

LangGraph 的 Postgres checkpointer 走 psycopg，而 psycopg 的 **async 模式
不支持 Windows 默认的 ProactorEventLoop**：

    psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run
    in async mode. Please use ... WindowsSelectorEventLoopPolicy()

直觉是 `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`，
但**对这个版本的 uvicorn 没用**：uvicorn ≥0.36 不再用事件循环策略，
改成在 `config.get_loop_factory()` 里挑一个"工厂"，而它的 win32 分支是硬编码的：

    def asyncio_loop_factory(use_subprocess=False):
        if sys.platform == "win32" and not use_subprocess:
            return asyncio.ProactorEventLoop      # ← 无视 policy
        return asyncio.SelectorEventLoop

所以只能把工厂整个换掉。uvicorn 允许 `loop` 传任意 "模块:属性" 路径，
它会把那个属性**直接当工厂调用**（注意：不走 `use_subprocess=...` 那条分支，
所以这里必须是无参可调用对象）。

用法（见 `scripts/run_api.py`）：

    uvicorn.run("rag.main:app", loop="rag.core.loop:platform_loop_factory")

Linux 上没有 Proactor，Docker / AutoDL 用裸 uvicorn 完全正常 ——
这个工厂在非 Windows 上返回平台默认循环，行为不变。
"""

from __future__ import annotations

import asyncio
import sys


def platform_loop_factory() -> asyncio.AbstractEventLoop:
    """给 uvicorn 用的无参循环工厂。

    Windows → SelectorEventLoop（psycopg async 唯一能跑的）；
    其他平台 → `asyncio.new_event_loop()`，即平台默认（装了 uvloop 就是 uvloop）。
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
