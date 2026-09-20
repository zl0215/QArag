"""后台 worker：从 Postgres 任务表领取并执行摄取任务。

    python -m rag.worker           # 常驻，容器里的默认用法
    python -m rag.worker --once    # 把当前排队的跑完就退出（补数据 / 冒烟）

★ 它是**独立进程**，和 API 同镜像不同命令（compose.yaml 里 worker 服务
  覆盖了 command，其余配置全部继承）。这样两边用的依赖版本、
  provider 实现、分块参数天然一致 —— 不会出现"API 分块和 worker 分块不一样"。
"""

from rag.worker.runner import (
    Worker,
    is_retryable,
    preflight_problems,
    retry_delay_seconds,
    run_worker,
)

__all__ = [
    "Worker",
    "is_retryable",
    "preflight_problems",
    "retry_delay_seconds",
    "run_worker",
]
