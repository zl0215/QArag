"""结构化日志。

★ 两个必须遵守的纪律：
1. contextvars 是 async-safe 的，但上下文会泄漏到下一个请求 —— 必须 try/finally 清理
2. 脱敏 processor 必须排在 JSONRenderer 之前，否则密钥会先被序列化进日志
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# ★ 脱敏规则的粒度问题（踩过）：
#   最初用 ("token", "auth", ...) 直接做子串匹配，结果 avg_tokens、token_count、
#   max_tokens、author 全被替换成 ***REDACTED*** —— 日志里最需要的分块统计
#   变成了星号，而真正的密钥字段反而不缺这一条保护。
#
#   现在分两类：
#     _SENSITIVE_SUBSTRINGS —— 语义无歧义，子串命中即脱敏
#     _SENSITIVE_EXACT      —— 有歧义的短词，只在**整个键名完全相同**时脱敏
#   带凭证含义的复合词（access_token / api_key）走子串表，
#   而 token / key / auth 单独出现时才认为是凭证。
_SENSITIVE_SUBSTRINGS = (
    "password", "passwd", "secret", "api_key", "apikey", "api-key",
    "authorization", "credential", "cookie", "private_key",
    "access_token", "refresh_token", "auth_token", "id_token", "bearer",
)
_SENSITIVE_EXACT = frozenset({"token", "auth", "key", "pwd", "passwd", "secret"})

_REDACTED = "***REDACTED***"


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SENSITIVE_EXACT:
        return True
    return any(part in lowered for part in _SENSITIVE_SUBSTRINGS)


def _redact_secrets(_logger: Any, _method: str, event_dict: dict) -> dict:
    """在渲染之前把敏感字段替换掉。"""
    for key in list(event_dict.keys()):
        if _is_sensitive(key):
            event_dict[key] = _REDACTED
    return event_dict


def setup_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """进程启动时调用一次。"""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # 这些库在 INFO 级别过于聒噪
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus", "neo4j", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,   # ★ 必须最前
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_secrets,                            # ★ 必须在 renderer 之前
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_request_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """★ 必须在请求结束时调用，否则上下文会泄漏到下一个请求。"""
    structlog.contextvars.clear_contextvars()
