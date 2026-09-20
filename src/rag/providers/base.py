"""Provider 协议定义。

用 Protocol 而不是 ABC：结构化子类型，实现类不需要显式继承，
测试里替换成 fake 也更自然。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# ★ "该 provider 未启用"的统一哨兵值。
#   所有空实现（NullLLM / NoopReranker）都用它，判断逻辑只有 is_enabled 一处。
#   之前这个字符串在两个文件里各写了一遍（"null" vs "none"），
#   导致"未启用"被当成"已启用"，降级路径静默失效。
DISABLED_MODEL_ID = "none"


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral tool call returned by a chat model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMChatResponse:
    """One assistant turn, including zero or more native tool calls."""

    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = ""


def is_enabled(provider: Any) -> bool:
    """provider 是否真的可用。

    用 `model_id != DISABLED_MODEL_ID` 而不是 `provider is not None`：
    空实现是**合法的构造结果**，配置成 none 时工厂返回的就是它。
    这样检索/生成链路不需要为"有没有模型"分叉，A/B 对比走同一条代码路径。
    """
    return provider is not None and getattr(provider, "model_id", DISABLED_MODEL_ID) != DISABLED_MODEL_ID


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_id: str
    dim: int

    @property
    def max_tokens(self) -> int | None:
        """模型最大输入长度；用于约束分块上限。未知返回 None。"""
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档向量。★ 不要加 query instruction —— BGE 的非对称检索要求只给 query 加。"""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class Reranker(Protocol):
    model_id: str

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """返回 (原始下标, 分数)，按分数降序。"""
        ...


@runtime_checkable
class LLMClient(Protocol):
    async def achat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMChatResponse: ...

    async def acomplete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str: ...

    async def astructured(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel: ...

    def astream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...
