"""LLM 客户端（OpenAI 兼容协议）。

覆盖 DeepSeek / 硅基流动 / 通义 / vLLM / OpenAI。
★ 结构化输出用 json_object + 手动校验，而不是 json_schema：
  多数国产 OpenAI 兼容端点只支持 json_object，用严格 schema 会直接 400。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from pydantic import BaseModel, ValidationError

from rag.core.errors import ProviderError, ServiceUnavailableError
from rag.core.logging import get_logger
from rag.providers.base import DISABLED_MODEL_ID, LLMChatResponse, LLMToolCall

logger = get_logger(__name__)


class NullLLM:
    """未配置 API Key 时的占位实现。

    让摄取与检索链路在没有 LLM 的情况下依然可用（/chat 返回 503）。

    ★ `model_id` 必须与其他"未启用"实现保持同一个哨兵值（见 providers.base.DISABLED_MODEL_ID）。
      这里曾经写的是 "null"，而判断处写的是 "none" —— 结果 agent 以为 LLM 可用，
      一路走到真实调用才抛异常，把本该走的抽取式降级路径整个跳过。
      两个字符串字面量散在两个文件里，是这类 bug 的温床。
    """

    model_id = DISABLED_MODEL_ID

    async def achat(self, messages: list[dict], **kwargs) -> LLMChatResponse:  # noqa: ANN003
        raise ServiceUnavailableError("未配置 LLM_API_KEY，问答功能不可用")

    async def acomplete(self, messages: list[dict], **kwargs) -> str:  # noqa: ANN003
        raise ServiceUnavailableError("未配置 LLM_API_KEY，问答功能不可用")

    async def astructured(self, messages: list[dict], schema: type[BaseModel], **kwargs):  # noqa: ANN003, ANN201
        raise ServiceUnavailableError("未配置 LLM_API_KEY，问答功能不可用")

    async def astream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:  # noqa: ANN003
        raise ServiceUnavailableError("未配置 LLM_API_KEY，问答功能不可用")
        yield ""  # pragma: no cover —— 让函数成为 async generator


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        cheap_model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
    ) -> None:
        if not base_url:
            raise ServiceUnavailableError("LLM_API_BASE 未配置")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                                   timeout=timeout, max_retries=2)
        # DeepSeek V4 默认开启思考模式，max_tokens 同时计算思考与可见输出。
        # RAG 的改写/判定通常只给 128~256 token，思考可能把预算耗尽，最终
        # content 为空。这里的任务有完整检索上下文，不需要隐藏推理链。
        self._disable_deepseek_thinking = "api.deepseek.com" in base_url.casefold()
        self.model_id = model
        self.cheap_model = cheap_model or model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---------------- 原生 tool calling ----------------
    async def achat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMChatResponse:
        """返回 provider-neutral 的 assistant turn。

        Harness 必须保留 assistant.tool_calls → tool.tool_call_id 的完整链条，
        因此这里不能复用只返回字符串的 ``acomplete``。
        """
        try:
            resp = await self._client.chat.completions.create(
                model=model or self.model_id,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
                **({"tools": tools, "tool_choice": "auto"} if tools else {}),
                **({"extra_body": {"thinking": {"type": "disabled"}}}
                   if self._disable_deepseek_thinking else {}),
            )
        except Exception as exc:
            logger.exception("llm.chat_failed", model=model or self.model_id)
            raise ProviderError(f"LLM 调用失败：{type(exc).__name__}") from exc

        if not resp.choices:
            raise ProviderError("LLM 返回空 choices")

        choice = resp.choices[0]
        message = choice.message
        parsed_calls: list[LLMToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Tool 参数不是合法 JSON：{call.function.name}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ProviderError(f"Tool 参数必须是 JSON 对象：{call.function.name}")
            parsed_calls.append(LLMToolCall(
                id=call.id,
                name=call.function.name,
                arguments=arguments,
            ))

        content = message.content or ""
        if not content.strip() and not parsed_calls:
            raise ProviderError("LLM 返回空响应且没有 tool call")
        return LLMChatResponse(
            content=content,
            tool_calls=parsed_calls,
            finish_reason=choice.finish_reason or "",
        )

    # ---------------- 纯文本 ----------------
    async def acomplete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        for attempt in range(2):
            try:
                resp = await self._client.chat.completions.create(
                    model=model or self.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                    **({"extra_body": {"thinking": {"type": "disabled"}}}
                       if self._disable_deepseek_thinking else {}),
                )
            except Exception as exc:
                logger.exception("llm.complete_failed", model=model or self.model_id)
                raise ProviderError(f"LLM 调用失败：{type(exc).__name__}") from exc

            if resp.choices:
                content = resp.choices[0].message.content or ""
                if content.strip():
                    return content
            logger.warning("llm.empty_response", attempt=attempt + 1,
                           model=model or self.model_id)

        raise ProviderError("LLM 连续返回空响应")

    # ---------------- 结构化输出 ----------------
    async def astructured(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """★ 把 schema 的 JSON Schema 塞进 system prompt，并用 json_object 模式。

        校验失败重试一次，把校验错误回灌给模型 —— 比直接抛异常友好得多。
        """
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        augmented = [
            *messages,
            {
                "role": "system",
                "content": (
                    "只输出一个 JSON 对象，不要任何解释或 Markdown 代码围栏。"
                    f"必须符合以下 JSON Schema：\n{schema_hint}"
                ),
            },
        ]

        last_error: Exception | None = None
        for attempt in range(2):
            raw = await self.acomplete(augmented, model=model, json_mode=True)
            try:
                return schema.model_validate_json(_strip_fence(raw))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("llm.structured_invalid", attempt=attempt + 1,
                               error=str(exc)[:200])
                augmented.append({"role": "assistant", "content": raw})
                augmented.append({
                    "role": "user",
                    "content": f"上面的 JSON 不符合 schema，错误：{exc}。请重新只输出合法 JSON。",
                })

        raise ProviderError(f"结构化输出校验失败：{last_error}")

    # ---------------- 流式 ----------------
    async def astream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=model or self.model_id,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
        except Exception as exc:
            logger.exception("llm.stream_open_failed")
            raise ProviderError(f"LLM 流式调用失败：{type(exc).__name__}") from exc

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            logger.exception("llm.stream_broken")
            raise ProviderError(f"LLM 流中断：{type(exc).__name__}") from exc


def _strip_fence(raw: str) -> str:
    """有些模型即使说了不要围栏也会加 ```json ... ```。"""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
