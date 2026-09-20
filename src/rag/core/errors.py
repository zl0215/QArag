"""领域异常 —— RFC 9457 Problem Details。

纯领域层，不依赖 FastAPI。API 层统一注册 handler 转成响应。

★ 关键纪律：不把上游 provider 的异常原文抛给客户端。
OpenAI/Anthropic 的异常里常带 request-id、组织 ID 甚至 prompt 片段。
"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    status: int = 500
    type_slug: str = "internal-error"
    title: str = "Internal error"

    # ★ 是否把 detail 原样返回给客户端。
    #
    #   默认为 True：这些 detail 都是**我们自己写的、给用户看的**话
    #   （"未配置 LLM_PROVIDER…"、"文件超过上限 50MB"），
    #   藏起来只会让前端拿到一句没有信息量的 "Service unavailable"。
    #
    #   只有 ProviderError 改成 False —— 它的消息可能裹着上游异常原文
    #   （含 request-id、prompt 片段、模型原始输出），见该类的 docstring。
    #
    #   注意这里**不能**用 `status >= 500` 来判断：那是个过宽的近似，
    #   会把 503 和 500 里我们主动写好的配置提示一并抹掉 —— 曾经就是。
    expose_detail: bool = True

    def __init__(self, detail: str = "", **extensions: Any) -> None:
        super().__init__(detail or self.title)
        self.detail = detail or self.title
        self.extensions = extensions

    def to_problem(self, instance: str | None = None, request_id: str | None = None) -> dict:
        problem: dict[str, Any] = {
            "type": f"https://rag-agent.local/problems/{self.type_slug}",
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        if instance:
            problem["instance"] = instance
        if request_id:
            problem["request_id"] = request_id
        problem.update(self.extensions)
        return problem


# ---------------- 4xx ----------------
class BadRequestError(DomainError):
    status, type_slug, title = 400, "bad-request", "Bad request"


class NotFoundError(DomainError):
    status, type_slug, title = 404, "not-found", "Resource not found"


class UnsupportedMediaError(DomainError):
    status, type_slug, title = 415, "unsupported-media-type", "Unsupported media type"


class PayloadTooLargeError(DomainError):
    status, type_slug, title = 413, "payload-too-large", "Payload too large"


class UnprocessableDocumentError(DomainError):
    status, type_slug, title = 422, "document-parse-failed", "Document parse failed"


class QuotaExceededError(DomainError):
    status, type_slug, title = 429, "quota-exceeded", "Quota exceeded"


# ---------------- 5xx ----------------
class ServiceUnavailableError(DomainError):
    status, type_slug, title = 503, "service-unavailable", "Service unavailable"


class ProviderError(DomainError):
    """上游模型/向量库故障。对外只暴露粗粒度信息，细节进日志。"""

    status, type_slug, title = 502, "upstream-provider-error", "Upstream provider error"
    expose_detail = False


class IngestionError(DomainError):
    status, type_slug, title = 500, "ingestion-failed", "Document ingestion failed"
