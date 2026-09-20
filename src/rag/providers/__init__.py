"""模型 Provider 工厂。

★ 所有 provider 都通过 Protocol 抽象，Settings 里一个开关就能切换实现。
   本地开发用 BGE，生产可以换成 API —— 检索层代码一行不用改。
"""

from __future__ import annotations

from rag.core.config import Settings
from rag.core.errors import ServiceUnavailableError
from rag.providers.base import EmbeddingProvider, LLMClient, Reranker
from rag.providers.embedding import (
    APIEmbeddingProvider,
    HashEmbeddingProvider,
    LocalEmbeddingProvider,
)
from rag.providers.llm import NullLLM, OpenAICompatLLM
from rag.providers.reranker import APIReranker, LocalReranker, NoopReranker


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embed_provider

    if provider == "local":
        if not settings.embed_model_path:
            raise ServiceUnavailableError(
                "EMBED_PROVIDER=local 但未配置 EMBED_MODEL_PATH"
            )
        return LocalEmbeddingProvider(
            model_path=settings.embed_model_path,
            model_id=settings.embed_model_id,
            dim=settings.embed_dim,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
            query_instruction=settings.embed_query_instruction,
            max_tokens=settings.embed_max_tokens,
        )

    if provider == "api":
        # ★ TEI 不校验 API key，所以这里不能强制要求配置 ——
        #   自建推理服务是最常见的用法，强制 key 会把最常见的路径堵死。
        if not settings.embed_api_base:
            raise ServiceUnavailableError("EMBED_PROVIDER=api 但未配置 EMBED_API_BASE")
        return APIEmbeddingProvider(
            base_url=settings.embed_api_base,
            api_key=settings.embed_api_key.get_secret_value(),
            model=settings.embed_api_model,
            dim=settings.embed_dim,
            batch_size=settings.embed_batch_size,
            query_instruction=settings.embed_query_instruction,
            api_style=settings.embed_api_style,
            timeout=settings.embed_timeout_seconds,
        )

    # hash：确定性假向量，仅测试用
    return HashEmbeddingProvider(dim=settings.embed_dim)


def build_reranker(settings: Settings) -> Reranker:
    provider = settings.rerank_provider

    if provider == "local":
        if not settings.rerank_model_path:
            raise ServiceUnavailableError(
                "RERANK_PROVIDER=local 但未配置 RERANK_MODEL_PATH"
            )
        return LocalReranker(model_path=settings.rerank_model_path)

    if provider == "api":
        return APIReranker(
            base_url=settings.rerank_api_base,
            api_key=settings.rerank_api_key.get_secret_value(),
            model=settings.rerank_api_model,
            api_style=settings.rerank_api_style,
            timeout=settings.rerank_timeout_seconds,
        )

    return NoopReranker()


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "none" or not settings.llm_api_key.get_secret_value():
        return NullLLM()

    return OpenAICompatLLM(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        cheap_model=settings.llm_model_cheap,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


__all__ = [
    "APIEmbeddingProvider",
    "APIReranker",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "LLMClient",
    "LocalEmbeddingProvider",
    "LocalReranker",
    "NoopReranker",
    "NullLLM",
    "OpenAICompatLLM",
    "Reranker",
    "build_embedding_provider",
    "build_llm",
    "build_reranker",
]
