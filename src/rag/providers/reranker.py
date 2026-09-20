"""Reranker 实现。

双塔（embedding）模型 query/doc 独立编码，精度有天花板；
cross-encoder 联合编码精度高得多，但复杂度是 O(N) 次前向。
所以做成漏斗：向量召回 top-50 → 重排 → top-5。
"""

from __future__ import annotations

import asyncio

from rag.core.errors import ProviderError
from rag.core.logging import get_logger
from rag.providers.base import DISABLED_MODEL_ID

logger = get_logger(__name__)


class NoopReranker:
    """不重排，按原顺序原分数透传。

    保留它是为了让"无 reranker"成为配置项而不是代码分支 ——
    这样 A/B 对比（评测里的模式 A vs 模式 B）走的是同一条代码路径。
    """

    model_id = DISABLED_MODEL_ID

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        ranked = [(i, 1.0 / (i + 1)) for i in range(len(documents))]
        return ranked[:top_k] if top_k else ranked


class LocalReranker:
    """sentence-transformers CrossEncoder（如 bge-reranker-v2-m3）。"""

    def __init__(self, *, model_path: str, max_length: int = 512, batch_size: int = 16) -> None:
        self.model_path = model_path
        self.model_id = model_path.rstrip("/\\").split("/")[-1].split("\\")[-1]
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _load_sync(self):  # noqa: ANN202
        from sentence_transformers import CrossEncoder

        logger.info("reranker.loading", path=self.model_path)
        model = CrossEncoder(self.model_path, max_length=self.max_length)
        logger.info("reranker.loaded", model=self.model_id)
        return model

    async def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load_sync)
                except Exception as exc:
                    logger.exception("reranker.load_failed")
                    raise ProviderError(f"重排模型加载失败：{type(exc).__name__}") from exc

    def _predict_sync(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        await self.ensure_loaded()
        try:
            scores = await asyncio.to_thread(self._predict_sync, query, documents)
        except Exception as exc:
            logger.exception("reranker.predict_failed")
            raise ProviderError(f"重排失败：{type(exc).__name__}") from exc

        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k] if top_k else ranked


class APIReranker:
    """远程重排服务。

    兼容两种响应格式（RERANK_API_STYLE）：

    - ``cohere``：``{"results": [{"index": 0, "relevance_score": 0.9}]}``
      —— Cohere / Jina / 硅基流动 都是这个形状。
    - ``tei``：``[{"index": 0, "score": 0.9}]``
      —— HuggingFace text-embeddings-inference 的裸数组。

    ★ 两种格式都**必须按 index 映射回原文下标**，不能假设返回顺序。
      TEI 默认按分数降序返回，而 Cohere 在 top_n 小于文档数时也只返回子集 ——
      直接按返回顺序当作 0,1,2… 会把分数安到错误的文档上，
      而且结果看起来"很正常"，是极难发现的错误。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        timeout: int = 60,
        api_style: str = "cohere",
    ) -> None:
        if not base_url:
            raise ProviderError("RERANK_API_BASE 未配置")
        import httpx

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )
        self.model_id = model
        self.api_style = api_style

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = {"query": query, "documents": documents}
        if self.api_style != "tei":
            payload["model"] = self.model_id
            payload["top_n"] = top_k or len(documents)

        try:
            resp = await self._client.post("/rerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.exception("reranker.api_failed", style=self.api_style)
            raise ProviderError(f"Rerank API 调用失败：{type(exc).__name__}") from exc

        if self.api_style == "tei":
            pairs = [(int(r["index"]), float(r["score"])) for r in data]
        else:
            results = data.get("results") or []
            pairs = [(int(r["index"]), float(r["relevance_score"])) for r in results]

        ranked = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k] if top_k else ranked

    async def aclose(self) -> None:
        await self._client.aclose()
