"""Embedding provider 实现。

★ 三个必须做对的地方：
1. **L2 归一化** —— 归一化后内积等价于余弦且更快；不归一化则 IP/COSINE 行为不一致
2. **非对称检索** —— BGE 系列要求只给 query 加 instruction 前缀，文档不加。
   加错了会稳定掉几个点，而且很难查。
3. **懒加载 + 加锁** —— 本地模型加载可能几十秒，不能在 import 期做；
   但 /readyz 要能反映"模型未就绪"。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Iterable

from rag.chunking.counter import read_max_seq_length
from rag.core.errors import ProviderError, ServiceUnavailableError
from rag.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider:
    """sentence-transformers 本地推理。

    默认读本机已有的 BGE 权重（见 .env 的 EMBED_MODEL_PATH）。
    CPU 上 bge-large-zh 编码 384 token 的文本约 15-30ms/条，16 条一批更划算。
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_id: str = "bge-large-zh-v1.5",
        dim: int = 1024,
        device: str = "cpu",
        batch_size: int = 16,
        query_instruction: str = "",
        max_tokens: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.dim = dim
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._max_tokens = max_tokens
        self._model = None
        self._load_lock = asyncio.Lock()

    # ---------------- 生命周期 ----------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _load_sync(self):  # noqa: ANN202
        from sentence_transformers import SentenceTransformer

        logger.info("embedding.loading", path=self.model_path, device=self.device)
        model = SentenceTransformer(self.model_path, device=self.device)
        # 以模型自身配置为准，避免配置写大了被 tokenizer 静默截断
        self._max_tokens = self._max_tokens or getattr(
            model, "max_seq_length", None
        ) or read_max_seq_length(self.model_path)
        # sentence-transformers 5.x 把 get_sentence_embedding_dimension 改名了，
        # 老名字还在但会打 FutureWarning。两个都试，兼容 3.x/4.x/5.x。
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        logger.info(
            "embedding.loaded",
            model=self.model_id, dim=get_dim(), max_seq_length=self._max_tokens,
        )
        return model

    async def ensure_loaded(self) -> None:
        """★ 双检锁：并发请求只触发一次加载。"""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load_sync)
                except Exception as exc:
                    logger.exception("embedding.load_failed")
                    raise ServiceUnavailableError(
                        f"本地 embedding 模型加载失败：{type(exc).__name__}"
                    ) from exc

    @property
    def max_tokens(self) -> int | None:
        if self._max_tokens is None:
            self._max_tokens = read_max_seq_length(self.model_path)
        return self._max_tokens

    # ---------------- 推理 ----------------
    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # ★ L2 归一化
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self.ensure_loaded()
        try:
            out: list[list[float]] = []
            for batch in _batched(texts, self.batch_size):
                # CPU 密集，必须丢线程池，否则阻塞事件循环
                out.extend(await asyncio.to_thread(self._encode_sync, batch))
            return out
        except Exception as exc:
            logger.exception("embedding.encode_failed")
            raise ProviderError(f"文本向量化失败：{type(exc).__name__}") from exc

    async def aembed_query(self, text: str) -> list[float]:
        await self.ensure_loaded()
        # ★ instruction 只加在 query 上（BGE 非对称检索）
        payload = f"{self.query_instruction}{text}" if self.query_instruction else text
        vectors = await asyncio.to_thread(self._encode_sync, [payload])
        return vectors[0]

    def warmup_sync(self) -> None:
        """供 lifespan 在启动时预热（避免首个请求等模型加载）。"""
        if self._model is None:
            self._model = self._load_sync()


class APIEmbeddingProvider:
    """远程推理服务。

    支持两种协议（EMBED_API_STYLE 切换）：

    - ``openai``：OpenAI 兼容的 ``POST /embeddings``（硅基流动 / 阿里百炼 /
      OpenAI / vLLM / Xinference）。用 openai SDK，自带重试。
    - ``tei``：HuggingFace text-embeddings-inference 的原生 ``POST /embed``，
      请求体是 ``{"inputs": [...]}``，响应是裸的二维数组。

    ★ 为什么要支持 TEI：它是最省的 BGE 部署方式 —— 一个容器同时提供
      HTTP 服务和 GPU 推理，不需要在业务容器里装 torch。
      代价是它的协议不是 OpenAI 形状，必须单独适配。

    ★ 无论哪种协议，返回前一律重新做 L2 归一化：
      不能假设服务端的 normalize 配置和我们的检索假设一致。
      服务端没归一化而我们按余弦检索，结果会静默变差而不报错。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        dim: int = 1024,
        batch_size: int = 32,
        query_instruction: str = "",
        api_style: str = "openai",
        timeout: int = 60,
    ) -> None:
        if not base_url:
            raise ServiceUnavailableError("EMBED_API_BASE 未配置")
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.dim = dim
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.api_style = api_style
        self._client = None
        self._http = None

        if api_style == "tei":
            import httpx

            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                timeout=timeout,
            )
        else:
            from openai import AsyncOpenAI

            # TEI 不校验 key，但 openai SDK 要求非空 —— 给个占位符
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key or "not-needed", max_retries=2
            )

    @property
    def max_tokens(self) -> int | None:
        # 主流 API embedding 模型（bge-m3 / text-embedding-3）都在 8k 以上
        return 8192

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self.api_style == "tei":
            resp = await self._http.post("/embed", json={"inputs": batch})
            resp.raise_for_status()
            data = resp.json()
            # TEI 的顺序与输入严格一致，没有 index 字段可排
            return [list(map(float, row)) for row in data]

        resp = await self._client.embeddings.create(model=self.model_id, input=batch)
        # ★ 按 index 排序：不能假设返回顺序与输入一致
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        try:
            for batch in _batched(texts, self.batch_size):
                out.extend(await self._embed_batch(batch))
        except Exception as exc:
            logger.exception("embedding.api_failed", style=self.api_style)
            raise ProviderError(f"Embedding API 调用失败：{type(exc).__name__}") from exc

        if out and len(out[0]) != self.dim:
            # ★ 维度写错是极常见的配置失误，而且 Milvus 要到写入时才报错，
            #   那时已经解析+分块完了。在这里提前失败，省掉一整轮无用功。
            raise ProviderError(
                f"向量维度不符：服务返回 {len(out[0])}，配置 EMBED_DIM={self.dim}"
            )
        return [_l2_normalize(v) for v in out]

    async def aembed_query(self, text: str) -> list[float]:
        payload = f"{self.query_instruction}{text}" if self.query_instruction else text
        vectors = await self.aembed_documents([payload])
        return vectors[0]

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
        if self._client is not None:
            await self._client.close()


class HashEmbeddingProvider:
    """确定性假向量 —— 只用于单元测试和离线冒烟。

    ★ 它不表达任何语义，绝不能用于评测检索质量。
    存在的意义是让 pipeline 测试不需要下载模型、不需要网络。
    """

    def __init__(self, dim: int = 1024, model_id: str = "hash-embedder") -> None:
        self.dim = dim
        self.model_id = model_id

    @property
    def max_tokens(self) -> int | None:
        return 8192

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split() or [text]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        # ★ 用 blake2b 而不是内置 hash()：Python 字符串 hash 有进程级随机化，
        #   每次运行结果都不同，是极隐蔽的 flaky 来源。
        return _l2_normalize(vec)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


# ----------------------------------------------------------------------
def _l2_normalize(vec: Iterable[float]) -> list[float]:
    values = list(vec)
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + size]
