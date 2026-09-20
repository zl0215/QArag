"""结构感知分块。"""

from __future__ import annotations

from rag.chunking.counter import HeuristicTokenCounter, TokenCounter, TransformersTokenCounter
from rag.chunking.splitter import StructureAwareChunker

__all__ = [
    "HeuristicTokenCounter",
    "StructureAwareChunker",
    "TokenCounter",
    "TransformersTokenCounter",
    "build_chunker",
]


def build_chunker(settings) -> StructureAwareChunker:  # noqa: ANN001
    """按配置构造分块器。

    ★ 这个工厂放在 chunking 包而不是 api/deps.py 里，是因为**两边都要它**：
      API（同步上传路径）和 worker（异步路径）用的是同一套分块参数。
      写在 api/deps.py 里的话 worker 就得 `from rag.api.deps import ...` ——
      一个纯后台进程为了拿一个分块器去 import FastAPI 那一整条依赖链，
      逻辑上也不对（worker 不提供 HTTP）。

    ★ tokenizer 从**模型目录**读，所以它依赖 embedder 的配置正确。
      走 API embedding（没有本地模型目录）时退到用 embed_model_id 去
      HuggingFace 找 —— 离线环境会失败，调用方需自行决定是降级还是退出。
    """
    source = settings.embed_model_path or settings.embed_model_id
    counter = TransformersTokenCounter.from_model_dir(source)
    return StructureAwareChunker(
        counter,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        parent_target_tokens=settings.parent_target_tokens,
        chunker_version=settings.chunker_version,
        embed_model_id=settings.embed_model_id,
        embed_dim=settings.embed_dim,
    )
