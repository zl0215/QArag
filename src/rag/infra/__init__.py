"""基础设施层：向量库、关系库、仓储。

★ 向量库工厂放在这里的原因：
   两种后端（milvus / memory）实现同一个 VectorStore Protocol，
   调用方只认 Protocol。**"用哪个后端"是纯配置问题，不该渗进业务代码。**
   所以 services 层拿到的永远是 VectorStore，不会出现 `if backend == ...`。

   （曾经还有第三种 sqlite 后端，已删除 —— 它"无 Docker 时的持久化"这个定位
     被 Milvus Lite 完全覆盖，而 Lite 有真正的 BM25，sqlite 那份是手写的简化实现。）
"""

from __future__ import annotations

from rag.core.config import Settings
from rag.core.logging import get_logger
from rag.infra.vectorstore import MemoryVectorStore, VectorHit, VectorRecord, VectorStore

logger = get_logger(__name__)


def build_repository(settings: Settings):
    """按配置返回仓储实现。

    ★ 返回类型故意不标注具体类：调用方只认 Repository Protocol。
      标注了 PostgresRepository 反而会诱导上层写出依赖具体实现的代码。
    """
    if settings.repository_backend == "memory":
        from rag.infra.repository import MemoryRepository

        logger.warning(
            "repository.memory",
            hint="MemoryRepository 重启即丢数据，且没有任务队列与 checkpoint，仅适合演示",
        )
        return MemoryRepository()

    from rag.infra.db import Database
    from rag.infra.repository import PostgresRepository

    return PostgresRepository(Database(settings.database_url))


def build_vector_store(settings: Settings) -> VectorStore:
    backend = settings.vector_backend

    if backend == "milvus":
        from rag.infra.milvus import MilvusVectorStore

        if settings.milvus_is_lite:
            logger.info("vectorstore.milvus_lite", uri=settings.milvus_uri)
        return MilvusVectorStore(
            uri=settings.milvus_uri,
            token=settings.milvus_token.get_secret_value(),
            collection=settings.milvus_collection,
            dim=settings.embed_dim,
            analyzer=settings.milvus_analyzer,
        )

    return MemoryVectorStore(dim=settings.embed_dim)


__all__ = [
    "MemoryVectorStore",
    "VectorHit",
    "VectorRecord",
    "VectorStore",
    "build_repository",
    "build_vector_store",
]
