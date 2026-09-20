"""Milvus 向量存储。

★ 四个关键决策：

1. **主键显式指定，禁用 auto_id** —— 幂等 upsert 与精确删除的前提。
   chunk_id 由 PostgreSQL 分配，是三库共享键。

2. **用内建 BM25 Function**，而不是自己算稀疏向量 ——
   插入时只给原文，Milvus 服务端分词并生成稀疏向量。省一路模型、省一次网络往返。
   代价：文本必须在 Milvus 侧，所以 content 是**派生副本**，权威仍在 PostgreSQL。

3. **所有标量字段都建 INVERTED 索引** ——
   Milvus 做的是预过滤（expr → bitmap → 只对命中 ID 跑 ANN）。
   未建索引的字段过滤会退化成全表扫描，实测能差 10–100 倍。

4. **pymilvus 的 MilvusClient 是同步的**，在 async 应用里必须走 asyncio.to_thread，
   否则会阻塞事件循环，QPS 断崖式下跌。

⚠️ 本文件无法在 Windows 上验证（需要 Docker），首次部署到 Ubuntu VM 时
   务必跑 scripts/smoke_milvus.py 做冒烟。
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag.core.errors import ProviderError
from rag.core.logging import get_logger
from rag.infra.vectorstore import VectorHit, VectorRecord

logger = get_logger(__name__)

# 中文 UTF-8 一字 3 字节。8192 字节 ≈ 2700 汉字，够 BM25 用且远离 65535 上限与 64MB RPC 上限。
CONTENT_MAX_BYTES = 8192


class MilvusVectorStore:
    def __init__(
        self,
        *,
        uri: str,
        token: str = "",
        collection: str = "rag_chunks",
        dim: int = 1024,
        analyzer: str = "chinese",
        consistency_level: str = "Bounded",
    ) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("未安装 pymilvus，无法使用 Milvus 后端") from exc

        self.collection = collection
        self.dim = dim
        self.analyzer = analyzer
        self.consistency_level = consistency_level
        self._client = MilvusClient(uri=uri, token=token or None)

    # ==================================================================
    # 建集合
    # ==================================================================
    def _build_schema(self):  # noqa: ANN202
        from pymilvus import DataType, Function, FunctionType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.INT64, is_primary=True)
        schema.add_field("tenant_id", DataType.INT64)
        schema.add_field("document_id", DataType.INT64)
        schema.add_field("is_active", DataType.BOOL)
        # ★ enable_analyzer + 中文分析器：BM25 的分词在这里发生。
        #   分析器建集合后不可改 —— 换分词器只能新建集合 + 迁移。
        schema.add_field(
            "content", DataType.VARCHAR, max_length=CONTENT_MAX_BYTES,
            enable_analyzer=True, analyzer_params={"type": self.analyzer},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("node_type", DataType.VARCHAR, max_length=32)
        schema.add_field("lang", DataType.VARCHAR, max_length=8)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("chunker_ver", DataType.VARCHAR, max_length=32)
        schema.add_field("embed_model", DataType.VARCHAR, max_length=64)

        # 插入时只提供 content，sparse 由服务端生成
        schema.add_function(Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse"],
        ))
        return schema

    def _build_index_params(self):  # noqa: ANN202
        params = self._client.prepare_index_params()
        params.add_index(
            field_name="dense", index_type="HNSW", metric_type="COSINE",
            # 显式传参：不要依赖默认值（不同版本的 AUTOINDEX 注入的参数不一样）
            params={"M": 16, "efConstruction": 200},
        )
        params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )
        # ★ 标量索引必建，否则过滤 = 全表扫描
        for field in ("tenant_id", "document_id", "is_active", "chunker_ver"):
            params.add_index(field_name=field, index_type="INVERTED")
        return params

    def _ensure_ready_sync(self) -> None:
        if self._client.has_collection(self.collection):
            logger.info("milvus.collection_exists", collection=self.collection)
        else:
            logger.info("milvus.creating_collection", collection=self.collection, dim=self.dim)
            self._client.create_collection(
                collection_name=self.collection,
                schema=self._build_schema(),
                index_params=self._build_index_params(),
                consistency_level=self.consistency_level,
            )
        self._client.load_collection(self.collection)

    async def ensure_ready(self) -> None:
        try:
            await asyncio.to_thread(self._ensure_ready_sync)
        except Exception as exc:
            logger.exception("milvus.ensure_ready_failed")
            raise ProviderError(f"Milvus 初始化失败：{type(exc).__name__}: {exc}") from exc

    # ==================================================================
    # 写入
    # ==================================================================
    def _upsert_sync(self, records: list[VectorRecord]) -> int:
        rows: list[dict[str, Any]] = [
            {
                "chunk_id": r.chunk_id,
                "tenant_id": r.tenant_id,
                "document_id": r.document_id,
                "is_active": r.is_active,
                "content": r.content[:CONTENT_MAX_BYTES],
                "dense": r.dense,
                # 注意：不提供 sparse —— 它是 BM25 Function 的输出
                "node_type": (r.node_type or "paragraph")[:32],
                "lang": (r.lang or "")[:8],
                "content_hash": (r.content_hash or "")[:64],
                "chunker_ver": (r.chunker_ver or "")[:32],
                "embed_model": (r.embed_model or "")[:64],
            }
            for r in records
        ]
        if not rows:
            return 0
        self._client.upsert(collection_name=self.collection, data=rows)
        return len(rows)

    async def upsert(self, records: list[VectorRecord]) -> int:
        try:
            return await asyncio.to_thread(self._upsert_sync, records)
        except Exception as exc:
            logger.exception("milvus.upsert_failed", n=len(records))
            raise ProviderError(f"Milvus 写入失败：{type(exc).__name__}") from exc

    def _delete_sync(self, document_id: int, tenant_id: int) -> int:
        """删除某个文档的全部分块，返回删掉的条数。

        ★ 返回值的**形状两种后端不一样**（实测）：
            standalone  → {"delete_count": N, ...}  —— 字典
            Milvus Lite → [1001, 1002, ...]         —— 被删主键的列表
          原来只认字典，Lite 上走 else 分支恒返回 0 —— 删除**成功**却报 0。
          排查时会以为"没删掉"，实际数据已经没了，是个会误导人的假象。
          （当前调用方 services/ingestion.py 丢弃了这个返回值，所以没有
            引发实际故障；但把"成功"报成"0 条"这种事留着早晚要坑人。）
        """
        expr = f"document_id == {int(document_id)} and tenant_id == {int(tenant_id)}"
        result = self._client.delete(collection_name=self.collection, filter=expr)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        if isinstance(result, list):
            return len(result)
        return 0

    async def delete_by_document(self, document_id: int, tenant_id: int = 1) -> int:
        try:
            return await asyncio.to_thread(self._delete_sync, document_id, tenant_id)
        except Exception as exc:
            logger.exception("milvus.delete_failed", document_id=document_id)
            raise ProviderError(f"Milvus 删除失败：{type(exc).__name__}") from exc

    # ==================================================================
    # 检索
    # ==================================================================
    @staticmethod
    def _tenant_expr(tenant_id: int) -> str:
        # ★ tenant_id 是服务端从鉴权信息解出的整数，不是用户输入。
        #   绝不把字符串拼进 filter —— 那是注入。
        return f"tenant_id == {int(tenant_id)} and is_active == true"

    @staticmethod
    def _to_hits(raw: list[dict]) -> list[VectorHit]:
        """把 pymilvus 的命中结果转成 VectorHit。

        ★ 主键的**键名在两种后端下不一样**，这里必须两种都认：
            standalone      → item["id"]
            Milvus Lite     → item["chunk_id"]（用的是主键的**字段名**）
          只认 "id" 的话，Lite 上每一次检索都 KeyError，而且异常被包在
          ProviderError 里，看起来像"检索失败"，实际是结果解析的问题 ——
          这个如果不在本地先打一遍，到 AutoDL 上很难定位。
        """
        hits: list[VectorHit] = []
        for rank, item in enumerate(raw, start=1):
            entity = item.get("entity") or {}
            pk = item.get("id")
            if pk is None:
                pk = item.get("chunk_id")
            if pk is None:
                pk = entity.get("chunk_id")
            if pk is None:
                # 字段名对不上时明确报出来，别让它退化成一句无信息的 KeyError
                raise ProviderError(
                    f"Milvus 返回的命中里找不到主键，实际字段：{sorted(item.keys())}"
                )
            hits.append(VectorHit(
                chunk_id=int(pk),
                score=float(item["distance"]),
                rank=rank,
                document_id=entity.get("document_id"),
                content=entity.get("content"),
                extra={"node_type": entity.get("node_type")},
            ))
        return hits

    def _search_dense_sync(
        self, vector: list[float], top_k: int, expr: str
    ) -> list[VectorHit]:
        raw = self._client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            filter=expr,
            output_fields=["document_id", "content", "node_type"],
        )
        hits = self._to_hits(raw[0] if raw else [])
        for hit in hits:
            hit.dense_score = hit.score
        return hits

    async def search_dense(
        self, vector: list[float], *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]:
        expr = self._tenant_expr(tenant_id)
        try:
            return await asyncio.to_thread(self._search_dense_sync, vector, top_k, expr)
        except Exception as exc:
            logger.exception("milvus.search_dense_failed")
            raise ProviderError(f"Milvus 向量检索失败：{type(exc).__name__}") from exc

    def _search_sparse_sync(self, query_text: str, top_k: int, expr: str) -> list[VectorHit]:
        # ★ BM25 直接传原文，服务端分词 —— 省一次本地分词 + 一次网络往返
        raw = self._client.search(
            collection_name=self.collection,
            data=[query_text],
            anns_field="sparse",
            search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            filter=expr,
            output_fields=["document_id", "content", "node_type"],
        )
        hits = self._to_hits(raw[0] if raw else [])
        for hit in hits:
            hit.sparse_score = hit.score
        return hits

    async def search_sparse(
        self, query_text: str, *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]:
        expr = self._tenant_expr(tenant_id)
        try:
            return await asyncio.to_thread(self._search_sparse_sync, query_text, top_k, expr)
        except Exception as exc:
            logger.exception("milvus.search_sparse_failed")
            raise ProviderError(f"Milvus 全文检索失败：{type(exc).__name__}") from exc

    def _search_hybrid_native_sync(
        self, dense_vector: list[float], query_text: str, top_k: int, expr: str
    ) -> list[VectorHit]:
        """Milvus 服务端融合 —— 只用于和自研 RRF 做对照实验。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[dense_vector], anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k, expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text], anns_field="sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=top_k, expr=expr,
        )
        raw = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["document_id", "content", "node_type"],
        )
        return self._to_hits(raw[0] if raw else [])

    async def search_hybrid_native(
        self, dense_vector: list[float], query_text: str, *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]:
        expr = self._tenant_expr(tenant_id)
        return await asyncio.to_thread(
            self._search_hybrid_native_sync, dense_vector, query_text, top_k, expr
        )

    # ==================================================================
    # 运维
    # ==================================================================
    def _count_sync(self, tenant_id: int) -> int:
        result = self._client.query(
            collection_name=self.collection,
            filter=f"tenant_id == {int(tenant_id)}",
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"]) if result else 0

    async def count(self, tenant_id: int = 1) -> int:
        return await asyncio.to_thread(self._count_sync, tenant_id)

    async def query_all_chunk_ids(self, tenant_id: int = 1) -> set[int]:
        """对账用。★ 必须用 query_iterator —— query 默认有 limit，会静默漏数据。"""

        def _run() -> set[int]:
            ids: set[int] = set()
            iterator = self._client.query_iterator(
                collection_name=self.collection,
                filter=f"tenant_id == {int(tenant_id)}",
                output_fields=["chunk_id"],
                batch_size=1000,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    iterator.close()
                    break
                ids.update(int(row["chunk_id"]) for row in batch)
            return ids

        return await asyncio.to_thread(_run)

    async def analyze(self, text: str) -> list[str]:
        """验证分词是否符合预期（建集合后应该立刻跑一次）。

        ★ Milvus Lite **没有实现 run_analyzer**（RPC 直接返回 UNIMPLEMENTED）。
          这是个**诊断接口**，不在检索主链路上，所以这里降级返回空列表，
          而不是把异常抛出去 —— 否则 /readyz 和冒烟脚本会因为一个可选能力
          把整个服务判成不健康，排查方向会被完全带偏。
        """
        try:
            return await asyncio.to_thread(
                self._client.run_analyzer,
                texts=text,
                analyzer_params={"type": self.analyzer},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "milvus.analyzer_unavailable",
                analyzer=self.analyzer, error=type(exc).__name__,
                hint="Milvus Lite 不支持 run_analyzer，属预期；standalone 上应该可用",
            )
            return []

    async def aclose(self) -> None:
        try:
            await asyncio.to_thread(self._client.close)
        except Exception:  # pragma: no cover
            logger.debug("milvus.close_failed", exc_info=True)


def escape_filter_string(value: str) -> str:
    """字符串型过滤值必须转义 —— 绝不直接拼接用户输入。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')
