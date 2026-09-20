"""摄取管道：解析 → 分块 → 嵌入 → 双写（Postgres + 向量库）。

★ 顺序与失败语义（这是整条链路最容易做错的地方）：

   1. Postgres 先分配 chunk_id（BIGSERIAL），向量库再按这个 id 写入。
      **绝不能反过来** —— Milvus 的 auto_id 会生成一套 id，两边就对不上了。
   2. 向量库写入放在最后，且失败不回滚 Postgres。代价是可能出现"库里有一批
      embed_status=stale 的块"，收益是**永远不会出现"向量库有、正文库里没有"
      的孤儿向量** —— 后者在检索时无法回表，是更难处理的坏状态。
   3. 对账脚本 scripts/reconcile.py 负责收敛这个偏差。

★ 幂等：content_hash（sha256 原始字节）命中且状态 ready 时直接返回已有文档。
   重复上传同一份文件不会产生第二次解析 —— 解析是整条链路最贵的一步。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from rag.core.errors import IngestionError
from rag.core.logging import get_logger
from rag.infra.models import DocStatus
from rag.infra.repository import DocumentRecord, Repository
from rag.infra.vectorstore import VectorRecord, VectorStore
from rag.parsers.base import _sha256_file, parse_document
from rag.providers.base import EmbeddingProvider

logger = get_logger(__name__)

# 嵌入分批大小。★ 不是越大越好：本地模型一次吃 64 条 512 token 的文本，
# 显存/内存峰值和延迟都会明显抖动；64 是本地 CPU 推理的稳健值。
EMBED_BATCH_SIZE = 64


def _title_from(source_uri: str | None) -> str:
    """从 source_uri 里取出给人看的标题。

    source_uri 可能是裸文件名（`手册.md`）、带目录的路径，也可能是 URL，
    这三种 `Path(...).stem` 都能拿到合理的名字。取不到就返回空串，
    由调用方退回 path.stem。
    """
    if not source_uri:
        return ""
    return Path(source_uri.split("?", 1)[0]).stem


@dataclass
class IngestionResult:
    document_id: int
    chunk_count: int
    status: str
    deduplicated: bool = False          # True 表示命中哈希、直接复用
    vectors_written: int = 0
    page_count: int = 0
    lang: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0


class IngestionService:
    def __init__(
        self,
        *,
        repository: Repository,
        store: VectorStore,
        embedder: EmbeddingProvider,
        chunker,                                    # noqa: ANN001
        chunker_version: str = "v1",
        embed_batch_size: int = EMBED_BATCH_SIZE,
        tenant_id: int = 1,
    ) -> None:
        self.repo = repository
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.chunker_version = chunker_version
        self.embed_batch_size = embed_batch_size
        self.tenant_id = tenant_id

    # ==================================================================
    async def ingest_file(
        self,
        path: Path,
        *,
        title: str | None = None,
        source_uri: str | None = None,
        force: bool = False,
    ) -> IngestionResult:
        started = dt.datetime.now(dt.UTC)
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"文件不存在：{path}")

        raw_hash = _sha256_file(path)

        # ---- 幂等短路 ----
        if not force:
            existing = await self.repo.find_document_by_hash(raw_hash, self.tenant_id)
            if existing and existing.status == DocStatus.READY:
                logger.info("ingest.dedup_hit", document_id=existing.id, hash=raw_hash[:12])
                return IngestionResult(
                    document_id=existing.id, chunk_count=existing.chunk_count,
                    status=existing.status, deduplicated=True,
                    page_count=existing.page_count or 0, lang=existing.lang,
                )

        doc = await self.repo.create_document(
            tenant_id=self.tenant_id,
            # ★ 标题优先取 source_uri（用户上传时的原始文件名），最后才退回 path.stem。
            #   不能直接用 path.stem：上传时文件是存成 `data/uploads/<8位uuid>_原名.md` 的，
            #   于是标题会变成 `d06dfdbd_ragtest` —— 把内部存储的实现细节泄漏到了界面上。
            title=title or _title_from(source_uri) or path.stem,
            source_uri=source_uri or str(path),
            mime_type="application/octet-stream",     # 解析后回填真实类型
            content_hash=raw_hash,
            size_bytes=path.stat().st_size,
            status=DocStatus.PARSING,
        )

        try:
            return await self._run(doc, path, started)
        except Exception as exc:
            logger.exception("ingest.failed", document_id=doc.id)
            await self.repo.update_document(
                doc.id, status=DocStatus.FAILED,
                error_message=f"{type(exc).__name__}: {exc}"[:2000],
            )
            raise

    # ------------------------------------------------------------------
    async def _run(self, doc: DocumentRecord, path: Path, started: dt.datetime) -> IngestionResult:
        warnings: list[str] = []

        # ---- 1. 解析 ----
        parsed = parse_document(
            path, doc_id=str(doc.id), file_name=path.name, sha256=doc.content_hash
        )
        meta = parsed.meta
        warnings.extend(meta.parse_warnings or [])
        await self.repo.update_document(
            doc.id, mime_type=meta.mime, parser=meta.parser, lang=meta.lang,
            page_count=meta.page_count, text_hash=meta.sha256_text,
            title=doc.title or meta.title, progress=30,
            status=DocStatus.CHUNKING,
        )
        if not parsed.nodes:
            raise IngestionError(f"解析后没有任何内容（可能是扫描版 PDF）：{path.name}")

        # ---- 2. 分块 ----
        # 文档标题必须进入 embedding 输入。科研论文的正文块经常只写“the proposed
        # method”，若不带题名，模型无法知道它属于 GBSVM 还是 ISFFSVM。
        chunks = self.chunker.split(parsed, doc_title=doc.title or meta.title)
        if not chunks:
            raise IngestionError(f"分块结果为空：{path.name}")
        logger.info("ingest.chunked", document_id=doc.id, n_chunks=len(chunks),
                    avg_tokens=sum(c.token_count for c in chunks) // max(len(chunks), 1))

        # ---- 3. Postgres 分配 chunk_id（必须先于向量库写入）----
        await self.repo.update_document(doc.id, status=DocStatus.EMBEDDING, progress=45)
        chunk_ids = await self.repo.replace_chunks(
            doc.id, doc.version, chunks,
            embed_model=self.embedder.model_id, embed_dim=self.embedder.dim,
        )
        if len(chunk_ids) != len(chunks):  # pragma: no cover - 防御性检查
            raise IngestionError(
                f"chunk_id 分配数量不一致：分配 {len(chunk_ids)}，预期 {len(chunks)}"
            )

        # ---- 4. 嵌入（分批）----
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[start:start + self.embed_batch_size]
            # ★ 送 embed_text（含标题面包屑），不是 content
            vectors.extend(await self.embedder.aembed_documents([c.embed_text for c in batch]))
        if len(vectors) != len(chunks):  # pragma: no cover
            raise IngestionError(
                f"向量数量不一致：{len(vectors)} vs {len(chunks)} 个块"
            )

        # ---- 5. 写向量库（最后一步，失败不回滚）----
        records = [
            VectorRecord(
                chunk_id=chunk_id,
                dense=vector,
                content=chunk.content,      # 供 Milvus 服务端 BM25 分词
                document_id=doc.id,
                tenant_id=self.tenant_id,
                node_type=str(chunk.node_type),
                lang=meta.lang,
                content_hash=chunk.content_hash,
                chunker_ver=self.chunker_version,
                embed_model=self.embedder.model_id,
                is_active=True,
            )
            for chunk_id, chunk, vector in zip(chunk_ids, chunks, vectors, strict=True)
        ]
        written = 0
        try:
            for start in range(0, len(records), self.embed_batch_size):
                written += await self.store.upsert(records[start:start + self.embed_batch_size])
        except Exception as exc:
            # ★ 不回滚 Postgres：宁可留下可对账的 stale 块，
            #   也不要出现"向量库有、正文库没有"的孤儿向量
            logger.exception("ingest.vector_write_failed", document_id=doc.id, written=written)
            await self.repo.update_document(
                doc.id, status=DocStatus.FAILED, progress=90,
                error_message=f"向量写入失败（已写入 {written}/{len(records)}）：{exc}"[:2000],
            )
            raise

        await self.repo.update_document(
            doc.id, status=DocStatus.READY, progress=100,
            error_message=None, chunk_count=len(chunks),
        )

        elapsed = int((dt.datetime.now(dt.UTC) - started).total_seconds() * 1000)
        result = IngestionResult(
            document_id=doc.id, chunk_count=len(chunks), status=DocStatus.READY,
            vectors_written=written, page_count=meta.page_count or 0,
            lang=meta.lang, warnings=warnings, duration_ms=elapsed,
        )
        logger.info("ingest.done", document_id=doc.id, chunks=len(chunks),
                    vectors=written, ms=elapsed, warnings=len(warnings))
        return result

    # ==================================================================
    async def delete_document(self, document_id: int) -> None:
        """★ 删除顺序与写入相反：先删向量库，再软删 Postgres。

        反过来的话，向量删失败就会留下孤儿向量 —— 检索命中却回不了表。
        """
        await self.store.delete_by_document(document_id, self.tenant_id)
        await self.repo.soft_delete_document(document_id, self.tenant_id)
        logger.info("ingest.deleted", document_id=document_id)
