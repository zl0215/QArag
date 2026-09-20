"""用新解析器/模型把现有文档重建到一个新 Milvus 集合。

默认只做解析预检；传 ``--apply`` 才会写 PostgreSQL 和目标集合。目标集合与线上
集合分开，因此可以在当前服务继续读取旧索引时完成构建，最后再改
``MILVUS_COLLECTION`` 一次性切换。

示例：
    python scripts/reindex_corpus.py --collection rag_chunks_v2
    python scripts/reindex_corpus.py --collection rag_chunks_v2 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking import build_chunker  # noqa: E402
from rag.core.config import get_settings  # noqa: E402
from rag.infra import build_repository, build_vector_store  # noqa: E402
from rag.providers import build_embedding_provider  # noqa: E402
from rag.services.ingestion import IngestionService  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _resolve_upload(upload_dir: Path, source_uri: str, expected_hash: str) -> Path:
    name = Path(source_uri.split("?", 1)[0]).name
    candidates = [upload_dir / name, *upload_dir.glob(f"*_{name}")]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if _sha256(candidate) == expected_hash:
            return candidate
    raise FileNotFoundError(f"找不到与文档哈希一致的上传文件：{name}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="rag_chunks_v2")
    parser.add_argument("--inference-url", default="http://127.0.0.1:8083")
    parser.add_argument("--documents", default="", help="逗号分隔的 document_id；默认全部")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    base = get_settings()
    settings = base.model_copy(update={
        "milvus_collection": args.collection,
        "embed_api_base": args.inference_url,
        "embed_api_model": "BAAI/bge-m3",
        "embed_model_id": "BAAI/bge-m3",
    })
    wanted = {int(value) for value in args.documents.split(",") if value.strip()}

    repository = build_repository(settings)
    await repository.ensure_ready(with_checkpointer=False)
    documents = await repository.list_documents(tenant_id=1, limit=10000)
    documents = [doc for doc in documents if not wanted or doc.id in wanted]
    documents.sort(key=lambda doc: doc.id)

    chunker = build_chunker(settings)
    paths = {
        doc.id: _resolve_upload(settings.upload_dir, doc.source_uri, doc.content_hash)
        for doc in documents
    }

    if not args.apply:
        from rag.parsers.base import parse_document

        total = 0
        for doc in documents:
            parsed = parse_document(
                paths[doc.id], doc_id=str(doc.id), file_name=paths[doc.id].name,
                sha256=doc.content_hash,
            )
            chunks = chunker.split(parsed, doc_title=doc.title or parsed.meta.title)
            total += len(chunks)
            print(
                f"doc={doc.id:<3} chunks {doc.chunk_count:>4} -> {len(chunks):>4}  "
                f"parser={parsed.meta.parser}  {doc.title}"
            )
        print(f"DRY RUN: {len(documents)} documents, {total} chunks; no data changed")
        await repository.aclose()
        return 0

    store = build_vector_store(settings)
    embedder = build_embedding_provider(settings)
    await store.ensure_ready()
    service = IngestionService(
        repository=repository,
        store=store,
        embedder=embedder,
        chunker=chunker,
        chunker_version=settings.chunker_version,
        embed_batch_size=settings.embed_batch_size,
    )

    failures = 0
    try:
        for index, doc in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] reindex doc={doc.id} {doc.title}", flush=True)
            try:
                # 只清目标 staging 集合中的同文档旧记录，线上集合完全不动。
                await store.delete_by_document(doc.id, doc.tenant_id)
                result = await service._run(doc, paths[doc.id], dt.datetime.now(dt.UTC))
                print(
                    f"  OK chunks={result.chunk_count} vectors={result.vectors_written} "
                    f"duration={result.duration_ms}ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - 批量重建需继续并汇总所有失败
                failures += 1
                await repository.update_document(
                    doc.id, status="failed", error_message=f"reindex: {type(exc).__name__}: {exc}"[:2000]
                )
                print(f"  FAIL {type(exc).__name__}: {exc}", flush=True)
    finally:
        await embedder.aclose()
        await store.aclose()
        await repository.aclose()

    if failures:
        print(f"FAILED: {failures}/{len(documents)} documents")
        return 1
    print(f"READY: collection={args.collection}, documents={len(documents)}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
