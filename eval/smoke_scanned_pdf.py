"""Real local OCR -> chunking -> BM25 retrieval acceptance on generated scan fixtures."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from ocr_samples import CHINESE_LINES, create_samples

from rag.chunking.counter import HeuristicTokenCounter
from rag.chunking.splitter import StructureAwareChunker
from rag.core.config import get_settings
from rag.core.errors import NeedsOCRError
from rag.infra.repository import MemoryRepository
from rag.infra.vectorstore import MemoryVectorStore
from rag.parsers.base import parse_document
from rag.parsers.ocr import OCRConfig
from rag.providers.embedding import HashEmbeddingProvider
from rag.services.ingestion import IngestionService
from rag.services.retrieval import RetrievalService


async def run(output: Path, font: Path | None) -> dict:
    samples = create_samples(output, font)
    config = OCRConfig.from_settings(get_settings())
    config = replace(config, mode="auto")
    report = {"scope": "synthetic OCR and BM25 pipeline acceptance; not semantic retrieval evaluation",
              "cases": []}
    for name in ("scanned", "mixed"):
        path = samples[name]
        started = time.monotonic()
        try:
            parse_document(path, doc_id="baseline", ocr=replace(config, mode="off"))
            baseline_rejected = False
        except NeedsOCRError:
            baseline_rejected = True
        parsed = parse_document(path, doc_id="scan", ocr=config)
        normalized = "".join(parsed.full_text().split())
        columns_ordered = None
        if name == "scanned":
            columns_ordered = normalized.index("Documenteachprocessingstep.") < normalized.index(
                "Resultsandlimitations"
            )
            if not columns_ordered:
                raise AssertionError("Two-column reading order was not preserved")
        phrases = [line[:12] for line in CHINESE_LINES[1:]]
        recovered = sum("".join(phrase.split()) in normalized for phrase in phrases)
        repo, store = MemoryRepository(), MemoryVectorStore()
        embedder = HashEmbeddingProvider(dim=64)
        service = IngestionService(
            repository=repo, store=store, embedder=embedder, ocr=config,
            chunker=StructureAwareChunker(HeuristicTokenCounter(), target_tokens=256, max_tokens=384),
        )
        ingested = await service.ingest_file(path)
        retrieval = RetrievalService(
            repository=repo, store=store, embedder=embedder, translate_default=False,
        )
        result = await retrieval.retrieve("数据备份", use_dense=False, use_rerank=False)
        hit = next((chunk for chunk in result.chunks if "两份" in chunk.content), None)
        row = {
            "case": name, "native_only_rejected": baseline_rejected,
            "ocr_pages": parsed.meta.ocr_pages,
            "phrase_checks": len(phrases), "phrases_recovered": recovered,
            "bm25_evidence_found": hit is not None,
            "retrieved_page": hit.page_start if hit else None,
            "extraction_method": hit.extraction_method if hit else None,
            "status": ingested.status, "elapsed_seconds": round(time.monotonic() - started, 2),
            "two_column_order_verified": columns_ordered,
        }
        report["cases"].append(row)
        if recovered != len(phrases) or hit is None or not baseline_rejected:
            raise AssertionError(row)
    blank = parse_document(samples["blank"], doc_id="blank", ocr=config)
    report["blank_page_detected"] = blank.meta.blank_pages == [1]
    if not report["blank_page_detected"]:
        raise AssertionError("Blank raster page was not detected")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ocr_acceptance"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--report", type=Path, default=Path("eval/results/ocr_smoke.json"))
    args = parser.parse_args()
    result = asyncio.run(run(args.output, args.font))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
