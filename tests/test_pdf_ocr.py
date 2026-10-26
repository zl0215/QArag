from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from eval.ocr_samples import create_samples
from rag.chunking.counter import HeuristicTokenCounter
from rag.chunking.splitter import StructureAwareChunker
from rag.core.errors import NeedsOCRError
from rag.infra.repository import MemoryRepository, _extraction_info
from rag.infra.vectorstore import MemoryVectorStore
from rag.parsers.base import parse_document
from rag.parsers.ocr import OCRConfig, needs_ocr, order_ocr_lines, recognize_page
from rag.providers.embedding import HashEmbeddingProvider
from rag.services.ingestion import IngestionService
from rag.services.retrieval import RetrievalService


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    try:
        return create_samples(tmp_path_factory.mktemp("ocr-fixtures"))
    except FileNotFoundError:
        pytest.skip("Scan fixture generation requires an installed Chinese font (Noto CJK)")


def page_with_graphics():
    return SimpleNamespace(width=600, height=800, images=[{
        "x0": 0, "x1": 600, "top": 0, "bottom": 800,
    }], curves=[], rects=[])


def fake_lines(text="数据备份至少保留两份，并分别存放在不同设备中。"):
    return [{"text": text, "top": 100, "bottom": 112, "x0": 30, "x1": 400,
             "chars": [{"size": 12}], "column": 0, "ocr_confidence": .98}]


def fake_recognize(page, config):
    return fake_lines(), None, {
        "blank": False, "accepted": 1, "rejected": 0, "mean_confidence": .98,
    }


def ingestion(config):
    repo, store = MemoryRepository(), MemoryVectorStore()
    embedder = HashEmbeddingProvider(dim=64)
    service = IngestionService(
        repository=repo, store=store, embedder=embedder, ocr=config,
        chunker=StructureAwareChunker(HeuristicTokenCounter()),
    )
    return repo, store, embedder, service


def test_page_detection_preserves_native_and_blank_pages():
    config = OCRConfig()
    page = page_with_graphics()
    assert needs_ocr(page, [], config)
    # A short digital header above a scanned image must not hide the scan.
    assert needs_ocr(page, fake_lines("header"), config)
    assert not needs_ocr(page, fake_lines("有效原文" * 100), config)
    assert needs_ocr(page, fake_lines("(cid:123)" * 30), config)
    page.images = []
    assert not needs_ocr(page, [], config)
    assert needs_ocr(page, [], replace(config, mode="force"))


def test_mixed_pdf_only_ocr_second_page(samples, monkeypatch):
    calls = []
    def recognize(page, config):
        calls.append(page.page_number)
        return fake_recognize(page, config)
    monkeypatch.setattr("rag.parsers.pdf.recognize_page", recognize)
    parsed = parse_document(samples["mixed"], doc_id="mixed", ocr=OCRConfig())
    assert calls == [2] and parsed.meta.ocr_pages == [2]
    assert "Native page:" in parsed.full_text()
    ocr_nodes = [n for n in parsed.nodes if n.extraction_method == "ocr"]
    assert ocr_nodes and ocr_nodes[0].page_start == 2
    assert ocr_nodes[0].bbox is not None
    assert ocr_nodes[0].ocr_confidence == .98


def test_native_only_mode_reports_missing_scan_instead_of_partial_success(samples):
    with pytest.raises(NeedsOCRError) as error:
        parse_document(samples["mixed"], doc_id="mixed", ocr=OCRConfig(mode="off"))
    assert error.value.extensions["pages"] == [2]


def test_ocr_page_limit_fails_closed(samples, monkeypatch):
    monkeypatch.setattr("rag.parsers.pdf.recognize_page", fake_recognize)
    with pytest.raises(NeedsOCRError) as error:
        parse_document(samples["scanned"], doc_id="s", ocr=OCRConfig(max_pages=1))
    assert error.value.extensions["reason"] == "ocr_page_limit"
    assert error.value.extensions["pages"] == [2]


def test_blank_raster_page_skips_engine(samples, monkeypatch):
    def must_not_load(*args):
        raise AssertionError("blank page loaded OCR")
    monkeypatch.setattr("rag.parsers.ocr._engine", must_not_load)
    parsed = parse_document(samples["blank"], doc_id="blank", ocr=OCRConfig())
    assert parsed.meta.blank_pages == [1]
    assert parsed.nodes == []


def test_missing_models_error_is_actionable(tmp_path):
    from rag.parsers.ocr import _model_paths
    with pytest.raises(NeedsOCRError, match="setup_ocr"):
        _model_paths(tmp_path)


def test_low_confidence_and_coordinate_mapping(monkeypatch):
    image = Image.new("RGB", (1000, 1600), "white")
    ImageDraw.Draw(image).rectangle((50, 100, 900, 150), fill="black")
    page = page_with_graphics()
    page.to_image = lambda **kwargs: SimpleNamespace(original=image)
    result = SimpleNamespace(
        txts=["清晰文字", "模糊文字"], scores=[.98, .2],
        boxes=[[[100, 200], [400, 200], [400, 240], [100, 240]]] * 2,
    )
    monkeypatch.setattr("rag.parsers.ocr._engine", lambda *args: lambda image: result)
    with pytest.raises(NeedsOCRError) as error:
        recognize_page(page, OCRConfig())
    assert error.value.extensions["reason"] == "ocr_low_confidence"
    lines, _, stats = recognize_page(page, OCRConfig(max_rejected_ratio=.6))
    assert len(lines) == 1 and stats["rejected"] == 1
    assert lines[0]["x0"] == 60 and lines[0]["top"] == 100


def test_two_column_reading_order():
    lines = []
    for col in range(2):
        for row in range(5):
            lines.append({"text": f"{col}:{row}", "x0": 20 + col * 320,
                          "x1": 240 + col * 320, "top": 100 + row * 30,
                          "bottom": 112 + row * 30, "column": 0})
    ordered, split = order_ocr_lines(lines, 600)
    assert split == 300
    assert [line["text"] for line in ordered] == [
        *[f"0:{i}" for i in range(5)], *[f"1:{i}" for i in range(5)],
    ]


async def test_failed_scan_reupload_recovers_same_document(samples, monkeypatch):
    repo, store, embedder, service = ingestion(OCRConfig(mode="off"))
    with pytest.raises(NeedsOCRError):
        await service.ingest_file(samples["mixed"])
    docs = await repo.list_documents()
    assert len(docs) == 1 and docs[0].status == "needs_ocr"
    assert await repo.count_chunks() == 0 and await store.count() == 0
    service.ocr = OCRConfig()
    monkeypatch.setattr("rag.parsers.pdf.recognize_page", fake_recognize)
    result = await service.ingest_file(samples["mixed"])
    assert result.document_id == docs[0].id and result.status == "ready"
    assert len(await repo.list_documents()) == 1
    retrieval = RetrievalService(
        repository=repo, store=store, embedder=embedder, translate_default=False,
    )
    found = await retrieval.retrieve("数据备份", use_dense=False, use_rerank=False)
    assert found.chunks[0].page_start == 2
    assert found.chunks[0].extraction_method == "ocr"
    assert found.chunks[0].ocr_confidence == .98
    doc = await repo.get_document(result.document_id)
    assert doc.extra["ocr_pages"] == [2]


def test_provenance_handles_native_ocr_and_mixed_spans():
    extra = {"ocr_pages": [2], "ocr_stats": {"2": {"mean_confidence": .9}}}
    assert _extraction_info(1, 1, extra)["extraction_method"] == "native"
    assert _extraction_info(2, 2, extra) == {"extraction_method": "ocr", "ocr_confidence": .9}
    assert _extraction_info(1, 2, extra)["extraction_method"] == "mixed"


async def test_parser_runs_outside_api_event_loop(samples, monkeypatch):
    import asyncio

    original = parse_document
    running = threading.Event()
    released = threading.Event()
    def parse(*args, **kwargs):
        running.set()
        assert released.wait(2), "event loop remained blocked by the parser"
        return original(*args, **kwargs)
    monkeypatch.setattr("rag.services.ingestion.parse_document", parse)
    monkeypatch.setattr("rag.parsers.pdf.recognize_page", fake_recognize)
    _, _, _, service = ingestion(OCRConfig())
    async def release():
        while not running.is_set():
            await asyncio.sleep(.001)
        released.set()
    await asyncio.gather(service.ingest_file(samples["mixed"]), release())


@pytest.mark.ocr
async def test_real_ocr_ingestion_and_page_aware_retrieval(samples):
    pytest.importorskip("rapidocr")
    config = OCRConfig()
    if not (config.model_dir / "det.onnx").is_file():
        pytest.skip("Run scripts/setup_ocr.py before real OCR acceptance")
    repo, store, embedder, service = ingestion(config)
    result = await service.ingest_file(samples["mixed"])
    assert result.status == "ready"
    retrieval = RetrievalService(
        repository=repo, store=store, embedder=embedder, translate_default=False,
    )
    found = await retrieval.retrieve("数据备份", use_dense=False, use_rerank=False)
    assert any(c.page_start == 2 and "两份" in c.content for c in found.chunks)


def test_pixel_limit_rejects_pathological_page_before_rendering():
    page = SimpleNamespace(width=10**8, height=10**8)
    with pytest.raises(NeedsOCRError) as error:
        recognize_page(page, OCRConfig())
    assert error.value.extensions["reason"] == "ocr_pixel_limit"


async def test_blank_metadata_never_becomes_ready_document(samples):
    repo, store, _, service = ingestion(OCRConfig())
    from rag.core.errors import IngestionError
    with pytest.raises(IngestionError):
        await service.ingest_file(samples["blank"])
    docs = await repo.list_documents()
    assert docs[0].status == "failed"
    assert await repo.count_chunks() == 0 and await store.count() == 0


async def test_upload_returns_422_and_document_list_retains_ocr_status(samples):
    import httpx

    from rag.api.routes import _doc_out
    from rag.api.schemas import ChatResponse
    from rag.core.config import Settings
    from rag.main import app

    config = Settings(_env_file=None, repository_backend="memory", vector_backend="memory",
                      upload_dir=samples["mixed"].parent / "uploads")
    config.ensure_dirs()
    repo, _, _, service = ingestion(OCRConfig(mode="off"))
    old = getattr(app.state, "ctx", None)
    app.state.ctx = SimpleNamespace(settings=config, repository=repo, ingestion=service)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            response = await client.post("/api/v1/documents?sync=true", files={
                "file": ("scan.pdf", samples["mixed"].read_bytes(), "application/pdf"),
            })
        assert response.status_code == 422
        assert response.json()["pages"] == [2]
        assert response.json()["reason"] == "ocr_disabled"
        doc = (await repo.list_documents())[0]
        assert _doc_out(doc).status == "needs_ocr"
    finally:
        app.state.ctx = old
    from rag.api.routes import _to_chat_response
    response: ChatResponse = _to_chat_response("s", {
        "answer": "保留两份。[1]", "route": "generate", "verified": True,
        "chunks": [{"rank": 1, "chunk_id": 1, "doc_title": "scan",
                    "page_start": 2, "extraction_method": "ocr", "ocr_confidence": .91}],
        "citations": [{"rank": 1, "chunk_id": 1, "quote": "保留两份"}],
    })
    assert response.citations[0].extraction_method == "ocr"
    assert response.citations[0].ocr_confidence == .91
