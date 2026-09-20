"""端到端冒烟：解析 → 分块 → 本地 BGE 嵌入 → 内存向量库 → 混合检索 → Agent。

★ 这个脚本是 Windows 本地开发的**唯一验收标准**。
   它不依赖 Docker / Postgres / Milvus，跑通了说明整条业务链路是活的。
   Milvus 路径另由 scripts/smoke_milvus.py 在 Ubuntu 上验证。

    python scripts/smoke_local.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ★ Windows 控制台默认 GBK，打印 ✓ / ✗ 会直接 UnicodeEncodeError 崩掉脚本。
#   必须在任何输出之前重配 stdout。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SAMPLE = """# RAG-Agent 部署手册

## 1. 环境要求

部署前需要准备一台 Ubuntu 22.04 或更高版本的服务器。
内存不低于 8GB，其中 Milvus 单独占用约 5GB。
CPU 必须支持 AVX2 指令集，否则 Milvus 会以 Illegal instruction 崩溃重启。

## 2. 部署步骤

第一步，克隆仓库并进入目录，执行 docker compose up -d 启动全部服务。
第二步，等待 Milvus 健康检查通过，首次启动大约需要 90 秒。
第三步，执行数据库迁移，创建 documents 与 chunks 两张表。

## 3. 关键配置

分块大小由 CHUNK_TARGET_TOKENS 控制，默认值为 256。
嵌入模型使用 bge-m3，向量维度为 1024，最大输入长度 8192 个 token。
检索阶段融合稠密与稀疏两路召回，RRF 的平滑常数 k 取 60。

## 4. 常见故障

如果 Milvus 反复重启，先检查 CPU 是否支持 AVX2，再检查内存是否被限制在 5GB 以下。
如果检索结果为空，检查 embedding 模型路径是否挂载正确。
"""


async def main() -> int:
    from rag.chunking.counter import TransformersTokenCounter
    from rag.chunking.splitter import StructureAwareChunker
    from rag.core.config import get_settings
    from rag.core.logging import setup_logging
    from rag.infra.repository import MemoryRepository
    from rag.infra.vectorstore import MemoryVectorStore
    from rag.parsers.base import parse_document
    from rag.providers import build_embedding_provider, build_llm, build_reranker
    from rag.services.ingestion import IngestionService
    from rag.services.retrieval import RetrievalService

    setup_logging("INFO", json_output=False)
    settings = get_settings()

    print("=" * 70)
    print("RAG-Agent 本地端到端冒烟")
    print("=" * 70)
    print(f"  向量后端   : {settings.vector_backend}")
    print(f"  嵌入模型   : {settings.embed_provider} / {settings.embed_model_id}")
    print(f"  模型路径   : {settings.embed_model_path}")
    print(f"  重排       : {settings.rerank_provider}")
    print(f"  LLM        : {settings.llm_provider}")
    print()

    model_dir = Path(settings.embed_model_path)
    if not model_dir.exists():
        print(f"[FAIL] 模型目录不存在：{model_dir}")
        return 1

    # ---- 0. 写一个样例文档 ----
    data_dir = ROOT / "data" / "samples"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = data_dir / "deploy_guide.md"
    sample_path.write_text(SAMPLE, encoding="utf-8")
    print(f"[0/6] 样例文档 {sample_path.name}（{len(SAMPLE)} 字符）")

    # ---- 1. 解析 ----
    t0 = time.perf_counter()
    parsed = parse_document(sample_path, doc_id="smoke-1")
    print(f"[1/6] 解析完成：{len(parsed.nodes)} 个节点，"
          f"lang={parsed.meta.lang}，{_ms(t0)}")
    for node in parsed.nodes[:6]:
        print(f"        {node.type!s:<10} L{node.level} | {node.text[:44]!r}")
    if len(parsed.nodes) > 6:
        print(f"        ... 还有 {len(parsed.nodes) - 6} 个节点")

    # ---- 2. 分块（tokenizer 用模型自己的）----
    t0 = time.perf_counter()
    counter = TransformersTokenCounter.from_model_dir(model_dir)
    print(f"[2/6] Tokenizer 就绪：max_tokens={counter.max_tokens}，{_ms(t0)}")

    chunker = StructureAwareChunker(
        counter,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        parent_target_tokens=settings.parent_target_tokens,
        chunker_version=settings.chunker_version,
        embed_model_id=settings.embed_model_id,
        embed_dim=settings.embed_dim,
        doc_title=parsed.meta.title or "",
    )
    t0 = time.perf_counter()
    chunks = chunker.split(parsed)
    print(f"[3/6] 分块完成：{len(chunks)} 块，{_ms(t0)}")
    for chunk in chunks:
        print(f"        #{chunk.chunk_index} {chunk.token_count:>3}tk "
              f"p{chunk.page_start} parent={chunk.parent_index} "
              f"| {chunk.section_path[:28]!r}")
        assert chunk.token_count <= counter.max_tokens, (
            f"块 {chunk.chunk_index} 超过模型上限：{chunk.token_count}"
        )

    # ---- 3. 嵌入 ----
    embedder = build_embedding_provider(settings)
    store = MemoryVectorStore(dim=settings.embed_dim)
    repo = MemoryRepository()

    t0 = time.perf_counter()
    vectors = await embedder.aembed_documents([c.embed_text for c in chunks])
    print(f"[4/6] 嵌入完成：{len(vectors)} 条，dim={len(vectors[0])}，{_ms(t0)}")
    norms = [sum(v * v for v in vec) ** 0.5 for vec in vectors]
    print(f"        向量模长 min={min(norms):.4f} max={max(norms):.4f}（应接近 1.0）")

    # ---- 4. 入库（走完整 IngestionService，而不是手工塞）----
    ingestion = IngestionService(
        repository=repo, store=store, embedder=embedder, chunker=chunker,
        chunker_version=settings.chunker_version,
    )
    result = await ingestion.ingest_file(sample_path, title="RAG-Agent 部署手册")
    print(f"[5/6] 入库完成：document_id={result.document_id} "
          f"chunks={result.chunk_count} vectors={result.vectors_written} "
          f"{result.duration_ms}ms")

    # 幂等验证：再传一次，应该走 dedup 而不是重新解析
    again = await ingestion.ingest_file(sample_path)
    assert again.deduplicated, "重复上传没有命中幂等短路"
    print(f"        幂等复验：deduplicated={again.deduplicated} ✓")

    # ---- 5. 检索 ----
    retrieval = RetrievalService(
        embedder=embedder, store=store, repository=repo,
        reranker=build_reranker(settings),
        rrf_k=settings.rrf_k,
        dense_top_k=settings.retrieve_dense_top_k,
        sparse_top_k=settings.retrieve_sparse_top_k,
        final_top_k=settings.retrieve_final_top_k,
    )

    queries = [
        ("Milvus 崩了怎么办", "AVX2"),
        ("分块大小默认多少", "256"),
        ("向量维度是多少", "1024"),
        ("部署需要多长时间", "90"),
    ]
    # ★ 判据是"正确答案出现在前 3 条"，不是"排第一"。
    #   排第一受 RRF 分数抖动影响很大，而 RAG 的容错在于喂给 LLM 的是 top-k ——
    #   只要正确证据在窗口内，生成就能用上。卡 top-1 会让冒烟测试变成噪声源。
    HIT_WINDOW = 3

    print("[6/6] 混合检索验证：")
    all_ok = True
    for query, expect in queries:
        res = await retrieval.retrieve(query)
        top = res.chunks[0] if res.chunks else None
        found_at = next(
            (i for i, c in enumerate(res.chunks[:HIT_WINDOW], start=1) if expect in c.content),
            None,
        )
        hit = found_at is not None
        all_ok &= hit
        print(f"\n  问：{query}")
        print(f"  诊断：{res.diagnostics['recall']} → 融合 {res.diagnostics['fused']} "
              f"→ 返回 {res.diagnostics['returned']}")
        if top:
            print(f"  首条：dense#{top.dense_rank} sparse#{top.sparse_rank} "
                  f"score={top.score:.5f}")
            print(f"  出处：{top.label}")
            print(f"  正文：{top.content[:70]!r}...")
        print(f"  期望 {expect!r} 落在前 {HIT_WINDOW} 条："
              f"{'✓ 第 ' + str(found_at) + ' 位' if hit else '✗ 未命中'}")

    # ---- 6. Agent 全图（无 LLM 时走抽取式兜底）----
    print("\n[7/7] Agent 图执行：")
    from rag.agent.graph import build_graph, run_agent

    graph = build_graph(
        retrieval=retrieval, llm=build_llm(settings),
        checkpointer=None, top_k=settings.retrieve_final_top_k,
        max_retries=settings.agent_max_retries,
    )
    state = await run_agent(
        graph, "Milvus 反复重启是什么原因？",
        thread_id="smoke-thread", max_retries=settings.agent_max_retries,
        recursion_limit=settings.agent_recursion_limit,
    )
    print(f"  路由      ：{state.get('route')}")
    print(f"  判定理由  ：{state.get('grade_reason')}")
    print(f"  引用校验  ：verified={state.get('verified')} "
          f"citations={len(state.get('citations') or [])}")
    print(f"  回答      ：{(state.get('answer') or '')[:160]}")

    print("\n" + "=" * 70)
    if all_ok:
        print("全部通过 ✓  本地链路可用")
        return 0
    print("检索断言未全部命中 ✗")
    return 1


def _ms(t0: float) -> str:
    return f"{(time.perf_counter() - t0) * 1000:.0f}ms"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
