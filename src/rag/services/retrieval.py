"""检索服务：混合召回 → RRF 融合 → 重排 → 组装上下文。

★ 检索流程（每一步都有存在的理由，删任何一步都要有消融数据支撑）：

    query
      ├─ 稠密召回（BGE 向量，语义匹配）      ─┐
      └─ 稀疏召回（BM25，关键词/术语/编号）   ─┴─→ RRF 融合 → 重排 → top-k
                                                        ↓
                                              small-to-big 扩窗（可选）
                                                        ↓
                                                 带编号的上下文

★ 为什么要混合：稠密召回对"专有名词、型号、条款编号、罕见词"几乎无感 ——
  这些词的向量在训练里没见过，会被平均掉。BM25 恰好补这一块。
  反过来 BM25 处理不了同义改写。两者是互补而非竞争关系。

★ 为什么召回的 top_k（50）远大于喂给 LLM 的 top_k（5）：
  召回阶段追求高 recall，排序阶段才追求 precision。50→5 的压缩比是经验值，
  太小会漏证据，太大会拖慢重排（cross-encoder 是 O(n) 次前向）。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from rag.core.logging import get_logger
from rag.infra.repository import ChunkRecord, Repository
from rag.infra.vectorstore import VectorHit, VectorStore
from rag.providers.base import EmbeddingProvider, Reranker, is_enabled
from rag.services.fusion import FusedItem, RankedItem, dedupe_by_content, rerank_order, rrf_fuse
from rag.services.translate import QueryTranslator, needs_translation

logger = get_logger(__name__)

_BIBLIOGRAPHIC_QUERY = re.compile(
    r"(参考文献|引用|出处|作者|课题组|发表|期刊|会议|哪一年|年份|卷号|"
    r"\b(?:reference|citation|author|published|journal|conference|volume|doi)\b)",
    re.I,
)
_MULTI_DOCUMENT_QUERY = re.compile(
    r"(哪些论文|哪几篇|这几篇|多篇(?:论文|文章|文献)|多份文档|各篇|跨文档|"
    r"文献(?:对比|比较|汇总|综述)|"
    r"\b(?:which papers|these papers|multiple papers|across papers|"
    r"multiple documents|across documents|list the papers)\b)",
    re.I,
)
_AUTHOR_COMPARISON_QUERY = re.compile(
    r"(课题组|研究组|相同作者|同一作者|共同作者|作者重合|"
    r"\b(?:research group|same authors?|shared authors?|co-?authors?)\b)",
    re.I,
)
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN_TERM = re.compile(r"\b[A-Za-z][A-Za-z-]{3,}\b")


@dataclass
class RetrievedChunk:
    """检索结果的最终形态 —— agent 与 API 层都只看这个结构。"""

    chunk_id: int
    content: str
    score: float
    document_id: int
    doc_title: str = ""
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    node_type: str = "paragraph"
    rank: int = 0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    # ★ 稠密通道的**原始余弦分**。`score` 是 RRF 融合分，只反映名次不反映相关性
    #   （rank 1 且 k=60 时恒等于 0.016393），拿它判断"这条到底像不像"必然误判。
    #   余弦分是这里唯一带绝对意义的量：领域内实测 0.51~0.60，领域外 0.29~0.49。
    #   注意它**不能当阈值用**（两者区间重叠），只用于展示。
    dense_score: float | None = None
    # 同 section 的兄弟块 —— small-to-big 用，不进上下文，按需展开
    sibling_ids: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        """给 LLM 看的定位串。有页码就带页码，方便用户核对。"""
        parts = [self.doc_title or f"doc-{self.document_id}"]
        if self.section_path:
            parts.append(self.section_path)
        if self.page_start:
            parts.append(f"p{self.page_start}" if self.page_start == self.page_end
                         else f"p{self.page_start}-{self.page_end}")
        return " · ".join(parts)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    # 诊断信息：评测与 debug 需要知道"每一路各召回了多少、融合后剩多少"
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)


class RetrievalService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        store: VectorStore,
        repository: Repository,
        reranker: Reranker | None = None,
        translator: QueryTranslator | None = None,
        translate_default: bool = True,
        translate_target: str = "en",
        rrf_k: int = 60,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        fused_top_k: int = 60,
        final_top_k: int = 5,
        max_context_chars: int = 12000,
        expand_siblings: bool = False,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.repo = repository
        self.reranker = reranker
        self.translator = translator
        self.translate_default = translate_default
        self.translate_target = translate_target
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fused_top_k = fused_top_k
        self.final_top_k = final_top_k
        self.max_context_chars = max_context_chars
        self.expand_siblings = expand_siblings

    # ==================================================================
    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: int = 1,
        top_k: int | None = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_rerank: bool = True,
        use_translate: bool | None = None,
    ) -> RetrievalResult:
        """各 use_* 开关是为消融实验留的 —— 评测里要能一键关掉任一路。

        `use_translate` 为 None 时取服务默认值（`translate_default`）。
        """
        query = (query or "").strip()
        if not query:
            return RetrievalResult(chunks=[], diagnostics={"reason": "empty_query"})

        top_k = top_k or self.final_top_k
        errors: list[str] = []

        # ---- 跨语言查询扩展 -------------------------------------------
        # ★ 是**追加一路**而不是替换原查询：原问题和译文各能召回对方漏掉的块，
        #   并集的召回上限高于任一路。见 services/translate.py 顶部的实测数字。
        #   后缀 `_x` 把扩展路和原路在 diagnostics 里分开，便于消融归因。
        variants: list[tuple[str, str]] = [("", query)]
        translate_on = (
            self.translate_default if use_translate is None else use_translate
        ) and self.translator is not None
        if translate_on and needs_translation(query, target=self.translate_target):
            try:
                translated = await self.translator.translate(query)
            except Exception as exc:  # noqa: BLE001 — 扩展失败只降级，不整体失败
                errors.append(f"translate: {type(exc).__name__}: {exc}")
                translated = None
            if translated and translated != query:
                variants.append(("_x", translated))

        # ★ 每一路召回并发。串行的话延迟直接相加，而它们之间没有任何依赖。
        tasks = []
        names = []
        for suffix, variant in variants:
            if use_dense:
                tasks.append(self._dense(variant, tenant_id=tenant_id))
                names.append(f"dense{suffix}")
            if use_sparse:
                tasks.append(self._sparse(variant, tenant_id=tenant_id))
                names.append(f"sparse{suffix}")
        raw_runs = await asyncio.gather(*tasks, return_exceptions=True)

        runs: dict[str, list[VectorHit]] = {}
        for name, outcome in zip(names, raw_runs, strict=False):
            if isinstance(outcome, BaseException):
                # ★ 单路失败不整体失败 —— 降级成单路检索，总比 500 好
                errors.append(f"{name}: {type(outcome).__name__}: {outcome}")
                logger.warning("retrieval.channel_failed", channel=name, exc_info=outcome)
                continue
            runs[name] = outcome

        if not runs:
            return RetrievalResult(chunks=[], diagnostics={"errors": errors})

        fused = rrf_fuse(
            {
                name: [RankedItem(chunk_id=h.chunk_id, rank=h.rank, score=h.score, source=name)
                       for h in hits]
                for name, hits in runs.items()
            },
            k=self.rrf_k,
        )

        # RRF 汇总分会偏爱“被多个通道同时命中”的块。如果直接截断，某一路头部的
        # 独有结果也可能被删掉（实测英文 BM25 第 3 的正确答案曾因此进不了重排）。
        # 保留全局 RRF 头部，并为每个通道保底前 10，再交给 cross-encoder 判断。
        fused = self._select_rerank_candidates(fused, runs)
        candidate_count = len(fused)
        rerank_on = use_rerank and is_enabled(self.reranker)
        if rerank_on and fused:
            translated_query = next(
                (value for suffix, value in variants if suffix == "_x"), None
            )
            fused = await self._rerank(
                query, fused, translated_query=translated_query
            )

        # 科研问答的两类常见偏差在最终截断前修正：参考文献列表不应挤掉正文证据；
        # “哪些论文/对比”类问题则需要覆盖多篇文献，不能让同一篇占满 top-k。
        fused = await self._postprocess_order(query, fused)

        fused = fused[:top_k]
        if not fused:
            return RetrievalResult(chunks=[], diagnostics={"errors": errors, "fused": 0})

        records = await self.repo.get_chunks([f.chunk_id for f in fused])
        by_id = {r.chunk_id: r for r in records}
        # ★ 回表可能缺数据（Milvus 有、Postgres 已删）。缺的必须丢，不能拿 Milvus 的
        #   content 顶替 —— 那份是派生副本，可能已过期。
        missing = [f.chunk_id for f in fused if f.chunk_id not in by_id]
        if missing:
            logger.warning("retrieval.missing_in_postgres", chunk_ids=missing[:10], n=len(missing))

        chunks: list[RetrievedChunk] = []
        for item in fused:
            record = by_id.get(item.chunk_id)
            if record is None:
                continue
            chunks.append(self._to_chunk(record, item))

        chunks = await self._attach_siblings(chunks)
        chunks = self._dedupe(chunks)

        diagnostics = {
            "recall": {name: len(hits) for name, hits in runs.items()},
            "fused": len(fused),
            "returned": len(chunks),
            "missing_in_postgres": len(missing),
            "rerank": rerank_on,
            "rerank_candidates": candidate_count if rerank_on else 0,
            # ★ 把实际用的查询变体打出来。跨语言扩展是否触发、译文长什么样，
            #   是排查"召回为什么变了"的第一现场 —— 不然只能靠猜。
            "queries": {name or "orig": q for name, q in variants},
            "errors": errors,
        }
        logger.info("retrieval.done", query_len=len(query), **{
            k: v for k, v in diagnostics.items() if k != "errors"
        })
        return RetrievalResult(chunks=chunks, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    async def _dense(self, query: str, *, tenant_id: int) -> list[VectorHit]:
        # ★ 查询侧的 instruction 前缀只加在 aembed_query 里，文档侧不加 ——
        #   BGE 是非对称模型，给文档也加前缀会让向量空间错位（掉点 3~8 个点）。
        vector = await self.embedder.aembed_query(query)
        return await self.store.search_dense(vector, top_k=self.dense_top_k, tenant_id=tenant_id)

    async def _sparse(self, query: str, *, tenant_id: int) -> list[VectorHit]:
        return await self.store.search_sparse(query, top_k=self.sparse_top_k, tenant_id=tenant_id)

    def _select_rerank_candidates(
        self, fused: list[FusedItem], runs: dict[str, list[VectorHit]]
    ) -> list[FusedItem]:
        """保留 RRF 头部和每个召回通道的头部独有结果。"""
        selected_ids = {item.chunk_id for item in fused[:self.fused_top_k]}
        for hits in runs.values():
            selected_ids.update(hit.chunk_id for hit in hits[:10])
        return [item for item in fused if item.chunk_id in selected_ids]

    @staticmethod
    def _has_mixed_language_term(query: str) -> bool:
        """中文问题里保留有意义的拉丁术语，纯缩写由英文译问覆盖。"""
        if not _CJK.search(query):
            return False
        return any(
            not term.isupper()
            for term in _LATIN_TERM.findall(query)
        )

    @staticmethod
    def _mostly_cjk(text: str) -> bool:
        cjk = len(_CJK.findall(text))
        latin = sum(ch.isascii() and ch.isalpha() for ch in text)
        return cjk >= 8 and cjk >= latin * 0.2

    async def _rerank(
        self,
        query: str,
        fused: list[FusedItem],
        *,
        translated_query: str | None = None,
    ) -> list[FusedItem]:
        records = await self.repo.get_chunks([f.chunk_id for f in fused])
        by_id = {r.chunk_id: r for r in records}
        ids = [f.chunk_id for f in fused if f.chunk_id in by_id]
        if not ids:
            return fused

        cjk_ratio = sum(self._mostly_cjk(by_id[cid].content) for cid in ids) / len(ids)
        if not translated_query:
            selected_query = query
        elif _BIBLIOGRAPHIC_QUERY.search(query) or self._has_mixed_language_term(query):
            selected_query = f"{query}\n{translated_query}"
        elif cjk_ratio <= 0.35:
            selected_query = translated_query
        elif cjk_ratio >= 0.65:
            selected_query = query
        else:
            selected_query = f"{query}\n{translated_query}"

        documents = [
            f"Document: {by_id[cid].doc_title}\n"
            f"Section: {by_id[cid].section_path}\n"
            f"{by_id[cid].content}"
            for cid in ids
        ]
        try:
            ranked = await self.reranker.arerank(selected_query, documents)
        except Exception:
            # ★ 重排失败降级为 RRF 顺序，而不是让整个请求失败
            logger.warning("retrieval.rerank_failed", exc_info=True)
            return fused
        scores = {ids[idx]: score for idx, score in ranked if 0 <= idx < len(ids)}
        return rerank_order(fused, scores)

    async def _postprocess_order(self, query: str, fused: list[FusedItem]) -> list[FusedItem]:
        if not fused:
            return fused
        records = await self.repo.get_chunks([item.chunk_id for item in fused])
        by_id = {record.chunk_id: record for record in records}

        ordered = list(fused)
        author_comparison = bool(_AUTHOR_COMPARISON_QUERY.search(query))
        if author_comparison:
            # “同一团队还发表过什么”首先需要题名页作者信息。cross-encoder
            # 往往会把含大量人名的参考文献误判得更相关，因此这里用结构类型纠偏。
            # 只提升已经被语义/关键词通道召回的 metadata，不绕过相关性召回。
            ordered.sort(key=lambda item: (
                by_id.get(item.chunk_id) is None
                or by_id[item.chunk_id].node_type != "metadata"
            ))
        elif not _BIBLIOGRAPHIC_QUERY.search(query):
            ordered.sort(key=lambda item: (
                by_id.get(item.chunk_id) is None
                or by_id[item.chunk_id].node_type in {"reference", "metadata"}
            ))

        if _MULTI_DOCUMENT_QUERY.search(query):
            first_per_doc: list[FusedItem] = []
            remainder: list[FusedItem] = []
            seen_docs: set[int] = set()
            for item in ordered:
                record = by_id.get(item.chunk_id)
                document_id = record.document_id if record else -item.chunk_id
                if document_id not in seen_docs:
                    seen_docs.add(document_id)
                    first_per_doc.append(item)
                else:
                    remainder.append(item)
            ordered = first_per_doc + remainder
        return ordered

    def _to_chunk(self, record: ChunkRecord, item: FusedItem) -> RetrievedChunk:
        dense_ranks = [rank for source, rank in item.ranks.items()
                       if source.startswith("dense")]
        sparse_ranks = [rank for source, rank in item.ranks.items()
                        if source.startswith("sparse")]
        dense_scores = [score for source, score in item.scores.items()
                        if source.startswith("dense")]
        return RetrievedChunk(
            chunk_id=record.chunk_id,
            content=record.content,
            score=round(item.score, 6),
            document_id=record.document_id,
            doc_title=record.doc_title,
            section_path=record.section_path,
            page_start=record.page_start,
            page_end=record.page_end,
            node_type=record.node_type,
            dense_rank=min(dense_ranks, default=None),
            sparse_rank=min(sparse_ranks, default=None),
            rerank_score=item.scores.get("rerank"),
            dense_score=(round(max(dense_scores), 6) if dense_scores else None),
        )

    async def _attach_siblings(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self.expand_siblings:
            return chunks
        # 只在需要时查同 section 的兄弟块，且同一 parent 只查一次
        seen: dict[tuple[int, int], list[int]] = {}
        records = await self.repo.get_chunks([c.chunk_id for c in chunks])
        by_id = {r.chunk_id: r for r in records}
        for chunk in chunks:
            record = by_id.get(chunk.chunk_id)
            if record is None or record.parent_index is None:
                continue
            key = (record.document_id, record.parent_index)
            if key not in seen:
                window = await self.repo.get_section_window(
                    record.document_id, record.version, record.parent_index
                )
                seen[key] = [w.chunk_id for w in window]
            chunk.sibling_ids = [cid for cid in seen[key] if cid != chunk.chunk_id]
        return chunks

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        from rag.services.fusion import FusedItem as _F

        contents = {c.chunk_id: c.content for c in chunks}
        proxy = [_F(chunk_id=c.chunk_id, score=c.score) for c in chunks]
        kept = {f.chunk_id for f in dedupe_by_content(proxy, contents)}
        out = [c for c in chunks if c.chunk_id in kept]
        for i, chunk in enumerate(out, start=1):
            chunk.rank = i
        return out

    # ==================================================================
    def build_context(self, chunks: list[RetrievedChunk], *, max_chars: int | None = None) -> str:
        """把检索结果拼成带编号的上下文。

        ★ 编号格式 `[1]` 必须和 system prompt 里要求的引用格式严格一致。
          模型只会模仿它看到的东西 —— 上下文里写 `[1]`、prompt 里写 `【1】`，
          引用就会乱。这个约束在 prompts.py 里有一份对应的常量。
        """
        budget = max_chars or self.max_context_chars
        blocks: list[str] = []
        used = 0
        for chunk in chunks:
            block = f"[{chunk.rank}] {chunk.label}\n{chunk.content}"
            if used + len(block) > budget and blocks:
                # ★ 截断要记日志 —— 静默丢弃会让"模型说找不到答案"变成一个查不出的 bug
                logger.info("context.truncated", kept=len(blocks), total=len(chunks),
                            budget=budget)
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)
