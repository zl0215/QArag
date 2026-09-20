"""向量存储抽象 + 内存实现。

★ 为什么要有内存实现（而不是"直接上 Milvus"）：
   ① Windows 上跑 Milvus 需要 Docker，开发期不该有这个门槛
   ② 单元测试不该依赖外部服务
   ③ 有了它，MemoryVectorStore 与 MilvusVectorStore 可以跑同一套契约测试，
      保证 fake 不会悄悄漂移

内存实现里的 BM25 是真实现的（不是假的），所以本地开发时混合检索的行为
与 Milvus 内建 BM25 在语义上一致 —— 只是 IDF 统计范围不同。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    chunk_id: int
    dense: list[float]
    content: str                       # 供 BM25 分词用；权威副本仍在 Postgres
    document_id: int
    tenant_id: int = 1
    node_type: str = "paragraph"
    lang: str | None = None
    content_hash: str = ""
    chunker_ver: str = "v1"
    embed_model: str = ""
    is_active: bool = True


@dataclass
class VectorHit:
    chunk_id: int
    score: float
    rank: int = 0
    document_id: int | None = None
    content: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def upsert(self, records: list[VectorRecord]) -> int: ...

    async def delete_by_document(self, document_id: int, tenant_id: int = 1) -> int: ...

    async def search_dense(
        self, vector: list[float], *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]: ...

    async def search_sparse(
        self, query_text: str, *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]: ...

    async def count(self, tenant_id: int = 1) -> int: ...

    async def aclose(self) -> None: ...


# ======================================================================
# 分词：CJK 用「单字 + 二元组」，拉丁用词
# ======================================================================
def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def tokenize(text: str) -> list[str]:
    """无需 jieba 的中文分词方案。

    jieba 最后发版是 2020 年，引入它是负债。字符二元组对 BM25 足够好：
    既保留了"词"的区分度（"向量" vs "向量库"），又不依赖词典。
    """
    out: list[str] = []
    latin: list[str] = []
    cjk: list[str] = []

    def flush_latin() -> None:
        if latin:
            out.append("".join(latin).lower())
            latin.clear()

    def flush_cjk() -> None:
        if cjk:
            run = "".join(cjk)
            out.extend(run)                                    # 单字
            out.extend(run[i:i + 2] for i in range(len(run) - 1))  # 二元组
            cjk.clear()

    for ch in text:
        if _is_cjk(ch):
            flush_latin()
            cjk.append(ch)
        elif ch.isalnum():
            flush_cjk()
            latin.append(ch)
        else:
            flush_latin()
            flush_cjk()
    flush_latin()
    flush_cjk()
    return out


class _BM25:
    """标准 BM25（k1=1.2, b=0.75）—— 与 Milvus 默认参数一致，便于对照。"""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: dict[int, Counter[str]] = {}
        self._len: dict[int, int] = {}
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0

    def add(self, doc_id: int, text: str) -> None:
        terms = tokenize(text)
        self._tf[doc_id] = Counter(terms)
        self._len[doc_id] = max(len(terms), 1)
        self._df.update(set(terms))
        self._n += 1
        self._avgdl = sum(self._len.values()) / self._n

    def remove(self, doc_id: int) -> None:
        tf = self._tf.pop(doc_id, None)
        length = self._len.pop(doc_id, None)
        if tf is None or length is None:
            return
        self._df.subtract(set(tf))
        self._df = +self._df            # 丢弃非正计数
        self._n = max(self._n - 1, 0)
        self._avgdl = (sum(self._len.values()) / self._n) if self._n else 0.0

    def score(self, query: str, doc_id: int) -> float:
        tf = self._tf.get(doc_id)
        if not tf or not self._n:
            return 0.0
        length = self._len[doc_id]
        total = 0.0
        for term in set(tokenize(query)):
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = self._df.get(term, 0)
            # BM25 的 IDF（带 +0.5 平滑），与 Lucene 口径一致
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
            total += idf * freq * (self.k1 + 1) / denom
        return total

    def __len__(self) -> int:
        return self._n


class MemoryVectorStore:
    """进程内向量库：暴力余弦 + 真 BM25。

    ★ 稠密检索走 numpy 矩阵乘法，不是纯 Python 循环。
      差距是数量级的：10 万条 1024 维向量，纯 Python 点积约 3~5 秒/次查询，
      numpy 的 (N,d)@(d,) 只要 10~30ms。没有这条快路径，
      "内存后端"就只是个玩具，没法在 AutoDL 这种没有 Docker 的环境里顶替 Milvus。
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._records: dict[int, VectorRecord] = {}
        self._bm25 = _BM25()
        # 稠密矩阵缓存：写入时置脏，检索时懒重建。
        # 批量入库时不能每条都重建矩阵，否则 N 次插入是 O(N²)。
        self._matrix = None            # np.ndarray (N, dim)
        self._matrix_ids: list[int] = []
        self._dirty = True

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, records: list[VectorRecord]) -> int:
        for record in records:
            # 幂等：同 chunk_id 直接覆盖
            self._records[record.chunk_id] = record
            self._bm25.remove(record.chunk_id)
            if record.is_active:
                self._bm25.add(record.chunk_id, record.content)
        self._dirty = True
        return len(records)

    def _rebuild_matrix(self) -> None:
        import numpy as np

        ids = [cid for cid, rec in self._records.items() if rec.is_active]
        if not ids:
            self._matrix, self._matrix_ids = None, []
        else:
            # 行顺序与 _matrix_ids 严格对应 —— 两个列表必须一起更新
            self._matrix = np.asarray([self._records[c].dense for c in ids], dtype=np.float32)
            self._matrix_ids = ids
        self._dirty = False

    async def delete_by_document(self, document_id: int, tenant_id: int = 1) -> int:
        victims = [
            cid for cid, rec in self._records.items()
            if rec.document_id == document_id and rec.tenant_id == tenant_id
        ]
        for cid in victims:
            self._records.pop(cid, None)
            self._bm25.remove(cid)
        self._dirty = True
        return len(victims)

    async def search_dense(
        self, vector: list[float], *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]:
        import numpy as np

        if self._dirty:
            self._rebuild_matrix()
        if self._matrix is None:
            return []

        # 向量已 L2 归一化，内积即余弦
        sims = self._matrix @ np.asarray(vector, dtype=np.float32)

        # 租户过滤在打分之后做：向量库是单租户部署时，为每条查询扫一遍
        # tenant_id 的 Python 判断，比矩阵乘法还慢。
        candidates = [
            (cid, float(sims[i]))
            for i, cid in enumerate(self._matrix_ids)
            if self._records[cid].tenant_id == tenant_id
        ]
        # ★ 显式 tie-break：不用 dict 迭代顺序，保证跨进程结果一致
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        return [
            VectorHit(chunk_id=cid, score=score, rank=i, dense_score=score,
                      document_id=self._records[cid].document_id,
                      content=self._records[cid].content)
            for i, (cid, score) in enumerate(candidates[:top_k], start=1)
        ]

    async def search_sparse(
        self, query_text: str, *, top_k: int, tenant_id: int = 1
    ) -> list[VectorHit]:
        scored: list[tuple[int, float]] = []
        for cid, rec in self._records.items():
            if rec.tenant_id != tenant_id or not rec.is_active:
                continue
            score = self._bm25.score(query_text, cid)
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [
            VectorHit(chunk_id=cid, score=score, rank=i, sparse_score=score,
                      document_id=self._records[cid].document_id,
                      content=self._records[cid].content)
            for i, (cid, score) in enumerate(scored[:top_k], start=1)
        ]

    async def count(self, tenant_id: int = 1) -> int:
        return sum(1 for r in self._records.values() if r.tenant_id == tenant_id)

    async def aclose(self) -> None:
        self._records.clear()

    # 测试辅助
    def all_chunk_ids(self) -> list[int]:
        return sorted(self._records)


def _dot(a: list[float], b: list[float]) -> float:
    """向量已 L2 归一化，内积即余弦。"""
    return sum(x * y for x, y in zip(a, b, strict=False))
