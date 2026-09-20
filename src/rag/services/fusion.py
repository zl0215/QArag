"""RRF（Reciprocal Rank Fusion）融合。

★ 为什么用 RRF 而不是加权归一化分数：

   稠密分（余弦，0~1）和稀疏分（BM25，0~+∞）**量纲不同**。做加权求和就得先归一化，
   而 min-max 归一化对异常值极其敏感 —— 一路召回的分数分布变化会直接改写另一路的权重。
   RRF 只看**名次**，天然免调参、免归一化，是工业界混合检索的默认选择。

★ 公式：score(d) = Σ_r 1 / (k + rank_r(d))，rank 从 1 开始。

   k=60 来自原论文（Cormack et al. 2009）。k 越大，名次差异被压得越平
   （头部文档的优势变小）；k 越小越倚重单路的头部命中。60 是经过大量实践验证的稳健值。

★ 平局处理必须是确定性的：
   `(-score, chunk_id)` 排序。若只按 score 排，Python 的 sorted 是稳定的，
   名次会退化成"取决于输入顺序"，而输入顺序又取决于 Milvus 返回顺序 ——
   在并发/分片场景下这个顺序不保证稳定，评测结果就会飘。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

RRF_K_DEFAULT = 60


@dataclass
class RankedItem:
    """一路召回的结果，按名次排列（rank 从 1 开始）。"""

    chunk_id: int
    rank: int
    score: float = 0.0
    source: str = ""


@dataclass
class FusedItem:
    chunk_id: int
    score: float
    # 每一路各自的名次与原始分，用于解释"为什么这条被召回"（评测与 debug 必备）
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values()) if self.ranks else 10**9


def rrf_fuse(
    runs: dict[str, Iterable[RankedItem]],
    *,
    k: int = RRF_K_DEFAULT,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedItem]:
    """把多路召回融合成一个有序列表。

    Args:
        runs: {"dense": [...], "sparse": [...]}，每路内部按名次升序。
        k: RRF 平滑常数。
        weights: 每路权重（消融实验用）。缺省全部为 1.0。
        top_k: 只返回前 N 条；None 表示全部。
    """
    if k <= 0:
        raise ValueError("RRF 的 k 必须为正数")

    weights = weights or {}
    acc: dict[int, FusedItem] = {}

    for source, items in runs.items():
        weight = weights.get(source, 1.0)
        if weight == 0:
            continue
        for item in items:
            contribution = weight / (k + item.rank)
            fused = acc.get(item.chunk_id)
            if fused is None:
                fused = FusedItem(chunk_id=item.chunk_id, score=0.0)
                acc[item.chunk_id] = fused
            fused.score += contribution
            fused.ranks[source] = item.rank
            fused.scores[source] = item.score
            fused.sources.add(source)

    # ★ 确定性排序：分数四舍五入到 9 位后再比，避免浮点误差导致顺序抖动；
    #   同分时用 chunk_id 升序兜底，保证任何两次运行结果完全一致。
    ordered = sorted(acc.values(), key=lambda x: (-round(x.score, 9), x.chunk_id))
    return ordered[:top_k] if top_k else ordered


def rerank_order(
    fused: list[FusedItem],
    rerank_scores: dict[int, float],
    *,
    top_k: int | None = None,
) -> list[FusedItem]:
    """按 reranker 分数重排。

    ★ 未被 reranker 打分的条目（理论上不该有）保留在末尾，并按原 RRF 顺序排列，
      而不是丢弃 —— 丢弃会让"重排后结果变少"成为一个隐蔽的召回损失。
    """
    scored = [f for f in fused if f.chunk_id in rerank_scores]
    unscored = [f for f in fused if f.chunk_id not in rerank_scores]

    for item in scored:
        # 把重排分记进 scores，便于 API 层与评测把"为什么排第一"讲清楚
        item.scores["rerank"] = rerank_scores[item.chunk_id]
        item.sources.add("rerank")

    scored.sort(key=lambda x: (-round(rerank_scores[x.chunk_id], 9), x.chunk_id))
    result = scored + unscored
    return result[:top_k] if top_k else result


def dedupe_by_content(items: list[FusedItem], contents: dict[int, str]) -> list[FusedItem]:
    """按正文去重，保留名次最高的一条。

    ★ 为什么需要：PDF 里页眉页脚重复、跨页表格续表、同一段落在多个章节被引用，
      会产生内容几乎相同的 chunk 占据多个 top-k 名额，白白挤掉真正多样的证据。
    """
    seen: set[str] = set()
    out: list[FusedItem] = []
    for item in items:
        key = " ".join((contents.get(item.chunk_id) or "").split())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(item)
    return out
