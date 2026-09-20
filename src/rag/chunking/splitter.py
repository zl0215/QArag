"""结构感知分块。

三层：
  第一层 结构切分 —— 按文档树切，一个语义元素一个单元（相邻小段落合并）
  第二层 token 约束 —— 超限才切、过小才并
  第三层 父块绑定 —— 同 section 的连续子块共享一个 parent_index

★ 关于 overlap 的决策：块间默认**不重叠**。
   依据：Bennani et al. (2026-03) 在 Natural Questions 上未测出 overlap 的可测量收益，
   只观察到索引成本上升。overlap 仅在**切分单个超长单元**时保留，
   目的是避免句子被腰斩，而不是为了让相邻块共享内容。

★ 表格与代码块享有原子性：绝不参与字符滑窗。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.chunking.counter import TokenCounter
from rag.core.logging import get_logger
from rag.schemas.chunk import Chunk, compute_chunk_hash
from rag.schemas.document import NodeType, ParsedDocument

logger = get_logger(__name__)

# 句末标点：中英文都要有，否则中文会被当成一个永不结束的句子
_SENTENCE_END = "。！？!?；;…\n"
# 从句标点：句子仍然超长时退到这一层
_CLAUSE_END = "，,、）)】》」』：:"

_UNIT_SEPARATOR = "\n\n"


@dataclass
class _Unit:
    """切分的最小语义单元，对应一个 DocNode。"""

    text: str
    node_type: NodeType
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0
    char_end: int = 0
    atomic: bool = False
    table_rows: list[list[str]] | None = None
    code_lang: str | None = None
    tokens: int = 0
    extra: dict = field(default_factory=dict)


class StructureAwareChunker:
    def __init__(
        self,
        counter: TokenCounter,
        *,
        target_tokens: int = 384,
        max_tokens: int = 512,
        overlap_tokens: int = 48,
        parent_target_tokens: int = 1200,
        chunker_version: str = "v1",
        embed_model_id: str = "unknown",
        embed_dim: int = 1024,
        doc_title: str = "",
        strip_furniture: bool = True,
    ) -> None:
        if target_tokens > max_tokens:
            raise ValueError("target_tokens 不能大于 max_tokens")

        # ★ 如果模型自身长度上限比配置更小，以模型为准。
        #   bge-large-zh-v1.5 上限 512，若配置成 1024 会被 tokenizer 静默截断。
        model_max = counter.max_tokens
        if model_max and max_tokens > model_max:
            logger.warning(
                "chunker.max_tokens_clamped_by_model",
                configured=max_tokens, model_max=model_max,
            )
            max_tokens = model_max
            target_tokens = min(target_tokens, max_tokens)

        self.counter = counter
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_target_tokens = parent_target_tokens
        self.chunker_version = chunker_version
        self.embed_model_id = embed_model_id
        self.embed_dim = embed_dim
        self.doc_title = doc_title
        self.strip_furniture = strip_furniture

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def split(self, parsed: ParsedDocument, *, doc_title: str | None = None) -> list[Chunk]:
        units = self._to_units(parsed)
        if not units:
            return []

        chunks = self._pack(units)
        # ★ 编号必须在 _assign_parents 之前完成 —— parent_index 依赖 chunk_index
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        self._assign_parents(chunks)
        self._finalize(chunks, doc_title=doc_title or self.doc_title or parsed.meta.title)
        logger.info(
            "chunker.done",
            doc_id=parsed.meta.doc_id,
            units=len(units),
            chunks=len(chunks),
            avg_tokens=round(sum(c.token_count for c in chunks) / len(chunks), 1),
        )
        return chunks

    # ------------------------------------------------------------------
    # 第一层：文档树 → 语义单元
    # ------------------------------------------------------------------
    def _to_units(self, parsed: ParsedDocument) -> list[_Unit]:
        units: list[_Unit] = []
        heading_stack: list[tuple[int, str]] = []

        for node in sorted(parsed.nodes, key=lambda n: n.reading_order):
            if self.strip_furniture and node.type is NodeType.FOOTER:
                continue

            if node.type is NodeType.TITLE:
                level = node.level or 1
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, node.text.strip()))
                # 标题本身不产出块：它已通过面包屑进入 embed_text，
                # 单独成块只会制造"只有标题、没有内容"的噪声块。
                continue

            text = (node.text or "").strip()
            if not text:
                continue

            # 节点自带的 heading_path 优先（解析器可能比我们更清楚层级）
            section = node.heading_path or " > ".join(t for _, t in heading_stack)

            units.append(_Unit(
                text=text,
                node_type=node.type,
                section_path=section,
                page_start=node.page_start,
                page_end=node.page_end,
                char_start=node.char_start,
                char_end=node.char_end,
                atomic=node.type in (NodeType.METADATA, NodeType.TABLE, NodeType.CODE),
                table_rows=node.table_cells if node.type is NodeType.TABLE else None,
                code_lang=node.code_lang,
                tokens=self.counter.count(text),
            ))

        return units

    # ------------------------------------------------------------------
    # 第二层：打包
    # ------------------------------------------------------------------
    def _pack(self, units: list[_Unit]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buf: list[_Unit] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if not buf:
                return
            text = _UNIT_SEPARATOR.join(u.text for u in buf).strip()
            if text:
                chunks.append(self._make_chunk(text, buf[0], buf[-1]))
            buf = []
            buf_tokens = 0

        for unit in units:
            # 原子单元：独占一个块（超长则特殊切分）
            if unit.atomic:
                flush()
                for piece in self._split_atomic(unit):
                    chunks.append(self._make_chunk(piece[0], piece[1], piece[2]))
                continue

            if unit.tokens > self.max_tokens:
                flush()
                for part in self._split_text(unit.text):
                    clone = _Unit(**{**unit.__dict__, "text": part,
                                     "tokens": self.counter.count(part)})
                    chunks.append(self._make_chunk(part, clone, clone))
                continue

            # ★ 章节边界必须切断。
            #   不切的话，几个短章节会被合并进同一个块，而块的 section_path
            #   取自 buf[0] —— 于是"常见故障"的正文被标成"环境要求"，
            #   引用定位、页码、面包屑全部指错地方。
            #   这是分块器最容易漏、后果又最隐蔽的一条约束：
            #   检索质量看起来正常，只是引用的出处是错的。
            if buf and buf[-1].section_path != unit.section_path:
                flush()

            if buf and buf_tokens + unit.tokens > self.target_tokens:
                flush()

            buf.append(unit)
            buf_tokens += unit.tokens
            if buf_tokens >= self.target_tokens:
                flush()

        flush()
        return chunks

    def _split_atomic(self, unit: _Unit) -> list[tuple[str, _Unit, _Unit]]:
        """表格 / 代码块：超长时的专用切分，绝不走通用滑窗。"""
        if unit.tokens <= self.max_tokens:
            return [(unit.text, unit, unit)]

        if unit.node_type is NodeType.TABLE and unit.table_rows:
            return self._split_table(unit)
        if unit.node_type is NodeType.CODE:
            return self._split_code(unit)
        # 理论上不该走到这里；退化为普通文本切分
        return [
            (part, _Unit(**{**unit.__dict__, "text": part}), _Unit(**{**unit.__dict__, "text": part}))
            for part in self._split_text(unit.text)
        ]

    def _split_table(self, unit: _Unit) -> list[tuple[str, _Unit, _Unit]]:
        """按行切表，★ 每一片都重复表头 —— 否则后续片段的列语义完全丢失。"""
        from rag.parsers.tables import table_to_markdown

        rows = unit.table_rows or []
        if len(rows) < 2:
            return [(unit.text, unit, unit)]

        header, body = rows[0], rows[1:]
        header_tokens = self.counter.count(table_to_markdown([header, header]))

        out: list[tuple[str, _Unit, _Unit]] = []
        bucket: list[list[str]] = []
        bucket_tokens = header_tokens

        def flush_bucket() -> None:
            nonlocal bucket, bucket_tokens
            if not bucket:
                return
            md = table_to_markdown([header, *bucket])
            out.append((md, unit, unit))
            bucket = []
            bucket_tokens = header_tokens

        for row in body:
            row_tokens = self.counter.count(table_to_markdown([header, row]))
            if row_tokens + bucket_tokens > self.max_tokens and bucket:
                flush_bucket()
            bucket.append(row)
            bucket_tokens += row_tokens
        flush_bucket()

        logger.debug("chunker.table_split", pieces=len(out), rows=len(body))
        return out or [(unit.text, unit, unit)]

    def _split_code(self, unit: _Unit) -> list[tuple[str, _Unit, _Unit]]:
        """按行切代码，★ 每片都补上语言标记，保持 Markdown 围栏闭合。"""
        lines = unit.text.splitlines()
        fence = f"```{unit.code_lang}" if unit.code_lang else "```"
        overhead = self.counter.count(fence) * 2

        out: list[tuple[str, _Unit, _Unit]] = []
        bucket: list[str] = []
        bucket_tokens = overhead

        def flush_bucket() -> None:
            nonlocal bucket, bucket_tokens
            if not bucket:
                return
            out.append((f"{fence}\n" + "\n".join(bucket) + "\n```", unit, unit))
            bucket = []
            bucket_tokens = overhead

        for line in lines:
            line_tokens = self.counter.count(line)
            if bucket and bucket_tokens + line_tokens > self.max_tokens:
                flush_bucket()
            bucket.append(line)
            bucket_tokens += line_tokens
        flush_bucket()

        return out or [(unit.text, unit, unit)]

    def _split_text(self, text: str) -> list[str]:
        """超长文本：句 → 从句 → 硬切，逐级退化。"""
        pieces = self._pack_pieces(_split_keep(text, _SENTENCE_END))
        result: list[str] = []
        for piece in pieces:
            if self.counter.count(piece) <= self.max_tokens:
                result.append(piece)
                continue
            clauses = self._pack_pieces(_split_keep(piece, _CLAUSE_END))
            for clause in clauses:
                if self.counter.count(clause) <= self.max_tokens:
                    result.append(clause)
                else:
                    result.extend(self._hard_split(clause))
        return [p.strip() for p in result if p.strip()]

    def _pack_pieces(self, pieces: list[str]) -> list[str]:
        """把已切好的片段贪心合并到 target_tokens。"""
        out: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = buf + piece
            if buf and self.counter.count(candidate) > self.target_tokens:
                out.append(buf)
                buf = piece
            else:
                buf = candidate
        if buf:
            out.append(buf)
        return out

    def _hard_split(self, text: str) -> list[str]:
        """连从句都超长（例如无标点的长串）—— 二分找最大可容纳的字符数。"""
        out: list[str] = []
        start, n = 0, len(text)
        overlap_chars = self._chars_for_tokens(text, self.overlap_tokens)

        while start < n:
            lo, hi = start + 1, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.counter.count(text[start:mid]) <= self.max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            end = lo
            out.append(text[start:end])
            if end >= n:
                break
            start = max(end - overlap_chars, start + 1)

        return out

    def _chars_for_tokens(self, text: str, tokens: int) -> int:
        if tokens <= 0 or not text:
            return 0
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.counter.count(text[:mid]) <= tokens:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ------------------------------------------------------------------
    # 第三层：父块绑定 + 收尾
    # ------------------------------------------------------------------
    def _assign_parents(self, chunks: list[Chunk]) -> None:
        """同一 section 的连续子块共享 parent_index（= 该组首块的 index）。

        ★ 不额外存父块行：检索时按 parent_index 回表取回整组，
          效果等同 small-to-big，但索引体积减半。
        """
        parent_index = 0
        accumulated = 0
        current_section: str | None = None

        for chunk in chunks:
            if chunk.section_path != current_section or accumulated >= self.parent_target_tokens:
                parent_index = chunk.chunk_index
                current_section = chunk.section_path
                accumulated = 0
            chunk.parent_index = parent_index
            accumulated += chunk.token_count

    def _finalize(self, chunks: list[Chunk], *, doc_title: str = "") -> None:
        for chunk in chunks:
            chunk.content_hash = compute_chunk_hash(
                chunk.content,
                chunker_version=self.chunker_version,
                embed_model_id=self.embed_model_id,
                embed_dim=self.embed_dim,
            )
            # ★ 面包屑只进 embedding 输入，不进返回给用户的正文
            prefix = f"《{doc_title}》" if doc_title else ""
            if chunk.section_path:
                prefix = f"{prefix} > {chunk.section_path}" if prefix else chunk.section_path
            chunk.embed_text = f"{prefix}\n\n{chunk.content}" if prefix else chunk.content

    def _make_chunk(self, text: str, first: _Unit, last: _Unit) -> Chunk:
        return Chunk(
            chunk_index=0,  # 打包完成后统一编号
            content=text,
            content_hash="",
            token_count=self.counter.count(text),
            char_count=len(text),
            page_start=first.page_start,
            page_end=last.page_end,
            char_start=first.char_start,
            char_end=last.char_end,
            section_path=first.section_path,
            node_type=first.node_type,
            lang=None,
        )


def _split_keep(text: str, delimiters: str) -> list[str]:
    """按分隔符切分，分隔符保留在前一片末尾。"""
    out: list[str] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in delimiters:
            piece = text[start:idx + 1]
            if piece.strip():
                out.append(piece)
            start = idx + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return out
