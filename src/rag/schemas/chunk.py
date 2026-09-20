"""分块与检索结果模型。"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from rag.schemas.document import NodeType

# 分隔符 \x1f (Unit Separator) —— 不会出现在正常文本里，避免拼接歧义
_SEP = "\x1f"


def compute_chunk_hash(
    text: str,
    *,
    chunker_version: str,
    embed_model_id: str,
    embed_dim: int,
) -> str:
    """内容哈希。

    ★ 把 chunker_version / embed_model 一起哈希是刻意的：
    换了分块策略或嵌入模型，即使文本完全相同，旧向量也不再可用，必须重算。
    """
    payload = _SEP.join([
        " ".join(text.split()),
        chunker_version,
        embed_model_id,
        str(embed_dim),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """待写入存储的检索单元。"""

    chunk_index: int
    content: str
    content_hash: str
    # ★ 送进 embedding 的文本（含标题面包屑）与返回给用户的正文必须分开存，
    #   否则答案里会出现重复的面包屑。
    embed_text: str = ""
    token_count: int = 0
    char_count: int = 0
    parent_index: int | None = None            # small-to-big：指向父块
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0
    char_end: int = 0
    section_path: str = ""
    node_type: NodeType = NodeType.PARAGRAPH
    lang: str | None = None

    # 写入存储后由数据库分配
    chunk_id: int | None = None
    document_id: int | None = None

    def model_post_init(self, _ctx) -> None:  # noqa: ANN001
        if not self.embed_text:
            self.embed_text = self.content
        if not self.char_count:
            self.char_count = len(self.content)


class Citation(BaseModel):
    """答案里的引用。"""

    chunk_id: int
    quote: str = Field(description="支撑结论的原文片段（逐字摘录）")


# 注意：检索结果的模型定义在 rag.services.retrieval（RetrievedChunk / RetrievalResult）。
# 这里**故意不再定义一份** —— 同一个概念有两份定义，早晚会漂移成两个不同的东西。
