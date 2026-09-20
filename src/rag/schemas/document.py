"""统一文档模型（解析层 IR）。

★ 核心决策：解析层产出富文档树，切分层才降维成 chunk。
不要用 LangChain Document 作为解析层 IR —— 它只有 page_content + metadata 两个字段，
表达不了标题层级、表格结构、阅读顺序、页码区间；一旦在这里丢掉结构，后面再也补不回来。
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    TITLE = "title"
    METADATA = "metadata"  # PDF 题名、作者/单位、期刊/会议、DOI 等书目信息
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    CAPTION = "caption"
    FORMULA = "formula"
    REFERENCE = "reference"  # 参考文献列表；默认检索时低于正文，但仍可回答书目信息问题
    FOOTER = "footer"      # 页眉页脚等 furniture，切分前丢弃
    PAGE_BREAK = "page_break"


class DocMeta(BaseModel):
    doc_id: str
    file_name: str
    mime: str                                  # magic bytes 嗅探结果，不信扩展名
    sha256_bytes: str                          # L1 哈希：原始字节
    sha256_text: str = ""                      # L2 哈希：归一化抽取文本
    size_bytes: int = 0
    page_count: int = 0
    lang: str = "unknown"
    parser: str = ""                           # 如 "pdfplumber:0.11.5"
    parser_cfg_hash: str = ""
    title: str = ""
    parse_warnings: list[str] = Field(default_factory=list)


class DocNode(BaseModel):
    """文档树节点。"""

    node_id: str
    doc_id: str
    parent_id: str | None = None
    type: NodeType = NodeType.PARAGRAPH
    level: int | None = None                   # title 的层级 1..6
    reading_order: int = 0
    text: str = ""
    page_start: int = 0
    page_end: int = 0
    heading_path: str = ""                     # "3 系统设计 > 3.2 存储层"
    lang: str | None = None                    # 节点级语言，中英混排文档必备
    # ★ 引用高亮与"跳转到原文"全靠这三个，后期无法补救
    char_start: int = 0
    char_end: int = 0
    bbox: tuple[float, float, float, float] | None = None
    # 表格
    table_cells: list[list[str]] | None = None
    table_markdown: str | None = None
    # 代码
    code_lang: str | None = None
    # 图片
    image_ref: str | None = None

    @property
    def is_structural(self) -> bool:
        """标题、表格、代码块在切分时享有原子性优先级。"""
        return self.type in (NodeType.TITLE, NodeType.TABLE, NodeType.CODE)


class ParsedDocument(BaseModel):
    meta: DocMeta
    nodes: list[DocNode] = Field(default_factory=list)

    def full_text(self) -> str:
        """按阅读顺序拼接全文 —— L2 哈希的输入。"""
        return "\n\n".join(n.text for n in self.nodes if n.text)


def make_node_id(doc_id: str, reading_order: int) -> str:
    return hashlib.sha1(f"{doc_id}:{reading_order}".encode()).hexdigest()[:16]


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_text_hash(text: str) -> str:
    """归一化后再哈希：空白折叠，避免纯格式变动触发重建。"""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
