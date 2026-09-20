"""解析路由。

★ 关键决策：按 magic bytes 判断真实类型，绝不信任客户端传来的 Content-Type 或扩展名。
一个 .pdf 后缀的文件可以是任何东西。
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from rag.core.errors import UnsupportedMediaError
from rag.core.logging import get_logger
from rag.schemas.document import DocMeta, NodeType, ParsedDocument, compute_text_hash

logger = get_logger(__name__)

# Postgres 的 text 列**不接受 0x00**，而 PDF 抽出的文本里经常带它 ——
# 字符映射表错位、嵌入字体缺 ToUnicode CMap 都会产生。症状是整篇文档摄取失败：
#
#   asyncpg.exceptions.CharacterNotInRepertoireError:
#     invalid byte sequence for encoding "UTF8": 0x00
#
# 其余 C0 控制字符（\x01-\x08、\x0b、\x0c、\x0e-\x1f）一并清掉：它们不是合法
# 正文，会污染 BM25 分词，在界面上也显示成方块。\t \n \r 保留 —— 它们是排版信息。
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """清掉会让入库失败或污染正文的控制字符。"""
    if not text:
        return text
    return _CONTROL_CHARS.sub("", text)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC = "application/msword"          # 遗留 OLE2 格式，一期不支持
MARKDOWN = "text/markdown"
PLAIN_TEXT = "text/plain"

SUPPORTED_MIME = {PDF, DOCX, MARKDOWN, PLAIN_TEXT}

# 扩展名 → 期望的 mime（仅用于无 magic 的纯文本类）
_TEXT_SUFFIXES = {".md": MARKDOWN, ".markdown": MARKDOWN, ".txt": PLAIN_TEXT}

_MAGIC_PDF = b"%PDF-"
_MAGIC_ZIP = b"PK\x03\x04"
_MAGIC_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"   # .doc / .xls / .ppt 共用


def _is_docx_zip(path: Path) -> bool:
    """PK 头是 zip 家族共用的，必须进一步确认里面是 Word 文档。

    否则 .xlsx / .pptx 甚至 zip 炸弹都会被当成 docx 放进来。
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False
    return "word/document.xml" in names


def sniff_mime(path: Path, file_name: str = "") -> str:
    """返回真实 MIME；无法识别时抛 UnsupportedMediaError。"""
    with path.open("rb") as fh:
        head = fh.read(8)

    if head.startswith(_MAGIC_PDF):
        return PDF
    if head.startswith(_MAGIC_OLE2):
        raise UnsupportedMediaError(
            "检测到旧版 .doc 格式。请用 LibreOffice 转换为 .docx："
            "soffice --headless --convert-to docx <file>"
        )
    if head.startswith(_MAGIC_ZIP):
        if _is_docx_zip(path):
            return DOCX
        raise UnsupportedMediaError("该压缩包不是 Word 文档（需要 word/document.xml）")

    suffix = Path(file_name or path.name).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        # 纯文本无 magic，做一次 UTF-8 解码试读
        try:
            with path.open("rb") as fh:
                fh.read(4096).decode("utf-8")
            return _TEXT_SUFFIXES[suffix]
        except UnicodeDecodeError as exc:
            raise UnsupportedMediaError("文本文件不是有效的 UTF-8 编码") from exc

    raise UnsupportedMediaError(f"不支持的文件类型：{suffix or '未知'}")


def _looks_chinese(text: str) -> float:
    """返回 CJK 字符占比，用于语言标注。"""
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / len(text)


def detect_lang(text: str) -> str:
    ratio = _looks_chinese(text[:4000])
    if ratio > 0.30:
        return "zh"
    if ratio < 0.05:
        return "en"
    return "mixed"


def finalize(
    meta: DocMeta,
    nodes: list,
    *,
    parser_name: str,
) -> ParsedDocument:
    """补齐 meta 的派生字段（L2 哈希、语言、页数）。

    ★ 这里是**所有解析器的唯一收口**，所以控制字符也在这里清：
      放在各个 parser 里就要改四份，且新增 parser 必然漏掉 ——
      而漏掉的代价是整篇文档摄取失败（见 sanitize_text 的说明）。
      先清再算哈希，`sha256_text` 才是对"真正入库的正文"取的。
    """
    for node in nodes:
        node.text = sanitize_text(node.text)
        # heading_path 会进 section_path、进而进引用标签，同样要清
        node.heading_path = sanitize_text(node.heading_path)
    if meta.title:
        meta.title = sanitize_text(meta.title)
    if meta.file_name:
        meta.file_name = sanitize_text(meta.file_name)

    # furniture 不进正文，也不参与哈希
    content_nodes = [n for n in nodes if n.type is not NodeType.FOOTER]
    full_text = "\n\n".join(n.text for n in content_nodes if n.text)

    meta.sha256_text = compute_text_hash(full_text)
    meta.parser = parser_name
    meta.lang = detect_lang(full_text)
    if not meta.page_count and content_nodes:
        meta.page_count = max((n.page_end for n in content_nodes), default=0)
    if not meta.title:
        meta.title = _infer_title(content_nodes) or meta.file_name

    return ParsedDocument(meta=meta, nodes=nodes)


def _infer_title(nodes: list) -> str:
    for node in nodes:
        if node.type is NodeType.TITLE and node.level == 1 and node.text.strip():
            return node.text.strip()[:200]
    for node in nodes:
        if node.text.strip():
            return node.text.strip()[:200]
    return ""


def parse_document(
    path: Path,
    *,
    doc_id: str,
    file_name: str | None = None,
    sha256: str | None = None,
) -> ParsedDocument:
    """解析入口。

    `sha256` 可由调用方传入 —— 摄取管道为了做幂等判断已经算过一次了，
    没必要对同一份文件读两遍。
    """
    file_name = file_name or path.name
    mime = sniff_mime(path, file_name)

    # ★ 流式算哈希，不 read_bytes()：一份 200MB 的 PDF 读进内存再算哈希，
    #   在容器的内存限额下会直接 OOM。解析器自己按需读文件即可。
    meta = DocMeta(
        doc_id=doc_id,
        file_name=file_name,
        mime=mime,
        sha256_bytes=sha256 or _sha256_file(path),
        size_bytes=path.stat().st_size,
        parser_cfg_hash="",
    )

    if mime == PDF:
        from rag.parsers.pdf import parse_pdf

        return parse_pdf(path, meta)
    if mime == DOCX:
        from rag.parsers.docx import parse_docx

        return parse_docx(path, meta)
    if mime in (MARKDOWN, PLAIN_TEXT):
        from rag.parsers.markdown import parse_markdown

        return parse_markdown(path, meta, plain=(mime == PLAIN_TEXT))

    raise UnsupportedMediaError(f"没有可用的解析器：{mime}")
