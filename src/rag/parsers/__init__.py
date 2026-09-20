"""文档解析层：PDF / DOCX / Markdown → 统一文档树。"""

from rag.parsers.base import SUPPORTED_MIME, parse_document, sniff_mime

__all__ = ["SUPPORTED_MIME", "parse_document", "sniff_mime"]
