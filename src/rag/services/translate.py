"""跨语言查询扩展：把问题翻成语料语言，再拿去召回。

★ 为什么需要这一步（实测，不是推测）：
  本项目语料是 9 篇**英文**论文，而用户用**中文**提问。两个通道各自的下场：

    稀疏（BM25）  R@50 = 0.175(中文查询) vs 0.650(同一批问题的英文版)
    稠密（bge-zh）R@50 = 0.675(中文)       vs 0.825(英文)

  稀疏通道的崩掉是**结构性**的，不是配置问题：BM25 是词法匹配，
  中文词条永远命中不了英文文档里的词条 —— 实测中文提问时它**一条都召不回**。
  稠密通道是模型问题：`bge-large-zh-v1.5` 是为中文语义空间训练的，
  中文 query 对英文 doc 的对齐能力远弱于它的中文-中文表现。

  ★ 注意这里**不是把原问题换掉**，而是**再加一路**：原问题和译文各自能召回
    对方漏掉的块，两路一起进 RRF，取并集。替换会丢掉原问题里的语感信息，
    并集不会 —— 代价只是多一次稠密前向（几毫秒）。

★ 为什么翻译放在检索侧而不是生成侧：
  生成侧的语言由用户决定（中文问就中文答），检索侧的语言由**语料**决定。
  把两者解耦，用户不需要知道自己的知识库是什么语言的。

★ 翻译用便宜模型：这是个短输入短输出的改写任务，不需要强模型。
"""

from __future__ import annotations

import re

from rag.core.logging import get_logger
from rag.providers.base import is_enabled

logger = get_logger(__name__)

# CJK 统一表意文字 + 日文假名 + 韩文谚文。只要出现就认为"查询语言不是英文"。
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

_PROMPT = """You are a query translator for an academic paper search engine.
Translate the user's question into English.

Rules:
- Output ONLY the translated question. No quotes, no explanation, no preamble.
- Keep technical terms, model names and abbreviations as-is
  (e.g. SVM, FSVM, GBSVM, BM25, RRF, DBSCAN, CKA).
- Keep the question's intent and specificity. Do not broaden or narrow it.
- Use wording likely to occur in source documents, not conversational paraphrases.
  For questions about a parameter's value or choice, prefer forms such as
  "set to", "defined as", or "chosen as" when they preserve the intent.
- If the input is already English, output it unchanged.

Question: {query}
English:"""


def needs_translation(query: str, *, target: str = "en") -> bool:
    """查询语言和语料语言是否不一致。

    判据故意做得极简：目标语言是英文、而查询里出现了 CJK 字符。
    ★ 不引入语言检测库：这个判断的**代价是不对称的** ——
      多翻一次只是多几毫秒，而漏翻一次是整条词法通道归零。
      所以宁可错翻，不可漏翻。
    """
    if target.lower().startswith("en"):
        return bool(_CJK.search(query or ""))
    return False


class QueryTranslator:
    """把查询翻成目标语言。带进程内缓存。"""

    def __init__(self, llm, *, target: str = "en", model: str = "") -> None:
        self.llm = llm
        self.target = target
        self.model = model
        # ★ 缓存的两个理由：① 评测算 6 组消融 × 40 题，同一个问题会被问 6 次，
        #   不缓存就是 240 次 LLM 调用换 40 次翻译；② 检索路径上的延迟对用户可见。
        #   键里带 target，换目标语言不会读到旧值。
        self._cache: dict[tuple[str, str], str] = {}

    async def translate(self, query: str) -> str | None:
        """返回译文；失败或没必要时返回 None（调用方应退回原查询）。

        ★ 失败必须返回 None 而不是抛异常：翻译只是个增强，
          它挂了不该让整个检索挂掉。
        """
        q = (query or "").strip()
        if not q or not needs_translation(q, target=self.target):
            return None

        key = (self.target, q)
        if key in self._cache:
            return self._cache[key]

        try:
            kwargs = {"model": self.model} if self.model else {}
            raw = await self.llm.acomplete(
                [{"role": "user", "content": _PROMPT.format(query=q)}],
                temperature=0.0,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — 翻译是增强，任何失败都只降级
            logger.warning("translate.failed", error=f"{type(exc).__name__}: {exc}")
            return None

        out = " ".join((raw or "").split()).strip().strip('"').strip("'")
        # 译文必须真的是英文：模型有时会把原句原样吐回来，或输出一段解释。
        # 含 CJK 或过长（>400 字符）的一律丢弃，退回原查询。
        if not out or _CJK.search(out) or len(out) > 400:
            logger.warning("translate.rejected", chars=len(out or ""))
            return None

        self._cache[key] = out
        logger.info("translate.done", src_chars=len(q), dst_chars=len(out))
        return out


def build_translator(llm, *, target: str = "en", model: str = "") -> QueryTranslator | None:
    return QueryTranslator(llm, target=target, model=model) if is_enabled(llm) else None


__all__ = ["QueryTranslator", "build_translator", "needs_translation"]
