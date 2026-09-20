"""Token 计数。

★ 必须用 embedding 模型自己的 tokenizer，不要用 tiktoken。
tiktoken 的 BPE 以英文为主，会把 CJK 切成接近"一字节一 token"：
一个 20 字的句子可能算出 60+ token。后果是
  ① 分块远小于预期，语义碎片化
  ② 上下文预算被高估，实际能塞的内容少 2/3
  ③ 这种退化在英文 MTEB 榜上完全看不出来
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    @property
    def max_tokens(self) -> int | None:
        """模型支持的最大输入长度；未知则 None。"""
        ...


class TransformersTokenCounter:
    """基于 HuggingFace tokenizer 的精确计数。"""

    def __init__(self, tokenizer, max_tokens: int | None = None) -> None:  # noqa: ANN001
        self._tok = tokenizer
        self._max = max_tokens

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> TransformersTokenCounter:
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在：{model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        return cls(tokenizer, max_tokens=read_max_seq_length(model_dir))

    def count(self, text: str) -> int:
        if not text:
            return 0
        # add_special_tokens=False：分块时我们关心的是净内容长度
        return len(self._tok.encode(text, add_special_tokens=False))

    @property
    def max_tokens(self) -> int | None:
        return self._max


class HeuristicTokenCounter:
    """无模型时的近似计数：CJK 一字≈1 token，其余按 4 字符≈1 token。

    ★ 只用于单元测试和模型不可用的降级路径。生产必须换成 TransformersTokenCounter。
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        self._max = max_tokens

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = sum(1 for ch in text if _is_cjk(ch))
        other = len(text) - cjk
        return cjk + math.ceil(other / 4)

    @property
    def max_tokens(self) -> int | None:
        return self._max


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一表意
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3000 <= code <= 0x303F   # CJK 标点
        or 0xFF00 <= code <= 0xFFEF   # 全角
    )


def read_max_seq_length(model_dir: str | Path) -> int | None:
    """从模型目录读最大序列长度。

    ★ bge-large-zh-v1.5 的 max_position_embeddings 是 512 —— 分块上限必须据此约束，
    写死 1024 会让超出部分的 token 被 tokenizer 静默截断，而截断是无声的。
    """
    model_dir = Path(model_dir)
    for name, key in (
        ("config.json", "max_position_embeddings"),
        ("sentence_bert_config.json", "max_seq_length"),
    ):
        path = model_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("counter.config_unreadable", file=str(path), error=str(exc))
            continue
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None
