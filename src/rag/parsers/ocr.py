"""Local OCR for scanned PDF pages; coordinates are mapped back to PDF points."""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag.core.errors import NeedsOCRError
from rag.core.logging import get_logger

logger = get_logger(__name__)
_ENGINE_LOCK = threading.Lock()
_RENDER_LOCK = threading.Lock()
_CID = re.compile(r"\(cid:\d+\)|\ufffd")


@dataclass(frozen=True)
class OCRConfig:
    mode: str = "auto"
    model_dir: Path = Path("./models/ocr")
    dpi: int = 200
    min_text_chars: int = 40
    min_confidence: float = 0.6
    max_rejected_ratio: float = 0.3
    max_pages: int = 100
    max_pixels: int = 12_000_000
    threads: int = 2

    @classmethod
    def from_settings(cls, settings) -> OCRConfig:
        return cls(**{
            name: getattr(settings, f"pdf_ocr_{name}")
            for name in cls.__dataclass_fields__
        })


def needs_ocr(page, lines: list[dict], config: OCRConfig) -> bool:
    if config.mode == "force":
        return True
    text = "".join(line.get("text", "") for line in lines)
    clean = _CID.sub("", text)
    useful = sum(ch.isalnum() for ch in clean)
    corrupted = len(_CID.findall(text)) > max(2, len(text) // 10)
    images = page.images
    # Native blank pages should not be sent through OCR.
    has_graphics = bool(images or page.curves or page.rects)
    area = max(float(page.width) * float(page.height), 1)
    image_area = sum(
        max(0, float(img.get("x1", 0)) - float(img.get("x0", 0)))
        * max(0, float(img.get("bottom", 0)) - float(img.get("top", 0)))
        for img in images
    )
    return (corrupted or (useful < config.min_text_chars and has_graphics)
            or (image_area / area > .55 and useful < config.min_text_chars * 4))


def _model_paths(model_dir: Path) -> dict[str, Path]:
    names = {"Det": "det.onnx", "Cls": "cls.onnx", "Rec": "rec.onnx"}
    paths = {kind: model_dir / filename for kind, filename in names.items()}
    if not all(path.is_file() for path in paths.values()):
        raise NeedsOCRError(
            "扫描页需要 OCR 模型。请安装 ocr 扩展并运行 python scripts/setup_ocr.py。",
            reason="ocr_models_missing",
        )
    return paths


@lru_cache(maxsize=2)
def _engine(model_dir: str, threads: int):
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise NeedsOCRError("缺少 OCR 依赖，请执行 uv sync --extra ocr。",
                            reason="ocr_dependency_missing") from exc
    paths = _model_paths(Path(model_dir))
    return RapidOCR(params={
        **{f"{kind}.model_path": str(path.resolve()) for kind, path in paths.items()},
        "Global.text_score": 0.0,
        "Global.log_level": "warning",
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def order_ocr_lines(lines: list[dict], width: float) -> tuple[list[dict], float | None]:
    """Use a stable central gutter, keeping full-width headings outside column detection."""
    split = width / 2
    margin = width * .025
    left = [line for line in lines if line["x1"] < split - margin]
    right = [line for line in lines if line["x0"] > split + margin]
    crossing = [line for line in lines if line not in left and line not in right]
    # Aligned rows on both sides with an empty central gutter indicate two columns.
    paired = sum(any(abs(a["top"] - b["top"]) < max(8, a["bottom"] - a["top"])
                     for b in right) for a in left)
    two_columns = len(left) >= 4 and len(right) >= 4 and paired >= 3
    if not two_columns:
        return sorted(lines, key=lambda line: (line["top"], line["x0"])), None
    # Wide headings separate sections into bands so a heading cannot land after its body.
    ordered: list[dict] = []
    remaining = left + right
    for heading in sorted(crossing, key=lambda line: line["top"]):
        band = [line for line in remaining if line["top"] < heading["top"]]
        for col, subset in enumerate((left, right)):
            for line in sorted((item for item in band if item in subset), key=lambda item: item["top"]):
                line["column"] = col
                ordered.append(line)
        remaining = [line for line in remaining if line not in band]
        ordered.append(heading)
    for col, subset in enumerate((left, right)):
        for line in sorted((item for item in remaining if item in subset), key=lambda item: item["top"]):
            line["column"] = col
            ordered.append(line)
    # Parser uses (column, top) sorting. Preserve this band-aware order as an explicit key.
    for index, line in enumerate(ordered):
        line["ocr_order"] = index
    return ordered, split


def recognize_page(page, config: OCRConfig) -> tuple[list[dict], float | None, dict]:
    """Serialize OCR inference and render one bounded page at a time."""
    from PIL import ImageStat

    width, height = float(page.width), float(page.height)
    if width <= 0 or height <= 0:
        raise NeedsOCRError("页面尺寸无效，需要检查 PDF。", reason="ocr_invalid_page")
    dpi = min(config.dpi, max(1, int(72 * math.sqrt(config.max_pixels / (width * height)))))
    if math.ceil(width * dpi / 72) * math.ceil(height * dpi / 72) > config.max_pixels:
        raise NeedsOCRError("页面尺寸超过 OCR 像素上限，请缩小页面。",
                            reason="ocr_pixel_limit")
    try:
        # PDFium rendering and RapidOCR engine state are both serialized per process.
        with _RENDER_LOCK:
            image = page.to_image(resolution=dpi, antialias=True).original.convert("RGB")
    except Exception as exc:
        raise NeedsOCRError("扫描页渲染失败，请检查 PDF。", reason="ocr_render_failed") from exc
    stat = ImageStat.Stat(image.convert("L"))
    if stat.mean[0] > 250 and stat.stddev[0] < 2:
        return [], None, {"blank": True, "accepted": 0, "rejected": 0, "mean_confidence": None}
    try:
        with _ENGINE_LOCK:
            engine = _engine(str(config.model_dir.resolve()), config.threads)
            # Passing a PIL image preserves RGB/BGR conversion in RapidOCR's own loader.
            result = engine(image)
    except NeedsOCRError:
        raise
    except Exception as exc:
        logger.warning("ocr.inference_failed", error_type=type(exc).__name__)
        raise NeedsOCRError("OCR 识别失败，请检查模型与扫描页质量。",
                            reason="ocr_inference_failed") from exc
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    boxes = getattr(result, "boxes", None)
    if texts is None or scores is None or boxes is None:
        raise NeedsOCRError("扫描页未识别到有效文字，需要人工检查。",
                            reason="ocr_empty")
    scale_x, scale_y = width / image.width, height / image.height
    lines, rejected = [], 0
    for text, score, box in zip(texts, scores, boxes, strict=True):
        score = float(score)
        if not str(text).strip():
            continue
        if not math.isfinite(score) or score < config.min_confidence:
            rejected += 1
            continue
        xs, ys = [float(p[0]) * scale_x for p in box], [float(p[1]) * scale_y for p in box]
        size = max(1., max(ys) - min(ys))
        lines.append({
            "text": str(text).strip(), "top": min(ys), "bottom": max(ys),
            "x0": min(xs), "x1": max(xs), "column": 0,
            "chars": [{"size": size}], "ocr_confidence": score,
        })
    if not lines or rejected / (len(lines) + rejected) > config.max_rejected_ratio:
        raise NeedsOCRError("OCR 低置信文字过多，需要更清晰扫描件或人工复核。",
                            reason="ocr_low_confidence")
    ordered, split = order_ocr_lines(lines, width)
    stats = {"blank": False, "accepted": len(lines), "rejected": rejected,
             "mean_confidence": round(sum(line["ocr_confidence"] for line in lines) / len(lines), 4)}
    logger.info("ocr.page_done", accepted=len(lines), rejected=rejected, dpi=dpi)
    return ordered, split, stats
