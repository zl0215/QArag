"""Copy bundled RapidOCR models to the configured read-only runtime model directory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from rag.core.config import get_settings

MODELS = {
    "det.onnx": "PP-OCRv6_det_small.onnx",
    "cls.onnx": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec.onnx": "PP-OCRv6_rec_small.onnx",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_models(directory: Path) -> dict:
    try:
        import rapidocr
    except ImportError as exc:
        raise SystemExit("Install OCR first: uv sync --extra ocr") from exc
    source = Path(rapidocr.__file__).parent / "models"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"rapidocr": importlib.metadata.version("rapidocr"), "models": {}}
    for destination, filename in MODELS.items():
        origin = source / filename
        if not origin.is_file():
            raise SystemExit(f"Bundled model missing: {filename}. Reinstall rapidocr==3.10.0.")
        target = directory / destination
        digest = sha256(origin)
        if target.exists() and sha256(target) != digest:
            raise SystemExit(f"Existing model differs: {target}. Select a new --model-dir.")
        if not target.exists():
            shutil.copyfile(origin, target)
        manifest["models"][destination] = digest
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=get_settings().pdf_ocr_model_dir)
    args = parser.parse_args()
    print(json.dumps(prepare_models(args.model_dir), indent=2))
