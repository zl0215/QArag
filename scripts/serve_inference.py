"""独立的嵌入 / 重排推理服务（跑在有 GPU 的机器上，比如 AutoDL）。

★ 为什么需要它：`EMBED_PROVIDER=local` 把 BGE-large 加载到 API/worker 进程里，
  上传文档时本机 CPU 被嵌入算力打满几十秒；而且 api 和 worker **各加载一份权重**。
  把模型拆出去之后，本机只发 HTTP，CPU 占用回到噪声水平。

★ 为什么用 sentence-transformers 而不是手写 CLS pooling：
  本地 `LocalEmbeddingProvider` 用的就是 `SentenceTransformer.encode(normalize_embeddings=True)`，
  池化方式（BGE 是 CLS pooling）由模型目录里的 `1_Pooling/config.json` 决定。
  手写一份 pooling 意味着"两边得靠人保证一致"，而这里**必须**一致 ——
  Milvus 里已有的向量是本地模型产出的，池化只要差一点，
  旧向量和新查询就不在同一个空间里了，而且不会报错，只会检索变差。
  用同一个库、同一个权重，一致性是构造出来的，不是测出来的。

★ 线格式必须和 `src/rag/providers/` 里的 `api_style=tei` 分支逐字对齐：
  - `POST /embed`  ← {"inputs": ["...", ...]}          → [[float, ...], ...]（顺序与输入一致）
  - `POST /rerank` ← {"query": "...", "documents": [...]} → [{"index": 0, "score": 0.9}, ...]

用法：
    python serve_inference.py \
        --embed-model  /root/autodl-tmp/models/bge-m3 \
        --rerank-model /root/autodl-tmp/models/bge-reranker-v2-m3 \
        --port 8080

本机 .env：
    EMBED_PROVIDER=api
    EMBED_API_BASE=http://127.0.0.1:8081     # 走 SSH 隧道
    EMBED_API_STYLE=tei
    RERANK_PROVIDER=api
    RERANK_API_BASE=http://127.0.0.1:8081
    RERANK_API_STYLE=tei
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("inference")

# 模型全局状态。用锁串行化推理：GPU 上并发提交反而会互相抢显存和 SM，
# 单请求排队对演示场景足够快（batch 内已经并行了）。
_state: dict[str, Any] = {"embed": None, "rerank": None}
_lock = asyncio.Lock()


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(default_factory=list)


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(default_factory=list)


def _load_embed(path: str, device: str, batch_size: int) -> Any:
    from sentence_transformers import SentenceTransformer

    logger.info("loading embedding model: %s", path)
    t0 = time.monotonic()
    model = SentenceTransformer(path, device=device)
    logger.info("embedding ready in %.1fs (dim=%s, max_seq=%s)",
                time.monotonic() - t0,
                model.get_sentence_embedding_dimension(),
                model.max_seq_length)
    return model


def _load_rerank(path: str, device: str, max_length: int) -> Any:
    from sentence_transformers import CrossEncoder

    logger.info("loading reranker model: %s", path)
    t0 = time.monotonic()
    model = CrossEncoder(path, max_length=max_length, device=device)
    logger.info("reranker ready in %.1fs", time.monotonic() - t0)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    # ★ 启动时就加载，而不是懒加载：这个服务的唯一职责就是推理，
    #   没有"先返回 /healthz 再说"的必要；早失败能立刻看到是哪份权重有问题。
    if args.embed_model:
        _state["embed"] = await asyncio.to_thread(
            _load_embed, args.embed_model, args.device, args.batch_size
        )
    if args.rerank_model:
        _state["rerank"] = await asyncio.to_thread(
            _load_rerank, args.rerank_model, args.device, args.max_length
        )
    if not args.embed_model and not args.rerank_model:
        raise RuntimeError("至少要给 --embed-model 或 --rerank-model 之一")
    yield
    _state.clear()


app = FastAPI(title="RAG inference", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    import torch

    return {
        "embed": _state["embed"] is not None,
        "rerank": _state["rerank"] is not None,
        "embed_model": Path(app.state.args.embed_model).name or None,
        "rerank_model": Path(app.state.args.rerank_model).name or None,
        "device": app.state.args.device,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/embed")
async def embed(req: EmbedRequest) -> list[list[float]]:
    model = _state["embed"]
    if model is None:
        raise HTTPException(503, "embedding model not loaded")
    if not req.inputs:
        return []
    args = app.state.args
    async with _lock:
        vectors = await asyncio.to_thread(
            model.encode,
            req.inputs,
            batch_size=args.batch_size,
            normalize_embeddings=True,     # 与 LocalEmbeddingProvider 一致
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return vectors.tolist()


@app.post("/rerank")
async def rerank(req: RerankRequest) -> list[dict[str, Any]]:
    model = _state["rerank"]
    if model is None:
        raise HTTPException(503, "reranker model not loaded")
    if not req.documents:
        return []
    args = app.state.args
    pairs = [(req.query, doc) for doc in req.documents]
    async with _lock:
        scores = await asyncio.to_thread(
            model.predict,
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
    return [
        {"index": i, "score": float(s)} for i, s in enumerate(scores)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 嵌入/重排推理服务")
    parser.add_argument("--embed-model", default="", help="嵌入模型目录，留空则不提供 /embed")
    parser.add_argument("--rerank-model", default="", help="重排模型目录，留空则不提供 /rerank")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import uvicorn

    app.state.args = args
    # ★ 只绑 127.0.0.1：AutoDL 的安全组不保证拦住端口，而这个服务**没有任何鉴权**。
    #   本机通过 SSH 隧道访问，不需要对外监听。
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
