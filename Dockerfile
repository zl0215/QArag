# syntax=docker/dockerfile:1.7
# ============================================================================
# RAG-Agent 应用镜像（api + worker 共用）
#
# 三种构建形态，用 build args 切换：
#
#   ① 纯 API 模式（镜像 ~450MB，不需要 GPU、不装 torch）
#      docker build -t rag-agent .
#      → EMBED_PROVIDER=api，把 BGE 交给独立推理服务（见 compose 的 models profile）
#
#   ② CPU + 本地模型（镜像 ~2.2GB，torch 是 CPU wheel）
#      docker build --build-arg WITH_LOCAL=true -t rag-agent:local .
#
#   ③ GPU + 本地模型（宿主机需有 NVIDIA 驱动 + nvidia-container-toolkit）
#      docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
#                   --build-arg WITH_LOCAL=true -t rag-agent:gpu .
#
# ★ 模型权重一律不打进镜像，运行时用 volume 挂载（见 compose 的 MODEL_DIR）。
#   权重 ~1.3GB，打进镜像会让每次改代码都重传一遍，而且换模型不该触发镜像重建。
# ============================================================================

ARG BASE_IMAGE=python:3.12-slim

# ---------------------------------------------------------------------------
# Stage 1: builder —— 只装依赖，不拷源码
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ARG WITH_LOCAL=false
# torch wheel 源。★ 必须在这里注入，不能写进 pyproject.toml ——
# 写进 [tool.uv.sources] 会让 uv 每次解析都去访问 download.pytorch.org，
# 国内网络下即使根本不安装 torch 也会卡住十几分钟。
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY --from=ghcr.io/astral-sh/uv:0.11.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# ★ 先只拷依赖清单。依赖没变时这一层走缓存 —— 改代码不触发重装依赖。
COPY pyproject.toml uv.lock* ./

# 两套安装路径，差异只在于"基础镜像里有没有 torch"：
#
#   uv sync        —— 按 lock 从零构建 venv，不参考系统包。
#                     用于 python:slim（系统里没有 torch）。
#   uv pip install —— 在已有环境里解析，已满足的依赖不重装。
#                     用于 pytorch/pytorch 基础镜像：系统里已有 CUDA 版 torch，
#                     再 uv sync 一遍不仅要下 2.5GB，还可能把 CUDA 版换成 CPU 版，
#                     症状是"容器起来了但 torch.cuda.is_available() 是 False"。
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    if [ "$WITH_LOCAL" = "true" ]; then \
        if python -c "import torch" 2>/dev/null; then \
            echo ">>> 基础镜像已带 torch，只补装 sentence-transformers"; \
            uv venv --system-site-packages /app/.venv; \
            VIRTUAL_ENV=/app/.venv uv pip install \
                "sentence-transformers>=3" "transformers>=4.40"; \
        else \
            echo ">>> 从 $TORCH_INDEX_URL 安装 torch"; \
            UV_EXTRA_INDEX_URL="$TORCH_INDEX_URL" uv sync --no-install-project --no-dev --extra local; \
        fi; \
    else \
        echo ">>> 纯 API 模式，不安装本地模型栈"; \
        uv sync --no-install-project --no-dev; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    VIRTUAL_ENV=/app/.venv \
    # ★ 离线加载：模型从挂载目录读，绝不在启动时联网。
    #   不加这两行，sentence-transformers 会去连 huggingface.co，
    #   国内网络下表现为"启动卡住两分钟然后超时"，而不是一个明确的报错。
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # tokenizers 在多进程/fork 场景会打警告并可能死锁
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts

# ★ 非 root 运行。uid 固定 10001，避免不同环境下的权限漂移。
RUN useradd -m -u 10001 app \
    && mkdir -p /app/data/uploads \
    && chown -R app:app /app
USER app

EXPOSE 8000

# ★ 这里**不写** HEALTHCHECK：api 和 worker 共用这个镜像，
#   worker 没有 HTTP 端口，塞一个 HTTP 健康检查会让它永远 unhealthy，
#   进而让 depends_on 的 service_healthy 卡死整条启动链。
#   健康检查放在 compose 里按服务分别定义。

CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
