#!/usr/bin/env bash
# AutoDL 一键环境准备（不用 Docker）。
#
#   bash scripts/setup_autodl.sh
#
# ★ 为什么不能用 Docker：
#   AutoDL 的实例**本身就是一个 Docker 容器**。容器里再跑 Docker 需要嵌套
#   支持（privileged + 宿主 docker.sock），标准实例两者都没有 ——
#   `docker info` 会直接 command not found 或报权限错。
#   所以 Milvus standalone / compose 那一套在这里全部不可用，必须换方案。
#
# ★ 这个脚本做了什么替换：
#   Milvus  → **Milvus Lite**：向量库变成本地一个文件（./data/milvus.db），
#             零网络依赖、不需要 Docker、不需要虚拟机、不需要隧道。
#             （VECTOR_MODE=standalone 可以切回"虚拟机上跑 standalone +
#               SSH 反向隧道"那条路，见下面的分支。）
#   Postgres→ apt 装一个真的 PostgreSQL（本机 127.0.0.1:5432）
#             装不了就退到 REPOSITORY_BACKEND=memory（重启丢数据）
#
# ★ 它**不做**的事：不下载模型。模型走 scp 或 hf-mirror，见 DEPLOY.md 第五节。
#
# ★ 为什么默认换成 Lite（踩过的坑）：
#   standalone 那条路要求 AutoDL 的 sshd 允许端口转发，且隧道必须一直活着。
#   实测两种情况都碰到过：隧道静默失败（ssh -N 成功失败长得一模一样）、
#   实例重启后 SSH 端口被重新分配（55154 → 42064，隧道指向了空气）。
#   而 Lite 把这一整类问题直接消掉 —— 没有网络、没有第二个进程可以死。
#   代价见 docs/DEPLOY.md 的对比表（BM25 是分段局部 IDF、单进程独占文件）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
PG_PASSWORD="${PG_PASSWORD:-rag_dev_pw}"

# 向量库模式：
#   lite       （默认）本地文件，零网络依赖
#   standalone 虚拟机跑 Milvus + SSH 反向隧道（要 AutoDL 允许端口转发）
VECTOR_MODE="${VECTOR_MODE:-lite}"

echo "============================================================"
echo " RAG-Agent AutoDL 环境准备"
echo " 项目目录 : $ROOT"
echo " 模型目录 : $MODEL_ROOT"
echo " 向量库   : $VECTOR_MODE"
echo "============================================================"

# ---------------------------------------------------------------- 0. 前置检查
echo
echo "[0/5] 环境自检"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "      ⚠ 检测到可用的 Docker。那用 compose 更省事："
    echo "        docker compose up -d --build"
    echo "        本脚本仍会继续，但你在走一条更麻烦的路。"
else
    echo "      Docker 不可用 → 走无容器方案（预期结果）"
fi

if grep -qm1 avx2 /proc/cpuinfo; then
    echo "      CPU 支持 AVX2 ✓"
else
    if [ "${VECTOR_MODE:-lite}" = "lite" ]; then
        echo "      ⚠ 不支持 AVX2 —— Milvus Lite 底层是 faiss，理论上要 AVX2。"
        echo "        先往下跑；[5/5] 建集合那步会直接告诉你行不行。"
    else
        echo "      ⚠ AutoDL 这侧不支持 AVX2 —— 不影响（Milvus 不跑在这台机器上），"
        echo "        但要确认**虚拟机**那侧支持，否则 Milvus 会 Illegal instruction 崩溃重启。"
    fi
fi

python -c "import torch; print(f'      PyTorch {torch.__version__} / CUDA {torch.version.cuda} / '
                             f'可用={torch.cuda.is_available()}')" 2>/dev/null \
    || echo "      ⚠ 实例没有预装 torch，uv sync 时会从 PyPI 装 CPU 版"

# ★ 这一步专门抓「无卡模式」。
#   AutoDL 开实例时可以选无卡模式（便宜很多），那种模式**不含 GPU 驱动**，
#   torch.cuda.is_available() 必然是 False。装环境的全过程都能正常跑完，
#   但真正要推理时必须关机 → 切回 GPU 模式重启。
#   不在这里说清楚，人会一直以为是自己把 torch 装坏了，然后反复重装。
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "      ⚠ 看不到 GPU（nvidia-smi 无输出）"
        echo "        → 你多半是在「无卡模式」下开的实例。装环境阶段这完全正常，"
        echo "          但推理前必须关机 → 切回 GPU 模式重启，否则 BGE 会退回 CPU。"
    else
        echo "      ⚠ 有 GPU 但 torch 用不了 —— torch 与驱动的版本不匹配，见 docs/DEPLOY.md"
    fi
fi

# ---------------------------------------------------------------- 1. Python 环境
echo
echo "[1/5] Python 环境"
if ! command -v uv >/dev/null 2>&1; then
    pip install -q uv || pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple uv
fi
echo "      uv $(uv --version)"

# ★ --system-site-packages 是关键：
#   实例自带的 torch 是**编译好 CUDA 的版本**，装在系统 site-packages 里。
#   不加这个参数，uv 会另装一个 CPU 版 torch（~200MB，且 GPU 直接失效）。
#   这和 Dockerfile 里 GPU 路径必须用 uv pip install 而不是 uv sync 是同一个坑。
# ★ 判据是 bin/activate 在不在，**不是 .venv 目录在不在**。
#   从 Windows scp 上来的 .venv 是 Scripts/ 布局：目录在，bin/activate 没有。
#   按"目录在不在"判断会跳过创建，然后在 source 那行报
#   "No such file or directory" —— 看起来像 uv 坏了，其实是一个用不了的
#   Windows venv 挡在那，而且它的 pyvenv.cfg 里还写着 D:\ 的绝对路径。
#   （git clone 不会带 .venv，.gitignore 里有；scp -r / PyCharm Deployment 会。）
if [ ! -f .venv/bin/activate ]; then
    if [ -d .venv ]; then
        echo "      ⚠ .venv 存在但不是可用的 Linux venv（多半是从 Windows 拷上来的）"
        echo "        → 改名到 .venv.bak，确认无用后自行删除"
        rm -rf .venv.bak
        mv .venv .venv.bak
    fi
    # 显式指定用系统 python：实例里可能有多个（conda / 系统自带），
    # 选错了就看不到系统那份 CUDA torch。
    uv venv --python "$(command -v python)" --system-site-packages .venv
fi
if [ ! -f .venv/bin/activate ]; then
    echo "      ✗ uv venv 没能生成 .venv/bin/activate —— 中止，"
    echo "        先手动看一下：uv venv --python \"\$(command -v python)\" --system-site-packages .venv"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ★ 镜像必须显式设，原因有两个：
#
#   ① uv **不读 pip.conf**。它是 Rust 实现，刻意不支持 pip 的那套配置。
#      AutoDL 预配的 pip 清华源对 uv 完全无效 —— 所以"AutoDL 应该很快啊"
#      这个直觉在这里不成立：uv sync 一直在从 pypi.org 直连拉，
#      国内速度可能只有几十 KB/s。这一步跑几十分钟是常态。
#      （同理，脚本上面用 pip 装 uv 时写的 -i 参数，对 uv sync 一点用没有。）
#
#   ② ⚠️ 千万别为了"求稳"加 --frozen。uv 有已知问题：
#      --frozen 会**忽略全部镜像配置**（环境变量、uv.toml、命令行 -i 都不认），
#      直接从 uv.lock 里写死的 files.pythonhosted.org 下载。
#      不加 --frozen 才走镜像 —— 这里是故意不写的。
#
#   为什么慢得这么具体：uv.lock 里写的是 registry = "https://pypi.org/simple"，
#   下载 URL 全是 files.pythonhosted.org。不换源就是直连这些地址。
#
#   ⚠️ 副作用：换源后 uv 可能把 uv.lock 改写成镜像地址。跑完 `git status` 看一下，
#      不想带上就 `git checkout uv.lock`（这只影响锁文件里记的源，不影响功能）。
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 慢链路下 uv 默认 30s 超时会反复重试，每次重试都从头来，越试越慢
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

# ★ 三个参数缺一不可，之前的版本这里写错了：
#   --extra local     —— sentence-transformers / transformers 在这个 extra 里。
#                        不加它，EMBED_PROVIDER=local 会因为 import 不到而失败。
#   --extra lite      —— milvus-lite。★ 必须走 extra，不能事后 `uv pip install`：
#                        uv sync 会把环境同步成和 uv.lock 完全一致，临时装的
#                        多余包会被卸载。表现是"昨天还好好的，今天起不来"。
#   --no-install-package torch —— 唯一可靠的"别动我的 CUDA torch"手段。
#                        光靠 --system-site-packages 是不够的：uv 的解析器
#                        会照常把 torch 写进安装计划，装完 GPU 就没了。
#
# ★ 不加 --no-progress：慢的时候能看见进度条，才知道是卡住了还是在动。
uv sync --extra local --extra dev --extra lite --no-install-package torch

# 装完立刻验证 torch 还是原来那份带 CUDA 的
python -c "import torch; print(f'      torch {torch.__version__} / CUDA {torch.version.cuda} / 可用={torch.cuda.is_available()}')"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || echo "      ⚠ torch.cuda.is_available()=False —— 见 docs/DEPLOY.md 的排查表"

# ★ 这里**故意不写 sys.path.insert(0, "src")**。
#   以前写了，于是即使 rag 根本没被装进 venv，这句也照样打印"导入正常" ——
#   检查通过，而 DEPLOY.md 里给的 `uvicorn rag.main:app` 必然 ModuleNotFoundError。
#   自检必须和真实启动方式走同一条 import 路径，否则它就是在骗人。
python -c "import rag; print(f'      rag 包导入正常 ✓  ({rag.__file__})')" \
    || { echo "      ✗ rag 没被装进 venv —— 检查 pyproject.toml 的 [build-system]/hatch 配置"; exit 1; }

# ---------------------------------------------------------------- 2. PostgreSQL
echo
echo "[2/5] PostgreSQL"
PG_OK=0
if command -v psql >/dev/null 2>&1; then
    PG_OK=1
    echo "      已安装"
else
    echo "      未安装，尝试 apt 安装（AutoDL 实例内是 root）"
    if apt-get update -qq && apt-get install -y -qq postgresql postgresql-contrib; then
        PG_OK=1
    else
        echo "      ⚠ apt 安装失败"
    fi
fi

if [ "$PG_OK" = "1" ]; then
    # 容器里没有 systemd，service 命令也常常不可用 —— 直接调 pg_ctlcluster
    if ! pg_isready -q 2>/dev/null; then
        PG_VER="$(ls /etc/postgresql 2>/dev/null | sort -V | tail -1 || true)"
        if [ -n "$PG_VER" ]; then
            pg_ctlcluster "$PG_VER" main start || true
        fi
    fi
    sleep 2

    if pg_isready -q 2>/dev/null; then
        echo "      PostgreSQL 已启动 ✓"
        # 建库建用户（幂等）。用 su postgres 是因为 PG 的 peer 认证只认同名系统用户。
        su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='rag'\"" \
            | grep -q 1 \
            || su postgres -c "psql -c \"CREATE USER rag WITH PASSWORD '$PG_PASSWORD'\""
        su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='rag'\"" \
            | grep -q 1 \
            || su postgres -c "psql -c \"CREATE DATABASE rag OWNER rag\""
        # pgvector 是可选的（只有走 SQL 侧向量检索才需要），装不上不影响主链路
        su postgres -c "psql -d rag -c 'CREATE EXTENSION IF NOT EXISTS vector'" 2>/dev/null \
            && echo "      pgvector ✓" \
            || echo "      pgvector 不可用（不影响：向量存在虚拟机的 Milvus 里）"
        echo "      CREATE DATABASE rag ✓"
    else
        PG_OK=0
        echo "      ⚠ PostgreSQL 起不来"
    fi
fi

if [ "$PG_OK" = "1" ]; then
    REPO_BACKEND=postgres
else
    REPO_BACKEND=memory
    echo "      → 退到 REPOSITORY_BACKEND=memory（重启丢数据，仅够演示）"
fi

# ---------------------------------------------------------------- 3. 目录与模型
echo
echo "[3/5] 数据目录与模型"
mkdir -p data/uploads data/vectors data/samples models

# ★ 模型放 autodl-tmp 而不是 /root：
#   /root 在系统盘上，AutoDL 的系统盘通常只有 30GB 且扩容要钱；
#   autodl-tmp 是数据盘，大得多。用软链接接进来，代码里路径不用改。
if [ ! -e models/bge-large-zh-v1.5 ] && [ -d "$MODEL_ROOT/bge-large-zh-v1.5" ]; then
    ln -sfn "$MODEL_ROOT/bge-large-zh-v1.5" models/bge-large-zh-v1.5
    echo "      已链接 embedding 模型"
fi
if [ ! -e models/bge-reranker-v2-m3 ] && [ -d "$MODEL_ROOT/bge-reranker-v2-m3" ]; then
    ln -sfn "$MODEL_ROOT/bge-reranker-v2-m3" models/bge-reranker-v2-m3
    echo "      已链接 reranker 模型"
fi

for m in bge-large-zh-v1.5 bge-reranker-v2-m3; do
    if [ -f "models/$m/config.json" ]; then
        echo "      $m ✓"
    else
        echo "      ✗ $m 缺失 —— 见下面提示"
    fi
done

# ---------------------------------------------------------------- 4. .env
echo
echo "[4/5] 生成 .env"
# ★ 托管块：脚本只负责这一段，段外的内容（比如你填的 LLM_API_KEY）永不改动。
#
#   为什么需要它 —— 之前这里是 `if [ -f .env ]` 就整个跳过，
#   于是"先 cp .env.example .env 再跑脚本"这个顺序会**静默失效**：
#   脚本什么都不写，而 .env.example 里的默认值是
#   VECTOR_BACKEND=memory —— 向量库变内存，重启即丢，
#   而且 worker 的 preflight 会拒绝启动。症状是"跑完了脚本，看起来都成功，
#   但数据存不住"。（AutoDL 上实际踩到的就是这个。）
#
#   加了哨兵之后：重复执行安全（先删旧块再写新块），
#   且对已存在的 .env 也能生效。
BEGIN_MARK="# >>> setup_autodl.sh managed block —— 本段由脚本生成，手改会在下次执行时丢失 >>>"
END_MARK="# <<< setup_autodl.sh managed block <<<"

strip_managed_block() {
    [ -f .env ] || return 0
    awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
        $0 == b { skip = 1; next }
        $0 == e { skip = 0; next }
        !skip   { print }
    ' .env > .env.tmp && mv .env.tmp .env
}

if [ -f .env ]; then
    echo "      .env 已存在：保留你的改动，只刷新脚本托管的那一段"
else
    cp .env.example .env
fi

# 执行到这里 .env 一定存在（上面两个分支都保证了）
strip_managed_block

# ★ 这个 heredoc 用不带引号的 <<EOF（不是 <<'EOF'）：
#   下面要展开 $REPO_BACKEND 和 $PG_PASSWORD。
#   代价是里面的 $ 和反引号都会被 shell 解释 —— 所以正文里
#   **不能出现反引号**（会被当命令执行）。
if [ "$VECTOR_MODE" = "lite" ]; then
    cat >> .env <<EOF
$BEGIN_MARK
APP_ENV=prod

# ---- 向量库：Milvus Lite（本地文件）----
# ★ MILVUS_URI 是**目录路径**，不是 http:// 地址。
#   milvus-lite 3.x 把它当 data_dir 用，跑起来会看到 ./data/milvus.db/ 是个目录。
#   所以别写成文件（xxx.db 单文件是老版本的行为），也别删这个目录当清空 ——
#   要清空整个 rmtree 掉。
VECTOR_BACKEND=milvus
MILVUS_URI=./data/milvus.db
# 本地文件没有鉴权，token 必须**留空**。
# 别填 root:xxx —— Lite 会因为"非法连接参数"直接拒连。
MILVUS_TOKEN=
MILVUS_COLLECTION=rag_chunks
# ★★ Lite 只认 standard / jieba，**不认 chinese**（standalone 才认）。
#    填 chinese 的报错是 unknown tokenizer type: 'chinese'，而且是在
#    create_collection 时才炸 —— 不在本地先跑一遍很难定位。
#    ⚠ 分析器建集合时固化，之后不可改：从 chinese 换成 jieba 之后，
#      老集合还在用老分词器，必须删库重建（MILVUS_COLLECTION 改个名也行）。
MILVUS_ANALYZER=jieba

REPOSITORY_BACKEND=$REPO_BACKEND
POSTGRES_HOST=127.0.0.1
POSTGRES_PASSWORD=$PG_PASSWORD

# 模型跑在 API 进程内 —— AutoDL 上跑不了 Docker，没有第二个容器放 TEI
EMBED_PROVIDER=local
EMBED_MODEL_PATH=./models/bge-large-zh-v1.5
EMBED_DEVICE=cuda
EMBED_BATCH_SIZE=16
RERANK_PROVIDER=local
RERANK_MODEL_PATH=./models/bge-reranker-v2-m3

# 本地模型不进 HuggingFace 联网校验，避免每次启动卡在联网超时
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
$END_MARK
EOF
        echo "      已写入 .env（Milvus Lite 模式）"
        echo "      ★ 还需要你自己填 LLM_API_KEY —— 这是唯一必须手改的一项"
    else
        cat >> .env <<EOF
$BEGIN_MARK
APP_ENV=prod

# 向量库在**你的虚拟机上**（VECTOR_MODE=standalone）。
#
# ★ 地址填 127.0.0.1，**不是**虚拟机 IP。原因：AutoDL 在公网机房，到你虚拟机
#   的 NAT 内网地址（192.168.x.x）没有路由 —— 填了也连不上。
#   走 SSH 反向隧道，在**虚拟机**上执行（端口换成控制台上当前那个）：
#
#     ssh -N -R 19530:127.0.0.1:19530 -p <当前SSH端口> root@region-41.seetacloud.com
#
#   ⚠ 实例每次重启，SSH 端口都可能变（见过 55154 → 42064）。隧道也要重建。
VECTOR_BACKEND=milvus
MILVUS_URI=${MILVUS_URI:-http://127.0.0.1:19530}
MILVUS_TOKEN=${MILVUS_TOKEN:-root:REPLACE_WITH_PASSWORD}
MILVUS_COLLECTION=rag_chunks
MILVUS_ANALYZER=chinese

REPOSITORY_BACKEND=$REPO_BACKEND
POSTGRES_HOST=127.0.0.1
POSTGRES_PASSWORD=$PG_PASSWORD

EMBED_PROVIDER=local
EMBED_MODEL_PATH=./models/bge-large-zh-v1.5
EMBED_DEVICE=cuda
EMBED_BATCH_SIZE=16
RERANK_PROVIDER=local
RERANK_MODEL_PATH=./models/bge-reranker-v2-m3

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
$END_MARK
EOF
        echo "      已写入 .env（standalone + SSH 隧道模式）"
        echo "      ★ 必须改 MILVUS_TOKEN / LLM_API_KEY"
    fi

# ★ 这里以前还有一个 fi —— 是给外层的 `if [ -f .env ]` 收尾的。
#   现在外层结构改了（存在就保留、只刷新托管块），那个 fi 不再需要。

# ---------------------------------------------------------------- 5. 向量库自检
echo
echo "[5/5] 向量库自检（$VECTOR_MODE）"
export VECTOR_MODE
# ★ 这里**不只是连一下**，而是真的建一次集合。
#   原因：连得上不代表能用。Lite 上真正会炸的是建集合那一步
#   （分词器名字不认、HNSW/BM25 参数不支持），只有 create_collection
#   跑成功才算这条路通了。分两步查是给自己找麻烦。
MILVUS_CHECK=$(python - <<'PY' 2>&1
import os, sys, traceback
from rag.core.config import get_settings
s = get_settings()
mode = os.environ.get("VECTOR_MODE", "lite")

if mode == "lite":
    # Lite 没有鉴权，token 必须空。填了反而会连不上。
    if s.milvus_token.get_secret_value():
        print("FAIL: Lite 模式下 MILVUS_TOKEN 必须留空（本地文件没有鉴权）")
        raise SystemExit(1)
else:
    # ★ standalone 才需要先探端口。隧道没建的时候 pymilvus 的报错是一长串
    #   gRPC 重试，看不出根因；单独探一次 TCP 才能说清"是隧道没建"。
    if "REPLACE_WITH" in s.milvus_token.get_secret_value():
        print("SKIP: MILVUS_TOKEN 还是占位符 —— 先去虚拟机上改掉 root 默认密码，再填进 .env")
        raise SystemExit(0)
    import socket
    host = s.milvus_uri.split("//")[-1].split(":")[0]
    port = int(s.milvus_uri.rsplit(":", 1)[-1])
    with socket.socket() as sock:
        sock.settimeout(3)
        if sock.connect_ex((host, port)) != 0:
            print(f"FAIL: {host}:{port} 连不上 —— SSH 反向隧道没建起来")
            raise SystemExit(1)

import asyncio
from rag.infra.milvus import MilvusVectorStore

async def main() -> int:
    store = MilvusVectorStore(
        uri=s.milvus_uri,
        token=s.milvus_token.get_secret_value(),
        collection=s.milvus_collection,
        dim=s.embed_dim,
        analyzer=s.milvus_analyzer,
    )
    try:
        # 建集合 = 分词器 / 索引参数 / schema 全过一遍，这才是真验证
        await store.ensure_ready()
        n = await store.count()
        print(f"OK: {s.milvus_uri} 可用 | 集合 {s.milvus_collection} | "
              f"分词器 {s.milvus_analyzer} | 现有 {n} 条")
    finally:
        await store.aclose()
    return 0

try:
    raise SystemExit(asyncio.run(main()))
except SystemExit:
    raise
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
    if os.environ.get("RAG_DEBUG"):
        traceback.print_exc()
    raise SystemExit(1)
PY
) && echo "      $MILVUS_CHECK" || {
    echo "      $MILVUS_CHECK"
    if [ "$VECTOR_MODE" = "lite" ]; then
        echo "      → Lite 建集合失败的常见原因："
        echo "        1. 报 unknown tokenizer type: 'xxx' → MILVUS_ANALYZER 得是 jieba 或 standard，"
        echo "           Lite **不认 chinese**"
        echo "        2. 报缺 milvus-lite → 装的时候没带 extra：uv sync --extra lite"
        echo "        3. 报文件被占用 → Lite 一个 data_dir 只能被**一个进程**打开。"
        echo "           先关掉正在跑的 uvicorn，再确认没有别的脚本占着 ./data/milvus.db"
        echo "        4. 换个集合名也不会好 → 想从零开始就整个删掉：rm -rf ./data/milvus.db"
        echo "        想看完整堆栈：RAG_DEBUG=1 bash scripts/setup_autodl.sh"
    else
        echo "      → 按这个顺序查（详见 docs/DEPLOY.md）："
        echo "        1. 隧道建了吗？AutoDL 上先确认端口在监听："
        echo "             ss -tlnp | grep 19530     （没有 ss 就用 netstat -tlnp）"
        echo "           没有 → 在**虚拟机**上执行（端口用控制台里当前那个，重启会变）："
        echo "             ssh -N -R 19530:127.0.0.1:19530 -p <当前SSH端口> root@region-41.seetacloud.com"
        echo "        2. 隧道在但连不上 → 虚拟机那侧 Milvus 没起来："
        echo "             docker compose -f compose.milvus.yaml logs --tail=50 milvus"
        echo "        3. 报认证失败 → .env 里的 MILVUS_TOKEN 还是占位符，"
        echo "           去虚拟机上把 root 默认密码（Milvus）改掉"
        echo "        4. 报 administratively prohibited → AutoDL 禁了 SSH 端口转发。"
        echo "           别再折腾隧道了，直接换 Lite：VECTOR_MODE=lite bash scripts/setup_autodl.sh"
    fi
}

cat <<'EOF'

============================================================
 完成。启动服务：

   source .venv/bin/activate
   uvicorn rag.main:app --host 0.0.0.0 --port 6006

 ★ AutoDL 只有 6006 端口能通过「自定义服务」对外访问，
   换别的端口就得走 SSH 隧道（ssh -L 8000:127.0.0.1:8000 ...）。

 ★ 启动前确认 .env 里的 LLM_API_KEY 已经填了 —— 这是唯一必须手改的一项。
============================================================
EOF

if [ "$VECTOR_MODE" = "lite" ]; then
    cat <<'EOF'

 数据落在哪：
   ./data/milvus.db/   ← 向量（Lite 把它当**目录**，不是单文件）
   PostgreSQL @5432    ← 正文、分块、任务表、checkpoint

 ★ 备份要两边一起做。只备一边不行：向量库里只有 chunk_id 和正文副本，
   引用定位要的 section_path / 页码在 Postgres 里，版本不一致会导致引用指错地方。
   Lite 的目录里有进程锁文件，**别在服务运行时拷** —— 先停服务再打包。

 ★ Lite 的两条硬限制（都是设计如此，不是 bug）：
   1. 一个 ./data/milvus.db 同时只能被**一个进程**打开 → 不能多开 uvicorn，
      也不能和 worker 同时跑。要并发就换 standalone。
   2. BM25 的 IDF 统计是**分段的**（每个 segment 各自算），不是全库统计。
      语料大了之后稀疏召回会比 standalone 差一截。
============================================================
EOF
else
    cat <<'EOF'

 数据落在哪（备份要两边一起做）：
   虚拟机  milvusdata 卷        ← 向量 + 内嵌 etcd 元数据
   AutoDL  PostgreSQL @5432     ← 正文、分块、任务表、checkpoint

 只备一边不行：向量库里只有 chunk_id 和正文副本，引用定位要的
 section_path / 页码在 Postgres 里，版本不一致会导致引用指错地方。

 ⚠ 隧道断了服务就连不上向量库。虚拟机那侧建议用 autossh 或重试循环保活，
   见 compose.milvus.yaml 末尾的说明。
============================================================
EOF
fi
