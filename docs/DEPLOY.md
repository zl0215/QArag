# 部署说明：模型跑在哪里，以及为什么

这份文档回答三个问题：

1. `EMBED_PROVIDER=local / api` 到底在切什么？
2. BGE、reranker 能不能都部署在自己的机器上？
3. 在 AutoDL 上具体怎么落地？

---

## 一、先厘清：三种 Provider 是"模型跑在哪"，不是"本地还是云端"

代码里每个模型能力都有一个 Protocol 接口和多个实现，用配置切换。**检索层代码一行都不用改。**

### Embedding（`EMBED_PROVIDER`）

| 取值 | 实现 | 模型在哪 | 什么时候用 |
|---|---|---|---|
| `local` | `LocalEmbeddingProvider` | **在 api 容器进程内**，sentence-transformers 加载 | 单容器图省事、没有第二个容器的位置 |
| `api` | `APIEmbeddingProvider` | **在另一个进程/容器里**，通过 HTTP 调用 | ★ 推荐。模型只加载一份，API 镜像不用装 torch |
| `hash` | `HashEmbeddingProvider` | 没有模型，blake2b 假向量 | 只用于单测。**不表达任何语义，绝不能用来评评测检索质量** |

关键点：**`api` 不等于"用云端 API"**。`EMBED_API_BASE` 完全可以指向你自己机器上的 `http://embed:80`。

`local` 和 `api` 的真实区别是：

```
local:  [ api 容器  ┌─────────────┐ ]
                    │ FastAPI     │
                    │ BGE 模型    │  ← 权重 1.3GB 在这里
                    └─────────────┘

api:    [ api 容器        ]      [ embed 容器 ]
        │ FastAPI         │ ───→ │ BGE 模型     │
        │ 不含 torch      │ HTTP │ 权重 1.3GB   │
        └─────────────────┘      └──────────────┘
```

### Reranker（`RERANK_PROVIDER`）

| 取值 | 实现 | 说明 |
|---|---|---|
| `api` | `APIReranker` | ★ 推荐。支持两种响应格式（`RERANK_API_STYLE=cohere \| tei`） |
| `local` | `LocalReranker` | 本进程内 CrossEncoder，要装 torch |
| `none` | `NoopReranker` | 不重排，直接用 RRF 顺序。**这是消融实验的基线，不是废物** |

`none` 保留的价值：消融表格里"去掉重排后指标掉多少"这一行，靠它跑出来。如果代码里把 reranker 写成 `if reranker:` 分支，基线和实验组走的就不是同一条代码路径，对比不成立。

### LLM（`LLM_PROVIDER`）

| 取值 | 说明 |
|---|---|
| `openai` | 任何 OpenAI 兼容端点：DeepSeek / 硅基流动 / 阿里百炼 / 自建 vLLM |
| `none` | `NullLLM`，`/chat` 返回 503，但**摄取和检索照常工作** |

**LLM 建议不要自建。** 理由：能跑出比 DeepSeek 更好中文效果的模型至少要 32B（约 20GB 显存），而 AutoDL 上 24GB 卡的时租比直接调 API 贵得多，效果还更差。把 GPU 留给 embedding 和 reranker —— 这两个自建收益明确（免费、无速率限制、数据不出机器、延迟低）。

---

## 二、BGE 是什么，为什么是这两个模型

**BGE = BAAI General Embedding**，智源研究院开源的中文检索模型系列。

### `bge-large-zh-v1.5`（embedding，约 1.3GB）

- 326M 参数，输出 1024 维向量，**最大输入 512 token**
- 中文检索榜上长期是开源第一梯队，中文语义匹配明显强于 `text-embedding-ada-002`
- **非对称检索**：query 加 instruction 前缀 `为这个句子生成表示以用于检索相关文章：`，文档**不加**。
  加错了会稳定掉 3~8 个点，而且不报错。代码里这个前缀只在 `aembed_query` 里加。

> ⚠️ 512 token 上限是本项目的硬约束。`CHUNK_MAX_TOKENS` 配成 1024 会被 tokenizer
> **静默截断** —— 不报错，只是后半段内容永远检索不到。代码里
> `StructureAwareChunker` 会用模型自己的 `max_seq_length` 反过来钳制配置值，
> 并在钳制时打警告。这是 SPEC 里列的坑之一。

### `bge-reranker-v2-m3`（reranker，约 2.3GB）

- Cross-encoder：把 (query, document) **拼在一起**送进模型，输出相关性分数
- 精度比双塔向量高得多，代价是复杂度 O(N) 次前向 —— 所以只能对召回的 top-50 用，不能对全库用
- 这就是"漏斗"结构：向量+BM25 召回 100 条 → RRF 融合 → reranker 精排 → 取 5 条

**两者能不能同时部署？能。** 显存占用（fp16）：

```
embed  ~1.3GB
rerank ~2.3GB
合计   ~3.6GB   → 6GB 显存够用，24GB 卡绰绰有余
```

CPU 上也能跑，只是慢：bge-large-zh 编码一条 384 token 的文本约 15~30ms，reranker 对 50 条打分约 200~500ms。做演示和评测完全够，做压测不够。

---

## 三、推荐拓扑

### 方案 A：全部自建（推荐，AutoDL / 自己的 Ubuntu VM）

```
┌──────────────────────────────────────────────────────────┐
│  宿主机                                                   │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ embed      │  │ rerank     │  │ milvus           │   │
│  │ TEI + BGE  │  │ TEI + BGE  │  │ 向量 + BM25      │   │
│  │ GPU ~1.3GB │  │ GPU ~2.3GB │  │ 内存 ~3GB        │   │
│  └─────┬──────┘  └─────┬──────┘  └────────┬─────────┘   │
│        │ HTTP          │ HTTP             │             │
│  ┌─────┴───────────────┴──────────────────┴─────────┐   │
│  │ api (FastAPI)  +  worker (摄取)                   │   │
│  │ 不含 torch，镜像 ~450MB                            │   │
│  └───────────────────────┬──────────────────────────┘   │
│                          │                              │
│  ┌───────────────────────┴──────────────────────────┐   │
│  │ postgres（文档/分块/任务表 + LangGraph checkpoint）│   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  LLM：外部 API（DeepSeek）—— 不占显存                     │
└──────────────────────────────────────────────────────────┘
```

启动命令：

```bash
WITH_LOCAL=false docker compose --profile models up -d --build
```

然后 `.env` 里：

```ini
EMBED_PROVIDER=api
EMBED_API_BASE=http://embed:80
EMBED_API_STYLE=tei

RERANK_PROVIDER=api
RERANK_API_BASE=http://rerank:80
RERANK_API_STYLE=tei

LLM_PROVIDER=openai
LLM_API_KEY=sk-xxx
```

有 GPU 就叠加：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile models up -d
```

### 方案 B：模型跑在应用进程内（最省事）

```bash
WITH_LOCAL=true docker compose up -d --build
```

```ini
EMBED_PROVIDER=local
EMBED_MODEL_PATH=/models/bge-large-zh-v1.5
RERANK_PROVIDER=local
RERANK_MODEL_PATH=/models/bge-reranker-v2-m3
```

代价：api 和 worker **各加载一份权重**（2 × 1.3GB），且每次改代码重启 API 都要重新加载模型。

### 两种方案的取舍

| | 方案 A（独立服务） | 方案 B（进程内） |
|---|---|---|
| 应用镜像大小 | ~450MB | ~2.2GB |
| 内存占用 | 权重一份 | api + worker 各一份 |
| 改代码重启 | 秒级 | 要等模型重新加载 |
| 多副本扩容 | 直接加 api 副本 | 每个副本一份权重 |
| 容器数量 | 6 | 4 |
| 调试难度 | 多一跳网络 | 单进程，栈更短 |

**面试时这是个好话题**：两种方案都实现了同一个 `EmbeddingProvider` Protocol，切换只改配置。能说清"什么时候该拆、拆的代价是什么"，比"我会用 Docker"有信息量得多。

---

## 四、AutoDL 落地形态（两轨 / 三轨）

AutoDL 的标准实例**跑不了 Docker**（官方明确：容器实例内不支持 Docker，
要用 Docker 得租裸金属整机、包月）。所以向量库必须另想办法，有两条路：

- **Milvus Lite**（★ 推荐）—— 向量库变成本地一个目录，**不需要第二台机器**
- **虚拟机跑 Milvus + SSH 反向隧道** —— 传统三轨，支持多进程，但多一条不可控的网络链路

**★ 默认走 A（两轨）**，因为 A 把"网络"这一整个失败面消掉了。B 是原来的三轨方案，
留在这里是因为它支持多进程（将来加 worker 时可能需要），但调试成本高得多。

```
【A】两轨 —— 推荐，scripts/setup_autodl.sh 的默认路径
┌─────────────────────────┐        ┌──────────────────────────┐
│ ① 本机（Windows/macOS）  │        │ ② AutoDL（11G 显存）      │
│    MemoryVectorStore    │        │    BGE embed + reranker  │
│    不需要 Docker         │        │    （API 进程内，~3.6GB） │
│    smoke_local.py       │        │    FastAPI  :6006        │
└─────────────────────────┘        │    PostgreSQL（apt 装）   │
      同一份代码，传上去 →          │    Milvus Lite（本地文件） │
                                   └──────────────────────────┘
                                     ./data/milvus.db —— 不连网，
                                     没有第二个进程，没有隧道

【B】三轨 —— 需要多进程/多副本时（VECTOR_MODE=standalone）
┌─────────────────────────┐        ┌──────────────────────────┐
│ ① 本机（Windows/macOS）  │        │ ③ AutoDL（11G 显存）      │
│    MemoryVectorStore    │        │    BGE embed + reranker  │
└─────────────────────────┘        │    FastAPI  :6006        │
                                   │    PostgreSQL（apt 装）   │
┌─────────────────────────┐        └────────────┬─────────────┘
│ ② 虚拟机（6GB 内存）      │                     ↑
│    Milvus standalone    │                     │
│    Docker               │  虚拟机主动连出去，把端口"带"过去
│    compose.milvus.yaml  │─────────────────────┘
│    192.168.126.130      │   ssh -N -R 19530:127.0.0.1:19530 \
└─────────────────────────┘       -p <当前SSH端口> root@region-41.seetacloud.com
                                  ⇒ AutoDL 的 127.0.0.1:19530 = 虚拟机的 Milvus
```

**为什么这么分**：需要 GPU 的放 AutoDL；纯开发的在本机。
B 里虚拟机那一轨的由来是"AutoDL 跑不了 Docker，向量库得放别处"——
但 Lite 出来之后这个前提不成立了，**能不放第二台机器就不放**：
每多一个服务就多一个"连不上"的失败点，而 AutoDL ↔ 虚拟机那条链路
是整个架构里唯一不可控的一环（NAT、SSH 端口漂移、端口转发策略，全都不在你手里）。

### ★ 为什么不能用 `192.168.126.130` 直连

这是最容易踩的坑，先把结论说清楚：**AutoDL 连不到这个地址，跟做不做端口映射无关。**

| | |
|---|---|
| `192.168.126.130` | RFC1918 私有地址（VMware NAT 网段），只在你**自己的宿主机虚拟网络里**有意义 |
| `region-41.seetacloud.com` | 公网机房里的另一台机器，和你的虚拟网络没有任何路由关系 |
| 结果 | 数据包出了 AutoDL 就被丢弃 —— 不是"被防火墙拦"，是**根本没有路由** |

`ssh -p <当前SSH端口> root@region-41.seetacloud.com` 是你**主动连进** AutoDL，
这条链路是单向的，不构成 AutoDL 反向访问你内网的通路。

**但反方向是通的**，这就是解法：虚拟机能出网（NAT 本来就允许 outbound），
AutoDL 有公网地址能 SSH 进去 —— 那就让虚拟机主动连出去，把端口带过去。
详见下面 ①。

> 顺带一提，这个方案比"开安全组放行 19530"**更安全**：隧道默认只监听
> AutoDL 的**回环地址**，公网上扫不到这个端口，Milvus 也可以安心绑 `127.0.0.1`。

### AutoDL 的两个硬约束

- **只有 6006 端口**能通过「自定义服务」对外访问，其余端口要走 SSH 隧道
- **`/root` 是系统盘**（通常 30GB，扩容收费），模型权重和大数据放 `/root/autodl-tmp`

### ① 虚拟机：起 Milvus

```bash
# 前置 1：CPU 必须支持 AVX2，否则 Milvus 会 Illegal instruction 崩溃循环
grep -m1 -o avx2 /proc/cpuinfo

# 前置 2：6GB 内存是"能跑但紧"的下限，先把 swap 开出来
#         Milvus 加载 segment 时有内存尖峰，超了 cgroup 上限就是 OOM-kill
#         （容器重启 + 索引重建，很慢）。有 swap 兜一下，尖峰变成"慢几秒"。
free -h
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 只需要这一个文件，自包含，不需要源码和模型
docker compose -f compose.milvus.yaml up -d
docker compose -f compose.milvus.yaml logs -f milvus    # 等到 "Milvus successfully started"
```

> 内存分配参考（6GB 机器）：系统 ~0.5G + Docker ~0.2G + Milvus 2.5~3G。
> compose 里 `MILVUS_MEM_LIMIT` 默认 **4G** 就是按这个定的。
> 这是 cgroup 硬上限，超了直接 OOM-kill；不设的话爆的是整台虚拟机，SSH 都会卡死。

**起来之后立刻改掉 root 默认密码**（`authorizationEnabled=true` 时 Milvus 会建
root 用户，默认密码就是 `Milvus`，不改等于没开认证）：

```bash
docker compose -f compose.milvus.yaml exec milvus python3 - <<'PY'
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
c.update_password(user="root", old_password="Milvus", new_password="你的强密码")
print("密码已更新")
PY
```

**然后建 SSH 反向隧道 —— 这是 AutoDL 能连过来的唯一通路。**

在**虚拟机**上执行（不是 AutoDL）：

```bash
ssh -N -R 19530:127.0.0.1:19530 -p <当前SSH端口> root@region-41.seetacloud.com
```

- `-N` 只做转发不开 shell；`-R` 在远端（AutoDL）监听 19530 并转回本地
- **不要加 `-g`**（GatewayPorts）。不加的话隧道只绑 AutoDL 的回环地址，公网扫不到
- ★ 端口从 AutoDL 控制台的「SSH 登录指令」里复制，**别用本文档里的旧值**。
  实例每次重启都可能重新分配（实测 `55154` → `42064`），
  端口变了隧道会握不上手，症状是 `Connection refused`

> ⚠️ **`ssh -N` 成功和失败长得一模一样**（都没有输出）。别靠"没报错"判断隧道通了 ——
> 加 `-v` 看有没有 `remote forward success`，或直接去 AutoDL 上
> `ss -tlnp | grep 19530`（AutoDL 容器里可能没装 `ss`，那就用 `netstat -tlnp`）。
> 这两条是唯一可靠的判据。

隧道会随 SSH 断开而消失，别用一次性终端。保活二选一：

```bash
# 方式一：autossh，断了自动重连（推荐）
sudo apt install -y autossh
autossh -M 0 -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -R 19530:127.0.0.1:19530 -p <当前SSH端口> root@region-41.seetacloud.com

# 方式二：不想装东西就循环重连
while true; do
    ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -R 19530:127.0.0.1:19530 -p <当前SSH端口> root@region-41.seetacloud.com
    echo "隧道断了，5 秒后重连"; sleep 5
done
```

> ⚠️ 前提是 AutoDL 的 sshd 没禁 `AllowTcpForwarding`（默认允许）。
> 若在 d 步排查中看到 `administratively prohibited: open failed`，
> 说明被禁了 —— 改用下面的 **Milvus Lite** 方案，或上 Tailscale/ZeroTier 组虚拟内网。

### ② AutoDL：装环境、连 Milvus、起服务

```bash
# ---- 1. 依赖（复用实例自带的 CUDA torch，别让它被换掉）----
pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /root/rag-agent
uv venv --python "$(command -v python)" --system-site-packages .venv
source .venv/bin/activate

# ★ uv 不读 pip.conf（Rust 实现，刻意不支持），所以镜像必须用环境变量给它。
#   不设的话 uv sync 会直连 pypi.org / files.pythonhosted.org，国内几十分钟起步。
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=300

# ★ 千万别加 --frozen：它会**忽略全部镜像配置**，直接从 uv.lock 里写死的
#   files.pythonhosted.org 下载，前面设的镜像全部作废（uv 已知问题 #19625）。
uv sync --extra local --extra dev --extra lite --no-install-package torch

# 立刻验证 torch 还带 CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ---- 2. PostgreSQL（正文 + 任务表 + checkpoint）----
apt-get update && apt-get install -y postgresql postgresql-contrib
pg_ctlcluster "$(ls /etc/postgresql | sort -V | tail -1)" main start
su postgres -c "psql -c \"CREATE USER rag WITH PASSWORD 'rag_dev_pw'\""
su postgres -c "psql -c \"CREATE DATABASE rag OWNER rag\""

# ---- 3. 模型（放数据盘，软链接进项目）----
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"
mkdir -p /root/autodl-tmp/models && cd /root/autodl-tmp/models
hf download BAAI/bge-large-zh-v1.5  --local-dir bge-large-zh-v1.5
hf download BAAI/bge-reranker-v2-m3 --local-dir bge-reranker-v2-m3
cd /root/rag-agent && mkdir -p models
ln -sfn /root/autodl-tmp/models/bge-large-zh-v1.5  models/bge-large-zh-v1.5
ln -sfn /root/autodl-tmp/models/bge-reranker-v2-m3 models/bge-reranker-v2-m3

# ---- 4. 起服务 ----
uvicorn rag.main:app --host 0.0.0.0 --port 6006
```

`.env` 里 AutoDL 相关的部分。**★ 推荐用 Milvus Lite**（理由见下面 ④）：

```ini
# ---- Milvus Lite：向量库是 AutoDL 上的一个本地目录，不连网 ----
VECTOR_BACKEND=milvus
MILVUS_URI=./data/milvus.db     # 目录路径，不是 http:// 地址
MILVUS_TOKEN=                   # ★ Lite 没有鉴权，必须留空
MILVUS_COLLECTION=rag_chunks
MILVUS_ANALYZER=jieba           # ★★ Lite 不认 chinese，填了建集合直接报错

REPOSITORY_BACKEND=postgres
POSTGRES_HOST=127.0.0.1

EMBED_PROVIDER=local
EMBED_MODEL_PATH=./models/bge-large-zh-v1.5
EMBED_DEVICE=cuda
EMBED_BATCH_SIZE=16

RERANK_PROVIDER=local
RERANK_MODEL_PATH=./models/bge-reranker-v2-m3

LLM_PROVIDER=openai
LLM_API_KEY=sk-xxx
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

> 模型路径写成 `./models/...` 相对路径即可（uvicorn 从 `/root/rag-agent` 启动）。
> 之前这里写的是 `/root/rag-agent/models/...` 绝对路径 —— 项目目录不叫这个名字时
> 会静默加载失败（报"找不到模型"而不是"路径不对"）。脚本生成的也是相对路径。

<details>
<summary>用 SSH 隧道连虚拟机的 Milvus（不推荐）</summary>

```ini
VECTOR_BACKEND=milvus
# ★ 是 127.0.0.1，不是虚拟机 IP —— 隧道把虚拟机的 19530 映射到了 AutoDL 的回环
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=root:<你改后的密码>
MILVUS_COLLECTION=rag_chunks
MILVUS_ANALYZER=chinese
```

其余同上。这条路要求隧道**一直活着**，且实例每次重启 SSH 端口都可能变
（实测 55154 → 42064），隧道必须跟着重建。
</details>

连通性自检（在 **AutoDL** 上跑）：

```bash
python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://127.0.0.1:19530', token='root:<密码>')
print('集合列表:', c.list_collections())
"
```

连不上按这个顺序查（从近到远，别跳步）：

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 虚拟机上 `ssh` 报 `Connection refused` | **SSH 端口变了**（实例重启过） | 去控制台重新复制 SSH 指令，改 `-p` |
| AutoDL 上 `ss -tlnp \| grep 19530` 没输出 | 隧道没建 / 已断 | 去虚拟机上执行 ① 里的 `ssh -N -R ...`；用 `ssh -v` 确认真建上了 |
| 有监听但连不上 / 超时 | 虚拟机那侧 Milvus 没起来 | `docker compose -f compose.milvus.yaml logs --tail=50 milvus` |
| 报认证失败 | token 不对 | 密码改了吗？`.env` 里同步了吗？ |
| `administratively prohibited` | AutoDL 禁了 SSH 端口转发 | **改用 Milvus Lite**（别再折腾隧道了） |

> 注意 **没有"安全组"这一步** —— 隧道走的是 SSH（就是 AutoDL 控制台给的那个端口），
> 不需要额外放行任何端口，也不需要在虚拟机上配 ufw。
> 这是这个方案比"公网直连 + 白名单"省事的地方。

**★ 说白了：这张表里的每一行，改用 Milvus Lite 就都不存在了。**
只有在明确需要多进程访问同一个向量库时才值得走隧道。

### ③ 备份：两边必须一起备

```
【A 两轨】AutoDL 上：
  ./data/milvus.db/    向量（Lite 把它当目录，里面还有锁文件）
  PostgreSQL @5432     正文、分块、任务表、checkpoint

【B 三轨】：
  虚拟机  milvusdata 卷   向量 + 内嵌 etcd 的元数据（schema/索引/segment 状态）
  AutoDL  PostgreSQL      正文、分块、任务表、checkpoint
```

> ⚠️ A 方案的 `data/milvus.db/` **不能热拷** —— 目录里有进程锁，
> 服务运行时拷出来的是不一致的快照。先停 uvicorn 再打包。

**只备一边是不够的**：向量库里只有 `chunk_id` 和正文副本，
引用定位需要的 `section_path` / 页码在 Postgres 的 `chunks` 表里。
两边版本不一致 → chunk_id 对不上 → 引用指错地方或回表失败。

### ④ Milvus Lite（AutoDL 侧的推荐方案）

早期版本的本文档写过"Milvus Lite 不支持 BM25，不能用"——**这个说法是错的**。
那说的是旧的 C++ 版。现在 pymilvus 3.x 配的 Python 重写版**支持 BM25**。
下面这些是**在本机实跑验证过的**，不是查文档抄的
（验证脚本：`scripts/smoke_milvus_lite.py`，12 项全过）：

| 能力 | Lite | 备注 |
|---|---|---|
| `FunctionType.BM25` 服务端生成稀疏向量 | ✅ | 插入只给原文，不用自己算 |
| `search(anns_field="sparse", data=[原文])` | ✅ | 服务端分词，实测有召回 |
| HNSW / COSINE、SPARSE_INVERTED_INDEX、标量 INVERTED | ✅ | 参数照传，不报错 |
| `query(count(*))` / `query_iterator` | ✅ | 对账逻辑可用 |
| `delete` | ✅ | ⚠ 返回值形状不同，见下 |
| `run_analyzer` | ❌ | RPC 直接 UNIMPLEMENTED，已在代码里降级 |
| `MILVUS_ANALYZER=chinese` | ❌ | **只认 `jieba` / `standard`** |

启用：`uv sync --extra lite`，然后 `.env` 里
`MILVUS_URI=./data/milvus.db` + `MILVUS_TOKEN=`（留空）+ `MILVUS_ANALYZER=jieba`。

#### ★ 四个实测踩出来的坑（都已修在代码/脚本里）

1. **不认 `chinese` 分词器** —— 报 `unknown tokenizer type: 'chinese'
   (supported: 'standard', 'jieba')`，而且是在 `create_collection` 时才炸。
   `config.py` 的默认值是 `chinese`（standalone 用），**所以 AutoDL 的 `.env`
   必须显式写 `MILVUS_ANALYZER=jieba`**。分析器建集合时固化，事后改不了：
   想换只能删库重建（或改 `MILVUS_COLLECTION` 换个集合名）。

2. **检索结果的主键键名不同** —— standalone 返回 `item["id"]`，
   Lite 返回 `item["chunk_id"]`（用的是主键的**字段名**）。
   只认 `"id"` 的话每次检索都 KeyError，还会被包成 `ProviderError`
   伪装成"检索失败"。`MilvusVectorStore._to_hits` 现在两种都认，
   认不出来时抛出带实际字段名的错误。

3. **`MILVUS_URI` 是目录，不是文件** —— Lite 把它当 `data_dir`，
   跑完躺着的是 `./data/milvus.db/` 目录（里面有锁文件）。
   所以清空要 `rm -rf`，备份要**先停服务**再打包。

4. **`delete()` 返回值形状不同** —— standalone 返回 `{"delete_count": N}`，
   Lite 返回**被删主键的列表** `[1001, 1002]`。
   原来的代码只认字典，Lite 上恒返回 0 —— 删除**成功却报 0 条**，
   排查时会以为没删掉。（当时的调用方丢弃了这个返回值，所以没造成实际故障。）

#### 两条真实的代价（架构层面，改代码解决不了）

- **单进程独占** —— 一个 `data_dir` 同时只能被一个进程打开。
  不能多开 uvicorn，**也不能和 worker 同时跑**。要并发只能上 standalone。
- **BM25 的 IDF 是分段局部的** —— 每个 segment 各自统计，不是全库统计。
  反复小批量插入会让稀疏分数漂移，跑消融实验时要认真对待。

**它和"虚拟机跑 Milvus"是互斥的选择，不是叠加。** 两个理由选它：
① SSH 隧道建不起来（AutoDL 禁 `AllowTcpForwarding`，或隧道静默失败）；
② 不想为了向量库一直开着一台虚拟机。**本文档把它作为 AutoDL 侧的默认方案**——
它把"隧道 + 虚拟机 + 第二个进程"这一整类失败点直接消掉了。

`scripts/setup_autodl.sh` 默认就装它、生成 Lite 配置；想走隧道那条约
`VECTOR_MODE=standalone bash scripts/setup_autodl.sh`。

### 向量后端的边界

| | `milvus` + standalone | `milvus` + Lite | `memory` |
|---|---|---|---|
| 需要 Docker | ✅ | ❌ | ❌ |
| 需要第二台机器 | ✅ | ❌ | ❌ |
| 重启后数据还在 | ✅ | ✅ | ❌ |
| 常驻 worker（与 API 并存） | ✅ | ❌ 文件锁，见 [六](#六摄取-worker) | ❌ |
| BM25 IDF | 全局 | 分段局部 | 全局（本地实现） |
| `chinese` 分词器 | ✅ | ❌ 只能用 jieba | — |
| 适用场景 | 生产 / 要 worker | **AutoDL 单机** | 单测、一次性冒烟 |

前两个是**同一个 `VECTOR_BACKEND=milvus`**，靠 `MILVUS_URI` 的格式区分
（`http://` vs `./xxx.db`），不是两个后端。切换只改 `.env`，
三个实现的是同一个 `VectorStore` Protocol。

> 曾经还有第四个 `sqlite` 后端（自研 numpy 单文件实现），**已删除**：
> 它"无 Docker 时的持久化"这个定位被 Milvus Lite 完全覆盖，
> 而 Lite 有真正的服务端 BM25，那份是手写的简化实现，两条路只会互相拖累。

**Postgres 装不上怎么办？** 设 `REPOSITORY_BACKEND=memory` 能跑起来，但代价要清楚：
正文、任务表、LangGraph checkpoint 全在进程内，**重启即丢**，且没有任务队列。
只够"先看一眼效果"，不是可交付状态。

### 本机应用 + 虚拟机依赖（不花 AutoDL 时长）

要验证"API 与 worker 两个进程共享一个 standalone"这条链路时，**不必用 AutoDL**。
把耗资源的依赖丢给虚拟机，应用跑在本机现成的 venv 里：

```
Windows（本机）                     虚拟机
  API 进程    ─┐                  ┌─ Postgres 容器   127.0.0.1:5432
  worker 进程 ─┴─ SSH 隧道 ────────┴─ Milvus 容器     127.0.0.1:19531
```

两个容器都只绑**回环**，隧道也是本机发起，所以没有任何东西暴露到网络。
两个进程都连 `127.0.0.1`，配置上完全看不出中间隔着一条隧道。

```bash
# 1) 虚拟机上起两个依赖（回环）
docker run -d --name rag-postgres -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag_dev_pw -e POSTGRES_DB=rag \
  -v rag-pgdata:/var/lib/postgresql/data --restart unless-stopped postgres:16-alpine

docker run -d --name rag-milvus-test -p 127.0.0.1:19531:19530 \
  -e DEPLOY_MODE=STANDALONE \
  -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e ETCD_CONFIG_PATH=/milvus/configs/advanced/etcd.yaml \
  -e COMMON_STORAGETYPE=local -e COMMON_STORAGEPATH=/var/lib/milvus/data \
  -e COMMON_SECURITY_AUTHORIZATIONENABLED=false \
  -v rag-milvus-test-data:/var/lib/milvus --memory 2g \
  milvusdb/milvus:v2.6.11 milvus run standalone

# 2) 本机开隧道（本机 19530 → 远端 19531，避开可能已有的 19530）
ssh -N -L 5432:127.0.0.1:5432 -L 19530:127.0.0.1:19531 ragvm

# 3) 两个进程，各开一个终端
VECTOR_BACKEND=milvus DATABASE_URL='postgresql+asyncpg://rag:rag_dev_pw@localhost:5432/rag' \
  .venv/Scripts/python.exe -m uvicorn rag.main:app --port 8000
VECTOR_BACKEND=milvus DATABASE_URL='postgresql+asyncpg://rag:rag_dev_pw@localhost:5432/rag' \
  .venv/Scripts/python.exe -m rag.worker
```

用环境变量覆盖而**不改 `.env`**，是因为真环境变量优先级高于 dotenv，
这样验证用的配置和日常开发用的 `.env` 互不干扰。

> **两个坑，都踩过：**
>
> 1. `milvus run standalone` **必须**同时给 `DEPLOY_MODE=STANDALONE`。
>    少了它，启动时直接 `panic: embedded etcd can not be used under distributed mode`
>    然后退出码 134 —— 报错在 Go 的堆栈里，看不出是缺了个环境变量。
> 2. Milvus 的 `/healthz` 在 **9091** 端口，不在 19530。
>    探测 19530 会一直拿到 404，误判成"起不来"。探活用
>    `POST /v2/vectordb/collections/list` 更准。

2026-09-14 实测结果：`?sync=false` 上传 → `queued`(16s，含 worker 冷加载 BGE)
→ `running` → `succeeded`（3 chunk / 3 向量）；`/api/v1/retrieve` 三问全部命中，
无关查询 `sparse=0`（BM25 正确不匹配）而降级为纯稠密召回。

### 四之补：GPU 算力放 AutoDL，应用留在本机（2026-09-14 实际形态）

上面 A/B 两轨都是"整个应用跑在 AutoDL"。实际跑起来之后改成了**算力与应用分离**，
因为把整个应用丢上云有几个说不通的地方：AutoDL 是 **GPU 计费**的，
而 Postgres / Milvus / FastAPI 一点 GPU 都不用；AutoDL 的系统盘
在"重建实例"时会重置，数据库放上去等于随时可能丢；
而解析、分块这些活还是要一个常驻进程。

所以变成三台机器各干各擅长的：

```
┌── 本机（Windows）────────────────┐
│  api  :8000    worker            │  ← 只做 HTTP 和解析，不加载任何模型
│  516MB / 449MB（原来各 ~2.5GB）   │
└───────┬──────────────┬───────────┘
        │ ssh -L       │ ssh -L
        │ 5432,19530   │ 8081
        ▼              ▼
┌── 虚拟机 ragvm ──┐  ┌── AutoDL（RTX 2080 Ti 11G）──────┐
│  Postgres        │  │  scripts/serve_inference.py      │
│  Milvus standalone│  │  bge-large-zh-v1.5  (embed)      │
│  （数据留本地）    │  │  bge-reranker-base  (rerank)     │
└──────────────────┘  │  127.0.0.1:8080，只绑回环         │
                      └──────────────────────────────────┘
```

`scripts/serve_inference.py` 提供 TEI 兼容的两个端点，本机 `.env` 这样指：

```ini
EMBED_PROVIDER=api
EMBED_API_BASE=http://127.0.0.1:8081     # 走 SSH 隧道
EMBED_API_STYLE=tei

RERANK_PROVIDER=api
RERANK_API_BASE=http://127.0.0.1:8081
RERANK_API_STYLE=tei
```

**为什么服务端也用 sentence-transformers，而不是手写 CLS pooling**：
本机原来的 `LocalEmbeddingProvider` 用的就是 `SentenceTransformer.encode(normalize_embeddings=True)`，
池化方式由模型目录里的 `1_Pooling/config.json` 决定。Milvus 里已有的向量是它产出的 ——
手写一份 pooling 意味着"两边一致"要靠人保证，而这里**必须**一致：
差一点点，旧向量和新查询就不在同一个空间，**不会报错，只会检索变差**。
用同一个库、同一份权重，一致性是构造出来的。

> **切过去之前务必验一次向量一致性**（换模型、升级 `sentence-transformers` 之后也要重验）：
> 同一批文本两边各编码一次，逐条算余弦。实测 `|cos − 1| ≈ 1e-13`，
> 说明池化、归一化、权重都对得上，已有向量不作废。差到 1e-3 以上就要停下来查。

**收益（2.2MB / 8 页 PDF，75 个块，实测）：**

| 阶段 | 迁移前 | 迁移后 |
|---|---|---|
| PDF 解析（pdfplumber，仍在本机） | ~8s | ~8s |
| 分块 | 0.2s | 0.2s |
| **嵌入** | **72.5s**（967 ms/块，本机 CPU） | **~0.3s**（GPU） |
| 合计 | ~81s | 11.5s |

本机 CPU 峰值从"打满近一分半"降到"解析这 8 秒"，api / worker 各瘦 ~2GB。

> **解析仍然占本机 CPU，这是有意留下的。** pdfplumber 是纯 Python 逐字符解析，
> 2.2MB 的 PDF 要 8 秒单核。把它也搬到 AutoDL 并没有好处 ——
> 那是 **CPU** 活，搬到 **GPU 计费**的机器上只是换个地方烧钱，
> 而且要把整篇文档的正文送到云端。真要治它得换解析后端
> （pypdfium2 快一个数量级，但没有 `extract_text_lines` 的行结构，
> 分块质量会变），属于另一件事。

**AutoDL 重启后隧道会断**（SSH 端口每次都变，见 `~/.ssh/config` 的 `autodl` 段）。
恢复顺序：

```bash
ssh autodl "cd /root/rag-infer && nohup /root/miniconda3/bin/python serve_inference.py \
  --embed-model /root/autodl-tmp/models/bge-large-zh-v1.5 \
  --rerank-model /root/autodl-tmp/models/bge-reranker-base --port 8080 &"
ssh -N -L 8081:127.0.0.1:8080 autodl &
curl -s http://127.0.0.1:8081/health   # 两个 true 才算好
```

模型放在 `/root/autodl-tmp/models/`（**数据盘，不会随实例重建被清**）。
国内下 HuggingFace 要两个环境变量，少一个都会失败：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1     # ★ 少了它会绕过镜像直连 cas-server.xethub.hf.co，401
```

---

## 五、把模型传到服务器

模型权重不进镜像，也不进 git（`.gitignore` 已排除 `models/`）。上传方式：

```bash
# 从 Windows（Git Bash）
scp -P <端口> -r "D:/PythonDemo_learn/shucang/embedding/bge-large-zh-v1.5" \
    root@<host>:/root/rag-agent/models/
```

国内服务器直接从 HuggingFace 下载通常不通，用镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3
```

目录结构必须是：

```
models/
├── bge-large-zh-v1.5/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
└── bge-reranker-v2-m3/
    └── ...
```

---

## 六、摄取 worker

摄取 = 解析 → 分块 → 嵌入，是**分钟级**的重活。放在 HTTP 请求里做的话，
请求会超时、uvicorn 的并发位会被长时间占死。所以有一条异步路径：

```
POST /api/v1/documents?sync=false   →  只入库，立刻返回 job_id
GET  /api/v1/jobs/{job_id}          →  轮询，直到 succeeded / failed
```

消费 `job_id` 的就是 worker —— 一个**独立进程，和 API 同镜像不同命令**：

```bash
# 容器里（compose.yaml 的 worker 服务就是这么写的）
python -m rag.worker

# 把当前排队的跑完就退出 —— 补数据、CI 冒烟、配合 cron
python -m rag.worker --once
```

同镜像这个选择是刻意的：两边用的依赖版本、provider 实现、分块参数天然一致。
如果 worker 单独打镜像，迟早会出现"API 切出来的块和 worker 切出来的块不一样"，
而这种不一致在检索质量上是**看不出来的**，只会表现为"某些文档搜不到"。

### 为什么不用 Celery

Postgres 的 `SELECT … FOR UPDATE SKIP LOCKED` 已经提供了原子领取，
不需要再引入 Redis/RabbitMQ 那一套中间件。少一个组件就少一类故障。
完整论证在 `src/rag/infra/repository.py` 顶部。

横向扩容同理：`docker compose up -d --scale worker=3`，任务不会被重复消费。

### 重试与租约

| 情况 | 行为 |
|---|---|
| 文件不存在、解析为空（`IngestionError` / `UnprocessableDocumentError`） | **不重试**，直接判 `failed` |
| `ProviderError`（向量库/模型抖动）、未知异常 | 退避重排：5s → 10s → 20s … 封顶 300s |
| 重试次数用尽（`max_attempts`，默认 3） | 判 `failed`，`last_error` 记下原因 |
| worker 被 `kill -9` | 任务停在 `running`；租约（900s）过期后被**别的** worker 自动捞回重跑 |
| `job_type` 不认识 | 判 `failed`（重试一万次也还是没有这个处理器） |

> 为什么"文件不存在"不重试：再跑一百次还是同一个结果。重试只会把一次失败
> 拖成三次失败，还占着队列。

> 为什么租约定 900 秒：它的作用是**故障恢复**，不是超时取消。
> 定得太短，一个正常跑着的"大 PDF 解析 + 嵌入"会被第二个 worker 抢走 ——
> 那就变成重复摄取了，比慢更糟。所以它必须大于最慢的一次摄取。

### 两条启动前就会拒绝的配置

`preflight_problems()` 在启动时拦死，因为这两个错误的症状**完全看不出原因**：

1. `REPOSITORY_BACKEND=memory` —— worker 是独立进程，看不见 API 进程里的内存队列。
   症状：任务永远停在排队，而 worker 日志里一片安静。
2. Milvus Lite（`MILVUS_URI=./data/milvus.db`）—— 一个 data_dir 只能被**一个进程**打开。
   症状：worker 和 API 抢同一个文件，**后启动的那个**起不来。
   这台机器"上次还好好的，重启一次就不行了"。

### AutoDL 上只有 Lite，怎么用 worker

Lite 是 AutoDL 侧的默认方案，而它和常驻 worker 天然冲突。正当的用法是
**停掉 API → 批量补数据 → 再起 API**：

```bash
# 1. 停掉 API（释放 ./data/milvus.db 的文件锁）
# 2. 批量跑完队列里的任务，跑完自己退出
python -m rag.worker --once --allow-lite
# 3. 重新起 API
```

`--allow-lite` 是**显式**出口，默认不开。理由是常见的错误用法（两个进程都常驻）
必然抢锁，而那个报错发生在后启动的进程上，很难往文件锁上想。
开了之后日志里会留一条 `worker.lite_lock_overridden` 警告 ——
它的作用是以后有人翻日志时能看出"当时是故意这么干的"，而不是怀疑配置写错了。

> 比拿几百个 HTTP 请求去打一个持锁的 API 好得多 —— 后者每个请求都在抢
> 那把本来就只能一个进程持有的锁。

### 本地验证到什么程度

- `python scripts/smoke_worker.py` —— **30 项全过**。覆盖 preflight 的两条拒绝
  与 `--allow-lite` 出口，以及整条任务生命周期：正常路径 + 哈希去重、
  永久性错误、路径越界、退避重排、重试耗尽、未知 job_type、
  租约过期回收、关停。用内存队列 + 假向量，不需要 Postgres / Milvus / 模型。
- `python scripts/smoke_worker_pg.py` —— **41 项全过**，需要一台真 Postgres。
  它单独存在是因为内存后端**验不了承载"任务表替代 Celery"这个决定的三条裸 SQL**：
  `claim_job` 的 `FOR UPDATE SKIP LOCKED`、`schedule_retry`、`requeue_stale_jobs`。
  内存实现是单进程里一个 dict 加锁，天然不会出现"两个 worker 领到同一条" ——
  Postgres 下会不会，只能真连上去试。最重要的是**并发领取**那一项：
  20 个任务、8 个 worker 同时抢，断言一条都不能被重复领取
  （写漏 `FOR UPDATE` 的表现是同一份文档被摄取两遍，而且只在并发时偶发）。

  其中 [8] [9] 两节覆盖幂等键的三个分支，都是**真出过问题**的：
  文档被删掉后再上传同一个文件，必须重新摄取（只看任务状态会返回陈旧的
  `succeeded`，界面上"已完成"而知识库空空如也）；旧任务状态是 `failed` 时
  重传必须复用那一行 —— `idempotency_key` 上有唯一约束，再插一行就是
  `IntegrityError`，表现为"上传一个曾经解析失败的文件"返回 500。

  ```bash
  # 必须先有测试库（脚本只建表，不建库）
  createdb -O rag rag_smoke
  SMOKE_DATABASE_URL='postgresql+asyncpg://rag:<密码>@localhost:5432/rag_smoke' \
      python scripts/smoke_worker_pg.py
  ```

  > ★ 库名里必须有 `smoke` 或 `test`，否则脚本拒绝执行 ——
  > 它会 `DELETE FROM`，这个闸门保证复制粘贴错一行 URL 也不会清掉生产库。
  > 它开头和结尾各清一次：开头那次是让脚本**可重复运行**
  > （只在结尾清的话，跑到一半挂掉留下的残留任务会命中幂等键，
  > 让下一次运行的断言以极具误导性的方式失败）。

---

## 七、常用命令

```bash
# 默认栈（postgres + milvus + api + worker），模型跑在进程内
docker compose up -d --build

# 加上独立推理服务
docker compose --profile models up -d

# 再加上图谱
docker compose --profile models --profile graph up -d

# GPU
docker compose -f compose.yaml -f compose.gpu.yaml --profile models up -d

# 看日志
docker compose logs -f api
docker compose logs -f embed

# 进容器排查
docker compose exec api python -c "import rag; print(rag.__file__)"
```

> 改了 `CHUNK_*` 分块参数后，必须 bump `CHUNKER_VERSION` 并重新入库 ——
> 否则增量索引会认为旧块仍然有效，检索结果不会变。重新入库的 CLI
> 与对账脚本属于接口层，随 `rag.main` 一起实现。

---

## 八、上线前检查清单

**基础设施**

- [ ] 虚拟机上 `grep -m1 -o avx2 /proc/cpuinfo` 有输出（否则 Milvus 会 `Illegal instruction` 崩溃循环）
- [ ] 虚拟机上开了 swap ≥ 2G（6GB 内存跑 Milvus，没 swap 会被内存尖峰 OOM-kill）
- [ ] 虚拟机上 Milvus 的 `authorizationEnabled=true`，且 **root 默认密码 `Milvus` 已改掉**
- [ ] 反向隧道建起来了，且配了保活（autossh 或重连循环）—— 隧道断了服务就静默失联
- [ ] 19530 绑的是 `127.0.0.1`（走隧道就不该对公网暴露；`ss -tlnp` 确认一下）
- [ ] 9091（metrics）永远只绑 `127.0.0.1`，绝不对外
- [ ] `POSTGRES_PASSWORD` 已从默认值改掉
- [ ] Postgres **绝不放公网**（要远程就开 SSH 隧道）—— 里面是全部文档原文
- [ ] `APP_ENV=prod`、`LOG_LEVEL=INFO`（`DEBUG` 会把请求体写进日志）

**配置正确性**

- [ ] `MILVUS_URI=http://127.0.0.1:19530`（走隧道就是回环地址，**不是**虚拟机 IP）
- [ ] `MILVUS_TOKEN` 不是 `REPLACE_WITH_*` 占位符
- [ ] `LLM_API_KEY` 已填（不填 `/chat` 返回 503，检索接口不受影响）
- [ ] `EMBED_MAX_TOKENS` 留空，让代码从模型配置推断
- [ ] 改了 `CHUNK_*` 参数时同步 bump `CHUNKER_VERSION`（否则增量索引复用旧块）
- [ ] `EMBED_DIM` 与 Milvus 集合建表时的维度一致（不一致要到写入时才报错）

**验收**

- [ ] `python scripts/smoke_local.py` 通过（本机，不需要 Docker / Milvus）
- [ ] AutoDL 上 `MILVUS_URI` 连通性自检通过（`setup_autodl.sh` 第 5 步会跑）
- [ ] `curl 127.0.0.1:6006/readyz` 六个组件全 `✓`
- [ ] 传一份文档，确认 `/api/v1/retrieve` 能召回它的内容（不是只返回 200）
- [ ] `python scripts/smoke_worker.py` 通过（本机，不需要 Docker / Postgres）
- [ ] `SMOKE_DATABASE_URL=... python scripts/smoke_worker_pg.py` 通过（**需要真 Postgres**，见下）
- [ ] 传一个 `sync=false` 的任务，确认 worker 日志里出现 `worker.job_start` → `worker.job_done`
- [ ] 确认 `GET /api/v1/jobs/{id}` 从 `queued` 变成 `succeeded` 且 `document_id` 非空
