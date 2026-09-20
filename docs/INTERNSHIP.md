# RAG-Agent 求职项目版（中小厂实习）

> 本文档是 `docs/SPEC.md` 的**裁剪版**。SPEC 是生产级设计（供你理解"业界怎么做"），
> 本文是**4 周能做完、能写进简历、能扛住面试追问**的最小版本。
> 两者冲突时，**以本文档为准**。

| 项 | 值 |
|---|---|
| 目标 | 拿到中小厂后端/AI 应用开发实习 offer |
| 周期 | 4 周（每天 3–4 小时）|
| 硬件 | Ubuntu 虚拟机，4 核 / 16 GB / 60 GB（最低 4 核 / 8 GB，见 §11）|
| 交付 | 可运行系统 + README + 评测报告 + 简历条目 + 面试话术 |

---

## 1. 与 SPEC.md 的差异（决定 + 理由）

| 项 | SPEC.md（生产） | **求职版** | 为什么改 |
|---|---|---|---|
| 关系库 | MySQL 8.4 | **PostgreSQL 16** | ★ LangGraph **官方** checkpointer 只支持 PG/SQLite/Redis/Mongo。换 PG 直接消除 SPEC 里标记的**头号风险 R1**（MySQL 9.6+ 硬不兼容、社区包单维护者）。JD 写 MySQL 也不怕——事务/SQL/索引是通的，而且"我为什么选 PG"本身就是个加分回答 |
| Milvus 部署 | standalone + 独立 etcd + MinIO（3 容器） | **standalone 单容器**（内嵌 etcd + local storage，`ETCD_USE_EMBED=true`） | 省 2 个容器约 1.5 GB 内存。VM 里这是生与死的差别 |
| Neo4j | 容器 + schema 预留，**不接流水线** | **真接一条图扩展召回路径** | 简历写"预留"= 没做。面试官一追问就穿帮。接一条路径反而好讲 |
| 异步任务 | Celery + Redis + 分队列 + Outbox + 对账 | **Postgres 任务表 + `FOR UPDATE SKIP LOCKED`** | 少 2 个中间件；且"我为什么不用 Celery"是**更好的面试回答**（见 §10 Q9）|
| Embedding / Reranker | 本地 BGE-M3 + bge-reranker（torch） | **API 优先 + Provider 抽象**（本地实现保留但不装进镜像） | ★ 镜像从 ~5 GB 降到 ~400 MB，VM 构建从 20 分钟降到 2 分钟。这是能不能跑起来的关键 |
| 多租户 | tenant_id + RBAC + 服务端强制过滤 | **单租户**（表结构保留 `tenant_id` 列） | 面试不考，但列留着，被问到能说"扩展路径已经留好了" |
| 测试 | 4 层 + 契约测试 + 属性测试 + RAGAS + DeepEval | **pytest 单测（分块/RRF）+ 1 个评测脚本** | 保留最有价值的部分（见 §7）|
| OCR / HITL / Text2Cypher / 社区检测 | 部分做 | **全不做** | |
| 前端 | 无 | **一个单页 HTML（可选，+2 天）** | 有界面演示效果好很多，但要放到第 4 周 |

**没变的（这些是简历的骨架，不能砍）**：混合检索 + RRF + 重排、LangGraph Agentic 图、引用溯源、结构感知切分、增量索引、Docker Compose。

---

## 2. JD 关键词 → 你项目里的落点

> 简历上写了的每个词，都要能指出**具体代码位置**和**一个取舍**。下表就是自查清单。

| JD 关键词 | 项目落点 | 面试会被问什么 |
|---|---|---|
| **LangGraph** | `agent/graph.py`：route → rewrite → retrieve → grade → generate → check 带环图 | 为什么用图不用链？回环怎么防死循环？ |
| **Milvus** | 集合 schema、HNSW 索引、**内建 BM25 Function**、`hybrid_search` + `RRFRanker` | 为什么不用 ES/pgvector？混合检索怎么做？ |
| **FastAPI** | 依赖注入、lifespan 单例、**SSE 流式**、RFC 9457 错误、上传 | 流式怎么实现的？异步里踩过什么坑？ |
| **PostgreSQL** | 业务表 + **LangGraph checkpointer** + 任务表（SKIP LOCKED） | checkpointer 存了什么？为什么不用 MySQL？ |
| **Neo4j** | 实体关系图 + **图扩展召回**（Cypher 1 跳） | 图带来多少提升？为什么不全量上 GraphRAG？ |
| **Docker** | compose 一键起 5 个服务，健康检查 + 依赖顺序 | 容器怎么保证启动顺序？ |
| **RAG / 向量检索** | 解析 → 切分 → 嵌入 → 混合检索 → 重排 → 生成 | 分块多大？怎么定的？ |
| **Prompt 工程** | 结构化输出（Pydantic）、引用强制、注入防御 | 怎么防幻觉？ |
| **（加分）评估** | 50 题评测集 + recall@k + **三组对照 ablation** | ★ 这是你和其他候选人拉开差距的地方 |

---

## 3. 系统范围

### 3.1 做（4 周）

```
1. 摄取：上传 PDF/DOCX/MD → 解析（保留标题层级+页码）→ 结构感知切分
        → embedding → Milvus；元数据入 Postgres
2. 检索：dense + BM25 混合 → RRF(k=60) → reranker → top-5
3. 生成：LangGraph 图（路由/改写/并行检索/评分/生成/校验/有界回环/弃答）
4. 图谱：LLM 抽实体关系 → Neo4j → 图扩展召回并入候选池
5. 服务：FastAPI，同步 + SSE 流式问答，文档管理，任务进度
6. 评估：50 题 golden set，三种检索模式对照，输出对比表
7. 部署：Docker Compose 一键起
```

### 3.2 不做（明确写进 README 的"未来工作"）

OCR 扫描件 · 图文多模态 · HITL 人工审批 · Text2Cypher · 社区检测/全局摘要 · 多租户 RBAC · 分布式部署 · 前端框架（最多一个单页 HTML）

> **在 README 里主动写"不做什么 + 为什么"**，比假装什么都做了强得多。面试官看得出边界。

---

## 4. 架构（求职版）

```
                    ┌─────────────────────────────────┐
   浏览器 / curl ───►│  FastAPI  :8000                 │
                    │  /chat  /chat/stream(SSE)        │
                    │  /documents  /jobs               │
                    └───┬──────────┬──────────┬────────┘
                        │          │          │
             ┌──────────▼──┐  ┌────▼─────┐  ┌─▼──────────┐
             │  LangGraph  │  │ Postgres │  │   Milvus   │
             │  Agent 图    │  │ 16       │  │ standalone │
             │  (进程内)    │  │ 业务+CP+  │  │ 单容器      │
             └──────┬──────┘  │ 任务表    │  └─▲──────────┘
                    │         └────▲─────┘    │
                    │              │          │
                    │        ┌─────┴──────────┴─────┐
                    │        │  Worker (轮询任务表)   │
                    │        │  解析→切分→嵌入→写库   │
                    │        └─────┬────────────────┘
                    │              │
             ┌──────▼───────┐  ┌───▼──────────┐
             │  LLM API     │  │   Neo4j 5.26 │
             │ +Embedding   │  │   实体关系图   │
             │ +Reranker    │  └──────────────┘
             └──────────────┘
```

**四条要能讲出来的设计原则**（比图本身重要）：

1. **Postgres 是唯一事实源**，Milvus / Neo4j 都是可重建的派生索引
2. **chunk_id 是三库共享键**（Postgres `BIGSERIAL` 分配，Milvus 显式主键，Neo4j 唯一约束）
3. **权限过滤下推到查询**（本期单租户，但 `tenant_id` 走服务端注入，不进 LLM 上下文）
4. **检索质量决定上限，编排只负责不失分**

---

## 5. 数据存储决定

### 5.1 为什么是 PostgreSQL 而不是 MySQL

| 理由 | 说明 |
|---|---|
| **★ 官方 checkpointer** | `langgraph-checkpoint-postgres` 是 LangGraph 官方维护，与 LangGraph 版本同步发版。MySQL 是社区包（单维护者），且 **MySQL ≥ 9.6 停用了生成列中的 MD5，项目方明确说无迁移路径** |
| 能力更强 | `JSONB`（存 chunk 元数据/评测结果）、原生数组、`FOR UPDATE SKIP LOCKED` 做任务表、`pgvector` 兜底（同镜像自带）|
| 面试可讲 | "我选 PG 是因为 LangGraph 的持久化层官方只支持 PG/SQLite/Redis/Mongo，用 MySQL 要引第三方包，且和 MySQL 9.6+ 有硬冲突"——这是**有依据的技术决策**，不是随大流 |

> JD 写 MySQL 怎么办？**照投**。关系型数据库的核心（事务隔离、索引、执行计划、锁）完全相通，
> 而且你能讲出"为什么选 PG"，比"我们公司用 MySQL 所以我用 MySQL"强。

### 5.2 为什么是 Milvus standalone 单容器

官方 compose 是 `etcd + minio + standalone` 三个容器。VM 里可以用**内嵌 etcd + 本地存储**压成一个：

```yaml
environment:
  ETCD_USE_EMBED: "true"
  ETCD_DATA_DIR: /var/lib/milvus/etcd
  ETCD_CONFIG_PATH: /milvus/configs/advanced/etcd.yaml
  COMMON_STORAGETYPE: local
  COMMON_STORAGEPATH: /var/lib/milvus/data
```

省约 1.5 GB 内存和 2 个容器。**代价**：单点、不能多副本、数据在容器卷里（个人项目完全可接受，README 里写清楚就行）。

> ⚠️ **两条硬前置**（起不来先查这两个）：
> 1. **CPU 必须有 AVX2**：`grep -m1 -o avx2 /proc/cpuinfo`。没有的话 Milvus 会 `Illegal instruction` **崩溃重启循环**。
>    VMware/VirtualBox 默认透传；Proxmox 的 VM 要把 CPU 类型设为 `host`。
> 2. **镜像 tag 必须显式指定**（禁止 `:latest`）。2.6.x 的最新 patch 号各来源不一致，
>    去 [Milvus releases](https://github.com/milvus-io/milvus/releases) 现查，然后跑一次冒烟（§9.3）。

### 5.3 为什么不塞 Redis / Celery

- 任务状态：Postgres 一张表就够（`SELECT ... FOR UPDATE SKIP LOCKED` 是标准的任务队列模式）
- 缓存：第 4 周如果还有时间，加 Redis 做**检索结果缓存**，然后写"缓存命中率 X%，检索 P95 从 A 降到 B"——**这样加才有意义**

---

## 6. 三个"能聊 20 分钟"的技术亮点

> 面试的胜负手不是功能多少，而是**你能不能就一个点往下聊三层**。下面三个各准备一个数字。

### 亮点 1：混合检索 + RRF + 重排（三层漏斗）

```
查询 ──┬─► Milvus dense (HNSW, COSINE)  ── top 50 ──┐
       └─► Milvus sparse (内建 BM25)     ── top 50 ──┴─► RRF(k=60) ── top 30 ──► reranker ── top 5
```

**要能讲的三层**：

| 层 | 问题 | 你的回答 |
|---|---|---|
| **为什么混合** | 纯向量检索对专有名词/型号/编号失手 | dense 管语义，BM25 管字面，两路互补 |
| **为什么 RRF 不用加权求和** | 两路分数量纲不可比（余弦 ∈ [-1,1]，BM25 无上界）| RRF 只看**排名**不看分数，跨模态可比且**免调参**；k=60 是原论文经验值，k 越小头部权重越大 |
| **为什么重排** | 双塔模型 query/doc 独立编码，精度有天花板 | cross-encoder 联合编码精度高但慢（O(N) 次前向），所以只对 top-30 做，**粗筛→精排** |

**必写的工程细节**（这些才是"你真做过"的证据）：
```python
# 1. 标量索引必须建，否则过滤会退化成全表扫描
index_params.add_index(field_name="tenant_id", index_type="INVERTED")

# 2. BM25 直接传原文，服务端分词（省一次网络往返）
AnnSearchRequest(data=[query_text], anns_field="sparse",
                 param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}, ...)

# 3. RRF 融合的确定性 tie-break —— 不用 dict 迭代顺序
sorted(scores.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
```

### 亮点 2：LangGraph Agentic RAG（有环 + 有界）

```
route ─► rewrite ─► [Send 并行检索] ─► grade ─► generate ─► check_grounded
  │                                                              │
  ├─► direct（寒暄，不检索）                          retries < 2 ─┘
  └─► clarify（问题歧义反问）                                │
                                                             ▼
                                                       abstain（明确弃答）
```

**要能讲的**：

| 问题 | 回答 |
|---|---|
| 为什么用 LangGraph 不用 LCEL 链？ | 链是 DAG，我要**环**（检索不满意回炉改写）和**中断恢复**（checkpointer 存状态）|
| 为什么不用 `create_react_agent`？ | 黑盒 ReAct 循环塞不进 RAG 的关键控制点：逐文档打分、阈值门控、**强制引用**。我用自己的图，每个节点可控可观测 |
| 怎么防死循环？ | **每个回环都有显式计数器**（`state["retries"]`）+ 路由函数里检查 + 显式传 `recursion_limit=50`（默认只有 25）。不依赖 prompt 里写"最多试 3 次" |
| 怎么防幻觉？ | 三层：① 检索分数阈值预筛（低分直接丢，不进 prompt）② 结构化输出**强制逐字引用** `quote` 字段 ③ `check_grounded` 节点校验，不过就回炉或弃答 |
| 多轮对话怎么记？ | checkpointer + `thread_id`，图状态里的 `messages`（配 `add_messages` reducer）**就是会话记忆本身**，不需要额外维护 history |

**面试加分句**："弃答是功能不是缺陷。我在评测集里专门放了 10 道知识库里没有答案的问题，测的是'该弃答时有没有编造'。"

### 亮点 3：图谱增强召回（Neo4j 真接入）

**不要**做全量 GraphRAG（成本高、增量维护复杂、实测只在多跳问题上 +3~5pp）。**只做一条路径**：

```
离线：每个 chunk ──LLM 抽取──► (实体, 关系) ──MERGE──► Neo4j
                               Entity.entity_id = sha256(归一化名称)[:16]
                               → 同名实体跨文档自动合并（天然的实体消歧）
在线：Milvus 召回 seed chunks
        │
        └─► Cypher 1 跳：找与 seed 共享实体的其他 chunk
              MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
              WHERE c.chunk_id IN $seed_ids AND NOT other.chunk_id IN $seed_ids
              WITH other, count(DISTINCT e) AS shared
              ORDER BY shared DESC LIMIT 20
        │
        └─► 并入候选池 ──► ★ 走同一个 reranker ──► 公平对比
```

**要能讲的**：

| 问题 | 回答 |
|---|---|
| 图带来多少提升？ | **用数据回答**（§8 的三组对照）。**如果只在多跳类问题上有提升，就按问题类型选择性启用**——这个诚实的结论比"图很有用"值钱得多 |
| 为什么不做全量 GraphRAG？ | 索引成本是普通 RAG 的几十倍（社区检测 + 摘要），增量维护最复杂（删一篇文档可能让多个社区摘要失效）。我的场景是文档问答，先用**图扩展召回**这个性价比最高的路径验证收益 |
| 实体重复怎么办？ | `entity_id = sha256(归一化名)[:16]`，同名自动同一节点；抽取 chunk 用 1024+ token 的窗口（**实体和它的代词必须在同一块内**，否则 LLM 无从链接）|

> ⚠️ **抽取成本控制**：`MAX_TRIPLETS_PER_CHUNK=30` 防单块爆炸；按 `content_hash` 缓存抽取结果（换 prompt 模板要让 `PROMPT_VERSION` 参与 hash）。
> 用便宜模型抽取（如 DeepSeek-chat），**别用最贵的模型做抽取**。

---

## 7. 评估：你和其他候选人的分水岭

> 90% 的实习简历项目**没有任何量化**。你只要有一个 50 题的评测集和一张对照表，就已经赢了。

### 7.1 评测集怎么建（3–4 小时）

**关键建议：选一个你自己很熟的语料**，否则标注 gold chunk 会痛苦到放弃。

推荐语料（任选其一，20–50 篇）：
- LangGraph / FastAPI / Milvus 的**官方中文文档**（你已经要读了）
- 你自己的课程资料 / 实验报告
- 一个开源项目的中文 README + docs 目录

```jsonl
// eval/golden.jsonl
{"q": "Milvus 的 partition key 最多支持多少个分区？", "gold_chunk_ids": [123, 124], "tags": ["single_hop"]}
{"q": "为什么 RRF 比加权求和更适合融合向量和关键词检索？", "gold_chunk_ids": [88], "tags": ["single_hop"]}
{"q": "文档更新后，系统怎么保证旧向量不会残留？", "gold_chunk_ids": [201, 202, 205], "tags": ["multi_hop"]}
{"q": "这个系统用的什么前端框架？", "gold_chunk_ids": [], "tags": ["unanswerable"]}
```

**配比**：30 题单跳 + 10 题多跳 + **10 题不可回答**（`gold_chunk_ids: []`）。

> ★ **必须有不可回答的问题**。只测可回答问题会系统性高估系统。重点看"不可回答却回答了"这一格——那是幻觉。

### 7.2 指标（自己写，20 行，别依赖黑盒）

```python
recall@k = |retrieved[:k] ∩ gold| / |gold|          # 检索
mrr      = 1 / rank(第一个相关项)
弃答准确率 = 正确弃答数 / 不可回答总数                 # 生成
引用真实率 = quote 能在原文逐字匹配的比例 / 总引用数      # ★ 这个指标别人没有
```

### 7.3 ★ 三组对照实验（写进简历的那张表）

| 模式 | recall@10（单跳） | recall@10（多跳） | MRR | P95 延迟 |
|---|---|---|---|---|
| A. 纯向量 | ? | ? | ? | ? |
| B. 混合 + RRF + 重排 | ? | ? | ? | ? |
| C. B + 图扩展召回 | ? | ? | ? | ? |

**跑完把真实数字填进去。** 这张表就是你简历上所有数字的来源，也是面试时最硬的证据。

> 如果 C 相比 B 提升 < 3pp，**照实写**："图扩展在多跳类问题上 recall@10 提升 X pp，单跳类基本持平，延迟增加 Y ms，因此我把它做成按问题类型可选启用的分支。" —— 这个回答展示的是**工程判断力**，比虚报数字强 10 倍。
>
> 另外做个 **chunk_size ablation**（256 / 512 / 1024）——"我用评测集选的分块大小，不是拍脑袋"是极强的信号。

---

## 8. Docker Compose（Ubuntu VM 可直接跑）

```yaml
# compose.yaml
name: rag-agent

x-app: &app
  build: {context: ., dockerfile: Dockerfile}
  env_file: [.env]
  restart: unless-stopped
  volumes:
    - ./data/uploads:/app/data/uploads

services:
  postgres:
    image: pgvector/pgvector:pg16          # 自带 pgvector，作 Milvus 的兜底方案
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-rag_dev_pw}
      POSTGRES_DB: rag
    volumes: [pgdata:/var/lib/postgresql/data]
    ports: ["127.0.0.1:5432:5432"]          # ★ 只绑本地，不暴露公网
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag"]
      interval: 5s
      timeout: 3s
      retries: 20
      start_period: 30s                     # ★ JVM/DB 冷启动慢，start_period 不能省

  milvus:
    image: milvusdb/milvus:v2.6.11          # ★ 去 releases 页确认最新 2.6.x patch
    command: ["milvus", "run", "standalone"]
    security_opt: ["seccomp:unconfined"]
    environment:
      ETCD_USE_EMBED: "true"                # ★ 内嵌 etcd，省一个容器
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      ETCD_CONFIG_PATH: /milvus/configs/advanced/etcd.yaml
      COMMON_STORAGETYPE: local             # ★ 本地存储，省掉 MinIO 容器
      COMMON_STORAGEPATH: /var/lib/milvus/data
      common.security.authorizationEnabled: "false"   # 个人项目；生产必须 true
    volumes: [milvusdata:/var/lib/milvus]
    ports: ["127.0.0.1:19530:19530", "127.0.0.1:9091:9091"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 90s                     # ★ Milvus 启动慢，给足
    deploy:
      resources:
        limits: {memory: 5G}

  neo4j:
    image: neo4j:5.26-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-rag_dev_pw}
      NEO4J_server_memory_heap_max__size: 1G        # ★ 注意是 server_memory，不是老的 dbms_memory
      NEO4J_server_memory_pagecache_size: 512M
      NEO4J_server_default__listen__address: 0.0.0.0
    volumes: [neo4jdata:/data]
    ports: ["127.0.0.1:7474:7474", "127.0.0.1:7687:7687"]
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-rag_dev_pw} 'RETURN 1' || exit 1"]
      interval: 10s
      timeout: 10s
      retries: 20
      start_period: 60s

  api:
    <<: *app
    command: uvicorn rag.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    depends_on:
      postgres: {condition: service_healthy}   # ★ 长格式，短格式只保证创建顺序
      milvus:   {condition: service_healthy}
      neo4j:    {condition: service_healthy}

  worker:
    <<: *app
    command: python -m rag.worker
    depends_on:
      postgres: {condition: service_healthy}
      milvus:   {condition: service_healthy}

volumes:
  pgdata:
  milvusdata:
  neo4jdata:
```

### Dockerfile

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /bin/     # ★ 锁版本
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev                        # ★ 先装依赖，利用层缓存
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
RUN useradd -m -u 10001 app && chown -R app /app
USER app
```

```toml
# pyproject.toml 里加，让 uv 不把项目本身当包安装
[tool.uv]
package = false
```

### 内存预算

| 服务 | 空闲 | 加载后 |
|---|---|---|
| postgres | ~120 MB | ~400 MB |
| milvus | ~1.5 GB | ~3–4 GB |
| neo4j | ~1.2 GB | ~1.6 GB |
| api | ~200 MB | ~400 MB |
| worker | ~150 MB | ~500 MB |
| **合计** | **~3.2 GB** | **~6.5 GB** |

→ **16 GB VM 很舒服；8 GB VM 能跑但要把 Milvus 限制调到 3G 并别同时开 IDE 编译**。

---

## 9. 4 周构建计划

### Week 1 — 地基 + 摄取链路（目标：上传一个 PDF，能搜到）

| 天 | 任务 | 完成标志 |
|---|---|---|
| 1 | Ubuntu VM 环境：Docker + compose plugin；**验证 AVX2** | `docker run --rm hello-world` 成功 |
| 2 | compose 起 postgres/milvus/neo4j，跑通健康检查 | `docker compose ps` 全 healthy |
| 3 | FastAPI 骨架 + lifespan + `/healthz`；Postgres 建表（documents/chunks/jobs） | 打开 `/docs` 看到接口 |
| 4 | 上传接口（分块读、限制大小、magic bytes 校验）→ 落盘 + 建 job | 上传 50 MB 文件不 OOM，返回 202 + job_id |
| 5 | worker：轮询任务表（SKIP LOCKED）→ 解析（pdfplumber / python-docx / markdown-it-py）| 状态从 queued → ready |
| 6 | 结构感知切分（保留 heading_path / page_no / char_offset）+ 单测 | pytest 通过，中文不被逐字切 |
| 7 | embedding + 写 Milvus（显式主键 upsert）| 手写一次 `hybrid_search` 能召回 |

**本周最容易卡住的**：Milvus 起不来（查 AVX2）、中文切分退化成按字符切（用中文标点做 separator）。

### Week 2 — 检索质量 + 评估集（★ 最关键的一周）

| 天 | 任务 | 完成标志 |
|---|---|---|
| 8 | Milvus schema：显式主键 + `chinese` analyzer + BM25 Function + HNSW + 标量索引 | `run_analyzer` 中文分词正确 |
| 9 | 混合检索 + RRF（自己实现一份，与 Milvus 内建对照）| 两路召回结果能融合 |
| 10 | reranker 接入（API 或本地）+ 阈值截断 | top-5 明显更相关 |
| 11–12 | **建 50 题评测集**（30 单跳 / 10 多跳 / 10 不可回答）| `eval/golden.jsonl` 进版本控制 |
| 13 | 评测脚本：recall@k / MRR / P95 延迟 | 跑出模式 A 和 B 的数字 |
| 14 | **chunk_size ablation**（256/512/1024）| 定下最终分块参数，写进 README |

### Week 3 — Agent 编排 + API

| 天 | 任务 | 完成标志 |
|---|---|---|
| 15 | LangGraph 图：route / rewrite / retrieve / generate | 能端到端回答一个问题 |
| 16 | grade_doc（阈值预筛 + 灰区 LLM 打分）+ check_grounded | 不可回答问题能弃答 |
| 17 | 有界回环（retries ≤ 2）+ abstain + `recursion_limit` | 无死循环，日志能看到回环次数 |
| 18 | Postgres checkpointer + `thread_id` 多轮会话 | 重启进程后会话还在 |
| 19 | 引用溯源（`quote` 逐字摘录 + `chunk_id` 回表）| 答案能定位到具体页码 |
| 20 | **SSE 流式**（`astream` + 反缓冲响应头 + 事件过滤）| curl 能看到逐 token 输出 |
| 21 | 错误契约（RFC 9457）+ 结构化日志 | 异常返回统一格式，不泄漏 API 错误 |

### Week 4 — 图谱 + 收尾

| 天 | 任务 | 完成标志 |
|---|---|---|
| 22 | LLM 抽实体关系 + 写 Neo4j（MERGE 幂等 + 唯一约束）| Neo4j Browser 里能看到图 |
| 23 | 图扩展召回（Cypher 1 跳）并入候选池 | 多跳问题能召回跨文档 chunk |
| 24 | **跑三组对照，填 §7.3 的表** | 拿到真实数字 |
| 25 | README（架构图 + 快速开始 + 设计取舍 + 不做什么）| 别人能一键跑起来 |
| 26 | 简历条目 + 面试话术（§10）| 能脱稿讲 20 分钟 |
| 27–28 | 缓冲（一定会有坑）+ 可选单页 HTML 演示界面 | |

> **时间不够时的裁剪顺序**（从后往前砍）：前端 → 图谱 → 流式 → 引用校验。
> **绝对不能砍**：混合检索、评测集、Docker 一键起。这三个是简历的根。

---

## 10. 简历怎么写 + 面试怎么答

### 10.1 简历条目（直接改数字用）

```
RAG 知识库问答 Agent                                        个人项目
技术栈：Python · FastAPI · LangGraph · Milvus · Neo4j · PostgreSQL · Docker
2026.09 – 2026.11                                        github.com/xxx/rag-agent

· 实现 PDF/Word/Markdown 多格式摄取管道：解析层输出带标题层级与页码的文档树，
  结构感知切分（512 token 子块 + 父块回填），基于内容哈希实现增量索引，
  重复上传零解析零嵌入
· 设计稠密向量 + BM25 稀疏向量混合检索，RRF(k=60) 融合后经 cross-encoder 重排，
  在自建 50 题评测集上 recall@10 由 0.72 提升至 0.94，MRR 由 0.58 提升至 0.81
· 基于 LangGraph 实现 Agentic RAG 有环图（路由→改写→并行检索→相关性评分→
  生成→幻觉校验），支持有界纠错回环与不可回答问题明确弃答，弃答准确率 9/10
· 接入 Neo4j 实体关系图，实现基于实体共现的 chunk 扩展召回，
  多跳类问题 recall@10 提升 X pp，检索 P95 延迟增加 Y ms
· Docker Compose 一键部署 5 个服务（Postgres/Milvus/Neo4j/API/Worker），
  含健康检查与启动依赖编排
```

**写简历的四条纪律**：

| 纪律 | 说明 |
|---|---|
| **每个数字都要能复现** | 面试官问"这个 0.94 怎么测的"，你要能当场打开 `eval/` 目录和评测报告 |
| **写"我做了什么"不写"我用了什么"** | "用 LangGraph 搭建 RAG" ❌ → "实现带相关性评分与有界纠错回环的有环图" ✅ |
| **不写"精通"** | 写"熟悉"、"了解"；被追问时留有余地 |
| **项目放 GitHub，README 要好** | 面试官大概率会点开。README 里要有架构图、快速开始、**设计取舍**、评测数据 |

### 10.2 高频面试题预演

| # | 问题 | 回答要点 |
|---|---|---|
| 1 | **为什么用 Milvus 不用 ES / pgvector？** | 数据规模假设（百万级 chunk）+ 需要内建稀疏向量与混合检索 + 分区横向扩展。**"小规模 pgvector 完全够用，我的向量层做了抽象，换实现只改一个类"** —— 这句话展示你知道边界 |
| 2 | **RRF 为什么比加权求和好？** | 量纲不可比；只看排名不看分数；免调参；单路失效时更鲁棒 |
| 3 | **为什么要重排？粗筛不能筛准点吗？** | 双塔模型 query/doc 独立编码，精度有天花板；cross-encoder 联合编码精度高但复杂度 O(N)。所以两层漏斗：top-50 粗筛 → top-5 精排 |
| 4 | **分块大小怎么定的？** | ★ "做 ablation 测出来的：256/512/1024 三组跑同一个评测集，512 的 recall@10 最高，1024 因为语义稀释下降，256 因为上下文断裂下降" |
| 5 | **怎么防幻觉？** | 三层：检索分数阈值预筛（低分不进 prompt）+ 结构化输出强制逐字 `quote` + groundedness 校验节点；兜底是弃答 |
| 6 | **LangGraph 和 LangChain 什么关系？** | LangChain 是组件库（模型/检索器/工具），LangGraph 是编排层（状态机）。LCEL 链是 DAG，我要**环**和**中断恢复**，所以用图 |
| 7 | **为什么不用 `create_react_agent`？** | 黑盒 ReAct 循环塞不进 RAG 的关键控制点（逐文档打分、阈值门控、强制引用）；且它的状态管理受限 |
| 8 | **怎么防 Agent 死循环？** | 显式计数器 + 路由函数检查 + `recursion_limit`（默认 25，我显式传 50）；**不依赖 prompt 里写"最多试 3 次"** |
| 9 | **为什么不用 Celery？** | ★ "我的场景只有一种任务、单机部署。Postgres 的 `FOR UPDATE SKIP LOCKED` 任务表已经提供了原子领取、幂等重试、崩溃恢复，少一个中间件就少一类故障。**如果要做多优先级队列、定时任务、跨机分布式，我会换 Celery**" |
| 10 | **图谱带来多少提升？为什么不全量上 GraphRAG？** | 报实测数字 + "只在多跳类问题上有收益，所以做成按问题类型可选启用"；全量 GraphRAG 索引成本几十倍且增量维护最复杂 |
| 11 | **文档更新了怎么保证检索不到旧内容？** | 版本化 + 按 `(document_id, version, chunk_index)` 唯一约束 + 幂等 upsert；新版本回填完才切 `is_active`，避免检索窗口期空结果 |
| 12 | **中文处理有什么特别的？** | ★ 三个坑：① tiktoken 对中文 token 膨胀约 3 倍（**用 embedding 模型自己的 tokenizer 数字数**）② `RecursiveCharacterTextSplitter` 默认分隔符对中文会退化成按字符切 ③ `SemanticChunker` 默认句分割正则匹配不到中文句号，整篇成一个块 |
| 13 | **遇到过最难的问题？** | ★ 提前准备 2 个**真实**的（见下） |
| 14 | **如果数据量涨 100 倍？** | Milvus standalone → 集群（etcd/MinIO 独立、QueryNode 水平扩）；`tenant_id` 标量 → partition key；worker 独立扩缩容；引入 Outbox 保证一致 |
| 15 | **你这个项目有什么不足？** | ★ **主动说**：单机部署无高可用、没有多租户隔离、图谱抽取成本未优化、没做 OCR。**能说出自己项目的缺点，是成熟度的信号** |

### 10.3 提前准备两个"真实踩坑"故事

面试官问"遇到什么困难"时，**不要编**。从下面挑两个你真遇到的，按 **现象 → 排查 → 根因 → 修复 → 学到什么** 讲：

| 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|
| **Milvus 起不来** | 容器反复重启，日志 `Illegal instruction` | VM 的 CPU 没透传 AVX2 | 改 VM CPU 类型，或换宿主机 |
| **过滤查询慢 100 倍** | 加了 `tenant_id` 过滤后 P95 从 20ms 涨到 2s | 标量字段**没建 INVERTED 索引**，退化成全表扫描 | 建标量索引，确认执行计划 |
| **中文被逐字切** | chunk 全是单字或超长块 | 默认分隔符 `["\n\n","\n"," ",""]` 对中文无空格文本失效 | 换成 `["\n\n","\n","。","！","？","；","，",""]` |
| **检索结果偶发顺序不同** | 同样输入，重启后 top-5 顺序变了 | RRF 的 tie-break 用了 `dict` 迭代顺序，Python hash 有进程级随机化 | 显式 `(-score, doc_id)` 排序 |
| **SSE 本地正常上云卡死** | 本地逐字输出，通过 Nginx 后一次性吐出 | 缺 `X-Accel-Buffering: no` 响应头 | 加反缓冲三件套 |
| **上传大文件 OOM** | 传 100 MB PDF 时容器被 kill | `await file.read()` 不带 size，整个文件进内存 | 分块读 + 边读边累加校验 |

### 10.4 投递策略（中小厂）

| 项 | 建议 |
|---|---|
| **目标岗位名** | AI 应用开发 / 大模型应用开发 / 后端开发（Python）/ RAG 算法工程（实习）|
| **公司类型** | 20–500 人的 AI 应用公司、垂直行业 SaaS、有知识库/客服/文档产品的团队 |
| **投递渠道** | BOSS 直聘（主力）+ 牛客网 + 实习僧 + 拉勾 |
| **关键词命中** | 简历里**必须出现 JD 里的原词**：LangGraph / Milvus / RAG / 向量检索 / Prompt 工程 / FastAPI。HR 筛简历是关键词匹配 |
| **GitHub** | 项目放上去，README 里放架构图 + 评测数据表 + `docker compose up` 三步启动 |
| **演示** | 录一个 2 分钟屏录（上传文档 → 提问 → 流式回答 → 显示引用），放在 README 顶部 |
| **别做** | 别写"精通 Python"；别堆砌没跑过的技术；别把教程项目原样放上去 |

---

## 11. 降级路径（VM 配置不够 / 时间不够）

### 11.1 VM 只有 8 GB

| 措施 | 省下 |
|---|---|
| Milvus `deploy.resources.limits.memory: 3G` | ~1 GB |
| Neo4j heap 降到 512M / pagecache 256M | ~700 MB |
| 关掉 Neo4j，图谱部分改成"只写不查"或直接跳过 | ~1.5 GB |
| 开发时 `docker compose up postgres milvus api worker`（不启 neo4j） | |
| IDE / 浏览器别和 Docker Desktop 抢内存（**用原生 Linux Docker，别用 Docker Desktop**）| |

### 11.2 Milvus 实在起不来（无 AVX2）

改用 **Milvus Lite**（`pip install pymilvus[milvus-lite]`，纯本地文件，无需 Docker）：

```python
client = MilvusClient("./data/milvus_lite.db")   # 接口与 standalone 几乎一致
```

**代价**（README 里要写明）：单进程文件锁、无认证、**BM25 的 IDF 是段内局部的，全文检索打分与 standalone 不一致**、索引调参被忽略。
**但简历上依然可以写 Milvus**——API 是同一套。面试时主动说明"本地开发用 Lite，生产部署用 standalone，两者 API 一致"。

### 11.3 时间只剩 2 周

砍到最小可讲版本：
1. 摄取（PDF + MD）→ Milvus
2. 混合检索 + RRF + 重排 + **评测集 30 题**（← 这个不能砍）
3. LangGraph 图（route/retrieve/generate/check）
4. Docker Compose
5. 跳过：Neo4j、流式、多轮会话

**简历改成**："实现混合检索 RAG 问答系统，在自建 30 题评测集上 recall@10 达 X" —— 依然能打。

---

## 12. 立即可做的第一步

```bash
# 1. 验证 VM 环境（★ 第一条不过后面全白搭）
grep -m1 -o 'avx2\|avx' /proc/cpuinfo          # 必须输出 avx2
free -g && nproc && df -h /                    # 内存 / 核数 / 磁盘

# 2. 装 Docker（Ubuntu 官方源，别用 snap）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker compose version

# 3. 起基础设施，先验证 Milvus 能跑起来
docker compose up -d postgres milvus neo4j
docker compose ps                              # 等全部 healthy（Milvus 首次约 60-90s）

# 4. 冒烟：Milvus 能不能建集合 + 中文分词对不对
python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
print(c.list_collections())
"
```

**这四步跑通了再写代码。** 环境问题占这类项目失败原因的一半。

---

## 附：本文档与 SPEC.md 的用法

| 场景 | 看哪份 |
|---|---|
| 决定这周写什么代码 | **本文档** |
| 面试前复习 | **本文档 §10** |
| 想知道"生产环境该怎么做" | SPEC.md（面试时能说出来 = 加分）|
| 被问到"你的方案有什么局限" | SPEC.md §3.5（图谱收益的实证）、§14（风险登记册）|
| 想深入某个技术点 | `docs/research/` 下的两份完整调研报告 |
