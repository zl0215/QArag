# 架构与流程拆解

> 这份文档回答一个问题：**一个请求从进来到出去，经过了哪些文件、哪些函数、为什么这么切。**
>
> 引用形式是 `文件:函数名`，不写行号 —— 行号会随改动漂移，函数名不会。
> 想快速定位实现，直接搜函数名即可。

---

## 0. 三条主线

系统里跑着三条**互不依赖**的主线，它们共用同一套存储层和 Provider 抽象：

```
主线 A  摄取   文件 → 解析 → 分块 → 嵌入 → Milvus + Postgres     异步，走 Worker
主线 B  问答   问题 → 检索 → 判定 → 改写/生成 → 引用校验          同步，走 API
主线 C  评测   golden → 检索/问答 → 分层归因                     离线脚本，不进运行时
```

主线 A 和 B **在代码上唯一的交汇点是 `services/` 和 `infra/`** —— API 和 Worker 各起各的进程，
各自 `build_context()` 装配一套组件，不共享内存、不共享单例。

---

## 1. 分层

依赖方向**严格单向向下**，没有任何反向 import。这条约束是可验证的：
`core/` 和 `schemas/` 里不出现 `services`、`agent`、`api` 的字样。

```
┌─ 编排层 ────────────────────────────────────────────┐
│  api/       HTTP 边界 + 静态前端    worker/   任务循环  │
│  agent/     LangGraph 图与节点                        │
└──────────────────────┬──────────────────────────────┘
┌──────────────────────▼──────────────────────────────┐
│  services/   业务用例：摄取、检索、融合、翻译            │
└──────────────────────┬──────────────────────────────┘
┌──────────────────────▼──────────────────────────────┐
│  parsers/  chunking/  providers/  infra/            │
│  能力层：不知道"业务"是什么，只提供机制                  │
└──────────────────────┬──────────────────────────────┘
┌──────────────────────▼──────────────────────────────┐
│  core/  schemas/    配置、日志、错误、事件循环、数据契约   │
└─────────────────────────────────────────────────────┘
```

**为什么这么切**：`providers/` 不知道什么叫"知识库"，`infra/` 不知道什么叫"问答"。
所以把嵌入从本机换到 AutoDL、把 Milvus 换成内存实现、把 Postgres 换成内存仓储，
都只需要动一个工厂函数或一个装配点，上层零改动。

---

## 2. 目录职责

| 目录 | 职责 | 关键约束 |
|---|---|---|
| `core/` | 配置、日志、错误、事件循环 | 无业务依赖，谁都能 import |
| `schemas/` | 跨层数据契约 | `DocNode` / `Chunk` / `Citation` |
| `parsers/` | 文件 → `ParsedDocument` | 格式差异**只允许存在于这一层** |
| `chunking/` | `ParsedDocument` → `Chunk[]` | 结构感知 + token 计数 |
| `providers/` | 模型能力（嵌入 / 重排 / LLM） | 本地实现与 API 实现可互换 |
| `infra/` | 存储（Postgres / Milvus / 内存） | 全部藏在 Protocol 后面 |
| `services/` | 业务用例 | 编排层共享 |
| `agent/` | LangGraph 编排 | 图、节点、提示词、状态 |
| `api/` | HTTP 边界 + 静态前端 | **只做转换，不含业务** |
| `worker/` | 后台任务循环 | 与 API 共享 `services/` |
| `eval/` | 评测与归因 | 离线，不参与运行时 |
| `scripts/` | 运维与冒烟 | 手动执行 |

---

## 3. 数据契约

三个结构贯穿全线，理解它们就理解了数据流：

```
parsers/ 产出   ParsedDocument ──┐
   schemas/document.py          │  DocNode 树（含 NodeType / DocMeta）
                                │  标题层级、页码、章节路径都在这里定好
                                ▼
chunking/ 产出  Chunk[]  ───────┐
   schemas/chunk.py             │  chunk_id、content、section_path、page_start/end
                                │  compute_chunk_hash() 用于去重
                                ▼
问答输出        Citation
   schemas/chunk.py             chunk_id + 逐字摘录 quote
```

**`DocNode` 是解析层唯一的出口**。PDF、Markdown、DOCX 三种解析器的内部实现
完全不同，但都收敛到同一个 `DocNode` 树 —— 所以下游的分块器**只需要写一份**。

`schemas/document.py:make_node_id` 用内容哈希生成稳定 ID，`compute_sha256` /
`compute_text_hash` 服务于幂等。

---

## 4. 主线 A：摄取

### 4.1 入口：只入队，不干活

```
api/routes.py:upload_document
  ├─ 落盘到 uploads/
  ├─ 算 raw_hash
  ├─ repository.enqueue_job()   → 写 Postgres，返回 job_id
  └─ 立即返回 202
```

**这里刻意不同步解析**。一份 2.2MB 的 PDF 解析要 8 秒以上，
同步做会把 HTTP 连接占死，用户看到的是一个转圈圈然后超时。

### 4.2 执行：`worker/runner.py:Worker`

```
startup()                  装配 services（和 API 用同一套 build_*）
run()                      主循环
 ├─ _wait_for_dependencies()   依赖没起来就不抢任务
 ├─ _reap_stale()              回收僵死任务（进程被 kill 留下的"进行中"）
 ├─ _process()                 取一个 job 执行
 │   └─ _handle_ingest()
 └─ _on_failure()              失败重试
```

`preflight_problems()` 在**启动时**检查配置（数据库连不连得上、Milvus 是不是 Lite 模式
却配了别的、API key 有没有），不满足直接拒绝启动。

> 这是踩过坑之后加的：任务跑了一半才失败，排查成本远高于启动即失败。

`retry_delay_seconds()` / `is_retryable()` 决定重试策略 —— **不是所有失败都值得重试**，
文件损坏重试一百次也是损坏。

### 4.3 核心：`services/ingestion.py:IngestionService._run`

五步，**顺序有硬约束**：

```python
# ---- 幂等短路 ----
#   raw_hash 命中已有 document 且产物还在 → 直接返回，不重复干活

1. 解析      parse_document()            → ParsedDocument
2. 分块      StructureAwareChunker.split() → Chunk[]
3. 分配 ID   Postgres 先写 chunk_id       ← ★ 必须先于向量库
4. 嵌入      embedder.embed(batch)        分批，带进度
5. 写向量    vectorstore.upsert()         ← 最后一步，失败不回滚
```

**为什么 3 必须在 4 之前**：`chunk_id` 由 Postgres 自增生成，而 Milvus 的主键就是它。
反过来做会出现"向量库有数据、关系库没记录"的孤儿向量 —— 这种状态无法自动修复，
只能全量重建。

**为什么 5 失败不回滚**：重试时整条链路重跑，靠开头的幂等短路收敛。
回滚反而引入"删了一半"的中间态。

### 4.4 解析层：`parsers/base.py:parse_document`

按 MIME 分发 → `parse_pdf` / `parse_markdown` / `parse_docx`，统一产出 `DocNode` 树。

| 函数 | 作用 |
|---|---|
| `sanitize_text` | 清洗不可见字符、统一空白 |
| `detect_lang` | 判语言（影响后续 BM25 分词策略） |
| `sniff_mime` / `_is_docx_zip` | 按内容而非扩展名判类型 |
| `finalize` | 收口：补 node_id、算哈希、规整层级 |

**PDF 是重灾区**，`parsers/pdf.py` 里大部分代码在处理它的病：

| 函数 | 解决什么 |
|---|---|
| `_collect_furniture` | 页眉页脚污染（统计每页重复出现的字符串并剔除） |
| `_detect_two_columns` | 双栏论文读成左右交错 |
| `_extract_layout_lines` | 按坐标重建行序 |
| `_join_wrapped_lines` | 英文断词粘连（`FuzzySetsandSystems394`） |
| `_is_page_number` / `_is_running_header` | 页码与running head |
| `_classify_heading_levels` / `_heading_level` | 用字号与字重推断标题层级 |
| `_render_heading_path` | 生成 `章 > 节 > 小节` 路径 |

### 4.5 分块：`chunking/splitter.py:StructureAwareChunker`

按章节树切，**标题与章节路径会拼进嵌入文本** —— 这是"标题语义缺失"的解法：
一个孤立的段落"…因此可以视为近似线性算法"没有任何检索价值，
加上 `GBSVM > 时间复杂度分析 > 粒度球分类器` 之后才能被召回。

`chunking/counter.py` 提供两种 token 计数：

- `TransformersTokenCounter` —— 精确，需要模型
- `HeuristicTokenCounter` —— 兜底，`_is_cjk()` 让中文按字算

> 目标 256 / 上限 384 / 重叠 48 token（`config.py`）。

---

## 5. 主线 B：问答

### 5.1 入口：`api/routes.py:chat` → `_run`

```python
history = await _history(ctx, thread_id)     # 取裁剪过的对话历史
state   = await run_agent(ctx.graph, question,
                          history=history, max_retries=...)
await _remember(ctx, thread_id, ...)         # 只写非空回答
```

`_history` 读的是 **`messages` 表，不是 LangGraph 的 checkpointer**。

> checkpointer 里存的是整个 state —— 带着 chunks、diagnostics、grade_reason，
> 回灌给 LLM 既浪费 token 又干扰判断。所以单独维护一份裁剪过的 history。
> 仓储没实现 `get_messages` 时降级为空历史，不影响单轮问答。

### 5.2 图拓扑：`agent/graph.py:build_graph`

```
START → prepare → retrieve → grade ─┬─ generate → verify → END
                          ↑         ├─ rewrite ─┐
                          └─────────┘          │
                                    refuse ←───┘
```

- `grade` 决定"直接答"还是"改写后重查"
- `rewrite` 决定"再查一轮"还是"用手上的料作答 / 拒答"
- **`refuse` 直达 END**：不进 `generate`（会被无关 chunk 诱惑着硬编出一段话并挂上引用），
  也不进 `verify`（没有引用可校验，拒答本身没有断言任何事实，是可信的）

`_route_of()` 对未知路由值兜底成 `generate`。LangGraph 遇到未声明的分支会抛
`InvalidUpdateError`，而且**错误信息不告诉你是哪个节点发出的** —— 兜底让流程走完并留下日志。

### 5.3 状态：`agent/state.py:AgentState`

三个"问题"字段，各司其职：

| 字段 | 含义 | 谁写 | 谁读 |
|---|---|---|---|
| `question` | 用户**原话**，永不改写 | `initial_state` | generate（作为 user message）、日志、引用 |
| `query` | 当前**用于检索**的查询 | prepare / rewrite | retrieve |
| `standalone` | 消解了上下文、**不带检索改写**的"用户到底在问什么" | prepare（只写一次） | grade、rewrite |

**为什么必须有 `standalone`**（这是踩坑后加的）：

- 用 `question` 去判：追问的原话是「那它大概要多久算完？」，「它」是什么只有对话历史知道，
  而 `GRADE_PROMPT` 里**不含历史**（故意的 —— 检索到的资料才是判据）。
  模型读不懂问题，只能判 insufficient → 拒答。
- 用 `query` 去判：`rewrite` 节点会把它换成关键词表，比较题的「A 和 B 哪个快」
  被压成「A 的指标词汇」，于是只答一半还判 sufficient。

**判据必须锚在用户意图上，而用户意图在 state 里需要一个不被改写的落点。**

### 5.4 节点逐个说

#### `prepare` — 消解上下文

把「那它大概要多久算完？」补成独立问题。三条防线，任一触发就退回原问题：

```python
history_text = "\n".join(f"{role}: {_clip(content, 300)}" for m in history[-6:])
rewritten = await llm.acomplete(CONTEXTUALIZE_PROMPT...)

if len(rewritten) > len(question) * 4 + 50:     退回   # 长度防线：模型返回了一整段解释
if _looks_like_keyword_salad(rewritten):        退回   # 词表防线
```

两个文本工具：

- `_clip()` —— 按**句子边界**截断。直接 `text[:300]` 会从「对于 GBSVM 的最大时间复杂」
  中间切断，半句话进提示词。
- `_looks_like_keyword_salad()` —— 判据故意保守：**既无疑问标记、又有 ≥4 个空格分隔的中文片段**。
  只打词表（`GBSVM 训练时间 实测 运行耗时 秒 分钟`），正常的改写不会被误伤。
  宁可漏放，不可误杀 —— 误杀会把一个好改写换成原问题。

#### `retrieve` — 检索 + 多轮证据合并

```python
fresh  = [chunk_to_dict(c) for c in result.chunks]
chunks = _merge_retrieval_rounds(state["chunks"], fresh, limit=top_k * 2)
```

只有 `retries > 0` 才合并。**作用**：比较题首轮召回 A、改写轮聚焦 B，
两轮并集才覆盖两边。直接覆盖第一轮会出现"补到了 B，却把已经找到的 A 丢掉"。
限制 `2×top_k` 防止无关重试无限膨胀。

单轮检索失败**不让整张图中断**，只写 `error` 并降级。

#### `grade` — 判定证据是否充分

```python
grade_question = state.get("standalone") or state.get("query") or state["question"]
verdict = await llm.astructured(GRADE_PROMPT.format(question=grade_question, context=...),
                                _GradeVerdict)
```

判定**失败**时倾向 `generate` 而非 `rewrite`：重试要再花一轮检索 + LLM 调用，
而生成至少能给出带引用的答案，由用户判断可信度。

#### `rewrite` — 聚焦缺口重查

`REWRITE_PROMPT` 接收三样东西：`question=standalone`（不是原话 —— 追问原话读不懂）、
`query`（上次用过的查询）、`reason`（grader 说缺什么）。

**核心是"聚焦 grader 指出的那个缺口"，一次只查一件事。**

> 比较题「A 和 B 哪个快」：如果 reason 显示只缺 B，就**只查 B**。
> 一条查询同时拉两个实体会被其中一个主导，结果还是捞回 A，缺口永远补不上。

`_exhausted_route()` 决定重试用尽后的去向，判据是 **`grade_verdict` 而不是 `chunks`**：

> 稠密通道一次召回 top_k×10 条，`chunks` 几乎永远非空。用它做判断等于永远走 generate，
> grader 的"证据不足"被静默吞掉 —— 用户看到的就是"重试两轮还是硬生成一段话，
> 并且挂着 5 条无关引用"。

#### `generate` / `refuse` / `verify`

- `generate` 带 top-k 上下文 + 最近 6 条历史生成。**不写 `citations`** ——
  该字段的归约器是 `operator.add`，两处都写会变成重复累加。
  未配置 LLM 时走**抽取式兜底**：附上 top-1 原文并明确标注，而不是假装有答案。
- `refuse` **不调 LLM**，确定性输出，不产生引用，不进 verify。
- `verify` 纯确定性，三件事：
  1. 剔除指向不存在编号的引用（模型写了 `[7]` 但上下文只有 5 条）
  2. `_best_quote()` 从**原文**里找与答案重叠最高的一段作为摘录
     （`_bigrams()` 算 CJK 字符二元组重合度 —— 不需要分词、不需要模型）
  3. 一条有效引用都没有 → `verified = False`

> **摘录必须是原文的连续子串**，不能是答案里的句子 ——
> 否则"引用校验"就变成了自己证明自己。

### 5.5 检索漏斗：`services/retrieval.py:RetrievalService.retrieve`

```
① 跨语言扩展   needs_translation() → translator.translate()
               追加 _x 一路，不替换原查询
② 并发召回     asyncio.gather(dense, dense_x, sparse, sparse_x)
               单路失败 → 降级成单路，不整体 500
③ RRF 融合     fusion.rrf_fuse(k=60)
④ 截断重排     _select_rerank_candidates() → _rerank()
⑤ 后处理       _postprocess_order() → _attach_siblings() → _dedupe()
```

参数（`config.py` / 构造函数默认值）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `rrf_k` | 60 | RRF 平滑常数 |
| `dense_top_k` / `sparse_top_k` | 50 / 50 | 每路召回条数 |
| `fused_top_k` | 60 | 融合后进重排的候选数 |
| `final_top_k` | 5 | 最终给 LLM 的条数 |
| `max_context_chars` | 12000 | 上下文硬上限 |

**① 跨语言扩展是"追加"而不是"替换"**：

> 本项目语料是英文论文，用户用中文提问。两个通道各自的下场：
> - 稀疏（BM25）：中文查询对英文文档 **一条都召不回**。这是结构性的 ——
>   BM25 是词法匹配，中文词条永远命中不了英文词条。
> - 稠密：中文 query 对英文 doc 的对齐能力远弱于中文-中文。
>
> 原问题和译文**各自能召回对方漏掉的块**，并集的上限高于任一路。
> 替换会丢掉原问题里的语感信息，追加不会 —— 代价只是多一次稠密前向。
>
> 翻译放在**检索侧**而不是生成侧：生成侧的语言由用户决定（中文问就中文答），
> 检索侧的语言由**语料**决定。解耦之后，用户不需要知道自己的知识库是什么语言的。

几个值得记住的函数：

| 函数 | 作用 |
|---|---|
| `_dense` / `_sparse` | 两路召回，各自独立 |
| `_select_rerank_candidates` | 决定哪些进重排器 |
| `_has_mixed_language_term` / `_mostly_cjk` | 中英混排的处理策略 |
| `_rerank` | 调重排器；**失败时退回融合顺序**，不让整条链路挂掉 |

**`_rerank` 里有一段容易忽略的逻辑：重排时到底把哪个查询喂给重排器。**
它不是简单地用原查询，而是按**候选文档本身**的 CJK 占比来选：

```python
cjk_ratio = 候选块里"以中文为主"的比例
if 查询像在找书目信息 或 含中英混合术语:  原查询 + 译文
elif cjk_ratio <= 0.35:                  只用译文   # 候选基本是英文
elif cjk_ratio >= 0.65:                  只用原查询 # 候选基本是中文
else:                                    原查询 + 译文
```

> 判据取自**文档侧**而不是查询侧，是因为同一份语料里可能中英文档混存。
> 拿中文查询去重排一批英文块，cross-encoder 的分数会整体偏低且区分度差。
| `_postprocess_order` | 书目/作者/多文档意图的结构化排序 |
| `_attach_siblings` | 把相邻块补回来（答案常跨块） |
| `_dedupe` | 去重 |
| `build_context` | 生成 `[1] [2]` 编号上下文 |

> `build_context` 的编号格式**必须与 `agent/nodes.py:_render_context` 一致** ——
> 否则 verify 阶段对不上号，引用会全部失效。

`services/fusion.py`：`rrf_fuse()`（只看排名不看分数）、`rerank_order()`、`dedupe_by_content()`。

---

## 6. 横切关注点

| 文件 | 作用 |
|---|---|
| `core/config.py:Settings` | pydantic-settings；`_empty_string_means_unset` 让空串回落到默认值 |
| `core/loop.py:platform_loop_factory` | **Windows 上 uvicorn 的事件循环工厂是硬编码的**，替换成 Selector 才能用 psycopg |
| `core/errors.py` | `DomainError` 家族，`main.py:_domain_error` 统一转 HTTP |
| `api/deps.py:AppContext` | **唯一的装配点**：`startup()` 装完所有组件，`readyz()` 逐组件健康检查 |
| `providers/__init__.py` | `build_embedding_provider` / `build_reranker` / `build_llm`，按配置选实现 |
| `infra/repository.py` | `Repository` Protocol，`MemoryRepository` 与 `PostgresRepository` 可互换 |
| `infra/vectorstore.py` | `VectorStore` 抽象；`MemoryVectorStore` 内含 `_BM25`，`tokenize` 对中文切字 |
| `infra/milvus.py` | `MilvusVectorStore`，`escape_filter_string` 防注入 |

**`AppContext` 是全项目唯一知道"所有组件怎么拼起来"的地方。**
测试里想换任何一个组件（内存仓储、假 LLM、Noop 重排器），改这一个文件就够。

---

## 7. 存储层的两个硬约束

### 7.1 软删除哨兵值

`documents.deleted_at` 是 **`NOT NULL`**，用 `EPOCH_ZERO`（1970-01-01）表示"未删除"，
**不是 NULL**。

> 所有查询都要带 `deleted_at == EPOCH_ZERO`。
> 写成 `IS NULL` 会一条都查不出来 —— 而且不报错，只是安静地返回空。

### 7.2 分页必须倒序再反转

`PostgresRepository.get_messages`：

```python
rows = (await session.scalars(
    select(Message).where(...).order_by(Message.seq.desc()).limit(limit)
)).all()
return [... for m in reversed(rows)]     # 反转回时间正序
```

> 正序 + LIMIT 取到的是**最早**的几轮，恰好是上下文里最不该留的。

---

## 8. 脚本与运维

| 脚本 | 用途 |
|---|---|
| `scripts/run_api.py` | 启 API |
| `scripts/serve_inference.py` | **AutoDL 上的 GPU 推理服务**（TEI 风格 `/embed` `/rerank`） |
| `scripts/reindex_corpus.py` | 重建索引（v3 蓝绿切换） |
| `scripts/setup_autodl.sh` | AutoDL 环境搭建 |
| `scripts/shot_ui.py` | CDP 截图 |
| `scripts/smoke_local.py` | 全链路冒烟 |
| `scripts/smoke_milvus_lite.py` / `smoke_worker.py` / `smoke_worker_pg.py` | 分级冒烟 |

**分级冒烟的意义**：出问题时能快速二分是"存储坏了"还是"编排坏了"，
不用每次都跑全链路。

---

## 9. 评测：`eval/`

| 脚本 | 回答什么问题 |
|---|---|
| `run_eval.py` | 端到端指标：chunk/doc Recall@k、MRR、弃答准确率、误拒率、引用逐字真实率 |
| `funnel.py` | **逐级天花板**：dense 池 / sparse 池 / 融合池 / 融合+重排，各自能到多少 |
| `diagnose.py` | 四假设归因，定位"到底哪一环丢的" |
| `analyze_failures.py` | 严格重判（用比 `diagnose` 更严的判据复核） |
| `probe_multiturn.py` | 多轮追问探针 |
| `probe_rewrite.py` | 提示词对照台：同一段历史下不同提示词的改写结果 |
| `dump_corpus.py` | 导出语料快照 |

### 9.1 `resolve_evidence`：为什么 gold 不能写死 ID

`run_eval.py:resolve_evidence` 把**稳定的证据短语**解析成当前索引的 `chunk_id`：

```python
# 数据库自增 ID 会在每次重新分块后变化，直接把 ID 当长期 gold 会让
# "改进分块"这件事本身破坏评测。
```

**这是一条容易忽视但很重要的设计**：如果 gold 写死 chunk_id，
那么每次调整分块策略，评测集就失效了 —— 你会看到指标暴跌，
然后去调检索参数，而真正的原因是标注过期了。

`cmd_check` 在跑指标**之前**先校验 golden 自洽（gold_doc_ids 与 gold_evidence 是否对得上）。

> 这一步不能省。gold 打错一个数字，评测不会报错，只会安静地把 recall 拉低几个点。

### 9.2 词表漏掉的边界：`is_junk` 的教训

早期用过一条启发式判断"这个 chunk 是不是参考文献碎片"：

```python
len(t) < 40 or len(REF_PAT.findall(t)) >= 3     # ✗ 这条判据是错的
```

它把 **10/59 = 16.9% 的 gold chunk 误判成了垃圾** —— 包括一个
字面包含答案的块，只因为里面有 `[0.7,1]` 这种像引用编号的数字。

换成严格判据（`vol./no./pp./doi` 里至少命中 2 个）后误判率降到 3.4%。

> **教训**：用来"清洗数据"的启发式，必须先拿它去跑一遍 gold。
> 一个把 gold 判成垃圾的过滤器，会让所有后续分析都建立在错误前提上。

---

## 10. 关键设计决策速查

| 决策 | 为什么 |
|---|---|
| 上传不同步解析，只入队 | 8 秒以上的解析占死 HTTP 连接 |
| chunk_id 先写 Postgres 再写 Milvus | 反过来会产生无法自动修复的孤儿向量 |
| 向量写入失败不回滚 | 重试靠 `raw_hash` 幂等短路收敛，回滚会引入中间态 |
| 跨语言用追加而非替换 | 两路并集的召回上限高于任一路 |
| 翻译放检索侧不放生成侧 | 生成语言由用户定，检索语言由语料定 |
| 拒答走独立节点，不调 LLM | 否则 generate 会把无关 chunk 拼进 prompt 硬编一段话 |
| `refuse` 不经过 `verify` | 它没有断言任何事实，本身可信 |
| 引用摘录取原文连续子串 | 取答案里的句子等于自己证明自己 |
| 图节点一律不抛异常 | 抛异常会中断整张图，checkpoint 停在中间态极难排查 |
| grader 锚 `standalone` | 见 §5.3 —— 用 question 读不懂，用 query 会被改写污染 |
| 重试用尽看 `grade_verdict` 不看 `chunks` | 稠密一次召回 50 条，chunks 永远非空 |
| `is_junk` 类启发式必须先跑 gold | 见 §9.2 |

---

## 11. 已知边界

写清楚**做不到什么**，比写清楚能做什么更重要：

1. **语料规模**：当前 12 份有效文档 / 967 个向量。这是工程验证规模，
   不是生产规模。百万级 chunk 需要重新考虑索引类型与分片策略。
2. **评测集**：50 题，40 可答 + 10 不可答。题目由本人构造，
   **不是多学科盲测**，因此不能用来主张"适用于所有学科"。
3. **跨语言方向是单向的**：实现的是"中文提问 + 英文语料"，
   反过来（英文提问 + 中文语料）没有专门处理。
4. **比较题受单查询检索限制**：一条查询同时拉两个实体时会被其中一个主导，
   目前靠"改写轮聚焦缺口 + 两轮证据合并"绕过，不是根本解法。
   真正的解法是查询分解（子问题各自检索再合并）。
5. **引用真实 ≠ 答案正确**：`verify` 只能证明"摘录确实来自原文"，
   不能证明"模型对这段原文的解读是正确的"。语义正确性需要人工或 LLM 评分，
   当前没有这项指标。
6. **拒答阈值是模型判断，不是分数阈值**：余弦分和 RRF 分都**判不了相关性**
   （RRF 只看排名，余弦分在同分布内可比、跨查询不可比）。
   所以拒答交给 grader 模型，代价是引入了不确定性。
