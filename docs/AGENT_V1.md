# Agent V1：RAG → Agent Tool

> 本文是 V1 交付快照。当前主线已升级到 V2，默认 8 轮并采用统一
> `success/data/error` Tool 信封；当前行为以 [AGENT_V2.md](AGENT_V2.md) 为准。

本文记录 V1 的代码审计、设计、调用链、启动与验收方法。V1 只增加
`PiAgent + search_knowledge + AgentHarness`，保留原有 RAG、LangGraph、Milvus、
PostgreSQL、摄取链路和前端协议。

## 1. 改造前代码审计

| 检查项 | 现有入口 | V1 的处理 |
|---|---|---|
| RAG 检索入口 | `RetrievalService.retrieve()` | 直接复用，工具不重写检索 |
| LangGraph 入口 | `build_graph()` / `run_agent()` | 保留完整旧图，不删除 |
| Milvus 入口 | `MilvusVectorStore.search_dense()` / `search_sparse()` | 仍由 RetrievalService 间接调用 |
| PostgreSQL 查询 | `PostgresRepository.get_chunks()` | 仍负责按 Milvus 命中 ID 回填权威正文 |
| Prompt | `rag/agent/prompts.py` | 原 RAG prompt 保留，新增独立 Agent prompt |
| FastAPI | `POST /api/v1/chat` | 请求/响应协议不变，内部改进 Harness |
| 纯检索 API | `POST /api/v1/retrieve` | 不变，仍直接调用 RetrievalService |
| 对话历史 | `get_messages()` 读取，`append_turn()` 写入 | 不变；Harness 收到最近历史消息 |
| 返回格式 | `ChatResponse`：answer/citations/verified/route 等 | 不变；Harness 结果转换成原 state 形状 |

原有 LangGraph 流程仍可单独调用：

```text
prepare → retrieve → grade → rewrite（可重试）→ generate/refuse → verify
```

`/chat` 的 V1 主流程改为：

```text
FastAPI /api/v1/chat
  → AgentHarness.run()
  → PiAgent.step()
  → 0..N 次原生 tool_calls
  → search_knowledge(query)
  → RetrievalService.retrieve()
  → Milvus 稠密/BM25 + RRF + rerank
  → PostgreSQL 回填正文
  → 结构化 Tool Result
  → PiAgent.step()
  → 引用校验
  → 原 ChatResponse
```

## 2. 为什么没有引入 Node sidecar

官方 `pi-agent-core` 是 TypeScript/Node 包，而本项目是单进程 Python/FastAPI 服务。
V1 若为它增加 Node sidecar、跨进程协议和第二套模型配置，会扩大部署面并违背“小改造”
目标。因此本项目增加了一个 Python `PiAgent` 适配层，保留 Pi 的关键执行契约：

1. 模型通过 OpenAI 兼容协议返回原生 `tool_calls`；
2. Harness 注册并执行工具；
3. 工具结果以 `role=tool` 和 `tool_call_id` 回填；
4. 模型再次判断，可结束，也可继续调用工具；
5. 达到最大迭代次数时安全终止。

它不是固定的 `LLM → Tool → Answer` 管道：问候可 0 次调用，单一事实可 1 次调用，
复合问题可多次改写查询并调用同一个工具。

## 3. 新增文件

| 文件 | 职责 |
|---|---|
| `src/rag/agent/agent.py` | `PiAgent`：组装 system/history/user 消息，请求模型选择 tool 或 final，并序列化 assistant tool call |
| `src/rag/agent/tools.py` | `search_knowledge(query)`、参数/结果模型和 `SearchKnowledgeTool`；只检索，不生成答案 |
| `src/rag/agent/harness.py` | 工具注册、上下文注入、0..N 次 loop、超时、错误回填、证据合并、最大迭代和统一输出 |
| `tests/test_agent_v1.py` | 覆盖 0 次、1 次、多次调用、学校规则护栏和知识缺失拒答 |
| `docs/AGENT_V1.md` | V1 审计、设计、运行、验收和已知限制 |

## 4. 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/rag/providers/base.py` | 增加 provider-neutral 的 `LLMToolCall`、`LLMChatResponse` 和 `achat` 协议 |
| `src/rag/providers/llm.py` | OpenAI 兼容客户端增加原生 tool calling；校验工具参数 JSON；NullLLM 保持 503 行为 |
| `src/rag/agent/prompts.py` | 增加 Agent system prompt、工具规则和最终回答规则；原 LangGraph prompt 保留 |
| `src/rag/agent/__init__.py` | 导出 V1 Agent、Harness 与 Tool |
| `src/rag/api/deps.py` | 启动时只构造一次 Tool、PiAgent 和 Harness；旧 graph 继续构造 |
| `src/rag/api/routes.py` | `/chat` 内部由 `run_agent()` 切到 `agent_harness.run()`；API schema 与 SSE 不变 |
| `src/rag/core/config.py` | 增加最大迭代与工具超时配置 |
| `.env.example` | 增加 V1 Agent 配置示例 |
| `README.md` | 更新架构、文档入口与 V1 测试命令 |

## 5. Tool Result

逻辑名称固定为 `search_knowledge`。公开函数签名为：

```python
await search_knowledge(
    query,
    retrieval=existing_retrieval_service,
    tenant_id=1,
    top_k=5,
)
```

给模型的结果示例：

```json
{
  "query": "学校 重修 规定",
  "results": [
    {
      "content": "……",
      "source": "学生手册 · 学籍管理 · p23",
      "page": 23,
      "page_end": 23,
      "section": "学籍管理",
      "score": 0.0317,
      "chunk_id": 812,
      "document_id": 15,
      "rank": 1,
      "dense_score": 0.61,
      "rerank_score": 0.98,
      "citation_index": 1
    }
  ],
  "diagnostics": {}
}
```

`score` 沿用现有 RRF 分数；`dense_score` 和 `rerank_score` 原样保留，避免混淆不同
量纲。`citation_index` 由 Harness 在多次检索合并后统一编号。

## 6. Loop、护栏和错误处理

- **0 次调用**：模型直接回答问候、闲聊或无需知识库的写作请求，`route=direct`。
- **1 次调用**：模型调用 `search_knowledge`，读取结果后回答，`route=generate`。
- **多次调用**：每次 Tool Result 都加入消息历史，模型可继续改写查询。
- **学校规定护栏**：模型若对学校规定、重修、毕业学分等问题误走直答，Harness 会
  强制先检索原问题，再把结果交回 Agent。这是 Prompt 规则的确定性兜底。
- **知识缺失**：检索后最终回答必须包含有效 `[n]`。无结果、无有效引用或只有不存在
  的编号时，Harness 返回原有确定性拒答，不传播模型猜测。
- **工具异常**：Tool Result 收到不含内部细节的错误对象，Agent 可重试；日志只记录
  异常类型，不记录 API Key、数据库 DSN 或文档正文。
- **最大迭代**：默认 4 次模型回合，耗尽后拒答，避免无限循环。

日志事件包括：

```text
[Agent] user request
[Agent] tool selected: search_knowledge
[Tool] search_knowledge started
[Tool] search_knowledge completed
[Agent] final response
```

## 7. 启动

环境和模型配置沿用原系统。新增配置有默认值，不修改 `.env` 也能启动：

```dotenv
AGENT_MAX_ITERATIONS=4
AGENT_TOOL_TIMEOUT_SECONDS=120
```

Windows：

```powershell
uv sync --extra dev
python scripts/run_api.py --port 8000
python -m rag.worker
```

Linux / AutoDL / Docker：

```bash
uv sync --extra dev
uvicorn rag.main:app --host 0.0.0.0 --port 8000
python -m rag.worker
```

或继续使用：

```bash
docker compose up -d --build
```

## 8. 测试

V1 单元验收：

```bash
uv run pytest tests/test_agent_v1.py
```

代码检查：

```bash
uv run ruff check src/rag/agent src/rag/providers/base.py \
  src/rag/providers/llm.py src/rag/api/deps.py src/rag/core/config.py tests/test_agent_v1.py
```

服务启动后的接口验收：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"学校重修有什么规定？","stream":false}'

curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"你好","stream":false}'

curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"毕业需要多少学分？","stream":false}'
```

预期：第一、第三条的 `diagnostics.tool_calls` 含 `search_knowledge`；第二条为空。
知识库中不存在的问题应返回 `route=refuse` 且 `citations=[]`。

## 9. V1 Demo

### 学校规定

```text
User: 学校重修有什么规定？
Agent: tool_call search_knowledge({"query":"学校 重修 规定"})
Tool:  返回学生手册的正文、页码、章节和分数
Agent: 根据学生手册……[1]
```

### 问候

```text
User: 你好
Agent: 你好！有什么可以帮你？
```

没有 Tool Call。

### 知识库缺失

```text
User: 火星交换生有多少住宿补贴？
Agent: tool_call search_knowledge(...)
Tool:  results=[]
Agent: 根据现有资料无法回答该问题。……
```

## 10. 当前问题和后续可优化项

1. **Pi 运行时边界**：V1 使用 Python 内的 Pi 风格核心契约，没有运行 TypeScript
   `pi-agent-core`。若以后必须与官方 Pi 的事件流、steering 或 extension 完全互操作，
   需要单独设计 Node sidecar 协议；这属于后续版本。
2. **证据充分性**：V1 延用确定性引用校验，能拦截空结果、无引用和越界引用，但还不能
   对每个自然语言主张做蕴含判定。后续可增加 claim-evidence verifier 和离线评测集。
3. **相关性阈值**：现有 RRF 分没有绝对相关性含义，不能直接当拒答阈值。后续应基于
   不同学科的标注集校准 rerank 分数或训练充分性判别器。
4. **流式首字延迟**：SSE 仍先跑完整 loop 再切片输出，协议兼容但不是模型原生流式。
5. **历史压缩**：V1 只复用最近历史消息，没有增加复杂 Memory；长会话仍需要摘要或
   token budget 管理。
6. **Tool 可观测性**：已有调用次数、查询和检索 diagnostics，但尚无 OpenTelemetry trace
   和分阶段延迟分布。
7. **真实环境验收依赖**：学校问题只有在知识库上传相应学校文件后才能生成有据答案；
   否则正确行为是拒答。不同学科应补各自的检索与拒答回归集。

这些都是 V1 之后的候选项，本次不自动进入 V2。
