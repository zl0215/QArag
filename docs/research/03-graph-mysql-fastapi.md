两个后台调研 agent 已完成，结合我自己的图谱侧调研，以下是完整报告。

---

# RAG-Agent 技术调研报告：知识图谱 + 元数据存储 + API/异步架构（2026-09）

> 调研方式：WebSearch 为主（本会话 200 次预算已用尽）。GitHub.com / pypi.org / neo4j.com 的直接抓取被网络策略阻断，部分结论来自搜索摘要与镜像站。**凡来源冲突或单一来源的，文中显式标注"未证实/冲突"**——这些点建议在 `pip index versions <pkg>` 与官方 release notes 上二次核实。

---

## 0. 结论速览（TL;DR）

| 决策点 | 推荐 | 核心理由 |
|---|---|---|
| Neo4j 版本 | **5.26 LTS**（补丁至 2028-06-06） | 稳定 > 新特性；CalVer 每版热修仅 4–6 周；5.26 是升级必经检查点 |
| GraphRAG 框架 | **`neo4j-graphrag`（官方）+ 自研编排** | 不要 MS GraphRAG（成本/无增量）；不要 LightRAG（AGE 依赖踩坑）；`langchain-neo4j` 只用于 LangGraph checkpointer |
| 图谱检索器 | **`VectorCypherRetriever`**（主力），**不用 Text2Cypher** | Text2Cypher 2026 实测 EM 仅个位数（见 §4.3） |
| 图 schema | `Document ←PART_OF- Chunk -NEXT_CHUNK→ Chunk`，`Chunk -MENTIONS→ Entity` | Neo4j 官方 `LexicalGraphConfig` 可重命名以对齐 |
| chunk 正文落点 | **MySQL 权威 + Milvus 只存向量/标量** | Milvus VARCHAR 上限 65535 字节、无 TEXT、无唯一约束 |
| 图社区检测 | **可选，MVP 不做** | 只在"全局摘要型"问题上有稳定收益；增量维护成本高 |
| MySQL | **8.4 LTS**（非 9.7） | 9.x 的 VECTOR 在 Community **无 ANN 索引**，对 RAG 无价值 |
| 异步驱动 | **asyncmy 0.2.14+**（`mysql+asyncmy://`） | 活跃维护 + 吞吐优于 aiomysql；注意 `caching_sha2_password` |
| 后台任务 | **Celery 5.6.3**（Linux）/ ARQ 0.28（Windows 友好但 maintenance-only） | 必须分队列分优先级；`BackgroundTasks` 绝对不可用于摄取 |
| 流式 | **SSE**（`sse-starlette` + `astream_events(version="v2")`） | LangGraph async generator 直通 `StreamingResponse` |
| 鉴权 | **PyJWT + pwdlib[argon2]** | python-jose 停更且有 CVE-2025-61152（接受 `alg=none`）；passlib 与 bcrypt 5.x 破裂 |
| 解析栈 | **pdfplumber(MIT) + python-docx + Docling 兜底** | **PyMuPDF 是 AGPL-3.0，闭源/SaaS 禁用** |
| 可观测 | **Phoenix 单容器起步**，需要 Prompt 管理再加 Langfuse v3 | Langfuse v3 需 ClickHouse（+4GB） |

---

# 1. Neo4j GraphRAG 生态选型

## 1.1 Neo4j 版本（**重要变更**）

Neo4j 已从 5.x 语义化版本**改为日历版本（CalVer，YY.MM）**：

- **5.26 LTS**：2024-12-06 发布，**5.x 系列最后一个版本**，热修支持到 **2028-06-06**，最新补丁 5.26.29/30（2026-08）。**只有安全补丁与 bug 修复，无新特性**（无 Cypher 25、无 `VECTOR` 原生类型、无 ABAC）。
- **CalVer 2026.x**：月度发布。2026.06.0（GA 2026-07-08）、**2026.07.1（2026-08-05，最新）**。**每个 CalVer 版本的热修只维持到下一个 minor 发布，实际生命期约 4–6 周**。截至 2026 年中，**没有任何 2025.x/2026.x 被指定为 LTS**（官方 release notes 已出现 "scheduled for after 2026.LTS" 的措辞，说明 CalVer LTS 在规划中）。
- 升级路径：**4.4 → 5.26 LTS → 2025.x/2026.x，不能跳**。升到 2025.01+ 必须 **Java 21**（Java 17 已删），集群发现服务 v1 已移除。
- **Community Edition 无保证支持，且缺 block format / 并行运行时**（这两项是 EE 特性）。

**推荐：5.26 LTS**。RAG 项目的图谱不是高频演进组件，稳定性 > 新特性；且它同时是未来的升级必经点。

**但有一个真实的取舍**：Cypher 25 的 `SEARCH` 子句（向量索引内联过滤）**只在 2026.01+ 可用**，`neo4j-graphrag` 1.16.0+ 的 `use_search_clause=True` 依赖它。5.26 上只能走 `db.index.vector.queryNodes()` 过程式查询。对中小项目这**不构成阻碍**（`queryNodes` 完全够用），但如果你明确要做"向量检索 + 复杂标量过滤"的深度优化，CalVer 的 `SEARCH ... WHERE` 有实质优势。

来源：[Neo4j 支持版本](https://neo4j.com/developer/kb/neo4j-supported-versions/)、[升级到 2025–2026 版本](https://neo4j.com/docs/upgrade-migration-guide/current/version-2025-2026/)、[Neo4j CalVer 中期盘点](https://www.youngju.dev/transcribe/2026-07-17-neo4j-calver-cypher-25-gql-block-format.zh)

## 1.2 `neo4j-graphrag`（官方包）

**版本状态（口径冲突，需实测）**：pyspect 列 **1.18.0（2026-06-24）**；mygit 快照（2026-03）列 **1.14.1（2026-03-25）**。以 `pip index versions neo4j-graphrag` 为准。包名 `neo4j-graphrag`，导入路径 `neo4j_graphrag.*`（旧包 `neo4j-genai` 已弃用）。要求 **Python ≥ 3.10**、**Neo4j ≥ 5.18.1**、driver ≥ 5.17.0（driver 6.x 支持）。

**核心组件清单**：

| 组件 | 路径 | 说明 |
|---|---|---|
| `SimpleKGPipeline` | `neo4j_graphrag.experimental.pipeline.kg_builder` | 封装建图全流程；支持 text 与 PDF；参数 `llm` / `driver` / `embedder` / `schema` |
| `Pipeline` | 同上 | 底层可定制版本 |
| `EntityRelationExtractor`（`LLMEntityRelationExtractor`） | `neo4j_graphrag.experimental.components.entity_relation_extractor` | 逐 chunk 抽取。v1.13.0 加 `use_structured_output` |
| `SchemaFromTextExtractor` | `experimental/components/schema.py` | **实验性**。LLM 从文本推导 `GraphSchema`。v1.13.0 加 `use_structured_output` |
| `LexicalGraphConfig` | — | **重命名标签/关系/embedding 属性名的关键**（见 §2.1） |
| `VectorRetriever` / `VectorCypherRetriever` | `neo4j_graphrag.retrievers` | `VectorCypherRetriever` = 向量命中后执行 `retrieval_query` Cypher 片段做图遍历 |
| `HybridRetriever` / `HybridCypherRetriever` | 同上 | Neo4j 向量索引 + 全文索引融合；`NAIVE` / `LINEAR`（HybridSearchRanker） |
| `Text2CypherRetriever` | 同上 | 自然语言 → Cypher，无需 embedder |
| `ToolsRetriever` | 同上 | **新**：把各 retriever 通过 `convert_to_tool()` 变成 LLM 可调用工具，由 LLM 动态路由（多 retriever 编排） |
| 三个 Resolver | `neo4j_graphrag.experimental.components.resolver` | 实体消歧（见 §3.4） |

**Schema-guided 抽取的三个宽松度**：`SimpleKGPipeline` 的 `schema` 参数接受 `GraphSchema`/dict，或字面量 **`"FREE"`**（不强制）与 **`"EXTRACTED"`**（自动推断）。约束分三层：属性级（`required`，缺失的实体被剔除）、节点/关系级（`additional_properties`）、图级（`additional_node_types` / `additional_relationship_types` / `additional_patterns`）。**schema 越详细默认越严格**。配套 `GraphPruning` 剔除违反 schema 的节点/关系，并做内置清理（空标签、无属性节点、孤立关系、关系方向修正）。

**⚠️ 结构化输出的硬限制**：`use_structured_output=True` **仅对 `OpenAILLM` 和 `VertexAILLM` 有效**（`supports_structured_output=True` 的 provider）；其他 provider 会抛 `ValueError`。且必须**不要**在 `model_params` / `generation_config` 里再传 `response_format`（extractor 在 `invoke()` 时自己设）。issue #493 → **Anthropic 用户目前必须保持 `use_structured_output=False`**。

**⚠️ Text2Cypher 安全（必须知道）**：**AIKIDO-2026-10787（2026-05-08 公布）**，影响 `neo4j-graphrag` **0.0.1–1.15.0**：`Text2CypherRetriever` 直接执行 LLM 生成的 Cypher，可被 prompt injection 诱导执行 `DETACH DELETE`。**1.16.0 修复**：所有生成 Cypher 先过 `EXPLAIN` 判定语句类型，非只读一律抛 `Text2CypherRetrievalError`，永不执行。类比的 Langroid 漏洞 **CVE-2026-55615 / GHSA-2pq5-3q89-j7cc（Critical）**：`Neo4jChatAgent` 无校验执行 LLM Cypher，配合 APOC 可达成 OS 命令执行；0.65.5 修复（默认 `allow_dangerous_operations=False` + 拒绝 `LOAD CSV` / `apoc.*` / `dbms.*` / `CALL db.*`）。

> **给你的规范**：**锁 `neo4j-graphrag >= 1.16.0`**（含 EXPLAIN 只读闸门）；即使不用 Text2Cypher，也应为图数据库建一个 **只读角色**（`GRANT MATCH ... DENY WRITE`）供 Agent 使用。

来源：[neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)、[Schema 能力博客](https://neo4j.com/blog/developer/unleashing-the-power-of-schema/)、[ToolsRetriever 介绍](https://neo4j.com/blog/developer/introducing-toolsretriever-graphrag-python-package/)、[AIKIDO-2026-10787](https://intel.aikido.dev/cve/AIKIDO-2026-10787)、[CVE-2026-55615](https://github.com/advisories/GHSA-2pq5-3q89-j7cc)

## 1.3 四家横向对比

| | **neo4j-graphrag** | **Microsoft GraphRAG** | **LightRAG** | **langchain-neo4j** |
|---|---|---|---|---|
| 定位 | 官方 SDK / 组件库 | 完整参考实现 | 轻量完整实现 | LangChain 集成层 |
| 图存储 | Neo4j（原生） | Parquet 文件（非图库） | Neo4j / AGE / 文件 | Neo4j |
| 社区检测 | **不含**（需自己接 GDS / leidenalg） | Leiden 内置 + 社区摘要 | **不做**社区检测 | 不含 |
| 索引成本 | 取决于你的模型选择 | **5–10× 源 token 膨胀，$4–7/文档**（来源冲突：另有 $50–200/万文档口径） | 号称比 MS GraphRAG 低 1–2 个数量级 | 取决于你 |
| 增量更新 | 需自研（推荐） | `update` 命令（title 为主键做 delta）；仍有"重建陷阱" | 一般 | 不涉及 |
| 查询成本 | 低（你可控） | **MS-GraphRAG ≈ 38,707 token/query；LightRAG ≈ 100,832**（Novel 语料） | 高 | — |
| 2026 评价 | 组件可靠、编排要自己做 | 全局归纳最强，但重、慢、Python-only | 多跳性价比高，但 Apache AGE 有事故记录（49B 行估算、407K 边迁移 17 小时），且**只支持无向图** | **内部依赖 `neo4j-graphrag`** |

**关键发现**：`langchain-neo4j` 从 **0.2.0 起把 `neo4j-graphrag` 加为依赖**，其内部的向量/全文索引查询、`Neo4jGraph` schema 获取、`Neo4jChatMessageHistory`、`GraphCypherQAChain` 用的 `construct_schema`、`extract_cypher` 都已换成 `neo4j-graphrag` 实现。**所以它不是替代品，是上层封装**。

> ⚠️ **`GraphCypherQAChain` 的 API 存在来源冲突**：官方文档仍列 `from langchain_neo4j import GraphCypherQAChain, Neo4jGraph` + `allow_dangerous_requests=True`；另有课程来源称 `from_llm()` 在 langchain 0.2.0 已移除、`allow_dangerous_requests` 不再是构造参数。**装哪个版本就在那个版本上验证，不要凭记忆写代码。**

## 1.4 ★ 中小项目推荐（明确）

**用 `neo4j-graphrag`（≥1.16.0）做抽取与检索组件，自己用 LangGraph 编排，不要引入 MS GraphRAG 或 LightRAG。**

具体分工：

1. **建图**：用 `SimpleKGPipeline` 起步（省 4–8 周；从零自研估计 12–16 周），但**必须自己补三件事**（`SimpleKGPipeline` 只做"同标签同名"的基础消歧）：
   - 分块层的中文分隔符改造（见 §3.5）
   - `FuzzyMatchResolver` / `SpaCySemanticMatchResolver` 做二次消歧
   - 增量更新与删除（§3.6）
2. **检索**：`VectorCypherRetriever` 为主力（Milvus 命中 → Neo4j 补图上下文），这与你的"Milvus 主检索 + Neo4j 补关系"架构天然对齐。
3. **`langchain-neo4j` 只用于一件事**：`Neo4jSaver` / `AsyncNeo4jSaver` 做 **LangGraph checkpointer**（如果你的会话状态要放 Neo4j）。其余场景直接用 `neo4j-graphrag` + `neo4j` driver，少一层抽象。
4. **MS GraphRAG 只在一种情况下考虑**：你的核心场景是"对语料做全局归纳/主题摘要"，且语料**稳定不变**、预算充足。它的增量更新至今是"重建陷阱"。
5. **LightRAG 的 AGPL 之外的坑**：Apache AGE 依赖导致的事故（49B 行估算、17 小时迁移）与**只支持无向图**，对需要 `PART_OF`/`NEXT_CHUNK` 有向结构的文档图谱是硬伤。

来源：[GraphRAG 实现指南](http://www.ideabosque.com/library/graphrag-implementation-guide-knowledge-graph-retrieval-production/)、[LeanKG 竞争研究 2026-08](https://github.com/freepeak/leankg/blob/b7f64728b393e84547966aa445e2dfb9df14d13c/docs/analysis/leankg-competitive-research-and-improvement-strategy-2026-08-02.md)、[pg-raggraph 实战复盘](https://github.com/yonk-labs/pg-raggraph/blob/374a7d5401550f0d1ddb63e92154eab9808d76bd/docs/blog-what-we-learned.md)、[langchain-neo4j 0.2.0 changelog](https://data.safetycli.com/packages/pypi/langchain-neo4j/changelog?page=2)、[LangChain Neo4j 集成文档](https://docs.langchain.com/oss/python/integrations/providers/neo4j)

---

# 2. 文档图谱 Schema 设计

## 2.1 三层词法图（Lexical Graph）

社区与 Neo4j 官方一致的推荐模型：

```
(:Document {doc_id, title, source_uri, tenant_id, version, content_hash})
      ↑
   [:PART_OF]
      |
(:Chunk {chunk_id, text, embedding, chunk_index, page_no, section_path, content_hash})
      ↓ [:NEXT_CHUNK]
(:Chunk) ...

(:Chunk)-[:MENTIONS {count, confidence}]->(:Entity {entity_id, name, label, embedding})
(:Entity)-[:RELATES_TO {type, weight, evidence_chunk_ids}]->(:Entity)
```

**关键约定：**
- **原文与 embedding 只放 `Chunk` 节点**；`Document` 只承载元数据与溯源。
- **方向统一**：官方 `SimpleKGPipeline` 默认是 `(Document)-[:FROM_DOCUMENT]->(Chunk)` 与 `(Chunk)-[:FROM_CHUNK]->(__Entity__)`；社区常见写法是 `(Chunk)-[:PART_OF]->(Document)` 与 `(Chunk)-[:MENTIONS]->(Entity)`。**选一套并全局一致**，因为 `VectorCypherRetriever` 的 `retrieval_query` 直接依赖方向。
- 用 **`LexicalGraphConfig`** 把官方默认名改成你自己的：

```python
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphConfig
config = LexicalGraphConfig(
    document_node_label="Document",
    chunk_node_label="Chunk",
    chunk_to_document_relationship_type="PART_OF",   # 默认 FROM_DOCUMENT
    node_to_chunk_relationship_type="MENTIONS",      # 默认 FROM_CHUNK
    next_chunk_relationship_type="NEXT_CHUNK",
    chunk_embedding_property="embedding",
)
```

## 2.2 把图实体链回 Milvus chunk id（核心问题）

**推荐：用 MySQL 的 `chunks.id`（BIGINT）作为三库共享的 `chunk_id`，把它同时写进 Neo4j 的 `Chunk.chunk_id` 与 Milvus 的主键字段。**

理由与做法：

- `chunk_id` 是**单一整数**，三处冗余存储成本可忽略，换来的是**零成本跨库 JOIN**：
  - Milvus 命中返回 `chunk_id` → 直接 `MATCH (c:Chunk {chunk_id: $id})` 拿图上下文
  - 图遍历得到 `chunk_id` 列表 → `SELECT id, content FROM chunks WHERE id IN (...)` 做 hydration，或直接回 Milvus 做向量相似度
- **不要用 `elementId()` 或 Neo4j 内部 id 做跨库键**（`elementId` 在重建后会变，内部 id 会复用）。
- **Milvus 主键必须显式指定，不要用 `autoID`** —— 这是幂等 upsert 与精确删除的前提。
- 给 `Chunk` 建**唯一约束**（见 §2.3），让 `MERGE (c:Chunk {chunk_id: $id})` 天然幂等。
- 实体侧同理：`Entity.entity_id` 建议用 **`sha256(label + normalized_name)` 的前 16 字节 hex**（而非 UUID），这样**跨文档同名实体自动落到同一节点**，天然完成一部分实体解析。

## 2.3 约束与索引（可直接抄的 Cypher）

```cypher
// ---------- 唯一性约束（Community 可用）----------
CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

// 关系唯一性（Neo4j 5.7+）
CREATE CONSTRAINT mentions_unique IF NOT EXISTS
FOR ()-[m:MENTIONS]-() REQUIRE m.id IS UNIQUE;

// 复合唯一（多租户）
CREATE CONSTRAINT chunk_tenant_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE (c.tenant_id, c.chunk_id) IS UNIQUE;

// ---------- 向量索引（Neo4j 5.13+，Community 可用）----------
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

// ---------- 全文索引（混合检索必需）----------
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text, c.title];

// 可选：调中文分析器（先列出可用分析器）
// CALL db.index.fulltext.listAvailableAnalyzers();

// ---------- 辅助范围索引 ----------
CREATE INDEX doc_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);
CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id, c.document_id);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

**必须知道的版本事实：**

| 特性 | 5.26 LTS | Community 可用？ |
|---|---|---|
| `IS UNIQUE` 约束 | ✅ | ✅（由 range 索引支撑） |
| **`IS NOT NULL`** | ✅ | ❌ **仅 EE** |
| **`IS NODE KEY`**（唯一+存在） | ✅ | ❌ **仅 EE** |
| 向量索引 | ✅（5.13+） | ✅ |
| 全文索引（Lucene） | ✅ | ✅ |
| **`SEARCH` 子句（向量索引内联过滤）** | ❌ | 需 **2026.01+** |
| 向量索引多属性/多标签过滤 | ❌ | 需 **2026.01+** |
| 原生 `VECTOR` 属性类型 | ❌ | 需 2025.10+ |

- `vector.dimensions` 自 5.23 起非强制，但**强烈建议显式写**（避免维度不匹配的诡异报错）。
- **无 `ALTER VECTOR INDEX`**：改维度/相似度函数必须 **DROP + CREATE + 重新生成全部 embedding**。
- 5.26 中 `indexProvider` 指定已弃用，**省略即用默认 `vector-2.0`**。
- 建完索引**必须等 ONLINE** 再查：

```cypher
SHOW VECTOR INDEXES YIELD name, state, populationPercent, indexConfig;
// 需 state='ONLINE' 且 populationPercent=100
```

- 查询（5.26 语法）：

```cypher
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $query_vector)
YIELD node, score
MATCH (node)-[:PART_OF]->(d:Document)
WHERE d.tenant_id = $tenant_id AND d.is_active = true
RETURN node.chunk_id AS chunk_id, node.text AS text, score
ORDER BY score DESC;
```

> ⚠️ 5.26 中向量索引是**单标签单属性**的。租户过滤只能在 `queryNodes` **之后**做 `WHERE` —— 这会让 top-k 被跨租户数据稀释。**缓解方案**：(a) 用 2026.01+ 的 `SEARCH` 内联过滤；(b) 多租户强隔离时按租户分 label（不优雅）；(c) **主要靠 Milvus 做带过滤的向量检索，Neo4j 只做图补全**（推荐，与你的架构一致）。

## 2.4 GDS 社区检测（可选）

**Community 版 GDS 包含全部算法，但限制 4 CPU 核**（并发上限 4）；EE 版无限核 + RBAC。Docker 安装：

```bash
--env NEO4J_PLUGINS='["apoc","graph-data-science"]'
```

- **官方明确：`NEO4J_PLUGINS` 自动下载只适合开发环境，生产不推荐**。生产应把 JAR 挂到 `/plugins`。许可证文件挂 `/licenses`。
- 插件名不能写 `"gds"`（会报 `'gds' is not a known Neo4j plugin`）。
- **5.26.x 对应 GDS 2.13.3**（历史上 5.26.3/5.26.4 因 `versions.json` 缺条目而报 `No compatible "graph-data-science" plugin found`，已修复）。安装后 `RETURN gds.version();`、`CALL gds.list();` 验证。

```cypher
CALL gds.graph.project('entities','Entity',{RELATES_TO:{orientation:'UNDIRECTED'}});
CALL gds.leiden.write('entities', {
  writeProperty: 'communityId',
  gamma: 0.5,          // 0.1–1.0：越低社区越大越宏观
  includeIntermediateCommunities: true
});
CALL gds.graph.drop('entities');
```

**但我的建议是 MVP 阶段不做社区检测**（见 §4.4 的成本-收益分析）。

来源：[Neo4j 向量索引手册](https://neo4j.com/docs/cypher-manual/25/indexes/semantic-indexes/vector-indexes/)、[约束语法](https://neo4j.com/docs/cypher-manual/5/constraints/create-constraints/)、[GDS 安装](https://neo4j.com/docs/operations-manual/current/docker/plugins/)、[GDS Community 4 核限制](https://neo4j.com/deployment-center/?gds-selfmanaged)

---

# 3. 实体/关系抽取

## 3.1 Schema-guided vs 开放抽取

**推荐：对生产用"半开放"——先 `SchemaFromTextExtractor` 从语料抽一版 schema，人工审阅裁剪后再固定下来用。**

- **纯 `"FREE"`**（不强制 schema）：抽取自由度最高，但**图会迅速腐化**——同一个语义被抽成十种关系类型，`RELATES_TO` 的 type 属性发散，检索时无法写稳定的 Cypher。
- **纯固定 schema**：稳定但会漏掉语料里的重要关系（schema 是人猜的）。
- **折中流程**（推荐）：
  1. 取 20–50 篇代表性文档跑 `SchemaFromTextExtractor`（**需 `use_structured_output=True`，即必须用 OpenAI/VertexAI，或用 Anthropic 时手工 prompt**）
  2. 人工把 node types 收敛到 10–25 个、relationship types 收敛到 15–40 个
  3. 固定为代码里的常量 schema 传给 `SimpleKGPipeline`
  4. 用 `additional_node_types="allow"` 保留一定开放性，并**定期统计"未命中 schema 的类型"**决定是否扩充
  5. 用 `GraphPruning` + `required` 属性剔除脏数据

**schema 严格度三档的取舍**：

| 档位 | 配置 | 结果 |
|---|---|---|
| 严 | `required` 属性多 + `additional_*=deny` | 图干净、召回低、大量实体被丢 |
| **中（推荐）** | 少量 `required`（如 name）+ `additional_node_types="allow"` | 平衡 |
| 松 | `"FREE"` | 脏 |

## 3.2 抽取 Prompt 的工程要点

`neo4j-graphrag` 的 `ERExtractionTemplate` 在迭代中持续加严了"必须返回合法 JSON"的指令，v1.13.0 还修了"LLM 返回合法 JSON 数组时 extractor 失败"的 bug。**你自己的 prompt 必须显式覆盖**：

1. **指代消解规则写进 prompt**（社区标准做法）：*"当实体以代词、简称、头衔等形式出现时，一律使用最完整的标识符（`John Doe` 而非 `Joe`/`he`）"*。
2. **实体名归一化**：明确要求"使用全称而非缩写"、"去掉冠词"、"保留原始语言（中文用中文名，不要翻译）"。
3. **关系方向**：给出 2–3 个具体示例（few-shot 对关系方向的作用远大于对实体识别的作用）。
4. **禁止编造**：明确"只抽取文本中明确陈述的关系，不要推断"。
5. **限制数量**：`MAX_TRIPLETS_PER_CHUNK`（如 30）防单 chunk 爆炸。
6. **时间与数值**：明确指出是否需要抽时间表达式，以及是否归一化。

## 3.3 指代消解（Coreference）

**结论：放在抽取之前，且必须用大 chunk。**

- **位置**：多个 2025/2026 实现（`llm4s` 的 `CoreferenceResolver`、FalkorDB 的 GraphRAG SDK 可选阶段）都把指代消解作为**抽取前**的预处理。理由：未消解的指代会**直接制造重复节点**。
- **两种实现**：
  - **LLM 重写**（推荐，中文友好）：temperature=0，指令"重写以下文本，把代词和间接引用替换为明确的实体名称，保持含义与结构完全一致，若无引用则原样返回"。
  - **NLP 管线**（spaCy + 指代模型）：英文成熟，**中文需要专门的中文指代消解模型**，工程成本高。
- **chunk 必须够大**：实体与其指代**必须在同一 chunk 内**，否则 LLM 无从链接。参考值：**2048 token + 24 token overlap**（tomasonjo/Neo4j 官方示例口径）。这与"中文 RAG 通常用 400–512 token"的检索最优值**冲突**。
- **解法：解密耦合** —— 抽取用大窗口（如 1024–2048 token），检索用小块（400–512 token）。即**抽取 chunk ≠ 检索 chunk**，通过 `(ExtractionChunk)-[:CONTAINS]->(RetrievalChunk)` 关联。或者在抽取时把**前一个 chunk 的末尾 N 个 token 作为上下文前缀**注入（更省 token，且不破坏检索分块）。
- **跨 chunk/跨文档一致性**：`LycheeMemory V2`（2026 预印本）的做法是把**消歧状态**（已解析的别名、规范实体名、引用关系）在分段间传递，避免 prompt 无限膨胀。可借用此模式维护一个"文档级别名表"。

## 3.4 实体消歧与合并

`neo4j-graphrag` 提供三个 resolver，**全部只在同一 label 内、默认比 `name` 属性**：

| Resolver | 算法 | 依赖 | 阈值建议 |
|---|---|---|---|
| **`SinglePropertyExactMatchResolver`** | 完全相等；用 `apoc.refactor.mergeNodes` 合并并重定向关系 | APOC | — |
| **`FuzzyMatchResolver`** | RapidFuzz 编辑距离 | `pip install "neo4j-graphrag[fuzzy-matching]"` | 高精度 >0.95 |
| **`SpaCySemanticMatchResolver`** | spaCy 向量余弦 | `pip install "neo4j-graphrag[nlp]"`，默认 `en_core_web_lg` | **0.85–0.95 平衡；0.95–1.0 严格；0.70–0.85 宽松；<0.70 不推荐** |

- **`SinglePropertyExactMatchResolver` 在 `SimpleKGPipeline` 中默认自动执行**（`perform_entity_resolution=True`）。
- **链式使用，从精确到宽松**：Exact → Fuzzy → Semantic。
- ⚠️ **过程是破坏性的**：原节点被删除替换（关系保留）。
- ⚠️ **spaCy 的 `nlp` extra 在 Python 3.14 上不支持**（上游 spaCy 问题）→ 用 3.13 或更低。
- ⚠️ **中文场景 spaCy 几乎无用**：`en_core_web_lg` 对中文无效，中文向量模型不在 spaCy 生态内。

**★ 中文项目的推荐消歧方案（自研，替代/补充官方 resolver）**：

```
1. 精确匹配：normalize(name) 相同 → 直接合并（用 entity_id = sha256(label+normalized_name) 天然实现）
2. 别名表：文档级 LLM 抽取阶段产出 {别名 → 规范名}，写入 Alias 节点或别名属性数组
3. 向量相似：用你自己的中文 embedding 模型（bge-m3 / Qwen3-Embedding）对 Entity.name 建 Milvus 索引
   → 相似度 > 0.92 且同 label → 候选
4. 阻断（blocking）：只在"label 相同 + 首字相同 / 共享 ≥1 个邻居实体"的候选对内比，避免 O(n²)
5. LLM 仲裁：仅对候选对（每天几十~几百对）用 LLM 判定"是否同一实体"，结果写审计表
6. 人工复核队列：合并前先落 `entity_merge_candidates` 表，高价值实体人工确认
```

**绝不要做无监督的全量自动合并** —— 一个错误的合并（"苹果公司" ↔ "苹果"水果）会污染整张图且难以回滚。

## 3.5 成本控制（这是最容易失控的地方）

**基准事实**：
- **KG 抽取占整个摄取流程 80%+ 的时间**（Flexible GraphRAG 实测）。`SKIP_GRAPH=true` 可让摄取快 ~90%。
- 模型分层：抽取用 `gpt-4o-mini` 级别，生成用强模型 → **索引成本降 60–70%，召回只掉 1–5%**。具体案例：100 篇 ×1500 token 从 **$14.85 → $4.22**；10,000 篇从 **$1,485 → $422**。

**七条必须落地的控制手段**：

1. **按 chunk hash 缓存（最重要）**。用 SQLite / 文件型 KV 存 `prompt_hash → LLM response`。GraphRAG 1.0 内置的磁盘缓存就是此机制。**prompt 模板一变，所有缓存失效** → 所以 prompt 必须**版本化**（`PROMPT_VERSION` 参与 hash）。
2. **`content_hash` 门控**：只有 hash 变化才重跑抽取。
   ```python
   content_hash = sha256(
       normalize(text) + "\x1f" + CHUNKER_CONFIG_VERSION
       + "\x1f" + EXTRACTION_PROMPT_VERSION + "\x1f" + LLM_MODEL_ID
   ).hexdigest()
   ```
3. **模型分层**：抽取/消歧用 mini 级，问答用强模型。
4. **chunk 尺寸右调**：太大浪费 token，太小丢跨句关系。抽取用 1024–2048 token 是常见甜点。
5. **`MAX_TRIPLETS_PER_CHUNK`** 限流：防止某个 chunk 抽出 200 条关系。
6. **并发 + 批量**：抽取是 IO bound，用 `asyncio.Semaphore(8–16)` 控并发；批量 embedding 优于逐条。
7. **成本可观测**：记录每个文档的 `prompt_tokens` / `completion_tokens` / `cost_usd`，写入 `ingestion_jobs` 表。**设置单文档成本上限**，超限告警而非静默烧钱。

**成本量级参考**：中型语料（数千文档）的 GraphRAG 索引成本在**数百到数千美元**量级。**这个数字要写进项目立项文档**，不要等到账单出来才发现。

## 3.6 增量更新与删除

**2026 年技术演进路线**：Microsoft GraphRAG 的 `update` 命令（title 为主键做 delta，合并实体/关系/社区）→ **hash/生命周期门控的按文档子图管理（Jigsaw-LightRAG 模式）**。

**Jigsaw 模式的核心思想（推荐采用）**：

```
每篇文档维护状态：New / Modified / Persistent / Deleted
  Hash(d_rev) ≠ Hash(d_curr) 才重新分块 + 抽取
  New / Modified → 重新抽取，替换该文档子图
  Persistent     → 完全跳过（零 LLM 调用）
  Deleted        → 从子图池移除（零 LLM 调用）
全局图重建 = 纯代码级聚合 + 去重（零 LLM 调用）
```

**删除实现（Neo4j 侧）**：

```cypher
// 删除文档及其全部 chunk，但保留被其他文档引用的实体
MATCH (d:Document {doc_id: $doc_id})-[:PART_OF|FROM_DOCUMENT*0..1]-(c:Chunk)
// 注意：实际应写成 (d)<-[:PART_OF]-(c)
WITH d, collect(DISTINCT c) AS chunks
UNWIND chunks AS c
DETACH DELETE c
WITH d
DETACH DELETE d;

// 清理孤儿实体（不再被任何 chunk 提及）
MATCH (e:Entity)
WHERE NOT (e)<-[:MENTIONS]-() AND NOT (e)<-[:FROM_CHUNK]-()
  AND e.updated_at < datetime() - duration('PT1H')   // 留出并发写入窗口
DELETE e;
```

**删除实现（Milvus 侧）**：

```python
client.delete(collection_name="rag_chunks", ids=chunk_ids_batch)  # 分批 1000
```

⚠️ **Milvus 的 delete 是软删**，空间回收依赖 compaction —— 容量规划时**不能按"已删即释放"计算**。

**⚠️ 删除的三条铁律**：
1. **孤儿实体清理必须延迟执行**（留 1 小时窗口），否则并发摄取时会把另一个文档刚建的实体删掉。
2. **共享实体的 `MENTIONS` 边要带 `chunk_id` 溯源**，删除某文档时只删它的边，不删节点。
3. **社区摘要的重建是真正的难点** —— 删一篇文档可能让多个社区摘要失效。这是"**不做社区检测**"建议的另一个理由（§4.4）。

来源：[Neo4j 实体解析文档](https://neo4j.com/docs/neo4j-graphrag-python/current/)、[DeepWiki 实体解析](https://deepwiki.com/neo4j/neo4j-graphrag-python/4.5-entity-resolution)、[MS GraphRAG 增量索引](https://deepwiki.com/microsoft/graphrag/4.7-incremental-indexing-and-updates)、[Jigsaw-LightRAG 论文](https://www.sciengine.com/AI4sci/doi/10.1088/3050-287X/ae4a3e)、[抽取成本优化](https://theneuralbase.com/graphrag/learn/advanced/cost-optimization-levers/)、[Flexible GraphRAG](https://deepwiki.com/stevereiner/flexible-graphrag/9-advanced-topics)、[GraphRAG 最佳实践](https://kindatechnical.com/knowledge-graphs/graphrag-best-practices-and-common-pitfalls.html)

---

# 4. 检索融合

## 4.1 分层架构

```
Query
  │
  ├─ ① 查询理解 / 改写（LLM，可选）
  │     → 抽取关键词 + 实体 + 时间/类型过滤条件
  │
  ├─ ② 并行多路召回（asyncio.gather + 超时）
  │     ├─ Milvus dense（带 tenant/collection/doc 过滤）      → top 50
  │     ├─ Milvus BM25 sparse（Function + SPARSE_WAND）       → top 50
  │     ├─ Neo4j 实体命中 → 1~2 跳图遍历 → 关联 chunk_id       → top 30
  │     └─ MySQL 结构化过滤（时间/标签/权限）→ 允许的 doc_id 集合
  │
  ├─ ③ RRF 融合（自研或 Milvus 内建）                        → top 30
  │
  ├─ ④ 重排（bge-reranker-v2-m3 或 Qwen3-Reranker-0.6B）     → top 5~8
  │
  └─ ⑤ 生成（带 citation）
```

**关键设计：MySQL 过滤条件必须先转成 Milvus 的 `expr` 与 Neo4j 的 `WHERE`，而不是检索后过滤。** Milvus 的标量过滤是**在 ANN 搜索过程中执行的**（不是 post-filter），后过滤会让 top-k 被过滤掉大半。

## 4.2 RRF 融合

**Milvus 内建（2.6.x，最省事）**：

```python
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker, Function, FunctionType

dense_req = AnnSearchRequest(
    data=[query_dense], anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=50, expr=f'tenant_id == {tid} and is_active == true',
)
sparse_req = AnnSearchRequest(
    data=[query_sparse], anns_field="sparse_vector",
    param={"metric_type": "BM25"}, limit=50, expr=...,
)
res = client.hybrid_search(
    collection_name="rag_chunks",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),          # k 默认 60，有效 (0,16384)，推荐 [10,100]
    limit=30,
    output_fields=["chunk_id", "document_id"],
)
```

- **`RRFRanker` 不需要调权重，是通用首选**（来源的排名位置而非原始分数，跨模态可比）。
- **`WeightedRanker(0.7, 0.3)`** 在你知道某路更重要时更好，但需要调参。
- 2.6+ 可用 **Function API**（`Function(name="weighted", ...)`）配置。
- BM25 稀疏向量可**自动生成**：schema 里 VARCHAR 字段设 `enable_analyzer=True`（中文需指定 analyzer），加 `Function(FunctionType.BM25, input_field_names=["content"], output_field_names=["sparse_vector"])`，插入时只需提供文本。
- 稀疏索引选 `SPARSE_WAND` 或 `SPARSE_INVERTED_INDEX`，`metric_type="BM25"`。
- **混合检索需 Milvus 2.5+**；2.4.x 会报错。

**跨源融合（Milvus + Neo4j + MySQL，需自研）**：

```python
def rrf_fuse(rankings: list[tuple[list[int], float]], k: int = 60) -> list[tuple[int, float]]:
    """rankings: [(id_list_ordered_by_rank, weight), ...]"""
    scores: dict[int, float] = {}
    for ids, w in rankings:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    # 关键：分数先量化再排序，tie-break 用 doc_id（保证跨进程确定性）
    return sorted(scores.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
```

**⚠️ 三个必须在代码里防住的坑**（这些是属性测试能直接抓到的不变量，见 §8.5）：
1. **浮点求和的非结合性**会让理论上相等的分数出现微小差异 → **必须先 `round(score, 9)` 再排序**。
2. **tie-break 用 `dict`/`set` 迭代顺序** → Python hash 随机化会让**同一输入在不同进程产生不同顺序**。**必须用 `(-score, doc_id)` 显式 tie-break**。
3. 自研 RRF 的 `k` 语义必须与 Milvus 侧对齐（都用 60），否则融合结果不可复现。

## 4.3 Local / Global / Text2Cypher —— 诚实的评估

| 模式 | 何时用 | 2026 实测效果 | 建议 |
|---|---|---|---|
| **Local（向量 + 1~2 跳图）** | 实体级、事实级问题 | **最稳**，图补全有效 | ✅ **主力** |
| **Global（社区摘要）** | 语料级归纳、主题摘要 | 优势最明显（Novel summary: HippoRAG2 64.10 vs basic RAG 51.30） | ⚠️ 只在有明确需求时做 |
| **Text2Cypher** | 结构化聚合/统计 | **很差**，见下 | ❌ **中小项目不要用** |

**Text2Cypher 的残酷现实（2026 论文数据，必须知道）**：

KG2Cypher（arXiv 2606.27742）用 gpt-oss-120B 测试：

| 配置 | EM | F1 |
|---|---|---|
| 仅规则 | 0.0 | 0.0024 |
| 规则 + 5 个 few-shot | 0.5 | 0.0049 |
| 规则 + few-shot + **完整 schema** | **4.2** | 0.0655 |
| Oracle 金标关系/实体 | 47.7 | 0.7139 |

**执行成功率 ~100%，但语义正确率接近 0** —— 模型"学会了可运行的查询形式，而不是正确的图接地"。典型失败：幻觉私有实体 URI、看似合理但错误的关系、错误的字面量子字段。

其他佐证：
- PIPE-Cypher（arXiv 2606.08481）：few-shot 的提升可能反映**泄漏**（同类别样例共享查询签名），需做"无签名"对照。**执行有效性不保证语义正确性**。
- CyVerACT（2026, IP&M）：按问题**裁剪 schema** + 执行反馈迭代，语法有效性最高 +52.7%、EM +13.5%。仍远不够生产。
- 一个扫描发现 Neo4j Labs 的 text2cypher 数据集中 **597/13939（4.3%）的 Cypher 本身有语义错误**（反向关系 198、计数错误 180、属性错误 160、schema 幻觉 59）。
- 领域特定 KG 上通用 Cypher 调优模型表现**最差**——难点是领域语义（关系选择、遍历方向、多跳），不是语法。

> **明确建议**：中小项目**不要上 Text2Cypher**。需要结构化查询时，用 **`ToolsRetriever` + 预置的 5–15 个参数化 Cypher 模板**（LLM 只负责选模板 + 填参数），正确率与可维护性都远高于自由生成。这本质是 "tool calling > code generation" 的老结论。

## 4.4 图到底值不值这个钱？（必须诚实）

**ICLR 2026 的 GraphRAG-Bench 是目前最公平的评测**（因为它用双密度语料：NCCN 医疗指南的显式层级 vs 20 世纪前小说，并分四级任务：事实检索 → 复杂推理 → 上下文摘要 → 创造性生成）。

**图没用的场景（证据）**：

- **简单事实检索：vanilla RAG 打平甚至赢**。Level 1 上基本 RAG（带 rerank）与图方法持平（Novel 语料 60.92 vs HippoRAG2 60.14），**图扩展引入的是噪声而非召回**。基本 RAG 在 Level 1 的检索召回 83.2%。
- **历史数据**：GraphRAG 在 Natural Questions 上**低 13.4%**、时效性查询上**低 16.6%**；HotpotQA 推理深度只提升 4.5%，但**延迟 2.3×**。
- **pg-raggraph 的诚实复盘**：在 NTSB 航空报告上，图**只赢了 5 个测试问题中的 1 个**，并且在**最该赢的技术文档查询上输了 8 个百分点**。图优先模式比朴素向量检索 **recall@10 低 75pp**（因为它必须先对问题做实体识别才能启动遍历，弱 NER 查询直接失败）。作者结论是"图在这些语料上不值它的成本"，而不是"图有害"。
- **一个反面教材**：pg-raggraph 曾发布"naive 0.74 vs graph 0.40"的 A/B 结果，后**撤回**——因为测试图有 2288 个实体但 **0 条关系**，测的是空的边表。**任何"图有用/无用"的结论都必须先验证边表非空。**

**图真正赚钱的场景（证据）**：

- **多跳关系推理**：Novel 复杂推理 HippoRAG2 53.38 vs 基本 RAG 42.93；Medical 61.98 vs 58.64。Level 2–3 上 HippoRAG 检索召回跳到 87.9–90.9%。
- **语料级/上下文摘要（最一致的优势）**：Novel summary 64.10 vs 51.30。
- **创造性生成的忠实度**：Medical LightRAG 78.76 vs 基本 RAG 36.74（但覆盖率有取舍）。
- **深度遍历**：正确实现的递归遍历（`WITH RECURSIVE ... depth < max_hops`）在 MuSiQue 上**比向量高 4–5pp**。
- **图密度是中介变量**：平均度 8.75（HippoRAG2）这类高密度图召回更好。**图构得好不好，直接决定图有没有用。**

**成本侧（"图税"）**：

| | 每 query token |
|---|---|
| MS-GraphRAG | ≈ 38,707 |
| LightRAG | ≈ 100,832 |
| HippoRAG2 | ≈ 1,008 |

延迟（pg-raggraph harness）：naive p50 **51ms** vs graph_leg **105ms** vs hybrid **~110ms**。

**★ 对你们项目的具体建议（这是最重要的决策）**：

1. **先建强基线**：好的分块（中文递归 + 400–512 token + 15% overlap）+ **混合检索（dense + BM25）+ rerank**。**先证明这个基线在你的真实问题上不够用**。GraphRAG-Bench 明确说：基础 RAG 在 Level 1 上"holds its own or wins"。
2. **建"图收益评测集"**：从真实用户问题里挑 **50–100 个跨文档/多跳/关系型问题**，标好 gold chunk。**测量"加图 vs 不加图"的 recall@k 与答案正确率**。如果提升 < 3pp，**不要上图**。
3. **渐进式引入**：
   - **阶段 1**：只把图作为**"相关 chunk 扩展器"**（Milvus 命中 → `MENTIONS` → 共现 chunk），不做社区检测、不做 Text2Cypher。成本最低，收益最直接。
   - **阶段 2**：加实体级向量检索（`entity_embedding` 索引），做"实体命中 → 邻居 chunk"。
   - **阶段 3**：只有明确需要"全局归纳"时才加 GDS Leiden + 社区摘要。**社区摘要是增量维护最贵的部分**（删一篇文档可能让多个摘要失效）。
4. **不要相信"图 = 更好"的营销**。现实收益是 **+3–5pp，且只在特定问题形态上**。
5. **图的真正杀手级价值不在检索，而在**：(a) **可解释的引用溯源**（答案 → 实体 → 关系 → chunk）；(b) **多跳问题的推理路径可视化**；(c) **结构化过滤**（"找出所有与 X 有合作关系且在 2024 年后成立的实体"）。**如果你们的产品不需要这三样，图的 ROI 会很差。**

来源：[GraphRAG-Bench / ICLR 2026](https://mlanthology.org/iclr/2026/xiang2026iclr-use/)、[论文笔记](https://en.papernotes.org/ICLR2026/information_retrieval/when_to_use_graphs_in_rag_a_comprehensive_analysis_for_graph_retrieval-augmented/)、[pg-raggraph 复盘](https://github.com/yonk-labs/pg-raggraph/blob/374a7d5401550f0d1ddb63e92154eab9808d76bd/docs/blog-what-we-learned.md)、[ML Digest: 何时用 KG-RAG](https://ml-digest.com/when-to-use-kg-rag/)、[KG2Cypher](https://arxiv-org.ezproxy.obspm.fr/html/2606.27742v1)、[PIPE-Cypher](https://huggingface.co/papers/2606.08481)

## 4.5 重排模型选型（中文）

| 模型 | 参数量 | BEIR nDCG@10 | 中文 | 许可 | 备注 |
|---|---|---|---|---|---|
| **bge-reranker-v2-m3** | ~0.6B | ≈71.5 | 稳 | MIT/Apache（口径不一，**以官方模型卡为准**） | **下载量最大的开放 reranker**；生产占比约 28%；中英混合最佳 |
| **Qwen3-Reranker-0.6B** | 0.6B | ≈71.4 | **强** | Apache/Tongyi（口径不一） | 支持 instruction 定制；延迟 45ms vs BGE 52ms（同测试集） |
| Qwen3-Reranker-8B | 8B | ≈77.0 | 最强 | 同上 | L40S 单对 38ms；32K 上下文 |

**推荐**：**`bge-reranker-v2-m3` 作为稳定基线**，自建 100–500 条真实 query 的黄金集测 nDCG@10 / MRR@10 / Recall@10。若中文细节排序不够，再试 **Qwen3-Reranker-0.6B**（延迟几乎相同）。**不要直接上 8B**——先证明 0.6B 不够。

**中文查询建议**：**不要因为检索用了 bge-m3 就固定选同系列精排**，换 reranker 可能提升精度但破坏长尾，**必须设 CI 评测门禁**。

来源：[2026 最佳 Reranker 对比](https://futureagi.com/blog/best-rerankers-for-rag-2026/)、[开放权重 Reranker 榜 2026](https://presenc.ai/research/best-open-weight-reranker-models-2026)、[中文 RAG Reranker 解析](https://blog.csdn.net/fuhanghang/article/details/161188440)

## 4.6 中文分块（影响检索质量的第一变量）

**2026 基线共识**：**递归分块 + 约 512 token（中文约 400–512 字）+ 10–20% overlap**。

- 中文分隔符优先级：**`\n\n → \n → 。→ ！→ ？→ ；→ ，→ 空格 → 字符`**。
- **⚠️ 默认分割器的中文隐形缺陷（必踩）**：
  - LangChain `RecursiveCharacterTextSplitter` 默认英文分隔符对中文（无空格）会**退化为按字符切**，导致列表条目被拆、标题与正文剥离、overlap 失效变为冗余。
  - **`SemanticChunker` 默认句分割正则 `(?<=[.?!])\s+` 只匹配英文句末标点**——中文（`。？！` 后无空格）**几乎匹配不到**，整篇文档被当成一句，产生 **10000+ 字符的超大块**。**必须扩展正则以支持 CJK 标点**，并对超长块做后置递归切分（LightRAG PR #3050 就是修的这个问题）。
  - 若用 LightRAG 的递归级联，**英文 `.?!` 应有意排除**——递归分割器做字面匹配会切开数字（如 `0.95` 被切成 `0` 和 `95`）。
- **注意 embedding 模型的 token 上限**：用 `bge-small-zh-v1.5`（512 token 上限）时，`chunk_size=1000` 字会**直接报错**。建议 250 字 + overlap 40。
- **父子分块（推荐用于 >10 页文档）**：子块 128–384 字符用于向量匹配，父块 4096 字符返回给 LLM 提供上下文。
- **评估**：用 **Recall@k** 衡量切分改造效果，用 Faithfulness 检查是否因断片导致幻觉（可用 RAGAS 对比不同策略）。

来源：[中文文本切分策略工程对比（腾讯云）](https://cloud.tencent.cn/developer/article/2729832)、[WeKnora CHUNKING.md](https://github.com/Tencent/WeKnora/blob/eab91d2f/docs/CHUNKING.md)、[LightRAG CJK 修复](https://github.com/HKUDS/LightRAG/pull/3050)、[RecursiveCharacterTextSplitter 中文缺陷复现](https://blog.nowcoder.net/n/cf2b53dd779e4ee098f5f7966c729a62)

---

# 5. MySQL 元数据存储

## 5.1 版本：8.4 LTS

| 版本 | 生命周期（2026-09） | 状态 |
|---|---|---|
| 8.0 | **2026-04 EOL** | 已终止 |
| **8.4 LTS** | Premier 到 **2029-04-30**，Extended 到 **2032-04-30** | 最新补丁 **8.4.9（2026-04）** 或 8.4.12（2026-08，口径不一） |
| 9.7 LTS | 2026-04-21，约到 2034 | 唯一 9.x LTS |
| 26.7.0 | 2026-07 | CalVer（YY.M）**不存在 9.8** |

**推荐 8.4 LTS**。不支持为 9.7 而升级的理由见 §5.6（VECTOR 无用）。

**⚠️ 认证变更（8.4 起）**：`mysql_native_password` **默认禁用**（9.0 移除），`caching_sha2_password` 成为默认。后果：
- 旧客户端报 `ERROR 1524: Plugin 'mysql_native_password' is not loaded`
- 使用 `caching_sha2_password` 的账号，客户端**必须走 TLS / Unix socket，或支持 RSA 公钥交换**（否则 `ERROR 2061: Authentication requires secure connection`）
- **这直接影响 asyncmy 的配置**（§5.3）

来源：[MySQL EOL 公告](https://www.mysql.com/support/eol-notice.html)、[8.4 vs 9.7 选型](https://www.modb.pro/db/2090616800686448640)、[SHA-256 认证](https://docs.oracle.com/cd/E17952_01/mysql-8.4-en/sha256-pluggable-authentication.html)

## 5.2 SQLAlchemy 2.0 异步

**版本：锁 `sqlalchemy[asyncio]==2.0.52`（2026-08-11）。不要用 2.1**（2.1.0b1/b2/b3 仍 beta，GA 预期"2026 夏末"，**未证实已发布**）。

**2.1 的破坏性变化（提前知悉）**：
- **greenlet 变为可选**，不再自动安装 → 异步必须 `pip install "sqlalchemy[asyncio]"`
- `postgresql://` 默认驱动改为 psycopg

**标准装配**：

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

engine = create_async_engine(
    "mysql+asyncmy://user:pwd@host:3306/rag?charset=utf8mb4",
    pool_size=10, max_overflow=10, pool_timeout=10,
    pool_pre_ping=True,
    pool_recycle=1200,        # 必须 < MySQL wait_timeout
    pool_use_lifo=True,
)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession,
                                  expire_on_commit=False,   # 关键
                                  autoflush=False)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(MEDIUMTEXT)
    document: Mapped["Document"] = relationship(back_populates="chunks", lazy="raise")
```

**五个必须写进规范的陷阱**：

1. **`expire_on_commit` 双向坑**：默认 `True` → `await session.commit()` 后访问任意属性（甚至 `obj.id`）触发 refresh → 异步下抛 `MissingGreenlet: greenlet_spawn has not been called`。设为 `False` 的代价是**对象可能持有陈旧数据** → 规范里写"长生命周期 session 中如需最新值，显式 `await session.refresh(obj)`"。
2. **异步下懒加载不可用**。三种应对（按推荐顺序）：查询时 `selectinload()`；模型上 `lazy="raise"`（让漏加载在开发期立刻炸）；`await obj.awaitable_attrs.chunks`（`AsyncAttrs`，需 2.0.13+）。
3. **"已 eager load 的空集合"仍可能触发懒加载**：对象已在 identity map 中时，访问空的 selectin 集合会再发一次查询 → `MissingGreenlet`。修法：`await session.get(Person, pk, populate_existing=True)` 或换新 session。
4. **`AsyncAttrs` 只能救关系，不能救"已过期属性"**。
5. **Session ≠ 全局单例**：全局 Session = 全局长事务 + 并发不安全 + identity map 无界增长。正确姿势：

```python
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:      # 每请求一个
        try:
            yield session
        except Exception:
            await session.rollback(); raise
```

来源：[SQLAlchemy asyncio 文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)、[2.1 迁移指南](https://docs.sqlalchemy.org/en/21/changelog/migration_21.html)、[Discussion #7196](https://github.com/sqlalchemy/sqlalchemy/discussions/7196)

## 5.3 驱动：asyncmy

| | asyncmy | aiomysql | mysqlclient |
|---|---|---|---|
| 2026 维护 | **活跃**：0.2.14（2026-08-12），Fedora 43/44 已打包 | **未找到 2026 发版证据（未证实）** | 活跃，但**不能 async** |
| URL | `mysql+asyncmy://` | `mysql+aiomysql://` | `mysql+mysqldb://`（同步） |
| 性能 | 自称大结果集比 aiomysql 快 **5.3x**，池吞吐 ~2x | 兼容性好 | C 快但被线程池吃掉 |

**推荐**：
- 运行期唯一异步驱动 **`asyncmy>=0.2.14`**（Cython 实现，API 兼容 aiomysql）
- Alembic / 运维脚本用同步 **PyMySQL**（`mysql+pymysql://`，需 ≥1.0.2），避免把 async 复杂度扩散到迁移层
- **禁止** `mysqlclient` + `run_in_executor` 的伪异步；**禁止** `mysql://` 裸 URL（会得到同步 Engine，`await` 直接报错）

**asyncmy 的已知问题**：
- **`caching_sha2_password` 需 0.2.9+ 且要 `ssl=True` 或 `auth_plugin='caching_sha2_password'`**（配合 RSA 公钥获取）—— **在 MySQL 8.4/9.x 上是硬门槛**
- 大批量 `executemany` 高并发下可能挂起 → 分批 100–500 行 + 超时
- 无默认 SQL 日志 → `logging.getLogger("asyncmy").setLevel(logging.DEBUG)`

来源：[asyncmy](https://github.com/long2ice/asyncmy)、[Fedora 安全更新](https://lists.fedoraproject.org/archives/list/package-announce@lists.fedoraproject.org/message/JVIX3GPID4MSEO772ZFG4K524T7RSAE3/)

## 5.4 连接池

`create_async_engine()` 默认用 **`AsyncAdaptedQueuePool`**。

| 参数 | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `pool_size` | 5 | 10–20 | 常驻连接 |
| `max_overflow` | 10 | 10–20 | 过大会打爆 `max_connections` |
| `pool_timeout` | 30 | 10 | 快速失败优于雪崩 |
| `pool_pre_ping` | False | **True** | checkout 时 ping，失效连接透明重连 |
| `pool_recycle` | -1 | **1200–3600** | **必须严格小于 `wait_timeout`** |
| `pool_use_lifo` | False | True | 减少空闲连接 |
| `pool_reset_on_return` | `rollback` | 保持 | 防脏事务泄漏 |

**与 `wait_timeout` 的交互（必写进部署检查清单）**：
- MySQL 默认 28800s（8h），但**云 RDS（阿里云/AWS）常被设为 60–300s**。`SHOW VARIABLES LIKE 'wait_timeout';` 必须检查。
- **只设 `pool_recycle` 不够**（环境会变）；**只设 `pool_pre_ping` 也不够**（每次 checkout 多一次 RTT）。**两者必须同时开**。
- `interactive_timeout` 应与 `wait_timeout` 一致。

**⚠️ 一个未修的边界**：即使两个都开了，**uvloop 下仍可能报 `RuntimeError: unable to perform operation on <TCPTransport closed=True>`**（SQLAlchemy discussion #11664，维护者判定为 uvloop 的 bug）。**缓解**：在 repository 层对只读/幂等查询加 `DBAPIError` 重试包装。

来源：[连接池文档](https://docs.sqlalchemy.org/en/20/core/pooling.html)、[Discussion #11664](https://github.com/sqlalchemy/sqlalchemy/discussions/11664)

## 5.5 Alembic

**版本 1.19.2（2026-09-04）**。初始化：`alembic init -t async migrations`。

**必须配置命名约定**（否则约束名随环境漂移）：

```python
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
class Base(AsyncAttrs, DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
```

⚠️ **已知副作用**：配了 `naming_convention` 后，autogenerate 可能对**未变的每个外键**报 "Detected removed/added foreign key"（Alembic issue #1623）→ 审阅阶段需识别并删除噪声 diff。

**autogenerate 的边界**：

| 能检测 | 检测不到 / 不可靠 |
|---|---|
| 表、列增删，nullable 变更 | **列重命名 → 生成为 drop + add → 数据丢失** |
| 显式索引、唯一约束、外键新增 | `server_default` 变更（仅部分） |
| | **CHECK 约束**、生成列、多值索引、JSON 表达式索引 |

⚠️ **CHECK 约束的 autogenerate 在 1.19 有变**：1.19.0 加入按名检测插件，**1.19.2 把它从默认通配中移出**，更名为 `alembic.ext.checkconstraint_byname`。官方建议**只在整个 schema 都用命名约定时启用**。

**MySQL 专属注意**：
- **MySQL DDL 是 autocommit，迁移无法回滚**。每个迁移"只做一件逻辑变更"，并显式指定在线 DDL：
  ```python
  op.execute("ALTER TABLE chunks ADD COLUMN embed_status VARCHAR(12) NOT NULL DEFAULT 'pending', ALGORITHM=INPLACE, LOCK=NONE")
  ```
- `op.create_table(..., mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_0900_ai_ci")` —— 否则依赖 `collation_server`，跨环境不一致。
- 生产流程：`--autogenerate` → **人工审阅** → `upgrade head --sql` 产出 SQL 交 DBA 审 → **绝不自动执行**。

来源：[Alembic + FastAPI 实践](https://dev.to/prasanna_kumar/mastering-database-migrations-in-fastapi-with-alembic-2062)、[Alembic issue #1623](https://github.com/sqlalchemy/alembic/issues/1623)

## 5.6 JSON 列、多值索引、以及 MySQL 9.x 的 VECTOR（重要结论）

**事实**：
- MySQL **没有 JSONB**；`JSON` 本身就二进制存储，**不能直接索引**（`ERROR 3152`）。
- 索引 JSON 的三条路径：**生成列 + 普通二级索引**、**多值索引**、**干脆提升为普通列（最优）**。
- **InnoDB 自 5.7.8 起支持对 VIRTUAL 生成列建二级索引**（"只有 STORED 能建索引"是过时说法，仅非 InnoDB 成立）。

```sql
ALTER TABLE documents
  ADD COLUMN lang VARCHAR(16) GENERATED ALWAYS AS (JSON_UNQUOTE(extra->>'$.lang')) VIRTUAL,
  ADD INDEX ix_documents_lang (tenant_id, lang);
```

- 用 `->>`（双箭头）而非 `->`（后者保留引号，等值匹配失败）
- 表达式必须**确定性**（禁 `NOW()`/`RAND()`/`UUID()`，否则 `ERROR 3105`）
- **VIRTUAL → STORED 不能原地转换**，必须 DROP 重建
- 查询必须**直接引用生成列**，不能引用原始 JSON 表达式

**多值索引（JSON 数组，8.0.17+）**：

```sql
ALTER TABLE documents ADD COLUMN tag_ids JSON NULL;
ALTER TABLE documents ADD INDEX ix_documents_tags ((CAST(tag_ids AS UNSIGNED ARRAY)));
-- 只有这三种谓词走索引：
-- WHERE 42 MEMBER OF (tag_ids)
-- WHERE JSON_CONTAINS(tag_ids, CAST('[42,43]' AS JSON))
-- WHERE JSON_OVERLAPS(tag_ids, CAST('[42,43]' AS JSON))
```

**限制清单（必踩）**：
- **`CAST(... AS JSON ARRAY)` 不合法**（JSON 不能作为 CAST 目标类型）
- 一个索引**只允许一个多值 key part**；不能排序、不能做主键、**不能覆盖索引**、不能外键、不支持前缀、**不能 ONLINE DDL（`ALGORITHM=COPY`）**
- 每记录索引值上限 **65221 字节**（`ERROR 3905`）
- 字符集只能 `binary` 或 `utf8mb4`
- **Collation 陷阱**：`CAST()` 返回 `utf8mb4_0900_ai_ci`，`JSON_UNQUOTE()` 返回 `utf8mb4_bin` → `WHERE data->>'$.name' = 'James'` **可能不走索引**

**判定规则**：

| 用 JSON 列 | 用关联表 / 普通列 |
|---|---|
| schema 易变的扩展属性（parser 配置、审计快照） | 需要 JOIN / 外键完整性 |
| 不参与（或极少参与）过滤的元数据 | 高频过滤、多列组合过滤 |
| 值的基数低、结构可不固定 | 统计聚合、枚举稳定（tags/ACL/collection 归属） |

**规范**：**核心检索维度一律列化 + 关联表；JSON 只放"不查或很少查"的扩展属性**。

**★ MySQL 9.x 的 VECTOR：2026 年明确不可用**

- **已具备**：原生 `VECTOR(N)`（float32，默认上限 2048 维，绝对上限 16383）、`TO_VECTOR()`/`STRING_TO_VECTOR()`/`FROM_VECTOR()`/`VECTOR_DIM()`
- **不具备（关键）**：VECTOR 列**不能做主键/唯一键/外键/分区键，不能建任何二级/全文/空间/BTREE/HASH 索引**；**MySQL 9.7 Community 没有 VECTOR INDEX、没有 ANN/KNN**。唯一距离函数 `DISTANCE()` **仅 HeatWave on OCI 与 MySQL AI 可用**（实测 9.7.0 Community 报 `FUNCTION ... DISTANCE does not exist`）。社区提案 `CREATE VECTOR INDEX ... USING SCANN` **仍是 feature request**。

> **结论：2026 年在 MySQL 上做向量检索不可行。Milvus 仍是必需组件；不要为了 VECTOR 类型选 9.7。**

来源：[MySQL 多值索引](https://oneuptime.com/blog/post/2026-03-31-mysql-multi-valued-indexes-json-arrays)、[JSON 列生产实践](https://cloud.tencent.com.cn/developer/article/2682722)、[MySQL 9.7 解析](https://www.modb.pro/db/2046752965533966336)、[VECTOR INDEX feature request](https://github.com/mysql/mysql-community/issues/3)

## 5.7 utf8mb4 与 Collation

**为什么 `utf8mb3` 是陷阱**：最多 3 字节，放不下 emoji 和部分 CJK 扩展区汉字（4 字节）；表现为写入报错或截断；MySQL 8.0.28+ 已标记弃用。**规范：所有表、列、连接一律 `utf8mb4`，CI 加 DDL lint 禁止出现 `utf8`/`utf8mb3`。**

| Collation | UCA 版本 | 中文排序 | 结论 |
|---|---|---|---|
| `utf8mb4_general_ci` | 非 UCA | **差** | MySQL 官方不推荐 |
| `utf8mb4_unicode_ci` | UCA 4.0 | 一般 | 官方点名有 "Sushi-Beer" 问题 |
| `utf8mb4_unicode_520_ci` | UCA 5.2.0 | 较好 | 官方点名 "Mother-Daddy" 问题；日文 p/b 音不区分 |
| **`utf8mb4_0900_ai_ci`** | **UCA 9.0.0** | **最好** | **推荐（8.0 默认）** |
| `utf8mb4_bin` / `utf8mb4_0900_as_cs` | — | 精确 | 用于唯一键、hash、token、邮箱 |

```ini
character_set_server = utf8mb4
collation_server     = utf8mb4_0900_ai_ci
default_time_zone    = '+00:00'
```
```python
"mysql+asyncmy://user:pwd@host:3306/rag?charset=utf8mb4"   # URL 必须显式带 charset
```

⚠️ **跨列比较 / JOIN 必须 charset+collation 完全一致**，否则索引失效（曾有 LEFT JOIN 因 collation 不一致导致全表扫描的案例）。

**索引前缀长度（关键数字）**：
- InnoDB **单列索引最大 3072 字节**（**DYNAMIC / COMPRESSED** 行格式；8.0 起 `innodb_large_prefix` 已移除并默认启用）
- utf8mb4 下即 **768 字符**：`VARCHAR(768)` 是临界；`VARCHAR(1024)` ❌
- **推荐模式**：长文本用前缀索引 `KEY (tenant_id, source_uri(255))`；需要全值唯一时用 hash 列 **`CHAR(64) CHARACTER SET ascii COLLATE ascii_bin`**（**仅 64 字节，索引体积缩到 1/4**）

来源：[MySQL 官方 charset/UCA collation 博客](https://dev.mysql.com/blog-archive/mysql-character-sets-unicode-and-uca-compliant-collations/)、[InnoDB Limits](https://dev.mysql.com/doc/refman/8.4/en/innodb-limits.html)

## 5.8 Schema 设计（DDL 草图）

**全局约定**：
- 引擎/字符集：`ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC`
- 主键：内部 `BIGINT UNSIGNED AUTO_INCREMENT`；对外 `public_id BINARY(16)`（UUIDv7，`UUID_TO_BIN(uuid, 1)` 保时间有序，避免页分裂）
- 时间：**`DATETIME(3)`**（不用 `TIMESTAMP`：2038 问题 + 隐式时区转换）；统一 UTC
- **软删除**：`deleted_at DATETIME(3) NOT NULL DEFAULT '1970-01-01 00:00:00.000'` —— **用纪元零值而非 NULL**，因为 MySQL 唯一索引中 **NULL 不参与去重**，用 NULL 会导致"同一业务键可存在多条未删除记录"的漏洞
- 多租户：**每张业务表第一列都是 `tenant_id`，所有索引以 `tenant_id` 为前导列**
- 枚举：**`VARCHAR(16)` + `CHECK` 约束**（8.0.16+ 真正强制），Python 侧用 `StrEnum` 单一来源；比 `ENUM` 更易演进

```sql
-- ========== 租户 / 用户 ==========
CREATE TABLE tenants (
  id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_id  BINARY(16)      NOT NULL,
  name       VARCHAR(128)    NOT NULL,
  status     VARCHAR(16)     NOT NULL DEFAULT 'active',
  created_at DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_tenants_public_id (public_id),
  UNIQUE KEY uk_tenants_name (name, deleted_at),
  CONSTRAINT ck_tenants_status CHECK (status IN ('active','suspended','deleted'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

CREATE TABLE users (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_id     BINARY(16)      NOT NULL,
  tenant_id     BIGINT UNSIGNED NOT NULL,
  email         VARCHAR(255)    NOT NULL,
  display_name  VARCHAR(128)    NOT NULL,
  password_hash VARCHAR(255)    NULL,           -- pwdlib/argon2id
  role          VARCHAR(16)     NOT NULL DEFAULT 'member',
  status        VARCHAR(16)     NOT NULL DEFAULT 'active',
  last_login_at DATETIME(3)     NULL,
  created_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at    DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_users_public_id (public_id),
  UNIQUE KEY uk_users_tenant_email (tenant_id, email, deleted_at),
  KEY ix_users_tenant_status (tenant_id, status, id),
  CONSTRAINT fk_users_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
  CONSTRAINT ck_users_role CHECK (role IN ('owner','admin','member','viewer'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- ========== 知识库 / 文档 ==========
CREATE TABLE collections (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  tenant_id   BIGINT UNSIGNED NOT NULL,
  name        VARCHAR(128)    NOT NULL,
  description VARCHAR(512)    NULL,
  embed_model VARCHAR(64)     NOT NULL,
  embed_dim   SMALLINT UNSIGNED NOT NULL,
  status      VARCHAR(16)     NOT NULL DEFAULT 'active',
  created_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at  DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_collections_tenant_name (tenant_id, name, deleted_at),
  CONSTRAINT fk_collections_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

CREATE TABLE documents (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_id     BINARY(16)      NOT NULL,
  tenant_id     BIGINT UNSIGNED NOT NULL,
  collection_id BIGINT UNSIGNED NOT NULL,
  title         VARCHAR(512)    NOT NULL,
  source_uri    VARCHAR(1024)   NOT NULL,
  source_type   VARCHAR(32)     NOT NULL,     -- upload/url/feishu/...
  mime_type     VARCHAR(128)    NOT NULL,
  size_bytes    BIGINT UNSIGNED NOT NULL DEFAULT 0,
  content_hash  CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,  -- sha256(raw bytes)
  version       INT UNSIGNED    NOT NULL DEFAULT 1,
  is_active     TINYINT(1)      NOT NULL DEFAULT 1,   -- 该 version 是否为当前生效版本
  status        VARCHAR(16)     NOT NULL DEFAULT 'pending',
  chunk_count   INT UNSIGNED    NOT NULL DEFAULT 0,
  error_message TEXT            NULL,
  extra         JSON            NULL,
  tag_ids       JSON            NULL,
  created_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at    DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_documents_public_id (public_id),
  UNIQUE KEY uk_documents_hash_ver (tenant_id, content_hash, version, deleted_at),
  KEY ix_documents_collection (tenant_id, collection_id, is_active, id),
  KEY ix_documents_status (tenant_id, status, updated_at),
  KEY ix_documents_src_prefix (tenant_id, source_uri(255)),
  CONSTRAINT fk_documents_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
  CONSTRAINT fk_documents_collection FOREIGN KEY (collection_id) REFERENCES collections(id),
  CONSTRAINT ck_documents_status CHECK (
    status IN ('pending','parsing','chunking','embedding','ready','failed','deleted'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- ========== 分块（chunk 正文的权威存储）==========
CREATE TABLE chunks (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,  -- ★ 同时作为 Milvus 主键与 Neo4j Chunk.chunk_id
  tenant_id    BIGINT UNSIGNED NOT NULL,
  document_id  BIGINT UNSIGNED NOT NULL,
  version      INT UNSIGNED    NOT NULL,
  chunk_index  INT UNSIGNED    NOT NULL,
  content      MEDIUMTEXT      NOT NULL,                 -- ★ 权威原文
  content_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  token_count  INT UNSIGNED    NOT NULL DEFAULT 0,
  char_count   INT UNSIGNED    NOT NULL DEFAULT 0,
  page_no      INT UNSIGNED    NULL,
  section_path VARCHAR(512)    NULL,                     -- "第3章 > 3.2 > ..."
  embed_model  VARCHAR(64)     NOT NULL,
  embed_dim    SMALLINT UNSIGNED NOT NULL,
  embed_status VARCHAR(12)     NOT NULL DEFAULT 'pending', -- pending/embedded/stale/failed
  embedded_at  DATETIME(3)     NULL,
  extra        JSON            NULL,
  created_at   DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at   DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at   DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_chunks_doc_ver_idx (document_id, version, chunk_index),
  KEY ix_chunks_tenant_doc (tenant_id, document_id, version, chunk_index),
  KEY ix_chunks_embed_backlog (embed_status, updated_at),   -- 补齐/重嵌 worker 扫描
  KEY ix_chunks_hash (content_hash),
  CONSTRAINT fk_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  CONSTRAINT ck_chunks_embed_status CHECK (embed_status IN ('pending','embedded','stale','failed'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- ========== 摄取任务（幂等 + 可重试）==========
CREATE TABLE ingestion_jobs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  tenant_id       BIGINT UNSIGNED NOT NULL,
  document_id     BIGINT UNSIGNED NULL,
  job_type        VARCHAR(24)     NOT NULL,     -- ingest/reindex/delete/reconcile
  status          VARCHAR(16)     NOT NULL DEFAULT 'queued',
  priority        TINYINT         NOT NULL DEFAULT 0,
  attempt         INT UNSIGNED    NOT NULL DEFAULT 0,
  max_attempts    INT UNSIGNED    NOT NULL DEFAULT 5,
  idempotency_key CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  payload         JSON            NULL,
  result          JSON            NULL,
  last_error      TEXT            NULL,
  next_run_at     DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  locked_by       VARCHAR(64)     NULL,          -- worker id
  locked_until    DATETIME(3)     NULL,
  started_at      DATETIME(3)     NULL,
  finished_at     DATETIME(3)     NULL,
  created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uk_jobs_idem (idempotency_key),
  KEY ix_jobs_claim (status, next_run_at, priority, id),    -- 轮询队列
  KEY ix_jobs_tenant_doc (tenant_id, document_id, id),
  CONSTRAINT fk_jobs_document FOREIGN KEY (document_id) REFERENCES documents(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- ========== Outbox（MySQL → Milvus/Neo4j 的可靠投递）==========
CREATE TABLE outbox_events (
  id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  tenant_id      BIGINT UNSIGNED NOT NULL,
  aggregate_type VARCHAR(32)     NOT NULL,       -- document/chunk/entity
  aggregate_id   BIGINT UNSIGNED NOT NULL,
  event_type     VARCHAR(32)     NOT NULL,       -- upsert/delete/reindex
  payload        JSON            NOT NULL,
  status         VARCHAR(12)     NOT NULL DEFAULT 'pending',
  attempt        INT UNSIGNED    NOT NULL DEFAULT 0,
  available_at   DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  last_error     TEXT            NULL,
  created_at     DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  processed_at   DATETIME(3)     NULL,
  PRIMARY KEY (id),
  KEY ix_outbox_poll (status, available_at, id),  -- FOR UPDATE SKIP LOCKED 扫描
  KEY ix_outbox_agg (aggregate_type, aggregate_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- ========== 会话 / 消息 ==========
CREATE TABLE conversations (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_id       BINARY(16)      NOT NULL,
  tenant_id       BIGINT UNSIGNED NOT NULL,
  user_id         BIGINT UNSIGNED NOT NULL,
  title           VARCHAR(255)    NULL,
  summary         VARCHAR(2048)   NULL,
  message_count   INT UNSIGNED    NOT NULL DEFAULT 0,
  last_message_at DATETIME(3)     NULL,
  extra           JSON            NULL,
  created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at      DATETIME(3)     NOT NULL DEFAULT '1970-01-01 00:00:00.000',
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_public_id (public_id),
  KEY ix_conv_user_recent (tenant_id, user_id, deleted_at, last_message_at DESC, id DESC),
  CONSTRAINT fk_conv_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

CREATE TABLE messages (
  id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id   BIGINT UNSIGNED NOT NULL,
  tenant_id         BIGINT UNSIGNED NOT NULL,
  seq               INT UNSIGNED    NOT NULL,     -- 会话内序号，避免依赖时间排序
  role              VARCHAR(12)     NOT NULL,     -- user/assistant/system/tool
  content           MEDIUMTEXT      NOT NULL,
  content_hash      CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  model             VARCHAR(64)     NULL,
  prompt_tokens     INT UNSIGNED    NULL,
  completion_tokens INT UNSIGNED    NULL,
  latency_ms        INT UNSIGNED    NULL,
  trace_id          CHAR(32) CHARACTER SET ascii COLLATE ascii_bin NULL,
  citations         JSON            NULL,         -- [{chunk_id, doc_id, score, span}]
  created_at        DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uk_msg_conv_seq (conversation_id, seq),
  KEY ix_msg_tenant_created (tenant_id, created_at, id),
  CONSTRAINT fk_msg_conv FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
  CONSTRAINT ck_msg_role CHECK (role IN ('user','assistant','system','tool'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;

-- 引用溯源（比 JSON 更适合做聚合与评估）
CREATE TABLE message_citations (
  message_id BIGINT UNSIGNED NOT NULL,
  chunk_id   BIGINT UNSIGNED NOT NULL,
  rank_no    TINYINT UNSIGNED NOT NULL,
  score      FLOAT           NOT NULL,
  used       TINYINT(1)      NOT NULL DEFAULT 0,
  PRIMARY KEY (message_id, chunk_id),
  KEY ix_cit_chunk (chunk_id),
  CONSTRAINT fk_cit_msg FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC;
```

## 5.9 ★ chunk 正文要不要存 MySQL？（明确回答）

**关键事实**：

1. **Milvus 的 `VARCHAR` 上限是 65535 字节**（约 21k 汉字）；`max_length` 是建 collection 时的必填参数；**Milvus 没有 TEXT 类型**。大字段批量插入/迁移容易撞 **64MB gRPC 上限**。
2. **Milvus 支持 `output_fields=["content"]` 直接返回 VARCHAR** —— 所以"Milvus 能否返回全文"的答案是**能**，但有字节上限与响应体积代价。
3. **Milvus 不是记录系统**：`insert` **不按主键去重**（同一 PK 可存多条）；`upsert` 是**读后写 + delete+insert**；`delete` 是**软删**，物理回收依赖 compaction。主键 `max_length` 不可变更（来源冲突，**未证实可改**）。
4. 若要用于 **BM25 Function 全文检索**，文本**必须**在 Milvus 侧（服务端分词，且 IDF 为段内统计）。

**★ 推荐：MySQL 存权威正文，Milvus 只存向量 + 少量标量。**

Milvus collection 建议只含：`chunk_id INT64 (PK, 显式指定)`、`tenant_id INT64`、`document_id INT64`、`collection_id INT64`、`embed_model VARCHAR(64)`、`content_hash VARCHAR(64)`、`is_active BOOL`、`vector FLOAT_VECTOR(dim)`（+ 可选 `sparse_vector`）。

| 论点 | 说明 |
|---|---|
| **单一事实来源** | 版本、软删、权限、审计、合规删除只需在 MySQL 一处正确；Milvus 侧可随时重建 |
| **避免双写不一致** | Milvus 的 delete/upsert 是异步软删语义，极易出现"向量已删、文本还在"或反之 |
| **存储成本** | InnoDB 压缩 + buffer pool 复用比 Milvus 段文件更可控；Milvus 的 delete 不立即回收空间 |
| **限制** | 65535 字节上限 + gRPC 64MB 响应上限，大 top-k / 长 chunk 必然踩坑 |
| **展示与引用** | 前端要高亮 span、跳转原文位置，需要 `page_no`/`section_path` 等结构化字段 |
| **额外成本可忽略** | 检索后 `SELECT id, content, page_no, section_path FROM chunks WHERE id IN (...)`，主键 IN 查询 P99 通常毫秒级 |

**唯一例外**：如果用 **Milvus 内建 BM25 Function + 混合检索**，文本必须在 Milvus 侧。此时改为：**Milvus 存 `content VARCHAR(8192)`（够 BM25 用、避开 64KB 极端值）+ MySQL 存权威全文**，由 outbox 保证最终一致，并在规范里明确 **"Milvus 的 content 是派生副本，禁止作为展示源"**。

来源：[Milvus 限额](https://milvus.io/docs/zh/v2.4.x/limitations.md)、[Milvus upsert 语义](https://milvus.io/docs/zh/v2.5.x/upsert-entities.md)、[VARCHAR max_length 讨论](https://github.com/milvus-io/milvus/discussions/34478)、[LightRAG 64MB gRPC 溢出 PR](https://github.com/HKUDS/LightRAG/pull/3228)

## 5.10 事实源 + 派生索引：Outbox / 幂等重索引 / 对账

**架构原则**：**MySQL 是唯一 source of truth；Milvus 与 Neo4j 是可丢弃、可重建的派生读模型。二者不一致时，以 MySQL 为准。**

**Outbox 模式**：

```
┌─ 单个 MySQL 事务 ────────────────────────────┐
│  INSERT/UPDATE documents, chunks, ...        │
│  INSERT INTO outbox_events (status='pending')│
│  COMMIT                                       │
└──────────────────────────────────────────────┘
              │
              ▼
  Relay Worker（独立进程 / 定时任务）
    SELECT * FROM outbox_events
     WHERE status='pending' AND available_at <= NOW(3)
     ORDER BY id LIMIT 100
     FOR UPDATE SKIP LOCKED;          ← 多 worker 安全
    → Milvus upsert / delete（幂等）
    → Neo4j MERGE
    → UPDATE status='sent', processed_at=NOW(3)
    失败：attempt+1, available_at = NOW() + 2^attempt 秒, last_error=...
```

要点：
- **MySQL 没有 `LISTEN/NOTIFY`**：投递靠 **1–5 秒轮询**（MVP 够用）或 **binlog CDC**（Debezium / canal，低延迟但引入运维复杂度）。**MVP 用轮询，规模上来再上 CDC。**
- **至少一次投递 → 消费端必须幂等**：Milvus 用 `upsert`（显式主键天然幂等）；Neo4j 用 `MERGE ... SET e += $props`。
- **顺序（容易忽略的致命点）**：同一 `aggregate_id` 的事件必须按 `id` 顺序处理，否则"先 upsert 后 delete"会留下**幽灵向量**。规范：relay 按 `aggregate_id` 分组串行，或事件带 `version` 并在消费端丢弃旧版本。
- **`outbox_lag_seconds`（最老 pending 事件的年龄）必须是 P1 告警项。**

**幂等重索引**：

```python
content_hash = sha256(
    normalize(text)                    # Unicode NFKC + 空白折叠
    + "\x1f" + CHUNKER_CONFIG_VERSION   # 分块器/参数变更时 bump
    + "\x1f" + EMBED_MODEL_ID           # 模型变更时 bump
    + "\x1f" + EMBED_DIM
).hexdigest()
```

- **模型升级用"双索引切换"**：写新 collection（`chunks_v2`）→ 回填 → 校验 → 原子切换读路径 → 删旧。**不要原地改维度**。
- **文档版本化**：新版本 chunk 以新 `(document_id, version, chunk_index)` 写入 → 回填完成后再把 `is_active` 从旧版本切到新版本 → 最后异步删旧。**避免检索窗口期出现空结果。**

**对账任务（每日 + 手动）**：

```python
mysql_ids = {r for r in await session.stream_scalars(
    select(Chunk.id).where(Chunk.tenant_id == t, Chunk.deleted_at == EPOCH_ZERO,
                           Chunk.embed_status == "embedded"))}
milvus_ids = set()
for batch in client.query_iterator(collection_name=..., filter=f"tenant_id == {t}",
                                   output_fields=["chunk_id"], batch_size=1000):
    milvus_ids.update(b["chunk_id"] for b in batch)

orphans = milvus_ids - mysql_ids   # 向量库有、MySQL 无 → 删除
missing = mysql_ids - milvus_ids   # MySQL 有、向量库无 → 重新嵌入
```

- ⚠️ **不要用 `query` 默认 limit 拉全量**（默认有 limit），**必须用 `query_iterator` 或分批**。
- 用同一套框架覆盖 Neo4j（对比 `chunks` 与 `:Chunk` 节点数，按 id diff，用 `MERGE` 修复）。
- 结果写入 `reconciliation_runs` 表，指标化 + 趋势告警。

来源：[Qdrant 与 Postgres 同步指南](https://qdrant.org.cn/documentation/data-synchronization/with-postgres/index.md)、[hexkit DAO Publisher](https://ghga-de.github.io/hexkit/user-guide/protocols/daopublisher.html)

---

# 6. FastAPI 架构

## 6.1 版本基线

| 组件 | 版本（2026-09） | 备注 |
|---|---|---|
| Python | **3.12.x**（保守）或 3.13.x | **3.14 的 free-threading 对 I/O 密集的 RAG 几乎无收益（4–8%）**，且需重编译 C 扩展；**不要用** `python3.13t` |
| FastAPI | **0.141.1**（2026-07-29）*（单一来源，建议实测）* | 已彻底移除 Pydantic v1；0.130.0 起内建 pydantic-core 序列化（~2x JSON 提速），**`ORJSONResponse` 已弃用** |
| Pydantic | **2.13.5**（2026-08-28） | 2.14.0b1 为 beta |
| pydantic-settings | **2.14.x** | |
| structlog | **26.1.0**（2026-06-06） | 25.x 终止于 25.5.0 |
| LangGraph | **1.2.6**（2026-06） | `langchain-core>=1.4.7` |

**FastAPI 与本项目相关的关键演进**：
- **0.126.0**：移除 Pydantic v1 支持；**0.127/0.128** 连 `pydantic.v1` 兼容层也删除 → **不可能再用 v1 语法**
- **0.134.0**：路径函数可直接 `yield` 流式输出
- **0.137.0**：router 内部改为 tree 结构
- **0.140.12/13**：修复 `format_sse_event` 换行切分（SSE 规范符合性）与 SSE 端点上 `status_code` 被忽略的问题
- **★ 0.140.x 起内建 `fastapi.sse` 模块**（`from fastapi.sse import ServerSentEvent`）

来源：[FastAPI release notes 镜像](https://fastapi.python.club.tw/release-notes/)、[Pydantic HISTORY](https://raw.githubusercontent.com/pydantic/pydantic/main/HISTORY.md)、[structlog CHANGELOG](https://raw.githubusercontent.com/hynek/structlog/refs/heads/main/CHANGELOG.md)

## 6.2 项目布局（推荐 monorepo）

```
rag-agent/
├── pyproject.toml / uv.lock / .python-version / .env.example
├── compose.yaml / compose.prod.yaml / Dockerfile / alembic.ini
├── src/rag/
│   ├── main.py              # create_app() 工厂 + lifespan
│   ├── api/
│   │   ├── deps.py          # ★ 所有 Depends 的聚合点（唯一）
│   │   ├── errors.py        # 异常类 + exception_handler 注册
│   │   └── v1/
│   │       ├── router.py    # APIRouter 汇总
│   │       ├── chat.py      # SSE 流式问答
│   │       ├── documents.py # 上传/列表/删除
│   │       ├── jobs.py      # 任务状态 + 进度 SSE
│   │       └── health.py    # /healthz /readyz（不进 OpenAPI）
│   ├── core/                # config / logging / security / ratelimit / telemetry
│   ├── schemas/             # Pydantic v2 出入参（与 ORM 解耦）
│   │   └── events.py        # ★ 流式事件的 discriminated union
│   ├── domain/              # 纯业务：异常、枚举、值对象（无框架依赖）
│   ├── services/            # 业务编排（API 与 worker 共用）：ingestion/retrieval/graph_store/chat
│   ├── repositories/        # document_repo(MySQL) / vector_repo(Milvus) / kg_repo(Neo4j)
│   ├── agent/               # ★ LangGraph: graph.py / state.py / nodes/ / tools.py / checkpointer.py
│   ├── infra/               # db.py / milvus.py / neo4j.py / redis.py / storage.py
│   └── worker/              # main.py (WorkerSettings) + tasks/ingest.py, reindex.py
└── tests/{conftest.py, unit/, integration/}
```

**分层纪律**：router 只做解析/调用/返回，**禁止直接碰 DB**；service 承载业务规则并抛领域异常；repository 只做查询。

**★ 推荐：API 与 worker 同仓、同镜像、不同 CMD。** 理由：
1. **共享面极大**：`schemas/`、`domain/`、`services/ingestion.py`、`agent/`、`core/config.py` 全部复用。拆仓会引入 schema 包发布/版本对齐开销，而"独立部署"用同镜像不同 CMD 即可拿到。
2. **依赖完全一致**：worker 天然需要 DB/Redis/Milvus/Neo4j 连接，与 API 一致（尤其 embedding 模型）。拆开要维护两份依赖。
3. **契约单一真相**：摄取写 `documents.status`、写 Milvus、写 Neo4j，schema 变更必须与 API 同步。**同仓 + 同一次 CI 才安全。**

**何时拆**：(a) 摄取需要 GPU 或超大内存，与 API 机器规格明显不同；(b) 摄取吞吐需独立扩缩容；(c) 团队边界清晰。届时提升为 **uv workspace 多包**（仍建议 monorepo），而非多 git 仓库。

来源：[Production-Ready FastAPI 2026](https://dev.to/datanestdigital/production-ready-fastapi-project-structure-2026-guide-b1g)、[FastAPI 项目结构](https://www.zestminds.com/blog/fastapi-project-structure/)

## 6.3 Pydantic v2 要点

```python
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")   # 入参 forbid，出参可 ignore
    id: int
    filename: str
    size_bytes: int

    @computed_field
    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / 1024 / 1024, 2)

doc = DocumentOut.model_validate(orm_obj)
raw = DocumentOut.model_validate_json(body)
```

- `ConfigDict(from_attributes=True)` 替代 v1 的 `orm_mode`
- `model_validate` / `model_validate_json` / `model_dump(mode="json")` 是 v2 唯一正道；**不要再写 `.dict()` / `.json()` / `.parse_obj()`**

**★ 流式事件的 discriminated union（本项目核心）**：

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Discriminator

class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str
class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool: str
    args: dict
class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    doc_id: int
    chunk_id: int
    score: float
class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    usage: dict | None = None

StreamEvent = Annotated[
    Union[TokenEvent, ToolStartEvent, CitationEvent, ErrorEvent, DoneEvent],
    Discriminator("type"),
]
```

⚠️ **必须避开的坑**（来自真实 PR）：Pydantic 的 union 判别器在某些路径下会**静默返回原始 dict** 而非类型化模型，导致嵌套字段未被解析，后续累加抛 `AttributeError`。**修复方式是兜底用完整校验 `model_validate`，而不是 `model_construct(**raw)`**（后者跳过校验，会让嵌套 union 停留在 dict）。

**工程约定**：事件契约**只增不改**（additive-only）；加 **schema 漂移 CI 门禁**（快照 canonical JSON Schema，未评审变更直接 fail）。校验开销 30–50 µs/事件，相对 LLM 延迟可忽略。

**pydantic-settings**：

```python
class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_", extra="ignore")
    uri: str = "http://milvus:19530"
    token: SecretStr | None = None
    collection: str = "rag_chunks"
    dim: int = 1024

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_nested_delimiter="__",        # APP__MILVUS__URI
        extra="ignore", case_sensitive=False,
    )
    env: Literal["dev", "staging", "prod"] = "dev"
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    jwt_secret: SecretStr
    upload_max_mb: int = 50

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- **`SecretStr` 一定要用**：`settings.jwt_secret.get_secret_value()` 显式取值，日志/`repr` 自动脱敏。**否则 `logger.info(settings)` 会直接把密钥写进日志。**
- 测试中用 `get_settings.cache_clear()` 重置

来源：[agentic-runtime-platform ADR-014](https://github.com/tafreeman/agentic-runtime-platform/blob/9383b3f8b91415e15bd9c1c729aa5f736d616cb2/docs/adr/ADR-014-pydantic-wire-format.md)、[anthropic-sdk-python PR #1542](https://github.com/anthropics/anthropic-sdk-python/pull/1542)

## 6.4 依赖注入

```python
from typing import Annotated, AsyncIterator
from fastapi import Depends, Request

async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback(); raise
        finally:
            await session.close()

SessionDep = Annotated[AsyncSession, Depends(get_session)]

def get_milvus(request: Request) -> MilvusClient:
    return request.app.state.milvus
MilvusDep = Annotated[MilvusClient, Depends(get_milvus)]

def get_neo4j(request: Request) -> AsyncDriver:
    return request.app.state.neo4j
Neo4jDep = Annotated[AsyncDriver, Depends(get_neo4j)]

def get_redis(request: Request) -> Redis:
    return request.app.state.redis
RedisDep = Annotated[Redis, Depends(get_redis)]
```

- **`Annotated[X, Depends(...)]` 别名是唯一推荐风格**（可复用、IDE 友好、避免默认值污染签名）
- `yield` 依赖的清理在**响应发送后**执行；`finally` 保证异常也回滚
- **同一次请求内依赖是缓存的**（`use_cache=True` 默认）
- **单例挂 `app.state`，用依赖从 `request.app.state` 取**（而非模块全局）—— 生命周期由 lifespan 统一管理，可测试，不依赖导入顺序

⚠️ **待验证**：检索中看到 FastAPI PR **#14301** `Fix Depends(func, scope='function') for top level dependencies`，强烈暗示 2026 年 `Depends` **增加了 `scope` 参数**。**未确认签名与语义，实现前查 `fastapi/params.py`。**

来源：[fastapi-best-practices](https://github.com/GONNE-2004/fastapi-best-practices)、[FastAPI Production Playbook 2026](https://dev.to/apaksh/building-production-ready-apis-with-fastapi-in-2026-the-complete-playbook-5hlb)

## 6.5 lifespan 与启动/关闭顺序

**`@app.on_event("startup"/"shutdown")` 自 0.93.0 起已弃用**，截至 0.138.0 仍存在但**不要赌它还在**。

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)          # 1. 日志最先
    setup_telemetry(settings)

    app.state.redis = Redis.from_url(settings.redis_url)      # 2. worker 队列与限流底座
    await app.state.redis.ping()

    app.state.db = create_async_engine(settings.database_url,
        pool_size=10, max_overflow=10, pool_pre_ping=True, pool_recycle=1200)
    app.state.sessionmaker = async_sessionmaker(app.state.db, expire_on_commit=False)
    async with app.state.db.connect() as c:                    # 3. 显式验证
        await c.execute(text("SELECT 1"))

    app.state.milvus = MilvusClient(uri=settings.milvus.uri, token=...)  # 4.
    await asyncio.to_thread(app.state.milvus.load_collection, settings.milvus.collection)

    app.state.neo4j = AsyncGraphDatabase.driver(                  # 5. 普通类方法，非协程
        settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password.get_secret_value()))
    await app.state.neo4j.verify_connectivity()

    app.state.agent = build_agent(settings)                       # 6. 编译一次，全局复用
    app.state.ready = True                                        # 7. /readyz 从此返回 200
    try:
        yield
    finally:
        app.state.ready = False
        await app.state.neo4j.close()      # ★ 协程，必须 await
        await app.state.db.dispose()
        await app.state.redis.aclose()
```

**关键细节**：
- **`AsyncDriver.close()` 是协程必须 `await`**（同步 API 里是 `driver.close()`）——**最容易写错的一处**。
- **`AsyncDriver` 可在并发协程中安全使用，但不是线程安全的**；`close()` 本身不并发安全。
- **`AsyncSession.cancel()`**（同步、无 await）专用于 `asyncio.CancelledError` 处理器内部。
- **不要用同步 `GraphDatabase` 在 asyncio 里**（阻塞事件循环）；**不要每请求建 driver**。
- **lifespan 每个 worker 进程执行一次**：`gunicorn -w 4` 会建 4 套连接池。按此计算总连接数（MySQL `max_connections`、Milvus 连接数）。
- **embedding 模型建议懒加载**（首次用时 + 加锁），因为加载可能几十秒到几分钟会拖垮 readiness 门禁；但要在 `/readyz` 反映"模型未就绪"。
- **LangGraph 图对象编译一次、全局复用**（它是无状态的，状态在 checkpointer 里）。

来源：[Neo4j Python driver async API](https://raw.githubusercontent.com/neo4j/neo4j-python-driver/656c79648c8e91e2afb8ccdc824e8b498e92a2d7/docs/source/async_api.rst)、[DeepWiki Async Driver](https://deepwiki.com/neo4j/neo4j-python-driver/7.1-async-driver-and-sessions)

## 6.6 ★ 异步后台摄取：明确推荐

### 为什么 `BackgroundTasks` 绝不能用

它在**同一进程、同一事件循环、同一块内存**里执行，是 fire-and-forget 钩子，**不是任务队列**：
- **进程挂了任务就丢**。有真实事故：**一次 47 秒的部署静默销毁了 400 个 AI 推理任务**，Sentry 无错误、HTTP 也无失败。
- **阻塞事件循环**：CPU 密集或卡住的同步调用会拖慢整个事件循环或占死线程池槽位。
- **无重试、无优先级、无任务 ID、无死信、无观测**；**无 at-least-once 语义**（FastAPI 官方文档自己承认）。
- **停机耦合**：FastAPI 关闭时会等待 BackgroundTasks，可能阻塞发布。

> **唯一适用场景**：缓存预热、审计日志、分析埋点等"丢了也无所谓"且 < 5 秒的副作用。判据是"这个任务丢了会怎样？"——答案必须是"没事"。

### 候选对比（2026-09 实际状态）

| 方案 | 版本/状态 | 优势 | 劣势 |
|---|---|---|---|
| **Celery** | **5.6.3**（2026-03） | 生态最成熟；队列路由/优先级、Canvas、Beat、Flower | 配置最重；**Windows 无官方支持**（prefork 不可用、无 soft time limit、`beat -B` 报错） |
| **Dramatiq** | 2.2.0（2026-06） | 比 Celery 简单，中间件式重试/限流/死信 | **非 async-native**；broker 仅 Redis/RabbitMQ；无内置调度 |
| **ARQ** | **0.28.0**（2026-04） | asyncio 原生、Redis 即 broker+backend、极轻 | ⚠️ **官方声明进入 maintenance-only**（issue #510）；**无优先级队列**、无官方 dashboard |
| **Taskiq** | 0.12.4 | async-first、有 DI、内置 scheduler；IO 密集吞吐比 Celery 高一个数量级 | 生态最年轻；**默认不重试** |
| **SAQ** | 活跃（1 maintainer） | ARQ 进化版：Redis 或 Postgres、有 Web UI、心跳/僵尸清扫、BLMOVE 免轮询（<5ms vs 0.5s） | 单维护者风险 |
| **Hatchet** | SDK 1.40.0 | 仅依赖 Postgres，durable workflow，~10k tasks/s | **需要 Postgres，你的是 MySQL** |
| **Temporal** | 企业级 | 最强持久化执行 | **对中小 RAG 严重过重** |

### ★ 推荐：Celery 5.6.3（Linux）/ ARQ 0.28（Windows 开发）

**首选 Celery 5.6.3**，理由：
1. **负载特征匹配**：PDF 解析（秒~分钟）、embedding（批量）、写 Milvus/Neo4j（重 IO）。**这类时间尺度远超 ARQ 这类轻队列的设计舒适区。**
2. **必须要有优先级与路由**：用户上传的单文档摄取（延迟敏感）与全量重建索引（可跑几小时）**必须分队列分 worker**，否则重建索引会把用户上传堵死。**Celery 的 `task_routes` + `-Q` 最成熟；Dramatiq 与 ARQ 都缺优先级队列。**
3. **可视与可运维性**：Flower 能直接看到"哪篇文档卡住了"，在调优期是刚需。
4. **prefork 模型反而规避了 asyncio 事件循环与同步 PDF 解析库的冲突。**

**代价与规避**：
- **Windows 开发机**：用 `--pool=solo` 或（推荐）在 Docker Linux 容器里跑。**无 soft time limit**、`beat -B` 不可用（beat 必须独立进程）。
- **若坚持 Windows 体验 + 轻量**：选 **ARQ 0.28**，但必须接受 maintenance-only 与无优先级队列。
- **不建议 Dramatiq**（简化程度不足以补偿失去的优先级/生态）；**不推荐 Taskiq / Hatchet / Temporal**。

### 幂等、重试、队列

```python
@app.task(
    bind=True,
    autoretry_for=(httpx.TimeoutException, ConnectionError, MilvusException),
    retry_backoff=True, retry_backoff_max=600, retry_jitter=True,   # jitter 必须开
    max_retries=5,
    acks_late=True,                    # 必须与幂等配套
    reject_on_worker_lost=True,
    soft_time_limit=540, time_limit=600,
)
def ingest_document(self, doc_id: int) -> None: ...
```

- **只重试瞬时错误**（超时、5xx、连接重置）；**永不重试** `ValueError`/校验失败/4xx → 直接进死信
- **幂等靠 DB 唯一约束，不靠应用层 if**：`documents(content_hash, collection_id)` 唯一索引 + `ingestion_jobs.idempotency_key` UNIQUE
- **摄取任务的天然幂等语义**："**按 doc_id 重建该文档的所有 chunk（先删后写）**" 比 "append chunk" 幂等得多

```python
task_routes = {
    "rag.worker.tasks.ingest.ingest_document":        {"queue": "ingest"},
    "rag.worker.tasks.reindex.rebuild_collection":    {"queue": "bulk"},
    "rag.worker.tasks.notify.push_progress":          {"queue": "notify"},
}
```

```bash
celery -A rag.worker.main worker -Q ingest -c 4 --prefetch-multiplier=1 -n ingest@%h
celery -A rag.worker.main worker -Q bulk   -c 1 --prefetch-multiplier=1 -n bulk@%h
celery -A rag.worker.main worker -Q notify -c 8 -n notify@%h
celery -A rag.worker.main beat -l INFO        # 必须独立进程
```

- **`worker_prefetch_multiplier=1`**：长任务必须设 1，否则一个 worker 预取一堆任务导致其他 worker 空闲
- **`--max-tasks-per-child=1000 --max-memory-per-child=200000`**：PDF 解析库的内存泄漏几乎是必然
- **★ `broker_transport_options` 的 `visibility_timeout` 必须大于最长任务时长**（否则长任务会被重新投递 → 重复执行，**这是 RAG 摄取最常见的生产事故**）
- **CPU 密集**：`-c $(nproc)`；**IO 密集**：2–4× CPU

### 进度回传

| 方案 | 机制 | 适用 |
|---|---|---|
| **A. DB 轮询** | worker 写 `documents.status/progress`；前端轮询 | **最简单最可靠，推荐默认**；粒度 5–10% 够用 |
| **B. Redis pub/sub → SSE** | worker `PUBLISH job:{id}`；API `SUBSCRIBE` 转 SSE | 需秒级/细粒度时。**⚠️ pub/sub 无持久化，断连期间进度会丢 → 必须同时落库** |
| C. 前端直连 result backend | 轮询 `AsyncResult` | **不建议**（暴露内部结构） |

**推荐组合：A 为真相源 + B 为体验优化。大结果绝不进 Redis**（用 S3/DB 存路径）。

来源：[400 个 AI 任务丢失事故复盘](https://python.plainenglish.io/python-fastapi-backgroundtasks-lost-400-ai-inference-jobs-on-deploy-a7d57d5224cf)、[Celery vs ARQ vs Dramatiq 2026](https://dev.to/datanestdigital/background-jobs-in-python-celery-vs-rq-vs-dramatiq-vs-arq-2026-decision-guide-37m6)、[ARQ maintenance 声明](https://github.com/python-arq/arq/blob/main/README.md)、[Celery changelog](https://docs.celeryq.dev/en/main/changelog.html)

## 6.7 文件上传

### 硬约束

- **`python-multipart` 必须显式钉 `>=0.0.32`** —— 修补了 **CVE-2026-53538 / 53539 / 53540**。**但 FastAPI 自身仍 pin 在 `^0.0.22`** → 依赖传递可能导致实际解析到低于安全线的版本。**必须在自己的 `pyproject.toml` 里显式声明。**
- **FastAPI 没有内置最大上传限制**。限额必须在**三层**做：
  1. **网关层**：Nginx `client_max_body_size 50m`（第一道闸，最省资源）
  2. **应用层**：边读边累加字节数，超限即 abort
  3. **存储层**：MinIO/S3 presigned policy 的 **`content-length-range`**
- **绝不相信 `Content-Length`**（客户端可伪造，chunked 传输根本不发）

### 流式落盘（防 OOM）

```python
CHUNK = 1024 * 1024

@router.post("/documents", status_code=202)
async def upload_document(file: Annotated[UploadFile, File()],
                          session: SessionDep, settings: SettingsDep) -> JobAccepted:
    max_bytes = settings.upload_max_mb * 1024 * 1024
    tmp = Path(tempfile.mkdtemp()) / f"{uuid.uuid4().hex}.part"
    total = 0
    try:
        async with aiofiles.open(tmp, "wb") as out:
            while chunk := await file.read(CHUNK):        # ★ 必须带 size 参数
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, "file too large")
                await out.write(chunk)
        mime = sniff_mime(tmp)                            # ★ magic bytes，不信 content_type
        if mime not in ALLOWED:
            raise HTTPException(415, "unsupported media type")
        object_key = f"uploads/{uuid.uuid4().hex}{suffix_for(mime)}"
        await asyncio.to_thread(storage.upload_file, tmp, object_key)
        ...
    finally:
        tmp.unlink(missing_ok=True)                       # ★ 失败必须清临时文件
```

**必须记住的坑**：
- **`await file.read()` 不传 size = 整个文件进 RAM**（`UploadFile` 底层是 `SpooledTemporaryFile`，但一次全读会绕过这个保护）。**永远 `file.read(CHUNK)`。**
- **文件只能读一次**，重读前必须 `await file.seek(0)`。
- **用 `aiofiles` 写盘**，不要同步 `open()` 阻塞事件循环。
- **保存路径用 UUID，绝不用客户端文件名**（路径穿越 + 覆盖风险）。
- **真实类型用 magic bytes 判断**：PDF `%PDF-`、DOCX `PK\x03\x04`（**注意 DOCX 与 XLSX/PPTX/zip-bomb 共享 PK 头**，需进一步检查 `[Content_Types].xml`）。
- **不用 `/tmp` 做持久化**（多实例部署必然失效）。

### MinIO / S3

```python
presigned = s3.generate_presigned_post(
    Bucket=BUCKET, Key=object_key,
    Fields={"Content-Type": content_type},
    Conditions=[{"Content-Type": content_type},
                ["content-length-range", 1, 100 * 1024 * 1024]],   # ★ 存储层硬限额
    ExpiresIn=3600,
)
```

- **> 100 MB 一律用 presigned URL 前端直传**，字节永不经过 FastAPI
- **`minio-py` 的 `put_object` 必须传 `length`**：SDK 不接受未知长度的流，不传会**挂住**
- **`get_object` 返回惰性流持有 TCP 连接**：必须 `response.close()` + `response.release_conn()`
- MinIO 客户端**启动时初始化单例挂 `app.state`**，不要每请求新建

### ★ 解析栈推荐

**2026 年的关键变化：PyMuPDF 的 AGPL-3.0 是商业闭源/SaaS 项目的硬伤。** Artifex 的 AGPL 含**网络条款**——即使不分发代码，只要用户通过网络与之交互，也必须开源整个应用。

| 库 | 许可 | 表格能力 | 速度 | 结论 |
|---|---|---|---|---|
| **PyMuPDF / pymupdf4llm** | **AGPL-3.0** ⚠️ | 基础 | 最快（121 页 1.2s） | **闭源/SaaS 禁用** |
| **pdfplumber** | **MIT** | 中等，**有显式 CJK 支持** | 慢 | **中文场景的 MIT 首选** |
| **Docling（IBM）** | **MIT** | **最强**（TEDS 97.9%, TableFormer） | 慢：CPU 上 121 页 10-K **240s 超时** | 复杂版式/扫描件首选，**需 GPU 才可批量** |
| **MarkItDown（MS）** | MIT | **最弱**：表格/公式碎裂，**不产 Markdown 标题**（破坏按标题分块） | 快 | 仅适合简单文本型 PDF |
| **unstructured** | Apache-2.0 核心 | 中等 | 中等 | 可作统一入口，但**内部可能拉起 PyMuPDF（AGPL 传染）→ 必须审计** |
| **python-docx** | MIT | Word 表格可靠（`doc.tables`） | 快 | **DOCX 首选** |

**★ 分层路由**：

```python
def parse(path: Path, mime: str) -> ParsedDoc:
    if mime == "application/pdf":
        quick = pdfplumber.open(path)
        if is_text_native(quick):          # 字符密度 + 字体覆盖率启发式
            return parse_with_pdfplumber(quick)     # 数字原生 PDF，含中文表格
        return parse_with_docling(path)             # 扫描件/复杂版式（+OCR）
    if mime.endswith("wordprocessingml.document"):
        return parse_with_python_docx(path)
    if mime == "text/markdown":
        return parse_markdown(path)                 # 直接按标题切分
```

- **PDF 是"两种东西"**（数字原生 vs 扫描件），**先分流**，不要让所有 PDF 都走重量级模型
- **Docling 必须异步化 + 设超时**，并**预下载模型权重进镜像**（避免运行时拉网）
- **表格保真度直接决定 RAG 质量**：建议把表格单独抽成 Markdown 表格 + `chunk_type="table"` metadata，检索时用不同 prompt
- ⚠️ **中文场景下 Docling / pdfplumber / MarkItDown 的 head-to-head 基准在检索中未找到** → **这是一个需要你自己做小规模评测（10–20 份真实中文文档）的决策点**

来源：[2026 PDF 抽取器对比](https://pdfmux.com/blog/pdf-extractor-comparison-2026/)、[PyMuPDF AGPL 分析](https://pdfmux.com/blog/alternatives/pymupdf-alternatives/)、[PDF→Markdown RAG 基准](https://dev.to/jeromebuilds/i-benchmarked-5-open-source-pdf-to-markdown-tools-for-rag-on-real-documents-2026-4heh)、[FastAPI 文件上传避坑](https://www.cnblogs.com/ymtianyu/p/19936310)、[hermes-agent 安全提交](https://github.com/NousResearch/hermes-agent/commit/a1782ea283cc51c6ef24243bed1654b02c562f17)

## 6.8 鉴权

**★ 用 PyJWT，弃 python-jose**：
- **`python-jose` 事实上已无人维护**：最后发布 **2022-12**，受 **CVE-2025-61152** 影响（**接受 `alg=none`，可伪造任意 token**）
- **`PyJWT` 活跃**，月下载 9000 万+（python-jose 约 1500 万），支持 HS/RS/ES/PS/EdDSA，`PyJWKClient` 支持 JWKS 轮转
- **迁移成本极低**：`jwt.encode`/`jwt.decode` 直接沿用；`JWTError` → `PyJWTError`；`python-jose[cryptography]` → `PyJWT[crypto]`
- **唯一例外**：需要 **JWE（加密 token）** 时 PyJWT 不支持——但本项目不需要

**★ 密码哈希用 `pwdlib`，弃 passlib**：
- **passlib 是 "semi-abandoned"**：**bcrypt 5.x 兼容性破裂**，社区被迫 pin `passlib==3.2.2`
- 推荐顺序：**`pwdlib`**（FastAPI 官方文档已采用，`PasswordHash.recommended()` 封装 argon2/bcrypt）> `argon2-cffi` > `bcrypt>=4`

```python
from pwdlib import PasswordHash
password_hash = PasswordHash.recommended()   # argon2id
hashed = password_hash.hash("s3cret")
ok = password_hash.verify("s3cret", hashed)
```

| 项 | 推荐 | 理由 |
|---|---|---|
| 算法 | **RS256**（多服务）或 HS256（单服务） | RS256 便于只分发公钥 |
| access TTL | **15 分钟** | 泄露窗口小 |
| refresh TTL | 14 天 | |
| **Refresh 轮转** | **每次刷新签发新 refresh + 旧 refresh 立即失效**；旧 refresh 被二次使用时**吊销整条链**（判定被盗） | |
| 存储 | refresh token 哈希入 DB（`refresh_tokens` 表：jti、user_id、expires_at、revoked_at、replaced_by） | 才能实现吊销与"登出所有设备" |
| 携带 | access 走 `Authorization: Bearer`；refresh 走 **HttpOnly + Secure + SameSite=Lax Cookie** | refresh 不能放 JS 可读处 |
| 服务间 | **独立 API Key**（`X-API-Key`），哈希存 DB，带 scope 与速率配额 | 不要用用户 JWT 做服务间调用 |

```python
payload = jwt.decode(token, settings.jwt_secret.get_secret_value(),
                     algorithms=[settings.jwt_alg],
                     options={"require": ["exp", "sub", "typ", "jti"]})
if payload["typ"] != "access": raise InvalidCredentialsError()
```

来源：[PyJWT vs python-jose](https://www.iamdevbox.com/posts/pyjwt-vs-python-jose-choosing-the-right-python-jwt-library/)、[JWT in Python 指南](https://jsonic.io/guides/jwt-python)

## 6.9 限流

| | slowapi | fastapi-limiter |
|---|---|---|
| 最新版本 | **0.1.10（2026-06-13）**，最近提交 2026-07 | 0.2.0（2026-02），最近提交 2026-02 |
| 引擎 | **`limits`**（Flask-Limiter 同源，成熟） | PyrateLimiter |
| API | `@limiter.limit("5/minute")` 装饰器 | `Depends(RateLimiter(...))` |
| 存储 | `storage_uri="redis://..."`，支持 Redis/Memcached/Valkey/Cluster/Sentinel | **0.2.0 起移除了内置 Redis** |

**⚠️ `fastapi-limiter` 0.2.0 是一次破坏性重写**：删除 `FastAPILimiter` 类、**不再内置 Redis**、API 全变、不再有全局启动初始化。有项目明确"保留 0.1.6 不升级"，社区维护者甚至建议"自己实现比适配 0.2.0 更省事"。

**★ 推荐 `slowapi` + 显式 Redis storage**：

```python
limiter = Limiter(
    key_func=user_or_ip_key,          # ★ 见下方陷阱
    storage_uri=settings.redis_url,   # ★ 多 worker 必须走 Redis
    default_limits=["200/minute"],
)
app.state.limiter = limiter

@router.post("/chat/stream")
@limiter.limit("10/minute")           # 廉价端点
@router.post("/documents")
@limiter.limit("20/hour")             # 昂贵端点
```

**两个库共有的陷阱**：
1. **`key_func` 配错会导致"全局限流"而非"按用户限流"** —— 最常见的生产事故
2. **静默回退到进程内内存存储**：Redis 配错时不报错，单机看起来正常，**多副本时限额被放大 N 倍** → **必须加启动自检**

**昂贵 LLM 端点的按用户配额（双层）**：
- **第一层（QPS）**：slowapi + Redis，`10/minute`
- **第二层（token 预算）**：**自研 Redis 计数器**（键 `quota:{user_id}:{yyyy-mm-dd}`，`INCRBY tokens_used`，在 LLM 调用后按实际 usage 扣减；用 **Lua 脚本**保证"检查+扣减"原子性）。超额返回 429 + `Retry-After` + RFC 9457 扩展字段 `quota_reset_at`

**为什么不用 fastapi-limiter**：0.2.0 丢了 Redis 内置支持，你仍要自己接 PyrateLimiter 的 RedisBucket，工作量与自研接近，却多一个依赖。

来源：[限流库整合研究](https://fastapi-rbac.mnfprofile.com/internal/research/rate-limiting-library-consolidation/)、[slowapi ADR](https://fastapi-rbac.mnfprofile.com/adr/0002-slowapi-sole-http-rate-limit/)、[fastapi-best-architecture 讨论](https://github.com/fastapi-practices/fastapi-best-architecture/discussions/70)

## 6.10 错误处理与结构化日志

**★ FastAPI 不原生支持 RFC 9457**。默认 422 用 `application/json` + `detail`，而 `detail` 是结构化对象数组 —— 这本身就是"FastAPI 自己的格式，不符合 RFC"。社区 PR #10370 仍未落地。**推荐自研 typed exception + handler**（错误契约是要长期维护的东西，第三方包价值有限）。

```python
# domain/errors.py —— 纯领域层，无 FastAPI 依赖
class DomainError(Exception):
    status = 500; type_slug = "internal-error"; title = "Internal error"
    def __init__(self, detail: str = "", **extensions):
        self.detail = detail; self.extensions = extensions

class NotFoundError(DomainError):
    status, type_slug, title = 404, "not-found", "Resource not found"
class QuotaExceededError(DomainError):
    status, type_slug, title = 429, "quota-exceeded", "Quota exceeded"
class DocumentParseError(DomainError):
    status, type_slug, title = 422, "document-parse-failed", "Document parse failed"

# api/errors.py
PROBLEM_BASE = "https://api.example.com/problems/"

def to_problem(exc: DomainError, request: Request) -> dict:
    body = {"type": f"{PROBLEM_BASE}{exc.type_slug}", "title": exc.title,
            "status": exc.status, "detail": exc.detail or None,
            "instance": str(request.url.path),
            "request_id": request.state.request_id}
    body.update(exc.extensions)
    return {k: v for k, v in body.items() if v is not None}

@app.exception_handler(DomainError)
async def _(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(exc.status, to_problem(exc, request),
                        media_type="application/problem+json",
                        headers={"X-Request-ID": request.state.request_id})

@app.exception_handler(Exception)
async def _(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error")                 # 堆栈只进日志
    problem = DomainError("Internal error")             # ★ 绝不外泄细节
    return JSONResponse(500, to_problem(problem, request),
                        media_type="application/problem+json")
```

**约定**：
- **problem type URI 放自己域名下**，kebab-case slug，稳定可文档化
- **抛 typed problem 异常，绝不抛裸 `HTTPException`**
- **★ 不泄漏 LLM/provider 错误**：OpenAI/Anthropic 的异常常含内部 request-id、组织 ID、甚至 prompt 片段。**必须映射**：
  ```python
  except openai.RateLimitError as e:
      raise QuotaExceededError("Upstream model quota exhausted",
                               retry_after=e.response.headers.get("retry-after"))
  ```
- **`request_id` 传播**：中间件从 `X-Request-ID` 取（**大小写不敏感**），没有则生成 `uuid4()`——**不要默认成 `"unknown"`**
- 生产对 5xx **剥离 extensions**

**structlog 26.1.0**：

```python
def configure_logging(settings: Settings) -> None:
    shared = [
        structlog.contextvars.merge_contextvars,     # ★ 必须最前
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_secrets,                              # ★ 必须在 JSONRenderer 之前
        otel_trace_processor,                        # 注入 trace_id/span_id
    ]
    structlog.configure(
        processors=shared + [structlog.processors.JSONRenderer()
                             if settings.env != "dev" else structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

@app.middleware("http")
async def correlation_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    structlog.contextvars.clear_contextvars()        # ★ 防串请求
    structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path, method=request.method)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()    # ★ 必须在 finally
    response.headers["X-Request-ID"] = rid
    return response
```

**要点**：
- **`contextvars` 是 async-safe（每个 asyncio task 隔离），但上下文会泄漏到下一个请求** → **必须 `try/finally` + `clear_contextvars()`**
- **`merge_contextvars` 必须在 processor 链最前，`JSONRenderer` 必须最后**，脱敏 processor 必须在 JSONRenderer 之前
- **中间件顺序**：`add_middleware` **最后调用的在最外层、最先执行** → 把 correlation 中间件**最后添加**
- **OTel 关联**：`FastAPIInstrumentor.instrument_app(app)` + 自定义 processor 从当前 span 注入 `trace_id`/`span_id` → **日志 ↔ trace ↔ LLM 调用链三者可互相跳转**

来源：[RFC 9457 FastAPI 实现](https://raw.githubusercontent.com/yonatangross/orchestkit/refs/heads/main/plugins/ork/skills/api-design/examples/fastapi-problem-details.md)、[FastAPI Telemetry Checklist 2026-08](https://devstacktips.com/backend-development/2026/08/28/fastapi-telemetry-checklist-implementing-structured-logs-prometheus-metrics-and-opentelemetry-tracing/)、[Structured Logging with Request-ID](https://ossaihub.com/code/structured-logging-request-id/)

## 6.11 健康检查与 OpenAPI

```python
health = APIRouter(include_in_schema=False)     # ★ 不进 OpenAPI

@health.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness：只证明进程活着、事件循环没卡死。禁止 I/O，< 1ms。"""
    return {"status": "ok"}

@health.get("/readyz")
async def readyz(request: Request, response: Response) -> dict:
    """Readiness：有界地并发探测依赖。全通过 200，否则 503。"""
    if not getattr(request.app.state, "ready", False):
        response.status_code = 503
        return {"status": "draining"}
    checks = await asyncio.gather(
        probe_mysql(request.app.state.db, timeout=2.0),
        probe_redis(request.app.state.redis, timeout=2.0),
        probe_milvus(request.app.state.milvus, timeout=2.0),
        probe_neo4j(request.app.state.neo4j, timeout=2.0),
        return_exceptions=True,
    )
    ok = all(c is True for c in checks)
    response.status_code = 200 if ok else 503
    return {"status": "ok" if ok else "degraded",
            "checks": dict(zip(("mysql","redis","milvus","neo4j"), checks))}
```

**铁律**：
- **liveness 绝不能查依赖**。否则 MySQL 抖动 10 秒 → **所有副本同时被重启**，把可恢复的依赖故障放大成自伤事故。
- **把 `livenessProbe` 指向 `/readyz` 是经典错误配置**：冷启动 30–120s 内 `/readyz` 返回 503 → **永久重启循环**。
- **`/readyz` 必须幂等且无副作用**（不能因探测而触发模型加载或重连）；**每个依赖检查用硬超时**（`asyncio.wait_for(..., 2.0)`）并**并发执行** —— 否则一个慢依赖能挂到 ~130s OS TCP 超时。
- **优雅停机**：开始关停时 `/readyz` 立即 503（给 LB 摘流量的排空信号），而 **`/healthz` 在排空期间继续 200**。
- **探测端点排除在鉴权、限流、Host/Origin 校验之外**（但要放内网或加 IP 白名单）。
- **日志噪声**：5s 间隔 × 3 端点 × 10 实例 ≈ **每天 51.8 万条健康检查日志** → 日志中间件必须 `skip_paths(["/healthz","/readyz","/metrics"])`。
- **Docker `HEALTHCHECK` 用退出码 0/1，不是 HTTP 200/503。**

**K8s 参数**：liveness `initialDelaySeconds: 10`、`periodSeconds: 5–15`、`timeoutSeconds: 2`、`failureThreshold: 3`；readiness `periodSeconds: 10–15`、`timeoutSeconds: 1–5`；慢启动加 `startupProbe`（`failureThreshold: 30`）。

**OpenAPI**：

```python
app = FastAPI(
    title="RAG Agent API", version="1.0.0",
    openapi_url="/openapi.json" if settings.env != "prod" else None,
    docs_url="/docs" if settings.env != "prod" else None,
    redoc_url=None, lifespan=lifespan,
)
```

- **`response_model` 必须显式声明**，配 `response_model_exclude_none=True`；用 `responses={404: {"model": ProblemDetail, "content_type": "application/problem+json"}}` 把错误契约写进 OpenAPI
- **`operation_id` 要显式指定**（否则生成的 SDK 方法名会随函数改名而漂移）
- **生产默认关闭文档**（`docs_url=None`）；若需对外，拆一个只含公开路由的独立 app 并加鉴权 —— **不要在主 API 上开 `/docs`**

来源：[LLM Pod 健康检查](https://ossaihub.com/code/healthcheck-readiness-llm/)、[Health Check ADR](https://docs.proteanhq.com/adr/0012-health-check-architecture/)

## 6.12 流式响应

| | **SSE** | **WebSocket** |
|---|---|---|
| 方向 | 单向 | 双向 |
| 重连 | **浏览器自动 + `Last-Event-ID` 断点续传** | 需自行实现 |
| 调试 | **curl 可调** | 需专用工具 |
| 基础设施 | 现有 HTTP/JWT/Nginx 全兼容 | 需单独处理 Upgrade/粘性会话 |
| 中途干预 | ❌ 需另开通道 | ✅ 原生 |

**★ 推荐 SSE**：**LangGraph 的 `astream_events` 返回 async generator，可以直接 yield 给 FastAPI 的 `StreamingResponse`，无需转换层**；同时复用现有 JWT + Nginx。**只有当产品明确需要"流式中途打断/追问"时才考虑 WebSocket。**

⚠️ **SSE 协议注入风险**：社区指出 **`format_sse_event()` 不做 `ServerSentEvent` 那套校验**，传用户可控输入会导致注入：

```python
format_sse_event(event="legit\ndata: injected", data_str="safe")
# → b'event: legit\ndata: injected\ndata: safe\n\n'
```

维护者定性其为"未文档化、应视为内部工具函数"。**结论：不要直接调用 `format_sse_event()` 传用户输入；用 `ServerSentEvent` 构造（它有 `_check_event_single_line` / `_check_id_valid`），或自行对换行做转义。**

```python
@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, agent: AgentDep, user: CurrentUserDep):
    async def gen():
        config = {"configurable": {"thread_id": body.thread_id}}
        try:
            async for ev in agent.astream_events(
                {"messages": [("user", body.question)]}, config=config, version="v2"):
                e = ev["event"]
                if e == "on_chat_model_stream":
                    chunk = ev["data"]["chunk"]
                    if chunk.content:
                        yield {"event": "token",
                               "data": TokenEvent(text=chunk.content).model_dump_json()}
                elif e == "on_tool_start":
                    yield {"event": "tool_start",
                           "data": ToolStartEvent(tool=ev["name"],
                                                  args=ev["data"].get("input", {})).model_dump_json()}
                elif e == "on_chain_end" and ev.get("name") == "LangGraph":
                    yield {"event": "done", "data": DoneEvent(...).model_dump_json()}
        except asyncio.CancelledError:
            raise                              # ★ 客户端断连，必须向上抛
        except Exception:
            logger.exception("stream_failed")
            yield {"event": "error",
                   "data": ErrorEvent(code="internal", message="生成失败").model_dump_json()}
    return EventSourceResponse(gen(), ping=15)   # ★ 心跳
```

**LangGraph 流式 API 现状（1.2.x）**：
- **`astream_events(version="v2")` 是 2026 年的生产主力**
- **1.2 新增 `stream_events(..., version="v3")`**，提供**类型化 projection**（beta，仍可能变）→ **本项目先用 v2（稳定），v3 列为观察项**
- **`astream_log()` 在 1.0 已软弃用，计划 2.0 移除**
- `stream_mode`：`"messages"`（token 级，聊天 UI 首选，需 LangGraph ≥1.1）／`"updates"`（每节点一个事件，适合进度）／`"values"`（每节点全量 state，**不适合浏览器**）／`"custom"`（节点内 `writer(...)`，适合推出 RAG 中间结果）

**★ 生产必备响应头（不可妥协）**：

```python
headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
```

**没有这三个头，本地能跑、放到 Nginx / Cloud Run / Cloudflare 后面就会"卡住"。** 另外：
- **SSE 心跳 `: heartbeat\n\n` 每 15 秒一次**（扛住 ~60s 空闲超时）；`sse-starlette` 的 `ping=15` 即此
- **Nginx 必须 `proxy_buffering off`**，`proxy_read_timeout` 调到 120s+
- **★ 服务端过滤事件**：一次 60 秒的 agent 运行可能产生 **3000+ 事件**。**只转发 `on_chat_model_stream`（及可选 `on_tool_start`/`on_tool_end`）**，丢弃 `on_chain_*`、`on_parser_*`、`on_prompt_*`、`on_retriever_*` —— 否则浏览器标签页会冻死
- **异步端点里绝不能调同步 `graph.stream()`**（阻塞事件循环）

来源：[LangGraph 生产级流式传输指南](https://skillui.com/zh/skill/show/jeremylongshore/claude-code-plugins-plus-skills/langchain-langgraph-streaming)、[部署 LangGraph Agent 为生产 API](https://markaicode.com/howto/how-to-deploy-langgraph-agents/)、[SSE 协议注入 Discussion #15648](https://github.com/fastapi/fastapi/discussions/15648)、[SSE vs WebSocket 决策](https://github.com/f-lab-edu/on-seoul/wiki/6.-Communication-Protocal)

## 6.13 LangGraph checkpointer

| 后端 | 包 | 类 | 用途 |
|---|---|---|---|
| Memory | `langgraph` | `InMemorySaver` | **仅开发** |
| SQLite | `langgraph-checkpoint-sqlite` | `SqliteSaver` | 本地开发 |
| **MySQL** | **`langgraph-checkpoint-mysql[asyncmy]`** | `AIOMySQLSaver` | **★ 与你的栈匹配** |
| Postgres | `langgraph-checkpoint-postgres` | `AsyncPostgresSaver` | 生产durability |
| Redis | `langgraph-checkpoint-redis` | `RedisSaver` | 低延迟读密集 |

- **绝不用 MemorySaver 生产**：进程内 RAM，重启丢失，多 worker 不共享
- **`thread_id` 是恢复键**，每次调用必须设 `config={"configurable": {"thread_id": ...}}`；**按租户隔离：`tenant:user:conversation`**
- **启动时和每次 LangGraph 升级后跑 `checkpointer.setup()`**（幂等）—— 跳过会导致升级后静默返回空 thread
- **保持 checkpoint 小（< ~50KB）**，用外部引用而非内容（S3 key，而不是 PDF 本身）。10 步图 ≈ 10 行/run，10k runs/day ≈ 10 万行/天，MySQL 轻松
- **⚠️ CVE-2025-67644（2026-03，CVSS 7.3）**：`langgraph-checkpoint-sqlite` 的 SQL 注入，**3.0.1 修复**（仅影响 SQLite 自托管）
- 1.0.0+ 的 `MemorySaver` 导入路径变更为 `langgraph_core.checkpoint.memory`（**未证实**，建议验证）

来源：[LangGraph 1.0 GA: Checkpoints](https://callsphere.ai/blog/vw1g-langgraph-1-stable-checkpoints-production)、[LangGraph 状态持久化](https://activewizards.com/blog/langgraph-state-management-checkpointing-recovery-and-the-persistence-layer-decision/)、[LangGraph Persistence Backends Spec](https://caipe.io/docs/0.5.69/specs/langgraph-redis-persistence/spec/)

---

# 7. 部署拓扑

## 7.1 镜像 tag（2026-09）

| 服务 | 镜像 | 说明 |
|---|---|---|
| Milvus | `milvusdb/milvus:v2.6.x`（**锁 patch**） | **3.0 GA 2026-07-29 但 pymilvus 3.0 弃用 ORM、有导入破坏 → 先 staging 验证** |
| etcd | `quay.io/coreos/etcd:v3.5.18` | 2.6 线实测 tag |
| MinIO | `minio/minio:RELEASE.2024-12-18T13-15-44Z` | **必须 ≥ 此版本**（2024-05~10 版本有内存泄漏等问题） |
| Neo4j | `neo4j:5.26-community-ubi9` 或 `neo4j:5.26.30-community` | CalVer 备选 `neo4j:2026.06-community` |
| MySQL | `mysql:8.4` | LTS 至 ~2032 |
| Redis | `redis:8-alpine` | 8.0+ 才有 AGPLv3 选项（7.4 **不含** AGPL） |
| Attu | `zilliz/attu:v2.6` | 可选 UI |
| Phoenix | `arizephoenix/phoenix` | 单容器可观测 |

## 7.2 compose.yaml 草图（关键片段）

```yaml
name: rag
x-logging: &default-logging
  driver: json-file
  options: { max-size: "20m", max-file: "5" }

services:
  mysql:
    image: mysql:8.4
    command: >
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_0900_ai_ci
      --innodb-buffer-pool-size=1G
      --max-connections=300
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:?err}
      MYSQL_DATABASE: rag
      MYSQL_USER: rag
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:?err}
    volumes: [mysql_data:/var/lib/mysql]
    healthcheck:
      test: ["CMD-SHELL", "mysqladmin ping -h 127.0.0.1 -u root -p$$MYSQL_ROOT_PASSWORD --silent"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 60s              # ★ 冷启动慢
    ports: ["127.0.0.1:3306:3306"]   # ★ 只绑本机
    networks: [backend]
    logging: *default-logging

  neo4j:
    image: neo4j:5.26-community-ubi9
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:?err}
      NEO4J_server_memory_heap_initial__size: 2G     # ★ 双下划线
      NEO4J_server_memory_heap_max__size: 2G
      NEO4J_server_memory_pagecache_size: 2G
      NEO4J_server_default__listen__address: 0.0.0.0
      # 开发环境可加：NEO4J_PLUGINS: '["apoc","graph-data-science"]'
    volumes: [neo4j_data:/data, neo4j_logs:/logs]
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p $$NEO4J_PASSWORD 'RETURN 1' || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 90s              # ★ JVM 冷启动慢，必须给足
    ports: ["127.0.0.1:7474:7474", "127.0.0.1:7687:7687"]
    networks: [backend]

  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
      ETCD_SNAPSHOT_COUNT: "50000"
    command: >
      etcd -advertise-client-urls=http://etcd:2379
           -listen-client-urls=http://0.0.0.0:2379 --data-dir /etcd
    volumes: [etcd_data:/etcd]
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks: [backend]

  minio:
    image: minio/minio:RELEASE.2024-12-18T13-15-44Z
    environment:
      MINIO_ROOT_USER: ${MINIO_USER:?err}
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD:?err}
    command: minio server /minio_data --console-address ":9001"
    volumes: [minio_data:/minio_data]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    ports: ["127.0.0.1:9000:9000", "127.0.0.1:9001:9001"]
    networks: [backend]

  milvus:
    image: milvusdb/milvus:v2.6.18
    command: ["milvus", "run", "standalone"]
    security_opt: ["seccomp:unconfined"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: ${MINIO_USER:?err}
      MINIO_SECRET_ACCESS_KEY: ${MINIO_PASSWORD:?err}
      MQ_TYPE: woodpecker       # 2.6 默认，不再依赖 Kafka/Pulsar
    volumes:
      - milvus_data:/var/lib/milvus
      - ./configs/milvus.yaml:/milvus/configs/milvus.yaml:ro   # ★ authorizationEnabled
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 5
      start_period: 90s
    ports: ["127.0.0.1:19530:19530", "127.0.0.1:9091:9091"]
    depends_on:
      etcd:  { condition: service_healthy }
      minio: { condition: service_healthy }
    networks: [backend]

  migrate:
    build: { context: ., target: runtime }
    command: ["alembic", "upgrade", "head"]
    env_file: [.env]
    depends_on:
      mysql: { condition: service_healthy }
    restart: "no"                    # ★ 一次性 init job
    networks: [backend]

  api:
    build: { context: ., target: runtime }
    command: >
      gunicorn rag.main:app -k uvicorn_worker.UvicornWorker
        -w 4 --bind 0.0.0.0:8000 --timeout 120 --graceful-timeout 30
        --preload --proxy-headers --forwarded-allow-ips='*'
    env_file: [.env]
    depends_on:
      mysql:  { condition: service_healthy }
      redis:  { condition: service_healthy }
      neo4j:  { condition: service_healthy }
      milvus: { condition: service_healthy }
      migrate: { condition: service_completed_successfully }   # ★ 迁移成功才启动
    ports: ["127.0.0.1:8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 40s
    networks: [backend]

  worker-ingest:
    build: { context: ., target: runtime }
    command: ["celery","-A","rag.worker.main:celery_app","worker",
              "-Q","ingest","-c","2","--prefetch-multiplier=1",
              "--max-tasks-per-child=200","-l","INFO"]
    env_file: [.env]
    depends_on:
      api:   { condition: service_healthy }   # 确保 Milvus 已 load_collection
      redis: { condition: service_healthy }
    networks: [backend]

  worker-bulk: { ... "-Q","bulk","-c","1" ... }
  beat:        { ... "beat","-l","INFO" ... }

networks:
  backend: { driver: bridge, name: rag_backend }
volumes:
  mysql_data: {} 
  redis_data: {}
  neo4j_data: {}
  neo4j_logs: {}
  etcd_data: {}
  minio_data: {}
  milvus_data: {}
```

## 7.3 compose 关键实践

1. **长格式 `depends_on` + `condition` 是唯一可靠的启动顺序控制**。短格式 `depends_on: [db]` **只保证创建顺序，不保证就绪** —— DB 还在初始化时应用就起来了，报 `ECONNREFUSED`，而 `docker compose ps` 看起来一切正常。
   - `service_started`（默认）／**`service_healthy`**（有状态依赖必用）／**`service_completed_successfully`**（一次性迁移）
2. **`start_period` 是最常被遗漏的设置**。MySQL / JVM 冷启动可达 60–90s，太短会被**误判为 unhealthy**。
3. **`healthcheck.test` 必须是数组**（`["CMD-SHELL", "..."]`）。
4. **健康检查在容器内部执行**：`curl http://db:5432` 在 db 自己的 check 里无效；要确认镜像里有对应客户端（Neo4j 自带 `cypher-shell`；Alpine 需 `apk add curl`）。
5. **IPv4/IPv6 陷阱**：Alpine 的 BusyBox wget 可能先解析 `::1` → 用 `127.0.0.1` 而非 `localhost`。
6. **`service_healthy` 只在启动时生效**：运行期依赖变 unhealthy **不会**重启依赖方。应用层仍需重试与优雅降级。
7. **`deploy.resources.limits` 自 Compose v2.15+ 在非 Swarm 模式下也可用**；`cpus` 要加引号（`"2.0"`）；**不要混用 `deploy.resources` 与旧式 `mem_limit`/`cpus`**。
8. **`docker compose up --wait --wait-timeout 600`** 可替代手写 sleep 循环。
9. **顶层 `version:` 键已废弃**，文件名用 `compose.yaml`。
10. **★ 端口只绑 `127.0.0.1`**：**Milvus 19530 与 Neo4j 7687 默认无认证**，绝不能直接暴露公网。

来源：[Compose depends_on 与 healthcheck 2026](https://dev.to/davidtio/docker-compose-dependson-and-health-checks-that-actually-protect-startup-2026-1bc1)、[depends_on 不够用](https://dev.to/jtorchia/dependson-isnt-enough-servicehealthy-in-compose-52g1)、[官方 Milvus standalone compose](https://github.com/milvus-io/milvus/blob/master/deployments/docker/standalone/docker-compose.yml)、[用 Compose 配置 Milvus](https://milvus.io/docs/zh/v2.6.x/configure-docker.md)

## 7.4 资源规格

### 开发笔记本（16 GB）—— 非常紧

| 服务 | 建议 |
|---|---|
| Milvus standalone（含 etcd+MinIO） | **4–6 GB**（官方**最低 8 GB / 推荐 16 GB**） |
| Neo4j | 2 GB（heap 768M / pagecache 1G） |
| MySQL | 1–1.5 GB（`innodb_buffer_pool_size=512M~1G`） |
| Redis | 256 MB |
| API (uvicorn) | 1 GB |
| Worker ×2 | 1.5 GB |
| **合计** | **约 11–13 GB** + OS |

**★ 开发降级方案（强烈推荐）**：

| 生产组件 | 开发替代 | 代价 |
|---|---|---|
| Milvus standalone | **Milvus Lite**（`pymilvus[milvus-lite]`，文件型） | 单进程、无 auth、无副本（限制见下） |
| Neo4j | 容器但 heap/pagecache 各 512M | 图查询变慢 |
| MySQL | **SQLite + aiosqlite** | 方言差异 —— **集成测试不要用 SQLite** |
| Redis | **`fakeredis`** | pub/sub 与 Lua 行为不完全一致 |
| MinIO | 本地文件系统（同一 `StorageBackend` 接口） | 无 presigned URL 演练 |
| Celery | `--pool=solo`（Windows 必须） | 无并发/无重试演练 |

**降级的关键纪律**：所有外部依赖都要有 **Protocol/ABC 抽象 + dev/prod 双实现**（`VectorStore`、`GraphStore`、`ObjectStorage`、`Queue`），通过 `app.dependency_overrides` 或 settings 切换。**否则"开发用 Lite、生产用 standalone"会在 API 差异上翻车。**

### 小生产 VM（32 GB）

| 服务 | 内存 | CPU |
|---|---|---|
| Milvus standalone + etcd + MinIO | 8–12 GB | 4 |
| Neo4j | 6 GB（heap 2G / pagecache 2G） | 2 |
| MySQL | 4 GB（`innodb_buffer_pool_size=2G`） | 2 |
| Redis | 2 GB | 1 |
| API × 2 副本 | 2 GB × 2 | 1 × 2 |
| Worker-ingest + bulk | 4 GB + 4 GB | 4 + 2 |
| Phoenix | 1 GB | 1 |
| **合计** | **约 27–29 GB** | |

**★ 若在这台机器上自托管 Langfuse v3（需 ClickHouse + Postgres + Redis + S3），建议加 8 GB**，否则改用 Phoenix。

### Neo4j 内存（env var 命名规则）

前缀 `NEO4J_`，点变下划线，**嵌套分隔符写成双下划线 `__`**：
- `server.memory.heap.max_size` → **`NEO4J_server_memory_heap_max__size`**
- `server.memory.pagecache.size` → **`NEO4J_server_memory_pagecache_size`**

**镜像默认值只有 512M / 512M**（为了一台机器跑多容器），**生产必须改**。

| 机器 RAM | heap | pagecache |
|---|---|---|
| 16 GB | 4 GB | 8 GB |
| **32 GB** | **8 GB** | **16 GB** |
| 64 GB | 12–16 GB | 32–40 GB |

- heap 建议取可用内存 **1/3~1/2**，且 **initial = max**（避免 GC 抖动）；pagecache 同理
- **heap 太大 → 饿死 pagecache → 查询变慢**；**pagecache 太小 → 磁盘 IO 爆炸**
- **容器内存上限对齐**：`--memory=4g` 时 heap 2–2.5G、pagecache 1.5G，留出 OS/Metaspace/direct memory
- 校验：`:sysinfo` 看 `dbms.memory.heap.max` / `dbms.memory.pagecache.size`；**pagecache 命中率目标 > 95%**
- ⚠️ 老文档的 `NEO4J_dbms_memory_*` 是 **5.x 之前**的命名，5.26/CalVer 用 `NEO4J_server_memory_*`

**MySQL `innodb_buffer_pool_size`**：专用机器 50–70%；**与其它服务同机时 25–40%**（32GB 机器上 2–4 GB）。**总连接数 = workers × (pool_size + max_overflow)，别超过 `max_connections`。**

来源：[Neo4j Docker 配置 2026.03](https://neo4j.net.cn/docs/operations-manual/2026.03/docker/configuration/)、[Neo4j 堆内存优化](https://www.php.cn/faq/2416844.html)

## 7.5 Milvus 部署要点

- **standalone 定位：可支撑到约 1 亿向量**，适合中小规模生产 → **本项目选 standalone**。
- **2.6 架构变化**：**Woodpecker WAL** 取代了对 Kafka/Pulsar 的硬依赖。⚠️ **Woodpecker 发布频繁（每周 1–2 次），v0.1.31（2026-06-04）被视为生产可用的最低版本，但项目整体仍偏早期 —— 这是 2.6 生产部署的主要风险点。**
- **端口**：19530（gRPC）、9091（metrics/healthz，`curl http://localhost:9091/healthz` 应返回 OK）。
- **★ 硬性前置：需要 AVX/AVX2**。无 AVX 的 CPU/VM 上 `milvus-standalone` 会以 `Illegal instruction` **崩溃重启循环**。检查 `grep -o -m1 'avx2\|avx' /proc/cpuinfo`；VM 需 CPU 透传（Proxmox `host`、KVM `-cpu host`）。
- **MinIO ≥ `RELEASE.2024-12-18T13-15-44Z`**。

**⚠️ 认证默认关闭**：任何人连上 19530 就能读你全部数据。启用：

```yaml
# milvus.yaml
common:
  security:
    authorizationEnabled: true
    defaultRootPassword: "<强口令>"     # 最长 72 字符，需双引号
```

启用后默认创建 `root`（默认口令 `Milvus`），用 `token="root:<password>"` 连接。**务必立刻改密** —— 改密后存在 etcd，`defaultRootPassword` 不再生效，**忘记口令可能需删卷重置（丢数据）**。

**RBAC 粒度**：实例级/数据库级/集合级。内置权限组 `COLL_RO`/`COLL_RW`/`COLL_ADMIN`、`DB_RO`/`DB_RW`/`DB_Admin`、`Cluster_RO`/`Cluster_RW`/`Cluster_Admin`。**建议**：API 用 `rag_api`（`COLL_RW`），worker 用 `rag_worker`（`COLL_ADMIN`）。

**★ pymilvus API**：**老 ORM（`connections.connect()` / `Collection()`）已弃用，一律用 `MilvusClient`。**

```python
client = MilvusClient(uri="http://milvus:19530", token="root:<password>")
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("chunk_id", DataType.INT64, is_primary=True)
schema.add_field("tenant_id", DataType.INT64, is_partition_key=True)  # 多租户
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
index_params = MilvusClient.prepare_index_params()
index_params.add_index(field_name="vector", index_type="HNSW",
                       metric_type="COSINE", params={"M": 32, "efConstruction": 200})
client.create_collection("rag_chunks", schema=schema, index_params=index_params)
```

- **顺序**：建集合 → 插数据 → 建索引 → load → search。向量字段**必须先有索引才能 load，先 load 才能 search**。
- **`M` 范围 4–64（默认 16），`efConstruction` 范围 8–512（默认 200）**。
- **默认建议 `AUTOINDEX`**，除非有明确需求（高召回 HNSW、超内存 DiskANN、内存受限 IVF_FLAT、小数据集 FLAT）。
- **schema 演进受限**：v2.6+ 可 `add_collection_field()` 加**可空新字段**，**已有字段不可改不可删** → **预留 `enable_dynamic_field=True`**。
- **⚠️ 版本 bug**：pymilvus **2.6.7 / 2.6.8 报 `SchemaNotReadyException`（"Collection not exist"）而 2.6.6 正常** → **上线前在候选 patch 版本上跑完整 create/insert/index/search 冒烟测试再锁定**（另一来源称最新为 2.6.21/2.6.22，**口径冲突**）。
- **⚠️ 版本对齐是硬约束**：`pymilvus 2.6.x ↔ Milvus server 2.6.x`；混用会出诡异问题。

**多租户**：partition key 是最可扩展的方案（**百万级租户**），声明 `is_partition_key=True` 后 Milvus 自动哈希路由，**默认 16 个物理分区**。⚠️ **过滤未索引字段会导致全集合扫描**（10–100x 延迟退化）→ **必须在首次查询前声明 partition key 或建索引**。

**Milvus Lite 用于开发：可行，但边界清楚**

| 限制 | 影响 |
|---|---|
| **一个 `data_dir` 只能被一个进程打开**（文件锁）；**同一 collection 的写入必须串行** | 不支持并发写 |
| 仅支持 pymilvus 本地工作流的子集 | 未实现的 RPC 返回 `UNIMPLEMENTED`；**不支持改 schema**；无分区级 load/release |
| **无认证/用户/角色/RBAC/TLS** | 绝不能暴露到不可信网络 |
| **不支持 binary / float16 / bfloat16 / int8 向量字段** | 量化方案无法验证 |
| **不支持 PQ 索引** | 索引选型受限 |
| **BM25 的 IDF 统计是段内局部而非全局** | **全文检索质量与 standalone 不一致** |
| pymilvus 2.6 把索引参数作为**扁平键**发送，而 **Lite 只读嵌套 `params` blob** | **索引调优参数在 Lite 上被静默忽略** |

**结论**：Lite 适合**原型、notebook、CI 测试、演示**（< 10 万向量）。**一旦涉及 RBAC、分区、标量索引调优、量化向量，就必须切 standalone。** 建议从一开始就把 `MilvusClient` 的 `uri` 参数化，切换成本≈0。

来源：[milvus-lite README](https://github.com/milvus-io/milvus-lite)、[Milvus Lite 限制](https://www.issoh.co.jp/tech/details/16456/)、[pymilvus issue #3263](https://github.com/milvus-io/pymilvus/issues/3263)、[Milvus 认证](https://milvus.io/docs/zh-hant/v2.6.x/authenticate.md)、[Milvus 多租户](https://milvus.io/docs/zh/v2.6.x/multi_tenancy.md)、[Milvus 元数据过滤生产实践](https://www.bestaiweb.ai/how-to-implement-metadata-filtering-in-qdrant-weaviate-milvus-and-pinecone-in-2026/)

## 7.6 配置管理：Dev vs Prod

```
.env.example        # 提交，所有键的占位 + 注释（唯一真相）
.env                # .gitignore，本地覆盖
.env.test
configs/milvus.yaml # 非 12-factor 组件的配置文件（挂载进容器）
```

**防泄漏清单**：
1. **所有密钥字段用 `SecretStr`**（`repr`/日志自动脱敏）
2. **`.gitignore` 包含 `.env*`（除 `.env.example`）**；pre-commit 加 secret 扫描（gitleaks / detect-secrets）
3. **structlog processor 加 redact**（`authorization`/`api_key`/`token`/`password`/`secret`），**且必须排在 `JSONRenderer` 之前**
4. **生产不用 `env_file` 明文**：用 **Docker secrets** 或 K8s Secret 挂文件，配合 pydantic-settings 的 `secrets_dir`；至少 `chmod 600`
5. **异常处理器绝不回传上游 provider 错误原文**
6. ⚠️ **`docker compose config` 输出会打印解析后的变量** —— CI 日志里要小心

**开发热重载**：

```bash
docker compose up -d mysql redis neo4j etcd minio milvus
uv run fastapi dev src/rag/main.py            # FastAPI CLI，自带 reload
# 或 uv run uvicorn rag.main:app --reload --reload-dir src
```

`--reload` 依赖 **`watchfiles`**（`uvicorn[standard]` 已含）。**生产绝不用 `--reload`。**

**生产 Dockerfile（多阶段 + uv）**：

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/   # ★ 锁 uv 版本
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_PROGRESS=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

FROM python:3.12-slim AS runtime
RUN groupadd -g 10001 app && useradd -u 10001 -g app -m app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/alembic /app/alembic
COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
CMD ["gunicorn","rag.main:app","-k","uvicorn_worker.UvicornWorker", ...]
```

**uv 关键纪律**：
- **`uv.lock` 必须提交**；Docker 构建用 `--frozen`；CI 用 **`uv lock --check`** 做门禁
- **`--no-editable`**：让 `.venv` 自包含，才能跨 stage 拷贝
- **`UV_COMPILE_BYTECODE=1`** 是生产镜像"性价比最高的设置"（预编译 `.pyc`，大幅降低冷启动）
- **`UV_LINK_MODE=copy`**：uv 缓存挂在 BuildKit cache mount 上时硬链接会失败（跨文件系统）
- **供应链**：`exclude-newer = "7 days"`（拒绝 7 天内发布的包）、`UV_MALWARE_CHECK=1`；uv 版本锁到 patch
- **不要用 Alpine 基础镜像**（musl libc 兼容性差）；用 `python:3.12-slim`
- **`.dockerignore` 必须排除 `.venv`、`.git`、`__pycache__`、`data/`**

**gunicorn + uvicorn worker（2026 口径）**：
- **⚠️ `uvicorn.workers.UvicornWorker` 自 uvicorn 0.30 起弃用** → 改用独立包 **`uvicorn-worker>=0.4,<0.5`**：`-k uvicorn_worker.UvicornWorker`
- **worker 数**：传统 `2×CPU+1` 是 WSGI 同步启发式，**对 async FastAPI 不适用**。IO 密集的 RAG API **从 4 开始**实测调优；**单机超 8 个 worker GIL 会让收益递减**，应水平扩副本
- **内存注意**：每个 worker 是独立进程，**4 worker 的本地方 embedding 模型 = 4 份内存**
- **`--timeout` 对 LLM 端点放宽到 120–300s**（默认 30s 会导致重启循环）
- `--preload` 避免每个 worker 各自初始化连接与全局对象；`--proxy-headers --forwarded-allow-ips` 在 LB 后必开

**迁移作为 init job**：`alembic init -t async`；`env.py` 用 `async_engine_from_config` + `poolclass=NullPool` + `await connection.run_sync(do_run_migrations)`；开 **`compare_type=True` + `compare_server_default=True`**；CI 用 `alembic check` 做漂移门禁。**★ 绝不在多个 API 副本的 lifespan 里跑迁移（并发迁移 = 灾难）**，必须是独立单次 init job。

来源：[uv in Dockerfile](https://pydevtools.com/handbook/how-to/how-to-use-uv-in-a-dockerfile/)、[uv.lock 工作流](https://stackharbor.com/en/knowledge-base/uv-python-project-lockfile-workflow/)、[FastAPI on Kubernetes 2026](https://khimananda.com/blog/run-fastapi-on-kubernetes)

## 7.7 可观测性

**★ 关键架构事实：Langfuse v3 自托管需要 web + worker + Postgres + ClickHouse + Redis + S3 ≈ 4–6 个后端服务**；**Arize Phoenix 是单容器**（SQLite 内置）。

| | Langfuse v3 | Arize Phoenix |
|---|---|---|
| 自托管组件 | 4–6 个 | **1 个** |
| 许可 | MIT 核心（企业模块另收费） | Elastic License 2.0（source-available，非 OSI） |
| 强项 | 追踪 + **Prompt 管理 + 数据集/评测 + 成本面板 + 生产监控**；ClickHouse 可到 10k+ spans/s | **最轻自托管**；**RAG 检索深度检查 + 向量空间分析**（UMAP 投影、聚类浏览器、漂移检测） |

**2026 两个重大变动**：**ClickHouse 于 2026-01 收购 Langfuse**；**Dynatrace 于 2026-08 宣布收购 Arize**（Phoenix 目前仍可用）。Thoughtworks 技术雷达（2026-04）把 Langfuse 列为 "Assess" 级，但指出 **v3 架构"更可扩展但也更难自托管"**。

**★ 分阶段推荐**：
- **阶段 1（开发 + 小生产，32GB VM）→ Arize Phoenix**（单容器 1GB）—— 得到完整 LLM 追踪 + **RAG 检索质量与 embedding 空间调试**（恰是 RAG 项目最需要的）
- **阶段 2（需要 Prompt 管理 / 评测 / 成本面板 / 团队协作）→ 加 Langfuse v3**（+ClickHouse 4GB + 一个 Postgres schema + Redis DB + MinIO bucket）

**★ 最关键纪律：用 OTel 埋点一次，后端可换**。两个平台都摄取 OTel span，所以应用侧只做 OTel/OpenInference 埋点：

```python
def setup_telemetry(settings: Settings) -> None:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=settings.otel_endpoint)   # Phoenix:4317 或 Langfuse:4317
    ))
    trace.set_tracer_provider(provider)

def instrument_app(app: FastAPI) -> None:
    FastAPIInstrumentor.instrument_app(app)
    LangChainInstrumentor().instrument()      # ★ 覆盖 LangGraph 的 graph/node/LLM/tool span
```

这样 LangGraph 的每个节点、每次 LLM 调用、每次工具调用都自动成为 span，`structlog` 注入的 `trace_id` 直通，**切换后端只改一个 endpoint**。

**Prometheus 抓取配置（注意各服务自带端点）**：

```yaml
scrape_configs:
  - job_name: api
    static_configs: [{ targets: ["api:8000"] }]              # prometheus-fastapi-instrumentator
  - job_name: milvus
    static_configs: [{ targets: ["milvus:9091"] }]           # ★ Milvus 自带 metrics
  - job_name: neo4j
    static_configs: [{ targets: ["neo4j:2004"] }]            # 需在 neo4j.conf 开 metrics.prometheus.enabled=true
  - job_name: mysql
    static_configs: [{ targets: ["mysqld-exporter:9104"] }]
  - job_name: redis
    static_configs: [{ targets: ["redis-exporter:9121"] }]   # oliver006/redis_exporter
```

⚠️ **已知坑**：追加 prometheus/grafana 到 Milvus compose 时会出现 **network 未定义**（`service "grafana" refers to undefined network milvus`）与**卷权限错误**（`/prometheus/queries.active` permission denied）→ **预创建卷目录 `mkdir -p volumes/{prometheus,grafana}` 并显式定义顶层 network**。

来源：[Langfuse vs Phoenix 自托管对比](https://www.morphllm.com/comparisons/arize-phoenix-vs-langfuse)、[2026 最佳开源 LLM 可观测性](https://futureagi.com/blog/best-open-source-llm-observability-2026/)、[Thoughtworks 雷达：Langfuse](https://www.thoughtworks.com/en-ec/radar/platforms/langfuse)

---

# 8. 测试

## 8.1 分层策略

| 层 | 范围 | 速度 | 何时跑 | 手段 |
|---|---|---|---|---|
| **L1 确定性内核** | chunker / normalizer / RRF / prompt builder / parser / citation 组装 | 毫秒 | 每次提交 | 纯函数断言 + Hypothesis + 假 LLM/Embedder |
| **L2 记录回放** | HTTP 契约、SDK 集成 | 5–15 ms | 每次提交 | vcrpy / pytest-recording cassettes |
| **L3 集成** | MySQL/Milvus/Neo4j 真容器 | 分钟 | PR / main | testcontainers + 事务回滚 |
| **L4 评估** | 检索质量 / 生成质量 | 分钟~小时 | 夜间 / 手动 | Golden set + recall@k/MRR/nDCG + LLM-judge |

**★ 最重要的判断：把"检索确定性指标"和"生成质量指标"分开。** 给定固定的 embedding 与索引，**recall@k / precision@k / MRR / nDCG@k / hit-rate 是完全确定的** —— 它们最适合做 **CI 硬门禁（不花一分钱 token）**；只有生成质量指标（faithfulness 等）才需要 LLM-judge，**只适合夜间/按需跑**。

## 8.2 testcontainers-python

**版本 4.15.0（2026-07-24）**。

⚠️ **Wait Strategy 大迁移**：从废弃的 `@wait_container_is_ready()` / `wait_for_logs` 迁到**结构化策略类**。4.14.0 引入 `ExecWaitStrategy` 并把 postgres 迁移过去；mysql/cassandra/kafka/elasticsearch/minio 同期迁移；**4.15.0-rc3 完成 neo4j 的迁移**。

可用策略：`LogMessageWaitStrategy`、`HttpWaitStrategy(port).for_status_code(200)`、`ExecWaitStrategy([...])`、`PortWaitStrategy`、`HealthcheckWaitStrategy`、`SqlAlchemyConnectWaitStrategy`（把 `DBAPIError` 视为可重试）、`CompositeWaitStrategy`；统一 `container.waiting_for(strategy)` 应用。

**★ 2026 共识："等待就绪，而不是等待时间"**（禁止 `sleep(5000)`）。wait strategy 超时时会把容器日志附加到 `TimeoutError`。

```python
# ---- MySQL ----
from testcontainers.mysql import MySqlContainer
with MySqlContainer("mysql:8.4", username="rag", password="rag", dbname="rag_test",
                    dialect="asyncmy") as mysql:      # dialect 让 get_connection_url() 产出 mysql+asyncmy://
    url = mysql.get_connection_url()
```

- ⚠️ **`MySqlContainer` 没有 `with_database()`**；库名走构造参数 `dbname`。默认 `MYSQL_USER`/`MYSQL_PASSWORD`/`MYSQL_DATABASE` 均为 `test`。支持 `seed=<目录>` 把 SQL 挂到 `/docker-entrypoint-initdb.d/`。
- ⚠️ 历史上 `get_connection_url()` **未对密码里的特殊字符（如 `%`）做 URL 编码** → **测试密码请用纯字母数字**。

```python
# ---- Milvus standalone（内嵌 etcd）----
from testcontainers.milvus import MilvusContainer
with MilvusContainer("milvusdb/milvus:v2.6.18") as milvus:   # 禁止 :latest
    uri = f"http://{milvus.get_container_host_ip()}:{milvus.get_exposed_port(19530)}"
```

- **★ 关键结论：不需要额外的 etcd / MinIO 容器。** `MilvusContainer` 以 `milvus run standalone` 启动并**自动配置内嵌 etcd**（`ETCD_USE_EMBED=true`、`ETCD_DATA_DIR=/var/lib/milvus/etcd`、`COMMON_STORAGETYPE=local`，配置文件 `embedEtcd.yaml`）。
- **健康检查：端口 9091 轮询 `/healthz`**，并额外等待日志 `"Welcome to use Milvus!"`。暴露端口 9091 与 19530。
- **默认镜像是 `milvusdb/milvus:latest` → CI 必须显式 pin。**

```python
# ---- Neo4j ----
from testcontainers.neo4j import Neo4jContainer
with Neo4jContainer("neo4j:5.26-community") as neo4j:
    driver = neo4j.get_driver()      # 已含认证配置
```

**复用（reuse）**：`~/.testcontainers.properties` 设 `testcontainers.reuse.enable=true` + `with_reuse=True`。
- **⚠️ 复用只用于本地开发，禁止用于 CI**（CI 必须 hermetic；复用会让测试间状态泄漏）。本地启动时间可降到 1s 以内。
- **★ 提升速度的第一手段不是 reuse，而是把容器 fixture 提到 pytest `scope="session"`**（一次启动、全测试共享，配合事务回滚保证隔离）。
- Ryuk 是默认资源回收器；受限 CI（DinD）中设 `TESTCONTAINERS_RYUK_DISABLED=true` 并显式 stop。

来源：[testcontainers-python CHANGELOG](https://raw.githubusercontent.com/testcontainers/testcontainers-python/main/CHANGELOG.md)、[v4.15.0-rc3](https://newreleases.io/project/github/testcontainers/testcontainers-python/release/testcontainers-v4.15.0-rc3)、[Wait Strategies (DeepWiki)](https://deepwiki.com/testcontainers/testcontainers-python/3.5-wait-strategies)、[MilvusContainer 实现](https://github.com/testcontainers/testcontainers-python/commit/78b137cfe53fc81eb8d5d858e98610fb6a8792ad)、[Testcontainers Best Practices 2026](https://qaskills.sh/blog/testcontainers-best-practices-2026)

## 8.3 测试异步 FastAPI 应用

```python
from httpx import ASGITransport, AsyncClient

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)       # 进程内 ASGI 调用，无 uvicorn、无端口
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

- **⚠️ `ASGITransport` 不触发 lifespan 事件** → 需 `pip install asgi-lifespan` 用 `LifespanManager(app)` 包裹，或 `async with app.router.lifespan_context(app)`。
- **⚠️ `TestClient` 的"魔法桥接"在 async 测试中失效** → 必须换 `AsyncClient`。
- **常见错误 `RuntimeError: Task attached to a different loop`**（如客户端在 import 期或模块级被实例化）→ **规范：任何依赖事件循环的对象都必须在 async fixture / lifespan 内创建。**

**pytest-asyncio vs anyio**：

| | anyio | pytest-asyncio |
|---|---|---|
| 标记 | `@pytest.mark.anyio` | `@pytest.mark.asyncio` 或 `asyncio_mode="auto"` |
| 后端 | asyncio + Trio | 仅 asyncio |
| 官方倾向 | **FastAPI 官方文档用的就是 anyio** | 社区主流 |

**★ 推荐：pytest-asyncio，`asyncio_mode = "auto"`，显式设 `asyncio_default_fixture_loop_scope = "function"`。**

- `auto` 模式下普通 `@pytest.fixture` 也能用于 async fixture。**`strict`（默认）下踩坑**：async fixture 用 `@pytest.fixture` 装饰会**返回协程对象而不是执行**，报 `requested async @pytest.fixture 'db_session' in strict mode`。
- **必须显式设 `asyncio_default_fixture_loop_scope`**，否则打 `PytestDeprecationWarning`（且未来行为可能变）。

**★ 依赖替换 + 每测试事务回滚（核心模式）**：

```python
@pytest_asyncio.fixture(scope="session")
async def engine(mysql_container):                 # session 级：DDL 只跑一次
    eng = create_async_engine(mysql_container.get_connection_url(), poolclass=NullPool)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)   # ★ DDL 在测试事务之外
    yield eng
    await eng.dispose()

@pytest_asyncio.fixture
async def db_session(engine):
    async with engine.connect() as conn:
        trans = await conn.begin()                     # 外层事务
        session = AsyncSession(bind=conn, expire_on_commit=False,
                               join_transaction_mode="create_savepoint")   # ★ 关键
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()                     # 一次性清空测试产生的所有数据

@pytest.fixture
def app_with_overrides(app, db_session):
    async def _get_session():
        yield db_session                               # 与测试共用同一事务
    app.dependency_overrides[get_session] = _get_session
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    yield app
    app.dependency_overrides.clear()                   # ★ 必须清理，否则污染后续测试
```

**★ 为什么需要 `join_transaction_mode="create_savepoint"`**：测试代码里的 `await session.commit()` 会结束外层事务，导致后续回滚失效。该模式让 `session.commit()` 只释放 SAVEPOINT，外层事务仍存活，最终 `trans.rollback()` 能撤销一切。

**三个必踩的坑**：
1. **DDL 绝不能在测试事务内执行**（`CREATE TABLE` 会隐式提交并破坏 savepoint），表现为 `SAVEPOINT does not exist`。**建表放 session 级 fixture。**
2. **"Connection is not acquired"**：fixture 必须在 `finally` 中显式 `close()` session。
3. **每个测试必须独占连接**（`NullPool` 或每测试独立 connect），否则异步驱动会报 "another operation is in progress"。

来源：[FastAPI 异步测试（官方中文）](https://fastapi.tiangolo.com/zh/advanced/async-tests/)、[pytest-asyncio patterns](https://tessl.io/registry/testland/pytest-asyncio-patterns/1.2.14/files/SKILL.md)、[异步 conftest 参考](https://github.com/alexvervloet/learning-python-backends/blob/main/backends/learning/testing-concepts/04_async_testing/conftest.py)、[pico-sqlalchemy 测试指南](https://dperezcabrera.github.io/pico-sqlalchemy/how-to/testing/)

## 8.4 测试非确定性 RAG 管线

**假 LLM**（脚本化，覆盖所有边界）：

```python
class FakeLLM:
    def __init__(self, script: list[dict]): self.script, self.i = script, 0
    async def ainvoke(self, messages, **kw):
        r = self.script[self.i % len(self.script)]; self.i += 1
        return AIMessage(content=r.get("content", ""), tool_calls=r.get("tool_calls", []),
                         response_metadata={"finish_reason": r.get("finish_reason", "stop")})
    async def astream(self, messages, **kw):
        for tok in self.script[0].get("content", "").split():
            yield AIMessageChunk(content=tok + " ")
```

**脚本必须覆盖的边界**（这才是"非确定性"里真正会炸的地方）：`finish_reason="length"`（被截断）、`content=None` 但 `tool_calls` 有值、多个并行 `tool_calls`、`tool_calls` 参数是非法 JSON、空字符串回复、超长回复、流式中途抛错、429/500 后重试。

**确定性 Embedder（hash-based）**：

```python
import hashlib, numpy as np

class HashEmbedder:
    """同文本 → 同向量；近似保留词袋相似性"""
    def __init__(self, dim=1024): self.dim = dim
    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v            # ★ 必须 L2 归一化
    async def aembed_documents(self, texts): return [self._embed_one(t) for t in texts]
    async def aembed_query(self, text):      return self._embed_one(text)
```

**要点**：**必须 L2 归一化**（否则余弦/内积行为不可控）；**用 `hashlib` 而不是内置 `hash()`**（Python 字符串 hash 有**进程级随机化** → 每次运行结果不同，是**隐蔽的 flaky 来源**）；用 hashing trick 才能让"共享词多 → 相似度高"这一**定性关系**成立，从而让 recall@k 类测试有意义。

**LangGraph 层**：单元测试用 `InMemorySaver` 而非 MySQL checkpointer；通过依赖注入替换节点函数；断言**状态迁移**而非节点内部实现。

**VCR.py 录制回放**：

**核心事实：VCR.py 拦截的是 HTTP 库而不是 SDK** —— 覆盖 `http.client`/`requests`/`urllib3`/`aiohttp`/`httpx`/`httpcore`。**OpenAI 与 Anthropic 官方 Python SDK 都基于 httpx**，所以无需包装即可录制。回放约 **5–15 ms/测试**，无需 API Key、无网络。

```python
llm_vcr = vcr.VCR(
    cassette_library_dir="tests/cassettes",      # 用绝对路径！相对路径相对 pytest CWD 解析
    record_mode="none",                          # ★ CI：永不录制，只回放
    decode_compressed_response=True,             # ★ 必须！否则 gzip 响应变成二进制垃圾
    filter_headers=[("authorization", "REDACTED")],
    match_on=["method", "scheme", "host", "port", "path", "body"],
)
```

**⚠️ 匹配陷阱（2026 年最值得注意的一条）**：VCR 默认匹配**不含请求体**，而所有 chat completion 都打到同一个 `POST /v1/chat/completions` → **一条录制的交互会应答所有调用**（测试"通过"但毫无意义）。把 `body` 加进 `match_on` 又会因 SDK 请求体含动态字段而失配；加 `headers` 更糟。**解法：自定义 matcher** —— 规范化请求体、忽略易变字段、只比对 prompt 内容与模型名，并单独处理流式响应。

其他必须知道的：
- ⚠️ **`record_mode="new_episodes"` 下未匹配的请求会静默走真实网络并追加到 cassette** → 测试永远绿、账单持续增长。**CI 必须 `record_mode="none"`。**
- ⚠️ **cassette 是不可信 YAML**：pin `vcrpy>=4`（现代版本用 `yaml.safe_load`；更早版本用可 RCE 的 `yaml.load`），并在 PR 里像审代码一样审 cassette diff。
- **脱敏**：至少过滤 `authorization`、`x-api-key`、`api-key`、`openai-organization`、`anthropic-version`、cookie，以及 URL query 里的 key。
- **★ 断言"代码而不是模型散文"**：断言**请求构造**（model、消息条数、tool schema 是否正确）和**解析器对 `tool_calls` / `finish_reason` / `content=None` 的处理**。**cassette 是给解析器用的 fixture，不是对模型质量的断言** —— 逐字断言散文，一次重新录制就会因为与你代码无关的原因失败。
- **`respx`（0.22+）是补充**：只用来定向模拟 HTTPX 的错误场景（429/500/超时/流中断），5–10 个测试即可。
- **不要用 VCR 的场景**：要求每次输出都不同的能力测试、prompt 高频演进的阶段（cassette 漂移）、短期小项目（维护成本可能超过收益）。

## 8.5 属性测试 / 蜕变测试（chunking 与 RRF）

工具：**Hypothesis ≥ 6.122**。适用性判断：**只对高价值纯逻辑用**（分块、融合、解析），不用于 CRUD / IO 密集路径。

**Chunking 的不变量（按价值排序）**：

```python
cjk_text = st.text(alphabet=st.sampled_from(
    "的一是不了在人有我他这中大来上个国说们為中文测试" " \n\t。，！？；：、" "abcXYZ019.,!?" "😀\n"),
    min_size=0, max_size=5000)

@settings(max_examples=200, deadline=None)
@given(text=cjk_text, max_size=st.integers(16, 512), overlap=st.integers(0, 64))
def test_1_no_text_loss(text, max_size, overlap):
    assume(overlap < max_size)
    chunks = chunk(text, max_size=max_size, overlap=overlap)
    assert normalize("".join(chunks)) == normalize(text)     # ★ 最强不变量：无文本丢失

@settings(max_examples=200, deadline=None)
@given(text=cjk_text, max_size=st.integers(16, 512), overlap=st.integers(0, 64))
def test_2_size_bounds(text, max_size, overlap):
    assume(overlap < max_size)
    chunks = chunk(text, max_size=max_size, overlap=overlap)
    assert all(len(c) <= max_size for c in chunks)                  # 上界
    assert all(len(c) >= max_size - overlap for c in chunks[:-1])   # 下界（非末块）

@settings(max_examples=100, deadline=None)
@given(text=cjk_text)
def test_3_idempotent_and_unicode_safe(text):
    assert chunk(text) == chunk(text)                        # 幂等
    assert [c.index for c in chunk(text)] == list(range(len(chunk(text))))  # 索引稳定
    for c in chunk(text):
        c.text.encode("utf-8")                               # 不抛 UnicodeEncodeError
    assert "".join(c.text for c in chunk(text)) == text      # 不产生非法切分（无孤立代理对）

# 蜕变关系：配置单调性
@settings(max_examples=100, deadline=None)
@given(text=cjk_text, a=st.integers(32,128), b=st.integers(129,512))
def test_4_size_monotonic(text, a, b):
    """max_size 增大 → chunk 数非增"""
    assert len(chunk(text, max_size=b, overlap=0)) <= len(chunk(text, max_size=a, overlap=0))
```

**必须显式测的边界**：空字符串（契约是 `[]` 还是 `[""]`？**必须明确并在测试中断言**）、纯空白、**单个超长无分隔段**（必须切分，不能一 chunk 无限长）、全角/半角标点混排、Markdown 代码块/表格不被切碎、标题与正文不分离。

**★"分区无关性"是最有价值的蜕变关系**：同一文档在不同分块粒度（page / window / line）下，**下游结果（去重、最终文本、归属）应保持一致** —— 2026 年已有项目用此模式（`alto-llm-corrector` 的 "metamorphic safety net"：在 default / tiny-window / small-window 三种分区下跑同一管线，断言最终文本与状态逐行一致）。

**RRF 的不变量**：

```python
ranked = st.lists(st.integers(min_value=0, max_value=999), unique=True, min_size=0, max_size=50)

@given(a=ranked, b=ranked, k=st.integers(1, 200), wa=st.floats(0,10), wb=st.floats(0,10))
def test_rrf_commutative(a, b, k, wa, wb):
    """★ 排列不变性：交换输入列表顺序，融合结果完全相同（含并列顺序）"""
    assert fuse([(a, wa), (b, wb)], k=k) == fuse([(b, wb), (a, wa)], k=k)

@given(a=ranked, b=ranked, k=st.integers(1, 200))
def test_rrf_deterministic_tie_break(a, b, k):
    """★ 确定性：同一输入多次调用结果完全一致（tie-break 必须用 (-score, doc_id)）"""
    assert fuse([(a,1.0),(b,1.0)], k=k) == fuse([(a,1.0),(b,1.0)], k=k)

@given(a=ranked, b=ranked, k=st.integers(1, 200))
def test_rrf_monotonicity(a, b, k):
    """单调性：某文档在某一列表中排名提升 → 融合分数非减"""
    if not a: return
    base = dict(fuse([(a,1.0),(b,1.0)], k=k))
    promoted = [a[-1]] + a[:-1]
    assert dict(fuse([(promoted,1.0),(b,1.0)], k=k))[a[-1]] >= base[a[-1]]

def test_rrf_degenerate():
    assert fuse([([], 1.0)], k=60) == []      # 空列表不崩
```

**★ 这两条不变量在真实代码中最容易失败**：
1. **排列不变性** —— 任何"用列表下标做 tie-break"或"按输入顺序累积浮点分数"的实现都会破坏它。**浮点求和的非结合性**也会让理论上相等的分数出现微小差异 → **规范要求：融合分数先量化（`round(score, 9)`）再排序，tie-break 用 `(-score, doc_id)`。**
2. **确定性 tie-break** —— 若用 `dict` 迭代顺序或 `set`，Python 的 hash 随机化会让**同一输入在不同进程产生不同顺序**，表现为"只在高并发/重启后偶发"的诡异失败。

**与 LangGraph 相关的属性**：checkpoint 重放幂等（从同一 checkpoint 重放得到相同最终状态）；中断/恢复一致性（`interrupt` 后恢复的 state 与不中断跑完在业务语义上等价）；**节点重试不产生重复副作用**（注入失败后重试，MySQL 侧写入条数不变 —— 依赖 outbox 的幂等键）。

来源：[Property-Based LLM Testing With Hypothesis](https://ossaihub.com/code/property-based-llm-testing-hypothesis/)、[Hypothesis 2026 指南](https://helpmetest.com/blog/hypothesis-python-property-testing/)、[Milvus RRF 文档（k=60）](https://milvus.io/docs/zh/reranking.md)

## 8.6 Golden 数据集与评估工具

**Golden set 结构**：

```python
@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    relevant_chunk_ids: list[int]        # 用于 recall@k / MRR / nDCG
    reference_answer: str | None = None
    tags: tuple[str, ...] = ()           # 分类：单跳/多跳/否定/时效/拒答
```

**指标实现要点（务必自己写、别依赖黑盒）**：
- `recall@k = |retrieved[:k] ∩ relevant| / |relevant|`
- `MRR = 1/rank(第一个相关项)`
- `nDCG@k`：相关度可分级（2=直接答案，1=部分相关）
- **★ 门禁写法：对每个 tag 分组统计**（单跳 recall@10 ≥ 0.95，多跳 ≥ 0.80），避免整体均值掩盖某类退化；**只设下限阈值 + 允许小样本容差**（如 `mean_recall >= baseline - 0.02`）

**2026 年评估工具格局**：

| 工具 | 版本/状态 | 定位 | 是否值得 |
|---|---|---|---|
| **RAGAS** | **0.4.3（2026-01-13，最新）**；repo 由 `explodinggradients` 迁到 `vibrantlabsai`，**2026-02-24 后无 commit**（维护放缓信号） | RAG 指标事实标准（faithfulness / answer relevancy / context precision/recall / noise sensitivity） | **值得作指标口径，但不要押注为 CI 框架** |
| **DeepEval** | **4.1.1（2026-07）** | "pytest for LLMs"：`deepeval test run`、`assert_test`、JUnit XML、50+ 指标、内置 RAGAS 指标 | **★ 值得作 CI 门禁**（与 pytest 栈天然契合） |
| **promptfoo** | CLI/YAML，**2026-03 被 OpenAI 收购**（仍 MIT） | 秒级 prompt 回归 + 红队/对抗测试 | 值得作为"prompt 变更快检" |
| LangSmith | 托管 SaaS | tracing / 数据集 / 实验 | 预算允许时；否则自托管 Langfuse/Phoenix |

**★ 推荐组合**：**DeepEval（pytest 门禁）+ RAGAS 指标（口径与看板）+ 自写 recall@k/MRR 断言（零成本 CI 门禁）+ Langfuse/Phoenix（tracing）**。理由：单一工具无法同时满足"pytest 原生门禁"和"RAG 指标权威口径"。

**⚠️ 已知坑**：**RAGAS 0.4.3 与 `langchain-community` 0.4.x 不兼容** → **pin `langchain-community<0.4`**；`from ragas.metrics import Faithfulness` 会 `ImportError`（实现移到了 `ragas.metrics._faithfulness`）。**DeepEval 坑**：把模型名当字符串传会回落到 OpenAI 模型，用 Anthropic 必须传 `AnthropicModel` 实例。

**LLM-as-judge 的偏差与缓解（必须进规范）**：
- **位置偏差**：成对比较时偏向第一个 → **交换顺序跑两次并取平均/要求一致**
- **自我偏好**：模型偏向自己家族的输出 → **用不同家族的模型做 judge**
- **冗长偏差**：偏向更长的回答 → rubric 里显式声明"简洁不扣分"，或对长度分桶统计
- 其他：给**明确的评分 rubric 与锚点示例**、**固定 judge 的 `temperature=0`**、**多次采样报告方差**、**人工抽检校准 judge**

**★ 容差断言取代精确字符串匹配**：

```python
assert 0.0 <= faithfulness <= 1.0 and faithfulness >= 0.75
assert cosine_sim(answer, reference) >= 0.82                  # 语义相似度下限
assert set(expected_entities) <= extract_entities(answer)     # 关键实体覆盖
assert 50 <= len(answer) <= 4000                              # 长度区间
assert json.loads(answer)["intent"] in ALLOWED_INTENTS        # 结构化输出 schema
```

**⚠️ seed / `temperature=0` 的确定性幻觉（必须写进规范的警示）**：
- **`temperature=0` 不保证可复现**：MoE 路由、batch 大小不同导致的浮点非结合性、推理服务并发调度、GPU/内核版本差异、vLLM 等引擎的连续批处理，都会改变输出。
- OpenAI 的 `seed` 参数是 **best-effort**，官方明确不保证确定性。
- **结论：即使 `temperature=0`，也必须用结构断言 + 容差断言；禁止在测试中做字符串相等比较。**

## 8.7 CI 成本控制与契约测试

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "-m 'not eval' --strict-markers -q"
markers = ["unit", "integration", "eval", "regression"]
```

```yaml
pr_fast:        pytest -m "unit"                       # 每次提交，<30s
pr_integration: pytest -m "unit or integration"        # PR，几分钟
nightly:        pytest -m "eval or regression" --maxfail=1   # 夜间/手动
```

- **CI 里禁止开启 testcontainers reuse**（必须 hermetic）
- `--strict-markers` 防止 marker 拼写错误导致测试被静默跳过
- 评估层加**输入哈希缓存**（prompt+context → 结果落盘），避免重复烧钱

**★ 明确反对的两种"省钱"做法**：

1. **用 SQLite 替 MySQL 跑单元测试** —— 方言差异会制造**假绿**：JSON 类型语义、`ENUM`/`CHECK`、多值索引、**`SELECT ... FOR UPDATE SKIP LOCKED`（SQLite 不支持）**、`ON DUPLICATE KEY UPDATE`、`utf8mb4` 行为、`DATETIME(3)` 精度。**这类差异恰好全都落在本项目的关键路径上。**
2. **把 Milvus 换成"纯 Python 列表检索"却不做契约测试** —— fake 会悄悄漂移。

**★ 契约测试（强烈推荐）**：

```python
@pytest.fixture(params=["memory", "milvus_lite",
                        pytest.param("milvus_tc", marks=pytest.mark.integration)])
def store(request): ...

async def test_upsert_is_idempotent(store):
    await store.upsert([ChunkVector(chunk_id=1, vector=[0.1]*1024)])
    await store.upsert([ChunkVector(chunk_id=1, vector=[0.2]*1024)])   # 同 PK 覆盖
    assert [h.chunk_id for h in await store.search([0.2]*1024, top_k=5)] == [1]  # 不出现重复 id

async def test_delete_removes_from_results(store): ...
async def test_tenant_filter_is_respected(store): ...   # ★ 租户隔离是最重要的契约
async def test_output_fields_roundtrip(store): ...      # content_hash 等标量能取回
```

为 `DocumentRepository`、`VectorStore`、`GraphStore` 各写**一套**契约测试，参数化跑在 fake 与真实现上。**这是 fake 不漂移的唯一可靠机制**，也是把"单元测试快"和"集成测试真"同时拿到的关键。

来源：[RAGAS vs DeepEval 2026](https://pythondatabench.com/article/rag-evaluation-python-ragas-deepeval-trulens-2026)、[AI Evals Frameworks 2026](https://jobsbyculture.com/blog/ai-evals-frameworks-compared-2026)、[LLM Evaluation Frameworks](https://machinelearningmastery.com/llm-evaluation-frameworks-compared-how-to-actually-measure-what-your-model-does/)

---

# 9. 依赖钉版清单（可直接落进 spec）

```toml
[project]
requires-python = ">=3.12,<3.13"     # 3.13 可接受；3.14 free-threading 不要用

dependencies = [
  # Web
  "fastapi==0.141.1",                # 版本来自单一来源，建议实测
  "uvicorn[standard]>=0.35",
  "uvicorn-worker>=0.4,<0.5",        # ★ uvicorn.workers 已弃用
  "gunicorn>=23,<24",
  "python-multipart>=0.0.32,<1",     # ★ 安全下限（CVE-2026-53538/53539/53540）
  "sse-starlette>=3,<4",

  # 校验 / 配置
  "pydantic>=2.13,<3",
  "pydantic-settings>=2.14,<3",

  # 数据
  "sqlalchemy[asyncio]==2.0.52",     # 禁止 2.1.0b*
  "asyncmy==0.2.14",
  "pymysql>=1.1",                    # 仅同步运维脚本 / Alembic 备选
  "alembic==1.19.2",
  "redis>=6,<7",
  "pymilvus==2.6.*",                 # ★ 避开 2.6.7/2.6.8 的 SchemaNotReady 回归
  "neo4j>=5.26,<7",                  # driver 6.x 支持
  "neo4j-graphrag>=1.16.0",          # ★ 含 Text2Cypher EXPLAIN 只读闸门

  # Agent
  "langgraph>=1.2,<2",
  "langchain-core>=1.4.7",
  "langgraph-checkpoint-mysql[asyncmy]>=3.0.0",   # AIOMySQLSaver

  # 任务
  "celery[redis]==5.6.3",            # Windows 开发改用 arq==0.28

  # 鉴权
  "PyJWT[crypto]>=2.10",             # ★ 不用 python-jose
  "pwdlib[argon2]>=0.2",             # ★ 不用 passlib

  # 上传 / 解析
  "aiofiles>=24",
  "minio>=7.2",
  "pdfplumber>=0.11",                # MIT + 显式 CJK
  "python-docx>=1.1",
  "docling>=2",                      # MIT；复杂表格/扫描件（需异步 + 超时）
  "python-magic-bin; sys_platform=='win32'",
  "python-magic; sys_platform!='win32'",
  "pypdf>=5",                        # 兜底（避免 PyMuPDF 的 AGPL）

  # 可观测
  "structlog==26.1.0",
  "opentelemetry-instrumentation-fastapi",
  "opentelemetry-exporter-otlp-proto-grpc",
  "openinference-instrumentation-langchain",
  "prometheus-fastapi-instrumentator",
  "slowapi>=0.1.10",
]

[dependency-groups]
dev = [
  "pytest>=9.0", "pytest-asyncio>=1.0", "anyio>=4.12",
  "testcontainers[mysql,neo4j]>=4.15.0",
  "milvus-lite>=3.2",                # 单元层；注意 setuptools<81
  "hypothesis>=6.122",
  "vcrpy>=7", "pytest-recording>=0.13.4", "respx>=0.22",
  "ragas==0.4.3",                    # 需 pin langchain-community<0.4
  "deepeval>=4.1.1",
  "asgi-lifespan>=2.1",              # ASGITransport 不触发 lifespan
  "fakeredis>=2.2",
]
```

**容器镜像（全部显式 pin）**：

| 组件 | 镜像 |
|---|---|
| MySQL | `mysql:8.4` |
| Milvus | `milvusdb/milvus:v2.6.x`（standalone，内嵌 etcd，健康检查 `:9091/healthz`） |
| etcd | `quay.io/coreos/etcd:v3.5.18` |
| MinIO | `minio/minio:RELEASE.2024-12-18T13-15-44Z` |
| Neo4j | `neo4j:5.26-community-ubi9`（或 `neo4j:5.26.30-community`） |
| Redis | `redis:8-alpine` |
| Phoenix | `arizephoenix/phoenix`（# 生产锁具体版本） |

---

# 10. 需要你方复核的不确定项（**请勿直接采信**）

以下条目来源冲突或为单一来源，**建议在实现前用 `pip index versions <pkg>` 或官方 release notes 验证**：

1. **`neo4j-graphrag` 最新版本**：1.18.0（2026-06，pyspect）vs 1.14.1（2026-03，mygit 快照）。**口径冲突。**
2. **`GraphCypherQAChain` 的 2026 API 面**：官方文档 vs 课程来源对 `from_llm()` / `allow_dangerous_requests` 的存废说法**互相矛盾**。
3. **FastAPI `Depends(scope=...)`**：只见 PR #14301 标题（`Fix Depends(func, scope='function') for top level dependencies`），**未确认签名/语义**。实现前查 `fastapi/params.py`。
4. **FastAPI 0.141.1 为最新稳定版**：来自单一博客来源，**无官方 release 页佐证**。
5. **SQLAlchemy 2.1.0 GA**：仅有"预期 2026 夏末 GA"说法，**未证实已发布**。
6. **`aiomysql` 的 2026 维护状态**：**未找到 2026 年发版证据**（只是"没找到"，**不等于停更**）。选 asyncmy 的理由基于其明确的活跃记录与性能数据。
7. **pymilvus / Milvus 具体 patch 号**：2.6.9 / 2.6.16 / 2.6.18 / 2.6.21 / 2.6.22 各来源不一。**必须在候选 patch 上跑完整冒烟测试再锁定**（2.6.7/2.6.8 有已知回归）。
8. **Milvus `VARCHAR.max_length` 是否可 `alter_collection_field` 修改**：两来源互相矛盾 → **设计上按"不可改"假设**。
9. **Milvus Lite 的索引能力**：一说 3.2.0 已支持 HNSW/稀疏/partition/BM25，另一说仍只有 FLAT 且 ≤100k 向量 → **CI 中先跑能力探测**。
10. **MySQL 9.7 Community 是否有 VECTOR INDEX**：一个教程站提到 `CREATE VECTOR INDEX`，但多个实测来源明确否认。**按"没有"规划。**
11. **LangGraph `MemorySaver` 的 1.0+ 导入路径**（`langgraph_core.checkpoint.memory`）：单一来源，**未证实**。
12. **Neo4j 5.26 的最新补丁号**：5.26.28 / 5.26.29 / 5.26.30 各来源不一。
13. **中文场景的 PDF 解析器 head-to-head 基准**（Docling vs pdfplumber vs MarkItDown）：**检索中未找到** → **必须自建 10–20 份真实中文文档的小评测**。
14. **Milvus Lite 在 Windows 原生支持**：仅出现 "Milvus_LiteWindows" 字样，**无明确支持声明** → **Windows 开发建议走 WSL2**。

---

# 附：十件最重要的事（如果只读一段）

1. **Neo4j 用 5.26 LTS**；**`neo4j-graphrag >= 1.16.0`**（Text2Cypher 的 EXPLAIN 只读闸门）；**不用 MS GraphRAG / LightRAG**。
2. **统一 `chunk_id`（MySQL BIGINT）作为三库共享键**；Milvus 主键显式指定、禁 autoID；Neo4j 建 `Chunk.chunk_id` 唯一约束。
3. **MySQL 存 chunk 权威正文，Milvus 只存向量 + 标量**（65535 字节上限 + 无唯一约束 + 软删语义）。用 **outbox + `FOR UPDATE SKIP LOCKED`** 保一致性，用**对账任务**兜底。
4. **`content_hash` 门控一切**（含 `PROMPT_VERSION` / `CHUNKER_VERSION` / `EMBED_MODEL_ID`）；**删除是 token-free 的**，但**孤儿实体清理必须延迟**。
5. **抽取成本控制三件套**：prompt 缓存（按 hash）+ 模型分层（mini 抽取）+ `MAX_TRIPLETS_PER_CHUNK`。**索引成本量级要写进立项文档。**
6. **先建强基线（混合检索 + rerank），再用 50–100 个真实跨文档问题测"加图的增量收益"**。**提升 < 3pp 就不要上图。** 现实收益是 +3–5pp 且只在特定问题形态上；图税是 1–2 个数量级的 query token。
7. **不用 Text2Cypher**（EM 4.2%）。需要结构化查询时用 `ToolsRetriever` + 预置参数化 Cypher 模板。
8. **摄取用 Celery（分 `ingest`/`bulk` 队列，`prefetch_multiplier=1`，`acks_late=True`，`visibility_timeout > 最长任务`）**；**绝不用 `BackgroundTasks`**；**幂等靠 DB 唯一约束而非应用层 if**。
9. **SSE + 三个反缓冲响应头 + 15s 心跳 + 服务端事件过滤**；**`/healthz` 零依赖、`/readyz` 有界并发探测**；**`await driver.close()` 是协程**。
10. **PyJWT + pwdlib**（python-jose 有 CVE-2025-61152，passlib 与 bcrypt 5.x 破裂）；**PyMuPDF 是 AGPL，闭源/SaaS 禁用**；**`python-multipart>=0.0.32` 必须显式钉**。

agentId: aefe5a91c8d96ff0e (use SendMessage with to: 'aefe5a91c8d96ff0e', summary: '<5-10 word recap>' to continue this agent)
<usage>subagent_tokens: 187758
tool_uses: 38
duration_ms: 697918</usage>