# Agent V2：教务任务编排

V2 在 V1 的原生 Tool Calling Loop 上增加教务只读工具、任务完成判断、持久化
Task State、多轮任务和错误恢复。旧 LangGraph、RAG 检索、`/api/v1/retrieve`、
`/api/v1/chat` 的原返回字段继续保留。

## 1. 现状审计与边界

改造前 PostgreSQL 只有文档、分块、摄取任务、会话和消息表，没有课程、成绩、课表、
考试表，也不存在可复用的教务查询 service。因此 V2 在同一个 Repository/SQLAlchemy
体系中增加最小只读教务域，而没有伪造“已有业务逻辑”。

V2 不实现选课写操作，不连接真实教务系统，不自动提交课程，不加入 Multi-Agent。
所有教务 Tool 都只读取事实；“还差多少学分”“哪些课值得选”由 Agent 分析。

## 2. 新增文件

| 文件 | 职责 |
|---|---|
| `src/rag/agent/academic_tools.py` | 五个教务 Tool 的 schema、参数校验和统一返回信封 |
| `src/rag/agent/task_state.py` | Task State 模型、任务分类、约束抽取和跨轮次合并 |
| `src/rag/services/academic.py` | 教务只读查询与确定性时间冲突计算 |
| `scripts/seed_academic_demo.py` | 幂等写入 PostgreSQL Demo 教务数据 |
| `tests/test_agent_v2.py` | 多工具、Task State、多轮、恢复和 Loop 上限验收 |
| `docs/AGENT_V2.md` | V2 设计、启动、测试、Demo 和限制 |

## 3. 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/rag/infra/models.py` | 新增五张表：课程、成绩、课表、考试、Agent Task |
| `src/rag/infra/repository.py` | Memory/PostgreSQL 统一实现教务查询和 Task State 读写 |
| `src/rag/agent/tools.py` | V1 Tool 升级为统一 `success/data/error` 协议 |
| `src/rag/agent/agent.py` | 将显式 Task State 注入每次模型上下文 |
| `src/rag/agent/harness.py` | 升级成 Observe/Decide/Execute/Evaluate Loop，默认最多 8 轮 |
| `src/rag/agent/prompts.py` | 增加计划、选工具、评估 Tool Result、最终回答规则 |
| `src/rag/api/deps.py` | 注册六个工具和 AcademicService，Repository 作为 Task Store |
| `src/rag/api/schemas.py` | `ChatRequest` 增加可选 `student_id`，原字段保持兼容 |
| `src/rag/api/routes.py` | 把学生上下文传给 Harness，不传给 LLM 参数 |
| `src/rag/core/config.py` / `.env.example` | `AGENT_MAX_ITERATIONS` 默认改为 8 |
| `README.md` | 更新 V2 架构、文档和测试入口 |

## 4. Tools

所有工具执行统一流程：

```text
LLM 参数 → Pydantic 校验 → Service/Repository 查询 → 结构化结果 → Agent
```

成功：

```json
{"success": true, "data": {"courses": [], "count": 0}, "error": null}
```

失败：

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "invalid_arguments",
    "message": "工具参数校验失败，请根据工具 schema 修正参数后重试。",
    "retryable": true
  }
}
```

| Tool | 参数 | 返回事实 |
|---|---|---|
| `search_knowledge(query)` | 检索查询 | 原文、来源、页码、章节、分数 |
| `search_courses(...)` | keyword、term、department、category、学分范围、limit | 课程、教学班 ID、学分、教师、容量、上课时间 |
| `query_grades(...)` | term、status、course_code | 当前学生成绩；学生 ID 由上下文注入 |
| `query_schedule(term)` | term | 当前学生已选课表和时间地点 |
| `query_exam(...)` | term、course_code、from_at | 考试时间、地点、座位和状态 |
| `check_schedule_conflict(course_ids, term)` | 真实教学班 ID 列表 | 各候选与现有课表的时间重叠事实 |

冲突计算比较星期、开始/结束时间和教学周。Tool 只返回 `has_conflict` 与重叠区间，
是否选择课程仍由 Agent 判断。

## 5. Agent Loop

```text
载入 Conversation History + Task State
  ↓
Observe：读取用户目标、约束、已完成步骤
  ↓
Decide：模型选择 0..N 个 Tool
  ↓
Execute：参数校验、超时、结构化错误、Tool Result 回填
  ↓
Evaluate：任务所需事实是否齐全
  ├─ 不足且可恢复 → 修正参数或选择下一 Tool
  ├─ 需要用户偏好 → 提一个澄清问题，status=waiting_input
  ├─ 足够 → Final Answer，status=completed
  └─ 8 轮仍未完成 → status=failed
```

Harness 使用“任务完成契约”防止模型过早结束：

- 毕业进度至少需要 `search_knowledge + query_grades`；
- 关注时间或要求无冲突的选课任务至少需要
  `query_schedule + search_courses + check_schedule_conflict`；
- 成绩、课表、考试、知识问答分别必须成功取得对应事实。

这些是按任务类型定义的事实依赖，不是对某一句用户输入写 `if` 固定流程。模型仍决定
查询参数、调用顺序、是否重试以及最终分析。

## 6. Task State

持久化表：`agent_tasks`，以 `(tenant_id, thread_id)` 唯一定位。

```json
{
  "version": 2,
  "task_type": "course_selection",
  "goal": "我想选人工智能方向的课。",
  "constraints": ["direction:人工智能", "preference:课程时间"],
  "slots": {"direction": "人工智能", "preference": "课程时间"},
  "completed_steps": ["query_schedule#1", "search_courses#1"],
  "tool_results": [
    {"tool": "search_courses", "success": true, "item_count": 2}
  ],
  "status": "running",
  "last_user_input": "那机器学习呢？",
  "turn_count": 3
}
```

Task State 只保存任务语义和 Tool 摘要，不复制大段文档或完整成绩正文。下一轮需要权威事实
时重新查询 Tool，避免使用过期数据。

## 7. 多轮对话

Repository 继续保存自然语言消息；`agent_tasks` 单独保存任务状态。每次 `/chat`：

1. 根据 `thread_id` 读取最近消息；
2. 读取同一线程的 Task State；
3. 合并新约束，例如“课程时间”“机器学习”；
4. 把精简后的 Task State 作为 system context 注入 PiAgent；
5. 每次 Tool Call 后立即保存状态；最终写入 completed/waiting_input/failed。

因此下面三轮共享同一任务：

```text
我想选人工智能方向的课。
→ 你更关注课程时间还是培养方案要求？

课程时间。
→ 保存 preference:课程时间，并执行课程时间相关查询

那机器学习呢？
→ 继承 direction:人工智能 和 preference:课程时间
```

## 8. 数据库与启动

应用启动时 `Base.metadata.create_all()` 会创建新增表。生产环境仍建议改为经过审阅的
Alembic migration；`create_all` 不负责已有表的结构迁移。

写入 Demo 数据：

```bash
uv run python scripts/seed_academic_demo.py
```

Demo 学生 ID：`demo-student`。然后启动：

```bash
python scripts/run_api.py --port 8000
python -m rag.worker
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id":"demo-v2",
    "student_id":"demo-student",
    "question":"帮我找下学期不会和现有课程冲突的人工智能课程。"
  }'
```

生产环境不能信任客户端提交的 `student_id`，应由认证中间件根据登录令牌覆盖该字段。

## 9. 测试与结果

```bash
uv run pytest
uv run ruff check .
uv run python -m compileall -q src tests scripts
```

V2 测试覆盖：

- V1 RAG 问答和拒答回归；
- 个人成绩、课表、考试查询；
- 毕业进度两 Tool 编排；
- 选课三 Tool 编排和真实时间冲突；
- 三轮任务状态继承；
- Tool 参数错误后修正重试；
- 无学生身份的结构化失败；
- 8 轮 Loop 上限。

## 10. Demo

### 毕业进度

```text
[Task] graduation_progress
[Agent] selecting tool: search_knowledge
[Tool] search_knowledge completed
[Agent] selecting tool: query_grades
[Tool] query_grades completed
[Task] completed

结果：毕业要求 160 学分 [1]；已通过 120 学分；还差 40 学分。
```

### 无冲突课程

```text
[Task] course_selection
[Agent] selecting tool: query_schedule
[Agent] selecting tool: search_courses
[Agent] selecting tool: check_schedule_conflict
[Task] completed

结果：机器学习不冲突；人工智能导论与周一已有课程冲突。
```

### 多轮追问

```text
Turn 1: 我想选人工智能方向的课。
Turn 2: 课程时间。
Turn 3: 那机器学习呢？

Task State 保留：direction=人工智能、preference=课程时间、course_interest=机器学习。
```

## 11. 当前问题

1. 当前只有只读镜像表，没有接入学校真实教务 API、CDC 或同步任务。
2. `student_id` 仅为 Demo/API 兼容入口，生产必须由认证层注入并做租户授权。
3. 新表使用 `create_all`，生产缺少 Alembic 版本化迁移和回滚脚本。
4. Task 类型和约束抽取目前是轻量规则；跨领域复杂表达仍可能分类不准。
5. Task State 保存 Tool 摘要，不保存完整事实；下一轮会重新查询，可靠但增加延迟。
6. 课程冲突只覆盖结构化时间，不处理通勤时间、容量变化、先修课和考试冲突。
7. SSE 仍是完整 Loop 结束后分片输出，无法实时展示每一步 Tool Event。
8. 真实 PostgreSQL/AutoDL 环境需在部署侧执行 Demo seed 或接入真实数据后再做联调。

## 12. 下一阶段建议

- 先接认证与教务只读同步，建立 student_id 的可信边界；
- 增加 Alembic migration、数据新鲜度字段和审计日志；
- 建立多学科、多任务离线评测集，评测 Tool 选择、任务完成率和错误恢复率；
- 增加先修课、容量、考试时间和培养方案类别学分校验；
- 将 Tool/Task Event 原生流式推给前端；
- 以上完成后再评估 Supervisor/Worker Multi-Agent，本次 V2 不实现。
