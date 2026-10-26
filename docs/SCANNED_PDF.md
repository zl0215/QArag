# 扫描 PDF 处理

## 处理链路

```text
Upload / Worker
  → IngestionService
  → 后台线程解析 PDF
  → 逐页检查文本层、乱码与页面图像
  → 原生页保留 pdfplumber 文本
  → 扫描页限像素渲染
  → RapidOCR 检测、方向分类、文字识别
  → 置信度过滤与双栏阅读顺序恢复
  → DocNode
  → 原生 / OCR 分界处分块
  → PostgreSQL 正文与 OCR 元数据
  → Milvus Dense + BM25
  → search_knowledge
  → Agent Answer + Citation
```

OCR 使用本地 CPU 推理，独立于 Embedding 与 LLM Provider。扫描页不上传到外部 OCR 服务。PDF 渲染与 OCR 推理分别使用进程内锁，模型按目录和线程配置缓存。PDF 解析在后台线程执行，避免占用 API 事件循环。

## 安装

```bash
uv sync --extra dev --extra ocr
uv run --extra ocr python scripts/setup_ocr.py
```

`setup_ocr.py` 将 RapidOCR 3.10 系列随包提供的模型复制到 `PDF_OCR_MODEL_DIR`，生成 SHA-256 清单。遇到内容不同的已有模型时停止，防止覆盖。运行时明确指定本地文件路径。

```text
models/ocr/
├── det.onnx
├── cls.onnx
├── rec.onnx
└── manifest.json
```

识别框架与模型接口见 [RapidOCR 官方项目](https://github.com/RapidAI/RapidOCR)。

## 配置

| 配置 | 默认值 | 用途 |
|---|---|---|
| `PDF_OCR_MODE` | `auto` | `auto` 按页检测；`off` 拒绝缺少可靠文本层的页；`force` 每页执行 OCR |
| `PDF_OCR_MODEL_DIR` | `./models/ocr` | 检测、方向分类、识别模型目录 |
| `PDF_OCR_DPI` | `200` | 目标渲染分辨率 |
| `PDF_OCR_MIN_TEXT_CHARS` | `40` | 扫描页检测的有效文本字符阈值 |
| `PDF_OCR_MIN_CONFIDENCE` | `0.6` | 文本框最低识别分数 |
| `PDF_OCR_MAX_REJECTED_RATIO` | `0.3` | 页面允许的低置信文本框占比 |
| `PDF_OCR_MAX_PAGES` | `100` | 单文档最多处理的 OCR 页数 |
| `PDF_OCR_MAX_PIXELS` | `12000000` | 单页渲染像素预算 |
| `PDF_OCR_THREADS` | `2` | ONNX Runtime CPU 线程数 |
| `CHUNKER_VERSION` | `v4-layout-ocr` | 分块策略版本 |

扫描页判定结合有效文字数量、乱码标记、页面图像覆盖率与绘图对象。短原生标题不会使其下方的大幅扫描图像被直接跳过。无图像、无绘图内容的原生空白页保留为空页。

渲染 DPI 会在像素预算内降低。纯白扫描页在模型加载前被排除。方向分类用于文本行的方向修正；整页大角度旋转与复杂阅读顺序仍需专项验证。

## 启动

```bash
uv run --extra ocr python scripts/run_api.py --port 8000
```

异步上传需要同时运行：

```bash
uv run --extra ocr python -m rag.worker
```

API 与 Worker 必须使用同一组 OCR 模型、阈值与分块版本。

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents?sync=false" \
  -F "file=@scanned-document.pdf"
```

文档成功入库后，继续使用 `/api/v1/retrieve` 和 `/api/v1/chat`。OCR 数据通过现有 `search_knowledge` Tool 返回 Agent。

## 容器部署

Compose 默认传入 `WITH_OCR=true` 构建 OCR 依赖，并将 `PDF_OCR_MODEL_DIR` 指向 `/models/ocr`。模型通过既有只读模型卷提供。

在宿主机准备模型：

```bash
uv sync --extra ocr
uv run --extra ocr python scripts/setup_ocr.py --model-dir ./models/ocr
docker compose up -d --build
```

仅使用容器准备模型：

```bash
docker compose build api worker
docker compose run --rm --no-deps --user 0:0 \
  -v ./models/ocr:/ocr-output api \
  python scripts/setup_ocr.py --model-dir /ocr-output
docker compose up -d
```

`MODEL_DIR` 指向自定义目录时，准备模型命令与可写卷的宿主机路径也应指向该目录下的 `ocr/`。

## 来源追溯

`DocNode` 保留 OCR 文本框对应的 PDF 坐标、页码、提取方式与识别分数。PostgreSQL `documents.extra` 保存 `ocr_pages`、`blank_pages`、逐页统计及解析告警，复用既有 JSONB 字段，无新增数据库表或列。

检索结果及问答引用增加：

```json
{
  "page_start": 2,
  "page_end": 2,
  "extraction_method": "ocr",
  "ocr_confidence": 0.97
}
```

问答引用的页码包含在原有 `label` 中。`extraction_method` 可为 `native`、`ocr` 或 `mixed`；`ocr_confidence` 是涉及 OCR 页的最低平均识别分数，不是答案正确率。引用校验检查识别后正文的一致性，不能证明扫描图像中的每个字符均已识别正确。

## 失败与恢复

| 情况 | 状态 / 响应 | 处理 |
|---|---|---|
| 未安装 OCR 依赖 | `needs_ocr` / HTTP 422 | 安装 `ocr` 扩展 |
| 模型文件缺失 | `needs_ocr` / HTTP 422 | 执行 `setup_ocr.py` |
| `PDF_OCR_MODE=off` 遇到扫描页 | `needs_ocr` / HTTP 422 | 改为 `auto` 后重新上传 |
| 识别失败或低置信框占比超限 | `needs_ocr` / HTTP 422 | 提供清晰扫描件或人工检查 |
| 超过页数或像素上限 | `needs_ocr` / HTTP 422 | 拆分文档或调整资源预算 |
| 全文为空，只有标题元数据 | `failed` | 检查源文档，不建立正文索引 |

混合文档中任一扫描页无法处理时，整份文档停止摄取，不会以部分正文标记为 `ready`。OCR 错误属于 Worker 永久错误，不会重复消耗同一任务的重试次数。安装依赖或修复配置后重新上传，系统复用失败文档记录；异步任务沿用已有重新排队逻辑。

已处于 `ready` 的文档仍按 SHA-256 去重。需要重新识别旧扫描文档时，先备份，再通过文档删除接口移除旧版本并重新上传。单纯重启服务不会自动重写既有正文或向量。

## 验证

```bash
uv run --extra dev --extra ocr pytest
uv run --extra dev --extra ocr ruff check .
uv run --extra dev --extra ocr python eval/smoke_scanned_pdf.py
```

真实 OCR 验收会生成纯扫描、原生与扫描混合、空白扫描三种 PDF，使用真实 RapidOCR 模型执行识别，再通过既有摄取和 BM25 检索验证来源页码。语义向量使用测试替身，验收结果不计入 BGE-M3 检索效果指标。

| 用例 | 结果 |
|---|---|
| 中文扫描页关键短语 | 5/5 |
| 混合文档扫描页关键短语 | 5/5 |
| 纯扫描文档来源页码 | 第 1 页 |
| 混合文档来源页码 | 第 2 页 |
| 英文双栏页面 | 恢复左栏后右栏顺序 |
| 空白扫描页 | 识别为空白页，跳过模型推理 |

验收结果保存在 `eval/results/ocr_smoke.json`。测试覆盖缺失模型、OCR 关闭、低置信识别、页面预算、失败后恢复、API 422 状态及正文来源字段。合成样本保存在 `data/ocr_acceptance/`，不进入 Git。

Linux 生成中文验收样本需要中文字体，可安装 Noto CJK 或显式指定：

```bash
uv run --extra dev --extra ocr python eval/smoke_scanned_pdf.py \
  --font /path/to/NotoSansCJK-Regular.ttc
```

## 后续任务

- 低清、噪点、阴影、手写与整页旋转文档的独立字符错误率评测
- OCR 队列按页进度、结果缓存、硬超时与任务恢复
- 扫描表格的单元格、合并格与跨页结构恢复
- 公式识别与图注关联
- 原始扫描页和 OCR 文本框的可视化校对

当前扫描表格以识别文本参与检索，不保证恢复单元格拓扑。严重模糊、复杂合并表格和手写内容仍需要人工核对。
