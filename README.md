# Doc RAG Evidence

> 🚀 **多模态文档检索增强生成系统** - 支持视觉理解与精确引用的智能问答平台

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/gradio-6.2.0-orange.svg)](https://gradio.app/)

**当前版本**：V1.0 (2026-01) | [版本详情](VERSION.md)

---

## 📖 简介

Doc RAG Evidence 是一款基于检索增强生成（RAG）技术的企业级文档问答系统，利用 Qwen3-VL 多模态大模型，实现对 PDF 文档的智能理解和精确引用。

### ✨ 核心亮点

- 🎨 **多模态理解**：同时处理文本和图像，理解图表、表格、公式
- 🔍 **智能检索**：BM25 + 语义向量 + 视觉检索三合一
- 🔗 **精确引用**：每条答案自动标注证据来源 [1][2]...
- 🎛️ **灵活配置**：Hybrid 融合检索，支持自定义权重
- 📦 **批量处理**：支持多文档上传和批量评估
- 🖥️ **Web 界面**：无代码操作，开箱即用

### 🎯 适用场景

| 场景 | 描述 |
|-----|------|
| 📚 **企业知识库** | 技术文档、标准规范、内部资料快速查询 |
| 📑 **合同审查** | 法律条款检索、风险点识别 |
| 🔬 **学术研究** | 论文阅读、文献调研、引用追踪 |
| 📊 **财报分析** | 图表理解、数据提取、趋势分析 |
| 🛠️ **技术手册** | 产品说明、操作指南、故障排查 |

---

## 🏗️ 系统架构

```mermaid
graph TD
    A[PDF 文档] --> B[HunyuanOCR 识别]
    B --> C[块级切分]
    C --> D[索引构建]
    D --> E1[BM25 索引]
    D --> E2[Dense 索引 FAISS]
    D --> E3[ColPali 索引]
    
    F[用户查询] --> G[混合检索]
    E1 --> G
    E2 --> G
    E3 --> G
    G --> H[证据选择]
    H --> I[Qwen3-VL 生成]
    I --> J[答案 + 引用]
```

### 技术栈

| 层级 | 组件 | 技术 |
|-----|-----|------|
| **前端** | Web UI | Gradio 6.2.0 |
| **检索** | 关键词 | BM25 (Rank-BM25) |
| | 语义 | Qwen3-Embedding-0.6B + FAISS |
| | 视觉 | ColPali (ColQwen3-Embed-4B) |
| **生成** | MLLM | Qwen3-VL-4B-Instruct (vLLM) |
| **OCR** | 识别 | HunyuanOCR (vLLM) |
| **存储** | 文档 | Local FS (JSON) |
| | 索引 | FAISS + BM25 Index |

### GPU 资源分配

```

  GPU0: HunyuanOCR          (Port 8000) │  ← 文档 OCR 识别
  GPU1: Qwen3-Embedding     (Port 8001) │  ← 语义向量编码
  GPU2: ColPali             (延迟加载)  │  ← 视觉检索
  GPU3: Qwen3-VL-4B         (Port 8002) │  ← 答案生成

```

---

## 🚀 快速开始

### 环境准备

**硬件要求**：
- 4x NVIDIA GPU（每张 ≥8GB 显存）
- 128GB 内存
- 500GB 存储空间

**软件要求**：
- Linux (Ubuntu 20.04+)
- Python 3.9+
- CUDA 11.8+

### 一键启动

```bash
# 1. 启动所有后台服务（OCR + Embedding + Generation）
bash scripts/start_services.sh

# 等待服务就绪（约 30-60 秒）

# 2. 启动 Web UI
bash scripts/start_ui.sh

# 3. 访问界面
# 浏览器打开: http://localhost:7860
```

### 基本使用

#### 1️⃣ 上传文档

```
 "Document Management" 标签
 点击 "Upload PDF Files"（支持多选）
 勾选 "Use OCR"（扫描版必选）
 点击 "Ingest Documents"
```

#### 2️⃣ 构建索引

```
 "Build Indices" 区域
 勾选索引类型：
   ☑ BM25（推荐）
   ☑ Dense（推荐）
   ☐ ColPali（可选）
 点击 "Build Selected Indices"
```

#### 3️⃣ 开始提问

```
 "Query & Answer" 标签
 输入问题（如："碳酸氢钠的用途是什么？"）
 选择检索模式：
   • BM25：关键词精确匹配
   • Dense：语义理解
   • ColPali：视觉理解
   • Hybrid：混合检索（推荐）
 选择证据格式：
   • text：文本片段（精确引用）
   • image：页面图片（适合图表）
 点击 "Ask Question"
```

#### 4️⃣ 查看结果

- **检索结果**：候选文档列表（doc_id, page, score）
- **证据**：选中的段落/页面（带来源标注）
- **答案**：生成的回答 + 引用标记 [1][2]...

---

## 💡 核心功能

### 1. 多模态生成

| 模式 | 输入 | 优势 | 适用场景 |
|-----|------|------|---------|
| **text** | OCR 文本片段 | 精确引用段落，token 消耗少 | 纯文本文档、合同、论文 |
| **image** | 页面图片 | 理解图表、表格、公式、布局 | 技术手册、财报、研究报告 |

**生成引擎**：Qwen3-VL-4B-Instruct
- 参数规模：4B
- 能力：多模态理解、引用控制、幻觉抑制
- 部署：vLLM 高性能推理

### 2. 混合检索（Hybrid Retrieval）

**可选组合**：
- `BM25 + Dense`：关键词 + 语义（通用推荐）
- `BM25 + ColPali`：关键词 + 视觉（图表文档）
- `Dense + ColPali`：语义 + 视觉（探索性查询）

**融合策略**：
1. **weighted_sum**（加权求和）
   ```python
   final_score = w1 * score1 + w2 * score2
   ```
   - 适合：分数尺度一致的检索器
   - 参数：调节 w1, w2 (和为 1.0)

2. **rrf**（倒数排名融合，推荐）
   ```python
   final_score = 1/(k + rank1) + 1/(k + rank2)
   ```
   - 适合：跨模态融合（如 BM25 + ColPali）
   - 优势：对分数尺度不敏感

**权重调节**：
- UI 滑块实时调整（0.0 - 1.0）
- 动态显示权重分配
- 支持快速 A/B 测试

### 3. 批量处理

**多文档上传**：
- Gradio 多选文件上传
- 队列式串行处理
- 实时进度反馈

**增量索引**：
- 新文档自动加入已有索引
- 无需重建全部索引
- 节省时间和计算资源

**批量评估**：
- 支持 CSV/JSON 数据集
- 自动运行问答
- 导出评估报告（predictions + metrics）

### 4. 文档处理

**高精度 OCR**：
- 引擎：HunyuanOCR（腾讯混元）
- 支持：中英文、表格、公式、手写体
- 输出：结构化文本 + 布局信息

**块级切分**：
- 粒度：page → blocks
- 策略：保留语义完整性
- 存储：text.json + blocks.json + page.png

---

## 📂 项目结构

```
doc-rag-evidence/
 app/ui/                 # Gradio Web UI
   └── main_v1.py          # 主界面
 core/                   # 核心抽象层
   ├── schemas.py          # 数据模型（Pydantic）
   └── pipeline.py         # RAG 处理流程
 impl/                   # 功能实现层
   ├── ingest_pdf_v1.py    # PDF 摄取 + OCR
   ├── index_bm25.py       # BM25 索引器
   ├── index_dense.py      # Dense 索引器（FAISS）
   ├── index_colpali.py    # ColPali 索引器
   ├── retriever_hybrid.py # 混合检索器
   ├── generator_qwen_vl.py # 多模态生成器
   ├── selector_topk.py    # 证据选择器
   └── eval_runner.py      # 评估运行器
 infra/                  # 基础设施层
   ├── store_local.py      # 文档存储
   └── runlog_local.py     # 运行日志
 scripts/                # 运维脚本
   ├── start_services.sh   # 一键启动
   ├── start_ui.sh         # 启动 UI
   └── stop_all_vllm.sh    # 停止服务
 configs/
   └── app.yaml            # 配置文件
 data/                   # 数据目录
    ├── docs/               # 文档工件
    ├── indices/            # 索引文件
    ├── runs/               # 查询日志
    └── reports/            # 评估报告
```

---

## ⚙️ 配置

### 核心参数（configs/app.yaml）

```yaml
# 检索配置
chunk_level: block          # page | block
top_k_retrieve: 20          # 检索召回数
top_k_evidence: 5           # 最终证据数

# 生成配置
generator:
  type: qwen3_vl            # qwen3_vl (MLLM) | qwen3_llm | template
  
llm:
  backend: vllm
  model: "Qwen/Qwen3-VL-4B-Instruct"
  endpoint: "http://localhost:8002"
  max_new_tokens: 1024
  temperature: 0.1          # 0.0-1.0，越低越确定
  citation_policy: strict   # 强制引用证据
```

### 服务端口

| 服务 | 端口 | GPU | 用途 |
|-----|------|-----|------|
| OCR | 8000 | 0 | 文档识别 |
| Embedding | 8001 | 1 | 向量编码 |
| Generation | 8002 | 3 | 答案生成 |
| ColPali | - | 2 | 视觉检索（延迟加载）|
| UI | 7860 | - | Web 界面 |

---

## 📊 功能清单

### ✅ 已实现

| 功能 | 状态 | 说明 |
|-----|:----:|------|
| PDF 文档摄取 | ✅ | 批量上传 + OCR 识别 |
| 索引构建 | ✅ | BM25/Dense/ColPali 增量更新 |
| 单一检索 | ✅ | 三种模式独立运行 |
| 混合检索 | ✅ | 任意组合 + 权重调节 |
| 文本生成 | ✅ | 基于 OCR 文本片段 |
| 图像生成 | ✅ | 基于页面图片（多模态）|
| Web UI | ✅ | 4 大功能模块 |
| 批量评估 | ✅ | CSV/JSON 数据集 |

### 🔄 待改进（V1.1）

| 问题 | 影响 | 优先级 |
|-----|------|:------:|
| ColPali 证据无文本 | 图像模式体验 | 🔴 P0 |
| 检索返回不统一 | 代码复杂度 | 🟡 P1 |
| 评估指标单一 | 效果评估 | 🟡 P1 |

---

## 📚 文档

- � [版本说明](VERSION.md) - 版本历史和开发路线图

---

## 🗺️ 路线图

### V1.1 - 证据统一与优化（进行中）

- [ ] **T1** - ColPali 证据落块（page → block expansion）
- [ ] **T2** - 统一检索返回契约（normalize_hits）
- [ ] **T3** - 优化 Qwen3-VL 生成质量（prompt 工程）
- [ ] **T4** - 证据上下文装配与去重
- [ ] **T5** - 多轮对话支持

### V1.2 - 评估与性能（规划中）

- [ ] **T6** - 三层评估指标（检索/证据/生成）
- [ ] **T7** - Reranker 集成
- [ ] **T8** - 性能优化（缓存、并发）

---

## 🤝 贡献

 Issue 和 Pull Request！

## 📄 许可证

 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**最后更新**：2026-01-14  
**维护团队**：Doc RAG Evidence Team

[🏠 首页](README.md) • [� 版本说明](VERSION.md) • [🐛 报告问题](https://github.com/576469377/doc-rag-evidence/issues)

</div>
