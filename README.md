# 📚 Doc RAG Evidence System

多模态文档检索增强问答与证据定位系统

## 🚀 Quick Start (推荐 - V1 UI集成版)

### 一键启动
```bash
cd /workspace/doc-rag-evidence
./start.sh
```

这将自动：
1. ✅ 启动 HunyuanOCR (GPU 0, Port 8000)
2. ✅ 启动 Qwen3-Embedding (GPU 1, Port 8001)
3. ✅ 启动 UI 界面 (Port 7860)

### 访问UI
浏览器打开: **http://localhost:7860**

### UI操作流程
1. **📤 上传文档**: Document Management → Upload PDF → ☑ Use OCR → Ingest
2. **⚙️ 构建索引**: Document Management → Build Indices → 选择索引类型 → Build
3. **🔍 查询**: Query & Answer → 输入问题 → Ask Question

### 停止服务
```bash
./scripts/stop_all_vllm.sh
```

### 📚 完整文档
- **[快速启动指南](docs/QUICKSTART.md)** - 详细使用说明
- **[系统升级说明](docs/VLLM_UPGRADE.md)** - vLLM双卡部署详解  
- **[HunyuanOCR配置](docs/HUNYUAN_OCR_GUIDE.md)** - OCR服务配置

---

## 🎯 系统简介

本系统提供完整的文档检索增强问答（RAG）能力，支持：

### V0 (Baseline) ✅
- ✅ PDF文档导入与结构化处理
- ✅ BM25文本检索（块级/页级可选）
- ✅ 问答生成与证据追溯
- ✅ 可视化Web界面（Gradio）
- ✅ 批量评测与报告导出
- ✅ 完整的运行日志记录（软著友好）

### V0.1+ (Multi-Modal) 🆕
- ✅ **Page Rendering**: PyMuPDF高质量页面渲染
- ✅ **OCR Integration**: SGLang API集成（DeepSeek/Hunyuan OCR）
- ✅ **Dense Text Retrieval**: FAISS + Qwen3-Embedding语义检索
- ✅ **ColPali Vision Retrieval**: 两阶段视觉检索（全局+Late Interaction）
- ✅ **Unified Block Builder**: 统一的文本块生成（OCR/文本分割）
- ✅ **Multi-Mode UI**: 检索模式切换器（BM25/Dense/ColPali）
- ✅ **GPU Resource Management**: 单GPU部署ColPali，避免显存冲突

---

## 🚀 快速开始

### V0 (Baseline)

**依赖**：
```bash
pip install pydantic pyyaml pdfplumber rank-bm25 gradio
```

**启动**：
```bash
python run.py
# 界面: http://127.0.0.1:7860
```

### V0.1+ (Multi-Modal)

**依赖**：
```bash
pip install -r requirements.txt
# 包含: pymupdf, Pillow, faiss-cpu, torch, transformers
```

**配置** (`configs/app.yaml`):
```yaml
# 启用OCR
ocr:
  provider: "sglang"
  model: "deepseek_ocr"
  endpoint: "http://127.0.0.1:30000"

# 启用Dense检索
dense:
  enabled: true
  model: "Qwen/Qwen3-Embedding-0.6B"
  endpoint: "http://127.0.0.1:30000"

# 启用ColPali视觉检索
colpali:
  enabled: true
  model: "vidore/colqwen2-v0.1"
  device: "cuda:0"
```

**一键启动完整流程**：
```bash
# 1. 启动SGLang服务器（另一终端）
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model Qwen/Qwen3-Embedding-0.6B \
  --port 30000

# 2. 导入文档 + 构建索引 + 启动UI
python run_v1.py \
  --ingest-dir data/pdfs \
  --use-ocr \
  --build-all \
  --ui

# 界面: http://localhost:7860
# 在UI中选择检索模式: BM25 / Dense / ColPali
```

### 3. 命令行使用

#### 构建索引
```bash
python scripts/build_index.py
```

#### 单次问答
```bash
python scripts/demo_run.py "What is the main topic of the document?"
```

#### 批量评测
```bash
python scripts/demo_eval.py data/sample_eval.csv
```

---

## 📁 项目结构

```
doc-rag-evidence/
├── configs/
│   └── app.yaml              # 系统配置（路径、粒度、参数）
├── core/
│   ├── schemas.py            # 数据模型定义
│   ├── inferences.py         # 接口协议（Protocol）
│   └── pipeline.py           # 核心Pipeline
├── infra/
│   ├── store_local.py        # 本地文件存储
│   └── runlog_local.py       # 运行日志记录
├── impl/
│   ├── ingest_pdf.py         # PDF导入器（pdfplumber）
│   ├── index_bm25.py         # BM25索引+检索
│   ├── selector_topk.py      # TopK证据选择
│   ├── generator_template.py # 模板式生成器
│   └── eval_runner.py        # 评测运行器
├── app/
│   └── ui/
│       └── main.py           # Gradio UI
├── scripts/
│   ├── build_index.py        # 构建索引脚本
│   ├── demo_run.py           # 单次问答脚本
│   └── demo_eval.py          # 批量评测脚本
├── data/
│   ├── docs/                 # 文档工件目录
│   ├── indices/              # 索引文件目录
│   ├── runs/                 # 运行日志目录
│   ├── reports/              # 评测报告目录
│   └── sample_eval.csv       # 示例评测数据集
└── run.py                    # 快速启动脚本
```

---

## 🔧 配置说明

配置文件：[configs/app.yaml](configs/app.yaml)

关键配置项：
```yaml
# 数据目录
data_root: "data"
docs_dir: "data/docs"
indices_dir: "data/indices"
runs_dir: "data/runs"
reports_dir: "data/reports"

# 索引粒度（page | block）
chunk_level: "block"

# 检索参数
top_k_retrieve: 20    # 初始召回数量
top_k_evidence: 5     # 最终证据数量

# 引用级别（page | block）
citation_level: "block"
```

---

## 📊 数据工件路径规范

系统遵循以下路径约定（软著友好，可追溯）：

```
data/
├── docs/
│   └── {doc_id}/
│       ├── meta.json                         # 文档元数据
│       └── pages/
│           └── {page_id:04d}/
│               ├── text.json                 # 页面文本
│               ├── blocks.json               # 块列表
│               └── page.png                  # 页面图片（可选）
├── indices/
│   └── bm25_default/
│       └── index.pkl                         # BM25索引
├── runs/
│   └── {query_id}.json                       # 单次问答完整日志
└── reports/
    └── {dataset}/
        └── {timestamp}/
            ├── predictions.csv               # 每个问题的结果
            └── report.json                   # 汇总指标
```

---

## 🎓 典型使用流程

### 场景1：文档问答（Web界面）

1. 打开Web界面：`python run.py`
2. 进入"文档管理"标签，上传PDF文件
3. 系统自动完成：文档导入 → 文本提取 → 索引构建
4. 进入"问答查询"标签，输入问题
5. 查看答案、证据列表、引用信息

### 场景2：批量评测（命令行）

1. 准备评测数据集（CSV或JSON格式）：
   ```csv
   qid,question,answer_gt
   q1,What is the main topic?,
   q2,What are the key findings?,
   ```

2. 运行评测：
   ```bash
   python scripts/demo_eval.py dataset.csv
   ```

3. 查看结果：
   - 控制台显示成功率、平均延迟
   - `data/reports/{dataset}/{timestamp}/predictions.csv` - 详细结果
   - `data/reports/{dataset}/{timestamp}/report.json` - 汇总指标

### 场景3：单次问答调试（命令行）

```bash
python scripts/demo_run.py "What is the publication date?"
```

输出包括：
- 答案文本
- 证据列表（doc_id、page_id、score、snippet）
- 运行日志路径（用于追溯）
- 计时信息（retrieval、evidence、generation）

---

## 📈 扩展方向

V0版本为后续扩展预留接口：

### 已实现（V0）
- ✅ PDF文本提取（pdfplumber）
- ✅ BM25检索
## ✅ V0 功能（已实现）
- ✅ PDF文档导入与分块
- ✅ BM25检索（block/page级）
- ✅ 证据选择与排序
- ✅ 模板式生成（无需API）
- ✅ 运行日志与追溯

## ✅ V0.1 功能（已实现）🆕
- ✅ PyMuPDF页面渲染（144 DPI PNG）
- ✅ OCR集成（SGLang API，DeepSeek/Hunyuan）
- ✅ 统一Block Builder（OCR/文本分割）
- ✅ Dense Text检索（FAISS + Qwen3-Embedding）
- ✅ ColPali视觉检索（两阶段Late Interaction）
- ✅ 多模态UI（BM25/Dense/ColPali切换）
- ✅ GPU资源管理（单卡ColPali）

## 🔲 待扩展（V0.2+）
- 🔲 Hybrid融合策略（多检索源加权）
- 🔲 重排序器（Qwen3-Reranker）
- 🔲 真实LLM集成（替换模板生成器）
- 🔲 扩展评测指标（recall@k, MRR, NDCG）
- 🔲 增量索引更新
- 🔲 多语言OCR支持
- 🔲 BBox定位与高亮

---

## 📚 文档索引

| 文档 | 描述 | 读者 |
|------|------|------|
| [README.md](README.md) | 快速入门和总览 | 所有用户 |
| [docs/user_manual_v0.md](docs/user_manual_v0.md) | V0详细用户手册 | V0用户 |
| [docs/v0.1_multimodal_retrieval.md](docs/v0.1_multimodal_retrieval.md) | V0.1多模态检索指南 | V0.1用户 |
| [docs/v0.1_implementation_summary.md](docs/v0.1_implementation_summary.md) | V0.1技术实现总结 | 开发者 |
| [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) | V0.1实现完成报告 | 项目经理 |

---

## 🛠 故障排除

### 问题1：索引为空，无法检索
```bash
# 重新构建索引
python scripts/build_index.py
```

### 问题2：PDF导入失败
- 确认 `pdfplumber` 已安装：`pip install pdfplumber`
- 检查PDF文件是否损坏
- 查看错误日志：`data/runs/{query_id}.json`

### 问题3：UI无法启动
- 确认 `gradio` 已安装：`pip install gradio`
- 检查端口7860是否被占用

---

## 📝 软著交付清单

V0版本已满足软著登记要求：

- ✅ 可运行程序（`python run.py`）
- ✅ 用户手册（本README + `docs/user_manual_v0.md`）
- ✅ 源代码（完整项目结构）
- ✅ 运行日志（`data/runs/` 目录）
- ✅ 界面截图（可从Web界面获取）
- ✅ 评测报告（`data/reports/` 目录）

---

## 📄 开源协议

MIT License

---

## 👥 贡献指南

欢迎提交Issue和Pull Request！

开发建议：
1. 遵循现有代码风格（类型提示、文档字符串）
2. 新增模块请实现对应的Protocol接口
3. 添加单元测试（可选，V0未强制要求）
4. 更新本README和配置文档

---

## 📞 联系方式

如有问题，请提交GitHub Issue。
