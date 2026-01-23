# Doc RAG Evidence - 版本说明

## 当前版本：V1.2.1 - ColPali 多进程优化

**发布时间**：2026-01-23
**状态**：✅ 已发布

---

## 版本历史

### V1.2.1 - ColPali 多进程优化（2026-01-23）

**核心改进**：ColPali 索引构建速度优化 + 图像分辨率配置

#### 新增功能

**ColPali Multiprocessing 支持**
- ✅ 支持多 worker 并行索引（2x 加速）
- ✅ 使用 spawn 模式避免 CUDA fork 冲突
- ✅ 懒加载优化：主进程不加载模型，节省 ~7GB GPU 显存
- ✅ 每个 worker 独立模型实例，支持 GPU 共享
- ✅ 实时进度显示：每个 worker 显示独立进度条和每页处理时间

**命令行索引构建工具**
- ✅ `scripts/build_indices_v1.py` 支持增量索引
- ✅ 支持单索引构建：`--colpali`, `--dense-vl`, `--bm25`
- ✅ 无需启动 UI 即可构建索引
- ✅ 自动进度显示和错误处理

#### 优化改进

**显存优化**
- ✅ ColPali 和 Dense-VL 图像 resize 参数（max_image_size）
- ✅ 默认 768px 分辨率（平衡速度和质量）
- ✅ 支持 512px（极快）/768px（平衡）/1024px（质量）/2048px（高质量）
- ✅ GPU2 共享机制：ColPali + Dense-VL 可共享同一 GPU

**UI 改进**
- ✅ 移除不准确的估算时间显示
- ✅ 显示实际处理进度和页数
- ✅ 启动进度条（7 步初始化）

**索引增量更新**
- ✅ 每 10 个文档自动保存检查点
- ✅ 支持 resume 和增量添加
- ✅ IndexTracker 追踪已索引文档

#### Bug 修复

- ✅ **Dense-VL 索引重复问题**：修复 `build_dense_vl_index.py` 缩进错误导致的重复向量
- ✅ **ColPali max_image_size 未传递**：修复 `index_incremental.py` 参数传递问题
- ✅ **ColPali worker 变量名 typo**：修复 `_worker_colpali_max_image_SIZE` 拼写错误

#### 性能数据

| 场景 | 配置 | 性能 |
|-----|------|------|
| ColPali 索引 | 2 workers, 768px | ~20-25s/page (40s → 25s) |
| Dense-VL 索引 | 4 workers, Flash Attention 2 | ~0.3s/page |
| 增量保存 | 每 10 docs | 支持断点续传 |

#### 配置示例

```yaml
colpali:
  enabled: true
  model: "/workspace/cache/tomoro-colqwen3-embed-4b"
  gpu: 2
  max_image_size: 768        # 图像分辨率优化
  num_workers: 2             # 并行 worker 数量
  batch_size: 8

dense_vl:
  enabled: true
  model_path: "/workspace/cache/Qwen3-VL-Embedding-2B"
  gpu: 2                     # 与 ColPali 共享 GPU
  max_image_size: 1024
  num_workers: 4
```

---

### V1.2 - Dense-VL 多模态检索（2026-01-21）

**核心新增**：Dense-VL 多模态检索 + 性能优化三板斧

#### 新增功能

**Dense-VL 多模态检索**
- ✅ Qwen3-VL-Embedding-2B 页面级索引
- ✅ 同时理解文本和图像内容
- ✅ 支持图表、表格、公式理解
- ✅ 与 ColPali 共享 GPU（延迟加载）
- ✅ Hybrid 检索支持 Dense-VL 组合

**性能优化三板斧（8-12x 加速）**
- ✅ **Flash Attention 2**：~2x 加速（自动检测，无则降级）
- ✅ **图像压缩**：长边压缩至 1024px，~2x 加速
- ✅ **并行索引**：4 worker 并行处理，~4x 加速

**Hit Normalization**
- ✅ 页面级检索结果自动扩展为块级证据
- ✅ 支持 ColPali/Dense-VL 页面 → 块展开
- ✅ 统一证据格式

#### 技术实现

**Flash Attention 2 集成**
```python
self.model = Qwen3VLEmbedder(
    model_name_or_path=model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 自动加速
)
```

**图像 Resize 优化**
```python
def _resize_image_if_needed(self, image_path: str) -> str:
    # 自动调整至 max_image_size
    if max_dim > self.max_image_size:
        scale = self.max_image_size / max_dim
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
```

**多进程索引构建**
```python
# Dense-VL: 4 workers
# ColPali: 2 workers (显存占用更大)
pool = mp.Pool(
    processes=num_workers,
    initializer=init_worker,
    initargs=(model_path, gpu_id, max_image_size)
)
```

#### 性能对比

| 场景 | 优化前 | 优化后 | 加速比 |
|-----|--------|--------|--------|
| 56 页 Dense-VL 索引 | 140s | 15s | **9.3x** |
| 单页 ColPali 处理 | 40s | 25s | **1.6x** |

---

### V1.1 - 多模态生成（2026-01-18）

**核心新增**：Qwen3-VL 多模态生成

#### 新增功能

**多模态生成引擎**
- ✅ Qwen3-VL-4B-Instruct 真实 LLM 生成
- ✅ 支持 text（OCR 文本）和 image（页面图片）两种证据格式
- ✅ 图像证据支持图表、表格、公式理解
- ✅ 引用控制：strict/relaxed/none 三种策略
- ✅ vLLM 高性能推理（GPU3）

**生成模式对比**

| 模式 | 输入 | 优势 | 场景 |
|-----|------|------|------|
| text | OCR 文本片段 | 精确引用，token 少 | 纯文本文档 |
| image | 页面图片 | 理解图表、表格 | 财报、技术手册 |

#### 配置示例

```yaml
llm:
  backend: vllm
  model: "Qwen/Qwen3-VL-4B-Instruct"
  endpoint: "http://localhost:8002"
  gpu: 3
  max_new_tokens: 2048
  temperature: 0.1
  citation_policy: strict
```

---

### V1.0 - 基础链路（2026-01-14）

**核心功能**：基础 RAG 链路跑通

#### 已实现功能

**文档处理**
- ✅ PDF 解析 + HunyuanOCR 识别
- ✅ 块级切分（page → blocks）
- ✅ 元数据管理（doc_id/page_id/block_id）

**索引构建**
- ✅ BM25 索引（关键词检索）
- ✅ Dense 索引（FAISS 语义检索）
- ✅ ColPali 索引（视觉检索）

**检索能力**
- ✅ 单一检索：BM25/Dense/ColPali
- ✅ 混合检索：自定义组合 + 权重调节
- ✅ 融合策略：weighted_sum / RRF

**Web UI**
- ✅ 文档管理：上传、列表、删除
- ✅ 索引构建：一键式操作
- ✅ 问答界面：查询、证据展示、答案生成
- ✅ 批量评估：CSV/JSON 数据集支持

---

## 技术栈总览

### 核心组件

| 层级 | 组件 | 技术 | GPU |
|-----|-----|------|-----|
| **前端** | Web UI | Gradio 6.2.0 | - |
| **OCR** | 文档识别 | HunyuanOCR | GPU0 |
| **检索** | 关键词 | BM25 (Rank-BM25) | - |
| | 语义 | Qwen3-Embedding-0.6B + FAISS | GPU1 |
| | 多模态 | Qwen3-VL-Embedding-2B + FAISS | GPU2 |
| | 视觉 | ColPali (ColQwen3-Embed-4B) | GPU2 |
| **生成** | MLLM | Qwen3-VL-4B-Instruct (vLLM) | GPU3 |

### GPU 资源分配

```
GPU0: HunyuanOCR (Port 8000)      │  ← 文档 OCR 识别
GPU1: Qwen3-Embedding (Port 8001) │  ← 语义向量编码
GPU2: ColPali / Dense-VL (延迟)   │  ← 视觉/多模态检索（共享）
GPU3: Qwen3-VL-4B (Port 8002)     │  ← 答案生成
```

---

## 路线图

### V1.3 - 体验优化（规划中）

- [ ] **Dense-VL API 模式** - vLLM/SGLang 在线服务
- [ ] **多轮对话** - 上下文记忆与引用追踪
- [ ] **Reranker** - 二阶段精排
- [ ] **缓存优化** - 向量/结果缓存加速

### V1.4 - 评估增强（规划中）

- [ ] **三层评估** - 检索/证据/生成分层指标
- [ ] **对比实验** - A/B 测试框架
- [ ] **可视化分析** - 性能仪表盘

---

## 快速开始

### 一键启动

```bash
# 1. 启动所有后端服务
bash scripts/start_services.sh

# 2. 启动 Web UI
bash scripts/start_ui.sh

# 3. 访问界面
# http://localhost:7860
```

### 命令行构建索引

```bash
# 构建所有索引
python scripts/build_indices_v1.py --all

# 只构建 ColPali 索引
python scripts/build_indices_v1.py --colpali

# 只构建 Dense-VL 索引
python scripts/build_indices_v1.py --dense-vl
```

---

## 文档索引

- **README.md**：项目概述与快速开始
- **VERSION.md**（本文档）：版本历史与路线图
- **CHANGELOG.md**：详细变更日志

---

**最后更新**：2026-01-23
**当前版本**：V1.2.1 - ColPali 多进程优化
**下一里程碑**：V1.3 - 体验优化

[🏠 首页](README.md) • [📜 详细日志](CHANGELOG.md) • [🐛 报告问题](https://github.com/your-org/doc-rag-evidence/issues)
