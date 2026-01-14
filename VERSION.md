# Doc RAG Evidence - 版本状态与路线图

## 当前版本：V1.0 - 基础链路完成

**完成时间**：2026-01-14  
**状态**：✅ 核心链路已跑通，进入下一阶段优化

---

## 当前已实现功能

### 1. 文档处理与索引构建 ✅

#### 文档摄取
- **PDF 解析**：基于 HunyuanOCR 的高质量 OCR，支持扫描版和文字版 PDF
- **块级切分**：page → blocks 粒度，支持文本块、表格和图像
- **元数据管理**：doc_id / page_id / block_id 三级索引体系
- **工件存储**：page.png、blocks.json、metadata.json 完整保存

#### 三路索引构建
```
✅ BM25 索引：基于 block text 的关键词检索
✅ Dense 索引：基于 FAISS 的语义向量检索（Qwen3-Embedding-0.6B）
✅ ColPali 索引：基于视觉理解的页面级检索
```

**当前状态**：单个 PDF 导入、索引构建流程完整可用

### 2. 检索能力 ✅

#### 三种检索模式
- **BM25 Retriever**：block 级精确匹配
  - ✅ 返回 block hits with text snippet
  - ✅ 适用于关键词查询

- **Dense Retriever**：block 级语义检索
  - ✅ 返回 block hits with text snippet
  - ✅ 适用于概念性查询

- **ColPali Retriever**：page 级视觉检索
  - ✅ 返回 page hits with score
  - ⚠️ **问题**：snippet 为空，未落到 block 证据

#### Hybrid 融合检索（新增）
- **灵活组合**：支持任意两个检索器的自定义组合
  - BM25 + Dense
  - BM25 + ColPali
  - Dense + ColPali
  
- **融合方法**：
  - weighted_sum：加权分数融合
  - rrf：倒数排名融合
  
- **权重控制**：UI 实时调整检索器权重

**当前状态**：检索能力完备，但 ColPali 证据不可读

### 3. 证据处理与生成 ⚠️

#### 证据选择
- ✅ TopK Evidence Selector：从检索结果中选择 top_k_evidence 条
- ⚠️ **问题**：ColPali 的 page hits 未展开为 block evidence

#### 答案生成
- ✅ Qwen3-VL-4B-Instruct：真实 LLM 生成（vLLM backend，GPU3）
- ✅ Template Generator：模板生成（fallback）
- ✅ 配置切换：configs/app.yaml 中 generator.type 控制

**当前输出**：
```
BM25/Dense 模式（LLM生成）：
  ✅ Evidence: block-level snippet 可读
  ✅ Answer: LLM 基于证据生成的答案
  ✅ Citations: [1][2] 引用支持

ColPali 模式：
  ❌ Evidence: snippet 为空（需要 page→block expansion）
  ⚠️ Answer: LLM 只能引用序号，无法展示原文
```

### 4. UI 与评估 ✅

#### Gradio Web UI
- **Document Management**：上传 PDF、查看文档列表、删除文档
- **Index Building**：一键构建 BM25/Dense/ColPali 索引
- **Query & Answer**：
  - 选择检索模式（BM25/Dense/ColPali/Hybrid）
  - 自定义 Hybrid 配置（检索器组合、融合方法、权重）
  - 证据展示（snippet、source、score）
  - 答案生成

- **Batch Evaluation**：
  - 上传评估数据集（CSV/JSON）
  - 支持 Hybrid 配置
  - 输出评估报告（predictions.csv、report.json）

#### 评估能力
- ✅ 批量问答评估
- ✅ 成功率、延迟等基础指标
- ❌ **缺失**：检索层/证据层/生成层的细分指标

### 5. 工程化 ✅

#### 服务化部署
- **OCR 服务**：HunyuanOCR vLLM (GPU0, port 8000)
- **Embedding 服务**：Qwen3-Embedding-0.6B vLLM (GPU1, port 8001)
- **Generation 服务**：Qwen3-VL-4B-Instruct vLLM (GPU3, port 8002)
- **ColPali**：延迟加载机制 (GPU2)

#### 配置管理
- `configs/app.yaml`：统一配置文件
- `AppConfig` Pydantic schema：类型安全

#### 日志与追踪
- `RunLogger`：每次查询的完整记录（问题、检索结果、证据、答案）
- `data/runs/` 持久化存储

---

## 核心问题与待解决

### 🔴 P0 - 阻塞性问题

#### 1. ColPali 证据不可用
**现象**：
- ColPali 检索返回 page hits，但 Evidence 表 snippet 为空
- 答案只有引用编号 [1][2]...，无法阅读

**根本原因**：
- ColPali Retriever 输出 page-level RetrieveHit
- 未做 page → block 的 expansion
- EvidenceSelector 无法获取 block text

**影响**：
- ColPali 模式完全不可用（证据不可读）
- Hybrid 模式中 ColPali 分支无贡献

### � P1 - 架构不统一

#### 2. 检索返回单元不统一
**现象**：
- BM25/Dense 返回 block hits
- ColPali 返回 page hits
- Pipeline 下游需要各自处理

**根本原因**：
- 缺少统一的 normalize_hits() 步骤
- EvidenceSelector 无法统一处理

**影响**：
- 代码重复、维护困难
- Hybrid 融合复杂度高

### 🟡 P1 - 工程体验

#### 3. 单文件导入限制
**现象**：一次只能上传一个 PDF

**影响**：
- 评估、演示时需要反复操作
- 软著截图不美观

---

## 下一阶段计划

### V1.1：证据统一与可读化（优先级最高）

**目标**：无论 BM25/Dense/ColPali，最终进入生成器的证据都变成统一的 block 证据（有 text/snippet，可引用），生成高质量答案。

#### T1 - ColPali 证据落块（最高优先级）⭐⭐⭐
**目标**：ColPali 的 page hits 展开为 block evidence，补齐 snippet

**改动**：
1. 新增 `expand_page_to_blocks()` 函数
   - 输入：page hits
   - 从 `blocks.json` 加载该页的所有 blocks
   - 页内轻量排序（BM25-in-page / Dense-in-page / 启发式）
   - 输出：block hits with snippet

2. 在 ColPali Retriever 或 Pipeline 中调用 expansion

**产物**：
- ColPali 模式下 Evidence 表 snippet 不为空
- Answer 包含可读文本（不只是引用编号）

**验收标准**：
- 用 ColPali 模式提问："食品中铅的限量是多少？"
- Evidence 表显示具体文本 snippet
- Answer 输出："根据 GB 2762 标准，食品中铅的限量为..."（LLM 真实生成）

#### T2 - 统一检索返回契约 ⭐⭐⭐
**目标**：三种检索模式最终都返回 block hits

**改动**：
1. 在 `core/pipeline.py` 增加 `normalize_hits()` 阶段：
   ```python
   retrieve → normalize_hits → select_evidence → generate
   ```

2. 规范：
   - BM25/Dense：已是 block hits，直接通过
   - ColPali：调用 expansion 转为 block hits

**产物**：
- Pipeline 逻辑清晰、可追踪
- 三种模式的 evidence schema 完全一致

**验收标准**：
- 查看 runlog，三种模式的 evidence 字段格式相同
- `doc_id/page_id/block_id/snippet/source/score` 都有值

#### T3 - 优化 Qwen3-VL-4B 生成质量 ⭐⭐
**目标**：改进 LLM 生成效果，强化引用控制

**改动**：
1. Prompt 工程优化
   - 强制引用格式（例如："根据证据[1]..."）
   - 防止幻觉（禁止编造不在证据中的内容）
   - 多轮对话支持（可选）

2. 参数调优
   - temperature / top_p 实验
   - max_new_tokens 控制
   - repetition_penalty 防止重复

**产物**：
- 答案质量提升，引用更准确
- 生成策略可配置

**验收标准**：
- 提问："碳酸氢钠是什么？"
- 答案："碳酸氢钠是一种白色结晶粉末，化学式为 NaHCO₃，常用作食品添加剂。[1][3]"
- 引用的 [1][3] 能对应到具体 evidence block
- 无证据外的信息

#### T4 - 证据上下文装配与去重 ⭐⭐
**目标**：规范化 context 构造，避免重复和超长

**改动**：
1. 新增 `impl/context_assembler.py`
   - 去重：高度重叠的 blocks 合并
   - 截断：按 max_context_chars 裁剪
   - 引用映射：稳定的 [1] → unit_id 对应

**产物**：
- context 在 token 预算内
- 引用编号稳定

**验收标准**：
- 同一 query_id 重跑，生成输入一致

#### T5 - 批量导入（工程体验）⭐
**目标**：支持一次上传多个 PDF

**改动**：
- UI：Gradio 支持多文件上传
- 后端：串行队列 + 进度反馈

**产物**：
- 一次上传 N 个 PDF，逐个处理

---

### V1.2：多模态生成 + 指标体系（后续）

#### T6 - 多模态生成（图文联合）
**目标**：把 page.png 送进 Qwen3-VL，支持图像辅助回答

**输入**：question + top_k pages 的 page.png + OCR blocks

**输出**：答案引用既可以引用 block，也可以引用 page

#### T7 - Hybrid 融合优化
**当前已有**：自定义检索器组合和权重

**下一步**：
- 加入 Reranker（可选）
- 优化分数归一化策略

#### T8 - 评估指标升级
**目标**：拆分三层指标

- **检索层**：Recall@K / MRR
- **证据层**：Evidence Hit Rate（引用是否落在正确页/块）
- **生成层**：EM / F1 / 关键词匹配

---

## 资源分配建议

### 当前 GPU 使用
```
GPU0: HunyuanOCR (OCR 服务)
GPU1: Qwen3-Embedding-0.6B (vLLM)
GPU2: ColPali (延迟加载)
GPU3: (未使用)
```

### V1.1 建议分配
```
GPU0: HunyuanOCR (保持)
GPU1: Qwen3-Embedding-0.6B + Qwen3-VL-4B 共享（embedding 负载轻）
GPU2: ColPali (保持)
GPU3: 预留（或单独部署 generation 服务更稳定）
```

**推荐方案**：
- **高稳定性**：GPU3 单独部署 Qwen3-VL-4B vLLM 服务
- **低资源**：GPU1 合并 embedding + generation

---

## 技术栈总结

### 核心框架
- **UI**：Gradio 6.2.0
- **检索**：BM25 (Rank-BM25) + FAISS + ColPali
- **Embedding**：Qwen3-Embedding-0.6B (vLLM)
- **OCR**：HunyuanOCR
- **生成**：Qwen3-VL-4B-Instruct（待接入）

### 数据模型
- **Pydantic V2**：类型安全的 schema 定义
- **存储**：本地文件系统（JSON + FAISS + 图片）

### 工程化
- **配置管理**：YAML + Pydantic
- **日志追踪**：RunLogger 持久化每次查询
- **服务化**：vLLM/SGLang OpenAI-compatible API

---

## 快速启动

### 启动服务
```bash
# 1. 启动 OCR 服务（GPU0）
bash scripts/start_ocr.sh

# 2. 启动 Embedding 服务（GPU1）
bash scripts/start_embedding.sh

# 3. 启动 UI
bash scripts/start_ui.sh
```

### 基本使用
1. 访问 `http://localhost:7860`
2. 上传 PDF → 构建索引（BM25 + Dense）
3. Query & Answer → 选择检索模式 → 提问
4. 查看 Evidence 和 Answer

### 当前限制
- ⚠️ ColPali 模式证据不可读（正在修复）
- ⚠️ Answer 质量低（模板生成，待接入 LLM）
- ⚠️ 一次只能上传一个 PDF

---

## 文档索引

- **README.md**：项目概述与快速开始
- **VERSION.md**（本文档）：版本状态与路线图
- **TROUBLESHOOTING.md**：常见问题排查

---

**更新时间**：2026-01-14  
**下一里程碑**：V1.1 - 证据统一与 LLM 生成（预计 1-2 周）
