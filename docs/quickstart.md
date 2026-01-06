# Doc RAG Evidence System - 快速上手指南

## 第一步：安装环境

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 第二步：启动Web界面

```bash
python run.py
```

浏览器访问：http://127.0.0.1:7860

## 第三步：上传文档

1. 切换到"📄 Document Management"标签
2. 点击"Upload PDF"上传PDF文件
3. 点击"📤 Ingest & Index"按钮
4. 等待处理完成（显示"Upload complete!"）

## 第四步：提问

1. 切换到"🔍 Query & Answer"标签
2. 在"Your Question"框中输入问题，例如：
   - "What is the main topic of this document?"
   - "What are the key findings?"
   - "Who are the authors?"
3. 点击"🚀 Ask Question"
4. 查看结果：
   - **Answer**：生成的答案
   - **Evidence**：支持答案的证据列表（包含文档ID、页码、分数、文本片段）
   - **Query ID**：用于追溯运行日志

## 第五步：批量评测（可选）

1. 准备评测数据集（CSV格式）：
   ```csv
   qid,question,answer_gt
   q1,What is the main topic?,
   q2,What are the key findings?,
   ```

2. 切换到"📊 Evaluation"标签
3. 上传CSV文件
4. 点击"▶️ Run Evaluation"
5. 下载结果：
   - predictions.csv（每个问题的结果）
   - report.json（汇总指标）

---

## 命令行使用

### 构建/重建索引

```bash
python scripts/build_index.py
```

### 单次问答

```bash
python scripts/demo_run.py "What is the publication date?"
```

### 批量评测

```bash
python scripts/demo_eval.py data/sample_eval.csv
```

---

## 常见问题

### Q: 为什么问答没有结果？

A: 请确保：
1. 已上传文档
2. 已构建索引（上传文档时自动构建）
3. 问题与文档内容相关

### Q: 如何查看运行日志？

A: 每次问答后会生成日志文件：
- 位置：`data/runs/{query_id}.json`
- Query ID会在问答结果中显示

### Q: 支持哪些文档格式？

A: V0版本支持：
- ✅ PDF（文本型）
- ❌ 图片型PDF（需OCR，暂不支持）
- ❌ Word/PPT（暂不支持）

### Q: 答案准确性如何提升？

V0使用模板式生成（无需API），准确性有限。后续版本可以：
- 集成真实LLM（OpenAI、Anthropic等）
- 优化检索策略（向量检索、重排序）
- 增加BBox定位（精确到文本位置）

---

## 下一步

- 📖 阅读完整文档：[README.md](../README.md)
- 🔧 修改配置：[configs/app.yaml](../configs/app.yaml)
- 📊 查看数据结构：[core/schemas.py](../core/schemas.py)
- 🚀 扩展功能：实现自定义的Retriever/Generator
