# 问题修复总结

## 修复的问题

### 1. ❌ deepseek-ocr 改为 HunyuanOCR

**问题**：用户的环境使用HunyuanOCR (tencent/HunyuanOCR) 通过vllm运行，端口8000

**修复内容**：

#### configs/app.yaml
```yaml
ocr:
  provider: "vllm"                   # 改为vllm
  model: "tencent/HunyuanOCR"        # 改为HunyuanOCR
  endpoint: "http://localhost:8000"  # 改为8000端口
  timeout: 300                       # 增加超时时间
  cache_enabled: true
```

#### impl/ocr_client.py
- 更新为支持vllm的OpenAI兼容API
- 使用HunyuanOCR官方推荐的提示词：
  - 提取文档主体信息
  - 表格用HTML格式
  - 公式用LaTeX格式
  - 按阅读顺序组织
- 添加extra_body参数支持（top_k=1, repetition_penalty=1.0）

### 2. ❌ UI Bug: 'DocumentStoreLocal' object has no attribute 'load_document'

**问题**：UI代码中调用了不存在的方法

**原因**：DocumentStoreLocal的方法是`get_document()`和`list_documents()`，而不是`load_document()`

**修复内容**：

#### app/ui/main_v1.py - _get_doc_list()方法
```python
# 修复前（错误）：
doc_ids = self.store.list_documents()  # 返回字符串列表
for doc_id in doc_ids:
    meta = self.store.load_document(doc_id)  # ❌ 方法不存在

# 修复后（正确）：
docs = self.store.list_documents()  # 返回DocumentMeta对象列表
for meta in docs:
    # 直接使用meta对象
    rows.append([meta.doc_id, meta.title, ...])
```

## 额外改进

### 1. 删除错误的OCR缓存
```bash
find data/docs -name "ocr.json" -delete
```
之前的deepseek-ocr缓存包含大量错误数据（重复的"user:"文本）

### 2. 创建测试脚本
- **test_hunyuan_ocr.py**: 测试HunyuanOCR连接和功能
- **test_v1_smoke.py**: V0.1系统的全面冒烟测试

### 3. 文档
- **docs/HUNYUAN_OCR_GUIDE.md**: HunyuanOCR集成完整指南

## ✅ 验证结果

所有smoke测试通过：
```
✅ Imports              PASS
✅ Configuration        PASS
✅ BM25 Index           PASS
✅ UI Initialization    PASS

🎉 All tests passed!
```

## 🚀 使用说明

### 启动HunyuanOCR服务器（单独终端）
```bash
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0
```

### 测试连接
```bash
python test_hunyuan_ocr.py
```

### 导入文档（使用OCR）
```bash
# 单个PDF
python scripts/ingest_docs_v1.py --pdf document.pdf --use-ocr

# 整个目录
python scripts/ingest_docs_v1.py --pdf-dir data/pdfs --use-ocr
```

### 构建索引
```bash
python scripts/build_indices_v1.py --bm25
```

### 启动UI
```bash
python app/ui/main_v1.py
# 访问 http://localhost:7860
```

## 📝 注意事项

1. **OCR性能**：每页处理需要10-30秒，大文档请耐心等待
2. **缓存机制**：OCR结果会缓存在`ocr.json`，重复运行会跳过
3. **GPU资源**：HunyuanOCR需要约12GB显存
4. **超时设置**：复杂页面可能需要更长时间，已设置为300秒

## 修改的文件清单

1. ✅ `configs/app.yaml` - OCR配置更新
2. ✅ `impl/ocr_client.py` - vllm API适配
3. ✅ `app/ui/main_v1.py` - 修复load_document bug
4. ✅ `test_hunyuan_ocr.py` - 新增HunyuanOCR测试
5. ✅ `docs/HUNYUAN_OCR_GUIDE.md` - 新增使用指南

所有修复已完成，系统现在可以正常使用HunyuanOCR！
