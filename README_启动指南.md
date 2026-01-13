# Doc RAG Evidence - 启动指南

## 快速启动

### 1. 激活conda环境

```bash
source /workspace/program/miniconda3/etc/profile.d/conda.sh
conda activate doc-rag-evidence-vllm
cd /workspace/doc-rag-evidence
```

### 2. 启动后台服务

```bash
bash scripts/start_services.sh
```

这个脚本会：
- 检查OCR服务 (端口8000) - 可选
- 检查Embedding服务 (端口8001) - Dense检索需要
- **自动启动Generation服务 (端口8002, Qwen3-VL)** - 必需

### 3. 启动UI界面

```bash
bash scripts/start_ui.sh
# 或者直接运行
python app/ui/main_v1.py
```

访问地址：**http://0.0.0.0:7860** 或 **http://localhost:7860**

---

## 详细说明

### 服务端口分配

| 服务 | 端口 | GPU | 用途 | 状态 |
|------|------|-----|------|------|
| HunyuanOCR | 8000 | GPU 0 | OCR文字识别 | 可选 |
| Qwen3-Embedding | 8001 | GPU 1 | Dense检索 | Dense模式必需 |
| **Qwen3-VL-4B** | 8002 | GPU 3 | LLM答案生成 | **必需** |
| Gradio UI | 7860 | - | Web界面 | - |

### 检索模式

1. **BM25** - 无需额外服务，使用本地索引
2. **Dense** - 需要Embedding服务 (端口8001)
3. **ColPali** - 使用GPU 2本地模型，无需额外服务

### 生成器模式

在 `configs/app.yaml` 中配置：

```yaml
generator:
  type: "template"   # 简单模板拼接（快速测试）
  # type: "qwen3_vl"   # 使用LLM生成（真实答案）
```

要使用LLM生成，修改为 `type: "qwen3_vl"`

---

## 手动启动（高级）

### 单独启动Generation服务

```bash
bash scripts/start_generation_vllm.sh > logs/generation_vllm.log 2>&1 &
```

等待30-40秒模型加载完成，然后验证：

```bash
curl -s http://localhost:8002/v1/models | python -m json.tool
```

### 直接启动UI

```bash
python app/ui/main_v1.py
```

---

## 故障排查

### Generation服务启动失败

1. **检查日志**：
   ```bash
   tail -100 logs/generation_vllm.log
   ```

2. **常见问题**：
   - GPU显存不足：检查 `nvidia-smi`，确认GPU 3有足够空闲显存
   - 端口被占用：`lsof -i :8002`，杀掉占用进程
   - 模型路径错误：确认 `/workspace/cache/Qwen3-VL-4B-Instruct` 存在

3. **重启服务**：
   ```bash
   pkill -f "vllm.*8002"
   sleep 3
   bash scripts/start_generation_vllm.sh > logs/generation_vllm.log 2>&1 &
   ```

### UI连接服务失败

检查所有服务是否运行：

```bash
curl http://localhost:8000/v1/models  # OCR
curl http://localhost:8001/v1/models  # Embedding
curl http://localhost:8002/v1/models  # Generation (必需)
```

---

## 性能优化建议

1. **首次启动**：Generation服务加载模型需要30-60秒
2. **推理速度**：
   - BM25: 最快 (~100ms)
   - Dense: 需要embedding计算 (~500ms)
   - ColPali: 视觉检索较慢 (~2-3s)
3. **LLM生成**：Qwen3-VL生成答案约400-1000ms

---

## 测试验证

### 端到端测试

```bash
python test_e2e_generation.py
```

会测试三种检索模式 + LLM生成。

### 单独测试生成器

```bash
python test_generator_simple.py
```

---

## 功能特性

✅ **V1.1 完成**
- 三种检索模式：BM25 / Dense / ColPali
- ColPali页面→块级证据扩展
- 统一的块级证据契约
- Qwen3-VL-4B真实LLM生成
- 引用提取和验证

🚀 **V1.2 规划**
- 证据去重和上下文组装
- 批量PDF导入
- 多模态生成（图片+文本）
- 混合检索融合
