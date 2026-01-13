# Doc RAG Evidence - 快速启动指南

## 🚀 两步启动

### 1️⃣ 启动后台服务

```bash
bash scripts/start_services.sh
```

这会自动：
- 激活conda环境 `doc-rag-evidence-vllm`
- 检查并启动Generation服务（Qwen3-VL-4B，端口8002）
- 等待模型加载完成（约30-60秒）

### 2️⃣ 启动UI界面

```bash
bash scripts/start_ui.sh
```

这会自动：
- 激活conda环境
- 检查服务状态
- 启动Gradio UI（端口7860）

**访问地址**: http://localhost:7860

---

## 📋 服务说明

| 服务 | 端口 | 必需 | 说明 |
|------|------|------|------|
| **Generation** | 8002 | ✅ | Qwen3-VL LLM生成，必需 |
| Embedding | 8001 | 可选 | Dense检索需要 |
| OCR | 8000 | 可选 | PDF导入需要 |

**核心功能**只需要Generation服务，支持：
- BM25检索（本地，无需额外服务）
- ColPali检索（GPU 2，无需额外服务）
- LLM答案生成（Generation服务）

---

## 🔧 故障排查

### Generation服务启动失败

查看日志：
```bash
tail -100 logs/generation_vllm.log
```

常见问题：
- **GPU显存不足**: `nvidia-smi` 检查GPU 3
- **端口占用**: `lsof -i :8002` 查看占用进程
- **进程卡死**: `pkill -f "vllm.*8002"` 然后重启

### UI无法访问

1. 检查服务是否运行：
   ```bash
   curl http://localhost:8002/v1/models
   ```

2. 确认UI进程正常：
   ```bash
   ps aux | grep main_v1.py
   ```

---

## 🎯 使用示例

启动后在UI中：

1. **选择检索模式**：
   - BM25：最快，无需额外服务
   - Dense：需要Embedding服务
   - ColPali：视觉检索，GPU 2本地模型

2. **输入问题**：
   ```
   磷酸氢钙的原料有哪些？
   ```

3. **查看结果**：
   - 检索到的证据
   - LLM生成的答案
   - 引用标注 [1][2]...

---

## ⚙️ 配置调整

### 启用LLM生成

编辑 `configs/app.yaml`：

```yaml
generator:
  type: "qwen3_vl"  # 启用真实LLM生成
  # type: "template"  # 简单模板拼接
```

### 调整GPU分配

编辑 `scripts/start_generation_vllm.sh`：

```bash
export CUDA_VISIBLE_DEVICES=3  # 使用GPU 3
```

---

## 📊 系统架构

```
用户请求
    ↓
检索器 (BM25/Dense/ColPali)
    ↓
证据规范化 (Page→Block)
    ↓
证据选择 (Top-K)
    ↓
LLM生成 (Qwen3-VL) ← Generation服务 (端口8002)
    ↓
答案+引用
```

---

## 🛑 停止服务

### 停止UI
按 `Ctrl+C` 在UI终端

### 停止Generation服务
```bash
pkill -f "vllm.*8002"
```

### 停止所有vLLM服务
```bash
pkill -f vllm
```
