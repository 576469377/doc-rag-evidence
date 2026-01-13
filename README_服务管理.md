# Doc RAG Evidence - 服务管理

## 🚀 启动服务

### 启动所有服务（推荐）

```bash
bash scripts/start_services.sh
# 或
bash scripts/start_services.sh all
```

启动：OCR + Embedding + Generation

### 仅启动特定服务

```bash
# 仅启动Generation服务（必需，用于LLM生成）
bash scripts/start_services.sh generation

# 仅启动Embedding服务（用于Dense检索）
bash scripts/start_services.sh embedding

# 仅启动OCR服务（用于PDF导入）
bash scripts/start_services.sh ocr
```

---

## 📊 服务说明

| 服务 | 端口 | GPU | 用途 | 脚本 |
|------|------|-----|------|------|
| OCR | 8000 | GPU 0 | PDF文字识别 | `start_ocr_vllm.sh` |
| Embedding | 8001 | GPU 1 | Dense检索 | `start_embedding_vllm.sh` |
| **Generation** | 8002 | GPU 3 | **LLM答案生成（必需）** | `start_generation_vllm.sh` |

### 功能依赖

- **BM25检索**: 无需额外服务
- **Dense检索**: 需要Embedding服务
- **ColPali检索**: GPU 2本地加载，无需额外服务
- **LLM生成**: 需要Generation服务（必需）
- **PDF导入**: 需要OCR服务

---

## 🔄 重启逻辑

启动脚本会自动：
1. 检测旧进程
2. 停止旧服务
3. 等待3秒
4. 启动新服务
5. 验证服务响应

---

## 🛑 停止服务

### 停止所有vLLM服务

```bash
pkill -f vllm
```

### 停止特定服务

```bash
# 停止OCR
pkill -f "vllm.*8000"

# 停止Embedding
pkill -f "vllm.*8001"

# 停止Generation
pkill -f "vllm.*8002"
```

---

## 📝 日志查看

```bash
# OCR日志
tail -f logs/ocr_vllm.log

# Embedding日志
tail -f logs/embedding_vllm.log

# Generation日志
tail -f logs/generation_vllm.log
```

---

## 🔍 检查服务状态

```bash
# 检查所有服务
curl http://localhost:8000/v1/models  # OCR
curl http://localhost:8001/v1/models  # Embedding
curl http://localhost:8002/v1/models  # Generation

# 检查进程
ps aux | grep vllm | grep -v grep
```

---

## 💡 使用建议

### 快速开发（最小配置）

仅启动必需服务：

```bash
bash scripts/start_services.sh generation
bash scripts/start_ui.sh
```

功能：BM25检索 + ColPali检索 + LLM生成

### 完整功能

启动所有服务：

```bash
bash scripts/start_services.sh all
bash scripts/start_ui.sh
```

功能：所有检索模式 + LLM生成 + PDF导入

---

## ⚠️ 故障排查

### GPU显存不足

```bash
# 查看GPU使用情况
nvidia-smi

# 按需启动服务，避免同时加载过多模型
bash scripts/start_services.sh generation  # 仅启动Generation
```

### 服务启动失败

1. 查看日志：`tail -100 logs/*_vllm.log`
2. 检查端口占用：`lsof -i :8000` / `lsof -i :8001` / `lsof -i :8002`
3. 检查模型路径：确认 `/workspace/cache/` 下模型存在
4. 重启服务：先停止再启动

### ColPali模型

ColPali不是独立服务，由UI启动时自动加载到GPU 2。
