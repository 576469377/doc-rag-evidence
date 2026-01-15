# Scripts 使用说明

本目录下的脚本已经重构为从 `configs/app.yaml` 动态读取配置，不再硬编码路径和端口。

## 配置文件

所有服务配置均在 `configs/app.yaml` 中定义：

```yaml
# OCR 配置
ocr:
  model_path: "/workspace/cache/HunyuanOCR"
  model: "tencent/HunyuanOCR"
  endpoint: "http://localhost:8000"
  gpu: 0                            # GPU 设备 ID

# Embedding 配置
dense:
  model_path: "/workspace/cache/Qwen3-Embedding-0.6B"
  model: "Qwen/Qwen3-Embedding-0.6B"
  endpoint: "http://localhost:8001"
  gpu: 1                            # GPU 设备 ID

# Generation 配置
llm:
  model_path: "/workspace/cache/Qwen3-VL-4B-Instruct"
  model: "Qwen/Qwen3-VL-4B-Instruct"
  endpoint: "http://localhost:8002"
  gpu: 3                            # GPU 设备 ID
```

## 配置加载器

`config_loader.py` 是一个 Python 工具脚本，用于从 app.yaml 读取配置：

```bash
# 读取 OCR 端点
python scripts/config_loader.py ocr.endpoint
# 输出: http://localhost:8000

# 读取模型路径
python scripts/config_loader.py llm.model_path
# 输出: /workspace/cache/Qwen3-VL-4B-Instruct
```

## 启动脚本

### 1. 启动所有服务

```bash
bash scripts/start_services.sh
```

自动从 app.yaml 读取配置并启动：
- OCR 服务（端口由 `ocr.endpoint` 决定）
- Embedding 服务（端口由 `dense.endpoint` 决定）
- Generation 服务（端口由 `llm.endpoint` 决定）

### 2. 启动单个服务

```bash
# 仅启动 Generation
bash scripts/start_services.sh generation

# 仅启动 OCR
bash scripts/start_services.sh ocr

# 仅启动 Embedding
bash scripts/start_services.sh embedding
```

### 3. 启动 UI

```bash
bash scripts/start_ui.sh
```

UI 会自动检测各服务端口（从 app.yaml 读取）。

### 4. 停止所有服务

```bash
bash scripts/stop_all_vllm.sh
```

## 跨机器部署

现在你可以轻松在不同机器上部署：

1. **修改 configs/app.yaml**：
   ```yaml
   # 更改模型路径
   llm:
     model_path: "/data/models/Qwen3-VL-4B-Instruct"
   
   # 更改端口
   ocr:
     endpoint: "http://localhost:9000"
   
   # 更改 GPU 分配
   ocr:
     gpu: 2  # 使用 GPU 2 而不是 GPU 0
   llm:
     gpu: 0  # 使用 GPU 0 而不是 GPU 3
   ```

2. **运行脚本**（无需修改）：
   ```bash
   bash scripts/start_services.sh
   ```

3. **脚本自动使用新配置**

## 注意事项

- ✅ 所有端口、模型路径、GPU分配从 app.yaml 读取
- ✅ 更换机器只需修改 app.yaml
- ✅ 脚本代码无需修改
- ✅ 支持灵活的GPU分配（可以将服务分配到不同GPU）
- ⚠️  确保 Python 环境已安装 PyYAML：`pip install pyyaml`
