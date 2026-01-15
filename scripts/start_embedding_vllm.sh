#!/bin/bash
# Start vLLM server for Qwen3-Embedding (Dense retrieval)

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# 从 app.yaml 读取配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_LOADER="$SCRIPT_DIR/config_loader.py"

MODEL_PATH=$(python "$CONFIG_LOADER" dense.model_path)
MODEL_NAME=$(python "$CONFIG_LOADER" dense.model)
PORT=$(python "$CONFIG_LOADER" dense.endpoint | sed 's|.*:||')
GPU=$(python "$CONFIG_LOADER" dense.gpu)

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    exit 1
fi

echo "Starting vLLM embedding server..."
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: $GPU"

# Start vLLM server
export CUDA_VISIBLE_DEVICES=$GPU
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
