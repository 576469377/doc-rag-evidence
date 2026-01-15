#!/bin/bash
# Start vLLM server for Qwen3-VL-4B-Instruct generation (multimodal)
# This service handles answer generation with citation

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# 从 app.yaml 读取配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_LOADER="$SCRIPT_DIR/config_loader.py"

MODEL_PATH=$(python "$CONFIG_LOADER" llm.model_path)
MODEL_NAME=$(python "$CONFIG_LOADER" llm.model)
PORT=$(python "$CONFIG_LOADER" llm.endpoint | sed 's|.*:||')
GPU=$(python "$CONFIG_LOADER" llm.gpu)

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "   Please download the model first"
    exit 1
fi

echo "Starting vLLM generation server..."
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
    --gpu-memory-utilization 0.80 \
    --trust-remote-code

# Note: This will share GPU with embedding server if on same GPU
# For production, consider dedicated GPU or adjust memory settings
