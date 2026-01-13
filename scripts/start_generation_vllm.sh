#!/bin/bash
# Start vLLM server for Qwen3-VL-4B-Instruct generation (multimodal)
# This service handles answer generation with citation

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

MODEL_PATH="/workspace/cache/Qwen3-VL-4B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "   Please download the model first"
    exit 1
fi

echo "Starting vLLM generation server..."
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Port: 8002"
echo "GPU: GPU 3 (CUDA_VISIBLE_DEVICES=3) - dedicated"

# Start vLLM server on GPU 3 (completely free)
export CUDA_VISIBLE_DEVICES=3
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8002 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code

# Note: This will share GPU with embedding server if on same GPU
# For production, consider dedicated GPU or adjust memory settings
