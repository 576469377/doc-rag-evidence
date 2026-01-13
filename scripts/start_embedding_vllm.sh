#!/bin/bash
# Start vLLM server for Qwen3-Embedding (Dense retrieval)

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

MODEL_PATH="/workspace/cache/Qwen3-Embedding-0.6B"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    exit 1
fi

echo "Starting vLLM embedding server..."
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Port: 8001"
echo "GPU: GPU 1"

# Start vLLM server on GPU 1
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
