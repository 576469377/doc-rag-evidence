#!/bin/bash
# 启动 Qwen3-Embedding 在 GPU 1

echo "======================================================"
echo "Starting Qwen3-Embedding on GPU 1"
echo "Model: Qwen/Qwen3-Embedding-0.6B"
echo "Port: 8001"
echo "GPU: 1"
echo "======================================================"

export CUDA_VISIBLE_DEVICES=1

# 使用本地模型路径
vllm serve /workspace/cache/Qwen3-Embedding-0.6B \
    --served-model-name Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code
