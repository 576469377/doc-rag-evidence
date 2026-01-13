#!/bin/bash
# Start vLLM server for HunyuanOCR (PDF text extraction)

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

MODEL_PATH="/workspace/cache/HunyuanOCR"
MODEL_NAME="tencent/HunyuanOCR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    exit 1
fi

echo "Starting vLLM OCR server..."
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Port: 8000"
echo "GPU: GPU 0"

# Start vLLM server on GPU 0
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code
