#!/bin/bash
# 启动 HunyuanOCR 在 GPU 0

echo "======================================================"
echo "Starting HunyuanOCR on GPU 0"
echo "Model: tencent/HunyuanOCR"
echo "Port: 8000"
echo "GPU: 0"
echo "======================================================"

export CUDA_VISIBLE_DEVICES=0

# 使用本地模型路径
vllm serve /workspace/cache/HunyuanOCR \
    --served-model-name tencent/HunyuanOCR \
    --port 8000 \
    --host 0.0.0.0 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
