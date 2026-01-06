#!/bin/bash
# 一键启动所有vllm服务

echo "======================================================"
echo "Starting All vLLM Services"
echo "======================================================"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "Warning: Less than 2 GPUs available. Some services may fail."
fi

# 启动HunyuanOCR (GPU 0, Port 8000)
echo ""
echo "Starting HunyuanOCR on GPU 0..."
export CUDA_VISIBLE_DEVICES=0
nohup vllm serve /workspace/cache/HunyuanOCR \
    --port 8000 \
    --host 0.0.0.0 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    > logs/vllm_ocr.log 2>&1 &
OCR_PID=$!
echo "HunyuanOCR started (PID: $OCR_PID)"

# 等待OCR服务启动
echo "Waiting for OCR service to be ready..."
echo "This may take 2-3 minutes for first-time model loading..."
sleep 30

# 健康检查
echo "Checking OCR service health..."
for i in {1..6}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ OCR service is ready"
        break
    fi
    echo "Waiting... ($i/6)"
    sleep 20
done

# 启动Qwen3-Embedding (GPU 1, Port 8001)
echo ""
echo "Starting Qwen3-Embedding on GPU 1..."
export CUDA_VISIBLE_DEVICES=1
nohup vllm serve /workspace/cache/Qwen3-Embedding-0.6B \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code \
    > logs/vllm_embedding.log 2>&1 &
EMBED_PID=$!
echo "Qwen3-Embedding started (PID: $EMBED_PID)"

echo ""
echo "======================================================"
echo "All services started!"
echo "======================================================"
echo "HunyuanOCR:      http://localhost:8000  (GPU 0, PID: $OCR_PID)"
echo "Qwen3-Embedding: http://localhost:8001  (GPU 1, PID: $EMBED_PID)"
echo ""
echo "Logs:"
echo "  OCR:       tail -f logs/vllm_ocr.log"
echo "  Embedding: tail -f logs/vllm_embedding.log"
echo ""
echo "To stop services:"
echo "  kill $OCR_PID $EMBED_PID"
echo "  or use: pkill -f vllm"
echo "======================================================"

# 保存PIDs到文件
mkdir -p logs
echo "$OCR_PID" > logs/vllm_ocr.pid
echo "$EMBED_PID" > logs/vllm_embedding.pid

echo ""
echo "Now you can start the UI:"
echo "  python app/ui/main_v1.py"
