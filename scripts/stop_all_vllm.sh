#!/bin/bash
# 停止所有vllm服务

echo "======================================================"
echo "Stopping All vLLM Services"
echo "======================================================"

# 从PID文件读取并停止
if [ -f logs/vllm_ocr.pid ]; then
    OCR_PID=$(cat logs/vllm_ocr.pid)
    echo "Stopping HunyuanOCR (PID: $OCR_PID)..."
    kill $OCR_PID 2>/dev/null && echo "  Stopped" || echo "  Not running"
    rm logs/vllm_ocr.pid
fi

if [ -f logs/vllm_embedding.pid ]; then
    EMBED_PID=$(cat logs/vllm_embedding.pid)
    echo "Stopping Qwen3-Embedding (PID: $EMBED_PID)..."
    kill $EMBED_PID 2>/dev/null && echo "  Stopped" || echo "  Not running"
    rm logs/vllm_embedding.pid
fi

# 备用方案：强制停止所有vllm进程
echo ""
echo "Checking for remaining vllm processes..."
pkill -f "vllm serve" && echo "  Killed remaining vllm processes" || echo "  No remaining processes"

echo ""
echo "All vLLM services stopped."
echo "======================================================"
