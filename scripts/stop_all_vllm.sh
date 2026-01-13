#!/bin/bash
# 停止所有vllm服务

echo "======================================================"
echo "Stopping All vLLM Services"
echo "======================================================"

# 方法1: 通过端口号精确停止服务
echo "Stopping services by port..."

# 停止端口 8000 (OCR)
OCR_PID=$(lsof -ti:8000 2>/dev/null)
if [ -n "$OCR_PID" ]; then
    echo "Stopping OCR service (port 8000, PID: $OCR_PID)..."
    kill -9 $OCR_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
else
    echo "OCR service (port 8000): Not running"
fi

# 停止端口 8001 (Embedding)
EMBED_PID=$(lsof -ti:8001 2>/dev/null)
if [ -n "$EMBED_PID" ]; then
    echo "Stopping Embedding service (port 8001, PID: $EMBED_PID)..."
    kill -9 $EMBED_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
else
    echo "Embedding service (port 8001): Not running"
fi

# 停止端口 8002 (Generation)
GEN_PID=$(lsof -ti:8002 2>/dev/null)
if [ -n "$GEN_PID" ]; then
    echo "Stopping Generation service (port 8002, PID: $GEN_PID)..."
    kill -9 $GEN_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
else
    echo "Generation service (port 8002): Not running"
fi

# 方法2: 强制停止所有 vllm 相关进程
echo ""
echo "Checking for remaining vllm processes..."
VLLM_PIDS=$(ps aux | grep -E "vllm.entrypoints.openai.api_server|VLLM::EngineCore" | grep -v grep | awk '{print $2}')
if [ -n "$VLLM_PIDS" ]; then
    echo "Found remaining vLLM processes: $VLLM_PIDS"
    kill -9 $VLLM_PIDS 2>/dev/null
    echo "  ✅ Killed all remaining vLLM processes"
else
    echo "  ℹ️  No remaining vLLM processes"
fi

# 清理资源跟踪器进程
echo ""
echo "Cleaning up resource tracker processes..."
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null && echo "  ✅ Cleaned up" || echo "  ℹ️  No tracker processes found"

# 清理 PID 文件
rm -f logs/vllm_*.pid 2>/dev/null

echo ""
echo "All vLLM services stopped."
echo "======================================================"
