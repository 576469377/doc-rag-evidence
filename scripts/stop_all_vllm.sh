#!/bin/bash
# 停止所有vllm服务

# 从 app.yaml 读取配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_LOADER="$SCRIPT_DIR/config_loader.py"

OCR_PORT=$(python "$CONFIG_LOADER" ocr.endpoint 2>/dev/null | sed 's|.*:||')
EMB_PORT=$(python "$CONFIG_LOADER" dense.endpoint 2>/dev/null | sed 's|.*:||')
GEN_PORT=$(python "$CONFIG_LOADER" llm.endpoint 2>/dev/null | sed 's|.*:||')

OCR_MODEL=$(python "$CONFIG_LOADER" ocr.model 2>/dev/null | sed 's|.*/||')
EMB_MODEL=$(python "$CONFIG_LOADER" dense.model 2>/dev/null | sed 's|.*/||')
GEN_MODEL=$(python "$CONFIG_LOADER" llm.model 2>/dev/null | sed 's|.*/||')

echo "======================================================"
echo "Stopping All vLLM Services"
echo "======================================================"
echo "配置端口: OCR=$OCR_PORT, Embedding=$EMB_PORT, Generation=$GEN_PORT"
echo ""

# 方法1: 通过端口号精确停止服务
echo "Stopping services by port..."

# 停止 OCR 服务
if [ -n "$OCR_PORT" ]; then
    OCR_PID=$(lsof -ti:$OCR_PORT 2>/dev/null)
    if [ -n "$OCR_PID" ]; then
        echo "Stopping OCR service ($OCR_MODEL, port $OCR_PORT, PID: $OCR_PID)..."
        kill -9 $OCR_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
    else
        echo "OCR service ($OCR_MODEL, port $OCR_PORT): Not running"
    fi
fi

# 停止 Embedding 服务
if [ -n "$EMB_PORT" ]; then
    EMBED_PID=$(lsof -ti:$EMB_PORT 2>/dev/null)
    if [ -n "$EMBED_PID" ]; then
        echo "Stopping Embedding service ($EMB_MODEL, port $EMB_PORT, PID: $EMBED_PID)..."
        kill -9 $EMBED_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
    else
        echo "Embedding service ($EMB_MODEL, port $EMB_PORT): Not running"
    fi
fi

# 停止 Generation 服务
if [ -n "$GEN_PORT" ]; then
    GEN_PID=$(lsof -ti:$GEN_PORT 2>/dev/null)
    if [ -n "$GEN_PID" ]; then
        echo "Stopping Generation service ($GEN_MODEL, port $GEN_PORT, PID: $GEN_PID)..."
        kill -9 $GEN_PID 2>/dev/null && echo "  ✅ Stopped" || echo "  ⚠️  Already stopped"
    else
        echo "Generation service ($GEN_MODEL, port $GEN_PORT): Not running"
    fi
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
