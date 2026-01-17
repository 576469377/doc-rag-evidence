#!/bin/bash
# 启动所有后台服务（OCR、Embedding、Generation）
# 使用方法: bash scripts/start_services.sh [all|generation|embedding|ocr]
# 默认启动所有服务

set -e

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# 从 app.yaml 读取端口配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_LOADER="$SCRIPT_DIR/config_loader.py"

OCR_PORT=$(python "$CONFIG_LOADER" ocr.endpoint | sed 's|.*:||')
EMB_PORT=$(python "$CONFIG_LOADER" dense.endpoint | sed 's|.*:||')
GEN_PORT=$(python "$CONFIG_LOADER" llm.endpoint | sed 's|.*:||')

OCR_GPU=$(python "$CONFIG_LOADER" ocr.gpu)
EMB_GPU=$(python "$CONFIG_LOADER" dense.gpu)
GEN_GPU=$(python "$CONFIG_LOADER" llm.gpu)

OCR_MODEL=$(python "$CONFIG_LOADER" ocr.model | sed 's|.*/||')
EMB_MODEL=$(python "$CONFIG_LOADER" dense.model | sed 's|.*/||')
GEN_MODEL=$(python "$CONFIG_LOADER" llm.model | sed 's|.*/||')

# 解析参数
MODE="${1:-all}"  # 默认启动所有服务

echo "🚀 启动 Doc RAG Evidence 后台服务"
echo "================================"
echo "模式: $MODE"
echo "端口配置: OCR=$OCR_PORT, Dense=$EMB_PORT, Dense-VL=$VL_EMB_PORT, Generation=$GEN_PORT"

# 激活conda环境
source /workspace/program/miniconda3/etc/profile.d/conda.sh
conda activate doc-rag-evidence-vllm
cd /workspace/doc-rag-evidence

# 创建日志目录
mkdir -p logs

# ========== OCR服务 ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "ocr" ]; then
    echo ""
    echo "🛑 停止旧的OCR服务 (端口$OCR_PORT)..."
    OLD_PIDS=$(ps aux | grep -E "vllm.*$OCR_PORT" | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_PIDS" ]; then
        echo "   发现运行中的进程: $OLD_PIDS"
        pkill -f "vllm.*$OCR_PORT" 2>/dev/null || true
        sleep 3
        echo "   ✅ 已停止旧OCR服务"
    else
        echo "   ℹ️  无运行中的OCR服务"
    fi
    
    echo ""
    echo "⏳ 启动OCR服务..."
    echo "   模型: $OCR_MODEL"
    echo "   GPU: $OCR_GPU"
    echo "   端口: $OCR_PORT"
    echo "   日志: logs/ocr_vllm.log"
    
    nohup bash scripts/start_ocr_vllm.sh > logs/ocr_vllm.log 2>&1 &
    OCR_PID=$!
    echo "   进程PID: $OCR_PID"
    echo "   等待启动..."
    
    for i in {1..40}; do
        sleep 2
        if curl -s --max-time 3 http://localhost:$OCR_PORT/v1/models >/dev/null 2>&1; then
            echo "   ✅ OCR服务启动成功！"
            break
        fi
        [ $((i % 5)) -eq 0 ] && echo -n " ${i}s" || echo -n "."
    done
fi

# ========== Embedding服务 ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "embedding" ]; then
    echo ""
    echo "🛑 停止旧的Embedding服务 (端口$EMB_PORT)..."
    OLD_PIDS=$(ps aux | grep -E "vllm.*$EMB_PORT" | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_PIDS" ]; then
        echo "   发现运行中的进程: $OLD_PIDS"
        pkill -f "vllm.*$EMB_PORT" 2>/dev/null || true
        sleep 3
        echo "   ✅ 已停止旧Embedding服务"
    else
        echo "   ℹ️  无运行中的Embedding服务"
    fi
    
    echo ""
    echo "⏳ 启动Embedding服务..."
    echo "   模型: $EMB_MODEL"
    echo "   GPU: $EMB_GPU"
    echo "   端口: $EMB_PORT"
    echo "   日志: logs/embedding_vllm.log"
    
    nohup bash scripts/start_embedding_vllm.sh > logs/embedding_vllm.log 2>&1 &
    EMB_PID=$!
    echo "   进程PID: $EMB_PID"
    echo "   等待启动..."
    
    for i in {1..40}; do
        sleep 2
        if curl -s --max-time 3 http://localhost:$EMB_PORT/v1/models >/dev/null 2>&1; then
            echo "   ✅ Embedding服务启动成功！"
            break
        fi
        [ $((i % 5)) -eq 0 ] && echo -n " ${i}s" || echo -n "."
    done
fi

# ========== Generation服务 ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "generation" ]; then

echo ""
echo "� 停止旧的vLLM服务..."

# 查找并停止端口8002的vLLM进程
OLD_PIDS=$(ps aux | grep -E "(vllm.*$GEN_PORT|python.*start_generation)" | grep -v grep | awk '{print $2}')
if [ -n "$OLD_PIDS" ]; then
    echo "   发现运行中的进程: $OLD_PIDS"
    pkill -f "vllm.*$GEN_PORT" 2>/dev/null || true
    sleep 3
    echo "   ✅ 已停止旧服务"
else
    echo "   ℹ️  无运行中的服务"
fi

echo ""
echo "⏳ 启动Generation服务..."
echo "   模型: $GEN_MODEL"
echo "   GPU: $GEN_GPU"
echo "   端口: $GEN_PORT"
echo "   日志: logs/generation_vllm.log"

# 后台启动服务
nohup bash scripts/start_generation_vllm.sh > logs/generation_vllm.log 2>&1 &
GEN_PID=$!
echo "   进程PID: $GEN_PID"

# 等待服务启动
echo "   等待模型加载（预计30-60秒）..."
SUCCESS=false

for i in {1..80}; do
    sleep 2
    
    # 检查进程是否还在运行
    if ! kill -0 $GEN_PID 2>/dev/null; then
        echo ""
        echo "   ❌ 服务进程已退出，请检查日志:"
        echo "   tail -50 logs/generation_vllm.log"
        exit 1
    fi
    
    # 检查服务是否响应
    if curl -s --max-time 3 http://localhost:$GEN_PORT/v1/models >/dev/null 2>&1; then
        echo ""
        echo "   ✅ Generation服务启动成功！"
        SUCCESS=true
        break
    fi
    
    # 每10秒显示一次进度
    if [ $((i % 5)) -eq 0 ]; then
        echo -n " ${i}s"
    else
        echo -n "."
    fi
done

if [ "$SUCCESS" = false ]; then
    echo ""
    echo "   ⏱️  服务启动超时（160秒），可能仍在加载"
    echo "   请手动检查:"
    echo "     tail -f logs/generation_vllm.log"
    exit 1
fi
fi

echo ""
echo "================================"
echo "✅ 服务启动完成"
echo ""
echo "📊 服务状态："

# 检查各服务状态
if curl -s --max-time 3 http://localhost:$OCR_PORT/v1/models >/dev/null 2>&1; then
    echo "   OCR ($OCR_MODEL):        http://localhost:$OCR_PORT ✅"
else
    echo "   OCR ($OCR_MODEL):        http://localhost:$OCR_PORT ❌"
fi

if curl -s --max-time 3 http://localhost:$EMB_PORT/v1/models >/dev/null 2>&1; then
    echo "   Dense Embedding ($EMB_MODEL):     http://localhost:$EMB_PORT ✅"
else
    echo "   Dense Embedding ($EMB_MODEL):     http://localhost:$EMB_PORT ❌"
fi

if curl -s --max-time 3 http://localhost:$GEN_PORT/v1/models >/dev/null 2>&1; then
    echo "   Generation ($GEN_MODEL): http://localhost:$GEN_PORT ✅"
else
    echo "   Generation ($GEN_MODEL): http://localhost:$GEN_PORT ❌"
fi

echo ""
echo "🎯 现在可以启动UI："
echo "   bash scripts/start_ui.sh"
echo "================================"
