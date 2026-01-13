#!/bin/bash
# 启动所有后台服务（OCR、Embedding、Generation）
# 使用方法: bash scripts/start_services.sh [all|generation|embedding|ocr]
# 默认启动所有服务

set -e

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# 解析参数
MODE="${1:-all}"  # 默认启动所有服务

echo "🚀 启动 Doc RAG Evidence 后台服务"
echo "================================"
echo "模式: $MODE"

# 激活conda环境
source /workspace/program/miniconda3/etc/profile.d/conda.sh
conda activate doc-rag-evidence-vllm
cd /workspace/doc-rag-evidence

# 创建日志目录
mkdir -p logs

# ========== OCR服务 (端口8000) ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "ocr" ]; then
    echo ""
    echo "🛑 停止旧的OCR服务 (端口8000)..."
    OLD_PIDS=$(ps aux | grep -E "vllm.*8000" | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_PIDS" ]; then
        echo "   发现运行中的进程: $OLD_PIDS"
        pkill -f "vllm.*8000" 2>/dev/null || true
        sleep 3
        echo "   ✅ 已停止旧OCR服务"
    else
        echo "   ℹ️  无运行中的OCR服务"
    fi
    
    echo ""
    echo "⏳ 启动OCR服务..."
    echo "   模型: HunyuanOCR"
    echo "   GPU: GPU 0"
    echo "   日志: logs/ocr_vllm.log"
    
    nohup bash scripts/start_ocr_vllm.sh > logs/ocr_vllm.log 2>&1 &
    OCR_PID=$!
    echo "   进程PID: $OCR_PID"
    echo "   等待启动..."
    
    for i in {1..40}; do
        sleep 2
        if curl -s --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
            echo "   ✅ OCR服务启动成功！"
            break
        fi
        [ $((i % 5)) -eq 0 ] && echo -n " ${i}s" || echo -n "."
    done
fi

# ========== Embedding服务 (端口8001) ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "embedding" ]; then
    echo ""
    echo "🛑 停止旧的Embedding服务 (端口8001)..."
    OLD_PIDS=$(ps aux | grep -E "vllm.*8001" | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_PIDS" ]; then
        echo "   发现运行中的进程: $OLD_PIDS"
        pkill -f "vllm.*8001" 2>/dev/null || true
        sleep 3
        echo "   ✅ 已停止旧Embedding服务"
    else
        echo "   ℹ️  无运行中的Embedding服务"
    fi
    
    echo ""
    echo "⏳ 启动Embedding服务..."
    echo "   模型: Qwen3-Embedding-0.6B"
    echo "   GPU: GPU 1"
    echo "   日志: logs/embedding_vllm.log"
    
    nohup bash scripts/start_embedding_vllm.sh > logs/embedding_vllm.log 2>&1 &
    EMB_PID=$!
    echo "   进程PID: $EMB_PID"
    echo "   等待启动..."
    
    for i in {1..40}; do
        sleep 2
        if curl -s --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
            echo "   ✅ Embedding服务启动成功！"
            break
        fi
        [ $((i % 5)) -eq 0 ] && echo -n " ${i}s" || echo -n "."
    done
fi

# ========== Generation服务 (端口8002) ==========
if [ "$MODE" = "all" ] || [ "$MODE" = "generation" ]; then

echo ""
echo "� 停止旧的vLLM服务..."

# 查找并停止端口8002的vLLM进程
OLD_PIDS=$(ps aux | grep -E "(vllm.*8002|python.*start_generation)" | grep -v grep | awk '{print $2}')
if [ -n "$OLD_PIDS" ]; then
    echo "   发现运行中的进程: $OLD_PIDS"
    pkill -f "vllm.*8002" 2>/dev/null || true
    sleep 3
    echo "   ✅ 已停止旧服务"
else
    echo "   ℹ️  无运行中的服务"
fi

echo ""
echo "⏳ 启动Generation服务..."
echo "   模型: Qwen3-VL-4B-Instruct"
echo "   GPU: GPU 3"
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
    if curl -s --max-time 3 http://localhost:8002/v1/models >/dev/null 2>&1; then
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
if curl -s --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "   OCR (HunyuanOCR):        http://localhost:8000 ✅"
else
    echo "   OCR (HunyuanOCR):        http://localhost:8000 ❌"
fi

if curl -s --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "   Embedding (Qwen3):       http://localhost:8001 ✅"
else
    echo "   Embedding (Qwen3):       http://localhost:8001 ❌"
fi

if curl -s --max-time 3 http://localhost:8002/v1/models >/dev/null 2>&1; then
    echo "   Generation (Qwen3-VL):   http://localhost:8002 ✅"
else
    echo "   Generation (Qwen3-VL):   http://localhost:8002 ❌"
fi

echo ""
echo "🎯 现在可以启动UI："
echo "   bash scripts/start_ui.sh"
echo "================================"
