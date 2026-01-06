#!/bin/bash
# 完整重启服务和清理数据

set -e

echo "========================================"
echo "完整重启系统"
echo "========================================"
echo ""

# 1. 停止所有服务
echo "步骤 1/5: 停止现有服务..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "python.*ui/main" 2>/dev/null || true
sleep 3
echo "✅ 服务已停止"
echo ""

# 2. 清理数据
echo "步骤 2/5: 清理文档和索引..."
rm -rf data/docs/*
rm -rf data/indices/*
echo "✅ 数据已清理"
echo ""

# 3. 启动 vLLM 服务
echo "步骤 3/5: 启动 vLLM 服务..."
cd /workspace/doc-rag-evidence

mkdir -p logs

./scripts/start_ocr_vllm.sh > logs/vllm_ocr.log 2>&1 &
OCR_PID=$!
echo "  OCR 服务启动中 (PID: $OCR_PID)..."

./scripts/start_embedding_vllm.sh > logs/vllm_embedding.log 2>&1 &
EMB_PID=$!
echo "  Embedding 服务启动中 (PID: $EMB_PID)..."

echo ""
echo "等待服务就绪（约 60 秒）..."
sleep 60

# 4. 验证服务
echo ""
echo "步骤 4/5: 验证服务状态..."

if ./scripts/wait_for_vllm.sh; then
    echo ""
else
    echo "❌ OCR 服务验证失败！"
    echo "查看日志: tail -f logs/vllm_ocr.log"
    exit 1
fi

# 检查 embedding 服务
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Embedding 服务就绪 (http://localhost:8001)"
else
    echo "⚠️  Embedding 服务未就绪（可能还在加载中）"
fi

echo ""
echo "步骤 5/5: 启动 UI..."
echo ""
echo "========================================"
echo "系统已就绪！"
echo "========================================"
echo ""
echo "服务状态:"
echo "  • OCR:       http://localhost:8000 (PID: $OCR_PID)"
echo "  • Embedding: http://localhost:8001 (PID: $EMB_PID)"
echo ""
echo "现在启动 UI:"
echo "  python app/ui/main_v1.py"
echo ""
echo "或查看日志:"
echo "  tail -f logs/vllm_ocr.log"
echo "  tail -f logs/vllm_embedding.log"
echo ""
