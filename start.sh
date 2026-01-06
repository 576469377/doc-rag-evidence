#!/bin/bash
# 快速启动 Doc-RAG-Evidence 系统的完整脚本

echo "======================================================"
echo "🚀 Doc-RAG-Evidence 系统启动"
echo "======================================================"
echo ""

# 检查是否在项目目录
if [ ! -f "configs/app.yaml" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "步骤 1/4: 启动 vLLM 服务..."
echo "----------------------------------------"
./scripts/start_all_vllm.sh

echo ""
echo "步骤 2/4: 等待服务启动 (30秒)..."
echo "首次启动需要加载模型，请耐心等待..."
echo "----------------------------------------"
sleep 30

echo ""
echo "步骤 3/4: 验证服务状态..."
echo "----------------------------------------"

# 检查 OCR 服务
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ HunyuanOCR (Port 8000) - 运行中"
else
    echo "⚠️  HunyuanOCR (Port 8000) - 未就绪 (可能需要更多时间)"
fi

# 检查 Embedding 服务
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Qwen3-Embedding (Port 8001) - 运行中"
else
    echo "⚠️  Qwen3-Embedding (Port 8001) - 未就绪 (可能需要更多时间)"
fi

echo ""
echo "步骤 4/4: 启动 UI 界面..."
echo "----------------------------------------"
echo "UI 将在 http://localhost:7860 启动"
echo ""
echo "提示:"
echo "  - 如果服务显示'未就绪'，请等待1-2分钟后刷新UI"
echo "  - 查看日志: tail -f logs/vllm_*.log"
echo "  - 停止服务: ./scripts/stop_all_vllm.sh"
echo ""
echo "======================================================"

python app/ui/main_v1.py
