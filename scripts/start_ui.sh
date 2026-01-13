#!/bin/bash
# 启动Gradio UI
# 使用方法: bash scripts/start_ui.sh

set -e

# 清除代理设置（避免localhost访问问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

echo "🎨 启动 Doc RAG Evidence UI"
echo "================================"

# 激活conda环境
source /workspace/program/miniconda3/etc/profile.d/conda.sh
conda activate doc-rag-evidence-vllm
cd /workspace/doc-rag-evidence

# 检查必需服务
echo "📋 检查服务状态..."
echo ""

services_ok=true

# 检查Generation服务（必需）
echo -n "🔍 Generation服务 (端口8002): "
if curl -s --max-time 5 http://localhost:8002/v1/models >/dev/null 2>&1; then
    echo "✅ 运行中"
else
    echo "❌ 未运行"
    echo ""
    echo "请先启动Generation服务："
    echo "  bash scripts/start_services.sh"
    services_ok=false
fi

# 检查Embedding服务（可选）
echo -n "🔍 Embedding服务 (端口8001): "
if curl -s --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "✅ 运行中 (Dense检索可用)"
else
    echo "⚠️  未运行 (仅BM25/ColPali可用)"
fi

# 检查OCR服务（可选）
echo -n "🔍 OCR服务 (端口8000): "
if curl -s --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "✅ 运行中 (PDF导入可用)"
else
    echo "⚠️  未运行 (PDF导入不可用)"
fi

if [ "$services_ok" = false ]; then
    exit 1
fi

echo ""
echo "================================"
echo "🚀 启动Gradio UI"
echo "   访问地址: http://localhost:7860"
echo "   按 Ctrl+C 停止"
echo "================================"
echo ""

# 启动UI
python app/ui/main_v1.py
