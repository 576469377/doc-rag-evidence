#!/bin/bash
# setup.sh - 一键安装和验证脚本

set -e  # Exit on error

echo "============================================================"
echo "Doc RAG Evidence System V0 - Setup Script"
echo "============================================================"

# 检查Python版本
echo ""
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# 安装依赖
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "🔍 Verifying installation..."
python -c "import pydantic; print('✅ pydantic:', pydantic.__version__)"
python -c "import yaml; print('✅ PyYAML installed')"
python -c "import pdfplumber; print('✅ pdfplumber installed')"
python -c "import rank_bm25; print('✅ rank-bm25 installed')"
python -c "import gradio; print('✅ gradio:', gradio.__version__)"

# 创建数据目录
echo ""
echo "📁 Creating data directories..."
mkdir -p data/docs data/indices data/runs data/reports
echo "✅ Data directories created"

# 运行测试
echo ""
echo "🧪 Running basic tests..."
python tests/test_basic.py

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Start UI:    python run.py"
echo "  2. Or use make: make run"
echo "  3. Visit:       http://127.0.0.1:7860"
echo ""
