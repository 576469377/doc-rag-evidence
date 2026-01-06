#!/bin/bash
# 清理旧文档数据并准备重新导入

echo "========================================"
echo "清理文档数据"
echo "========================================"
echo ""

# 备份旧数据（如果需要）
if [ -d "data/docs" ] && [ "$(ls -A data/docs)" ]; then
    echo "备份旧文档到 data/docs_backup_$(date +%Y%m%d_%H%M%S)/"
    cp -r data/docs "data/docs_backup_$(date +%Y%m%d_%H%M%S)"
    echo "✅ 备份完成"
    echo ""
fi

# 清理文档目录
echo "清理 data/docs/ ..."
rm -rf data/docs/*
echo "✅ 文档目录已清空"
echo ""

# 清理索引
echo "清理 data/indices/ ..."
rm -rf data/indices/*
echo "✅ 索引目录已清空"
echo ""

# 检查 OCR 服务状态
echo "检查 OCR 服务状态..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ OCR 服务正常 (http://localhost:8000)"
    
    # 检查模型名称
    MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    if [ -n "$MODEL" ]; then
        echo "   模型: $MODEL"
    fi
else
    echo "❌ OCR 服务未运行!"
    echo ""
    echo "请先启动 OCR 服务:"
    echo "  ./scripts/start_ocr_vllm.sh"
    echo "  或"
    echo "  ./scripts/start_all_vllm.sh"
    exit 1
fi

echo ""
echo "========================================"
echo "准备就绪!"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 在 UI 中上传 PDF (http://localhost:7860)"
echo "  2. ☑ 勾选 'Use OCR'"
echo "  3. 等待处理完成 (每页约10-30秒)"
echo "  4. 验证文本: cat data/docs/YOUR_DOC_ID/pages/0000/text.json"
echo "  5. 构建索引: 选择 BM25 并点击 'Build Indices'"
echo ""
