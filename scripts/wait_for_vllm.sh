#!/bin/bash
# 等待 vLLM 服务完全就绪

echo "等待 vLLM OCR 服务完全就绪..."
echo ""

MAX_WAIT=60  # 最多等待60秒
COUNTER=0

while [ $COUNTER -lt $MAX_WAIT ]; do
    # 检查健康端点
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        # 检查模型列表
        MODEL=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null)
        
        if [ "$MODEL" = "tencent/HunyuanOCR" ]; then
            echo "✅ vLLM OCR 服务就绪!"
            echo "   端点: http://localhost:8000"
            echo "   模型: $MODEL"
            echo ""
            
            # 测试一个简单的请求
            echo "测试 API 调用..."
            RESULT=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{"model":"tencent/HunyuanOCR","messages":[{"role":"user","content":"test"}],"max_tokens":5}' 2>&1)
            
            if echo "$RESULT" | grep -q "choices"; then
                echo "✅ API 测试成功!"
                echo ""
                echo "========================================"
                echo "服务已完全就绪，可以使用了！"
                echo "========================================"
                exit 0
            else
                echo "⚠️  API 调用失败，继续等待..."
            fi
        fi
    fi
    
    echo -n "."
    sleep 2
    COUNTER=$((COUNTER + 2))
done

echo ""
echo "❌ 等待超时！服务可能未正确启动。"
echo ""
echo "检查日志:"
echo "  tail -f logs/vllm_ocr.log"
exit 1
