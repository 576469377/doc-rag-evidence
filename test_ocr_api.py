#!/usr/bin/env python3
"""测试 OCR API 调用"""

import sys
import yaml
from pathlib import Path
from impl.ocr_client import SGLangOcrClient

# 加载配置
with open("configs/app.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ocr_config = config['ocr']

print("=" * 60)
print("测试 OCR API 调用")
print("=" * 60)
print(f"端点: {ocr_config['endpoint']}")
print(f"模型: {ocr_config['model']}")
print()

# 创建 OCR 客户端
client = SGLangOcrClient(
    endpoint=ocr_config['endpoint'],
    model=ocr_config['model'],
    timeout=ocr_config.get('timeout', 60)
)

# 查找测试图片
test_images = list(Path("data/docs").glob("*/pages/0000/page.png"))
if not test_images:
    print("❌ 找不到测试图片 (data/docs/*/pages/0000/page.png)")
    sys.exit(1)

test_image = test_images[0]
print(f"测试图片: {test_image}")
print()

# 调用 OCR
print("正在调用 OCR API...")
try:
    result = client.ocr_page(str(test_image))
    print("✅ OCR 调用成功!")
    print()
    print("提取的文本长度:", len(result.text))
    print("提取的文本预览 (前500字符):")
    print("-" * 60)
    print(result.text[:500])
    print("-" * 60)
    
    if result.text:
        print()
        print("✅ OCR 功能正常工作!")
    else:
        print()
        print("⚠️  OCR 返回了空文本")
        
except Exception as e:
    print(f"❌ OCR 调用失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
