#!/usr/bin/env python3
"""测试 OCR API 调用（使用临时测试图片）"""

import sys
import yaml
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# 创建一个简单的测试图片
print("创建测试图片...")
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)

# 绘制文本
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
except:
    font = ImageFont.load_default()

draw.text((50, 150), "Test OCR Recognition", fill='black', font=font)
draw.text((50, 250), "这是一个测试文本", fill='black', font=font)

# 编码为 base64
buffer = BytesIO()
img.save(buffer, format='PNG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# 加载配置
with open("configs/app.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ocr_config = config['ocr']

print("=" * 60)
print("测试 OCR API 调用")
print("=" * 60)
print(f"端点: {ocr_config['endpoint']}/v1/chat/completions")
print(f"模型: {ocr_config['model']}")
print()

# 准备请求
url = f"{ocr_config['endpoint']}/v1/chat/completions"

prompt = (
    "Extract all information from the main body of the document image "
    "and represent it in markdown format, ignoring headers and footers. "
    "Tables should be expressed in HTML format, formulas in the document "
    "should be represented using LaTeX format, and the parsing should be "
    "organized according to the reading order."
)

payload = {
    "model": ocr_config['model'],
    "messages": [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 4096
}

print("正在调用 OCR API...")
print(f"图片大小: {len(image_b64)} 字符 (base64)")
print()

try:
    response = requests.post(url, json=payload, timeout=60)
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        text = result['choices'][0]['message']['content']
        
        print("✅ OCR 调用成功!")
        print()
        print("提取的文本:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        
        if text and len(text) > 10:
            print()
            print("✅ OCR 功能正常工作!")
        else:
            print()
            print("⚠️  OCR 返回了空文本或文本太短")
    else:
        print(f"❌ HTTP 错误: {response.status_code}")
        print(f"响应: {response.text[:500]}")
        
except Exception as e:
    print(f"❌ OCR 调用失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
