#!/usr/bin/env python3
"""
对比OCR客户端的请求与测试脚本的差异
找出400错误的原因
"""

import requests
import base64
from pathlib import Path
import json

# 测试图片
test_image = Path("/workspace/doc-rag-evidence/data/docs/gb_1886.3-2021/pages/0001/page.png")

with open(test_image, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print("=" * 70)
print("测试1: 使用测试脚本的请求格式 (成功的)")
print("=" * 70)

payload1 = {
    "model": "tencent/HunyuanOCR",
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
                {"type": "text", "text": "Extract text from this image."}
            ]
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.0
}

print(f"Payload keys: {list(payload1.keys())}")
print(f"Model: {payload1['model']}")
print(f"Messages count: {len(payload1['messages'])}")
print(f"Temperature: {payload1['temperature']}")
print(f"Max tokens: {payload1['max_tokens']}")

try:
    response1 = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload1,
        timeout=30
    )
    print(f"Status: {response1.status_code}")
    if response1.status_code == 200:
        print("✅ 成功!")
    else:
        print(f"❌ 失败: {response1.text[:500]}")
except Exception as e:
    print(f"❌ 异常: {e}")

print("\n" + "=" * 70)
print("测试2: 使用OCR客户端的请求格式 (400错误的)")
print("=" * 70)

# 完全复制 ocr_client.py 的格式
prompt = (
    "Extract all information from the main body of the document image "
    "and represent it in markdown format, ignoring headers and footers. "
    "Tables should be expressed in HTML format, formulas in the document "
    "should be represented using LaTeX format, and the parsing should be "
    "organized according to the reading order."
)

payload2 = {
    "model": "tencent/HunyuanOCR",
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

# 添加 extra_body (ocr_client.py line 191-195)
payload2["extra_body"] = {
    "top_k": 1,
    "repetition_penalty": 1.0
}

print(f"Payload keys: {list(payload2.keys())}")
print(f"Model: {payload2['model']}")
print(f"Messages count: {len(payload2['messages'])}")
print(f"Temperature: {payload2['temperature']}")
print(f"Max tokens: {payload2['max_tokens']}")
print(f"Extra body: {payload2.get('extra_body')}")

try:
    response2 = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload2,
        timeout=30
    )
    print(f"Status: {response2.status_code}")
    if response2.status_code == 200:
        print("✅ 成功!")
        result = response2.json()
        text = result["choices"][0]["message"]["content"]
        print(f"提取文本长度: {len(text)} 字符")
    else:
        print(f"❌ 失败: {response2.text}")
except Exception as e:
    print(f"❌ 异常: {e}")

print("\n" + "=" * 70)
print("差异对比")
print("=" * 70)
print("主要差异:")
print(f"  1. max_tokens: 2048 vs 4096")
print(f"  2. extra_body: 无 vs {{'top_k': 1, 'repetition_penalty': 1.0}}")
print(f"  3. prompt长度: 短 vs 长")
