#!/usr/bin/env python3
"""
测试OCR服务的API调用
验证 /v1/chat/completions 端点是否正常工作
"""

import requests
import base64
import json
from pathlib import Path

def test_ocr_service():
    """测试OCR服务"""
    
    # 1. 检查服务状态
    print("=" * 60)
    print("1. 检查OCR服务状态")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 服务运行正常")
            print(f"   可用模型: {models['data'][0]['id']}")
        else:
            print(f"❌ 服务响应异常: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return
    
    # 2. 查找一个测试图片
    print("\n" + "=" * 60)
    print("2. 准备测试图片")
    print("=" * 60)
    
    # 查找任意一个页面图片
    data_dir = Path("/workspace/doc-rag-evidence/data/docs")
    image_files = list(data_dir.glob("*/pages/*/page.png"))
    
    if not image_files:
        print("❌ 未找到测试图片")
        return
    
    test_image = image_files[0]
    print(f"✅ 使用测试图片: {test_image}")
    
    # 3. 编码图片
    with open(test_image, "rb") as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
    
    print(f"   图片大小: {len(image_data)} bytes")
    print(f"   Base64长度: {len(image_b64)} chars")
    
    # 4. 测试 /v1/chat/completions 端点
    print("\n" + "=" * 60)
    print("3. 测试 /v1/chat/completions 端点")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    # HunyuanOCR的标准提示词
    prompt = (
        "Extract all information from the main body of the document image "
        "and represent it in markdown format, ignoring headers and footers. "
        "Tables should be expressed in HTML format, formulas in the document "
        "should be represented using LaTeX format, and the parsing should be "
        "organized according to the reading order."
    )
    
    payload = {
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
        "max_tokens": 2048,
        "temperature": 0.0
    }
    
    print(f"   请求URL: {url}")
    print(f"   模型: {payload['model']}")
    print(f"   Max tokens: {payload['max_tokens']}")
    print("\n   发送请求...")
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"   响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            
            print(f"\n✅ OCR识别成功!")
            print(f"   提取文本长度: {len(extracted_text)} 字符")
            print(f"\n   前200字符预览:")
            print("   " + "-" * 56)
            print("   " + extracted_text[:200].replace("\n", "\n   "))
            print("   " + "-" * 56)
            
            return True
            
        else:
            print(f"\n❌ API调用失败: {response.status_code}")
            print(f"   响应内容: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时 (60秒)")
        return False
    except Exception as e:
        print(f"\n❌ 请求异常: {e}")
        return False

if __name__ == "__main__":
    print("\n🧪 OCR服务API测试")
    print("=" * 60)
    success = test_ocr_service()
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过！OCR服务工作正常")
    else:
        print("⚠️  测试失败，请检查服务配置")
    print("=" * 60 + "\n")
