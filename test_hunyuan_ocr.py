#!/usr/bin/env python3
"""
Test HunyuanOCR integration via vllm.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from impl.ocr_client import SGLangOcrClient
import yaml
from core.schemas import AppConfig


def test_hunyuan_connection():
    """Test connection to HunyuanOCR vllm server."""
    print("Testing HunyuanOCR connection...")
    
    try:
        import requests
        
        # Load config
        with open("configs/app.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        
        endpoint = config.ocr["endpoint"]
        
        # Test health endpoint
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ vllm server is running at {endpoint}")
            else:
                print(f"⚠️  vllm server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to vllm server at {endpoint}")
            print(f"   Error: {e}")
            print(f"\n💡 Please start vllm server first:")
            print(f"   vllm serve tencent/HunyuanOCR --no-enable-prefix-caching --mm-processor-cache-gb 0")
            return False
        
        # Test models endpoint
        try:
            response = requests.get(f"{endpoint}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Available models: {models.get('data', [])}")
            else:
                print(f"⚠️  Cannot list models (status {response.status_code})")
        except Exception as e:
            print(f"⚠️  Cannot list models: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_ocr_client():
    """Test OCR client initialization."""
    print("\nTesting OCR client...")
    
    try:
        import yaml
        from core.schemas import AppConfig
        
        # Load config
        with open("configs/app.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        
        # Create OCR client
        client = SGLangOcrClient(
            endpoint=config.ocr["endpoint"],
            model=config.ocr["model"],
            timeout=config.ocr.get("timeout", 60),
            cache_dir=Path(config.docs_dir) if config.ocr.get("cache_enabled") else None
        )
        
        print(f"✅ OCR client initialized")
        print(f"   Model: {client.model}")
        print(f"   Endpoint: {client.endpoint}")
        print(f"   Cache: {client.cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client init failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_on_sample():
    """Test OCR on a sample page if available."""
    print("\nTesting OCR on sample page...")
    
    try:
        from impl.ocr_client import SGLangOcrClient
        import yaml
        from core.schemas import AppConfig
        
        # Load config
        with open("configs/app.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        
        # Find a sample page image
        docs_dir = Path(config.docs_dir)
        sample_images = list(docs_dir.glob("*/pages/*/page.png"))
        
        if not sample_images:
            print("⚠️  No page images found. Run ingestion first:")
            print("   python scripts/ingest_docs_v1.py --pdf-dir data/pdfs")
            return True
        
        sample_image = sample_images[0]
        print(f"   Using: {sample_image}")
        
        # Create OCR client
        client = SGLangOcrClient(
            endpoint=config.ocr["endpoint"],
            model=config.ocr["model"],
            timeout=config.ocr.get("timeout", 60),
            cache_dir=None  # Disable cache for test
        )
        
        # Run OCR
        print("   Running OCR (may take 10-30 seconds)...")
        result = client.ocr_page(str(sample_image))
        
        if result.text:
            print(f"✅ OCR successful")
            print(f"   Extracted {len(result.text)} characters")
            print(f"   Preview: {result.text[:100]}...")
            return True
        else:
            print(f"❌ OCR returned empty text")
            if result.metadata.get("error"):
                print(f"   Error: {result.metadata['error']}")
            return False
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("HunyuanOCR Integration Test")
    print("="*60)
    
    results = []
    
    results.append(("Server Connection", test_hunyuan_connection()))
    results.append(("OCR Client Init", test_ocr_client()))
    
    # Only test actual OCR if server is running
    if results[0][1]:
        results.append(("OCR Execution", test_ocr_on_sample()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n📝 Next steps:")
        print("   1. Ingest documents with OCR:")
        print("      python scripts/ingest_docs_v1.py --pdf-dir data/pdfs --use-ocr")
        print("   2. Build indices:")
        print("      python scripts/build_indices_v1.py --bm25")
        print("   3. Launch UI:")
        print("      python app/ui/main_v1.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
