#!/usr/bin/env python3
"""
完整的系统验证测试 - V1 UI集成版
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块导入"""
    print("🔍 Testing imports...")
    try:
        from app.ui.main_v1 import DocRAGUIV1
        from impl.index_dense import VLLMEmbedder, SGLangEmbedder
        from impl.ingest_pdf_v1 import PDFIngestorV1
        from impl.index_bm25 import BM25IndexerRetriever
        from impl.ocr_client import SGLangOcrClient
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_scripts():
    """测试启动脚本存在性和权限"""
    print("\n🔍 Testing launch scripts...")
    scripts = [
        "scripts/start_ocr_vllm.sh",
        "scripts/start_embedding_vllm.sh",
        "scripts/start_all_vllm.sh",
        "scripts/stop_all_vllm.sh",
        "start.sh"
    ]
    
    all_ok = True
    for script in scripts:
        path = Path(script)
        if not path.exists():
            print(f"  ❌ Missing: {script}")
            all_ok = False
        elif not path.stat().st_mode & 0o111:
            print(f"  ⚠️  Not executable: {script}")
            all_ok = False
        else:
            print(f"  ✅ {script}")
    
    return all_ok

def test_config():
    """测试配置文件"""
    print("\n🔍 Testing configuration...")
    try:
        import yaml
        with open("configs/app.yaml") as f:
            config = yaml.safe_load(f)
        
        # Check OCR config
        ocr = config.get("ocr", {})
        assert ocr.get("provider") == "vllm", "OCR provider should be vllm"
        assert ocr.get("model") == "tencent/HunyuanOCR", "OCR model should be HunyuanOCR"
        assert ocr.get("endpoint") == "http://localhost:8000", "OCR endpoint should be :8000"
        print(f"  ✅ OCR: {ocr['model']} @ {ocr['endpoint']}")
        
        # Check Dense config
        dense = config.get("dense", {})
        assert dense.get("embedder_type") == "vllm", "Dense embedder should be vllm"
        assert dense.get("endpoint") == "http://localhost:8001", "Embedding endpoint should be :8001"
        print(f"  ✅ Dense: {dense['model']} @ {dense['endpoint']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_ui():
    """测试UI初始化和新增功能"""
    print("\n🔍 Testing UI...")
    try:
        from app.ui.main_v1 import DocRAGUIV1
        
        ui = DocRAGUIV1("configs/app.yaml")
        
        # Check handlers exist
        assert hasattr(ui, "_handle_upload"), "Missing _handle_upload"
        assert hasattr(ui, "_handle_build_indices"), "Missing _handle_build_indices"
        assert hasattr(ui, "_handle_query"), "Missing _handle_query"
        print("  ✅ UI handlers present")
        
        # Check retrievers
        print(f"  ✅ Available retrievers: {list(ui.retrievers.keys())}")
        
        # Check document list
        docs = ui._get_doc_list()
        print(f"  ✅ Documents loaded: {len(docs)} docs")
        
        return True
    except Exception as e:
        print(f"  ❌ UI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedder():
    """测试VLLMEmbedder类"""
    print("\n🔍 Testing VLLMEmbedder class...")
    try:
        from impl.index_dense import VLLMEmbedder
        
        # Just check class exists and can be instantiated
        embedder = VLLMEmbedder(
            endpoint="http://localhost:8001",
            model="Qwen/Qwen3-Embedding-0.6B"
        )
        
        assert embedder.endpoint == "http://localhost:8001"
        assert embedder.model == "Qwen/Qwen3-Embedding-0.6B"
        print("  ✅ VLLMEmbedder class works")
        
        return True
    except Exception as e:
        print(f"  ❌ VLLMEmbedder test failed: {e}")
        return False

def test_docs():
    """测试文档存在性"""
    print("\n🔍 Testing documentation...")
    docs = [
        "docs/QUICKSTART.md",
        "docs/VLLM_UPGRADE.md",
        "docs/HUNYUAN_OCR_GUIDE.md",
        "README.md"
    ]
    
    all_ok = True
    for doc in docs:
        path = Path(doc)
        if not path.exists():
            print(f"  ❌ Missing: {doc}")
            all_ok = False
        else:
            print(f"  ✅ {doc}")
    
    return all_ok

def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 Doc-RAG-Evidence V1 System Validation")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Scripts": test_scripts(),
        "Config": test_config(),
        "UI": test_ui(),
        "Embedder": test_embedder(),
        "Docs": test_docs()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. ./start.sh           - 启动所有服务")
        print("2. http://localhost:7860 - 访问UI界面")
        print("3. docs/QUICKSTART.md   - 查看使用指南")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
