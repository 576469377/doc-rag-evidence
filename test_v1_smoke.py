#!/usr/bin/env python3
"""
Quick smoke test for V0.1 implementation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all V0.1 modules can be imported."""
    print("Testing imports...")
    
    try:
        from impl.ocr_client import SGLangOcrClient, MockOcrClient
        from impl.block_builder import BlockBuilder
        from impl.index_dense import DenseIndexerRetriever, SGLangEmbedder
        from impl.index_colpali import ColPaliRetriever
        from impl.ingest_pdf_v1 import PDFIngestorV1
        from app.ui.main_v1 import DocRAGUIV1
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        from core.schemas import AppConfig
        
        with open("configs/app.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = AppConfig(**config_dict)
        print(f"✅ Config loaded: {config.retrieval_mode} mode")
        print(f"   OCR provider: {config.ocr.get('provider', 'not set')}")
        print(f"   Dense enabled: {config.dense.get('enabled', False)}")
        print(f"   ColPali enabled: {config.colpali.get('enabled', False)}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_bm25_index():
    """Test BM25 index loading."""
    print("\nTesting BM25 index...")
    
    try:
        from core.schemas import AppConfig
        from infra.store_local import DocumentStoreLocal
        from impl.index_bm25 import BM25IndexerRetriever
        import yaml
        
        with open("configs/app.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        
        store = DocumentStoreLocal(config)
        indexer = BM25IndexerRetriever(store)
        
        try:
            indexer.load(config, index_name="bm25_default")
            print(f"✅ BM25 index loaded: {len(indexer.units)} units")
            return True
        except FileNotFoundError:
            print("⚠️  BM25 index not found (run: python scripts/build_indices_v1.py --bm25)")
            return True  # Not a failure, just not built yet
    except Exception as e:
        print(f"❌ BM25 test failed: {e}")
        return False


def test_ui_init():
    """Test UI initialization."""
    print("\nTesting UI initialization...")
    
    try:
        from app.ui.main_v1 import DocRAGUIV1
        
        ui = DocRAGUIV1()
        print(f"✅ UI initialized")
        print(f"   Available modes: {list(ui.retrievers.keys())}")
        return True
    except Exception as e:
        print(f"❌ UI init failed: {e}")
        return False


def main():
    print("="*60)
    print("V0.1 Smoke Tests")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("BM25 Index", test_bm25_index()))
    results.append(("UI Initialization", test_ui_init()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
