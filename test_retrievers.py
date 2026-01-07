#!/usr/bin/env python3
"""Test retriever initialization."""
import sys
import yaml
from pathlib import Path

from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever

def main():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    print("=" * 60)
    print("Testing Retriever Initialization")
    print("=" * 60)
    
    store = DocumentStoreLocal(config)
    indices_dir = Path(config.indices_dir)
    
    retrievers = {}
    
    # Test BM25
    print("\n1. Testing BM25...")
    bm25_index_name = "bm25_default"
    try:
        retriever = BM25IndexerRetriever(store)
        retriever.load(config, index_name=bm25_index_name)
        retrievers["bm25"] = retriever
        print(f"   ✅ Loaded BM25 index: {len(retriever.units)} units")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Dense
    print("\n2. Testing Dense...")
    if config.dense.get("enabled"):
        dense_index_name = "dense_default"
        dense_index_dir = indices_dir / dense_index_name
        if dense_index_dir.exists():
            try:
                embedder = VLLMEmbedder(
                    endpoint=config.dense["endpoint"],
                    model=config.dense["model"],
                    batch_size=config.dense.get("batch_size", 32)
                )
                retriever = DenseIndexerRetriever.load(dense_index_dir, embedder)
                retrievers["dense"] = retriever
                print(f"   ✅ Loaded Dense index: {len(retriever.units)} units")
                print(f"      Endpoint: {config.dense['endpoint']}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  Index not found: {dense_index_dir}")
    else:
        print("   ⏭️  Disabled in config")
    
    # Test ColPali
    print("\n3. Testing ColPali...")
    if config.colpali.get("enabled"):
        colpali_index_name = "colpali_default"
        colpali_index_dir = indices_dir / colpali_index_name
        if colpali_index_dir.exists():
            try:
                device = config.colpali.get("device", "cuda:2")
                retriever = ColPaliRetriever.load(
                    colpali_index_dir,
                    model_name=config.colpali["model"],
                    device=device
                )
                retrievers["colpali"] = retriever
                print(f"   ✅ Loaded ColPali index")
                print(f"      Device: {device}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  Index not found: {colpali_index_dir}")
    else:
        print("   ⏭️  Disabled in config")
    
    print("\n" + "=" * 60)
    print(f"Available retrievers: {list(retrievers.keys())}")
    print("=" * 60)
    
    if len(retrievers) > 0:
        print("\n✅ Success! At least one retriever loaded.")
    else:
        print("\n❌ No retrievers loaded!")
        sys.exit(1)


if __name__ == "__main__":
    main()
