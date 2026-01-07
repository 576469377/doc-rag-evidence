#!/usr/bin/env python3
"""Test query with all three retrieval modes."""
import sys
import yaml
from pathlib import Path

from core.schemas import AppConfig, QueryInput
from infra.store_local import DocumentStoreLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever

def main():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    store = DocumentStoreLocal(config)
    indices_dir = Path(config.indices_dir)
    
    print("=" * 60)
    print("Testing Query with All Retrieval Modes")
    print("=" * 60)
    
    # Test query
    query = "碳酸钙的使用范围"
    print(f"\nQuery: {query}\n")
    
    # Load retrievers
    retrievers = {}
    
    # BM25
    try:
        retriever = BM25IndexerRetriever(store)
        retriever.load(config, index_name="bm25_default")
        retrievers["bm25"] = retriever
    except Exception as e:
        print(f"BM25 load failed: {e}")
    
    # Dense
    if config.dense.get("enabled"):
        try:
            embedder = VLLMEmbedder(
                endpoint=config.dense["endpoint"],
                model=config.dense["model"]
            )
            retriever = DenseIndexerRetriever.load(indices_dir / "dense_default", embedder)
            retrievers["dense"] = retriever
        except Exception as e:
            print(f"Dense load failed: {e}")
    
    # ColPali
    if config.colpali.get("enabled"):
        try:
            retriever = ColPaliRetriever.load(
                indices_dir / "colpali_default",
                model_name=config.colpali["model"],
                device=config.colpali.get("device", "cuda:2")
            )
            retrievers["colpali"] = retriever
        except Exception as e:
            print(f"ColPali load failed: {e}")
    
    # Test each retriever
    for mode, retriever in retrievers.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {mode.upper()}")
        print("=" * 60)
        
        try:
            # Different APIs for different retrievers
            if mode == "bm25":
                # BM25 uses QueryInput and AppConfig
                query_input = QueryInput(
                    query_id="test_001",
                    question=query,
                    doc_filter=[]
                )
                result = retriever.retrieve(query_input, config)
                hits = result.hits[:3]  # Top 3
            else:
                # Dense and ColPali use string query and top_k
                hits = retriever.retrieve(query, top_k=3)
            
            print(f"✅ Retrieved {len(hits)} results:")
            
            for i, hit in enumerate(hits, 1):
                print(f"\n  {i}. Doc: {hit.doc_id}, Page: {hit.page_id}")
                print(f"     Score: {hit.score:.4f}")
                print(f"     Source: {hit.source}")
                if hit.text:
                    text_preview = hit.text[:80].replace('\n', ' ')
                    print(f"     Text: {text_preview}...")
                if hit.metadata:
                    print(f"     Metadata: {hit.metadata}")
        
        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Query Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
