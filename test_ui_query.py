#!/usr/bin/env python3
"""Test UI query handling with all three modes."""
import sys
import yaml
from pathlib import Path

from core.schemas import AppConfig, QueryInput
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator
from core.pipeline import Pipeline

def main():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    store = DocumentStoreLocal(config)
    logger = RunLoggerLocal(config)
    selector = TopKEvidenceSelector(snippet_length=500)
    generator = TemplateGenerator(mode="summary")
    
    print("=" * 60)
    print("Testing UI Query with All Modes")
    print("=" * 60)
    
    # Test query
    query_input = QueryInput(
        query_id="ui_test_001",
        question="碳酸钙的使用范围",
        doc_filter=[]
    )
    print(f"\nQuery: {query_input.question}\n")
    
    # Load retrievers
    retrievers = {}
    
    # BM25
    try:
        retriever = BM25IndexerRetriever(store)
        retriever.load(config, index_name="bm25_default")
        retrievers["bm25"] = retriever
        print(f"✅ Loaded BM25")
    except Exception as e:
        print(f"❌ BM25 load failed: {e}")
    
    # Dense
    if config.dense.get("enabled"):
        try:
            embedder = VLLMEmbedder(
                endpoint=config.dense["endpoint"],
                model=config.dense["model"]
            )
            retriever = DenseIndexerRetriever.load(Path("data/indices/dense_default"), embedder)
            retrievers["dense"] = retriever
            print(f"✅ Loaded Dense")
        except Exception as e:
            print(f"❌ Dense load failed: {e}")
    
    # ColPali
    if config.colpali.get("enabled"):
        try:
            retriever = ColPaliRetriever.load(
                Path("data/indices/colpali_default"),
                model_name=config.colpali["model"],
                device=config.colpali.get("device", "cuda:2")
            )
            retrievers["colpali"] = retriever
            print(f"✅ Loaded ColPali")
        except Exception as e:
            print(f"❌ ColPali load failed: {e}")
    
    # Test each retriever in pipeline
    for mode, retriever in retrievers.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {mode.upper()} Pipeline")
        print("=" * 60)
        
        try:
            # Create pipeline with this retriever
            pipeline = Pipeline(
                retriever=retriever,
                selector=selector,
                generator=generator,
                logger=logger,
                reranker=None
            )
            
            # Run full pipeline
            result = pipeline.answer(query_input, config)
            
            print(f"✅ Pipeline executed successfully")
            print(f"   Retrieval hits: {len(result.retrieval.hits)}")
            print(f"   Evidence items: {len(result.evidence.evidence) if result.evidence else 0}")
            print(f"   Answer length: {len(result.generation.output.answer) if result.generation else 0} chars")
            
            if result.evidence and result.evidence.evidence:
                print(f"\n   Top evidence:")
                for i, ev in enumerate(result.evidence.evidence[:3], 1):
                    print(f"     {i}. Doc: {ev.doc_id}, Page: {ev.page_id}, Score: {ev.score:.4f}")
        
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("UI Query Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
