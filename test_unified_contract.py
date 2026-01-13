#!/usr/bin/env python3
"""Test unified retrieval contract: all three modes return block-level evidence."""
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


def test_mode(mode_name, retriever, config, store, logger):
    """Test a single retrieval mode."""
    print(f"\n{'=' * 60}")
    print(f"Testing {mode_name.upper()} Mode")
    print("=" * 60)
    
    # Create pipeline with normalization
    selector = TopKEvidenceSelector(snippet_length=500)
    generator = TemplateGenerator(mode="summary")
    
    pipeline = Pipeline(
        retriever=retriever,
        selector=selector,
        generator=generator,
        logger=logger,
        reranker=None,
        store=store
    )
    
    # Test query
    query_input = QueryInput(
        query_id=f"{mode_name}_contract_test",
        question="碳酸钙的使用范围",
        doc_filter=[]
    )
    
    # Run pipeline
    result = pipeline.answer(query_input, config)
    
    # Validate contract
    print(f"\n📊 Results:")
    print(f"   Retrieval hits: {len(result.retrieval.hits) if result.retrieval else 0}")
    print(f"   Evidence items: {len(result.evidence.evidence) if result.evidence else 0}")
    
    # Contract validation
    contract_ok = True
    
    # Check 1: All hits have text
    if result.retrieval and result.retrieval.hits:
        hits_with_text = sum(1 for h in result.retrieval.hits if h.text and h.text.strip())
        print(f"\n   ✓ Hits with text: {hits_with_text}/{len(result.retrieval.hits)}")
        if hits_with_text < len(result.retrieval.hits):
            print(f"      ⚠️  Some hits lack text!")
            contract_ok = False
    
    # Check 2: All evidence has snippet
    if result.evidence and result.evidence.evidence:
        evidence_with_snippet = sum(1 for e in result.evidence.evidence if e.snippet and e.snippet.strip())
        print(f"   ✓ Evidence with snippet: {evidence_with_snippet}/{len(result.evidence.evidence)}")
        if evidence_with_snippet < len(result.evidence.evidence):
            print(f"      ⚠️  Some evidence lacks snippet!")
            contract_ok = False
    
    # Check 3: Evidence schema fields
    if result.evidence and result.evidence.evidence:
        sample = result.evidence.evidence[0]
        print(f"\n   📝 Sample evidence schema:")
        print(f"      doc_id: {sample.doc_id}")
        print(f"      page_id: {sample.page_id}")
        print(f"      block_id: {sample.block_id or 'N/A'}")
        print(f"      score: {sample.score:.4f}")
        print(f"      snippet length: {len(sample.snippet) if sample.snippet else 0}")
        
        # Block_id should exist for block-level evidence
        has_block_id = bool(sample.block_id)
        print(f"      Has block_id: {has_block_id}")
    
    # Check 4: Answer quality
    if result.generation:
        answer = result.generation.output.answer
        print(f"\n   📄 Answer preview:")
        print(f"      Length: {len(answer)} chars")
        print(f"      Citations: {len(result.generation.output.cited_units)} units")
        print(f"      Preview: {answer[:150]}...")
    
    print(f"\n{'✅' if contract_ok else '❌'} Contract validation: {'PASSED' if contract_ok else 'FAILED'}")
    
    return contract_ok


def main():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    store = DocumentStoreLocal(config)
    logger = RunLoggerLocal(config)
    
    print("=" * 60)
    print("Testing Unified Retrieval Contract")
    print("=" * 60)
    print("\nContract requirements:")
    print("  1. All hits must have text content")
    print("  2. All evidence must have non-empty snippet")
    print("  3. Evidence schema: doc_id, page_id, block_id, score, snippet")
    
    results = {}
    
    # Test BM25
    try:
        print("\n\nLoading BM25...")
        retriever = BM25IndexerRetriever(store)
        retriever.load(config, index_name="bm25_default")
        results["bm25"] = test_mode("bm25", retriever, config, store, logger)
    except Exception as e:
        print(f"\n❌ BM25 test failed: {e}")
        results["bm25"] = False
    
    # Test Dense
    if config.dense.get("enabled"):
        try:
            print("\n\nLoading Dense...")
            embedder = VLLMEmbedder(
                endpoint=config.dense["endpoint"],
                model=config.dense["model"]
            )
            retriever = DenseIndexerRetriever.load(Path("data/indices/dense_default"), embedder)
            results["dense"] = test_mode("dense", retriever, config, store, logger)
        except Exception as e:
            print(f"\n❌ Dense test failed: {e}")
            import traceback
            traceback.print_exc()
            results["dense"] = False
    
    # Test ColPali
    if config.colpali.get("enabled"):
        try:
            print("\n\nLoading ColPali...")
            retriever = ColPaliRetriever.load(
                Path("data/indices/colpali_default"),
                model_name=config.colpali["model"],
                device=config.colpali.get("device", "cuda:2")
            )
            results["colpali"] = test_mode("colpali", retriever, config, store, logger)
        except Exception as e:
            print(f"\n❌ ColPali test failed: {e}")
            import traceback
            traceback.print_exc()
            results["colpali"] = False
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for mode, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {mode.upper():12} {status}")
    
    all_passed = all(results.values())
    print(f"\n{'🎉' if all_passed else '⚠️ '} Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
