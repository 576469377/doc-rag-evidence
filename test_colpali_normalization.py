#!/usr/bin/env python3
"""Test ColPali with hit normalization (page → block expansion)."""
import sys
import yaml
from pathlib import Path

from core.schemas import AppConfig, QueryInput
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
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
    
    print("=" * 60)
    print("Testing ColPali with Hit Normalization")
    print("=" * 60)
    
    # Load ColPali retriever
    print("\nLoading ColPali retriever...")
    retriever = ColPaliRetriever.load(
        Path("data/indices/colpali_default"),
        model_name=config.colpali["model"],
        device=config.colpali.get("device", "cuda:2")
    )
    print(f"✅ Loaded ColPali ({len(retriever.store.page_ids) if retriever.store else 0} pages)")
    
    # Create pipeline WITH store for normalization
    selector = TopKEvidenceSelector(snippet_length=500)
    generator = TemplateGenerator(mode="summary")
    
    pipeline = Pipeline(
        retriever=retriever,
        selector=selector,
        generator=generator,
        logger=logger,
        reranker=None,
        store=store  # Enable hit normalization
    )
    
    # Test query
    query_input = QueryInput(
        query_id="colpali_norm_test_001",
        question="碳酸钙的使用范围",
        doc_filter=[]
    )
    
    print(f"\nQuery: {query_input.question}")
    print("\nRunning pipeline with normalization enabled...")
    
    # Run full pipeline
    result = pipeline.answer(query_input, config)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    # Check retrieval
    if result.retrieval:
        print(f"\n✅ Retrieval: {len(result.retrieval.hits)} hits")
        print(f"   First 3 hits:")
        for i, hit in enumerate(result.retrieval.hits[:3], 1):
            has_text = bool(hit.text and hit.text.strip())
            has_block = bool(hit.block_id)
            print(f"   {i}. Doc: {hit.doc_id}, Page: {hit.page_id}, Block: {hit.block_id or 'N/A'}")
            print(f"      Score: {hit.score:.4f}, Source: {hit.source}")
            print(f"      Has text: {has_text}, Has block_id: {has_block}")
            if hit.text:
                snippet = hit.text[:80].replace('\n', ' ')
                print(f"      Text: {snippet}...")
            if "expanded_from" in hit.metadata:
                print(f"      ⭐ Expanded from: {hit.metadata['expanded_from']}")
    
    # Check evidence
    if result.evidence and result.evidence.evidence:
        print(f"\n✅ Evidence: {len(result.evidence.evidence)} items")
        print(f"   Top 3 evidence items:")
        for i, ev in enumerate(result.evidence.evidence[:3], 1):
            print(f"   {i}. Doc: {ev.doc_id}, Page: {ev.page_id}, Block: {ev.block_id or 'N/A'}")
            print(f"      Score: {ev.score:.4f}")
            if ev.snippet:
                snippet = ev.snippet[:100].replace('\n', ' ')
                print(f"      Snippet: {snippet}...")
    
    # Check answer
    if result.generation:
        answer = result.generation.output.answer
        print(f"\n✅ Answer ({len(answer)} chars):")
        print(f"   {answer[:300]}...")
        
        if result.generation.output.cited_units:
            print(f"\n   Cited units: {result.generation.output.cited_units}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    # Check DoD criteria
    print("\n📋 DoD Checklist:")
    
    evidence_has_snippet = (
        result.evidence and 
        result.evidence.evidence and 
        all(ev.snippet and ev.snippet.strip() for ev in result.evidence.evidence)
    )
    print(f"   {'✅' if evidence_has_snippet else '❌'} Evidence has non-empty snippets")
    
    evidence_has_block = (
        result.evidence and 
        result.evidence.evidence and 
        any(ev.block_id for ev in result.evidence.evidence)
    )
    print(f"   {'✅' if evidence_has_block else '❌'} Evidence includes block_id")
    
    answer_readable = (
        result.generation and 
        len(result.generation.output.answer) > 50 and
        not result.generation.output.answer.strip().startswith("[")
    )
    print(f"   {'✅' if answer_readable else '❌'} Answer is readable (not just citations)")


if __name__ == "__main__":
    main()
