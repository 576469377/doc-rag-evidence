#!/usr/bin/env python3
# scripts/demo_run.py
"""
Demo script for single question answering.

Usage:
    python scripts/demo_run.py "What is the main topic?" [--config configs/app.yaml]
"""
import argparse
import sys
import uuid
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import AppConfig, QueryInput
from core.pipeline import Pipeline
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator


def main():
    parser = argparse.ArgumentParser(description="Demo: Ask a question")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--doc-filter",
        nargs="+",
        help="Filter by document IDs"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    print("=" * 60)
    print("Doc RAG Evidence Demo - Single Query")
    print("=" * 60)

    # Initialize components
    store = DocumentStoreLocal(config)
    logger = RunLoggerLocal(config)
    indexer = BM25IndexerRetriever(store)
    selector = TopKEvidenceSelector(snippet_length=500)
    generator = TemplateGenerator(mode="summary")

    # Load index
    print("\n📚 Loading index...")
    if not indexer.load(config, index_name="bm25_default"):
        print("❌ Index not found. Please run build_index.py first.")
        return
    print(f"✅ Index loaded: {len(indexer.units)} units")

    # Create pipeline
    pipeline = Pipeline(
        retriever=indexer,
        selector=selector,
        generator=generator,
        logger=logger,
        reranker=None
    )

    # Create query
    query_id = f"demo_{uuid.uuid4().hex[:8]}"
    query = QueryInput(
        query_id=query_id,
        question=args.question,
        doc_filter=args.doc_filter
    )

    print(f"\n❓ Question: {args.question}")
    if args.doc_filter:
        print(f"   Filter: {args.doc_filter}")

    # Run pipeline
    print("\n🔍 Running pipeline...")
    record = pipeline.answer(query, config)

    # Display results
    if not record.status.ok:
        print(f"\n❌ Error: {record.status.error_message}")
        if record.status.stack:
            print(f"\nStack trace:\n{record.status.stack}")
        return

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Answer
    if record.generation and record.generation.output:
        print(f"\n📝 Answer:\n{record.generation.output.answer}\n")
        print(f"Citations: {record.generation.output.cited_units}")

    # Evidence
    if record.evidence and record.evidence.evidence:
        print(f"\n📑 Evidence ({len(record.evidence.evidence)} items):")
        for ev in record.evidence.evidence:
            print(f"\n  [{ev.rank + 1}] {ev.unit_id}")
            print(f"      Doc: {ev.doc_id}, Page: {ev.page_id}")
            print(f"      Score: {ev.score:.4f}")
            print(f"      Snippet: {ev.snippet[:150]}...")

    # Traceability
    print(f"\n🔗 Query ID: {query_id}")
    print(f"   Run log: {config.runs_dir}/{query_id}.json")

    # Timing
    if record.retrieval:
        print(f"\n⏱️  Timing:")
        print(f"   Retrieval: {record.retrieval.elapsed_ms}ms")
        if record.evidence:
            print(f"   Evidence selection: {record.evidence.elapsed_ms}ms")
        if record.generation:
            print(f"   Generation: {record.generation.elapsed_ms}ms")


if __name__ == "__main__":
    main()
