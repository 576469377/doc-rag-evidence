#!/usr/bin/env python3
# scripts/demo_eval.py
"""
Demo script for batch evaluation.

Usage:
    python scripts/demo_eval.py dataset.csv [--config configs/app.yaml]
    python scripts/demo_eval.py dataset.json [--config configs/app.yaml]
"""
import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import AppConfig
from core.pipeline import Pipeline
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator
from impl.eval_runner import EvalRunner, load_dataset_from_csv, load_dataset_from_json


def main():
    parser = argparse.ArgumentParser(description="Demo: Batch evaluation")
    parser.add_argument("dataset", help="Path to dataset file (CSV or JSON)")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    print("=" * 60)
    print("Doc RAG Evidence Demo - Batch Evaluation")
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

    # Create eval runner
    eval_runner = EvalRunner(pipeline)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return

    print(f"\n📊 Loading dataset: {args.dataset}")
    if dataset_path.suffix.lower() == ".csv":
        dataset = load_dataset_from_csv(str(dataset_path))
    elif dataset_path.suffix.lower() == ".json":
        dataset = load_dataset_from_json(str(dataset_path))
    else:
        print("❌ Unsupported file format. Use .csv or .json")
        return

    print(f"✅ Loaded: {dataset.name}")
    print(f"   Questions: {len(dataset.items)}")

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60 + "\n")

    report = eval_runner.evaluate(dataset, config)

    # Display summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nDataset: {report.dataset_name}")
    print(f"Total questions: {report.metrics.n}")
    print(f"Success rate: {report.metrics.extra['success_rate']:.2%}")
    print(f"Average latency: {report.metrics.avg_latency_ms:.0f}ms")

    # Show failed cases
    failed_rows = [row for row in report.rows if not row.status_ok]
    if failed_rows:
        print(f"\n⚠️  Failed cases ({len(failed_rows)}):")
        for row in failed_rows[:5]:  # Show first 5
            print(f"  - {row.qid}: {row.error_type}")

    # Artifacts
    print("\n📁 Output files:")
    for key, path in report.artifact_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
