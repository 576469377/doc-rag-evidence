#!/usr/bin/env python3
# scripts/build_index.py
"""
Build or rebuild BM25 index from all ingested documents.

Usage:
    python scripts/build_index.py [--config configs/app.yaml]
"""
import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.index_bm25 import BM25IndexerRetriever


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from documents")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--index-name",
        default="bm25_default",
        help="Index name"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    print("=" * 60)
    print("Building BM25 Index")
    print("=" * 60)

    # Initialize components
    store = DocumentStoreLocal(config)
    indexer = BM25IndexerRetriever(store)

    # Get all documents
    docs = store.list_documents()
    print(f"\nFound {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.doc_id}: {doc.title} ({doc.page_count} pages)")

    if not docs:
        print("\n⚠️  No documents found. Please ingest documents first.")
        return

    # Build index units from all documents
    print("\nBuilding index units...")
    all_units = []
    for doc in docs:
        units = indexer.build_units(doc.doc_id, config)
        all_units.extend(units)
        print(f"  {doc.doc_id}: {len(units)} units")

    print(f"\nTotal units: {len(all_units)}")

    # Build index
    print("\nBuilding BM25 index...")
    stats = indexer.build_index(all_units, config)

    print(f"\n✅ Index built successfully!")
    print(f"  Documents: {stats.doc_count}")
    print(f"  Pages: {stats.page_count}")
    print(f"  Index units: {stats.unit_count}")
    print(f"  Time: {stats.elapsed_ms}ms")

    # Persist index
    print(f"\nPersisting index to: {config.indices_dir}/{args.index_name}")
    indexer.persist(config, index_name=args.index_name)
    print("✅ Index saved!")


if __name__ == "__main__":
    main()
