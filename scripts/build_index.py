#!/usr/bin/env python3
# scripts/build_index.py
"""
Build or rebuild indices from all ingested documents.
Supports: BM25, Dense (vLLM), ColPali (vision)

Usage:
    python scripts/build_index.py --types bm25,dense,colpali [--config configs/app.yaml]
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
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever


def build_bm25_index(config: AppConfig, store: DocumentStoreLocal, index_name: str = "bm25_default"):
    """Build BM25 index."""
    print("\n" + "=" * 60)
    print("Building BM25 Index")
    print("=" * 60)
    
    indexer = BM25IndexerRetriever(store)
    
    # Get all documents
    docs = store.list_documents()
    if not docs:
        print("⚠️  No documents found. Skipping BM25 index.")
        return
    
    print(f"\nFound {len(docs)} documents")
    
    # Build index units
    all_units = []
    for doc in docs:
        units = indexer.build_units(doc.doc_id, config)
        all_units.extend(units)
        print(f"  {doc.doc_id}: {len(units)} units")
    
    print(f"\nTotal units: {len(all_units)}")
    
    # Build index
    print("Building BM25 index...")
    stats = indexer.build_index(all_units, config)
    
    print(f"✅ BM25 index built!")
    print(f"  Documents: {stats.doc_count}")
    print(f"  Pages: {stats.page_count}")
    print(f"  Units: {stats.unit_count}")
    print(f"  Time: {stats.elapsed_ms}ms")
    
    # Save
    indexer.persist(config, index_name=index_name)
    print(f"✅ Saved to: {config.indices_dir}/{index_name}")


def build_dense_index(config: AppConfig, store: DocumentStoreLocal, index_name: str = "dense_default"):
    """Build Dense index using vLLM embeddings."""
    print("\n" + "=" * 60)
    print("Building Dense Index (vLLM Embedding)")
    print("=" * 60)
    
    if not config.dense.get("enabled"):
        print("⚠️  Dense retrieval not enabled in config. Skipping.")
        return
    
    # Initialize embedder
    embedder = VLLMEmbedder(
        endpoint=config.dense["endpoint"],
        model=config.dense["model"],
        batch_size=config.dense.get("batch_size", 32)
    )
    print(f"Embedder: {config.dense['model']} @ {config.dense['endpoint']}")
    
    indexer = DenseIndexerRetriever(
        embedder=embedder,
        index_type=config.dense.get("index_type", "Flat"),
        nlist=config.dense.get("nlist", 100),
        nprobe=config.dense.get("nprobe", 10)
    )
    
    # Get all documents
    docs = store.list_documents()
    if not docs:
        print("⚠️  No documents found. Skipping Dense index.")
        return
    
    print(f"\nFound {len(docs)} documents")
    
    # Build index units
    all_units = []
    for doc in docs:
        units = indexer.build_units(doc.doc_id, config)
        all_units.extend(units)
        print(f"  {doc.doc_id}: {len(units)} units")
    
    print(f"\nTotal units: {len(all_units)}")
    
    # Build index (this will call embedder API)
    print("Building Dense index (calling vLLM API for embeddings)...")
    indexer.build_index(all_units, config)
    
    print(f"✅ Dense index built!")
    print(f"  Units: {len(indexer.units)}")
    print(f"  Embedding dim: {indexer.index.d}")
    
    # Save
    index_dir = Path(config.indices_dir) / index_name
    indexer.save(index_dir)
    print(f"✅ Saved to: {index_dir}")


def build_colpali_index(config: AppConfig, store: DocumentStoreLocal, index_name: str = "colpali_default"):
    """Build ColPali vision index."""
    print("\n" + "=" * 60)
    print("Building ColPali Vision Index")
    print("=" * 60)
    
    if not config.colpali.get("enabled"):
        print("⚠️  ColPali retrieval not enabled in config. Skipping.")
        return
    
    # Initialize retriever
    device = config.colpali.get("device", "cuda:2")
    indexer = ColPaliRetriever(
        model_name=config.colpali["model"],
        device=device,
        max_global_pool_pages=config.colpali.get("max_global_pool", 100)
    )
    print(f"Model: {config.colpali['model']}")
    print(f"Device: {device}")
    
    # Get all documents and build page list
    docs = store.list_documents()
    if not docs:
        print("⚠️  No documents found. Skipping ColPali index.")
        return
    
    print(f"\nFound {len(docs)} documents")
    
    # Build page list (doc_id, page_id) tuples
    page_list = []
    for doc in docs:
        for page_id in range(doc.page_count):
            page_list.append((doc.doc_id, page_id))
    
    print(f"Total pages: {len(page_list)}")
    
    # Build index (this will load images and encode with ColQwen3)
    print("Building ColPali index (encoding page images, this may take a while)...")
    indexer.build_index(page_list, config)
    
    print(f"✅ ColPali index built!")
    print(f"  Pages: {len(indexer.store.page_ids)}")
    
    # Save
    index_dir = Path(config.indices_dir) / index_name
    indexer.save(index_dir)
    print(f"✅ Saved to: {index_dir}")


def main():
def main():
    parser = argparse.ArgumentParser(description="Build indices from ingested documents")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--types",
        default="bm25,dense,colpali",
        help="Comma-separated list of index types to build (bm25, dense, colpali)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    # Initialize store
    store = DocumentStoreLocal(config)
    
    # Get all documents
    docs = store.list_documents()
    print("=" * 60)
    print("Document Index Builder")
    print("=" * 60)
    print(f"\nAvailable documents ({len(docs)}):")
    for doc in docs:
        print(f"  - {doc.doc_id}: {doc.title} ({doc.page_count} pages)")

    if not docs:
        print("\n⚠️  No documents found. Please ingest documents first.")
        print("   Use: python scripts/demo_run.py --ingest <pdf_path>")
        return
    
    # Parse types
    index_types = [t.strip().lower() for t in args.types.split(",")]
    
    # Build requested indices
    if "bm25" in index_types:
        build_bm25_index(config, store)
    
    if "dense" in index_types:
        build_dense_index(config, store)
    
    if "colpali" in index_types:
        build_colpali_index(config, store)
    
    print("\n" + "=" * 60)
    print("Index Building Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

