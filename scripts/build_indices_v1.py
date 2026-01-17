#!/usr/bin/env python3
"""
Build indices for all retrieval modes (BM25, Dense, ColPali).
Supports selective building via command-line flags.
"""
import argparse
from pathlib import Path
import yaml
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import AppConfig, IndexUnit
from infra.store_local import DocumentStoreLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, SGLangEmbedder
from impl.index_colpali import ColPaliRetriever
from impl.index_dense_vl import VLEmbedder, DenseVLRetriever


def load_config(config_path: str) -> AppConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)


def build_index_units(store: DocumentStoreLocal, chunk_level: str) -> list[IndexUnit]:
    """
    Build index units from all documents in store.
    
    Args:
        store: Document store
        chunk_level: "page" or "block"
        
    Returns:
        List of IndexUnit objects
    """
    units = []
    
    # List all documents
    docs = store.list_documents()
    print(f"Found {len(docs)} documents")
    
    for meta in docs:
        doc_id = meta.doc_id
        
        # Process each page
        for page_id in range(meta.page_count):
            artifact = store.load_page_artifact(doc_id, page_id)
            if not artifact:
                continue
            
            if chunk_level == "page":
                # Page-level unit
                unit = IndexUnit(
                    unit_id=f"{doc_id}__page{page_id:04d}",
                    doc_id=doc_id,
                    page_id=page_id,
                    block_id=None,
                    text=artifact.text.text if artifact.text else "",
                    bbox=None
                )
                units.append(unit)
            
            elif chunk_level == "block":
                # Block-level units
                for block in artifact.blocks:
                    unit = IndexUnit(
                        unit_id=block.block_id,
                        doc_id=doc_id,
                        page_id=page_id,
                        block_id=block.block_id,
                        text=block.text,
                        bbox=block.bbox
                    )
                    units.append(unit)
    
    print(f"Built {len(units)} index units (chunk_level={chunk_level})")
    return units


def build_bm25_index(config: AppConfig, store: DocumentStoreLocal):
    """Build BM25 index."""
    print("\n=== Building BM25 Index ===")
    
    # Build units
    units = build_index_units(store, config.chunk_level)
    
    # Create indexer
    indexer = BM25IndexerRetriever(store)
    stats = indexer.build_index(units, config)
    
    # Persist
    indexer.persist(config, index_name="bm25_default")
    
    index_dir = Path(config.indices_dir) / "bm25_default"
    print(f"BM25 index saved to {index_dir}")
    print(f"Stats: {stats.unit_count} units indexed")


def build_dense_index(config: AppConfig, store: DocumentStoreLocal):
    """Build dense embedding index."""
    print("\n=== Building Dense Embedding Index ===")
    
    dense_config = config.dense
    if not dense_config.get("enabled"):
        print("Dense retrieval not enabled in config. Set dense.enabled=true")
        return
    
    # Build units
    units = build_index_units(store, config.chunk_level)
    
    # Create embedder
    embedder = SGLangEmbedder(
        endpoint=dense_config["endpoint"],
        model=dense_config["model"],
        batch_size=dense_config.get("batch_size", 32)
    )
    
    # Create indexer
    indexer = DenseIndexerRetriever(
        embedder=embedder,
        index_type=dense_config.get("index_type", "Flat"),
        nlist=dense_config.get("nlist", 100),
        nprobe=dense_config.get("nprobe", 10)
    )
    
    indexer.build_units(units)
    indexer.build_index()
    
    # Persist
    index_dir = Path(config.indices_dir) / "dense"
    indexer.persist(index_dir)
    
    print(f"Dense index saved to {index_dir}")


def build_colpali_index(config: AppConfig, store: DocumentStoreLocal):
    """Build ColPali vision index."""
    print("\n=== Building ColPali Vision Index ===")
    
    colpali_config = config.colpali
    if not colpali_config.get("enabled"):
        print("ColPali retrieval not enabled in config. Set colpali.enabled=true")
        return
    
    # Collect page images
    page_images = []
    docs = store.list_documents()
    
    for meta in docs:
        doc_id = meta.doc_id
        
        for page_id in range(meta.page_count):
            artifact = store.load_page_artifact(doc_id, page_id)
            if artifact and artifact.image_path:
                page_images.append((doc_id, page_id, artifact.image_path))
    
    print(f"Found {len(page_images)} page images")
    
    if not page_images:
        print("No page images found. Run ingestion with page rendering first.")
        return
    
    # Create retriever
    retriever = ColPaliRetriever(
        model_name=colpali_config["model"],
        device=colpali_config.get("device", "cuda:0"),
        max_global_pool_pages=colpali_config.get("max_global_pool", 100),
        cache_dir=Path(colpali_config.get("cache_dir", "data/cache/colpali"))
    )
    
    # Build index
    retriever.build_index(page_images)
    
    # Persist
    index_dir = Path(config.indices_dir) / "colpali"
    retriever.persist(index_dir)
    
    print(f"ColPali index saved to {index_dir}")


def build_dense_vl_index(config: AppConfig, store: DocumentStoreLocal):
    """Build dense-vl multimodal embedding index."""
    print("\n=== Building Dense-VL Multimodal Index ===")
    
    dense_vl_config = config.dense_vl
    if not dense_vl_config.get("enabled"):
        print("Dense-VL retrieval not enabled in config. Set dense_vl.enabled=true")
        return
    
    # Collect page images with captions
    page_images = []
    docs = store.list_documents()
    
    for meta in docs:
        doc_id = meta.doc_id
        
        for page_id in range(meta.page_count):
            artifact = store.load_page_artifact(doc_id, page_id)
            if artifact and artifact.image_path:
                # Use page text as caption
                caption = artifact.text.text if artifact.text else ""
                page_images.append((doc_id, page_id, artifact.image_path, caption))
    
    print(f"Found {len(page_images)} page images")
    
    if not page_images:
        print("No page images found. Run ingestion with page rendering first.")
        return
    
    # Create embedder
    embedder = VLEmbedder(
        endpoint=dense_vl_config["endpoint"],
        model=dense_vl_config["model"],
        batch_size=dense_vl_config.get("batch_size", 16)
    )
    
    # Create retriever
    retriever = DenseVLRetriever(
        embedder=embedder,
        index_type=dense_vl_config.get("index_type", "Flat"),
        nlist=dense_vl_config.get("nlist", 100),
        nprobe=dense_vl_config.get("nprobe", 10)
    )
    
    # Build index
    retriever.build_index(page_images)
    
    # Persist
    index_dir = Path(config.indices_dir) / "dense_vl"
    retriever.persist(index_dir)
    
    print(f"Dense-VL index saved to {index_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/app.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="Build BM25 index"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Build dense embedding index"
    )
    parser.add_argument(
        "--colpali",
        action="store_true",
        help="Build ColPali vision index"
    )
    parser.add_argument(
        "--dense-vl",
        action="store_true",
        help="Build Dense-VL multimodal index"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all enabled indices"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize store
    store = DocumentStoreLocal(config)
    
    # Determine what to build
    build_all = args.all or not (args.bm25 or args.dense or args.colpali or args.dense_vl)
    
    try:
        if build_all or args.bm25:
            build_bm25_index(config, store)
        
        if build_all or args.dense:
            if config.dense.get("enabled") or args.dense:
                build_dense_index(config, store)
            else:
                print("\nSkipping dense index (not enabled)")
        
        if build_all or args.colpali:
            if config.colpali.get("enabled") or args.colpali:
                build_colpali_index(config, store)
            else:
                print("\nSkipping ColPali index (not enabled)")
        
        if build_all or args.dense_vl:
            if config.dense_vl.get("enabled") or args.dense_vl:
                build_dense_vl_index(config, store)
            else:
                print("\nSkipping Dense-VL index (not enabled)")
        
        print("\n=== Index building complete ===")
        
    except Exception as e:
        print(f"\nError building indices: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
