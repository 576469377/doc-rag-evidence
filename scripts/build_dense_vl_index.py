#!/usr/bin/env python3
"""
Build Dense-VL index using Qwen3VLEmbedder (offline mode).
This script loads the model directly and builds embeddings for all pages.
Supports parallel processing with multiple model instances on same GPU.
"""
import sys
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from multiprocessing import Pool, cpu_count

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Qwen3VL embedding model from cloned repository
try:
    # Add Qwen3-VL-Embedding repository to path (relative to project root)
    project_root = Path(__file__).parent.parent
    qwen_vl_repo = project_root / "Qwen3-VL-Embedding"
    if qwen_vl_repo.exists():
        sys.path.insert(0, str(qwen_vl_repo))
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        print(f"✅ Imported Qwen3VLEmbedder from {qwen_vl_repo}")
    else:
        raise ImportError(f"Qwen3-VL-Embedding repository not found at {qwen_vl_repo}")
except ImportError as e:
    print(f"⚠️  Could not import Qwen3VLEmbedder: {e}")
    print("   Please clone the repository:")
    print("   cd doc-rag-evidence && git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git")
    Qwen3VLEmbedder = None

from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.index_tracker import IndexTracker


def process_batch_worker(batch_data, model_path, gpu, max_image_size, worker_id):
    """
    Worker function to process a batch of pages.
    Each worker loads its own model instance.
    
    Args:
        batch_data: List of (doc_id, page_id, image_path, text) tuples
        model_path: Path to model
        gpu: GPU ID
        max_image_size: Max image size
        worker_id: Worker ID for tracking
        
    Returns:
        np.ndarray: Embeddings for the batch
    """
    # Each worker gets its own embedder instance
    embedder = OfflineVLEmbedder(
        model_path=model_path,
        gpu=gpu,
        gpu_memory=0.45,
        max_image_size=max_image_size
    )
    
    # Process pages one by one with progress bar for this worker
    all_embeddings = []
    pbar = None
    try:
        pbar = tqdm(total=len(batch_data), 
                    desc=f"Worker {worker_id}", 
                    position=worker_id,
                    leave=True,
                    unit="page")
        for doc_id, page_id, image_path, text in batch_data:
            # Prepare single input
            inputs = [(text, image_path)]
            
            # Get embedding for single page
            embedding = embedder.embed_batch(inputs)
            all_embeddings.append(embedding)
            
            # Update progress
            pbar.update(1)
    finally:
        # Explicitly close progress bar to clean up resources
        if pbar is not None:
            pbar.close()
    
    # Combine all embeddings
    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = np.array([])
    
    # Clean up
    del embedder
    torch.cuda.empty_cache()
    
    return embeddings


class OfflineVLEmbedder:
    """Offline VL embedder using Qwen3VLEmbedder."""
    
    def __init__(self, model_path: str, gpu: int = 1, gpu_memory: float = 0.45, max_image_size: int = None):
        """
        Initialize Qwen3VL embedding model.
        
        Args:
            model_path: Path to model
            gpu: GPU ID
            gpu_memory: GPU memory fraction
            max_image_size: Max image size (pixels) on longest side. If None, use original size.
                           Recommended: 1024 for faster indexing, 2048 for better quality.
        """
        self.max_image_size = max_image_size
        print(f"Loading Qwen3VL Embedding model: {model_path}")
        print(f"Original GPU ID: {gpu}")
        if max_image_size:
            print(f"Image resize: max {max_image_size}px on longest side")
        
        # Set GPU device via environment variable (safer for multiprocessing)
        # Note: If CUDA_VISIBLE_DEVICES is set externally, use device 0
        if torch.cuda.is_available():
            # Check if CUDA_VISIBLE_DEVICES is set
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                # When CUDA_VISIBLE_DEVICES is set, the visible GPU is mapped to device 0
                device = "cuda:0"
                print(f"Using device 0 (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
            else:
                device = f"cuda:{gpu}"
                # Don't use torch.cuda.set_device in multiprocessing context
                # Set CUDA_VISIBLE_DEVICES instead if needed
                print(f"Using device {gpu}")
        else:
            device = "cpu"
            print("⚠️  CUDA not available, using CPU")
        
        if Qwen3VLEmbedder is None:
            raise RuntimeError(
                "Qwen3VLEmbedder not available. Please clone the repository:\n"
                "cd /workspace && git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git"
            )
        
        # Use official Qwen3VLEmbedder with Flash Attention 2
        try:
            self.model = Qwen3VLEmbedder(
                model_name_or_path=model_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2",
            )
            print("  ✅ Using Flash Attention 2 (fast)")
        except Exception as e:
            # Fallback to standard attention if flash-attn not available
            if "flash_attn" in str(e).lower() or "flash" in str(e).lower():
                print("  ⚠️  Flash Attention 2 not available, using standard attention (slower)")
                print("     To enable: pip install flash-attn --no-build-isolation")
                self.model = Qwen3VLEmbedder(
                    model_name_or_path=model_path,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
            else:
                raise
        
        self.device = device
        print(f"✅ Model loaded on {device}")
    
    def _resize_image_if_needed(self, image_path: str) -> str:
        """
        Resize image if it's too large.
        
        Args:
            image_path: Original image path
            
        Returns:
            Path to (possibly resized) image
        """
        if not self.max_image_size:
            return image_path
        
        try:
            img = Image.open(image_path)
            w, h = img.size
            max_dim = max(w, h)
            
            # No need to resize if already small enough
            if max_dim <= self.max_image_size:
                return image_path
            
            # Calculate new size maintaining aspect ratio
            scale = self.max_image_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize and save to temp file
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create temp file in same directory
            temp_path = Path(image_path).parent / f"temp_resized_{self.max_image_size}.jpg"
            img_resized.save(str(temp_path), quality=95)
            
            return str(temp_path)
        except Exception as e:
            print(f"Warning: Failed to resize {image_path}: {e}")
            return image_path
    
    def embed_batch(self, inputs: list) -> np.ndarray:
        """
        Embed a batch of inputs.
        
        Args:
            inputs: List of (text, image_path) tuples
            
        Returns:
            Embeddings array (N, D)
        """
        # Prepare inputs in Qwen3VL format
        qwen_inputs = []
        temp_files = []  # Track temp files for cleanup
        
        for text, image_path in inputs:
            input_dict = {}
            
            if image_path and os.path.exists(image_path):
                # Resize if needed
                processed_path = self._resize_image_if_needed(image_path)
                input_dict["image"] = processed_path
                
                # Track temp file for cleanup
                if processed_path != image_path:
                    temp_files.append(processed_path)
            
            if text and text.strip():
                input_dict["text"] = text.strip()
            
            # If both are empty, add empty text
            if not input_dict:
                input_dict["text"] = ""
            
            qwen_inputs.append(input_dict)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.process(qwen_inputs)
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Convert to numpy (convert to float32 first to handle BFloat16)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        
        return embeddings.astype(np.float32)


def build_dense_vl_index(
    config_path: str = "configs/app.yaml",
    index_name: str = "dense_vl_default",
    force_rebuild: bool = False,
    max_image_size: int = 1024,
    num_workers: int = 1
):
    """
    Build Dense-VL index using offline embedding.
    
    Args:
        config_path: Path to config file
        index_name: Name of index
        force_rebuild: Whether to rebuild from scratch
        max_image_size: Max image size (pixels) on longest side.
                       1024 = fast but lower quality
                       2048 = slower but better quality
                       None = original size (slowest)
        num_workers: Number of parallel workers (model instances).
                    Each worker uses ~5GB GPU memory.
                    Recommended: 4 for 24GB GPU, 2 for 12GB GPU.
    """
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    # Initialize store
    store = DocumentStoreLocal(config)
    
    # Get documents
    docs = store.list_documents()
    if not docs:
        print("❌ No documents found")
        return
    
    print(f"📚 Found {len(docs)} document(s)")
    
    # Check what needs indexing
    index_dir = Path(config.indices_dir) / index_name
    tracker = IndexTracker(index_dir)
    
    indexed_docs = tracker.get_indexed_docs()
    all_doc_ids = {doc.doc_id for doc in docs}
    new_doc_ids = sorted(list(all_doc_ids - indexed_docs))
    
    if not new_doc_ids and not force_rebuild:
        print("ℹ️  All documents already indexed")
        return
    
    if force_rebuild:
        print("🔄 Force rebuild mode")
        doc_ids_to_index = [doc.doc_id for doc in docs]
        existing_page_images = []
    else:
        print(f"📝 New documents: {len(new_doc_ids)}")
        doc_ids_to_index = new_doc_ids
        
        # Load existing index if present
        existing_page_images = []
        if index_dir.exists():
            print("📚 Loading existing index...")
            meta_path = index_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                for (doc_id, page_id) in metadata["page_ids"]:
                    page_key = f"{doc_id}__page{page_id:04d}"
                    meta = metadata["page_metadata"][page_key]
                    existing_page_images.append((
                        doc_id,
                        page_id,
                        meta["image_path"],
                        meta.get("text", "")
                    ))
                print(f"   Loaded {len(existing_page_images)} existing pages")
    
    # Collect new page images
    new_page_images = []
    for doc_id in doc_ids_to_index:
        doc = store.get_document(doc_id)
        
        for page_id in range(doc.page_count):
            # Get page image
            page_image_path = Path(config.docs_dir) / doc_id / "pages" / f"{page_id:04d}" / "page.png"
            if not page_image_path.exists():
                print(f"  ⚠️  Missing image: {doc_id} page {page_id}")
                continue
            
            # Get page text
            page_artifact = store.load_page_artifact(doc_id, page_id)
            page_text = ""
            if page_artifact and page_artifact.text:
                page_text = page_artifact.text.text
            
            new_page_images.append((doc_id, page_id, str(page_image_path), page_text))
        
        # Update tracker
        tracker.mark_indexed(doc_id, doc.page_count, doc.page_count)
        print(f"  + {doc_id}: {doc.page_count} pages")
    
    if not new_page_images:
        print("❌ No pages to index")
        return
    
    # Combine old and new
    all_page_images = existing_page_images + new_page_images
    print(f"🔨 Total pages to embed: {len(all_page_images)}")
    
    # Get config
    model_path = config.dense_vl["model_path"]
    gpu = config.dense_vl.get("gpu", 1)
    
    # Set CUDA_VISIBLE_DEVICES for multiprocessing (must be set before spawning workers)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"🎯 Set CUDA_VISIBLE_DEVICES={gpu} (all workers will use GPU {gpu})")
    
    # Parallel processing
    if num_workers > 1:
        print(f"⚡ Using {num_workers} parallel workers")
        print(f"   Each worker loads one model instance (~5GB GPU memory)")
        print(f"   Expected GPU usage: ~{num_workers * 5}GB")
        
        # Split pages into chunks for workers
        chunk_size = max(1, len(all_page_images) // num_workers)
        chunks = [all_page_images[i:i+chunk_size] 
                 for i in range(0, len(all_page_images), chunk_size)]
        
        print(f"   Split into {len(chunks)} chunks of ~{chunk_size} pages each")
        print(f"\n📊 Progress (each worker shows page-level progress):\n")
        
        # Process in parallel - each worker shows its own progress bar
        all_embeddings = []
        pool = None
        try:
            pool = Pool(processes=num_workers)
            # Prepare arguments for each worker: (chunk, model_path, gpu, max_image_size, worker_id)
            worker_args = [(chunk, model_path, gpu, max_image_size, i) 
                          for i, chunk in enumerate(chunks)]
            
            # Use starmap to pass multiple arguments - workers will show their own progress
            for embeddings in pool.starmap(process_batch_worker, worker_args):
                all_embeddings.append(embeddings)
        finally:
            # Explicitly close and join pool to clean up resources
            if pool is not None:
                pool.close()
                pool.join()
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        print(f"✅ Embeddings shape: {all_embeddings.shape}")
        
    else:
        # Single worker (original sequential processing)
        print("📄 Using single worker (sequential processing)")
        
        embedder_sdk = OfflineVLEmbedder(
            model_path=model_path,
            gpu=gpu,
            gpu_memory=0.45,
            max_image_size=max_image_size
        )
        
        # Build embeddings in batches
        print("Embedding pages...")
        batch_size = 8
        all_embeddings = []
        
        with tqdm(total=len(all_page_images), desc="📄 Processing pages", unit="page", 
                  file=sys.stdout, dynamic_ncols=True, mininterval=0.5) as pbar:
            for i in range(0, len(all_page_images), batch_size):
                batch = all_page_images[i:i+batch_size]
                
                # Prepare inputs
                inputs = [(text, image_path) for _, _, image_path, text in batch]
                
                # Embed
                embeddings = embedder_sdk.embed_batch(inputs)
                all_embeddings.append(embeddings)
                
                # Update progress
                pbar.update(len(batch))
        
        # Combine embeddings
        all_embeddings = np.vstack(all_embeddings)
        print(f"✅ Embeddings shape: {all_embeddings.shape}")
    
    # Build FAISS index
    print("Building FAISS index...")
    import faiss
    
    embed_dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    print(f"✅ FAISS index built: {index.ntotal} vectors")
    
    # Save index
    print(f"Saving to {index_dir}...")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(index_dir / "dense_vl.index"))
    
    # Save metadata
    page_ids = [(doc_id, page_id) for doc_id, page_id, _, _ in all_page_images]
    page_metadata = {}
    for doc_id, page_id, image_path, text in all_page_images:
        page_key = f"{doc_id}__page{page_id:04d}"
        page_metadata[page_key] = {
            "doc_id": doc_id,
            "page_id": page_id,
            "image_path": image_path,
            "text": text
        }
    
    metadata = {
        "index_type": "Flat",
        "nlist": 100,
        "nprobe": 10,
        "embed_dim": embed_dim,
        "num_pages": len(page_ids),
        "page_ids": page_ids,
        "page_metadata": page_metadata
    }
    
    with open(index_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    tracker.save()
    
    print(f"✅ Index saved to {index_dir}")
    print(f"   Total pages: {len(page_ids)}")
    print(f"   Embedding dim: {embed_dim}")


if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    # Set start method to 'spawn' for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description="Build Dense-VL index (offline)")
    parser.add_argument("--config", default="configs/app.yaml", help="Config file")
    parser.add_argument("--index-name", default="dense_vl_default", help="Index name")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--max-image-size", type=int, default=None,
                       help="Max image size (px) on longest side. Default: from config or 1024")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers. Each uses ~5GB GPU. Default: from config or 1")
    
    args = parser.parse_args()
    
    # Load config to get defaults
    import yaml
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    max_image_size = args.max_image_size
    if max_image_size is None:
        max_image_size = config_dict.get('dense_vl', {}).get('max_image_size', 1024)
    
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = config_dict.get('dense_vl', {}).get('num_workers', 1)
    
    build_dense_vl_index(
        config_path=args.config,
        index_name=args.index_name,
        force_rebuild=args.force_rebuild,
        max_image_size=max_image_size,
        num_workers=num_workers
    )
