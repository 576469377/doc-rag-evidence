"""
Dense multimodal (VL) embedding indexer and retriever using FAISS.
Uses Qwen3-VL-Embedding-2B with lazy loading via Qwen3VLEmbedder.
Each page is represented as one image embedding vector.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np
import os
import sys
import torch

try:
    import faiss
except ImportError:
    faiss = None

from PIL import Image

from core.schemas import RetrieveHit, QueryInput, AppConfig, RetrievalResult


class VLEmbedderLazy:
    """
    Lazy-loaded VL embedder using Qwen3VLEmbedder.
    Loads model on-demand instead of using API service.
    """
    
    def __init__(
        self,
        model_path: str,
        gpu: int = 1,
        gpu_memory: float = 0.45,
        batch_size: int = 8,
        max_image_size: int = None
    ):
        """
        Initialize lazy VL embedder.
        
        Args:
            model_path: Path to model checkpoint
            gpu: GPU index to use
            gpu_memory: GPU memory utilization (0-1, not used with new API)
            batch_size: Maximum batch size
            max_image_size: Max image size (pixels) on longest side. If None, use original.
        """
        self.model_path = model_path
        self.gpu = gpu
        self.gpu_memory = gpu_memory
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        self.model = None
    
    def _resize_image_if_needed(self, image_path: str) -> str:
        """Resize image if needed. Returns path to (possibly resized) image."""
        if not self.max_image_size:
            return image_path
        
        try:
            img = Image.open(image_path)
            w, h = img.size
            max_dim = max(w, h)
            
            if max_dim <= self.max_image_size:
                return image_path
            
            scale = self.max_image_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            temp_path = Path(image_path).parent / f"temp_resized_{self.max_image_size}.jpg"
            img_resized.save(str(temp_path), quality=95)
            
            return str(temp_path)
        except Exception as e:
            print(f"Warning: Failed to resize {image_path}: {e}")
            return image_path
        
    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self.model is None:
            print(f"⏳ Loading Dense-VL model from {self.model_path}...")
            
            # Set GPU device
            # Note: If CUDA_VISIBLE_DEVICES is set externally, use device 0
            if torch.cuda.is_available():
                # Check if CUDA_VISIBLE_DEVICES is set
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    # When CUDA_VISIBLE_DEVICES is set, the visible GPU is mapped to device 0
                    device = "cuda:0"
                    print(f"Using device 0 (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
                else:
                    device = f"cuda:{self.gpu}"
                    torch.cuda.set_device(self.gpu)
                    print(f"Using device {self.gpu}")
            else:
                device = "cpu"
                print("⚠️  CUDA not available, using CPU")
            
            # Try to import Qwen3VLEmbedder from cloned repository
            try:
                # Get project root and locate Qwen3-VL-Embedding
                project_root = Path(__file__).parent.parent
                qwen_vl_repo = project_root / "Qwen3-VL-Embedding"
                if qwen_vl_repo.exists():
                    sys.path.insert(0, str(qwen_vl_repo))
                    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
                    
                    # Try to use Flash Attention 2 for speedup
                    try:
                        self.model = Qwen3VLEmbedder(
                            model_name_or_path=self.model_path,
                            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            attn_implementation="flash_attention_2",
                        )
                        print("  ✅ Using Flash Attention 2 (fast)")
                    except Exception as e:
                        # Fallback to standard attention
                        if "flash_attn" in str(e).lower() or "flash" in str(e).lower():
                            print("  ⚠️  Flash Attention 2 not available, using standard attention (slower)")
                            print("     To enable: pip install flash-attn --no-build-isolation")
                            self.model = Qwen3VLEmbedder(
                                model_name_or_path=self.model_path,
                                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            )
                        else:
                            raise
                    print(f"✅ Dense-VL model loaded on {device}")
                else:
                    raise ImportError(f"Qwen3-VL-Embedding repository not found at {qwen_vl_repo}")
            except ImportError as e:
                raise RuntimeError(
                    f"Could not import Qwen3VLEmbedder: {e}\n"
                    "Please clone the repository:\n"
                    "cd doc-rag-evidence && git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git"
                )
    
    def embed(self, text: Optional[str] = None, image_path: Optional[str] = None) -> np.ndarray:
        """
        Embed single input.
        
        Args:
            text: Optional text content
            image_path: Optional path to image
            
        Returns:
            Embedding vector (numpy array)
        """
        return self.embed_batch([(text, image_path)])[0]
    
    def embed_batch(self, inputs: List[tuple]) -> np.ndarray:
        """
        Embed batch of inputs.
        
        Args:
            inputs: List of (text, image_path) tuples
            
        Returns:
            Batch of embeddings (numpy array, shape [N, D])
        """
        self._ensure_loaded()
        
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


class DenseVLRetrieverLazy:
    """
    Dense VL retriever with lazy loading support.
    Loads model on-demand using vLLM Python SDK.
    """
    
    def __init__(
        self,
        embedder: VLEmbedderLazy,
        index_type: str = "Flat",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize Dense VL retriever.
        
        Args:
            embedder: VL embedder (lazy-loaded)
            index_type: FAISS index type
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search
        """
        self.embedder = embedder
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.page_ids = []
        self.page_metadata = {}
    
    def retrieve(
        self,
        query_input: QueryInput,
        config: AppConfig
    ) -> RetrievalResult:
        """
        Retrieve relevant pages using multimodal embedding.
        
        Args:
            query_input: Query with text/image
            config: App configuration with top_k settings
            
        Returns:
            RetrievalResult with hits
        """
        if self.index is None or len(self.page_ids) == 0:
            return RetrievalResult(hits=[], metadata={"error": "Index not loaded"})
        
        # Get top_k from config
        top_k = config.top_k_retrieve
        
        # Embed query
        query_text = query_input.question if hasattr(query_input, 'question') else str(query_input)
        query_vec = self.embedder.embed(text=query_text, image_path=None)
        query_vec = query_vec.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vec, top_k)
        
        # Build hits
        hits = []
        index_size = self.index.ntotal  # FAISS index size
        page_ids_size = len(self.page_ids)  # page_ids list size

        # Debug: Check for size mismatch
        if index_size != page_ids_size:
            print(f"⚠️  Dense-VL index size mismatch: FAISS has {index_size} vectors, but page_ids has {page_ids_size} entries")

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue

            # Boundary check
            if idx >= page_ids_size or idx < 0:
                print(f"⚠️  Dense-VL: FAISS returned invalid index {idx}, page_ids size is {page_ids_size}")
                continue

            doc_id, page_id = self.page_ids[idx]
            
            # Apply doc filter from query if present
            if hasattr(query_input, 'doc_ids') and query_input.doc_ids:
                if doc_id not in query_input.doc_ids:
                    continue
            
            # Generate unit_id (page-level format for dense_vl)
            unit_id = f"p{doc_id}_p{page_id:04d}"
            
            hit = RetrieveHit(
                unit_id=unit_id,
                doc_id=doc_id,
                page_id=page_id,
                score=float(dist),
                text="",  # Will be populated by hit normalizer
                source="dense_vl",
                metadata=self.page_metadata.get(f"{doc_id}_{page_id}", {})
            )
            hits.append(hit)
        
        return RetrievalResult(
            query_id=query_input.query_id,
            hits=hits[:top_k],
            elapsed_ms=0
        )
    
    @classmethod
    def load(
        cls,
        index_dir: Path,
        model_path: str,
        gpu: int = 1,
        gpu_memory: float = 0.45,
        batch_size: int = 8,
        max_image_size: int = None
    ) -> "DenseVLRetrieverLazy":
        """
        Load index and prepare lazy model loading.

        Args:
            index_dir: Directory containing FAISS index
            model_path: Path to VL model checkpoint
            gpu: GPU index
            gpu_memory: GPU memory utilization
            batch_size: Batch size for embeddings
            max_image_size: Max image size (px) on longest side

        Returns:
            DenseVLRetrieverLazy instance with loaded index
        """
        import os
        index_dir = Path(index_dir)

        if not faiss:
            raise ImportError("faiss-cpu or faiss-gpu is required")

        # Load metadata
        meta_path = index_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Create lazy embedder (model not loaded yet)
        embedder = VLEmbedderLazy(
            model_path=model_path,
            gpu=gpu,
            gpu_memory=gpu_memory,
            batch_size=batch_size,
            max_image_size=max_image_size
        )

        # Create retriever instance
        retriever = cls(
            embedder=embedder,
            index_type=metadata["index_type"],
            nlist=metadata["nlist"],
            nprobe=metadata["nprobe"]
        )

        # Load FAISS index
        index_path = index_dir / "dense_vl.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        index = faiss.read_index(str(index_path))

        # Sanity check: verify index and metadata are consistent
        faiss_size = index.ntotal
        metadata_size = metadata.get("num_pages", len(metadata.get("page_ids", [])))

        if faiss_size != metadata_size:
            print(f"⚠️  Dense-VL Warning: Index size mismatch detected!")
            print(f"   FAISS index has {faiss_size} vectors")
            print(f"   Metadata has {metadata_size} pages")
            print(f"   Using minimum size: {min(faiss_size, metadata_size)}")

            # Truncate to the smaller size to prevent index errors
            safe_size = min(faiss_size, metadata_size)
            if safe_size < faiss_size:
                # Cannot truncate FAISS index, warn user
                print(f"   ⚠️  FAISS index is larger, some pages may be inaccessible")
            else:
                # Truncate page_ids to match FAISS index
                metadata["page_ids"] = metadata["page_ids"][:safe_size]
                print(f"   ✅ Truncated page_ids to {safe_size} to match FAISS index")

        # Set nprobe for IVF index
        if hasattr(index, 'nprobe'):
            index.nprobe = metadata["nprobe"]

        retriever.index = index
        retriever.page_ids = [tuple(p) for p in metadata["page_ids"]]
        retriever.page_metadata = metadata["page_metadata"]

        print(f"✅ Loaded Dense-VL index: {index.ntotal} pages (model will load on first query)")

        return retriever