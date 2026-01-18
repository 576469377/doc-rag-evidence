"""
ColPali-based vision retrieval for document pages.
Two-stage retrieval: coarse (global vectors) + fine (late interaction).
"""
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass

if TYPE_CHECKING:
    from core.schemas import QueryInput, AppConfig, RetrievalResult

try:
    import torch
    from transformers import AutoProcessor, AutoModel
except ImportError:
    torch = None
    AutoProcessor = None
    AutoModel = None

try:
    import faiss
except ImportError:
    faiss = None

from PIL import Image

from core.schemas import RetrieveHit, AppConfig


@dataclass
class PageVectorStore:
    """Store for page-level embeddings."""
    page_ids: List[Tuple[str, int]]  # List of (doc_id, page_id)
    global_vectors: np.ndarray  # Shape: (num_pages, embed_dim)
    patch_vectors: List[np.ndarray]  # List of patch embeddings per page


class ColPaliRetriever:
    """
    ColPali/ColQwen3-based vision retriever.
    
    Two-stage retrieval:
    1. Coarse: Use global pooled vectors for fast top-N retrieval
    2. Fine: Late interaction scoring on patch vectors for top-N
    """
    
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        device: str = "cuda:0",
        max_global_pool_pages: int = 100,
        cache_dir: Optional[Path] = None,
        max_image_size: int = None
    ):
        """
        Initialize ColPali retriever.
        
        Args:
            model_name: HuggingFace model name
            device: CUDA device (e.g., "cuda:0")
            max_global_pool_pages: Max pages to retrieve in coarse stage
            cache_dir: Optional cache directory for embeddings
            max_image_size: Max image size (pixels) on longest side. If None, use original size.
        """
        if torch is None or AutoProcessor is None:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")
        if faiss is None:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")
        
        self.model_name = model_name
        self.device = device
        self.max_global_pool_pages = max_global_pool_pages
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_image_size = max_image_size
        
        # Load model and processor
        print(f"Loading ColPali model: {model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_num_visual_tokens=1280  # ColQwen3 parameter
        )
        
        # Try to use Flash Attention 2 for speedup
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device
            ).eval()
            print("  ✅ Using Flash Attention 2 (fast)")
        except Exception as e:
            # Fallback to standard attention if flash-attn not available
            if "flash_attn" in str(e).lower() or "flash" in str(e).lower():
                print("  ⚠️  Flash Attention 2 not available, using standard attention (slower)")
                print("     To enable: pip install flash-attn --no-build-isolation")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device
                ).eval()
            else:
                raise
        
        # Vector store
        self.store: Optional[PageVectorStore] = None
        self.index: Optional[faiss.Index] = None
    
    def _resize_image_if_needed(self, image_path: str) -> str:
        """Resize image if needed to reduce GPU memory usage. Returns path to (possibly resized) image."""
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
            temp_path = Path(image_path).parent / f"temp_colpali_resized_{self.max_image_size}.jpg"
            img_resized.save(str(temp_path), quality=95)
            
            return str(temp_path)
        except Exception as e:
            print(f"Warning: Failed to resize {image_path}: {e}")
            return image_path
    
    def _get_cache_path(self, doc_id: str, page_id: int) -> Optional[Path]:
        """Get cache path for page embeddings."""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / doc_id / f"page_{page_id:04d}_colpali.npz"
        return cache_path
    
    def _load_cached_embeddings(self, doc_id: str, page_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load cached embeddings if available."""
        cache_path = self._get_cache_path(doc_id, page_id)
        if cache_path and cache_path.exists():
            try:
                data = np.load(cache_path)
                return data['global_vec'], data['patch_vecs']
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings: {e}")
        return None
    
    def _save_cached_embeddings(
        self,
        doc_id: str,
        page_id: int,
        global_vec: np.ndarray,
        patch_vecs: np.ndarray
    ):
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(doc_id, page_id)
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    global_vec=global_vec,
                    patch_vecs=patch_vecs
                )
            except Exception as e:
                print(f"Warning: Failed to save cached embeddings: {e}")
    
    @torch.no_grad()
    def _embed_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed a single page image using ColQwen3 API.
        
        Returns:
            global_vec: Shape (embed_dim,) - pooled representation
            patch_vecs: Shape (num_patches, embed_dim) - patch-level embeddings
        """
        # Resize if needed to save GPU memory
        resized_path = self._resize_image_if_needed(image_path)
        
        # Load image
        image = Image.open(resized_path).convert('RGB')
        
        # Process image using ColQwen3 API
        features = self.processor.process_images(images=[image])
        features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in features.items()}
        
        # Get embeddings
        outputs = self.model(**features)
        
        # Extract embeddings - ColQwen3 returns .embeddings attribute
        # Shape: [1, num_patches, embed_dim]
        # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
        patch_vecs = outputs.embeddings[0].cpu().float().numpy()  # [num_patches, embed_dim]
        
        # Global pooling (mean over patches)
        global_vec = patch_vecs.mean(axis=0)  # [embed_dim]
        
        return global_vec, patch_vecs
    
    @torch.no_grad()
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a text query using ColQwen3 API.
        
        Returns:
            query_vecs: Shape (query_tokens, embed_dim) for late interaction
        """
        # Process query text using ColQwen3 API
        batch = self.processor.process_texts(texts=[query])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get embeddings
        outputs = self.model(**batch)
        
        # Extract token embeddings - ColQwen3 returns .embeddings attribute
        # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
        query_vecs = outputs.embeddings[0].cpu().float().numpy()  # [num_tokens, embed_dim]
        
        return query_vecs
    
    @torch.no_grad()
    def _embed_images_batch(self, image_paths: List[str], batch_size: int = 16) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        批量embed图像，提升索引构建速度。
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小（默认16，24GB显存可支持）
        
        Returns:
            List of (global_vec, patch_vecs) tuples
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Resize images if needed
            resized_paths = [self._resize_image_if_needed(path) for path in batch_paths]
            
            # Load images
            images = [Image.open(path).convert('RGB') for path in resized_paths]
            
            # Batch process
            features = self.processor.process_images(images=images)
            features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in features.items()}
            
            # Get embeddings
            outputs = self.model(**features)
            
            # Extract embeddings for each image in batch
            # Shape: [batch_size, num_patches, embed_dim]
            embeddings = outputs.embeddings.cpu().float().numpy()
            
            for j in range(len(batch_paths)):
                patch_vecs = embeddings[j]  # [num_patches, embed_dim]
                global_vec = patch_vecs.mean(axis=0)  # [embed_dim]
                results.append((global_vec, patch_vecs))
        
        return results
    
    def build_index(
        self,
        page_data: Union[List[Tuple[str, int]], List[Tuple[str, int, str]]],
        config: Optional[AppConfig] = None
    ):
        """
        Build index from page images.
        
        Args:
            page_data: Either:
                - List of (doc_id, page_id) tuples (requires config to locate images)
                - List of (doc_id, page_id, image_path) tuples (deprecated)
            config: AppConfig (required if page_data is (doc_id, page_id) format)
        """
        print(f"Building ColPali index for {len(page_data)} pages...")
        
        page_ids = []
        global_vectors = []
        patch_vectors = []
        
        # 第一步：收集所有需要处理的图像
        items_to_process = []  # (doc_id, page_id, image_path)
        items_cached = []  # (doc_id, page_id, global_vec, patch_vecs)
        
        for item in page_data:
            if len(item) == 2:
                doc_id, page_id = item
                if config is None:
                    raise ValueError("config is required when page_data is (doc_id, page_id) format")
                image_path = Path(config.docs_dir) / doc_id / "pages" / f"{page_id:04d}" / "page.png"
                image_path = str(image_path)
            elif len(item) == 3:
                doc_id, page_id, image_path = item
            else:
                raise ValueError(f"Invalid page_data item format: {item}")
            
            if not Path(image_path).exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                continue
            
            # Check cache
            cached = self._load_cached_embeddings(doc_id, page_id)
            if cached:
                items_cached.append((doc_id, page_id, cached[0], cached[1]))
            else:
                items_to_process.append((doc_id, page_id, image_path))
        
        # 第二步：批量处理未缓存的图像（并行加速）
        if items_to_process:
            print(f"⏳ 批量处理 {len(items_to_process)} 个图像（batch_size=16）...")
            image_paths = [item[2] for item in items_to_process]
            embeddings = self._embed_images_batch(image_paths, batch_size=16)
            
            # Save to cache and collect
            for (doc_id, page_id, _), (global_vec, patch_vecs) in zip(items_to_process, embeddings):
                self._save_cached_embeddings(doc_id, page_id, global_vec, patch_vecs)
                page_ids.append((doc_id, page_id))
                global_vectors.append(global_vec)
                patch_vectors.append(patch_vecs)
        
        # 第三步：添加缓存的数据
        for doc_id, page_id, global_vec, patch_vecs in items_cached:
            page_ids.append((doc_id, page_id))
            global_vectors.append(global_vec)
            patch_vectors.append(patch_vecs)
        
        if not page_ids:
            raise ValueError("No valid pages found to build index")
        
        # Stack global vectors
        global_vectors_array = np.stack(global_vectors).astype(np.float32)
        
        # Build FAISS index on global vectors
        embed_dim = global_vectors_array.shape[1]
        self.index = faiss.IndexFlatIP(embed_dim)  # Inner product (cosine similarity)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(global_vectors_array)
        self.index.add(global_vectors_array)
        
        # Store vectors
        self.store = PageVectorStore(
            page_ids=page_ids,
            global_vectors=global_vectors_array,
            patch_vectors=patch_vectors
        )
        
        print(f"ColPali index built with {len(page_ids)} pages")
    
    def add_pages(
        self,
        page_data: Union[List[Tuple[str, int]], List[Tuple[str, int, str]]],
        config: Optional[AppConfig] = None
    ):
        """
        Add new pages to existing index (incremental update).
        
        Args:
            page_data: List of (doc_id, page_id) or (doc_id, page_id, image_path) tuples
            config: AppConfig (required if using (doc_id, page_id) format)
        """
        if self.index is None or self.store is None:
            # No existing index, just build from scratch
            return self.build_index(page_data, config)
        
        print(f"Adding {len(page_data)} new pages to ColPali index...")
        
        new_page_ids = []
        new_global_vectors = []
        new_patch_vectors = []
        
        # 收集需要处理的图像（与build_index相同的批量处理逻辑）
        items_to_process = []  # (doc_id, page_id, image_path)
        items_cached = []  # (doc_id, page_id, global_vec, patch_vecs)
        
        for item in page_data:
            if len(item) == 2:
                doc_id, page_id = item
                if config is None:
                    raise ValueError("config is required when page_data is (doc_id, page_id) format")
                image_path = Path(config.docs_dir) / doc_id / "pages" / f"{page_id:04d}" / "page.png"
                image_path = str(image_path)
            elif len(item) == 3:
                doc_id, page_id, image_path = item
            else:
                raise ValueError(f"Invalid page_data item format: {item}")
            
            # Skip if already in index
            if (doc_id, page_id) in self.store.page_ids:
                print(f"  ⏭️  Skipping {doc_id} page {page_id} (already indexed)")
                continue
            
            # Check if image exists
            if not Path(image_path).exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                continue
            
            # Check cache
            cached = self._load_cached_embeddings(doc_id, page_id)
            if cached:
                items_cached.append((doc_id, page_id, cached[0], cached[1]))
            else:
                items_to_process.append((doc_id, page_id, image_path))
        
        # 批量处理未缓存的图像
        if items_to_process:
            print(f"⏳ 批量处理 {len(items_to_process)} 个新图像（batch_size=16）...")
            image_paths = [item[2] for item in items_to_process]
            embeddings = self._embed_images_batch(image_paths, batch_size=16)
            
            # Save to cache and collect
            for (doc_id, page_id, _), (global_vec, patch_vecs) in zip(items_to_process, embeddings):
                self._save_cached_embeddings(doc_id, page_id, global_vec, patch_vecs)
                new_page_ids.append((doc_id, page_id))
                new_global_vectors.append(global_vec)
                new_patch_vectors.append(patch_vecs)
        
        # 添加缓存的数据
        for doc_id, page_id, global_vec, patch_vecs in items_cached:
            new_page_ids.append((doc_id, page_id))
            new_global_vectors.append(global_vec)
            new_patch_vectors.append(patch_vecs)
        
        if not new_page_ids:
            print("No new pages to add (all already indexed or invalid)")
            return
        
        # Stack new global vectors
        new_global_vectors_array = np.stack(new_global_vectors).astype(np.float32)
        
        # Normalize and add to FAISS index
        faiss.normalize_L2(new_global_vectors_array)
        self.index.add(new_global_vectors_array)
        
        # Merge with existing store
        combined_page_ids = self.store.page_ids + new_page_ids
        combined_global_vectors = np.vstack([
            self.store.global_vectors,
            new_global_vectors_array
        ])
        combined_patch_vectors = self.store.patch_vectors + new_patch_vectors
        
        self.store = PageVectorStore(
            page_ids=combined_page_ids,
            global_vectors=combined_global_vectors,
            patch_vectors=combined_patch_vectors
        )
        
        print(f"✅ Added {len(new_page_ids)} pages. Total: {len(self.store.page_ids)} pages")
    
    def save(self, index_dir: Path):
        """Save index to disk (alias for persist)."""
        return self.persist(index_dir)
    
    def persist(self, index_dir: Path):
        """Save index to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = index_dir / "colpali_global.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save page IDs
        page_ids_path = index_dir / "colpali_page_ids.json"
        with open(page_ids_path, 'w', encoding='utf-8') as f:
            json.dump([
                {"doc_id": doc_id, "page_id": page_id}
                for doc_id, page_id in self.store.page_ids
            ], f, indent=2)
        
        # Save patch vectors (compressed)
        patch_vecs_path = index_dir / "colpali_patch_vectors.npz"
        np.savez_compressed(
            patch_vecs_path,
            **{f"page_{i}": vecs for i, vecs in enumerate(self.store.patch_vectors)}
        )
        
        print(f"ColPali index saved to {index_dir}")
    
    def load_instance(self, index_dir: Path):
        """
        Load index from disk (instance method).
        
        Args:
            index_dir: Directory containing the saved index
        """
        index_dir = Path(index_dir)
        
        # Load FAISS index
        index_path = index_dir / "colpali_global.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load page IDs
        page_ids_path = index_dir / "colpali_page_ids.json"
        with open(page_ids_path, 'r', encoding='utf-8') as f:
            page_ids_data = json.load(f)
        page_ids = [(item["doc_id"], item["page_id"]) for item in page_ids_data]
        
        # Load global vectors (needed for incremental updates)
        # For Flat index, we can reconstruct vectors
        num_vectors = self.index.ntotal
        if num_vectors > 0:
            # Reconstruct all vectors from FAISS index
            global_vectors = np.zeros((num_vectors, self.index.d), dtype=np.float32)
            for i in range(num_vectors):
                global_vectors[i] = self.index.reconstruct(i)
        else:
            global_vectors = np.array([]).reshape(0, self.index.d)
        
        # Load patch vectors
        patch_vecs_path = index_dir / "colpali_patch_vectors.npz"
        patch_data = np.load(patch_vecs_path)
        patch_vectors = [patch_data[f"page_{i}"] for i in range(len(page_ids))]
        
        # Reconstruct store
        self.store = PageVectorStore(
            page_ids=page_ids,
            global_vectors=global_vectors,
            patch_vectors=patch_vectors
        )
        
        print(f"Loaded ColPali index with {len(page_ids)} pages")
    
    @classmethod
    def load(
        cls,
        index_dir: Path,
        model_name: str,
        device: str = "cuda:0",
        max_image_size: int = None
    ) -> 'ColPaliRetriever':
        """
        Load index from disk (class method for compatibility).
        
        Args:
            index_dir: Directory containing the saved index
            model_name: ColPali model name/path
            device: Device to load model on
            max_image_size: Max image size for resizing (None = no resize)
            
        Returns:
            ColPaliRetriever instance with loaded index
        """
        index_dir = Path(index_dir)
        
        # Create retriever with max_image_size
        retriever = cls(model_name=model_name, device=device, max_image_size=max_image_size)
        
        # Load index using instance method
        retriever.load_instance(index_dir)
        
        return retriever
    
    def _late_interaction_score(
        self,
        query_vecs: np.ndarray,
        page_patch_vecs: np.ndarray
    ) -> float:
        """
        Compute late interaction score between query and page using ColQwen3 MaxSim.
        
        MaxSim scoring: for each query token, find max similarity with page patches.
        
        Args:
            query_vecs: Shape (num_query_tokens, embed_dim)
            page_patch_vecs: Shape (num_patches, embed_dim)
            
        Returns:
            Similarity score (float)
        """
        # Use processor's score_multi_vector if available (ColQwen3 API)
        if hasattr(self.processor, 'score_multi_vector'):
            # Convert to tensor format: (sequence_length, embedding_dim)
            # Note: score_multi_vector expects list of 2D tensors
            query_tensor = torch.from_numpy(query_vecs).to(self.device)  # [num_tokens, dim]
            doc_tensor = torch.from_numpy(page_patch_vecs).to(self.device)  # [num_patches, dim]
            
            # score_multi_vector expects list of 2D tensors for each query/doc
            scores = self.processor.score_multi_vector([query_tensor], [doc_tensor], device=self.device)
            # Returns shape [n_queries, n_passages] -> [1, 1]
            return float(scores[0][0])
        
        # Fallback: manual MaxSim implementation
        # Normalize
        query_vecs = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        page_patch_vecs = page_patch_vecs / (np.linalg.norm(page_patch_vecs, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix: (num_query_tokens, num_patches)
        sim_matrix = query_vecs @ page_patch_vecs.T
        
        # MaxSim: for each query token, take max over page patches
        max_sims = sim_matrix.max(axis=1)
        
        # Average over query tokens
        score = max_sims.mean()
        
        return float(score)
    
    def _batch_late_interaction_score(
        self,
        query_vecs: np.ndarray,
        page_indices: List[int]
    ) -> np.ndarray:
        """
        Batch compute late interaction scores for multiple pages (optimized & parallelized).
        
        Uses vectorized NumPy operations and multiprocessing for speed.
        
        Args:
            query_vecs: Query embeddings, shape (num_query_tokens, embed_dim)
            page_indices: List of page indices to score
            
        Returns:
            Array of scores for each page
        """
        # Normalize query vectors once
        query_vecs_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        
        scores = np.zeros(len(page_indices), dtype=np.float32)
        
        # Use ThreadPoolExecutor for parallel computation
        from concurrent.futures import ThreadPoolExecutor
        import os
        
        def compute_score(args):
            """Compute score for a single page."""
            idx, page_idx = args
            page_patch_vecs = self.store.patch_vectors[page_idx]
            
            # Normalize page patches
            page_patch_vecs_norm = page_patch_vecs / (
                np.linalg.norm(page_patch_vecs, axis=1, keepdims=True) + 1e-8
            )
            
            # Compute similarity matrix: (num_query_tokens, num_patches)
            sim_matrix = query_vecs_norm @ page_patch_vecs_norm.T
            
            # MaxSim: for each query token, take max over page patches
            max_sims = sim_matrix.max(axis=1)
            
            # Average over query tokens
            return idx, max_sims.mean()
        
        # Determine number of workers (use available CPUs, but cap at 8 to avoid overhead)
        num_workers = min(8, len(page_indices), os.cpu_count() or 4)
        
        # Parallel computation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(compute_score, enumerate(page_indices))
            
            for idx, score in results:
                scores[idx] = score
        
        return scores
    
    def retrieve(
        self,
        query: Union[str, 'QueryInput'],
        config: Optional['AppConfig'] = None,
        top_k: Optional[int] = None,
        coarse_k: Optional[int] = None
    ) -> Union[List[RetrieveHit], 'RetrievalResult']:
        """
        Two-stage retrieval.
        
        Stage 1: Retrieve top coarse_k pages using global vectors
        Stage 2: Rerank using late interaction on patch vectors
        
        Args:
            query: Query text or QueryInput object
            config: AppConfig (optional, for compatibility with Pipeline)
            top_k: Final number of results (defaults to config.top_k_retrieve)
            coarse_k: Number of candidates in coarse stage (default: max_global_pool_pages)
            
        Returns:
            List of RetrieveHit objects (if query is str) or RetrievalResult (if QueryInput)
        """
        import time
        from core.schemas import QueryInput, RetrievalResult
        
        # Handle QueryInput vs string
        if isinstance(query, QueryInput):
            query_str = query.question
            query_id = query.query_id
            return_result = True
            if config:
                top_k = top_k or config.top_k_retrieve
        else:
            query_str = query
            query_id = None
            return_result = False
        
        top_k = top_k or 10
        start_time = time.time()
        
        if self.index is None or self.store is None:
            if return_result:
                return RetrievalResult(query_id=query_id, hits=[], elapsed_ms=0)
            return []
        
        if coarse_k is None:
            coarse_k = min(self.max_global_pool_pages, len(self.store.page_ids))
        
        # Embed query
        query_vecs = self._embed_query(query_str)
        
        # Stage 1: Coarse retrieval using global query vector (mean pooling)
        global_query_vec = query_vecs.mean(axis=0).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(global_query_vec)
        
        distances, indices = self.index.search(global_query_vec, coarse_k)
        
        # Stage 2: Late interaction scoring (batch processing for speed)
        # Filter valid indices
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(self.store.page_ids)]
        
        if len(valid_indices) == 0:
            if return_result:
                elapsed_ms = int((time.time() - start_time) * 1000)
                return RetrievalResult(query_id=query_id, hits=[], elapsed_ms=elapsed_ms)
            return []
        
        # Batch compute scores using vectorized operations
        scores = self._batch_late_interaction_score(query_vecs, valid_indices)
        
        # Combine indices with scores
        idx_score_pairs = list(zip(valid_indices, scores))
        
        # Sort by score and take top-k
        idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
        idx_score_pairs = idx_score_pairs[:top_k]
        
        # Build results
        hits = []
        for idx, score in idx_score_pairs:
            doc_id, page_id = self.store.page_ids[idx]
            
            hit = RetrieveHit(
                unit_id=f"{doc_id}_p{page_id:04d}",  # Page-level unit ID
                doc_id=doc_id,
                page_id=page_id,
                block_id=None,  # Page-level retrieval
                text="",  # No text available in vision retrieval
                score=score,
                source="colpali",
                metadata={
                    "coarse_rank": int(list(indices[0]).index(idx)) + 1
                }
            )
            hits.append(hit)
        
        # Return appropriate type
        if return_result:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return RetrievalResult(
                query_id=query_id,
                hits=hits,
                elapsed_ms=elapsed_ms
            )
        return hits
