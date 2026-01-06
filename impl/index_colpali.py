"""
ColPali-based vision retrieval for document pages.
Two-stage retrieval: coarse (global vectors) + fine (late interaction).
"""
from typing import List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass

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

from core.schemas import RetrieveHit


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
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize ColPali retriever.
        
        Args:
            model_name: HuggingFace model name
            device: CUDA device (e.g., "cuda:0")
            max_global_pool_pages: Max pages to retrieve in coarse stage
            cache_dir: Optional cache directory for embeddings
        """
        if torch is None or AutoProcessor is None:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")
        if faiss is None:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")
        
        self.model_name = model_name
        self.device = device
        self.max_global_pool_pages = max_global_pool_pages
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load model and processor
        print(f"Loading ColPali model: {model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_num_visual_tokens=1280  # ColQwen3 parameter
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # Requires flash-attn
            device_map=device
        ).eval()
        
        # Vector store
        self.store: Optional[PageVectorStore] = None
        self.index: Optional[faiss.Index] = None
    
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
        # Load image
        image = Image.open(image_path).convert('RGB')
        
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
    
    def build_index(self, page_image_paths: List[Tuple[str, int, str]]):
        """
        Build index from page images.
        
        Args:
            page_image_paths: List of (doc_id, page_id, image_path) tuples
        """
        print(f"Building ColPali index for {len(page_image_paths)} pages...")
        
        page_ids = []
        global_vectors = []
        patch_vectors = []
        
        for doc_id, page_id, image_path in page_image_paths:
            # Check cache first
            cached = self._load_cached_embeddings(doc_id, page_id)
            if cached:
                global_vec, patch_vecs = cached
            else:
                # Embed image
                global_vec, patch_vecs = self._embed_image(image_path)
                
                # Save to cache
                self._save_cached_embeddings(doc_id, page_id, global_vec, patch_vecs)
            
            page_ids.append((doc_id, page_id))
            global_vectors.append(global_vec)
            patch_vectors.append(patch_vecs)
        
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
    
    @classmethod
    def load(
        cls,
        index_dir: Path,
        model_name: str,
        device: str = "cuda:0"
    ) -> 'ColPaliRetriever':
        """Load index from disk."""
        index_dir = Path(index_dir)
        
        # Create retriever
        retriever = cls(model_name=model_name, device=device)
        
        # Load FAISS index
        index_path = index_dir / "colpali_global.faiss"
        retriever.index = faiss.read_index(str(index_path))
        
        # Load page IDs
        page_ids_path = index_dir / "colpali_page_ids.json"
        with open(page_ids_path, 'r', encoding='utf-8') as f:
            page_ids_data = json.load(f)
        page_ids = [(item["doc_id"], item["page_id"]) for item in page_ids_data]
        
        # Load patch vectors
        patch_vecs_path = index_dir / "colpali_patch_vectors.npz"
        patch_data = np.load(patch_vecs_path)
        patch_vectors = [patch_data[f"page_{i}"] for i in range(len(page_ids))]
        
        # Reconstruct store
        retriever.store = PageVectorStore(
            page_ids=page_ids,
            global_vectors=None,  # Not needed for retrieval
            patch_vectors=patch_vectors
        )
        
        print(f"Loaded ColPali index with {len(page_ids)} pages")
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
            # Convert to tensor format expected by score_multi_vector
            query_tensor = torch.from_numpy(query_vecs).unsqueeze(0)  # [1, num_tokens, dim]
            doc_tensor = torch.from_numpy(page_patch_vecs).unsqueeze(0)  # [1, num_patches, dim]
            
            # score_multi_vector expects lists of tensors
            scores = self.processor.score_multi_vector([query_tensor], [doc_tensor])
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
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        coarse_k: Optional[int] = None
    ) -> List[RetrieveHit]:
        """
        Two-stage retrieval.
        
        Stage 1: Retrieve top coarse_k pages using global vectors
        Stage 2: Rerank using late interaction on patch vectors
        
        Args:
            query: Query text
            top_k: Final number of results
            coarse_k: Number of candidates in coarse stage (default: max_global_pool_pages)
            
        Returns:
            List of RetrieveHit objects sorted by score (descending)
        """
        if self.index is None or self.store is None:
            return []
        
        if coarse_k is None:
            coarse_k = min(self.max_global_pool_pages, len(self.store.page_ids))
        
        # Embed query
        query_vecs = self._embed_query(query)
        
        # Stage 1: Coarse retrieval using global query vector (mean pooling)
        global_query_vec = query_vecs.mean(axis=0).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(global_query_vec)
        
        distances, indices = self.index.search(global_query_vec, coarse_k)
        
        # Stage 2: Late interaction scoring
        scores = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.store.page_ids):
                continue
            
            page_patch_vecs = self.store.patch_vectors[idx]
            score = self._late_interaction_score(query_vecs, page_patch_vecs)
            scores.append((idx, score))
        
        # Sort by score and take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]
        
        # Build results
        hits = []
        for idx, score in scores:
            doc_id, page_id = self.store.page_ids[idx]
            
            hit = RetrieveHit(
                doc_id=doc_id,
                page_id=page_id,
                block_id=None,  # Page-level retrieval
                text="",  # No text available in vision retrieval
                score=score,
                metadata={
                    "source": "colpali",
                    "coarse_rank": int(list(indices[0]).index(idx)) + 1
                }
            )
            hits.append(hit)
        
        return hits
