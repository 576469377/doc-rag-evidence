"""
Dense text embedding indexer and retriever using FAISS.
Supports embedding via vLLM/SGLang-served models (OpenAI-compatible API).
"""
from typing import List, Optional, Protocol, Union, TYPE_CHECKING
from pathlib import Path
import json
import numpy as np

if TYPE_CHECKING:
    from core.schemas import QueryInput, AppConfig, RetrievalResult

try:
    import faiss
except ImportError:
    faiss = None

import requests

from core.schemas import IndexUnit, RetrieveHit


class Embedder(Protocol):
    """Protocol for text embedding models."""
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embed_dim)
        """
        ...
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (embed_dim,)
        """
        ...


class VLLMEmbedder:
    """
    Text embedder using vLLM-served embedding models.
    Compatible with OpenAI embeddings API.
    """
    
    def __init__(
        self,
        endpoint: str,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        timeout: int = 30,
        batch_size: int = 32
    ):
        """
        Initialize vLLM embedder.
        
        Args:
            endpoint: vLLM server endpoint (e.g., http://localhost:8001)
            model: Model name/path
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for embedding
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._embed_dim = None  # Lazy init
    
    @property
    def embed_dim(self) -> int:
        """Get embedding dimension (lazy initialization)."""
        if self._embed_dim is None:
            # Embed a dummy text to get dimension
            dummy_embedding = self.embed_query("test")
            self._embed_dim = len(dummy_embedding)
        return self._embed_dim
    
    def _call_embedding_api(self, texts: List[str]) -> np.ndarray:
        """Call vLLM embedding API."""
        url = f"{self.endpoint}/v1/embeddings"
        
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embeddings in order
            embeddings = []
            for item in sorted(result["data"], key=lambda x: x["index"]):
                embeddings.append(item["embedding"])
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"Error calling embedding API: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Automatically handles batching for large inputs.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embed_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embed_dim)
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._call_embedding_api(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (embed_dim,)
        """
        result = self._call_embedding_api([query])
        return result[0]


class SGLangEmbedder:
    """
    Text embedder using SGLang-served embedding models.
    Compatible with OpenAI embeddings API.
    (Alias for VLLMEmbedder for backward compatibility)
    """
    
    def __init__(
        self,
        endpoint: str,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        timeout: int = 30,
        batch_size: int = 32
    ):
        """
        Initialize SGLang embedder.
        
        Args:
            endpoint: SGLang/vLLM server endpoint
            model: Model name/path
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for embedding
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._embed_dim = None  # Lazy init
    
    @property
    def embed_dim(self) -> int:
        """Get embedding dimension (lazy initialization)."""
        if self._embed_dim is None:
            # Embed a dummy text to get dimension
            dummy_embedding = self.embed_query("test")
            self._embed_dim = len(dummy_embedding)
        return self._embed_dim
    
    def _call_embedding_api(self, texts: List[str]) -> np.ndarray:
        """Call SGLang/vLLM embedding API."""
        url = f"{self.endpoint}/v1/embeddings"
        
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embeddings in order
            embeddings = []
            for item in sorted(result["data"], key=lambda x: x["index"]):
                embeddings.append(item["embedding"])
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"Error calling embedding API: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Automatically handles batching for large inputs.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embed_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embed_dim)
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._call_embedding_api(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (embed_dim,)
        """
        embeddings = self._call_embedding_api([query])
        return embeddings[0]


class DenseIndexerRetriever:
    """
    Dense retrieval using FAISS index.
    Supports both flat and IVF index types.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        index_type: str = "Flat",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize dense indexer/retriever.
        
        Args:
            embedder: Embedder instance for encoding texts
            index_type: FAISS index type ("Flat", "IVF")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF index
        """
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu is required. Install with: pip install faiss-cpu")
        
        self.embedder = embedder
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index: Optional[faiss.Index] = None
        self.units: List[IndexUnit] = []
    
    def build_units(self, units: List[IndexUnit]) -> List[IndexUnit]:
        """
        Store units for later indexing.
        
        Args:
            units: List of IndexUnit objects
            
        Returns:
            Same list of units
        """
        self.units = units
        return units
    
    def build_index(self, units=None, config=None):
        """
        Build FAISS index from units.
        Embeds all unit texts and creates index.
        
        Args:
            units: Optional list of IndexUnit (uses self.units if not provided)
            config: Optional config (for compatibility, not used)
        """
        # Use provided units or fall back to self.units
        if units is not None:
            self.units = units
        
        if not self.units:
            print("Warning: No units to index")
            return
        
        print(f"Embedding {len(self.units)} units...")
        texts = [unit.text for unit in self.units]
        embeddings = self.embedder.embed_texts(texts)
        
        print(f"Building FAISS index (type={self.index_type})...")
        embed_dim = embeddings.shape[1]
        
        if self.index_type == "Flat":
            # Flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(embed_dim)
            self.index.add(embeddings)
        
        elif self.index_type == "IVF":
            # IVF index (approximate search)
            quantizer = faiss.IndexFlatL2(embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embed_dim, self.nlist)
            
            # Train index
            print("Training IVF index...")
            self.index.train(embeddings)
            self.index.add(embeddings)
            
            # Set search parameters
            self.index.nprobe = self.nprobe
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save(self, index_dir: Path):
        """
        Save index to directory (alias for persist).
        
        Args:
            index_dir: Directory to save index files
        """
        return self.persist(index_dir)
    
    def persist(self, index_dir: Path):
        """
        Save index and units to disk.
        
        Args:
            index_dir: Directory to save index files
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = index_dir / "dense.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save units
        units_path = index_dir / "units.jsonl"
        with open(units_path, 'w', encoding='utf-8') as f:
            for unit in self.units:
                f.write(unit.model_dump_json() + '\n')
        
        # Save metadata
        meta_path = index_dir / "dense_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "num_vectors": self.index.ntotal,
                "embed_dim": self.index.d
            }, f, indent=2)
        
        print(f"Dense index saved to {index_dir}")
    
    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> 'DenseIndexerRetriever':
        """
        Load index and units from disk.
        
        Args:
            index_dir: Directory containing index files
            embedder: Embedder instance
            
        Returns:
            DenseIndexerRetriever instance with loaded index
        """
        index_dir = Path(index_dir)
        
        # Load metadata
        meta_path = index_dir / "dense_meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # Create retriever instance
        retriever = cls(
            embedder=embedder,
            index_type=meta["index_type"],
            nlist=meta.get("nlist", 100),
            nprobe=meta.get("nprobe", 10)
        )
        
        # Load FAISS index
        index_path = index_dir / "dense.faiss"
        retriever.index = faiss.read_index(str(index_path))
        
        # Set nprobe for IVF index
        if hasattr(retriever.index, 'nprobe'):
            retriever.index.nprobe = retriever.nprobe
        
        # Load units
        units_path = index_dir / "units.jsonl"
        units = []
        with open(units_path, 'r', encoding='utf-8') as f:
            for line in f:
                units.append(IndexUnit.model_validate_json(line))
        retriever.units = units
        
        print(f"Loaded dense index with {len(units)} units")
        return retriever
    
    def retrieve(
        self,
        query: Union[str, 'QueryInput'],
        config: Optional['AppConfig'] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Union[List[RetrieveHit], 'RetrievalResult']:
        """
        Retrieve top-k most similar units to query.
        
        Args:
            query: Query string or QueryInput object
            config: AppConfig (optional, for compatibility with Pipeline)
            top_k: Number of results to return (defaults to config.top_k_retrieve)
            score_threshold: Optional minimum similarity threshold
            
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
        
        if self.index is None or not self.units:
            if return_result:
                return RetrievalResult(query_id=query_id, hits=[], elapsed_ms=0)
            return []
        
        # Embed query
        query_embedding = self.embedder.embed_query(query_str)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search index (returns L2 distances)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert L2 distances to similarity scores (higher is better)
        # Simple conversion: similarity = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Build results
        hits = []
        for idx, sim in zip(indices[0], similarities):
            if idx < 0 or idx >= len(self.units):
                continue
            
            # Apply threshold if specified
            if score_threshold is not None and sim < score_threshold:
                continue
            
            unit = self.units[idx]
            hit = RetrieveHit(
                unit_id=unit.unit_id,
                doc_id=unit.doc_id,
                page_id=unit.page_id,
                block_id=unit.block_id,
                text=unit.text,
                score=float(sim),
                source="dense",
                metadata={
                    "distance": float(distances[0][list(indices[0]).index(idx)])
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
