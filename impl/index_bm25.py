# impl/index_bm25.py
"""
BM25 Indexer and Retriever V0 implementation.

Uses rank_bm25 library for simple BM25 search.
Stores index as pickle file for persistence.

V0 features:
  - Text-based BM25 retrieval
  - Page or block level indexing
  - Persistence via pickle
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from core.schemas import (
    AppConfig, IndexUnit, IndexBuildStats, QueryInput, RetrieveHit, RetrievalResult
)
from infra.store_local import DocumentStoreLocal


class BM25IndexerRetriever:
    """BM25-based indexer and retriever."""

    def __init__(self, store: DocumentStoreLocal):
        self.store = store
        self.index = None
        self.units = []  # List of IndexUnit
        self.tokenized_corpus = []

        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            )

    def build_units(self, doc_id: str, config: AppConfig) -> List[IndexUnit]:
        """
        Build index units from a document's page artifacts.
        
        Args:
            doc_id: Document ID
            config: App configuration
            
        Returns:
            List of IndexUnit objects
        """
        units = []
        pages = self.store.get_all_pages(doc_id)

        for page in pages:
            if config.chunk_level == "page":
                # Page-level indexing
                if page.text and page.text.text:
                    unit_id = f"{doc_id}_p{page.page_id:04d}"
                    unit = IndexUnit(
                        unit_id=unit_id,
                        doc_id=doc_id,
                        page_id=page.page_id,
                        block_id=None,
                        text=page.text.text,
                        bbox=None,
                        metadata={}
                    )
                    units.append(unit)

            elif config.chunk_level == "block":
                # Block-level indexing
                for block in page.blocks:
                    if block.text:
                        unit = IndexUnit(
                            unit_id=block.block_id,
                            doc_id=doc_id,
                            page_id=page.page_id,
                            block_id=block.block_id,
                            text=block.text,
                            bbox=block.bbox,
                            metadata={"block_type": block.block_type}
                        )
                        units.append(unit)

        return units

    def build_index(self, units: List[IndexUnit], config: AppConfig) -> IndexBuildStats:
        """
        Build BM25 index from index units.
        
        Args:
            units: List of index units
            config: App configuration
            
        Returns:
            IndexBuildStats with build metrics
        """
        start_time = time.time()

        self.units = units

        # Check if units is empty
        if not units:
            raise ValueError(
                "Cannot build BM25 index: no index units provided. "
                "Make sure documents have text content (check OCR if using images)."
            )

        # Tokenize corpus (simple word-level tokenization)
        self.tokenized_corpus = [
            self._tokenize(unit.text) for unit in units
        ]

        # Build BM25 index
        self.index = BM25Okapi(self.tokenized_corpus)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Compute stats
        doc_ids = set(unit.doc_id for unit in units)
        page_ids = set((unit.doc_id, unit.page_id) for unit in units)

        stats = IndexBuildStats(
            doc_count=len(doc_ids),
            page_count=len(page_ids),
            unit_count=len(units),
            elapsed_ms=elapsed_ms,
            index_type="bm25"
        )

        return stats

    def persist(self, config: AppConfig, index_name: str = "bm25_default") -> None:
        """
        Persist index to disk.
        
        Args:
            config: App configuration
            index_name: Name of the index
        """
        index_dir = Path(config.indices_dir) / index_name
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save index and units
        index_path = index_dir / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "units": [u.model_dump() for u in self.units],
                "tokenized_corpus": self.tokenized_corpus
            }, f)

    def load(self, config: AppConfig, index_name: str = "bm25_default") -> bool:
        """
        Load index from disk.
        
        Args:
            config: App configuration
            index_name: Name of the index
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = Path(config.indices_dir) / index_name / "index.pkl"
        if not index_path.exists():
            return False

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.index = data["index"]
        self.units = [IndexUnit(**u) for u in data["units"]]
        self.tokenized_corpus = data["tokenized_corpus"]

        return True

    def retrieve(self, query: QueryInput, config: AppConfig) -> RetrievalResult:
        """
        Retrieve top-k units for a query.
        
        Args:
            query: Query input
            config: App configuration
            
        Returns:
            RetrievalResult with top-k hits
        """
        start_time = time.time()

        if self.index is None or not self.units:
            return RetrievalResult(
                query_id=query.query_id,
                hits=[],
                elapsed_ms=0
            )

        # Tokenize query
        tokenized_query = self._tokenize(query.question)

        # Get BM25 scores
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_k = config.top_k_retrieve
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Build hits
        hits = []
        for idx in top_indices:
            unit = self.units[idx]
            score = float(scores[idx])

            # Apply doc_filter if specified
            if query.doc_filter and unit.doc_id not in query.doc_filter:
                continue

            hit = RetrieveHit(
                unit_id=unit.unit_id,
                doc_id=unit.doc_id,
                page_id=unit.page_id,
                block_id=unit.block_id,
                text=unit.text,
                score=score,
                bbox=unit.bbox,
                source="bm25",
                metadata=unit.metadata
            )
            hits.append(hit)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RetrievalResult(
            query_id=query.query_id,
            hits=hits,
            elapsed_ms=elapsed_ms
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization (lowercase, split by whitespace)."""
        return text.lower().split()
