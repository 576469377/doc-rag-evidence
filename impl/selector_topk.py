# impl/selector_topk.py
"""
TopK Evidence Selector V0 implementation.

Selects top-k retrieval hits as evidence items.
Adds snippet truncation for display.

V0 strategy:
  - Take top-k hits directly (no additional filtering)
  - Truncate text to snippet (first N chars)
  - Preserve all metadata for traceability
"""
from __future__ import annotations

import time
from typing import List

from core.schemas import (
    AppConfig, QueryInput, RetrievalResult, EvidenceItem, EvidenceSelectionResult
)


class TopKEvidenceSelector:
    """Simple top-k evidence selector."""

    def __init__(self, snippet_length: int = 500):
        """
        Args:
            snippet_length: Maximum length of text snippet (chars)
        """
        self.snippet_length = snippet_length

    def select(
        self, query: QueryInput, retrieval: RetrievalResult, config: AppConfig
    ) -> EvidenceSelectionResult:
        """
        Select top-k evidence items from retrieval hits.
        
        Args:
            query: Query input
            retrieval: Retrieval result with hits
            config: App configuration
            
        Returns:
            EvidenceSelectionResult with selected evidence
        """
        start_time = time.time()

        top_k = config.top_k_evidence
        hits = retrieval.hits[:top_k]

        evidence_items: List[EvidenceItem] = []

        for rank, hit in enumerate(hits):
            # Truncate text to snippet
            snippet = self._make_snippet(hit.text)

            evidence = EvidenceItem(
                rank=rank,
                unit_id=hit.unit_id,
                doc_id=hit.doc_id,
                page_id=hit.page_id,
                block_id=hit.block_id,
                snippet=snippet,
                bbox=hit.bbox,
                score=hit.score,
                source=hit.source,
                rationale=f"BM25 score: {hit.score:.4f}",
                metadata=hit.metadata
            )
            evidence_items.append(evidence)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return EvidenceSelectionResult(
            query_id=query.query_id,
            evidence=evidence_items,
            elapsed_ms=elapsed_ms
        )

    def _make_snippet(self, text: str) -> str:
        """
        Create a display snippet from full text.
        
        V0: Simple truncation with ellipsis.
        Future: Could do smart sentence boundary detection.
        """
        if len(text) <= self.snippet_length:
            return text

        # Truncate and add ellipsis
        return text[:self.snippet_length].strip() + "..."
