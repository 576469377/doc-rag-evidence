"""
Hit normalizer: convert page-level hits to block-level hits.
Used for ColPali retrieval to ensure evidence has text snippets.
"""
from typing import List, Optional
from pathlib import Path

from core.schemas import RetrieveHit, AppConfig
from infra.store_local import DocumentStoreLocal


class HitNormalizer:
    """
    Normalize retrieval hits to block-level evidence.
    
    For page-level hits (e.g., from ColPali), expands them into block-level
    hits by loading blocks from the page and ranking them.
    """
    
    def __init__(self, store: DocumentStoreLocal):
        """
        Initialize normalizer.
        
        Args:
            store: Document store for loading page artifacts
        """
        self.store = store
    
    def normalize_hits(
        self,
        hits: List[RetrieveHit],
        config: AppConfig,
        query: str,
        source: str
    ) -> List[RetrieveHit]:
        """
        Normalize hits to block-level evidence.
        
        Args:
            hits: Retrieved hits (may be page-level or block-level)
            config: Application config
            query: Original query string
            source: Retrieval source (bm25/dense/colpali)
            
        Returns:
            Block-level hits with text snippets
        """
        normalized = []
        
        for hit in hits:
            # Check if hit is already block-level (has text and block_id)
            if hit.text and hit.text.strip() and hit.block_id:
                # Already block-level, keep as is
                normalized.append(hit)
            else:
                # Page-level hit, needs expansion
                expanded = self._expand_page_hit(hit, query, config)
                normalized.extend(expanded)
        
        return normalized
    
    def _expand_page_hit(
        self,
        page_hit: RetrieveHit,
        query: str,
        config: AppConfig
    ) -> List[RetrieveHit]:
        """
        Expand a page-level hit into block-level hits.
        
        Strategy:
        1. Load blocks from page
        2. Rank blocks by relevance (simple keyword scoring)
        3. Return top blocks as evidence
        
        Args:
            page_hit: Page-level retrieve hit
            query: Query string
            config: Application config
            
        Returns:
            List of block-level hits from this page
        """
        # Load page artifact
        artifact = self.store.load_page_artifact(page_hit.doc_id, page_hit.page_id)
        
        if not artifact or not artifact.blocks:
            # No blocks available, try to use page text as single block
            if artifact and artifact.text and artifact.text.text.strip():
                return [RetrieveHit(
                    unit_id=f"{page_hit.doc_id}_p{page_hit.page_id:04d}",
                    doc_id=page_hit.doc_id,
                    page_id=page_hit.page_id,
                    block_id=None,
                    text=artifact.text.text,
                    score=page_hit.score,
                    source=page_hit.source,
                    metadata={**page_hit.metadata, "expanded_from": "page"}
                )]
            return []
        
        # Score blocks by simple keyword matching
        query_keywords = set(query.lower().split())
        block_scores = []
        
        for block in artifact.blocks:
            if not block.text or not block.text.strip():
                continue
            
            # Simple scoring: keyword overlap + length bonus
            block_text_lower = block.text.lower()
            keyword_hits = sum(1 for kw in query_keywords if kw in block_text_lower)
            
            # Normalize by query length and add length bonus
            relevance_score = keyword_hits / max(len(query_keywords), 1)
            length_bonus = min(len(block.text) / 1000, 0.2)  # Up to 0.2 bonus
            
            block_score = relevance_score + length_bonus
            
            # Inherit original page score and combine with block score
            # Use weighted average: 70% page score, 30% block score
            final_score = 0.7 * page_hit.score + 0.3 * block_score
            
            block_scores.append((block, final_score))
        
        # Sort by score and take top blocks
        block_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 blocks from this page (configurable)
        top_blocks_per_page = 3
        results = []
        
        for block, score in block_scores[:top_blocks_per_page]:
            hit = RetrieveHit(
                unit_id=f"{page_hit.doc_id}_p{page_hit.page_id:04d}_b{block.block_id}",
                doc_id=page_hit.doc_id,
                page_id=page_hit.page_id,
                block_id=block.block_id,
                text=block.text,
                score=score,
                source=page_hit.source,
                metadata={
                    **page_hit.metadata,
                    "expanded_from": "page",
                    "original_page_score": page_hit.score
                }
            )
            results.append(hit)
        
        return results


class DenseHitExpander:
    """
    Optional: expand page/block hits with dense re-ranking.
    For V1.1, this is a future enhancement after basic expansion works.
    """
    pass
