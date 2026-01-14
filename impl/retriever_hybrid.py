# impl/retriever_hybrid.py
"""Hybrid retriever that combines multiple retrieval methods."""

from typing import List, Dict, Any
from core.schemas import QueryInput, AppConfig, RetrieveHit
from core.inferences import Retriever


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining multiple retrieval methods.
    
    Supports:
    - Dense + ColPali fusion
    - BM25 + Dense fusion
    - Three-way fusion (BM25 + Dense + ColPali)
    
    Fusion strategy: Weighted score normalization + merge
    """
    
    def __init__(
        self,
        retrievers: Dict[str, Retriever],
        weights: Dict[str, float] = None,
        fusion_method: str = "weighted_sum"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            retrievers: Dict of {name: retriever_instance}
            weights: Dict of {name: weight}. Default: equal weights
            fusion_method: "weighted_sum" | "rrf" (reciprocal rank fusion)
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method
        
        # Default equal weights
        if weights is None:
            num_retrievers = len(retrievers)
            weights = {name: 1.0 / num_retrievers for name in retrievers}
        
        self.weights = weights
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        
        print(f"✅ Initialized HybridRetriever with {len(retrievers)} methods")
        print(f"   Methods: {list(retrievers.keys())}")
        print(f"   Weights: {self.weights}")
        print(f"   Fusion: {fusion_method}")
    
    def retrieve(self, query: QueryInput, config: AppConfig) -> "RetrievalResult":
        """
        Retrieve using hybrid fusion.
        
        Returns:
            RetrievalResult with merged hits
        """
        from core.schemas import RetrievalResult
        
        # Get hits from each retriever
        all_hits = {}
        retrieval_times = {}
        
        for name, retriever in self.retrievers.items():
            print(f"🔍 Retrieving from {name}...")
            result = retriever.retrieve(query, config)
            all_hits[name] = result.hits
            retrieval_times[name] = result.elapsed_ms
            print(f"   Got {len(result.hits)} hits in {result.elapsed_ms}ms")
        
        # Fusion
        if self.fusion_method == "rrf":
            merged_hits = self._reciprocal_rank_fusion(all_hits)
        else:  # weighted_sum
            merged_hits = self._weighted_score_fusion(all_hits)
        
        # Sort by fused score and take top-k
        merged_hits.sort(key=lambda h: h.score, reverse=True)
        top_k = config.top_k_retrieve
        merged_hits = merged_hits[:top_k]
        
        total_time = sum(retrieval_times.values())
        
        return RetrievalResult(
            query_id=query.query_id,
            hits=merged_hits,
            elapsed_ms=total_time,
            metadata={
                "fusion_method": self.fusion_method,
                "weights": self.weights,
                "source_counts": {name: len(hits) for name, hits in all_hits.items()},
                "retrieval_times": retrieval_times
            }
        )
    
    def _weighted_score_fusion(self, all_hits: Dict[str, List[RetrieveHit]]) -> List[RetrieveHit]:
        """
        Weighted score fusion with score normalization.
        
        Steps:
        1. Normalize scores within each method (min-max to [0,1])
        2. Apply weights
        3. Merge by unit_id (sum weighted scores)
        """
        # Step 1: Normalize scores per method
        normalized_hits = {}
        
        for name, hits in all_hits.items():
            if not hits:
                normalized_hits[name] = []
                continue
            
            scores = [h.score for h in hits]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                # Min-max normalization to [0, 1]
                norm_hits = []
                for hit in hits:
                    norm_score = (hit.score - min_score) / score_range
                    norm_hit = RetrieveHit(
                        unit_id=hit.unit_id,
                        doc_id=hit.doc_id,
                        page_id=hit.page_id,
                        block_id=hit.block_id,
                        text=hit.text,
                        score=norm_score,
                        bbox=hit.bbox,
                        source=hit.source,
                        metadata={**hit.metadata, "original_score": hit.score, "source": name}
                    )
                    norm_hits.append(norm_hit)
                normalized_hits[name] = norm_hits
            else:
                # All scores equal, use uniform 0.5
                normalized_hits[name] = [
                    RetrieveHit(
                        unit_id=h.unit_id,
                        doc_id=h.doc_id,
                        page_id=h.page_id,
                        block_id=h.block_id,
                        text=h.text,
                        score=0.5,
                        bbox=h.bbox,
                        source=h.source,
                        metadata={**h.metadata, "original_score": h.score, "source": name}
                    ) for h in hits
                ]
        
        # Step 2 & 3: Apply weights and merge
        merged_scores = {}  # unit_id -> (total_score, contributing_sources, best_hit)
        
        for name, hits in normalized_hits.items():
            weight = self.weights[name]
            
            for hit in hits:
                weighted_score = hit.score * weight
                
                if hit.unit_id not in merged_scores:
                    merged_scores[hit.unit_id] = (weighted_score, [name], hit)
                else:
                    prev_score, sources, best_hit = merged_scores[hit.unit_id]
                    new_score = prev_score + weighted_score
                    sources.append(name)
                    merged_scores[hit.unit_id] = (new_score, sources, best_hit)
        
        # Build final hits
        final_hits = []
        for unit_id, (score, sources, base_hit) in merged_scores.items():
            fused_hit = RetrieveHit(
                unit_id=unit_id,
                doc_id=base_hit.doc_id,
                page_id=base_hit.page_id,
                block_id=base_hit.block_id,
                text=base_hit.text,
                score=score,
                bbox=base_hit.bbox,
                source="hybrid",
                metadata={
                    **base_hit.metadata,
                    "fusion_sources": sources,
                    "fusion_score": score,
                    "num_sources": len(sources)
                }
            )
            final_hits.append(fused_hit)
        
        return final_hits
    
    def _reciprocal_rank_fusion(self, all_hits: Dict[str, List[RetrieveHit]], k: int = 60) -> List[RetrieveHit]:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank_i)) for all methods
        Default k=60 from original RRF paper.
        """
        rrf_scores = {}  # unit_id -> (rrf_score, contributing_sources, best_hit)
        
        for name, hits in all_hits.items():
            for rank, hit in enumerate(hits, start=1):
                rrf_score = 1.0 / (k + rank)
                # Apply weight to RRF score
                weighted_rrf = rrf_score * self.weights[name]
                
                if hit.unit_id not in rrf_scores:
                    rrf_scores[hit.unit_id] = (weighted_rrf, [name], hit)
                else:
                    prev_score, sources, best_hit = rrf_scores[hit.unit_id]
                    new_score = prev_score + weighted_rrf
                    sources.append(name)
                    rrf_scores[hit.unit_id] = (new_score, sources, best_hit)
        
        # Build final hits
        final_hits = []
        for unit_id, (score, sources, base_hit) in rrf_scores.items():
            fused_hit = RetrieveHit(
                unit_id=unit_id,
                doc_id=base_hit.doc_id,
                page_id=base_hit.page_id,
                block_id=base_hit.block_id,
                text=base_hit.text,
                score=score,
                bbox=base_hit.bbox,
                source="hybrid",
                metadata={
                    **base_hit.metadata,
                    "fusion_sources": sources,
                    "rrf_score": score,
                    "num_sources": len(sources)
                }
            )
            final_hits.append(fused_hit)
        
        return final_hits
