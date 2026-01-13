# core/pipeline.py
from __future__ import annotations

from typing import Optional
from .schemas import (
    AppConfig, QueryInput, RetrievalResult, EvidenceSelectionResult,
    GenerationRequest, GenerationResult, QueryRunRecord, RunStatus
)
from .inferences import Retriever, Reranker, EvidenceSelector, Generator, RunLogger


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        selector: EvidenceSelector,
        generator: Generator,
        logger: RunLogger,
        reranker: Optional[Reranker] = None,
        store = None,  # For hit normalization
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.selector = selector
        self.generator = generator
        self.logger = logger
        self.store = store
        
        # Initialize hit normalizer if store available
        self.normalizer = None
        if store:
            from impl.hit_normalizer import HitNormalizer
            self.normalizer = HitNormalizer(store)

    def answer(self, query: QueryInput, config: AppConfig) -> QueryRunRecord:
        # You can add real timing; V0 keeps schema stable.
        record = QueryRunRecord(
            query=query,
            retrieval=None,
            evidence=None,
            generation=None,
            status=RunStatus(ok=True),
            started_at=_iso_now(),
            finished_at=None,
            config_snapshot=config,
        )
        try:
            # Step 1: Retrieve (raw hits from retriever)
            retrieval: RetrievalResult = self.retriever.retrieve(query, config)
            
            # Step 2: Normalize hits (page → block expansion for ColPali/page-level retrievers)
            if self.normalizer and retrieval.hits:
                # Detect if normalization needed (check if any hit lacks text)
                needs_normalization = any(
                    not hit.text or not hit.text.strip() 
                    for hit in retrieval.hits
                )
                
                if needs_normalization:
                    # Determine source from first hit
                    source = retrieval.hits[0].source if retrieval.hits else "unknown"
                    original_hit_count = len(retrieval.hits)
                    
                    normalized_hits = self.normalizer.normalize_hits(
                        retrieval.hits,
                        config,
                        query.question,
                        source
                    )
                    retrieval.hits = normalized_hits
                    
                    # Log normalization info
                    print(f"🔄 Hit normalization: {original_hit_count} page hits → {len(normalized_hits)} block hits")
            
            # Step 3: Optional reranking
            if self.reranker is not None and config.top_k_rerank > 0:
                retrieval = self.reranker.rerank(query, retrieval, config)
            
            record.retrieval = retrieval

            # Step 4: Evidence selection
            evidence: EvidenceSelectionResult = self.selector.select(query, retrieval, config)
            record.evidence = evidence

            # Step 5: Generation
            req = GenerationRequest(
                query_id=query.query_id,
                question=query.question,
                evidence=evidence.evidence,
                config=config,
            )
            gen: GenerationResult = self.generator.generate(req)
            record.generation = gen

            record.finished_at = _iso_now()
            self.logger.save_run(record)
            return record

        except Exception as e:
            record.status = RunStatus(ok=False, error_type=type(e).__name__, error_message=str(e))
            record.finished_at = _iso_now()
            self.logger.save_run(record)
            return record


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
