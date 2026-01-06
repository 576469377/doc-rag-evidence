# core/pipeline.py
from __future__ import annotations

from typing import Optional
from .schemas import (
    AppConfig, QueryInput, RetrievalResult, EvidenceSelectionResult,
    GenerationRequest, GenerationResult, QueryRunRecord, RunStatus
)
from .interfaces import Retriever, Reranker, EvidenceSelector, Generator, RunLogger


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        selector: EvidenceSelector,
        generator: Generator,
        logger: RunLogger,
        reranker: Optional[Reranker] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.selector = selector
        self.generator = generator
        self.logger = logger

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
            retrieval: RetrievalResult = self.retriever.retrieve(query, config)
            if self.reranker is not None and config.top_k_rerank > 0:
                retrieval = self.reranker.rerank(query, retrieval, config)
            record.retrieval = retrieval

            evidence: EvidenceSelectionResult = self.selector.select(query, retrieval, config)
            record.evidence = evidence

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
