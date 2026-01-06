# core/interfaces.py
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable
from .schemas import (
    AppConfig, DocumentMeta, PageArtifact, IndexBuildStats, IndexUnit,
    QueryInput, RetrievalResult, EvidenceSelectionResult,
    GenerationRequest, GenerationResult, QueryRunRecord,
    EvalDataset, EvalReport
)


@runtime_checkable
class DocumentStore(Protocol):
    """Persistence for document artifacts + metadata."""
    def save_document(self, meta: DocumentMeta) -> None: ...
    def get_document(self, doc_id: str) -> Optional[DocumentMeta]: ...
    def list_documents(self) -> List[DocumentMeta]: ...
    def delete_document(self, doc_id: str) -> None: ...

    def save_page_artifact(self, artifact: PageArtifact) -> None: ...
    def load_page_artifact(self, doc_id: str, page_id: int) -> Optional[PageArtifact]: ...


@runtime_checkable
class Ingestor(Protocol):
    """Import PDF/images into structured artifacts."""
    def ingest(self, source_path: str, config: AppConfig) -> DocumentMeta: ...
    def build_page_artifacts(self, doc_id: str, config: AppConfig) -> List[PageArtifact]: ...


@runtime_checkable
class Indexer(Protocol):
    """Build/search indices from index units."""
    def build_units(self, doc_id: str, config: AppConfig) -> List[IndexUnit]: ...
    def build_index(self, units: List[IndexUnit], config: AppConfig) -> IndexBuildStats: ...
    def persist(self, config: AppConfig) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieve candidate units for a query."""
    def retrieve(self, query: QueryInput, config: AppConfig) -> RetrievalResult: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional reranking; can be a no-op for V0."""
    def rerank(self, query: QueryInput, retrieval: RetrievalResult, config: AppConfig) -> RetrievalResult: ...


@runtime_checkable
class EvidenceSelector(Protocol):
    """Select final evidence list from retrieval hits."""
    def select(self, query: QueryInput, retrieval: RetrievalResult, config: AppConfig) -> EvidenceSelectionResult: ...


@runtime_checkable
class Generator(Protocol):
    """Generate answer based on evidence + policy."""
    def generate(self, req: GenerationRequest) -> GenerationResult: ...


@runtime_checkable
class RunLogger(Protocol):
    """Persist query-level run records for traceability."""
    def save_run(self, record: QueryRunRecord) -> str: ...
    def load_run(self, run_id: str) -> QueryRunRecord: ...


@runtime_checkable
class Evaluator(Protocol):
    """Batch evaluation over a dataset."""
    def evaluate(self, dataset: EvalDataset, config: AppConfig) -> EvalReport: ...
