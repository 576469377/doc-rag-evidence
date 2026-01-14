# core/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field


# ---------- Common types ----------
BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized [0,1] or absolute px; define in config


class AppConfig(BaseModel):
    # Paths
    data_root: str = Field(default="data", description="Root directory for all runtime artifacts")
    docs_dir: str = Field(default="data/docs", description="Document artifacts directory")
    indices_dir: str = Field(default="data/indices", description="Indices directory")
    runs_dir: str = Field(default="data/runs", description="Per-query run logs directory")
    reports_dir: str = Field(default="data/reports", description="Evaluation reports directory")

    # Granularity
    chunk_level: Literal["page", "block"] = Field(default="block", description="Index/retrieval unit")

    # Retrieval params
    top_k_retrieve: int = Field(default=20)
    top_k_rerank: int = Field(default=10)
    top_k_evidence: int = Field(default=5)

    # Evidence / citation policy
    citation_level: Literal["page", "block"] = Field(default="block")
    bbox_mode: Literal["none", "normalized", "absolute_px"] = Field(default="none")

    # Model providers (V0 can be 'stub' for offline demo)
    embedder_name: str = Field(default="stub-embedder")
    reranker_name: str = Field(default="none")
    llm_name: str = Field(default="stub-llm")
    
    # V1.1+: Generator and LLM configuration
    generator: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "template"  # template | qwen3_vl
    })
    
    llm: Dict[str, Any] = Field(default_factory=lambda: {
        "backend": "transformers",  # transformers | vllm | sglang
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_path": None,
        "endpoint": "http://localhost:8002",
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "citation_policy": "strict"  # strict | relaxed | none
    })

    # Safety / behavior
    max_context_chars: int = Field(default=12000, description="Hard cap for context assembly")
    require_citations: bool = Field(default=True, description="Force output citations if possible")

    # V0.1+: OCR configuration
    ocr: Dict[str, Any] = Field(default_factory=lambda: {
        "provider": "sglang",
        "model": "deepseek_ocr",
        "endpoint": "http://127.0.0.1:30000",
        "timeout": 60,
        "cache_enabled": True
    })

    # V0.1+: Dense embedding retrieval
    dense: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "embedder_type": "sglang",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "endpoint": "http://127.0.0.1:30000",
        "index_type": "Flat",
        "nlist": 100,
        "nprobe": 10,
        "batch_size": 32
    })

    # V0.1+: ColPali vision retrieval
    colpali: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "model": "vidore/colqwen2-v0.1",
        "device": "cuda:0",
        "max_global_pool": 100,
        "cache_dir": "data/cache/colpali"
    })

    # V0.1+: Retrieval mode selection
    retrieval_mode: Literal["bm25", "dense", "colpali", "hybrid"] = Field(default="bm25")


# ---------- Document artifacts ----------
class DocumentMeta(BaseModel):
    doc_id: str
    title: str = ""
    source_path: str
    sha256: Optional[str] = None
    created_at: str  # ISO time string
    page_count: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class PageMeta(BaseModel):
    """Lightweight page metadata."""
    doc_id: str
    page_id: int
    has_text: bool = False
    has_image: bool = False


class PageText(BaseModel):
    doc_id: str
    page_id: int
    text: str = ""  # raw extracted or OCR output
    language: Optional[str] = None


class Block(BaseModel):
    doc_id: str
    page_id: int
    block_id: str  # stable ID, e.g., "p0003_b0007"
    text: str
    bbox: Optional[BBox] = None
    block_type: Literal["paragraph", "table", "figure", "header", "footer", "other"] = "paragraph"
    score_hint: Optional[float] = None  # optional, for parsing confidence etc.


class PageArtifact(BaseModel):
    doc_id: str
    page_id: int
    image_path: Optional[str] = None  # rendered page image
    text: Optional[PageText] = None
    blocks: List[Block] = Field(default_factory=list)
    layout_path: Optional[str] = None  # e.g., layout.json


# ---------- Index units ----------
class IndexUnit(BaseModel):
    # The atomic unit to be indexed/retrieved
    unit_id: str  # e.g., "p0003_b0007" or "p0003"
    doc_id: str
    page_id: int
    block_id: Optional[str] = None
    text: str
    bbox: Optional[BBox] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexBuildStats(BaseModel):
    doc_count: int
    page_count: int
    unit_count: int
    elapsed_ms: int
    index_type: Literal["bm25", "faiss", "hybrid"]


# ---------- Query / retrieval / rerank ----------
class QueryInput(BaseModel):
    query_id: str
    question: str
    doc_filter: Optional[List[str]] = None  # restrict to specific doc_ids
    user_meta: Dict[str, Any] = Field(default_factory=dict)


class RetrieveHit(BaseModel):
    unit_id: str
    doc_id: str
    page_id: int
    block_id: Optional[str] = None
    text: str
    score: float
    bbox: Optional[BBox] = None
    source: Literal["bm25", "dense", "colpali", "hybrid", "rerank"] = "bm25"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query_id: str
    hits: List[RetrieveHit] = Field(default_factory=list)
    elapsed_ms: int = 0


class EvidenceItem(BaseModel):
    # Final evidence used for generation + citation
    rank: int
    unit_id: str
    doc_id: str
    page_id: int
    block_id: Optional[str] = None
    snippet: str
    bbox: Optional[BBox] = None
    score: float
    source: Literal["bm25", "dense", "colpali", "hybrid", "rerank"] = "bm25"
    rationale: Optional[str] = None  # optional explanation from selector
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceSelectionResult(BaseModel):
    query_id: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    elapsed_ms: int = 0


# ---------- Generation ----------
class PromptPackage(BaseModel):
    system_prompt: str
    user_prompt: str
    context: str
    citations: List[EvidenceItem] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    query_id: str
    question: str
    evidence: List[EvidenceItem]
    config: AppConfig


class GenerationOutput(BaseModel):
    answer: str
    # citations returned by generator (should map to evidence items)
    cited_units: List[str] = Field(default_factory=list)  # list of unit_id
    warnings: List[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    query_id: str
    output: GenerationOutput
    prompt: Optional[PromptPackage] = None
    elapsed_ms: int = 0


# ---------- Run log (traceability) ----------
class RunStatus(BaseModel):
    ok: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack: Optional[str] = None


class QueryRunRecord(BaseModel):
    # One complete trace for one question
    query: QueryInput
    retrieval: Optional[RetrievalResult] = None
    evidence: Optional[EvidenceSelectionResult] = None
    generation: Optional[GenerationResult] = None
    status: RunStatus = Field(default_factory=RunStatus)

    started_at: str  # ISO
    finished_at: Optional[str] = None
    config_snapshot: AppConfig


# ---------- Evaluation ----------
class EvalItem(BaseModel):
    qid: str
    question: str
    answer_gt: Optional[str] = None
    citations_gt: Optional[List[Dict[str, Any]]] = None  # optional ground-truth evidence


class EvalDataset(BaseModel):
    name: str
    items: List[EvalItem]


class EvalMetrics(BaseModel):
    n: int
    # Minimal V0 metrics
    exact_match: Optional[float] = None
    contains_match: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    evidence_hit_rate: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class EvalRow(BaseModel):
    qid: str
    question: str
    answer_pred: str
    cited_units: List[str] = Field(default_factory=list)
    latency_ms: int = 0
    status_ok: bool = True
    error_type: Optional[str] = None


class EvalReport(BaseModel):
    dataset_name: str
    created_at: str
    metrics: EvalMetrics
    rows: List[EvalRow] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)  # csv/json paths
