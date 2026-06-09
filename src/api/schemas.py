"""API 요청/응답 Pydantic 모델."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SearchStrategyEnum(str, Enum):
    vector_only = "vector_only"
    bm25_only = "bm25_only"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


# ── Health ──────────────────────────────────────────────────


class SourceStatus(BaseModel):
    last_run: str | None = None
    status: str | None = None
    count: int = 0


class DataPipelineStatus(BaseModel):
    last_ingestion: str | None = None
    total_policies: int = 0
    index_sync_status: str = "unknown"
    sources: dict[str, SourceStatus] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    faiss_loaded: bool
    faiss_doc_count: int
    faiss_dim: int | None = None
    faiss_last_updated: str | None = None
    mongodb_connected: bool
    gcs_accessible: bool | None = None
    uptime_seconds: float
    version: str
    data_pipeline: DataPipelineStatus | None = None


# ── Search ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    strategy: SearchStrategyEnum = SearchStrategyEnum.hybrid
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResultItem(BaseModel):
    content: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)
    rank: int


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    strategy: str
    total: int
    latency_ms: float


# ── Generate ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    model: str | None = Field(None, description="모델 키 (예: gpt-4o-mini) 또는 전체 모델 ID")
    strategy: SearchStrategyEnum = SearchStrategyEnum.hybrid
    top_k: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=100, le=4096)
    no_rag: bool = False


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class SourceItem(BaseModel):
    source_id: str = ""
    policy_id: str = ""
    chunk_id: str = ""
    content: str
    title: str = ""
    category: str = ""
    source_name: str = ""
    score: float = 0.0
    rank: int = 0


class GenerateResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model: str
    strategy: str
    token_usage: TokenUsage
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


# ── Policies ────────────────────────────────────────────────


class PolicyItem(BaseModel):
    policy_id: str
    title: str = ""
    category: str = ""
    summary: str | None = None
    description: str | None = None
    eligibility: str | None = None
    benefits: str | None = None
    how_to_apply: str | None = None
    application_period: str | None = None
    managing_department: str | None = None
    region: str | None = None
    source_url: str | None = None
    source_name: str | None = None
    last_updated: str | None = None


class PoliciesResponse(BaseModel):
    policies: list[PolicyItem]
    total: int
    page: int
    limit: int


# ── Models ──────────────────────────────────────────────────


class ModelInfo(BaseModel):
    key: str
    model_id: str
    description: str
    temperature: float
    max_tokens: int


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


# ── Evaluate ────────────────────────────────────────────────


class EvalSample(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=10000)
    ground_truth: str = Field(..., min_length=1, max_length=10000)
    contexts: list[str] = Field(..., min_length=1, max_length=20)


class EvalRequest(BaseModel):
    samples: list[EvalSample] = Field(..., min_length=1, max_length=20)
    judge_model: str | None = None


class RagasScores(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class JudgeScores(BaseModel):
    citation_accuracy: float = 0.0
    completeness: float = 0.0
    readability: float = 0.0
    average: float = 0.0


class SafetyScores(BaseModel):
    hallucination_score: float | None = None


class EvalResultItem(BaseModel):
    id: str
    ragas: RagasScores | None = None
    judge: JudgeScores | None = None
    safety: SafetyScores | None = None
    latency: float = 0.0
    error: str | None = None


class EvalResponse(BaseModel):
    results: list[EvalResultItem]
    total: int
    evaluated: int
    errors: int


# ── Error ───────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    status_code: int
