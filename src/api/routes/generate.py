"""생성 엔드포인트."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from config.models import resolve_model_key
from src.api.auth import require_api_key
from src.api.costs import estimate_cost_usd
from src.api.deps import get_mongo, get_rag_pipeline
from src.api.logging_config import log_structured
from src.api.monitoring import get_monitoring_client
from src.api.rate_limit import limiter
from src.api.schemas import GenerateRequest, GenerateResponse, SourceItem, TokenUsage
from src.generation.llm_client import LLMError
from src.generation.pipeline import RAGPipeline
from src.ingestion.mongo_client import PolicyMetadataStore

logger = logging.getLogger(__name__)
usage_logger = logging.getLogger("api.rag")
router = APIRouter(prefix="/api/v1", tags=["generate"], dependencies=[Depends(require_api_key)])

def _to_log_dict(item: object) -> dict:
    """Pydantic v1/v2 모델을 로그용 dict로 변환한다."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    if isinstance(item, dict):
        return item
    return {}


def _truncate_text(text: str, max_length: int = 4000) -> str:
    """Cloud Logging 로그 크기와 개인정보 노출 위험을 줄이기 위해 긴 텍스트를 제한한다."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"


def _prepare_sources_for_eval_log(sources: list[SourceItem]) -> list[dict]:
    """서비스 로그 평가에 사용할 sources 구조를 만든다."""
    prepared = []

    for source in sources:
        source_dict = _to_log_dict(source)

        if "content" in source_dict and isinstance(source_dict["content"], str):
            source_dict["content"] = _truncate_text(source_dict["content"])

        prepared.append(source_dict)

    return prepared

@router.post("/generate", response_model=GenerateResponse)
@limiter.limit("30/minute")
def generate(
    request: Request,
    body: GenerateRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    mongo: PolicyMetadataStore | None = Depends(get_mongo),
) -> GenerateResponse:
    model_id = resolve_model_key(body.model)

    try:
        if body.no_rag:
            resp = pipeline.run_no_rag(
                query=body.query,
                model=model_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        else:
            resp = pipeline.run(
                query=body.query,
                model=model_id,
                strategy=body.strategy.value,
                top_k=body.top_k,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception:
        logger.exception("generate 엔드포인트 예기치 않은 오류 (query=%s)", body.query[:50])
        raise

    sources = [
        SourceItem(
            source_id=s.get("source_id", ""),
            policy_id=s.get("policy_id", ""),
            chunk_id=s.get("chunk_id", ""),
            content=s["content"],
            title=s.get("title", ""),
            category=s.get("category", ""),
            source_name=s.get("source_name", ""),
            score=s.get("score", 0.0),
            rank=s.get("rank", 0),
        )
        for s in resp.sources
    ]

    lr = resp.llm_response
    usage = TokenUsage(
        prompt_tokens=lr.prompt_tokens if lr else 0,
        completion_tokens=lr.completion_tokens if lr else 0,
        total_tokens=lr.total_tokens if lr else 0,
    )

    total_latency = resp.retrieval_latency + resp.generation_latency
    retrieval_latency_ms = round(resp.retrieval_latency * 1000, 1)
    generation_latency_ms = round(resp.generation_latency * 1000, 1)
    total_latency_ms = round(total_latency * 1000, 1)
    estimated_cost = estimate_cost_usd(resp.model, usage.prompt_tokens, usage.completion_tokens)
    request_id = getattr(request.state, "request_id", "")

    log_structured(
        usage_logger,
        logging.INFO,
        "rag_request",
        event="rag_request",
        request_id=request_id,
        query=body.query[:200],
        model=resp.model,
        strategy=resp.search_strategy,
        retrieval_ms=retrieval_latency_ms,
        generation_ms=generation_latency_ms,
        total_ms=total_latency_ms,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        estimated_cost_usd=estimated_cost,
        source_count=len(sources),
        status="success",
    )

    log_structured(
        usage_logger,
        logging.INFO,
        "rag_eval_log",
        event="rag_eval_log",
        request_id=request_id,
        question=body.query,
        answer=_truncate_text(resp.answer),
        model=resp.model,
        strategy=resp.search_strategy,
        top_k=body.top_k,
        no_rag=body.no_rag,
        sources=_prepare_sources_for_eval_log(sources),
        citations=[],
        should_abstain=None,
        retrieval_latency=resp.retrieval_latency,
        generation_latency=resp.generation_latency,
        latency=total_latency,
        token_usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        source_count=len(sources),
        status="success",
    )

    get_monitoring_client().record_generation(
        model=resp.model,
        strategy=resp.search_strategy,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        tokens_used=usage.total_tokens,
        estimated_cost_usd=estimated_cost,
    )
    if mongo:
        try:
            mongo.log_api_usage(
                {
                    "request_id": request_id,
                    "model": resp.model,
                    "tokens_in": usage.prompt_tokens,
                    "tokens_out": usage.completion_tokens,
                    "cost_usd": estimated_cost,
                    "latency_ms": total_latency_ms,
                    "strategy": resp.search_strategy,
                    "status": "success",
                }
            )
        except Exception as exc:
            logger.debug("api_usage_logs insert failed: %s", exc)

    return GenerateResponse(
        answer=resp.answer,
        sources=sources,
        model=resp.model,
        strategy=resp.search_strategy,
        token_usage=usage,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )
