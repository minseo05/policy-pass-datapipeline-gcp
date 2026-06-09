"""RAG 생성 파이프라인 — 검색 + 프롬프트 + LLM 생성 오케스트레이션.

사용법:
    python -m src.generation.pipeline \
        --query "청년 월세 지원 신청 자격이 뭐야?" \
        --model openai/gpt-4o-mini \
        --strategy hybrid_rerank
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import settings
from src.generation import RAGResponse
from src.generation.llm_client import generate
from src.generation.prompt import build_no_rag_prompt, build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline, SearchStrategy

logger = logging.getLogger(__name__)

def _metadata_value(metadata: dict, *keys: str) -> str:
    """metadata에서 첫 번째로 존재하는 값을 문자열로 반환한다."""
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _build_source_identity(metadata: dict, rank: int) -> dict[str, str]:
    """평가와 citation 추적에 사용할 source 식별자를 만든다."""
    policy_id = _metadata_value(
        metadata,
        "policy_id",
        "policyId",
        "policy_no",
        "policyNo",
        "id",
    )

    chunk_id = _metadata_value(
        metadata,
        "chunk_id",
        "chunkId",
        "chunk_index",
        "chunk_idx",
    )

    source_id = _metadata_value(
        metadata,
        "source_id",
        "sourceId",
        "doc_id",
        "document_id",
    )

    if not source_id:
        source_id = chunk_id or policy_id or f"rank_{rank}"

    return {
        "source_id": source_id,
        "policy_id": policy_id,
        "chunk_id": chunk_id,
    }

class RAGPipeline:
    """검색 → 프롬프트 → 생성 통합 파이프라인."""

    def __init__(
        self,
        index_dir: str | Path | None = None,
        default_model: str | None = None,
        top_k: int | None = None,
    ):
        self.retrieval = RetrievalPipeline(index_dir=index_dir, top_k=top_k)
        self.default_model = default_model or settings.default_model

    def run(
        self,
        query: str,
        model: str | None = None,
        strategy: SearchStrategy | str = SearchStrategy.HYBRID_RERANK,
        top_k: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RAGResponse:
        """RAG 파이프라인 실행: 검색 → 프롬프트 → 생성."""
        model = model or self.default_model

        strategy_str = strategy if isinstance(strategy, str) else strategy.value

        retrieval_start = time.monotonic()
        results = self.retrieval.search(query, strategy=strategy, top_k=top_k)
        retrieval_latency = round(time.monotonic() - retrieval_start, 3)

        messages = build_rag_prompt(query, results)

        generation_start = time.monotonic()
        llm_response = generate(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        generation_latency = round(time.monotonic() - generation_start, 3)

        sources = []

        for r in results:
            identity = _build_source_identity(r.metadata, r.rank)

            sources.append(
                {
                    "source_id": identity["source_id"],
                    "policy_id": identity["policy_id"],
                    "chunk_id": identity["chunk_id"],
                    "content": r.content,
                    "title": r.metadata.get("title", ""),
                    "category": r.metadata.get("category", ""),
                    "source_name": r.metadata.get("source_name", ""),
                    "score": r.score,
                    "rank": r.rank,
                }
            )

        logger.info(
            "RAG 완료: model=%s, strategy=%s, sources=%d, retrieval=%.3fs, generation=%.3fs",
            model,
            strategy_str,
            len(sources),
            retrieval_latency,
            generation_latency,
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            model=model,
            search_strategy=strategy_str,
            llm_response=llm_response,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
        )

    def run_no_rag(
        self,
        query: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RAGResponse:
        """No-RAG 모드 — 비교 실험용, 검색 없이 LLM 직접 호출."""
        model = model or self.default_model
        messages = build_no_rag_prompt(query)

        llm_response = generate(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        return RAGResponse(
            answer=llm_response.content,
            sources=[],
            model=model,
            search_strategy="no_rag",
            llm_response=llm_response,
            retrieval_latency=0.0,
            generation_latency=llm_response.latency,
        )


def _resolve_model(model_key: str) -> str:
    """모델 키를 LiteLLM 모델 ID로 변환."""
    from config.models import resolve_model_key

    return resolve_model_key(model_key) or model_key


if __name__ == "__main__":
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="RAG 생성 파이프라인")
    parser.add_argument("--query", required=True, help="질문")
    parser.add_argument("--index-dir", default="data/index", help="FAISS 인덱스 디렉토리")
    parser.add_argument("--model", default=None, help="모델 (예: gpt-4o-mini, openai/gpt-4o-mini)")
    parser.add_argument(
        "--strategy",
        default="hybrid_rerank",
        choices=[s.value for s in SearchStrategy],
    )
    parser.add_argument("--no-rag", action="store_true", help="No-RAG 비교 모드")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    model_id = _resolve_model(args.model) if args.model else None

    pipeline = RAGPipeline(index_dir=args.index_dir, default_model=model_id)

    if args.no_rag:
        response = pipeline.run_no_rag(args.query, model=model_id)
    else:
        response = pipeline.run(args.query, model=model_id, strategy=args.strategy, top_k=args.top_k)

    print(f"\n{'=' * 60}")
    print(f"질문: {args.query}")
    print(f"모델: {response.model}")
    print(f"전략: {response.search_strategy}")
    print(f"검색 시간: {response.retrieval_latency:.3f}s")
    print(f"생성 시간: {response.generation_latency:.3f}s")
    print(f"{'=' * 60}")
    print(f"\n답변:\n{response.answer}\n")

    if response.sources:
        print(f"출처 ({len(response.sources)}건):")
        for s in response.sources:
            print(f"  - {s['title']} [{s['category']}] (score={s['score']:.4f})")

    if response.llm_response:
        lr = response.llm_response
        print(f"\n토큰: prompt={lr.prompt_tokens}, completion={lr.completion_tokens}, total={lr.total_tokens}")

    output = {
        "answer": response.answer,
        "model": response.model,
        "strategy": response.search_strategy,
        "sources": response.sources,
        "retrieval_latency": response.retrieval_latency,
        "generation_latency": response.generation_latency,
    }
    print(f"\n{json.dumps(output, ensure_ascii=False, indent=2)}")
