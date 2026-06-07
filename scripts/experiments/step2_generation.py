"""실험 B 데이터 수집: 멀티 LLM 답변 생성 — 5모델 × RAG + 1모델 × No-RAG.

검색 캐시 최적화: 동일 query+strategy의 검색 결과를 메모리 캐시하여 임베딩 호출 절감.

사용법:
    python -m scripts.experiments.step2_generation [--resume]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.models import MODELS, resolve_model_key
from scripts.experiments._common import (
    BASE_OUTPUT_DIR,
    CHECKPOINT_INTERVAL,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    EXPERIMENT_MODELS,
    NO_RAG_MODEL,
    CostTracker,
    Timer,
    load_latest_checkpoint,
    load_qa_samples,
    make_run_id,
    save_checkpoint,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_OUTPUT_DIR / "step2_generation"


def main(resume: bool = False) -> Path:
    setup_logging("step2_generation", OUTPUT_DIR)
    logger.info("=== 실험 B: 멀티 LLM 답변 생성 시작 ===")

    from src.generation.llm_client import generate
    from src.generation.prompt import build_no_rag_prompt, build_rag_prompt
    from src.retrieval.pipeline import RetrievalPipeline, SearchStrategy

    qa_samples = load_qa_samples()
    logger.info("QA 샘플 %d건 로드", len(qa_samples))

    retrieval = RetrievalPipeline()
    cost_tracker = CostTracker()
    strategy = SearchStrategy(DEFAULT_STRATEGY)

    conditions = [(f"{mk}__rag", resolve_model_key(mk), mk) for mk in EXPERIMENT_MODELS]
    conditions.append((f"{NO_RAG_MODEL}__no_rag", resolve_model_key(NO_RAG_MODEL), NO_RAG_MODEL))

    checkpoint_dir = OUTPUT_DIR / "checkpoint"
    results: dict[str, list] = {cond: [] for cond, _, _ in conditions}

    completed_keys: set[str] = set()
    if resume:
        ckpt, _ = load_latest_checkpoint(checkpoint_dir, "step2")
        if ckpt and isinstance(ckpt, dict) and "results" in ckpt:
            for cond, samples in ckpt["results"].items():
                results[cond] = samples
                for s in samples:
                    completed_keys.add(f"{cond}_{s['id']}")
            logger.info("체크포인트 복원: %d건 완료", len(completed_keys))

    search_cache: dict[str, list] = {}

    total_work = len(conditions) * len(qa_samples)
    done = len(completed_keys)

    for cond_key, model_id, model_key in conditions:
        is_no_rag = cond_key.endswith("__no_rag")

        for idx, sample in enumerate(qa_samples):
            sample_id = sample.get("id", f"q{idx:03d}")
            key = f"{cond_key}_{sample_id}"

            if key in completed_keys:
                done += 1
                continue

            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")

            retrieval_latency = 0.0
            contexts: list[str] = []
            search_results = []

            if not is_no_rag:
                if question in search_cache:
                    search_results = search_cache[question]
                else:
                    with Timer() as rt:
                        search_results = retrieval.search(question, strategy=strategy, top_k=DEFAULT_TOP_K)
                    retrieval_latency = rt.elapsed
                    search_cache[question] = search_results

                contexts = [r.content for r in search_results]
                messages = build_rag_prompt(question, search_results)
            else:
                messages = build_no_rag_prompt(question)

            try:
                with Timer() as gt:
                    mcfg = MODELS.get(model_key, {})
                    llm_resp = generate(
                        messages,
                        model=model_id,
                        temperature=mcfg.get("temperature", 0.0),
                        max_tokens=mcfg.get("max_tokens", 2048),
                    )
                generation_latency = gt.elapsed

                cost_tracker.record(
                    model=model_id,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                    latency=llm_resp.latency,
                    purpose="generation",
                )

                results[cond_key].append(
                    {
                        "id": sample_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": llm_resp.content,
                        "contexts": contexts,
                        "model": model_id,
                        "strategy": "no_rag" if is_no_rag else DEFAULT_STRATEGY,
                        "retrieval_latency": retrieval_latency,
                        "generation_latency": generation_latency,
                        "prompt_tokens": llm_resp.prompt_tokens,
                        "completion_tokens": llm_resp.completion_tokens,
                        "total_tokens": llm_resp.total_tokens,
                    }
                )

            except Exception:
                logger.exception("[%s] %s — 생성 실패", cond_key, sample_id)
                results[cond_key].append(
                    {
                        "id": sample_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": "",
                        "contexts": contexts,
                        "model": model_id,
                        "error": True,
                    }
                )

            done += 1
            if done % CHECKPOINT_INTERVAL == 0:
                save_checkpoint({"results": results}, checkpoint_dir, "step2", done)
                logger.info("[%d/%d] 체크포인트 저장", done, total_work)

        logger.info("조건 %s 완료 (%d건)", cond_key, len(results[cond_key]))

    output = {
        "run_id": make_run_id("step2"),
        "config": {
            "strategy": DEFAULT_STRATEGY,
            "top_k": DEFAULT_TOP_K,
            "models": EXPERIMENT_MODELS,
            "no_rag_model": NO_RAG_MODEL,
        },
        "results": results,
        "cost": cost_tracker.summary(),
    }

    output_path = OUTPUT_DIR / "generation_results.json"
    save_json(output, output_path)
    logger.info("=== 실험 B 완료: %s ===", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 B: 멀티 LLM 답변 생성")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    main(resume=args.resume)
