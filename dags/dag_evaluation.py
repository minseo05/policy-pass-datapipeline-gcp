"""평가 파이프라인 DAG.

스케줄: 수동 트리거 (schedule=None). 모델별 RAG 응답 생성 → 평가 수행.
QA 데이터셋이 REPO_ROOT/data/eval/qa_pairs.json에 존재해야 함.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param
from utils.notifications import on_failure_callback

from config.models import MODELS

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "rag-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "on_failure_callback": on_failure_callback,
}

REPO_ROOT = Path("/opt/rag-pipeline")
QA_DATASET_PATH = REPO_ROOT / "data" / "eval" / "qa_pairs.json"
RESULTS_DIR = REPO_ROOT / "data" / "eval" / "results"


def _resolve_model(model_key: str) -> str:
    """모델 키를 실제 LiteLLM 모델 ID로 변환한다."""
    return str(MODELS.get(model_key, {}).get("id", model_key))


@dag(
    dag_id="evaluation_pipeline",
    default_args=DEFAULT_ARGS,
    description="모델별 RAG 응답 생성 + 평가 (RAGAS / LLM Judge)",
    schedule=None,
    start_date=pendulum.datetime(2026, 4, 25, tz="Asia/Seoul"),
    catchup=False,
    tags=["evaluation", "manual"],
    params={
        "models": Param(
            default=["gpt-4o-mini", "gemini-flash"],
            type="array",
            description="평가할 모델 키 (config/models.py MODELS dict)",
        ),
        "strategy": Param(
            default="hybrid_rerank",
            type="string",
            enum=["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
            description="검색 전략",
        ),
        "max_samples": Param(
            default=0,
            type="integer",
            description="평가 샘플 수 (0=전체)",
        ),
    },
)
def evaluation_pipeline():
    @task()
    def load_qa_dataset(**context) -> list[dict]:
        """QA 데이터셋 로드."""
        if not QA_DATASET_PATH.exists():
            raise FileNotFoundError(f"QA 데이터셋 없음: {QA_DATASET_PATH}")

        with open(QA_DATASET_PATH) as f:
            dataset = json.load(f)

        samples = dataset.get("samples", [])
        max_samples = context["params"].get("max_samples", 0)
        if max_samples > 0:
            samples = samples[:max_samples]

        logger.info("QA 데이터셋 로드: %d건", len(samples))
        return samples

    @task()
    def generate_rag_responses(qa_samples: list[dict], **context) -> dict:
        """모델별 RAG 응답 생성."""
        from src.generation.pipeline import RAGPipeline

        models = context["params"]["models"]
        strategy = context["params"]["strategy"]

        pipeline = RAGPipeline()
        all_results: dict[str, list[dict]] = {}

        for model_key in models:
            model_id = _resolve_model(model_key)
            model_results: list[dict] = []
            error_count = 0
            for sample in qa_samples:
                query = sample["question"]
                try:
                    response = pipeline.run(
                        query=query,
                        model=model_id,
                        strategy=strategy,
                    )
                    llm_resp = response.llm_response
                    model_results.append(
                        {
                            **sample,
                            "id": sample["id"],
                            "question": query,
                            "ground_truth": sample.get("ground_truth", ""),
                            "answer": response.answer,
                            "model": response.model,
                            "strategy": strategy,
                            "retrieval_latency": response.retrieval_latency,
                            "generation_latency": response.generation_latency,
                            "token_usage": {
                                "prompt_tokens": llm_resp.prompt_tokens if llm_resp else 0,
                                "completion_tokens": llm_resp.completion_tokens if llm_resp else 0,
                                "total_tokens": llm_resp.total_tokens if llm_resp else 0,
                            },
                            "contexts": [s["content"] for s in response.sources[:3] if s.get("content")],
                            "sources": response.sources[:3],
                        }
                    )
                except Exception:
                    logger.exception("응답 생성 실패: model=%s, q=%s", model_id, query[:50])
                    error_count += 1
                    model_results.append(
                        {
                            "id": sample["id"],
                            "question": query,
                            "error": True,
                        }
                    )

            if error_count == len(qa_samples):
                raise RuntimeError(f"모델 {model_id}: 전체 {error_count}건 응답 생성 실패")

            all_results[model_key] = model_results
            logger.info("모델 %s: %d건 성공, %d건 실패", model_id, len(qa_samples) - error_count, error_count)

        return all_results

    @task()
    def evaluate_results(rag_results: dict, **context) -> dict:
        """3단계 평가 수행."""
        from src.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        run_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", context["run_id"])
        checkpoint_root = RESULTS_DIR / "checkpoints" / run_id

        evaluated_results: dict[str, list[dict]] = {}
        for model_key, samples in rag_results.items():
            evaluated_results[model_key] = evaluator.evaluate_batch(
                samples,
                checkpoint_dir=checkpoint_root / model_key,
            )

        return evaluated_results

    @task()
    def evaluate_compact_results(rag_results: dict, **context) -> dict:
        """Compact QA 평가 수행."""
        from src.evaluation.compact_evaluator import CompactRAGEvaluator

        config_path = REPO_ROOT / "config" / "evaluation_compact.json"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                compact_config = json.load(f).get("qa", {})
        else:
            compact_config = {}

        judge_model = compact_config.get("judge_model", "openai/gpt-4o-mini")
        thresholds = compact_config.get("thresholds", {})

        evaluator = CompactRAGEvaluator(
            judge_model=judge_model,
            thresholds=thresholds,
        )

        compact_results: dict[str, list[dict]] = {}

        for model_key, samples in rag_results.items():
            logger.info("Compact 평가 시작: model=%s, samples=%d", model_key, len(samples))
            compact_results[model_key] = evaluator.evaluate_batch(
                samples=samples,
                mode="qa",
            )

        return compact_results


    @task()
    def save_results(eval_results: dict, **context) -> str:
        """평가 결과 리포트 저장."""
        from src.evaluation.report import generate_report

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        run_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", context["run_id"])
        output_path = generate_report(
            eval_results,
            RESULTS_DIR,
            run_id=run_id,
            metadata={
                "airflow_run_id": context["run_id"],
                "strategy": context["params"]["strategy"],
                "models": context["params"]["models"],
            },
        )

        logger.info("결과 저장: %s", output_path)
        return str(output_path)

    @task()
    def save_compact_results(compact_results: dict, **context) -> dict[str, str]:
        """Compact 평가 결과 리포트 저장."""
        from src.evaluation.compact_evaluator import write_compact_report

        run_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", context["run_id"])
        compact_dir = RESULTS_DIR / "compact"

        output_paths: dict[str, str] = {}

        for model_key, results in compact_results.items():
            model_run_id = f"{run_id}_{model_key}"
            output_path = write_compact_report(
                results=results,
                output_dir=compact_dir,
                run_id=model_run_id,
                metadata={
                    "airflow_run_id": context["run_id"],
                    "mode": "qa",
                    "strategy": context["params"]["strategy"],
                    "model": model_key,
                    "sample_count": len(results),
                },
            )
            logger.info("Compact 결과 저장: model=%s, path=%s", model_key, output_path)
            output_paths[model_key] = str(output_path)

        return output_paths



    qa = load_qa_dataset()
    rag_results = generate_rag_responses(qa)
    eval_results = evaluate_results(rag_results)
    compact_results = evaluate_compact_results(rag_results)
    save_results(eval_results)
    save_compact_results(compact_results)


evaluation_pipeline()
