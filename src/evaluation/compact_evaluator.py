"""Compact RAG evaluation orchestrator.

It reuses the project's existing three-stage evaluator for QA and adds only a
small set of service-ready metrics:
- claim_f1
- citation_support_rate
- abstention_accuracy
- operational latency/token metrics

Use QA mode when a reference/ground_truth exists.
Use service mode for production logs where references usually do not exist.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.compact_metrics import build_operational_metrics, compute_extra_metrics, quality_gate

logger = logging.getLogger(__name__)


def _obj_to_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _contexts_from_sample(sample: dict[str, Any]) -> list[str]:
    contexts = sample.get("contexts")
    if isinstance(contexts, list):
        return [str(c) for c in contexts]
    sources = sample.get("sources") or []
    return [str(s.get("content", "")) for s in sources if isinstance(s, dict) and s.get("content")]


class CompactRAGEvaluator:
    """Small wrapper around RAGAS + Judge + DeepEval + custom service metrics."""

    def __init__(self, judge_model: str = "vertex_ai/openai/gpt-4o-mini", thresholds: dict[str, float] | None = None):
        self.judge_model = judge_model
        self.thresholds = thresholds or {}

    def evaluate_qa(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a QA sample with reference answer/ground truth."""
        from src.evaluation.evaluator import RAGEvaluator

        question = str(sample.get("question") or sample.get("input") or "")
        answer = str(sample.get("answer") or sample.get("actual_output") or "")
        ground_truth = str(sample.get("ground_truth") or sample.get("reference") or sample.get("expected_output") or "")
        contexts = _contexts_from_sample(sample)

        core = RAGEvaluator(judge_model=self.judge_model).evaluate_single(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        result = {
            "sample_id": sample.get("id") or sample.get("request_id"),
            "mode": "qa",
            "ragas": _obj_to_dict(core.ragas),
            "judge": _obj_to_dict(core.judge),
            "safety": _obj_to_dict(core.safety),
            "extra": compute_extra_metrics(sample),
            "ops": build_operational_metrics(sample),
            "eval_latency_seconds": core.latency,
        }
        result["quality_gate"] = quality_gate(result, self.thresholds)
        return result

    def evaluate_service(self, sample: dict[str, Any], run_llm_metrics: bool = True) -> dict[str, Any]:
        """Evaluate a live service log sample.

        Service mode can run without a reference answer. If run_llm_metrics=False,
        only deterministic extra/ops metrics are calculated.
        """
        question = str(sample.get("question") or sample.get("input") or "")
        answer = str(sample.get("answer") or sample.get("actual_output") or "")
        contexts = _contexts_from_sample(sample)

        ragas_result: dict[str, Any] | None = None
        judge_result: dict[str, Any] | None = None
        safety_result: dict[str, Any] | None = None

        if run_llm_metrics:
            try:
                from src.evaluation.ragas_service_metrics import evaluate_ragas_service

                ragas_result = evaluate_ragas_service(question=question, contexts=contexts, answer=answer)
            except Exception:
                logger.exception("service RAGAS metrics failed")

            try:
                from src.evaluation.llm_judge import judge_response

                judge_result = _obj_to_dict(
                    judge_response(question=question, contexts=contexts, answer=answer, judge_model=self.judge_model)
                )
            except Exception:
                logger.exception("service LLM judge failed")

            try:
                from src.evaluation.safety_metrics import evaluate_safety

                safety_result = _obj_to_dict(evaluate_safety(question=question, contexts=contexts, answer=answer))
            except Exception:
                logger.exception("service safety metrics failed")

        result = {
            "sample_id": sample.get("request_id") or sample.get("id"),
            "mode": "service",
            "ragas": ragas_result,
            "judge": judge_result,
            "safety": safety_result,
            "extra": compute_extra_metrics(sample),
            "ops": build_operational_metrics(sample),
        }
        result["quality_gate"] = quality_gate(result, self.thresholds)
        return result

    def evaluate_batch(self, samples: list[dict[str, Any]], mode: str, run_llm_metrics: bool = True) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples, start=1):
            sample_id = sample.get("id") or sample.get("request_id") or f"sample_{idx}"
            try:
                if mode == "qa":
                    eval_result = self.evaluate_qa(sample)
                elif mode == "service":
                    eval_result = self.evaluate_service(sample, run_llm_metrics=run_llm_metrics)
                else:
                    raise ValueError(f"unsupported mode: {mode}")
                results.append({**sample, "compact_eval": eval_result})
                logger.info("[%d/%d] evaluated: %s", idx, len(samples), sample_id)
            except Exception as exc:
                logger.exception("[%d/%d] evaluation failed: %s", idx, len(samples), sample_id)
                results.append({**sample, "compact_eval": None, "eval_error": str(exc)})
        return results


def summarize_compact_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a simple average summary for compact evaluation results."""
    buckets: dict[str, list[float]] = {}
    total = len(results)
    errors = 0
    passed = 0
    pass_checked = 0

    def add(key: str, value: Any) -> None:
        if isinstance(value, (int, float)):
            buckets.setdefault(key, []).append(float(value))

    for row in results:
        ev = row.get("compact_eval")
        if not ev:
            errors += 1
            continue
        gate = ev.get("quality_gate") or {}
        if gate.get("passed") is not None:
            pass_checked += 1
            passed += 1 if gate.get("passed") else 0
        for block_name in ("ragas", "judge", "safety", "extra", "ops"):
            block = ev.get(block_name) or {}
            for key, value in block.items():
                add(f"{block_name}.{key}", value)

    return {
        "total_samples": total,
        "errors": errors,
        "quality_gate_pass_rate": None if pass_checked == 0 else round(passed / pass_checked, 4),
        "averages": {key: round(sum(values) / len(values), 4) for key, values in sorted(buckets.items()) if values},
    }


def write_compact_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = output_dir / f"compact_eval_{run_id}_{timestamp}.json"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "metadata": metadata or {},
        "summary": summarize_compact_results(results),
        "details": results,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path
