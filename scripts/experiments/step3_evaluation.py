"""실험 B 평가 + C/D 데이터 수집: 3단계 평가 — 2종 Judge 모델로 실행.

최적화: RAGAS/Safety는 1회만 실행, Judge만 2종(GPT-4o, GPT-4o-mini)으로 나눠 실행.

사용법:
    python -m scripts.experiments.step3_evaluation [--resume]
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

from scripts.experiments._common import (
    BASE_OUTPUT_DIR,
    CHECKPOINT_INTERVAL,
    CostTracker,
    Timer,
    load_json,
    load_latest_checkpoint,
    make_run_id,
    save_checkpoint,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_OUTPUT_DIR / "step3_evaluation"
GENERATION_RESULTS_PATH = BASE_OUTPUT_DIR / "step2_generation" / "generation_results.json"

JUDGE_PRIMARY = "openai/gpt-4o-mini"
JUDGE_EXPENSIVE = "vertex_ai/gemini-3.1-pro-preview"

SAMPLE_TIMEOUT = 180


class _SampleTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SampleTimeout("sample timeout")


def main(resume: bool = False, conditions: list[str] | None = None) -> tuple[Path, Path]:
    setup_logging("step3_evaluation", OUTPUT_DIR)
    logger.info("=== 실험 B+C+D: 3단계 평가 시작 ===")

    gen_data = load_json(GENERATION_RESULTS_PATH)
    gen_results = gen_data["results"]

    all_samples = []
    for cond_key, samples in gen_results.items():
        for s in samples:
            all_samples.append({**s, "condition": cond_key})

    existing_primary: list[dict] = []
    existing_expensive: list[dict] = []

    if conditions:
        cond_set = set(conditions)
        target_samples = [s for s in all_samples if s["condition"] in cond_set]
        logger.info("재평가 대상: %d건 (%s)", len(target_samples), ", ".join(conditions))

        primary_file = OUTPUT_DIR / "eval_gpt4o_mini_judge.json"
        expensive_file = OUTPUT_DIR / "eval_gemini_pro_judge.json"
        if primary_file.exists():
            existing_primary = [r for r in load_json(primary_file)["results"] if r["condition"] not in cond_set]
        if expensive_file.exists():
            existing_expensive = [r for r in load_json(expensive_file)["results"] if r["condition"] not in cond_set]

        resume = False
    else:
        target_samples = all_samples
        logger.info("평가 대상: %d건 (%d 조건)", len(target_samples), len(gen_results))

    primary_path = _run_evaluation(
        target_samples,
        judge_model=JUDGE_PRIMARY,
        label="gpt4o_mini",
        run_ragas=True,
        run_safety=True,
        resume=resume,
        existing_results=existing_primary,
    )

    primary_data = load_json(primary_path)
    ragas_safety_cache = _build_ragas_safety_cache(primary_data)

    expensive_path = _run_evaluation(
        target_samples,
        judge_model=JUDGE_EXPENSIVE,
        label="gemini_pro",
        run_ragas=False,
        run_safety=False,
        ragas_safety_cache=ragas_safety_cache,
        resume=resume,
        existing_results=existing_expensive,
    )

    logger.info("=== 실험 B+C+D 완료 ===")
    return primary_path, expensive_path


def _run_evaluation(
    samples: list[dict],
    judge_model: str,
    label: str,
    run_ragas: bool,
    run_safety: bool,
    ragas_safety_cache: dict[str, dict] | None = None,
    resume: bool = False,
    existing_results: list[dict] | None = None,
) -> Path:
    """단일 Judge 모델로 평가 실행."""
    from src.evaluation.llm_judge import judge_response
    from src.evaluation.ragas_metrics import _reset_ragas_clients, evaluate_ragas
    from src.evaluation.safety_metrics import evaluate_safety

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    cost_tracker = CostTracker()
    checkpoint_dir = OUTPUT_DIR / "checkpoint"
    output_file = OUTPUT_DIR / f"eval_{label}_judge.json"

    evaluated: list[dict] = []
    completed_ids: set[str] = set()

    if resume:
        ckpt, _ = load_latest_checkpoint(checkpoint_dir, f"step3_{label}")
        if ckpt and isinstance(ckpt, dict) and "results" in ckpt:
            evaluated = ckpt["results"]
            completed_ids = {f"{s['condition']}_{s['id']}" for s in evaluated}
            logger.info("[%s] 체크포인트 복원: %d건", label, len(completed_ids))

    total = len(samples)

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"unknown_{idx}")
        cond = sample.get("condition", "")
        unique_key = f"{cond}_{sample_id}"

        if unique_key in completed_ids:
            continue

        if sample.get("error"):
            evaluated.append({**sample, "eval": None, "eval_error": "생성 실패"})
            continue

        question = sample.get("question", "")
        contexts = sample.get("contexts", [])
        answer = sample.get("answer", "")
        ground_truth = sample.get("ground_truth", "")

        eval_entry: dict = {}

        signal.alarm(SAMPLE_TIMEOUT)
        try:
            if run_ragas:
                try:
                    ragas = evaluate_ragas(question, contexts, answer, ground_truth)
                    eval_entry["ragas"] = {
                        "faithfulness": ragas.faithfulness,
                        "answer_relevancy": ragas.answer_relevancy,
                        "context_precision": ragas.context_precision,
                        "context_recall": ragas.context_recall,
                    }
                except _SampleTimeout:
                    raise
                except Exception:
                    logger.exception("[%d/%d] %s RAGAS 실패", idx + 1, total, sample_id)
                    eval_entry["ragas"] = None
            elif ragas_safety_cache and unique_key in ragas_safety_cache:
                eval_entry["ragas"] = ragas_safety_cache[unique_key].get("ragas")
            else:
                eval_entry["ragas"] = None

            try:
                with Timer() as jt:
                    judge = judge_response(question, contexts, answer, judge_model=judge_model)

                eval_entry["judge"] = {
                    "citation_accuracy": judge.citation_accuracy,
                    "completeness": judge.completeness,
                    "readability": judge.readability,
                    "average": judge.average,
                    "raw_scores": _format_raw_scores(judge.raw_scores),
                }
                eval_entry["judge_latency"] = jt.elapsed
            except _SampleTimeout:
                raise
            except Exception:
                logger.exception("[%d/%d] %s Judge 실패", idx + 1, total, sample_id)
                eval_entry["judge"] = None
                eval_entry["judge_latency"] = 0.0

            if run_safety:
                try:
                    safety = evaluate_safety(question, contexts, answer)
                    eval_entry["safety"] = {"hallucination_score": safety.hallucination_score}
                except _SampleTimeout:
                    raise
                except Exception:
                    logger.exception("[%d/%d] %s Safety 실패", idx + 1, total, sample_id)
                    eval_entry["safety"] = None
            elif ragas_safety_cache and unique_key in ragas_safety_cache:
                eval_entry["safety"] = ragas_safety_cache[unique_key].get("safety")
            else:
                eval_entry["safety"] = None

        except _SampleTimeout:
            logger.error("[%d/%d] %s/%s — %ds 타임아웃, 건너뜀", idx + 1, total, cond, sample_id, SAMPLE_TIMEOUT)
            _reset_ragas_clients()
            eval_entry = {
                "ragas": eval_entry.get("ragas"),
                "judge": eval_entry.get("judge"),
                "safety": eval_entry.get("safety"),
            }
        finally:
            signal.alarm(0)

        evaluated.append({**sample, "eval": eval_entry})
        logger.info("[%d/%d] %s/%s — 평가 완료", idx + 1, total, cond, sample_id)

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint({"results": evaluated}, checkpoint_dir, f"step3_{label}", idx + 1)
            logger.info("[%s] 체크포인트 저장: %d건", label, len(evaluated))

    signal.signal(signal.SIGALRM, old_handler)

    merged = (existing_results or []) + evaluated

    output = {
        "run_id": make_run_id(f"step3_{label}"),
        "config": {"judge_model": judge_model, "source": str(GENERATION_RESULTS_PATH)},
        "results": merged,
        "cost": cost_tracker.summary(),
    }

    save_json(output, output_file)
    logger.info("[%s] 평가 완료: %s (%d건)", label, output_file, len(evaluated))
    return output_file


def _format_raw_scores(raw: tuple[dict[str, int], ...]) -> dict:
    if len(raw) >= 2:
        return {"original_order": raw[0], "shuffled_order": raw[1]}
    if len(raw) == 1:
        return {"original_order": raw[0], "shuffled_order": {}}
    return {"original_order": {}, "shuffled_order": {}}


def _build_ragas_safety_cache(primary_data: dict) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for sample in primary_data.get("results", []):
        cond = sample.get("condition", "")
        sid = sample.get("id", "")
        key = f"{cond}_{sid}"
        ev = sample.get("eval")
        if ev:
            cache[key] = {"ragas": ev.get("ragas"), "safety": ev.get("safety")}
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 B+C+D: 3단계 평가")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="재평가할 조건 (콤마 구분, 예: gpt-5.4-mini__rag,gemini-3.5-flash__rag)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    cond_list = args.conditions.split(",") if args.conditions else None
    main(resume=args.resume, conditions=cond_list)
