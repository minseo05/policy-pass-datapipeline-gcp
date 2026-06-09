"""Run compact evaluation for production/service RAG logs.

Expected input fields per log row:
- request_id or id
- question
- answer
- contexts or sources

Recommended optional fields:
- citations: list[{claim, source_ids}]
- should_abstain or no_answer_expected: bool
- retrieval_latency, generation_latency, latency
- tokens or token_usage

This script can run in two modes:
1. full mode: RAGAS faithfulness/relevancy + LLM Judge + DeepEval + extra metrics
2. cheap mode: only deterministic extra/ops metrics with --skip-llm-metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.evaluation.compact_evaluator import (  # noqa: E402
    CompactRAGEvaluator,
    write_compact_report,
)

logger = logging.getLogger(__name__)


def load_logs(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("logs") or data.get("samples") or data
    if not isinstance(rows, list):
        raise ValueError("service log input must be a JSON list, {'logs': [...]}, or JSONL")
    return rows[:limit] if limit else rows


def load_config(path: Path, section: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(section, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact service-log RAG evaluation")
    parser.add_argument("--log-path", type=Path, default=Path("data/eval/service_log_sample.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/evaluation_compact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/service_eval"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--skip-llm-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config, "service")
    judge_model = args.judge_model or config.get("judge_model") or "vertex_ai/openai/gpt-4o-mini"
    thresholds = config.get("thresholds", {})
    run_id = args.run_id or datetime.now(timezone.utc).strftime("service_%Y%m%dT%H%M%S")

    logs = load_logs(args.log_path, args.limit)
    evaluator = CompactRAGEvaluator(judge_model=judge_model, thresholds=thresholds)
    results = evaluator.evaluate_batch(samples=logs, mode="service", run_llm_metrics=not args.skip_llm_metrics)
    report_path = write_compact_report(
        results=results,
        output_dir=args.output_dir,
        run_id=run_id,
        metadata={
            "mode": "service",
            "log_path": str(args.log_path),
            "judge_model": judge_model,
            "run_llm_metrics": not args.skip_llm_metrics,
            "sample_count": len(logs),
        },
    )
    print(f"Compact service evaluation report: {report_path}")


if __name__ == "__main__":
    main()
