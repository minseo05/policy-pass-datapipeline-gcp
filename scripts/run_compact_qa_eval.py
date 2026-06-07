"""Run compact QA evaluation.

Input file formats:
- JSONL: one sample per line
- JSON: list[dict] or {"samples": list[dict]}

Required fields if --generate-missing is not used:
- question
- answer
- contexts or sources
- ground_truth/reference/expected_output

Optional fields:
- expected_claims: list[str]
- citations: list[{claim, source_ids}]
- expected_abstain: bool
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.evaluation.compact_evaluator import CompactRAGEvaluator, write_compact_report

logger = logging.getLogger(__name__)


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("QA input must be a JSON list, {'samples': [...]}, or JSONL")
    return rows[:limit] if limit else rows


def load_config(path: Path, section: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(section, {})


def maybe_generate_missing(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if sample.get("answer") and (sample.get("contexts") or sample.get("sources")):
        return sample
    if not args.generate_missing:
        raise ValueError(f"sample {sample.get('id')} has no answer/contexts. Use --generate-missing.")

    from config.models import resolve_model_key
    from src.generation.pipeline import RAGPipeline

    model = resolve_model_key(args.model) or args.model
    pipeline = RAGPipeline(index_dir=args.index_dir, default_model=model)
    response = pipeline.run(
        query=sample["question"],
        model=model,
        strategy=args.strategy,
        top_k=args.top_k,
    )
    usage = response.llm_response
    return {
        **sample,
        "answer": response.answer,
        "contexts": [source.get("content", "") for source in response.sources],
        "sources": response.sources,
        "model": response.model,
        "strategy": response.search_strategy,
        "retrieval_latency": response.retrieval_latency,
        "generation_latency": response.generation_latency,
        "tokens": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact QA RAG evaluation")
    parser.add_argument("--qa-path", type=Path, default=Path("data/eval/qa_pairs.json"))
    parser.add_argument("--config", type=Path, default=Path("config/evaluation_compact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/compact_qa"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--strategy", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config, "qa")
    judge_model = args.judge_model or config.get("judge_model") or "vertex_ai/openai/gpt-4o-mini"
    thresholds = config.get("thresholds", {})
    run_id = args.run_id or datetime.now(timezone.utc).strftime("qa_%Y%m%dT%H%M%S")

    samples = [maybe_generate_missing(s, args) for s in load_samples(args.qa_path, args.limit)]
    evaluator = CompactRAGEvaluator(judge_model=judge_model, thresholds=thresholds)
    results = evaluator.evaluate_batch(samples=samples, mode="qa")
    report_path = write_compact_report(
        results=results,
        output_dir=args.output_dir,
        run_id=run_id,
        metadata={
            "mode": "qa",
            "qa_path": str(args.qa_path),
            "judge_model": judge_model,
            "generate_missing": args.generate_missing,
            "sample_count": len(samples),
        },
    )
    print(f"Compact QA evaluation report: {report_path}")


if __name__ == "__main__":
    main()
