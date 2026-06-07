"""실험 파이프라인 공유 유틸리티 — 체크포인트, 비용 추적, JSON I/O."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BASE_OUTPUT_DIR = Path("data/experiments")

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-5.4-mini": (0.40, 1.60),
    "vertex_ai/gemini-2.5-flash": (0.15, 0.60),
    "vertex_ai/gemini-2.5-pro": (1.25, 5.00),
    "vertex_ai/gemini-3.5-flash": (1.50, 9.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "huggingface/meta-llama/Llama-3.3-70B-Instruct": (0.0, 0.0),
    "vertex_ai/gemini-3.1-pro-preview": (2.00, 12.00),
}

EXPERIMENT_MODELS = ["gpt-4o-mini", "gpt-4o", "claude-sonnet", "gemini-flash", "llama3", "gpt-5.4-mini", "gemini-3.5-flash"]
NO_RAG_MODEL = "gpt-4o-mini"
DEFAULT_STRATEGY = "hybrid_rerank"
DEFAULT_TOP_K = 5
QA_PATH = Path("data/eval/qa_pairs.json")
CHECKPOINT_INTERVAL = 10


@dataclass(frozen=True)
class CostRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    purpose: str
    timestamp: str = ""


@dataclass
class CostTracker:
    records: list[CostRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        purpose: str,
    ) -> None:
        self.records.append(
            CostRecord(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency,
                purpose=purpose,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    def summary(self) -> dict:
        by_model: dict[str, dict[str, int]] = {}
        by_purpose: dict[str, int] = {}

        for r in self.records:
            m = by_model.setdefault(r.model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens
            m["calls"] += 1
            by_purpose[r.purpose] = by_purpose.get(r.purpose, 0) + 1

        return {
            "total_calls": len(self.records),
            "by_model": by_model,
            "by_purpose": by_purpose,
            "estimated_usd": self.estimated_usd(),
        }

    def estimated_usd(self) -> float:
        total = 0.0
        for r in self.records:
            input_price, output_price = MODEL_PRICING.get(r.model, (0.0, 0.0))
            total += (r.prompt_tokens / 1_000_000) * input_price
            total += (r.completion_tokens / 1_000_000) * output_price
        return round(total, 4)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_qa_samples(qa_path: Path | None = None) -> list[dict]:
    path = qa_path or QA_PATH
    raw = load_json(path)
    if isinstance(raw, list):
        return raw
    return raw.get("samples", raw.get("data", []))


def save_checkpoint(data: dict | list, checkpoint_dir: Path, step_name: str, index: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{step_name}_ckpt_{index}.json"
    save_json(data, path)
    return path


def load_latest_checkpoint(checkpoint_dir: Path, step_name: str) -> tuple[dict | list | None, int]:
    if not checkpoint_dir.exists():
        return None, 0

    ckpts = list(checkpoint_dir.glob(f"{step_name}_ckpt_*.json"))
    if not ckpts:
        return None, 0

    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    idx = int(latest.stem.split("_")[-1])
    data = load_json(latest)
    return data, idx


def make_run_id(step_name: str) -> str:
    return f"{step_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def setup_logging(step_name: str, output_dir: Path | None = None) -> None:
    log_dir = output_dir or (BASE_OUTPUT_DIR / step_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "step.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_step_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_exp_step", None) == step_name for h in root.handlers
    )
    if not has_step_handler:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        fh._exp_step = step_name  # type: ignore[attr-defined]
        root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(ch)


class Timer:
    """컨텍스트 매니저 타이머."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = round(time.monotonic() - self._start, 4)
