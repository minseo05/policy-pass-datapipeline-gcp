"""Lightweight custom RAG metrics for QA and service-log evaluation.

This module intentionally stays deterministic and dependency-light.
It complements RAGAS, the existing LLM Judge, and DeepEval with only three
service-friendly checks:

1. claim_level_f1: compares answer claims against expected claims or ground truth.
2. citation_support_rate: checks whether answer claims are supported by cited/retrieved context.
3. abstention_accuracy: checks whether the system refuses when it should not answer.

These are not meant to replace semantic/NLI evaluation. They are cheap guardrail
signals that can run on every service log or on a larger sample than LLM metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+|(?<=다\.)\s*|(?<=요\.)\s*")
_ABSTENTION_PATTERNS = [
    "문서에서 확인",
    "근거가 부족",
    "답변하기 어렵",
    "알 수 없",
    "확인할 수 없",
    "제공된 정보만으로",
    "정확히 안내하기 어렵",
    "not enough information",
    "cannot determine",
    "i don't know",
    "insufficient context",
]


@dataclass(frozen=True)
class ClaimScore:
    precision: float | None
    recall: float | None
    f1: float | None
    matched_claims: int
    answer_claims: int
    expected_claims: int


@dataclass(frozen=True)
class CitationScore:
    support_rate: float | None
    supported_claims: int
    checked_claims: int
    mode: str


@dataclass(frozen=True)
class AbstentionScore:
    accuracy: float | None
    expected_abstain: bool | None
    did_abstain: bool


def tokenize(text: str) -> set[str]:
    """Tokenize Korean/English text with a small stop-word filter."""
    tokens = {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}
    stop = {
        "그리고",
        "또는",
        "대한",
        "관련",
        "있습니다",
        "합니다",
        "입니다",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "is",
        "are",
    }
    return {t for t in tokens if len(t) >= 2 and t not in stop}


def token_overlap_score(left: str, right: str) -> float:
    """Return token containment-style overlap from left to right."""
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def extract_simple_claims(text: str, min_chars: int = 12) -> list[str]:
    """Split an answer/reference into simple claim-like sentences.

    It is deliberately conservative: bullets and sentence endings become claims,
    very short fragments are removed.
    """
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text.replace("•", "\n").replace("- ", "\n- ")).strip()
    parts = [p.strip(" -\t") for p in _SENTENCE_SPLIT_RE.split(cleaned)]
    claims: list[str] = []
    for part in parts:
        if len(part) >= min_chars and part not in claims:
            claims.append(part)
    return claims


def claim_level_f1(
    answer: str,
    expected_claims: list[str] | None = None,
    ground_truth: str | None = None,
    min_overlap: float = 0.55,
) -> ClaimScore:
    """Compute a lightweight claim-level precision/recall/F1.

    expected_claims is preferred. If missing, ground_truth is split into claims.
    Answer claims are split from the answer text.
    """
    answer_claims = extract_simple_claims(answer)
    expected = expected_claims or extract_simple_claims(ground_truth or "")
    if not answer_claims or not expected:
        return ClaimScore(None, None, None, 0, len(answer_claims), len(expected))

    matched_answer_indices: set[int] = set()
    matched_expected_indices: set[int] = set()

    for a_idx, answer_claim in enumerate(answer_claims):
        best_idx = None
        best_score = 0.0
        for e_idx, expected_claim in enumerate(expected):
            if e_idx in matched_expected_indices:
                continue
            # symmetric-ish match: expected tokens contained in answer, and answer not completely unrelated
            expected_in_answer = token_overlap_score(expected_claim, answer_claim)
            answer_in_expected = token_overlap_score(answer_claim, expected_claim)
            score = max(expected_in_answer, answer_in_expected)
            if score > best_score:
                best_score = score
                best_idx = e_idx
        if best_idx is not None and best_score >= min_overlap:
            matched_answer_indices.add(a_idx)
            matched_expected_indices.add(best_idx)

    precision = len(matched_answer_indices) / len(answer_claims)
    recall = len(matched_expected_indices) / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return ClaimScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched_claims=len(matched_answer_indices),
        answer_claims=len(answer_claims),
        expected_claims=len(expected),
    )


def _source_id(source: dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "source_id", "chunk_id", "doc_id", "policy_id", "rank"):
        value = source.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _source_text(source: dict[str, Any]) -> str:
    return str(source.get("content") or source.get("text") or source.get("body") or source.get("page_content") or "")


def citation_support_rate(
    answer: str,
    contexts: list[str] | None = None,
    sources: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    min_overlap: float = 0.45,
) -> CitationScore:
    """Estimate whether answer claims are supported by cited/retrieved context.

    Preferred input: citations with claim/text and cited source IDs.
    Fallback input: answer sentences checked against all retrieved contexts.
    """
    source_lookup: dict[str, str] = {}
    if sources:
        for idx, source in enumerate(sources, start=1):
            sid = _source_id(source, idx)
            source_lookup[sid] = _source_text(source)
            source_lookup[str(idx)] = _source_text(source)
    if contexts:
        for idx, context in enumerate(contexts, start=1):
            source_lookup.setdefault(str(idx), context)

    if citations:
        checked = 0
        supported = 0
        for citation in citations:
            claim = str(citation.get("claim") or citation.get("text") or citation.get("sentence") or "")
            raw_ids = (
                citation.get("source_ids")
                or citation.get("citation_ids")
                or citation.get("cited_source_ids")
                or citation.get("source_id")
                or citation.get("citation_id")
            )
            ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
            ids = [str(i) for i in ids if i is not None]
            cited_text = "\n".join(source_lookup.get(i, "") for i in ids).strip()
            if not claim or not cited_text:
                continue
            checked += 1
            if token_overlap_score(claim, cited_text) >= min_overlap:
                supported += 1
        if checked == 0:
            return CitationScore(None, 0, 0, "explicit_citation")
        return CitationScore(round(supported / checked, 4), supported, checked, "explicit_citation")

    claims = extract_simple_claims(answer)
    context_text = "\n".join(source_lookup.values())
    if not claims or not context_text:
        return CitationScore(None, 0, len(claims), "retrieved_context_fallback")
    supported = sum(1 for claim in claims if token_overlap_score(claim, context_text) >= min_overlap)
    return CitationScore(round(supported / len(claims), 4), supported, len(claims), "retrieved_context_fallback")


def is_abstention_answer(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(pattern in lowered for pattern in _ABSTENTION_PATTERNS)


def abstention_accuracy(answer: str, expected_abstain: bool | None) -> AbstentionScore:
    """Check whether the answer abstained exactly when expected.

    expected_abstain should come from QA labels or service no-answer detection.
    If it is missing, accuracy is None but did_abstain is still returned.
    """
    did_abstain = is_abstention_answer(answer)
    if expected_abstain is None:
        return AbstentionScore(None, None, did_abstain)
    return AbstentionScore(1.0 if did_abstain == expected_abstain else 0.0, expected_abstain, did_abstain)


def build_operational_metrics(sample: dict[str, Any]) -> dict[str, float | int | None]:
    """Extract service-operational metrics from one sample/log."""
    retrieval_latency = sample.get("retrieval_latency")
    generation_latency = sample.get("generation_latency")
    explicit_total = sample.get("latency") or sample.get("total_latency") or sample.get("elapsed")

    if explicit_total is not None:
        total_latency = float(explicit_total)
    else:
        total_latency = float(retrieval_latency or 0.0) + float(generation_latency or 0.0)

    tokens = sample.get("tokens") or sample.get("token_usage") or {}
    prompt_tokens = int(tokens.get("prompt") or tokens.get("prompt_tokens") or 0)
    completion_tokens = int(tokens.get("completion") or tokens.get("completion_tokens") or 0)
    total_tokens = int(tokens.get("total") or tokens.get("total_tokens") or prompt_tokens + completion_tokens)

    return {
        "retrieval_latency_seconds": float(retrieval_latency) if retrieval_latency is not None else None,
        "generation_latency_seconds": float(generation_latency) if generation_latency is not None else None,
        "total_latency_seconds": round(total_latency, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def compute_extra_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    """Compute all lightweight custom metrics for one QA or service sample."""
    answer = str(sample.get("answer") or sample.get("actual_output") or "")
    contexts = sample.get("contexts") or [
        s.get("content", "") for s in sample.get("sources", []) if isinstance(s, dict)
    ]
    sources = sample.get("sources") or []
    expected_abstain = sample.get("expected_abstain")
    if expected_abstain is None:
        expected_abstain = sample.get("should_abstain")
    if expected_abstain is None:
        expected_abstain = sample.get("no_answer_expected")

    claim_score = claim_level_f1(
        answer=answer,
        expected_claims=sample.get("expected_claims"),
        ground_truth=sample.get("ground_truth") or sample.get("reference") or sample.get("expected_output"),
    )
    citation_score = citation_support_rate(
        answer=answer,
        contexts=contexts,
        sources=sources,
        citations=sample.get("citations"),
    )
    abstention_score = abstention_accuracy(answer=answer, expected_abstain=expected_abstain)

    return {
        "claim_precision": claim_score.precision,
        "claim_recall": claim_score.recall,
        "claim_f1": claim_score.f1,
        "claim_matched": claim_score.matched_claims,
        "claim_answer_count": claim_score.answer_claims,
        "claim_expected_count": claim_score.expected_claims,
        "citation_support_rate": citation_score.support_rate,
        "citation_supported_claims": citation_score.supported_claims,
        "citation_checked_claims": citation_score.checked_claims,
        "citation_mode": citation_score.mode,
        "abstention_accuracy": abstention_score.accuracy,
        "expected_abstain": abstention_score.expected_abstain,
        "did_abstain": abstention_score.did_abstain,
    }


def flatten_metrics(result: dict[str, Any]) -> dict[str, float | int | None]:
    """Flatten nested result blocks into dot-key metrics for threshold checks."""
    flat: dict[str, float | int | None] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (int, float)) or value is None:
            flat[prefix] = value

    walk("", result)
    return flat


def quality_gate(result: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Return pass/fail details for configured thresholds.

    Keys ending with _max mean lower-is-better. For example:
    safety.hallucination_score_max <= 0.35
    """
    flat = flatten_metrics(result)
    checks: dict[str, dict[str, Any]] = {}
    passed = 0
    total = 0
    for raw_key, threshold in thresholds.items():
        lower_is_better = raw_key.endswith("_max")
        metric_key = raw_key[:-4] if lower_is_better else raw_key
        value = flat.get(metric_key)
        if value is None:
            checks[raw_key] = {"value": None, "threshold": threshold, "passed": None}
            continue
        total += 1
        ok = value <= threshold if lower_is_better else value >= threshold
        passed += 1 if ok else 0
        checks[raw_key] = {"value": value, "threshold": threshold, "passed": ok}
    pass_rate = None if total == 0 else round(passed / total, 4)
    return {"passed": passed == total if total else None, "pass_rate": pass_rate, "checks": checks}
