from src.evaluation.compact_metrics import (
    abstention_accuracy,
    build_operational_metrics,
    citation_support_rate,
    claim_level_f1,
    is_abstention_answer,
    quality_gate,
)


def test_claim_level_f1_matches_expected_claims():
    score = claim_level_f1(
        answer="청년 월세 지원은 주거비 부담이 큰 청년에게 월세 일부를 지원한다.",
        expected_claims=["청년 월세 지원은 주거비 부담이 큰 청년에게 월세 일부를 지원한다"],
    )

    assert score.f1 == 1.0


def test_citation_support_rate_with_explicit_citation():
    score = citation_support_rate(
        answer="신청 전에는 연령, 소득, 거주 요건을 확인해야 합니다.",
        sources=[
            {
                "id": "A",
                "content": "신청 자격은 연령, 소득, 거주 요건을 기준으로 확인한다.",
            }
        ],
        citations=[
            {
                "claim": "신청 전에는 연령, 소득, 거주 요건을 확인해야 합니다.",
                "source_ids": ["A"],
            }
        ],
    )

    assert score.support_rate == 1.0
    assert score.mode == "explicit_citation"


def test_citation_support_rate_with_retrieved_context_fallback():
    score = citation_support_rate(
        answer="청년 월세 지원은 소득 요건을 확인해야 합니다.",
        contexts=["청년 월세 지원 신청자는 소득 요건을 확인해야 한다."],
    )

    assert score.support_rate == 1.0
    assert score.mode == "retrieved_context_fallback"


def test_abstention_detection():
    assert is_abstention_answer("제공된 정보만으로는 확인할 수 없습니다.")

    score = abstention_accuracy(
        "제공된 정보만으로는 확인할 수 없습니다.",
        expected_abstain=True,
    )

    assert score.accuracy == 1.0
    assert score.did_abstain is True


def test_abstention_accuracy_fails_when_answer_refuses_unnecessarily():
    score = abstention_accuracy(
        "제공된 정보만으로는 확인할 수 없습니다.",
        expected_abstain=False,
    )

    assert score.accuracy == 0.0
    assert score.expected_abstain is False
    assert score.did_abstain is True


def test_quality_gate_passes_when_metrics_meet_thresholds():
    result = {
        "ragas": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
        },
        "extra": {
            "citation_support_rate": 0.8,
            "abstention_accuracy": 1.0,
        },
        "safety": {
            "hallucination_score": 0.2,
        },
        "ops": {
            "total_latency_seconds": 5.0,
            "total_tokens": 2500,
        },
    }

    thresholds = {
        "ragas.faithfulness": 0.85,
        "ragas.answer_relevancy": 0.80,
        "extra.citation_support_rate": 0.70,
        "extra.abstention_accuracy": 1.0,
        "safety.hallucination_score_max": 0.30,
        "ops.total_latency_seconds_max": 10.0,
        "ops.total_tokens_max": 4000,
    }

    gate = quality_gate(result, thresholds)

    assert gate["passed"] is True
    assert gate["pass_rate"] == 1.0


def test_quality_gate_fails_when_lower_is_better_metric_is_too_high():
    result = {
        "safety": {
            "hallucination_score": 0.8,
        },
        "ops": {
            "total_latency_seconds": 15.0,
        },
    }

    thresholds = {
        "safety.hallucination_score_max": 0.30,
        "ops.total_latency_seconds_max": 10.0,
    }

    gate = quality_gate(result, thresholds)

    assert gate["passed"] is False
    assert gate["checks"]["safety.hallucination_score_max"]["passed"] is False
    assert gate["checks"]["ops.total_latency_seconds_max"]["passed"] is False


def test_quality_gate_marks_missing_metric_as_not_checked():
    result = {
        "ragas": {
            "faithfulness": 0.9,
        }
    }

    thresholds = {
        "ragas.faithfulness": 0.85,
        "extra.citation_support_rate": 0.70,
    }

    gate = quality_gate(result, thresholds)

    assert gate["passed"] is True
    assert gate["checks"]["ragas.faithfulness"]["passed"] is True
    assert gate["checks"]["extra.citation_support_rate"]["passed"] is None


def test_build_operational_metrics_from_latency_and_token_usage():
    sample = {
        "retrieval_latency": 0.8,
        "generation_latency": 3.2,
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "total_tokens": 1300,
        },
    }

    metrics = build_operational_metrics(sample)

    assert metrics["retrieval_latency_seconds"] == 0.8
    assert metrics["generation_latency_seconds"] == 3.2
    assert metrics["total_latency_seconds"] == 4.0
    assert metrics["prompt_tokens"] == 1000
    assert metrics["completion_tokens"] == 300
    assert metrics["total_tokens"] == 1300


def test_build_operational_metrics_calculates_total_tokens_when_missing():
    sample = {
        "retrieval_latency": 1.0,
        "generation_latency": 2.5,
        "token_usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 500,
        },
    }

    metrics = build_operational_metrics(sample)

    assert metrics["total_latency_seconds"] == 3.5
    assert metrics["total_tokens"] == 1700
