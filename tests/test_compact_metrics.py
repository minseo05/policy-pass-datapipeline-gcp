from src.evaluation.compact_metrics import (
    abstention_accuracy,
    citation_support_rate,
    claim_level_f1,
    is_abstention_answer,
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
        sources=[{"id": "A", "content": "신청 자격은 연령, 소득, 거주 요건을 기준으로 확인한다."}],
        citations=[{"claim": "신청 전에는 연령, 소득, 거주 요건을 확인해야 합니다.", "source_ids": ["A"]}],
    )
    assert score.support_rate == 1.0


def test_abstention_detection():
    assert is_abstention_answer("제공된 정보만으로는 확인할 수 없습니다.")
    score = abstention_accuracy("제공된 정보만으로는 확인할 수 없습니다.", expected_abstain=True)
    assert score.accuracy == 1.0
