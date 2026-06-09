"""3단계 평가 파이프라인 테스트 — RAGAS, LLM Judge, DeepEval 모두 mock 처리."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation import EvalResult, JudgeResult, RagasResult, SafetyResult

# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def sample_question() -> str:
    return "청년 월세 한시 특별지원 신청 자격은?"


@pytest.fixture()
def sample_contexts() -> list[str]:
    return [
        "청년 월세 한시 특별지원은 만 19~34세 무주택 청년을 대상으로 한다.",
        "청년가구 소득이 기준 중위소득 60% 이하이고 원가구 소득이 100% 이하인 경우 신청 가능하다.",
    ]


@pytest.fixture()
def sample_answer() -> str:
    return (
        "만 19~34세 독립거주 무주택 청년으로, "
        "청년가구 소득이 기준 중위소득 60% 이하이고 원가구 소득이 100% 이하인 경우 신청 가능합니다."
    )


@pytest.fixture()
def sample_ground_truth() -> str:
    return (
        "만 19~34세 독립거주 무주택 청년으로, "
        "청년가구 소득이 기준 중위소득 60% 이하이고 원가구 소득이 기준 중위소득 100% 이하인 경우 신청 가능하다."
    )


# ── dataclass 테스트 ──────────────────────────────────────


class TestDataclasses:
    def test_ragas_result_defaults(self):
        r = RagasResult()
        assert r.faithfulness is None
        assert r.answer_relevancy is None
        assert r.context_precision is None
        assert r.context_recall is None

    def test_ragas_result_with_values(self):
        r = RagasResult(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7, context_recall=0.6)
        assert r.faithfulness == 0.9
        assert r.context_recall == 0.6

    def test_ragas_result_frozen(self):
        r = RagasResult()
        with pytest.raises(AttributeError):
            r.faithfulness = 0.5  # type: ignore[misc]

    def test_judge_result_defaults(self):
        j = JudgeResult()
        assert j.citation_accuracy == 0.0
        assert j.average == 0.0

    def test_judge_result_with_values(self):
        j = JudgeResult(citation_accuracy=4.0, completeness=3.5, readability=4.5, average=4.0)
        assert j.average == 4.0

    def test_safety_result_defaults(self):
        s = SafetyResult()
        assert s.hallucination_score is None

    def test_eval_result_composition(self):
        e = EvalResult(
            ragas=RagasResult(faithfulness=0.9),
            judge=JudgeResult(average=4.0),
            safety=SafetyResult(hallucination_score=0.1),
            latency=2.5,
        )
        assert e.ragas.faithfulness == 0.9
        assert e.judge.average == 4.0
        assert e.safety.hallucination_score == 0.1
        assert e.latency == 2.5


# ── RAGAS 메트릭 테스트 ──────────────────────────────────


class TestRagasMetrics:
    def test_evaluate_ragas_success(self, sample_question, sample_contexts, sample_answer, sample_ground_truth):
        mock_result = MagicMock()
        mock_result.value = 0.9

        mock_metric = MagicMock()
        mock_metric.ascore = AsyncMock(return_value=mock_result)

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {
                    "ragas": MagicMock(),
                    "ragas.llms": MagicMock(llm_factory=lambda *args, **kw: MagicMock()),
                    "ragas.embeddings": MagicMock(OpenAIEmbeddings=lambda *args, **kw: MagicMock()),
                    "ragas.metrics": MagicMock(),
                    "ragas.metrics.collections": MagicMock(
                        Faithfulness=lambda **kw: mock_metric,
                        AnswerRelevancy=lambda **kw: mock_metric,
                        ContextPrecision=lambda **kw: mock_metric,
                        ContextRecall=lambda **kw: mock_metric,
                    ),
                },
            ),
        ):
            import importlib

            import src.evaluation.ragas_metrics as rm

            importlib.reload(rm)
            rm._ragas_llm = MagicMock()
            rm._ragas_embeddings = MagicMock()

            result = rm.evaluate_ragas(sample_question, sample_contexts, sample_answer, sample_ground_truth)

        assert isinstance(result, RagasResult)
        assert result.faithfulness == 0.9
        assert result.answer_relevancy == 0.9
        assert result.context_precision == 0.9
        assert result.context_recall == 0.9

    def test_evaluate_ragas_partial_failure(self, sample_question, sample_contexts, sample_answer, sample_ground_truth):
        """일부 메트릭 실패 시 None으로 반환."""
        mock_result_ok = MagicMock()
        mock_result_ok.value = 0.9

        mock_metric_ok = MagicMock()
        mock_metric_ok.ascore = AsyncMock(return_value=mock_result_ok)

        mock_metric_fail = MagicMock()
        mock_metric_fail.ascore = AsyncMock(side_effect=Exception("metric error"))

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {
                    "ragas": MagicMock(),
                    "ragas.llms": MagicMock(llm_factory=lambda *args, **kw: MagicMock()),
                    "ragas.embeddings": MagicMock(OpenAIEmbeddings=lambda *args, **kw: MagicMock()),
                    "ragas.metrics": MagicMock(),
                    "ragas.metrics.collections": MagicMock(
                        Faithfulness=lambda **kw: mock_metric_ok,
                        AnswerRelevancy=lambda **kw: mock_metric_fail,
                        ContextPrecision=lambda **kw: mock_metric_ok,
                        ContextRecall=lambda **kw: mock_metric_fail,
                    ),
                },
            ),
        ):
            import importlib

            import src.evaluation.ragas_metrics as rm

            importlib.reload(rm)
            rm._ragas_llm = MagicMock()
            rm._ragas_embeddings = MagicMock()

            result = rm.evaluate_ragas(sample_question, sample_contexts, sample_answer, sample_ground_truth)

        assert result.faithfulness == 0.9
        assert result.answer_relevancy is None
        assert result.context_precision == 0.9
        assert result.context_recall is None


# ── LLM Judge 테스트 ─────────────────────────────────────


class TestLLMJudge:
    def test_parse_scores_valid(self):
        from src.evaluation.llm_judge import _parse_scores

        raw = '{"citation_accuracy": 4, "completeness": 5, "readability": 3}'
        result = _parse_scores(raw)
        assert result == {"citation_accuracy": 4, "completeness": 5, "readability": 3}

    def test_parse_scores_with_code_block(self):
        from src.evaluation.llm_judge import _parse_scores

        raw = '```json\n{"citation_accuracy": 4, "completeness": 5, "readability": 3}\n```'
        result = _parse_scores(raw)
        assert result == {"citation_accuracy": 4, "completeness": 5, "readability": 3}

    def test_parse_scores_out_of_range(self):
        from src.evaluation.llm_judge import _parse_scores

        raw = '{"citation_accuracy": 6, "completeness": 5, "readability": 3}'
        assert _parse_scores(raw) is None

    def test_parse_scores_missing_field(self):
        from src.evaluation.llm_judge import _parse_scores

        raw = '{"citation_accuracy": 4, "completeness": 5}'
        assert _parse_scores(raw) is None

    def test_parse_scores_invalid_json(self):
        from src.evaluation.llm_judge import _parse_scores

        assert _parse_scores("not json") is None

    @patch("src.generation.llm_client.generate")
    def test_judge_response_success(self, mock_generate, sample_question, sample_contexts, sample_answer):
        mock_resp = MagicMock()
        mock_resp.content = '{"citation_accuracy": 4, "completeness": 5, "readability": 4}'
        mock_generate.return_value = mock_resp

        from src.evaluation.llm_judge import judge_response

        result = judge_response(sample_question, sample_contexts, sample_answer)

        assert isinstance(result, JudgeResult)
        assert result.citation_accuracy == 4.0
        assert result.completeness == 5.0
        assert result.readability == 4.0
        assert result.average == pytest.approx(4.33, abs=0.01)
        assert mock_generate.call_count == 2

    @patch("src.generation.llm_client.generate")
    def test_judge_response_all_fail(self, mock_generate, sample_question, sample_contexts, sample_answer):
        mock_generate.side_effect = RuntimeError("LLM 호출 실패")

        from src.evaluation.llm_judge import judge_response

        result = judge_response(sample_question, sample_contexts, sample_answer)

        assert result.citation_accuracy == 0.0
        assert result.average == 0.0

    @patch("src.generation.llm_client.generate")
    def test_judge_response_one_parse_fail(self, mock_generate, sample_question, sample_contexts, sample_answer):
        """첫 번째는 파싱 실패, 두 번째만 성공 → 1회 결과로 반환."""
        resp_ok = MagicMock()
        resp_ok.content = '{"citation_accuracy": 3, "completeness": 4, "readability": 5}'
        resp_bad = MagicMock()
        resp_bad.content = "invalid"
        mock_generate.side_effect = [resp_bad, resp_ok]

        from src.evaluation.llm_judge import judge_response

        result = judge_response(sample_question, sample_contexts, sample_answer)

        assert result.citation_accuracy == 3.0
        assert result.completeness == 4.0
        assert result.readability == 5.0


# ── DeepEval 안전성 테스트 ────────────────────────────────


class TestSafetyMetrics:
    def test_evaluate_safety_success(self, sample_question, sample_contexts, sample_answer):
        mock_metric = MagicMock()
        mock_metric.score = 0.15
        mock_metric.reason = "low hallucination"

        mock_test_case_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "deepeval": MagicMock(),
                "deepeval.metrics": MagicMock(HallucinationMetric=lambda **kw: mock_metric),
                "deepeval.test_case": MagicMock(LLMTestCase=mock_test_case_cls),
            },
        ):
            import importlib

            import src.evaluation.safety_metrics as sm

            importlib.reload(sm)

            result = sm.evaluate_safety(sample_question, sample_contexts, sample_answer)

        assert isinstance(result, SafetyResult)
        assert result.hallucination_score == 0.15

    def test_evaluate_safety_failure(self, sample_question, sample_contexts, sample_answer):
        with patch.dict(
            "sys.modules",
            {
                "deepeval": MagicMock(),
                "deepeval.metrics": MagicMock(HallucinationMetric=MagicMock(side_effect=Exception("deepeval error"))),
                "deepeval.test_case": MagicMock(),
            },
        ):
            import importlib

            import src.evaluation.safety_metrics as sm

            importlib.reload(sm)

            result = sm.evaluate_safety(sample_question, sample_contexts, sample_answer)

        assert result.hallucination_score is None


# ── 통합 Evaluator 테스트 ────────────────────────────────


class TestRAGEvaluator:
    @patch("src.evaluation.evaluator.RAGEvaluator._run_safety")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_judge")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_ragas")
    def test_evaluate_single(
        self, mock_ragas, mock_judge, mock_safety, sample_question, sample_contexts, sample_answer, sample_ground_truth
    ):
        mock_ragas.return_value = RagasResult(faithfulness=0.9, answer_relevancy=0.8)
        mock_judge.return_value = JudgeResult(citation_accuracy=4.0, completeness=4.0, readability=4.0, average=4.0)
        mock_safety.return_value = SafetyResult(hallucination_score=0.1)

        from src.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        result = evaluator.evaluate_single(sample_question, sample_contexts, sample_answer, sample_ground_truth)

        assert isinstance(result, EvalResult)
        assert result.ragas.faithfulness == 0.9
        assert result.judge.average == 4.0
        assert result.safety.hallucination_score == 0.1
        assert result.latency >= 0

    @patch("src.evaluation.evaluator.RAGEvaluator._run_safety")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_judge")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_ragas")
    def test_evaluate_batch(self, mock_ragas, mock_judge, mock_safety):
        mock_ragas.return_value = RagasResult(faithfulness=0.9)
        mock_judge.return_value = JudgeResult(average=4.0)
        mock_safety.return_value = SafetyResult(hallucination_score=0.1)

        samples = [
            {
                "id": "q001",
                "question": "질문1",
                "answer": "답변1",
                "ground_truth": "정답1",
                "contexts": ["컨텍스트1"],
            },
            {
                "id": "q002",
                "question": "질문2",
                "answer": "답변2",
                "ground_truth": "정답2",
                "contexts": ["컨텍스트2"],
                "error": True,
            },
        ]

        from src.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        results = evaluator.evaluate_batch(samples)

        assert len(results) == 2
        assert results[0]["eval_result"] is not None
        assert results[0]["eval_result"]["ragas"]["faithfulness"] == 0.9
        assert results[1]["eval_result"] is None
        assert results[1]["eval_error"] == "응답 생성 실패"

    @patch("src.evaluation.evaluator.RAGEvaluator._run_safety")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_judge")
    @patch("src.evaluation.evaluator.RAGEvaluator._run_ragas")
    def test_evaluate_batch_checkpoint(self, mock_ragas, mock_judge, mock_safety, tmp_path):
        mock_ragas.return_value = RagasResult(faithfulness=0.9)
        mock_judge.return_value = JudgeResult(average=4.0)
        mock_safety.return_value = SafetyResult(hallucination_score=0.1)

        samples = [
            {
                "id": f"q{i:03d}",
                "question": f"질문{i}",
                "answer": f"답변{i}",
                "ground_truth": f"정답{i}",
                "contexts": [f"ctx{i}"],
            }
            for i in range(10)
        ]

        from src.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        evaluator.evaluate_batch(samples, checkpoint_dir=tmp_path)

        checkpoint = tmp_path / "checkpoint_10.json"
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert len(data) == 10


# ── 리포트 생성 테스트 ────────────────────────────────────


class TestReport:
    def test_generate_report(self, tmp_path):
        results = {
            "gpt-4o-mini": [
                {
                    "id": "q001",
                    "question": "질문",
                    "eval_result": {
                        "ragas": {
                            "faithfulness": 0.9,
                            "answer_relevancy": 0.8,
                            "context_precision": 0.7,
                            "context_recall": 0.6,
                        },
                        "judge": {
                            "citation_accuracy": 4.0,
                            "completeness": 5.0,
                            "readability": 4.0,
                            "average": 4.33,
                        },
                        "safety": {"hallucination_score": 0.1},
                        "latency": 2.5,
                    },
                },
            ],
        }

        from src.evaluation.report import generate_report

        output_path = generate_report(results, tmp_path, run_id="test_run")

        assert output_path.exists()
        assert output_path.with_suffix(".html").exists()
        report = json.loads(output_path.read_text())
        assert report["run_id"] == "test_run"
        assert "gpt-4o-mini" in report["summary"]
        assert report["summary"]["gpt-4o-mini"]["ragas_avg"]["faithfulness"] == 0.9

    def test_generate_report_with_errors(self, tmp_path):
        results = {
            "gemini-flash": [
                {"id": "q001", "eval_result": None, "eval_error": "평가 실패"},
                {
                    "id": "q002",
                    "eval_result": {
                        "ragas": {
                            "faithfulness": 0.8,
                            "answer_relevancy": None,
                            "context_precision": None,
                            "context_recall": None,
                        },
                        "judge": None,
                        "safety": None,
                        "latency": 1.0,
                    },
                },
            ],
        }

        from src.evaluation.report import generate_report

        output_path = generate_report(results, tmp_path)
        report = json.loads(output_path.read_text())
        summary = report["summary"]["gemini-flash"]
        assert summary["errors"] == 1
        assert summary["ragas_avg"]["faithfulness"] == 0.8
        assert summary["ragas_avg"]["answer_relevancy"] is None

    def test_safe_mean_empty(self):
        from src.evaluation.report import _safe_mean

        assert _safe_mean([]) is None

    def test_safe_mean_values(self):
        from src.evaluation.report import _safe_mean

        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0
