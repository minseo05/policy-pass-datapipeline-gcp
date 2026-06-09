# Compact RAG Evaluation 사용 가이드

이 구성은 기존 3단계 평가인 RAGAS, LLM-as-Judge, DeepEval을 유지하면서 서비스 운영에 필요한 최소 지표만 추가한다.

## 1. 추가되는 파일

```text
config/evaluation_compact.json
src/evaluation/compact_metrics.py
src/evaluation/ragas_service_metrics.py
src/evaluation/compact_evaluator.py
scripts/run_compact_qa_eval.py
scripts/run_service_log_eval.py
data/eval/qa_compact_sample.jsonl
data/eval/service_log_sample.jsonl
tests/test_compact_metrics.py
```

## 2. QA 오프라인 평가

정답 또는 기준 답변이 있는 QA 세트를 평가한다.

```bash
python scripts/run_compact_qa_eval.py \
  --qa-path data/eval/qa_pairs.json \
  --output-dir data/results/compact_qa \
  --limit 20
```

QA 파일에 `answer`, `contexts`가 이미 있으면 그대로 평가한다. 질문만 있고 답변이 없으면 아래처럼 생성까지 같이 실행한다.

```bash
python scripts/run_compact_qa_eval.py \
  --qa-path data/eval/qa_pairs.json \
  --generate-missing \
  --index-dir data/index \
  --model gpt-4o-mini \
  --strategy hybrid_rerank \
  --limit 20
```

## 3. 실서비스 로그 평가

운영 로그에는 보통 ground truth가 없으므로 reference-free 지표만 사용한다.

```bash
python scripts/run_service_log_eval.py \
  --log-path data/eval/service_log_sample.jsonl \
  --output-dir data/results/service_eval
```

비용을 줄이고 싶으면 RAGAS/Judge/DeepEval을 끄고 deterministic 지표만 돌린다.

```bash
python scripts/run_service_log_eval.py \
  --log-path data/eval/service_log_sample.jsonl \
  --skip-llm-metrics
```

## 4. 권장 입력 스키마

### QA

```json
{
  "id": "qa_001",
  "question": "질문",
  "ground_truth": "기준 답변",
  "expected_claims": ["필수 주장 1", "필수 주장 2"],
  "answer": "생성 답변",
  "contexts": ["검색 chunk 1", "검색 chunk 2"],
  "sources": [{"id": "1", "content": "검색 chunk 1", "title": "문서명"}],
  "expected_abstain": false
}
```

### Service Log

```json
{
  "request_id": "req_001",
  "question": "질문",
  "answer": "생성 답변",
  "contexts": ["검색 chunk"],
  "sources": [{"id": "1", "content": "검색 chunk", "title": "문서명"}],
  "citations": [{"claim": "답변 문장", "source_ids": ["1"]}],
  "should_abstain": false,
  "retrieval_latency": 0.32,
  "generation_latency": 1.21,
  "tokens": {"prompt": 800, "completion": 120, "total": 920}
}
```

## 5. 결과 해석

- `ragas`: 기본 RAG 품질. QA에서는 context precision/recall까지, service에서는 faithfulness/relevancy 중심.
- `judge`: 사용자 관점의 인용 정확성, 완결성, 가독성.
- `safety`: DeepEval hallucination score. 낮을수록 좋다.
- `extra.claim_f1`: 답변이 기준 claim을 얼마나 포함했는지.
- `extra.citation_support_rate`: 인용 또는 검색 context가 답변 claim을 지지하는지.
- `extra.abstention_accuracy`: 답이 없을 때 제대로 거절했는지.
- `ops`: latency와 token 사용량.
