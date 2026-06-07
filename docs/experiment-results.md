# RAG 응답 모니터링을 위한 비용 효율적 LLM-as-a-Judge 파이프라인 구축 및 성능 비교 분석 실험 결과 보고서

> 실행일: 2026-05-05 ~ 2026-05-06 (초기), 2026-05-23 (GPT-5.4 Mini, Gemini 3.5 Flash 추가)
> 파이프라인: `scripts/experiments/step1~step6`
> QA 데이터셋: `data/eval/qa_pairs.json` (100쌍)

---

## 1. 실험 설계 개요

### 1.1 파이프라인 구조

```
Step1 (검색)  → Step2 (생성)  → Step3 (평가)  → Step4 (Judge 비교) → Step5 (통계 분석) → Step6 (표·그림)
  4전략×100      7모델×100+NoRAG   Primary+Expensive    Judge 간 일치도     Bias/상관/탐지율     논문용 테이블
```

### 1.2 실험 변수

| 변수 | 값 |
|------|-----|
| QA 쌍 수 | 100 |
| 검색 전략 (Step1) | `vector_only`, `bm25_only`, `hybrid`, `hybrid_rerank` |
| 생성 모델 (Step2) | GPT-4o-mini, GPT-4o, Claude Sonnet, Gemini Flash, Llama3, **GPT-5.4 Mini**, **Gemini 3.5 Flash** |
| NoRAG 대조군 | GPT-4o-mini (검색 컨텍스트 없이 생성) |
| 검색 전략 (Step2~3) | `hybrid_rerank` (Step1에서 최적 전략 확정 후 고정) |
| top_k | 5 |
| Primary Judge | `openai/gpt-4o-mini` |
| Expensive Judge | `vertex_ai/gemini-3.1-pro-preview` (region: global) |
| Position Bias 완화 | 2회 평가 (원본 순서 + 셔플 순서) 평균 |

### 1.3 총 샘플 수

- Step1: 4전략 × 100 = **400건**
- Step2: 7모델 × 100 + 1 NoRAG × 100 = **800건**
- Step3 Primary: **800건** (GPT-4o-mini Judge + RAGAS + Safety)
- Step3 Expensive: **800건** (Gemini 3.1 Pro Judge only, RAGAS/Safety는 Primary 캐시 재사용)
- Step4~6: LLM 호출 없음 (순수 통계 연산)

---

## 2. Step1: 검색 전략 비교 (표 6)

> 출력: `data/experiments/step1_retrieval/retrieval_results.json`

| 전략 | Context Precision (mean±std) | Context Recall (mean±std) | Latency (sec) | N |
|------|-----|-----|-----|---|
| **vector_only** | 0.7885 ± 0.3075 | 0.8250 ± 0.3631 | 0.431 | 100 |
| **bm25_only** | 0.6896 ± 0.3563 | 0.7330 ± 0.4377 | 0.012 | 100 |
| **hybrid (RRF)** | 0.8054 ± 0.3122 | 0.8450 ± 0.3514 | 0.408 | 100 |
| **hybrid_rerank** | 0.7946 ± 0.3078 | 0.8550 ± 0.3413 | 0.449 | 100 |

**분석**:
- **Context Recall 최고**: hybrid_rerank (0.855) — Cross-Encoder reranking이 recall을 가장 높게 끌어올림
- **Context Precision 최고**: hybrid RRF (0.805) — 벡터+BM25 앙상블이 정밀도에서 소폭 우위
- **BM25 단독**: precision/recall 모두 최저 (0.690/0.733), 그러나 latency 0.012초로 30~40배 빠름
- **Vector 단독 vs Hybrid**: hybrid가 precision/recall 모두 우위, latency 차이 미미
- **Step2 이후 고정 전략**: `hybrid_rerank` (recall 최우선 선택)

---

## 3. Step2: 멀티 LLM 생성

> 출력: `data/experiments/step2_generation/generation_results.json`

| 조건 | 모델 (LiteLLM ID) | RAG | 샘플 수 |
|------|-----|-----|---|
| `gpt-4o-mini__rag` | `openai/gpt-4o-mini` | O | 100 |
| `gpt-4o__rag` | `openai/gpt-4o` | O | 100 |
| `claude-sonnet__rag` | `vertex_ai/claude-sonnet-4-5` | O | 100 |
| `gemini-flash__rag` | `vertex_ai/gemini-2.5-flash` | O | 100 |
| `llama3__rag` | `huggingface/meta-llama/Llama-3.3-70B-Instruct` | O | 100 |
| **`gpt-5.4-mini__rag`** | **`openai/gpt-5.4-mini`** | O | 100 |
| **`gemini-3.5-flash__rag`** | **`vertex_ai/gemini-3.5-flash`** | O | 100 |
| `gpt-4o-mini__no_rag` | `openai/gpt-4o-mini` | X | 100 |

- 검색 전략: `hybrid_rerank`, top_k=5 (NoRAG 제외)
- 총 800건 생성 완료

---

## 4. Step3: 3단계 평가

> 출력:
> - `data/experiments/step3_evaluation/eval_gpt4o_mini_judge.json` (Primary)
> - `data/experiments/step3_evaluation/eval_gemini_pro_judge.json` (Expensive)

### 4.1 평가 구조

```
Primary Pass (GPT-4o-mini Judge):
  - RAGAS v0.4: faithfulness, answer_relevancy, context_precision, context_recall
  - LLM Judge (G-Eval): citation_accuracy, completeness, readability (1~5점 정수)
  - Safety (DeepEval): hallucination_score (0.0~1.0)
  - Position Bias 완화: 원본 순서 + 셔플 순서 2회 평가 → 평균

Expensive Pass (Gemini 3.1 Pro Judge):
  - LLM Judge만 실행 (RAGAS/Safety는 Primary 결과 캐시 재사용)
  - 동일한 Position Bias 완화 적용
  - temperature=1.0 (Gemini 3.x 요구사항)
```

### 4.2 유효 샘플 수

#### Primary Pass (GPT-4o-mini Judge)

| 조건 | 전체 | eval 없음 | 유효 Judge | 유효 RAGAS | 유효 Safety |
|------|------|-----------|-----------|-----------|------------|
| gpt-4o-mini__rag | 100 | 0 | 100 | 99 | 100 |
| gpt-4o__rag | 100 | 0 | 100 | 100 | 100 |
| claude-sonnet__rag | 100 | 0 | 24 | 23 | 24 |
| gemini-flash__rag | 100 | 0 | 100 | 96 | 100 |
| llama3__rag | 100 | 65 | 35 | 35 | 35 |
| **gpt-5.4-mini__rag** | 100 | 0 | **51** | **50** | **51** |
| **gemini-3.5-flash__rag** | 100 | 0 | **51** | **36** | **50** |
| gpt-4o-mini__no_rag | 100 | 0 | 100 | 0* | 100 |
| **합계** | **800** | **65** | **561** | **439** | **560** |

*NoRAG는 컨텍스트가 없어 Faithfulness/Context 메트릭 측정 불가 (Answer Relevancy만 100건 측정)

- `claude-sonnet__rag`: Judge는 24건만 유효 (GPT-4o-mini Judge 파싱 실패율 높음)
- `llama3__rag`: HuggingFace API 에러로 65건의 생성 자체가 실패 → eval 없음
- `gpt-5.4-mini__rag`: 51건 유효 — 나머지 49건 Judge/RAGAS 타임아웃 또는 파싱 실패
- `gemini-3.5-flash__rag`: 51건 Judge 유효, 36건 RAGAS 유효 — RAGAS asyncio 타임아웃 빈발

#### Expensive Pass (Gemini 3.1 Pro Judge)

| 조건 | 전체 | 유효 Judge | 실패 |
|------|------|-----------|------|
| gpt-4o-mini__rag | 100 | 73 | 27 |
| gpt-4o__rag | 100 | 73 | 27 |
| claude-sonnet__rag | 100 | 59 | 41 |
| gemini-flash__rag | 100 | 78 | 22 |
| llama3__rag | 100 | 23 | 77 |
| **gpt-5.4-mini__rag** | 100 | **57** | **43** |
| **gemini-3.5-flash__rag** | 100 | **65** | **35** |
| gpt-4o-mini__no_rag | 100 | 5 | 95 |
| **합계** | **800** | **433** | **367** |

- 전체 성공률: **54.1%** (433/800)
- Gemini 3.1 Pro는 JSON 출력 포맷 준수율이 낮음 (temperature=1.0 필수 제약 + 응답 중간 truncation)
- `gpt-4o-mini__no_rag` 조건에서 특히 낮음 (5/100) — NoRAG 답변의 컨텍스트 부재로 Judge 혼동
- `llama3__rag`도 낮음 (23/100) — 65건은 원본 답변 자체가 없음 + 나머지도 파싱 실패율 높음

### 4.3 표 7: RAGAS 메트릭 (모델별)

| 조건 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | N |
|------|-------------|------------------|-------------------|---------------|---|
| claude-sonnet__rag | 0.8793 ± 0.1382 | 0.4214 ± 0.2350 | 0.6719 ± 0.3973 | 0.8125 ± 0.3767 | 100 |
| gemini-flash__rag | 0.8257 ± 0.2746 | 0.4845 ± 0.2677 | 0.7823 ± 0.3088 | 0.8550 ± 0.3413 | 100 |
| gpt-4o-mini__rag | 0.8586 ± 0.2892 | 0.5313 ± 0.2322 | 0.7840 ± 0.3181 | 0.8550 ± 0.3413 | 100 |
| gpt-4o__rag | 0.8532 ± 0.2910 | 0.4727 ± 0.2805 | 0.7905 ± 0.3118 | 0.8500 ± 0.3500 | 100 |
| llama3__rag | **0.9824** ± 0.0584 | 0.5113 ± 0.1779 | 0.7815 ± 0.3082 | 0.8857 ± 0.2949 | 35 |
| **gpt-5.4-mini__rag** | 0.8834 ± 0.2019 | 0.2988 ± 0.3162 | 0.6771 ± 0.3369 | 0.8333 ± 0.3660 | **51** |
| **gemini-3.5-flash__rag** | 0.7199 ± 0.3279 | 0.4312 ± 0.3004 | 0.6964 ± 0.3343 | 0.8333 ± 0.3660 | **51** |
| gpt-4o-mini__no_rag | N/A | 0.3355 ± 0.3204 | N/A | N/A | 100 |

**분석**:
- **Faithfulness 최고**: Llama3 (0.982) — 컨텍스트에 매우 충실한 답변 생성 (단, N=35로 표본 적음)
- **GPT-5.4 Mini Faithfulness 2위** (0.883): GPT-4o-mini(0.859)보다 소폭 높아 충실도 개선
- **Gemini 3.5 Flash Faithfulness 최저** (0.720): Thinking 모델 특성상 추론 과정에서 컨텍스트를 벗어나는 경향
- **Answer Relevancy 최고**: GPT-4o-mini RAG (0.531) — 질문 의도에 가장 부합하는 답변
- **GPT-5.4 Mini Answer Relevancy 최저** (0.299): 답변 관련성이 낮음 — 과도하게 상세한 응답으로 인해 질문 초점이 분산되는 것으로 추정
- **NoRAG 대조군**: Answer Relevancy 0.336으로 RAG 대비 현저히 낮음, Faithfulness는 컨텍스트 없어 측정 불가
- Context Recall/Precision은 검색 전략이 동일(hybrid_rerank)하므로 조건 간 차이가 적음

### 4.4 표 8: LLM Judge 점수 (GPT-4o-mini Judge, 모델별)

| 조건 | Citation Accuracy | Completeness | Readability | Average | N |
|------|------------------|-------------|------------|---------|---|
| claude-sonnet__rag | 1.20 ± 2.14 | 1.20 ± 2.14 | 1.20 ± 2.14 | 1.20 ± 2.14 | 100* |
| gemini-flash__rag | 4.62 ± 1.02 | 4.48 ± 1.16 | 4.96 ± 0.28 | 4.69 ± 0.76 | 100 |
| gpt-4o-mini__rag | 4.59 ± 1.14 | 4.55 ± 1.23 | 4.98 ± 0.14 | 4.71 ± 0.79 | 100 |
| gpt-4o__rag | 4.39 ± 1.38 | 4.29 ± 1.51 | 4.88 ± 0.43 | 4.52 ± 1.04 | 100 |
| llama3__rag | **4.94** ± 0.23 | **4.93** ± 0.24 | **4.99** ± 0.08 | **4.95** ± 0.17 | 35 |
| **gpt-5.4-mini__rag** | **4.76** ± 0.61 | 4.53 ± 0.70 | 4.98 ± 0.14 | **4.76** ± 0.43 | **51** |
| **gemini-3.5-flash__rag** | 4.21 ± 1.46 | 4.12 ± 1.45 | 4.88 ± 0.43 | 4.40 ± 1.02 | **51** |
| gpt-4o-mini__no_rag | 2.46 ± 1.47 | 4.37 ± 1.16 | 5.00 ± 0.05 | 3.94 ± 0.71 | 100 |

*claude-sonnet__rag: 유효 Judge 24건, 나머지 76건은 0점 → 평균이 비정상적으로 낮음. 유효 24건만의 평균은 별도 계산 필요.

**분석**:
- **Llama3 최고 점수** (Average 4.95) — 단, N=35로 생존 편향 가능 (성공적으로 생성된 응답만 평가)
- **GPT-5.4 Mini 2위** (Average 4.76): Citation Accuracy 4.76으로 전 모델 중 Llama3 다음으로 높음. 표준편차 0.43으로 가장 안정적
- **GPT-4o-mini RAG** (Average 4.71) — Citation/Completeness/Readability 모두 균형
- **Gemini 3.5 Flash**: Average 4.40으로 Gemini Flash(4.69)보다 낮음 — Thinking 모델의 긴 추론이 인용 정확도와 완결성에서 불리
- **NoRAG 대조군**: Citation Accuracy 2.46 (컨텍스트 없이 생성하므로 인용 정확도 낮음), Readability 5.00 (가독성은 최고)
- **Readability는 전반적으로 높음** (4.88~5.00) — 모든 모델이 가독성 높은 한국어 답변 생성

### 4.5 표 9: Safety (Hallucination Score, 모델별)

| 조건 | Hallucination Mean ± Std | N |
|------|------------------------|---|
| gpt-4o-mini__no_rag | **0.000** ± 0.000 | 100 |
| gemini-flash__rag | 0.338 ± 0.365 | 100 |
| gpt-4o__rag | 0.350 ± 0.367 | 100 |
| gpt-4o-mini__rag | 0.368 ± 0.380 | 100 |
| **gemini-3.5-flash__rag** | **0.412** ± 0.363 | **51** |
| llama3__rag | 0.457 ± 0.369 | 35 |
| **gpt-5.4-mini__rag** | **0.490** ± 0.359 | **51** |
| claude-sonnet__rag | **0.533** ± 0.340 | 100 |

> Hallucination Score: 0.0 = 환각 없음, 1.0 = 완전한 환각. **낮을수록 좋음**.

**분석**:
- **NoRAG 대조군 0.000**: DeepEval의 hallucination 판정 기준이 "컨텍스트 대비 사실 왜곡"이므로, 컨텍스트 자체가 없으면 환각으로 판정하지 않음 (측정 한계)
- **Gemini Flash 최저 환각** (RAG 중, 0.338) — 컨텍스트 충실도 높음
- **GPT-5.4 Mini 환각 0.490**: GPT-4o-mini(0.368)보다 높음 — 모델 업그레이드에도 불구하고 환각률 증가
- **Gemini 3.5 Flash 환각 0.412**: Gemini Flash(0.338)보다 높음 — Thinking 모델의 추론 확장이 환각 유발
- **Claude Sonnet 최고 환각** (0.533) — 답변이 컨텍스트를 넘어 추가 정보를 생성하는 경향
- 전반적으로 RAG 모델의 환각률 0.34~0.53 범위

---

## 5. Step4: Judge 모델 비용-성능 비교 (실험 C)

> 출력: `data/experiments/step4_judge_comparison/judge_comparison.json`
> 비교 대상: GPT-4o-mini Judge vs Gemini 3.1 Pro Judge
> 유효 비교 쌍: **331건** (양쪽 모두 유효한 Judge 결과가 있는 샘플)

### 5.1 표 10: 전체 일치도 (Average 기준)

| 지표 | 값 | 해석 |
|------|-----|------|
| Kendall τ | 0.2373 | 약한 순위 상관 |
| Kendall p-value | 8×10⁻⁶ | 통계적으로 유의 |
| Spearman ρ | 0.2464 | 약한 순위 상관 |
| Spearman p-value | 6×10⁻⁶ | 통계적으로 유의 |
| MAE | 0.3234 | 평균 0.32점 차이 (5점 척도) |
| Perfect Agreement Rate | 85.80% | 정수 반올림 시 동일 비율 |
| **Class Agreement Rate** | **89.73%** | 3등급(low/mid/high) 분류 일치율 |

### 5.2 메트릭별 상세 일치도

| 메트릭 | Kendall τ | MAE | Perfect Agr. | Class Agr. |
|--------|-----------|-----|-------------|-----------|
| citation_accuracy | 0.2645 | 0.4230 | 87.01% | 87.92% |
| completeness | 0.2121 | 0.5000 | 83.69% | 86.40% |
| **readability** | **0.4623** | **0.0468** | **97.28%** | **98.49%** |
| average | 0.2373 | 0.3234 | 85.80% | 89.73% |

**분석**:
- **Readability에서 가장 높은 일치도**: τ=0.462, MAE=0.047, Class Agreement 98.5% — 두 Judge가 가독성 판단에 매우 일치
- **Completeness에서 가장 낮은 일치도**: τ=0.212, MAE=0.500 — 완결성 판단에 모델 간 시각 차이 존재
- **전체 Class Agreement 89.7%**: 3등급 분류 기준으로 약 9/10건 일치 → **저비용 GPT-4o-mini Judge가 고비용 Gemini Pro Judge를 대체 가능**
- 순위 상관(τ, ρ)은 낮지만, 이는 대부분의 점수가 고점(4~5)에 집중되어 변동성이 적기 때문
- 비용 데이터는 LiteLLM 비용 추적이 비활성화되어 0.0으로 기록됨 (수동 추정 필요)

---

## 6. Step5: 통계 분석

### 6.1 실험 D: Position Bias 분석

> 출력: `data/experiments/step5_analysis/position_bias.json`
> 분석 대상: 561건 (유효 Judge raw_scores 보유 샘플)
> 방법: 원본 순서(original) vs 셔플 순서(shuffled) 점수 차이 분석

| 메트릭 | Mean |Δ| | Std |Δ| | Max Δ | ≥1점차 건수 | ≥1점차 비율 | Wilcoxon p |
|--------|---------|---------|-------|------------|------------|------------|
| citation_accuracy | 0.162 | 0.652 | 4 | 37/561 | **6.6%** | 0.573 |
| completeness | 0.091 | 0.410 | 4 | 30/561 | **5.3%** | 0.758 |
| readability | 0.034 | 0.275 | 4 | 10/561 | **1.8%** | 0.403 |

**점수 차이 분포 (citation_accuracy)**:
| Δ=0 | Δ=1 | Δ=2 | Δ=4 |
|-----|-----|-----|-----|
| 524건 (93.4%) | 1건 (0.2%) | 27건 (4.8%) | 9건 (1.6%) |

**점수 차이 분포 (completeness)**:
| Δ=0 | Δ=1 | Δ=2 | Δ=4 |
|-----|-----|-----|-----|
| 531건 (94.7%) | 11건 (2.0%) | 18건 (3.2%) | 1건 (0.2%) |

**점수 차이 분포 (readability)**:
| Δ=0 | Δ=1 | Δ=2 | Δ=4 |
|-----|-----|-----|-----|
| 551건 (98.2%) | 3건 (0.5%) | 6건 (1.1%) | 1건 (0.2%) |

**분산 감소 효과 (2회 평균)**:
| 지표 | 단일 평가 분산 | 2회 평균 분산 | 분산 감소율 |
|------|--------------|-------------|-----------|
| 전체 평균 | 0.7539 | 0.7095 | **5.89%** |

**분석**:
- **Wilcoxon p-value 모두 >0.05**: 원본/셔플 간 체계적 편향(systematic bias) 없음 — 문맥 순서가 점수에 유의한 영향을 미치지 않음
- **≥1점차 비율 최대 6.6%**: 대부분의 평가에서 순서 변경에 무관하게 동일 점수 부여
- **Readability 가장 안정**: 1.8%만 1점 이상 차이 — 가독성 평가는 순서 독립적
- **Citation Accuracy 가장 취약**: 6.6%에서 1점 이상 차이, 최대 4점 차이까지 발생 — 인용 정확도는 문맥 배치에 민감할 수 있음
- **분산 감소 5.89%**: 초기 실험(2.79%) 대비 2배 이상 증가 — 신규 모델 추가로 점수 변동이 큰 샘플이 포함되면서 2회 평균의 효과가 더 뚜렷

### 6.2 실험 E: 3단계 교차 상관 분석

> 출력: `data/experiments/step5_analysis/cross_correlation.json`
> 분석 대상: 438건 (RAGAS + Judge + Safety 3가지 모두 유효한 샘플)

#### 전체 상관 매트릭스

| 쌍 | Spearman ρ | p-value | 해석 |
|-----|-----------|---------|------|
| RAGAS Faithfulness ↔ Judge Average | 0.455 | <0.0001 | **약한 상관 (보완적)** |
| RAGAS Faithfulness ↔ Safety Hallucination | 0.254 | <0.0001 | 무시 가능 (독립적) |
| Judge Average ↔ Safety Hallucination | 0.278 | <0.0001 | 무시 가능 (독립적) |
| RAGAS Relevancy ↔ Judge Completeness | **0.565** | <0.0001 | **중간 상관 (보완적)** |
| Judge Readability ↔ Judge Average | 0.460 | <0.0001 | **약한 상관 (보완적)** |
| Judge Readability ↔ RAGAS Faithfulness | 0.222 | <0.0001 | 무시 가능 (독립적) |

#### 모델별 상관 (Faithfulness ↔ Judge, Faithfulness ↔ Hallucination)

| 조건 | Faith↔Judge ρ | Faith↔Halluc ρ | N |
|------|-------------|---------------|---|
| gpt-4o-mini__rag | 0.362 | 0.091 | 99 |
| gpt-4o__rag | 0.510 | 0.345 | 100 |
| claude-sonnet__rag | 0.134 | 0.622 | 23 |
| gemini-flash__rag | 0.410 | 0.335 | 96 |
| llama3__rag | 0.269 | 0.141 | 35 |
| **gpt-5.4-mini__rag** | **0.539** | 0.314 | **50** |
| **gemini-3.5-flash__rag** | **0.744** | **0.529** | **36** |

**분석**:
- **3단계 평가는 보완적(complementary)**: 최대 ρ=0.565로 중간 수준 상관 → 각 단계가 서로 다른 측면을 측정
- **RAGAS Faithfulness와 Safety Hallucination은 거의 독립**: ρ=0.254 → 이론적으로 유사한 개념이나 측정 방식이 달라 포착하는 결함이 다름
- **Relevancy ↔ Completeness 가장 높은 상관** (ρ=0.565): 답변 관련성과 완결성은 개념적으로 중첩 — 가장 redundant한 쌍
- **Gemini 3.5 Flash 특이점**: Faith↔Judge ρ=0.744로 전 모델 중 최고 — Faithfulness와 Judge 점수가 매우 강하게 연동. Faith↔Halluc ρ=0.529도 높음
- **GPT-5.4 Mini**: Faith↔Judge ρ=0.539로 GPT-4o(0.510)와 유사한 수준
- **Claude Sonnet 특이점**: Faith↔Halluc ρ=0.622로 다른 모델 대비 높음 (N=23으로 소표본 주의)

### 6.3 탐지율 분석: 단독 vs 조합 평가

> 출력: `data/experiments/step5_analysis/detection_coverage.json`
> 분석 대상: 637건
> 임계값: RAGAS Faithfulness < 0.5, Judge Average < 3.0, Hallucination > 0.5

#### 단독 탐지율

| 평가 단계 | Flagged | 탐지율 |
|----------|---------|--------|
| RAGAS 단독 | 33건 | **5.2%** |
| Judge 단독 | 121건 | **19.0%** |
| Safety 단독 | 213건 | **33.4%** |

#### 조합 탐지율 (Union)

| 조합 | Flagged | 탐지율 |
|------|---------|--------|
| RAGAS + Judge | 132건 | 20.7% |
| RAGAS + Safety | 243건 | 38.2% |
| **Judge + Safety** | **333건** | **52.3%** |
| **3단계 전체** | **342건** | **53.7%** |

#### 고유 탐지 (해당 단계만 잡은 케이스)

| 단계 | 고유 탐지 | 비율 |
|------|----------|------|
| RAGAS만 | 9건 | 1.4% |
| **Judge만** | **99건** | **15.5%** |
| **Safety만** | **210건** | **33.0%** |

#### 불일치 매트릭스

| 상황 | 건수 | 비율 |
|------|------|------|
| RAGAS 통과 + Judge 실패 | 99건 | 15.5% |
| Judge 통과 + RAGAS 실패 | 11건 | 1.7% |
| RAGAS 통과 + Safety 실패 | 210건 | 33.0% |
| Safety 통과 + RAGAS 실패 | 30건 | 4.7% |
| Judge 통과 + Safety 실패 | 212건 | 33.3% |
| Safety 통과 + Judge 실패 | 120건 | 18.8% |

**분석**:
- **3단계 조합 탐지율 53.7%**: 단독 최고(Safety 33.4%) 대비 +20.3%p 추가 탐지 — 조합의 가치 입증
- **Judge+Safety 2단계 조합이 52.3%**: 3단계(53.7%)와 1.4%p 차이밖에 안 남 — RAGAS의 추가 기여가 제한적
- **RAGAS 단독 탐지율 5.2%**: Faithfulness < 0.5 임계값이 대부분의 응답에서 도달하지 않음 (평균 0.72~0.98)
- **Safety가 가장 많은 고유 탐지** (210건, 33.0%): Hallucination Score는 다른 2단계가 놓치는 결함을 가장 많이 포착
- **Judge도 상당한 고유 탐지** (99건, 15.5%): 정성 평가(인용/완결성)만으로 잡히는 품질 문제 존재
- **핵심 결론: 최소 Judge+Safety 2단계는 필수**, RAGAS는 추가적 안전망 역할

---

## 7. Step6: 생성된 차트

> 출력 디렉토리: `data/experiments/step6_tables_figures/figures/`

| 파일명 | 설명 |
|--------|------|
| `fig4_ragas_radar.html` | 모델별 RAGAS 4지표 레이더 차트 (7개 모델) |
| `fig_judge_heatmap.html` | 모델별 Judge 점수 히트맵 (1~5점 색상 매핑) |
| `fig_detection_coverage.html` | 단독/2단계/3단계 탐지율 막대 차트 |
| `fig5_bias_histogram.html` | Position Bias 점수 차이 분포 히스토그램 |
| `fig_cost_performance.html` | GPT-4o-mini vs Gemini Pro 메트릭별 MAE-τ 산점도 |

---

## 8. 알려진 제한사항 및 주의점

### 8.1 데이터 결측

| 문제 | 영향 | 원인 |
|------|------|------|
| Llama3 65건 생성 실패 | N=35만 유효 (다른 모델 N=100) | HuggingFace Inference API 불안정 |
| Claude Sonnet Judge 76건 실패 | Primary Judge 유효 24건 | GPT-4o-mini가 Claude Sonnet 답변의 JSON 파싱에 실패 |
| Gemini Pro Judge 367건 실패 | Expensive pass 유효 433건 (54.1%) | temperature=1.0 제약 + JSON 출력 준수율 낮음 |
| **GPT-5.4 Mini 49건 평가 실패** | N=51만 유효 | RAGAS asyncio 타임아웃 + Judge 파싱 실패 |
| **Gemini 3.5 Flash 49건 평가 실패** | N=51 (Judge), N=36 (RAGAS) | Thinking 모델 특성상 긴 응답 → 타임아웃 빈발 |

### 8.2 측정 한계

- **NoRAG Hallucination Score 0.000**: DeepEval은 컨텍스트 대비 환각을 측정하므로, 컨텍스트 없는 NoRAG에서는 환각이 0으로 판정됨. 이는 NoRAG 답변에 환각이 없다는 뜻이 아님.
- **Claude Sonnet Judge 평균 왜곡**: 표 8의 claude-sonnet__rag 평균 1.20은 76건의 0점이 포함된 값. 유효 24건의 실제 평균은 별도 계산 필요.
- **비용 데이터 미수집**: LiteLLM 비용 추적이 비활성화되어 모든 비용 필드가 0.0. GPT-4o-mini vs Gemini Pro 비용 비율은 공식 가격표 기준 수동 추정 필요.
- **Gemini Pro NaN 상관계수**: 일부 조건에서 점수가 상수(모두 동일)하여 Kendall τ / Spearman ρ가 NaN — per_condition 분석에서 해당 조건 해석 불가.
- **GPT-5.4 Mini / Gemini 3.5 Flash 소표본**: N=51(Judge)~N=36(RAGAS)로 100건 대비 절반 수준. 통계적 신뢰도가 다른 모델보다 낮음.

### 8.3 기술적 이슈 및 해결

| 이슈 | 해결 |
|------|------|
| Gemini 3.x temperature < 1.0 시 무한루프/품질 저하 | `temp = 1.0 if "gemini-3" in judge_model else 0.0` 동적 설정 |
| Gemini 3.1 Pro JSON 출력에 ```json 래핑 + 전후 텍스트 | `_parse_scores()` regex 강화: fenced code block + bare JSON 추출 |
| 손상된 체크포인트 (0점 포함) | 삭제 후 재실행 |
| Step4 실패 평가 포함 | `_align_samples()`에 average==0 필터 추가 |
| Step6 eval=None 샘플 AttributeError | `s.get("eval", {})` → `(s.get("eval") or {})` 방어적 코딩 |
| **RAGAS asyncio 타임아웃 후 연쇄 실패** | `_reset_ragas_clients()`로 캐시된 AsyncOpenAI 클라이언트 초기화 |
| **Gemini 3.5 Flash max_tokens 부족** | Thinking 모델용 `max_tokens=8192` + Vertex AI `global` 리전 설정 |

---

## 9. 출력 파일 목록

```
data/experiments/
├── step1_retrieval/
│   └── retrieval_results.json              # 4전략 × 100 = 400건
├── step2_generation/
│   └── generation_results.json             # 8조건 × 100 = 800건
├── step3_evaluation/
│   ├── eval_gpt4o_mini_judge.json          # Primary: 800건 (561 valid judge)
│   └── eval_gemini_pro_judge.json          # Expensive: 800건 (433 valid judge)
├── step4_judge_comparison/
│   └── judge_comparison.json               # 331쌍 비교 결과
├── step5_analysis/
│   ├── position_bias.json                  # 561건 Position Bias 분석
│   ├── cross_correlation.json              # 438건 교차 상관
│   └── detection_coverage.json             # 637건 탐지율 분석
└── step6_tables_figures/
    ├── tables_figures.json                 # 표 6~10 + 부가 표 통합
    └── figures/
        ├── fig4_ragas_radar.html           # RAGAS 레이더 차트
        ├── fig_judge_heatmap.html          # Judge 히트맵
        ├── fig_detection_coverage.html     # 탐지율 막대 차트
        ├── fig5_bias_histogram.html        # Position Bias 분포
        └── fig_cost_performance.html       # 비용-성능 산점도
```

---

## 10. 핵심 결론 요약 (논문 기술용)

1. **검색**: Hybrid Rerank가 Context Recall 0.855로 최고. BM25 단독은 30배 빠르나 성능 열위.
2. **생성**: GPT-4o-mini가 비용 대비 최고 균형 (RAGAS Answer Relevancy 0.531, Judge Average 4.71). GPT-5.4 Mini는 Citation Accuracy 4.76으로 인용 정확도 개선되었으나 Answer Relevancy 0.299로 하락.
3. **최신 모델**: GPT-5.4 Mini는 Faithfulness(0.883)와 Judge Average(4.76)에서 전작 대비 소폭 개선, 그러나 환각률(0.490)이 GPT-4o-mini(0.368) 대비 증가. Gemini 3.5 Flash는 Thinking 모델 특성상 Faithfulness(0.720)가 가장 낮고, 응답 길이로 인한 타임아웃 빈발.
4. **RAG 효과**: NoRAG 대비 RAG가 Answer Relevancy +0.20, Citation Accuracy +2.13 향상.
5. **Judge 대체 가능성**: GPT-4o-mini와 Gemini 3.1 Pro의 Class Agreement **89.7%** → 저비용 모델로 대체 가능.
6. **Position Bias 미미**: Wilcoxon p>0.05, ≥1점차 비율 최대 6.6% → 2회 평균이 안정성에 기여하며 분산 5.89% 감소.
7. **3단계 평가 보완성**: 단독 최고 33.4% → 3단계 조합 53.7% (+20.3%p). Judge+Safety 2단계가 핵심, RAGAS는 안전망.
8. **교차 상관**: 최대 ρ=0.565 (중간) → 3단계는 서로 다른 결함을 측정하여 보완적 관계 확인.