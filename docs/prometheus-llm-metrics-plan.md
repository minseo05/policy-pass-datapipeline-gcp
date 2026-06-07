# Prometheus LLM Metrics 구현 계획

> FastAPI 앱에 커스텀 Prometheus 메트릭을 추가하여 LLM 사용량(토큰, 비용, 레이턴시)을 모니터링한다.

## 배경

- FastAPI 앱은 LiteLLM SDK로 LLM 호출 중 (Proxy 서버 아님)
- AWS Prometheus(`3.35.247.34:9090`)가 이미 `3.35.151.233:8080`을 `aws-api` job으로 스크래핑 시도 중
- 현재 `/metrics` 엔드포인트가 없어 404 반환 → `prometheus_client` 추가 필요
- 기존 GCP Cloud Monitoring(`src/api/monitoring.py`)은 그대로 유지 (병행 운영)

## 아키텍처

```
FastAPI App (EC2:8080)          Prometheus (EC2:9090)         Grafana (EC2:3000)
  /metrics ──────────────────►  scrape every 15s ──────────►  LLM Usage Dashboard
  ├─ http_requests_total                                      ├─ Request Rate by Model
  ├─ http_request_duration_seconds                            ├─ Latency Percentiles
  ├─ llm_requests_total                                       ├─ Token Usage
  ├─ llm_tokens_total                                         ├─ Cost Tracking
  ├─ llm_cost_usd_total                                       └─ Error Rate
  ├─ llm_request_duration_seconds
  └─ rag_retrieval_duration_seconds
```

## 메트릭 정의

### HTTP 메트릭 (기존 API Overview 대시보드 호환)

| 메트릭 | 타입 | 라벨 | 용도 |
|--------|------|------|------|
| `http_requests_total` | Counter | method, handler, status | HTTP 요청 수 |
| `http_request_duration_seconds` | Histogram | method, handler | HTTP 레이턴시 |
| `http_requests_in_progress` | Gauge | method, handler | 현재 처리 중 요청 |

### LLM 메트릭

| 메트릭 | 타입 | 라벨 | 용도 |
|--------|------|------|------|
| `llm_requests_total` | Counter | model, strategy, status | LLM 호출 수 (success/error) |
| `llm_request_duration_seconds` | Histogram | model, strategy | LLM 생성 레이턴시 |
| `llm_tokens_total` | Counter | model, strategy, token_type | 토큰 사용량 (prompt/completion/total) |
| `llm_cost_usd_total` | Counter | model, strategy | 추정 비용 (USD) |
| `rag_retrieval_duration_seconds` | Histogram | strategy | RAG 검색 레이턴시 |

### 라벨 값 예시

- `model`: `openai/gpt-4o-mini`, `vertex_ai/gemini-2.5-flash`, `vertex_ai/claude-sonnet-4-5`
- `strategy`: `hybrid_rerank`, `hybrid_rrf`, `semantic`, `keyword`, `no_rag`
- `token_type`: `prompt`, `completion`, `total`
- `status`: `success`, `error`

## 변경 파일

### 1. `pyproject.toml` — 의존성 추가

```toml
api = [
    ...
    "prometheus-client>=0.20",    # 추가
]
```

### 2. `src/api/prometheus_metrics.py` — 새 파일

모든 Prometheus 메트릭을 모듈 레벨 싱글턴으로 정의. `prometheus_client`의 기본 레지스트리에 등록된다.

```python
from prometheus_client import Counter, Gauge, Histogram

# HTTP 메트릭
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "handler", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds",
    ["method", "handler"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress", "HTTP requests in progress",
    ["method", "handler"],
)

# LLM 메트릭
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total", "Total LLM requests",
    ["model", "strategy", "status"],
)
LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds", "LLM generation latency in seconds",
    ["model", "strategy"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total", "Total LLM tokens consumed",
    ["model", "strategy", "token_type"],
)
LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total", "Cumulative LLM cost in USD",
    ["model", "strategy"],
)
RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds", "RAG retrieval latency in seconds",
    ["strategy"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
```

### 3. `src/api/middleware.py` — HTTP 메트릭 기록

기존 `dispatch()`에 Prometheus 기록 추가:

```python
from src.api.prometheus_metrics import (
    HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_IN_PROGRESS,
)

async def dispatch(self, request, call_next):
    ...
    handler = request.url.path
    HTTP_REQUESTS_IN_PROGRESS.labels(method=request.method, handler=handler).inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        elapsed_ms = ...
        HTTP_REQUESTS_IN_PROGRESS.labels(method=request.method, handler=handler).dec()
        HTTP_REQUESTS_TOTAL.labels(method=request.method, handler=handler, status="500").inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, handler=handler).observe(elapsed_ms / 1000)
        raise

    elapsed_ms = ...
    HTTP_REQUESTS_IN_PROGRESS.labels(method=request.method, handler=handler).dec()
    HTTP_REQUESTS_TOTAL.labels(method=request.method, handler=handler, status=str(status_code)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, handler=handler).observe(elapsed_ms / 1000)
    ...
```

### 4. `src/api/main.py` — `/metrics` 엔드포인트

```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import src.api.prometheus_metrics as _prom  # noqa: F401 — 싱글턴 초기화

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### 5. `src/api/routes/generate.py` — LLM 메트릭 기록

**성공 경로** (line ~101, `record_generation()` 호출 뒤):

```python
from src.api.prometheus_metrics import (
    LLM_REQUESTS_TOTAL, LLM_REQUEST_DURATION_SECONDS,
    LLM_TOKENS_TOTAL, LLM_COST_USD_TOTAL, RAG_RETRIEVAL_DURATION_SECONDS,
)

# 성공
LLM_REQUESTS_TOTAL.labels(model=resp.model, strategy=resp.search_strategy, status="success").inc()
LLM_REQUEST_DURATION_SECONDS.labels(model=resp.model, strategy=resp.search_strategy).observe(resp.generation_latency)
LLM_TOKENS_TOTAL.labels(model=resp.model, strategy=resp.search_strategy, token_type="prompt").inc(usage.prompt_tokens)
LLM_TOKENS_TOTAL.labels(model=resp.model, strategy=resp.search_strategy, token_type="completion").inc(usage.completion_tokens)
LLM_TOKENS_TOTAL.labels(model=resp.model, strategy=resp.search_strategy, token_type="total").inc(usage.total_tokens)
LLM_COST_USD_TOTAL.labels(model=resp.model, strategy=resp.search_strategy).inc(estimated_cost)
if resp.retrieval_latency > 0:
    RAG_RETRIEVAL_DURATION_SECONDS.labels(strategy=resp.search_strategy).observe(resp.retrieval_latency)
```

**에러 경로** (`except LLMError`):

```python
except LLMError as exc:
    model_id = resolve_model_key(body.model)
    strategy_label = "no_rag" if body.no_rag else body.strategy.value
    LLM_REQUESTS_TOTAL.labels(model=model_id or "unknown", strategy=strategy_label, status="error").inc()
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
```

### 6. Grafana 대시보드 (인프라 레포)

`policy-pass-infra-aws/monitoring/grafana/dashboards/llm-usage.json`

| Row | 패널 | PromQL |
|-----|------|--------|
| Request Overview | LLM Request Rate by Model | `sum(rate(llm_requests_total{job="aws-api"}[5m])) by (model)` |
| | Success vs Error Rate | `sum(rate(llm_requests_total{job="aws-api"}[5m])) by (status)` |
| Latency | Generation Latency (p50/p95/p99) | `histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket{job="aws-api"}[5m])) by (le, model))` |
| | RAG Retrieval Latency | `histogram_quantile(0.95, sum(rate(rag_retrieval_duration_seconds_bucket{job="aws-api"}[5m])) by (le, strategy))` |
| Token Usage | Token Rate (prompt/completion) | `sum(rate(llm_tokens_total{job="aws-api",token_type="prompt"}[5m])) by (model)` |
| | Cumulative Tokens | `sum(increase(llm_tokens_total{job="aws-api",token_type="total"}[$__range])) by (model)` |
| Cost | Cost Rate ($/hour) | `sum(rate(llm_cost_usd_total{job="aws-api"}[5m])) by (model) * 3600` |
| | Total Cost | `sum(increase(llm_cost_usd_total{job="aws-api"}[$__range])) by (model)` |

## 변경 불필요

- `monitoring/prometheus.yml` — 이미 `3.35.151.233:8080` 스크래핑 중
- `monitoring/docker-compose.yml` — Grafana provisioner가 새 대시보드 자동 감지
- `src/api/monitoring.py` — 기존 GCP Cloud Monitoring 유지

## 배포 순서

1. GCP 레포에서 코드 변경 (Steps 1-5)
2. Docker 이미지 빌드 & ECR push
3. EC2 API 인스턴스에서 새 이미지 pull & 재시작
4. `curl http://3.35.151.233:8080/metrics` 로 확인
5. 인프라 레포에서 Grafana 대시보드 추가 (Step 6)
6. EC2 Monitor에 대시보드 SCP & Grafana 재시작

## 검증

1. `/metrics` 엔드포인트 응답 확인
2. `/api/v1/generate` 호출 후 메트릭 카운터 증가 확인
3. Prometheus UI에서 `llm_requests_total` 쿼리 확인
4. Grafana "Policy Pass LLM Usage" 대시보드 데이터 표시 확인
5. 기존 "Policy Pass API Overview" 대시보드 HTTP 메트릭도 데이터 표시 확인
