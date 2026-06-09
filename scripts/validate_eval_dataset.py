"""Validate QA evaluation dataset.

This script checks data/eval/qa_pairs.json before running RAG evaluation.

It intentionally treats expected_claims and expected_abstain as warnings,
because existing QA samples may not have those fields yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REFERENCE_FIELDS = ("ground_truth", "reference", "expected_output")
REQUIRED_FIELDS = ("id", "question")

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_QA_TYPES = {
    "factual",
    "comparison",
    "reasoning",
    "no_answer",
    "out_of_scope",
    "abstention",
}

ABSTAIN_HINTS = (
    "답변할 수",
    "확인할 수",
    "알 수 없",
    "제공된 문맥",
    "제공된 정보",
    "근거가 부족",
    "확정할 수",
    "문맥만으로",
    "정보만으로",
)


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSON or JSONL dataset.

    Supported formats:
    - JSON list[dict]
    - JSON {"samples": list[dict]}
    - JSONL one dict per line
    """

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if path.suffix.lower() == ".jsonl":
        samples = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL {line_no}번째 줄 JSON 오류: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"JSONL {line_no}번째 줄은 object여야 합니다.")

            samples.append(row)

        return samples, {"format": "jsonl"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 문법 오류: {exc}") from exc

    if isinstance(data, list):
        return data, {"format": "json_list"}

    if isinstance(data, dict):
        samples = data.get("samples")
        if not isinstance(samples, list):
            raise ValueError("JSON object 형식이면 'samples' 배열이 필요합니다.")
        return samples, data

    raise ValueError("지원하지 않는 데이터셋 형식입니다. list 또는 {'samples': [...]} 형식이어야 합니다.")


def get_reference_text(sample: dict[str, Any]) -> str:
    for field in REFERENCE_FIELDS:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def looks_like_abstain_text(text: str) -> bool:
    normalized = text.replace(" ", "")
    return any(hint.replace(" ", "") in normalized for hint in ABSTAIN_HINTS)


def validate_sample(
    sample: dict[str, Any],
    index: int,
    seen_ids: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    label = sample.get("id") or f"row_{index + 1}"

    if not isinstance(sample, dict):
        return [f"row_{index + 1}: sample은 object여야 합니다."], []

    for field in REQUIRED_FIELDS:
        value = sample.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: 필수 필드 누락 또는 빈 값 - {field}")

    sample_id = sample.get("id")
    if isinstance(sample_id, str) and sample_id.strip():
        if sample_id in seen_ids:
            errors.append(f"{label}: 중복 id입니다.")
        seen_ids.add(sample_id)

    reference_text = get_reference_text(sample)
    if not reference_text:
        errors.append(f"{label}: ground_truth/reference/expected_output 중 하나는 필요합니다.")

    question = sample.get("question")
    if isinstance(question, str) and len(question.strip()) < 3:
        warnings.append(f"{label}: question이 너무 짧습니다.")

    difficulty = sample.get("difficulty")
    if difficulty is not None and difficulty not in ALLOWED_DIFFICULTIES:
        warnings.append(
            f"{label}: difficulty 값이 권장 범위를 벗어났습니다. 현재={difficulty}, 권장={sorted(ALLOWED_DIFFICULTIES)}"
        )

    qa_type = sample.get("qa_type")
    if qa_type is not None and qa_type not in ALLOWED_QA_TYPES:
        warnings.append(
            f"{label}: qa_type 값이 권장 범위를 벗어났습니다. 현재={qa_type}, 권장={sorted(ALLOWED_QA_TYPES)}"
        )

    expected_claims = sample.get("expected_claims")
    if expected_claims is None:
        warnings.append(f"{label}: expected_claims가 없습니다. claim_f1 평가 보강 대상입니다.")
    elif not isinstance(expected_claims, list):
        errors.append(f"{label}: expected_claims는 list[str]이어야 합니다.")
    else:
        for claim_index, claim in enumerate(expected_claims, start=1):
            if not isinstance(claim, str) or not claim.strip():
                errors.append(f"{label}: expected_claims[{claim_index}]는 비어 있지 않은 문자열이어야 합니다.")

    expected_abstain = sample.get("expected_abstain")
    if expected_abstain is None:
        warnings.append(f"{label}: expected_abstain이 없습니다. abstention 평가 보강 대상입니다.")
    elif not isinstance(expected_abstain, bool):
        errors.append(f"{label}: expected_abstain은 boolean이어야 합니다.")
    elif expected_abstain is True:
        if reference_text and not looks_like_abstain_text(reference_text):
            warnings.append(f"{label}: expected_abstain=true인데 ground_truth가 답변 불가 문장처럼 보이지 않습니다.")

    citations = sample.get("citations")
    if citations is not None:
        if not isinstance(citations, list):
            errors.append(f"{label}: citations는 list여야 합니다.")
        else:
            for citation_index, citation in enumerate(citations, start=1):
                if not isinstance(citation, dict):
                    errors.append(f"{label}: citations[{citation_index}]는 object여야 합니다.")
                    continue

                if "claim" in citation and not isinstance(citation["claim"], str):
                    errors.append(f"{label}: citations[{citation_index}].claim은 문자열이어야 합니다.")

                if "source_ids" in citation and not isinstance(citation["source_ids"], list):
                    errors.append(f"{label}: citations[{citation_index}].source_ids는 list여야 합니다.")

    return errors, warnings


def summarize_warnings(warnings: list[str], max_items: int) -> None:
    print(f"\n경고: {len(warnings)}개")

    if not warnings:
        return

    for warning in warnings[:max_items]:
        print(f"- {warning}")

    remaining = len(warnings) - max_items
    if remaining > 0:
        print(f"- ... 외 {remaining}개 경고 생략")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RAG evaluation QA dataset")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/eval/qa_pairs.json",
        help="검증할 QA 데이터셋 경로",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="warning이 있어도 실패 처리합니다.",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=30,
        help="출력할 warning 최대 개수",
    )
    args = parser.parse_args()

    path = Path(args.path)

    try:
        samples, metadata = load_dataset(path)
    except Exception as exc:
        print(f"검증 실패: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    if not samples:
        errors.append("samples가 비어 있습니다.")

    total_count = metadata.get("total_count")
    if isinstance(total_count, int) and total_count != len(samples):
        warnings.append(f"metadata.total_count({total_count})와 실제 samples 개수({len(samples)})가 다릅니다.")

    for index, sample in enumerate(samples):
        sample_errors, sample_warnings = validate_sample(sample, index, seen_ids)
        errors.extend(sample_errors)
        warnings.extend(sample_warnings)

    if errors:
        print(f"검증 실패: error {len(errors)}개")
        for error in errors:
            print(f"- {error}")
        summarize_warnings(warnings, args.max_warnings)
        return 1

    summarize_warnings(warnings, args.max_warnings)

    if warnings and args.strict_warnings:
        print("\n검증 실패: --strict-warnings 옵션으로 warning도 실패 처리합니다.")
        return 1

    print("\n검증 통과")
    print(f"- 파일: {path}")
    print(f"- 샘플 수: {len(samples)}")
    print(f"- 고유 ID 수: {len(seen_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
