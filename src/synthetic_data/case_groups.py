"""Synthetic QA에 사용할 사례 집합을 한 번의 규칙으로 확정한다.

이 모듈은 CLI나 파일 출력이 없는 공통 로직 계층이다.
build_decision_chunks.py가 prepare_case_groups() 결과를 canonical case 입력으로
직렬화하고, 최종 Synthetic builder는 그 결과를 그대로 소비한다.

content_text는 artifact LLM에 넘길 방송내용 중심의 deterministic source
view이고, raw_text는 rationale 생성과 원문 대조에 사용하는 전체 의결행이다.
현재 content_text 절단기는 최종 artifact 생성기가 아니라 LLM 입력과 중복
통합을 위한 전처리다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .build_synthetic_data import (
    ANSWERABILITY_EXCLUDE_RE,
    content_from_raw,
    dedup_fingerprint,
    neutral_artifact,
    outcome_from_raw,
    review_required_from_outcome,
    rule_ids_from_raw,
)

PAGE_MARKER_RE = re.compile(r"\s*\[\[PAGE_\d+\]\]\s*")
CONTENT_BULLET_RE = re.compile(r"(?m)^○\s+")


def raw_text_from_row(raw: str) -> str:
    """내부 page marker만 제거하고 공식 의결행 전체를 보존한다."""
    return PAGE_MARKER_RE.sub("\n", raw).strip()


def apply_raw_content_override(raw: str, override: dict[str, Any]) -> str:
    """표 열 추출 순서 때문에 앞 사건 내용이 붙은 행의 본문 시작점을 보정한다.

    PDF 원문과 페이지 provenance는 그대로 두고, qa_overrides.yaml에 사람이
    확인한 1-based `raw_content_start_occurrence`가 있을 때만 적용한다.
    의결번호·방송정보·관련조항·결과가 있는 첫 본문 앞 header는 보존한다.
    """
    occurrence = override.get("raw_content_start_occurrence")
    if occurrence is None:
        return raw
    if not isinstance(occurrence, int) or occurrence < 1:
        raise ValueError(
            "raw_content_start_occurrence는 1 이상의 정수여야 합니다."
        )
    markers = list(CONTENT_BULLET_RE.finditer(raw))
    if occurrence > len(markers):
        raise ValueError(
            "raw_content_start_occurrence가 추출된 본문 표식 수를 넘었습니다: "
            f"requested={occurrence}, found={len(markers)}"
        )
    header = raw[:markers[0].start()].rstrip()
    selected = raw[markers[occurrence - 1].start():].lstrip()
    return f"{header}\n{selected}".strip()


def prepare_case_groups(
    rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    manual: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """행을 검증·선별하고 deterministic fingerprint로 case group을 만든다.

    반환값은 (groups, excluded, stats)다. 어떤 파일도 쓰지 않는다.
    """
    prepared: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_decision_ids: set[tuple[str, str]] = set()
    fallback_total = 0

    for row in rows:
        source_decision_key = (row["source_id"], row["decision_id"])
        if source_decision_key in seen_decision_ids:
            excluded.append({
                "decision_id": row["decision_id"],
                "source_id": row["source_id"],
                "reason": "duplicate_decision_id_in_source",
            })
            continue
        seen_decision_ids.add(source_decision_key)

        override = manual.get(row["decision_id"], {})
        if override.get("exclude") is True:
            excluded.append({
                "decision_id": row["decision_id"],
                "source_id": row["source_id"],
                "reason": override.get("exclude_reason", "manual_exclusion"),
            })
            continue
        corrected_raw = apply_raw_content_override(row["raw"], override)
        outcome = outcome_from_raw(corrected_raw)
        review_required: bool | None = None
        if outcome:
            try:
                review_required = review_required_from_outcome(outcome, config)
            except ValueError:
                review_required = None

        source_content = content_from_raw(corrected_raw)
        # 최종 artifact가 아니다. artifact LLM용 source view이자 기존
        # 205개 case grouping을 유지하는 deterministic dedup seed다.
        content_text = override.get("artifact_text") or neutral_artifact(source_content)
        parsed_rule_ids, fallback_count, raw_documents = rule_ids_from_raw(
            corrected_raw, units
        )
        reviewed_rule_ids = override.get("reviewed_rule_ids", [])
        rule_ids = parsed_rule_ids or reviewed_rule_ids

        reason: str | None = None
        if not outcome:
            reason = "outcome_parse_failed"
        elif review_required is None:
            reason = "unmapped_official_result"
        elif ANSWERABILITY_EXCLUDE_RE.search(source_content):
            reason = "external_post_broadcast_evidence"
        elif not source_content or not content_text:
            reason = "content_parse_failed"
        elif not rule_ids:
            reason = "no_indexed_gold_rule"
        elif any(rule_id not in units for rule_id in rule_ids):
            reason = "manual_rule_missing_from_corpus"
        if reason:
            excluded.append({
                "decision_id": row["decision_id"],
                "source_id": row["source_id"],
                "reason": reason,
                "detected_rule_documents": raw_documents,
            })
            continue

        fallback_total += fallback_count
        prepared.append({
            **row,
            "raw_text": raw_text_from_row(corrected_raw),
            "source_content": source_content,
            "content_text": content_text,
            "content_text_source": (
                "manual_override" if override.get("artifact_text")
                else "deterministic_parser"
            ),
            "outcome": outcome,
            "review_required": review_required,
            "rule_ids": rule_ids,
            "fingerprint": dedup_fingerprint(content_text, rule_ids, outcome),
            "manual_annotation": bool(reviewed_rule_ids and not parsed_rule_ids),
            "rationale_override": override.get("rationale"),
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        grouped[row["fingerprint"]].append(row)

    groups = [grouped[key] for key in sorted(grouped)]
    return groups, excluded, {
        "prepared_row_count": len(prepared),
        "case_group_count": len(groups),
        "duplicate_rows_collapsed": len(prepared) - len(groups),
        "citation_fallback_to_parent_count": fallback_total,
    }
