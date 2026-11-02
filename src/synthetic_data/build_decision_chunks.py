"""공식 PDF에서 Synthetic QA 생성용 canonical case input을 만든다.

사례 선별·제외·중복 통합·qa_id 계약은 case_groups.prepare_case_groups가
유일하게 담당한다. 이 CLI는 그 결과를 decision_chunks.jsonl과 report로
직렬화할 뿐이며 artifact/rationale을 생성하지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ..common import ROOT, flatten_rule_units, load_jsonl, write_json, write_jsonl
from .build_synthetic_data import (
    DEFAULT_INPUT_DIR,
    DOMAIN_BY_CATEGORY,
    OVERRIDES_CONFIG,
    SOURCE_CONFIG,
    SYNTHETIC_CONFIG,
    evidence_structure,
    extract_rows,
    load_yaml,
    qa_id_for_group,
    source_inventory,
    validate_outcome_mapping,
)
from .case_groups import prepare_case_groups


def build_chunk_rows(
    groups: list[list[dict[str, Any]]],
    units: dict[str, dict[str, Any]],
    board_urls: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in groups:
        first = group[0]
        output.append({
            "qa_id": qa_id_for_group(group),
            "domain": DOMAIN_BY_CATEGORY[first["source_category"]],
            "official_result": first["outcome"],
            "review_required": first["review_required"],
            "gold_rule_ids": first["rule_ids"],
            "evidence_structure": evidence_structure(first["rule_ids"], units),
            "chunks": [
                {
                    "decision_id": row["decision_id"],
                    # 기존 decision_date parser가 실제로 잡는 것은 방송일이다.
                    "broadcast_date": row["decision_date"],
                    "source_id": row["source_id"],
                    "source_category": row["source_category"],
                    "pages": row["pages"],
                    "board_url": board_urls[row["source_id"]],
                    # LLM 입력은 raw_text뿐이다. content_text는 case_groups의
                    # dedup fingerprint 내부 로직으로만 남고 출력하지 않는다.
                    "raw_text": row["raw_text"],
                }
                for row in group
            ],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_CONFIG)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "legal_corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "decision_chunks.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "decision_chunks_report.json",
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    corpus = load_jsonl(args.corpus)
    units = flatten_rule_units(corpus)
    sources = source_inventory(args.input_dir, args.source_config)
    config = load_yaml(SYNTHETIC_CONFIG)
    validate_outcome_mapping(config)
    manual = load_yaml(args.overrides).get("decisions", {})

    rows: list[dict[str, Any]] = []
    for source in sources:
        rows.extend(extract_rows(args.input_dir / source["local_file"], source))
    groups, excluded, preparation = prepare_case_groups(rows, units, manual, config)
    board_urls = {source["source_id"]: source["board_url"] for source in sources}
    chunk_rows = build_chunk_rows(groups, units, board_urls)

    # 같은 내용이 서로 다른 Gold/결과로 남은 case를 자동 병합하지 않고
    # 사람이 검수할 수 있도록 별도 경고 목록으로 보존한다.
    content_variants: dict[str, list[list[dict[str, Any]]]] = {}
    for group in groups:
        content_key = "".join(group[0]["content_text"].split())
        content_variants.setdefault(content_key, []).append(group)
    same_content_different_gold = []
    for variants in content_variants.values():
        if len(variants) < 2:
            continue
        same_content_different_gold.append({
            "cases": [
                {
                    "qa_id": qa_id_for_group(group),
                    "decision_ids": [row["decision_id"] for row in group],
                    "gold_rule_ids": group[0]["rule_ids"],
                    "official_result": group[0]["outcome"],
                }
                for group in variants
            ]
        })

    representative_raw_lengths = sorted(
        len(row["chunks"][0]["raw_text"]) for row in chunk_rows
    )
    validation = {
        "duplicate_qa_ids": len({row["qa_id"] for row in chunk_rows}) != len(chunk_rows),
        "empty_raw_text": [
            row["qa_id"] for row in chunk_rows if not row["chunks"][0]["raw_text"]
        ],
    }
    validation["ok"] = not any(validation.values())

    def length_summary(values: list[int]) -> dict[str, int]:
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
        }

    report = {
        "source_count": len(sources),
        "sources": sources,
        "raw_decision_row_count": len(rows),
        "rows_with_indexed_gold": preparation["prepared_row_count"],
        "independent_case_count": preparation["case_group_count"],
        "duplicate_rows_collapsed": preparation["duplicate_rows_collapsed"],
        "same_content_different_gold_group_count": len(same_content_different_gold),
        "same_content_different_gold_case_count": sum(
            len(item["cases"]) for item in same_content_different_gold
        ),
        "same_content_different_gold": same_content_different_gold,
        "citation_fallback_to_parent_count": preparation[
            "citation_fallback_to_parent_count"
        ],
        "excluded_count": len(excluded),
        "excluded_by_reason": dict(sorted(
            Counter(item["reason"] for item in excluded).items()
        )),
        "excluded_decisions": excluded,
        "distribution": {
            "domain": dict(sorted(Counter(row["domain"] for row in chunk_rows).items())),
            "official_result": dict(sorted(
                Counter(row["official_result"] for row in chunk_rows).items()
            )),
            "evidence_structure": dict(sorted(
                Counter(row["evidence_structure"] for row in chunk_rows).items()
            )),
        },
        "representative_raw_text_length": length_summary(
            representative_raw_lengths
        ),
        "validation": validation,
    }
    if args.validate and not validation["ok"]:
        write_json(args.report, report)
        raise SystemExit(f"decision chunk 검증 실패: {validation}")

    write_jsonl(args.output, chunk_rows)
    write_json(args.report, report)
    print(f"decision_chunks: {len(chunk_rows)} independent cases -> {args.output}")
    print(
        f"rows: {len(rows)} parsed, {len(excluded)} excluded, "
        f"{preparation['duplicate_rows_collapsed']} duplicates collapsed"
    )
    print(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
