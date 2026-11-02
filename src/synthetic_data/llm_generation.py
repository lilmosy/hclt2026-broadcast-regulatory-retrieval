"""Synthetic QA의 artifact와 rationale을 2단 LLM 호출로 생성한다.

  ① domain + broadcast_date + raw_text          → artifact
  ② 생성된 artifact + raw_text + Gold rule 조문 → rationales + overall_answer

이 모듈은 provider 호출, structured output, stage별 cache와 call log만 담당한다.
case 선별과 Gold ID는 변경하지 않는다. 실제 호출은 build_synthetic_data.py의
--generate와 model/가격 인자가 모두 주어진 경우에만 일어난다.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from ..common import (
    ROOT,
    compact_hash,
    flatten_rule_units,
    load_jsonl,
    normalize_space,
    rule_context_text,
    write_jsonl,
)


class ArtifactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact: str = Field(min_length=1)


class RuleRationaleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    rationale_sentences: list[str] = Field(min_length=4, max_length=4)


class RationaleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationales: list[RuleRationaleOutput] = Field(min_length=1)
    overall_answer: str | None = None


RATIONALE_OUTPUT_VALIDATION_ATTEMPTS = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt가 비어 있습니다: {path}")
    return text, compact_hash(text, 64)


def render_prompt(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for name, value in replacements.items():
        rendered = rendered.replace("{" + name + "}", value)
    unresolved = [
        token
        for token in (
            "{DOMAIN}",
            "{BROADCAST_DATE}",
            "{RAW_TEXT}",
            "{ARTIFACT}",
            "{GOLD_RULES}",
        )
        if token in rendered
    ]
    if unresolved:
        raise ValueError(f"prompt placeholder가 남았습니다: {unresolved}")
    return rendered


class OpenAIGenerator:
    def __init__(
        self,
        *,
        model: str,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> None:
        self.model = model
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        load_dotenv(ROOT / ".env", override=False)
        import openai

        self._openai = openai
        self._client = openai.OpenAI()

    def call(
        self,
        *,
        qa_id: str,
        stage: str,
        system: str,
        prompt: str,
        output_schema: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, Any]]:
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=output_schema,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("structured output이 비어 있습니다.")
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost = (
                    input_tokens * self.input_price
                    + output_tokens * self.output_price
                ) / 1_000_000
                return parsed, {
                    "qa_id": qa_id,
                    "stage": stage,
                    "model": self.model,
                    "attempt": attempt,
                    "timestamp": utc_now(),
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": round(cost, 8),
                    "success": True,
                }
            except self._openai.RateLimitError as exc:
                last_error = exc
                if attempt == 8:
                    break
                time.sleep(20 * attempt)
            except (
                self._openai.APIConnectionError,
                self._openai.APITimeoutError,
            ) as exc:
                last_error = exc
                if attempt == 8:
                    break
                time.sleep(5 * attempt)
        raise RuntimeError(
            f"{qa_id}/{stage}: OpenAI 호출 실패: {last_error}"
        ) from last_error


def validate_rationale_output(
    output: RationaleOutput,
    gold_rules: list[dict[str, str]],
) -> dict[str, str]:
    gold_rule_ids = [row["rule_id"] for row in gold_rules]
    gold_by_id = {row["rule_id"]: row for row in gold_rules}
    received = [row.rule_id for row in output.rationales]
    if len(received) != len(set(received)):
        raise ValueError(f"중복 rationale rule_id: {received}")
    if received != gold_rule_ids:
        raise ValueError(
            f"rationale rule mismatch: expected={gold_rule_ids}, received={received}"
        )
    if len(gold_rule_ids) == 1 and output.overall_answer is not None:
        raise ValueError("단일 Gold rule의 overall_answer는 null이어야 합니다.")
    if len(gold_rule_ids) > 1 and not output.overall_answer:
        raise ValueError("복수 Gold rule에는 overall_answer가 필요합니다.")

    rationale_by_rule: dict[str, str] = {}
    for row in output.rationales:
        sentences = [sentence.strip() for sentence in row.rationale_sentences]
        if any(not sentence for sentence in sentences):
            raise ValueError(f"빈 rationale 문장: {row.rule_id}")
        expected_quote = normalize_space(gold_by_id[row.rule_id]["rule_text"])
        actual_quote_sentence = normalize_space(sentences[2])
        if expected_quote not in actual_quote_sentence:
            # 모델이 법령 원문의 문장부호를 "교정"하는 경우가 있다. 인용부가
            # 사실상 동일할 때만 canonical corpus 문자열로 되돌리고, 생략이나
            # 의미 변경처럼 차이가 큰 응답은 아래 exact-match 검사에서 거절한다.
            quote_start = sentences[2].find("‘")
            quote_end = sentences[2].find("’", quote_start + 1)
            if quote_start >= 0 and quote_end > quote_start:
                generated_quote = normalize_space(
                    sentences[2][quote_start + 1:quote_end]
                )
                similarity = SequenceMatcher(
                    None, generated_quote, expected_quote
                ).ratio()
                if similarity >= 0.995:
                    sentences[2] = (
                        sentences[2][:quote_start + 1]
                        + expected_quote
                        + sentences[2][quote_end:]
                    )
                    actual_quote_sentence = normalize_space(sentences[2])
        if expected_quote not in actual_quote_sentence:
            raise ValueError(
                f"세 번째 문장에 Gold 규정 원문이 그대로 인용되지 않았습니다: "
                f"{row.rule_id}"
            )
        problem_judgment = normalize_space(sentences[1])
        if "문제" not in problem_judgment:
            raise ValueError(
                f"두 번째 문장에 문제성 판단이 없습니다: {row.rule_id}"
            )
        requirement_application = actual_quote_sentence.partition(
            expected_quote
        )[2].strip()
        application_markers = ("요건", "해당", "충족", "연결", "위반", "의무")
        if (
            len(requirement_application) < 15
            or not any(
                marker in requirement_application
                for marker in application_markers
            )
        ):
            raise ValueError(
                f"세 번째 문장의 규정 인용 뒤에 요건 적용 설명이 부족합니다: "
                f"{row.rule_id}"
            )
        conclusion = normalize_space(sentences[3])
        conclusion_markers = (
            "필요",
            "유지",
            "수정",
            "삭제",
            "보완",
            "재검토",
            "방송",
        )
        if not any(marker in conclusion for marker in conclusion_markers):
            raise ValueError(
                f"네 번째 문장의 방송 전 결론이 불명확합니다: {row.rule_id}"
            )
        rationale_by_rule[row.rule_id] = " ".join(sentences)
    return rationale_by_rule


def generate_qa_rows(
    *,
    cases: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    artifact_prompt_path: Path,
    rationale_prompt_path: Path,
    model: str,
    input_price_per_million: float,
    output_price_per_million: float,
    pricing_version: str,
    cache_path: Path,
    calls_path: Path,
    rule_reference: Callable[[str, dict[str, dict[str, Any]]], dict[str, str]],
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = cases[:limit] if limit is not None else cases
    units = flatten_rule_units(corpus)
    artifact_template, artifact_hash = load_prompt(artifact_prompt_path)
    rationale_template, rationale_hash = load_prompt(rationale_prompt_path)
    # stage별 system prompt.
    artifact_system = "지시를 정확히 따르고 JSON object만 출력한다."
    rationale_system = (
        "당신은 공식 방송심의 사례의 Gold rationale을 만드는 데이터 편집자다. "
        "제공된 사실과 Gold 규정 밖의 내용을 만들지 않고 JSON object만 출력한다."
    )
    cache_rows = load_jsonl(cache_path)
    calls = load_jsonl(calls_path)
    cache_by_id = {
        row["qa_id"]: row for row in cache_rows if row.get("model") == model
    }
    generator: OpenAIGenerator | None = None
    qa_rows: list[dict[str, Any]] = []
    run_calls: list[dict[str, Any]] = []
    reused_count = 0

    def ensure_generator() -> OpenAIGenerator:
        nonlocal generator
        if generator is None:
            generator = OpenAIGenerator(
                model=model,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
            )
        return generator

    for case in selected:
        qa_id = case["qa_id"]
        representative = case["chunks"][0]
        gold_rules = [
            {
                **rule_reference(rule_id, units),
                "rule_text": rule_context_text(rule_id, units),
            }
            for rule_id in case["gold_rule_ids"]
        ]
        broadcast_date = representative.get("broadcast_date") or "미상"
        case_new_calls = 0

        # ① artifact stage — 실제 입력 전체가 hash에 들어간다.
        artifact_input_hash = compact_hash(
            json.dumps(
                {
                    "model": model,
                    "system": artifact_system,
                    "prompt_sha256": artifact_hash,
                    "domain": case["domain"],
                    "broadcast_date": broadcast_date,
                    "raw_text": representative["raw_text"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            64,
        )
        cached = cache_by_id.get(qa_id)
        if cached and cached.get("artifact_input_sha256") == artifact_input_hash:
            artifact_text = cached["artifact"]
        else:
            artifact_prompt = render_prompt(
                artifact_template,
                {
                    "DOMAIN": case["domain"],
                    "BROADCAST_DATE": broadcast_date,
                    "RAW_TEXT": representative["raw_text"],
                },
            )
            artifact_result, artifact_call = ensure_generator().call(
                qa_id=qa_id,
                stage="artifact",
                system=artifact_system,
                prompt=artifact_prompt,
                output_schema=ArtifactOutput,
            )
            artifact_text = artifact_result.artifact.strip()
            artifact_call["prompt_sha256"] = artifact_hash
            artifact_call["input_sha256"] = artifact_input_hash
            run_calls.append(artifact_call)
            calls.append(artifact_call)
            write_jsonl(calls_path, calls)
            case_new_calls += 1
            # artifact 성공 즉시 기록 → rationale이 실패해도 재실행 시
            # artifact를 다시 호출하지 않는다.
            cache_rows = [
                row
                for row in cache_rows
                if not (row.get("qa_id") == qa_id and row.get("model") == model)
            ]
            cached = {
                "qa_id": qa_id,
                "model": model,
                "artifact_prompt_sha256": artifact_hash,
                "artifact_input_sha256": artifact_input_hash,
                "artifact": artifact_text,
                "rationale_prompt_sha256": None,
                "rationale_input_sha256": None,
                "rationales": None,
                "overall_answer": None,
                "generated_at": utc_now(),
            }
            cache_rows.append(cached)
            cache_by_id[qa_id] = cached
            write_jsonl(cache_path, cache_rows)

        # ② rationale stage — 생성된 artifact가 입력이자 hash 재료다.
        rationale_input_hash = compact_hash(
            json.dumps(
                {
                    "model": model,
                    "system": rationale_system,
                    "prompt_sha256": rationale_hash,
                    "artifact": artifact_text,
                    "raw_text": representative["raw_text"],
                    "gold_rules": gold_rules,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            64,
        )
        if (
            cached.get("rationale_input_sha256") == rationale_input_hash
            and cached.get("rationales")
        ):
            rationale_by_rule = cached["rationales"]
            overall_answer = cached.get("overall_answer")
        else:
            rationale_prompt = render_prompt(
                rationale_template,
                {
                    "ARTIFACT": artifact_text,
                    "RAW_TEXT": representative["raw_text"],
                    "GOLD_RULES": json.dumps(
                        gold_rules, ensure_ascii=False, indent=2
                    ),
                },
            )
            retry_feedback = ""
            for validation_attempt in range(
                1, RATIONALE_OUTPUT_VALIDATION_ATTEMPTS + 1
            ):
                request_prompt = rationale_prompt + retry_feedback
                rationale_result, rationale_call = ensure_generator().call(
                    qa_id=qa_id,
                    stage="rationale",
                    system=rationale_system,
                    prompt=request_prompt,
                    output_schema=RationaleOutput,
                )
                rationale_call["prompt_sha256"] = rationale_hash
                rationale_call["input_sha256"] = rationale_input_hash
                rationale_call["request_sha256"] = compact_hash(request_prompt, 64)
                rationale_call["output_validation_attempt"] = validation_attempt
                try:
                    rationale_by_rule = validate_rationale_output(
                        rationale_result, gold_rules
                    )
                except ValueError as exc:
                    # API 호출 자체는 성공했으므로 토큰·비용을 버리지 않고
                    # 실패 원인과 함께 기록한 뒤 이 rationale만 다시 요청한다.
                    rationale_call["success"] = False
                    rationale_call["api_success"] = True
                    rationale_call["output_validation_success"] = False
                    rationale_call["error"] = str(exc)
                    rationale_call["rejected_output"] = rationale_result.model_dump()
                    run_calls.append(rationale_call)
                    calls.append(rationale_call)
                    write_jsonl(calls_path, calls)
                    case_new_calls += 1
                    if validation_attempt == RATIONALE_OUTPUT_VALIDATION_ATTEMPTS:
                        raise
                    retry_feedback = (
                        "\n\n[이전 응답 검증 실패 — 전체 JSON을 다시 생성할 것]\n"
                        f"실패 원인: {exc}\n"
                        "특히 각 rationale_sentences의 세 번째 문자열에는 위 GOLD_RULES의 "
                        "해당 rule_text 전체를 글자·문장부호·순서 변경 없이 그대로 복사한 "
                        "뒤, 같은 문자열 안에서 요건 적용 설명을 이어 써라. "
                        "요약·생략·교정·따옴표 변경을 하지 마라."
                    )
                    continue

                overall_answer = rationale_result.overall_answer
                rationale_call["output_validation_success"] = True
                run_calls.append(rationale_call)
                calls.append(rationale_call)
                write_jsonl(calls_path, calls)
                case_new_calls += 1
                break
            cached["rationale_prompt_sha256"] = rationale_hash
            cached["rationale_input_sha256"] = rationale_input_hash
            cached["rationales"] = rationale_by_rule
            cached["overall_answer"] = overall_answer
            cached["generated_at"] = utc_now()
            write_jsonl(cache_path, cache_rows)

        if case_new_calls == 0:
            reused_count += 1

        regulations = [
            {
                **rule_reference(rule_id, units),
                "rationale": rationale_by_rule[rule_id],
            }
            for rule_id in case["gold_rule_ids"]
        ]
        source_refs: list[dict[str, Any]] = []
        for source_id in sorted({c["source_id"] for c in case["chunks"]}):
            source_chunks = [
                c for c in case["chunks"] if c["source_id"] == source_id
            ]
            source_refs.append({
                "source_id": source_id,
                "pages": sorted({
                    page for chunk in source_chunks for page in chunk["pages"]
                }),
            })
        qa_rows.append({
            "qa_id": qa_id,
            "artifact": {"domain": case["domain"], "text": artifact_text},
            "gold": {
                "review_required": case["review_required"],
                "applicable_regulations": regulations,
                "overall_answer": overall_answer,
            },
            "provenance": {
                "decision_ids": sorted({
                    chunk["decision_id"] for chunk in case["chunks"]
                }),
                "sources": source_refs,
                "official_result": case["official_result"],
            },
            "review_status": "auto_draft",
        })

    selected_qa_ids = {case["qa_id"] for case in selected}
    cumulative_calls = [
        row
        for row in calls
        if row.get("model") == model and row.get("qa_id") in selected_qa_ids
    ]

    return qa_rows, {
        "provider": "openai",
        "model": model,
        "pricing_version": pricing_version,
        "input_price_per_million_usd": input_price_per_million,
        "output_price_per_million_usd": output_price_per_million,
        "artifact_system_sha256": compact_hash(artifact_system, 64),
        "rationale_system_sha256": compact_hash(rationale_system, 64),
        "artifact_prompt_sha256": artifact_hash,
        "rationale_prompt_sha256": rationale_hash,
        "selected_case_count": len(selected),
        "cache_reused_case_count": reused_count,
        "new_call_count": len(run_calls),
        "input_tokens": sum(row["input_tokens"] for row in run_calls),
        "output_tokens": sum(row["output_tokens"] for row in run_calls),
        "estimated_cost_usd": round(
            sum(row["estimated_cost_usd"] for row in run_calls), 8
        ),
        "cumulative_logged_call_count": len(cumulative_calls),
        "cumulative_input_tokens": sum(
            row.get("input_tokens", 0) for row in cumulative_calls
        ),
        "cumulative_output_tokens": sum(
            row.get("output_tokens", 0) for row in cumulative_calls
        ),
        "cumulative_estimated_cost_usd": round(
            sum(row.get("estimated_cost_usd", 0) for row in cumulative_calls), 8
        ),
        "cumulative_failed_output_validation_count": sum(
            row.get("output_validation_success") is False
            for row in cumulative_calls
        ),
        "cache_path": str(cache_path),
        "calls_path": str(calls_path),
    }
