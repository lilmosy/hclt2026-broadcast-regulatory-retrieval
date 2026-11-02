from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

from ..common import (
    ROOT,
    compact_hash,
    file_sha256,
    flatten_rule_units,
    load_config,
    load_jsonl,
    normalize_space,
    rule_context_text,
    write_json,
    write_jsonl,
)
SYNTHETIC_CONFIG_DIR = ROOT / "configs" / "synthetic_data"
SOURCE_CONFIG = SYNTHETIC_CONFIG_DIR / "decision_sources.yaml"
OVERRIDES_CONFIG = SYNTHETIC_CONFIG_DIR / "qa_overrides.yaml"
SYNTHETIC_CONFIG = SYNTHETIC_CONFIG_DIR / "synthetic_data.yaml"
DEFAULT_INPUT_DIR = ROOT / "data" / "decisions"
BOARD_URL = (
    "https://www.kocsc.or.kr/main/cop/bbs/selectBoardArticle.do"
    "?bbsId=info_Opinion_main&nttId={ntt_id}&menuNo=040000&subMenuNo=040700"
)
SUBITEM_LETTERS = {
    label: chr(ord("A") + index)
    for index, label in enumerate("가나다라마바사아자차카타파하")
}

DECISION_ID_RE = re.compile(
    r"(?:제)?(?P<year>20\d{2})-"
    r"(?:(?P<kind>광고|방송)-)?"
    r"(?P<session>\d{1,2})-(?P<number>\d{4})(?:호)?(?!\d)"
)
DATE_RE = re.compile(r"(?P<year>20\d{2})[.\-](?P<month>\d{1,2})[.\-](?P<day>\d{1,2})")
OUTCOME_RE = re.compile(r"문제없음|의견제시|권고|주의|경고|관계자\s*징계|과징금")
RULE_MARKER_RE = re.compile(
    r"[「｢]?\s*(?P<document>"
    r"상품소개\s*및\s*판매방송\s*심의에\s*관한\s*규정|"
    r"방송광고심의에\s*관한\s*규정|"
    r"방송심의에\s*관한\s*규정(?:\s*\(\s*규칙\s*제\d+호\s*\))?"
    r")\s*[」｣]?"
)
CITATION_TOKEN_RE = re.compile(
    r"제(?P<article>\d+)조(?:의(?P<branch>\d+))?(?:\([^)]*\))?"
    r"|제(?P<paragraph>\d+)항"
    r"|제(?P<item>\d+)호"
    r"|(?P<subitem>[가-하])목"
)
ANSWERABILITY_EXCLUDE_RE = re.compile(
    r"약\s*\d+개월\s*후\s*동일\s*방송사|"
    r"방송\s*(?:후|이후)\s*(?:제출|확보)된?\s*(?:자료|소명)|"
    r"추가\s*(?:제출|확보)\s*자료"
)

DOCUMENT_IDS = {
    "방송심의에관한규정": "BR_REVIEW",
    "방송광고심의에관한규정": "AD_REVIEW",
    "상품소개및판매방송심의에관한규정": "SALES_REVIEW",
}
DOMAIN_BY_CATEGORY = {
    "terrestrial": "broadcast_program",
    "comprehensive_news": "broadcast_program",
    "specialty": "broadcast_program",
    "broadcast_ad": "broadcast_ad",
    "product_sales": "product_sales",
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def canonical_decision_id(match: re.Match[str]) -> str:
    middle = f"-{match.group('kind')}" if match.group("kind") else ""
    return (
        f"{match.group('year')}{middle}-"
        f"{int(match.group('session')):02d}-{match.group('number')}"
    )


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00a0", " ")
    text = re.sub(
        r"상\s*품\s*소\s*개\s*및\s*판\s*매\s*방\s*송\s*심\s*의\s*에\s*관\s*한\s*규\s*정",
        "상품소개 및 판매방송 심의에 관한 규정",
        text,
    )
    text = re.sub(
        r"방\s*송\s*광\s*고\s*심\s*의\s*에\s*관\s*한\s*규\s*정",
        "방송광고심의에 관한 규정",
        text,
    )
    text = re.sub(
        r"방\s*송\s*심\s*의\s*에\s*관\s*한\s*규\s*정",
        "방송심의에 관한 규정",
        text,
    )
    text = re.sub(r"(20\d{2})\s*-\s*방\s*송\s*-\s*", r"\1-방송-", text)
    text = re.sub(r"(20\d{2})\s*-\s*광\s*고\s*-\s*", r"\1-광고-", text)
    text = re.sub(r"(?<=\d)\s*[-‐‑–—]\s*(?=\d)", "-", text)
    text = re.sub(r"(?<=\d)\s*[-‐‑–—]\s*(?=광고|방송)", "-", text)
    text = re.sub(r"(?<=광고|방송)\s*[-‐‑–—]\s*(?=\d)", "-", text)
    text = re.sub(r"제\s*(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", text)
    text = re.sub(r"제\s*(\d+)\s*(조|항|호)", r"제\1\2", text)
    text = re.sub(r"([가-하])\s*목", r"\1목", text)
    # The committee PDFs sometimes wrap a Korean word in the middle of a
    # syllable sequence.  Keep this list deliberately narrow: a general
    # Hangul-newline-Hangul join would also erase genuine word boundaries.
    split_word_repairs = {
        "염 증": "염증",
        "증 가": "증가",
        "통 증": "통증",
        "내 용": "내용",
        "표 현": "표현",
        "언 급": "언급",
        "장 면": "장면",
        "대 해": "대해",
        "발생하 지": "발생하지",
        "제품 과": "제품과",
        "내용 을": "내용을",
        "등 을": "등을",
        "연골세 포": "연골세포",
        "것 에": "것에",
        "발 생하지": "발생하지",
        "묘사하 는": "묘사하는",
        "방 송한": "방송한",
        "않도 록": "않도록",
        "것으 로": "것으로",
        "제 품": "제품",
        "종 합적으로": "종합적으로",
        "충전부 터": "충전부터",
        "판매조 건": "판매조건",
        "위 반되는": "위반되는",
        "지저 분": "지저분",
        "기어다 니는": "기어다니는",
        "패키 지": "패키지",
        "인 덕션": "인덕션",
        "무료체 험": "무료체험",
        "전립 선건강": "전립선건강",
        "있도 록": "있도록",
        "청 소년": "청소년",
        "가톨릭평화방 송": "가톨릭평화방송",
        "서 류": "서류",
        "어린 이": "어린이",
        "시 청자": "시청자",
    }
    for broken, repaired in split_word_repairs.items():
        text = re.sub(r"\s+".join(map(re.escape, broken.split())), repaired, text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def normalize_date(raw: str) -> str | None:
    match = DATE_RE.search(raw)
    if not match:
        return None
    return (
        f"{match.group('year')}-{int(match.group('month')):02d}-"
        f"{int(match.group('day')):02d}"
    )


def source_inventory(input_dir: Path, source_config: Path) -> list[dict[str, Any]]:
    configured = load_yaml(source_config).get("sources", [])
    sources: list[dict[str, Any]] = []
    for item in configured:
        path = input_dir / item["local_file"]
        if not path.exists():
            raise FileNotFoundError(f"공식 의결 PDF 누락: {path}")
        reader = PdfReader(str(path))
        sources.append({
            "source_id": item["source_id"],
            "period": item["period"],
            "category": item["category"],
            "coverage": item.get("coverage", "complete"),
            "title": item["title"],
            "board_url": BOARD_URL.format(ntt_id=item["ntt_id"]),
            "ntt_id": int(item["ntt_id"]),
            "local_file": item["local_file"],
            "sha256": file_sha256(path),
            "page_count": len(reader.pages),
        })
    return sources


def extract_rows(path: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    page_texts = [normalize_pdf_text(page.extract_text() or "") for page in reader.pages]
    spans: list[tuple[int, int, int]] = []
    parts: list[str] = []
    cursor = 0
    for page_number, page_text in enumerate(page_texts, 1):
        separator = f"\n\n[[PAGE_{page_number}]]\n\n"
        parts.append(separator)
        cursor += len(separator)
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        spans.append((start, cursor, page_number))
    text = "".join(parts)
    # 긴 의결행은 다음 페이지에 같은 의결번호 머리글을 반복한다.
    # 연속 반복을 경계로 자르지 않아야 한 사건의 raw_text가 온전히 보존된다.
    raw_matches = list(DECISION_ID_RE.finditer(text))
    matches: list[re.Match[str]] = []
    previous_decision_id: str | None = None
    for match in raw_matches:
        decision_id = canonical_decision_id(match)
        if decision_id == previous_decision_id:
            continue
        matches.append(match)
        previous_decision_id = decision_id
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = text[start:end]
        nearby = text[max(0, start - 120):match.end() + 80]
        dates = list(DATE_RE.finditer(nearby))
        decision_date = normalize_date(dates[-1].group(0)) if dates else None
        pages = [
            page_number
            for page_start, page_end, page_number in spans
            if page_start < end and page_end > start
        ]
        rows.append({
            "decision_id": canonical_decision_id(match),
            "decision_date": decision_date,
            "source_id": source["source_id"],
            "source_period": source["period"],
            "source_category": source["category"],
            "pages": pages,
            "raw": raw,
        })
    return rows


def outcome_from_raw(raw: str) -> str | None:
    matches = list(OUTCOME_RE.finditer(raw))
    if not matches:
        return None
    return normalize_space(matches[-1].group(0))


def document_id_from_marker(raw: str) -> str | None:
    compact = re.sub(r"\s+", "", raw)
    compact = re.sub(r"\(규칙제\d+호\)", "", compact)
    return DOCUMENT_IDS.get(compact)


def deepest_rule_id(
    document_id: str,
    article: str,
    branch: str | None,
    paragraph: str | None,
    item: str | None,
    subitem: str | None,
) -> str:
    rule_id = f"{document_id}_ART{article}"
    if branch:
        rule_id += f"_{branch}"
    if paragraph:
        rule_id += f"_P{paragraph}"
    if item:
        rule_id += f"_I{item}"
    if subitem:
        rule_id += f"_{SUBITEM_LETTERS.get(subitem, subitem)}"
    return rule_id


def citation_ids(citation: str, document_id: str) -> list[str]:
    output: list[str] = []
    article: str | None = None
    branch: str | None = None
    paragraph: str | None = None
    item: str | None = None
    subitem: str | None = None

    def flush() -> None:
        if not article:
            return
        rule_id = deepest_rule_id(document_id, article, branch, paragraph, item, subitem)
        if rule_id not in output:
            output.append(rule_id)

    for token in CITATION_TOKEN_RE.finditer(citation):
        if token.group("article"):
            flush()
            article = token.group("article")
            branch = token.group("branch")
            paragraph = item = subitem = None
        elif token.group("paragraph") and article:
            if paragraph is not None:
                flush()
            paragraph = token.group("paragraph")
            item = subitem = None
        elif token.group("item") and article:
            if item is not None:
                flush()
            item = token.group("item")
            subitem = None
        elif token.group("subitem") and article:
            if subitem is not None:
                flush()
            subitem = token.group("subitem")
    flush()
    return output


def resolve_to_corpus(rule_id: str, units: dict[str, dict[str, Any]]) -> tuple[str | None, bool]:
    candidate = rule_id
    while candidate:
        if candidate in units:
            return candidate, candidate != rule_id
        if "_" not in candidate:
            break
        candidate = candidate.rsplit("_", 1)[0]
    return None, False


def rule_ids_from_raw(
    raw: str,
    units: dict[str, dict[str, Any]],
) -> tuple[list[str], int, list[str]]:
    markers = list(RULE_MARKER_RE.finditer(raw))
    ids: list[str] = []
    fallback_count = 0
    raw_documents: list[str] = []
    for index, marker in enumerate(markers):
        document_id = document_id_from_marker(marker.group("document"))
        if not document_id:
            continue
        raw_documents.append(normalize_space(marker.group("document")))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw)
        tail = raw[marker.end():end]
        outcome = OUTCOME_RE.search(tail)
        if outcome:
            tail = tail[:outcome.start()]
        for candidate in citation_ids(tail, document_id):
            resolved, used_fallback = resolve_to_corpus(candidate, units)
            if resolved and resolved not in ids:
                ids.append(resolved)
                fallback_count += int(used_fallback)
    return ids, fallback_count, raw_documents


def content_from_raw(raw: str) -> str:
    marker = raw.find("○")
    if marker < 0:
        return ""
    text = raw[marker + 1:]
    rule_marker = RULE_MARKER_RE.search(text)
    if rule_marker:
        text = text[:rule_marker.start()]
    text = re.sub(r"\[\[PAGE_\d+\]\]", " ", text)
    return normalize_space(text)


def neutral_artifact(content: str) -> str:
    text = content
    review_marker = re.search(
        r"(?:해당\s*)?방송\s*내용을?\s*확인하고\s*논의한\s*결과\s*[,，]?",
        text,
    )
    if review_marker and "민원" in text[:review_marker.start()]:
        text = text[review_marker.end():]
    split_patterns = [
        r"\s*사안에\s*대해(?:\s*논의한\s*결과)?",
        r"\s*방송한\s*바\s*[,，]?",
        r"\s*(?:내용|장면)[^.!?]{0,160}?방송한\s*바\s*[,，]?",
        r"\s*방송한\s*것(?:은|으로)",
    ]
    for pattern in split_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break

    legal = re.search(
        r"관련\s*심의\s*규정(?:에|의|을|를|\s*위반)|"
        r"관련\s*심의규정(?:에|의|을|를|\s*위반)|"
        r"문제없음\s*으로\s*의결|"
        r"[｢「]방송법[｣」]\s*제100조",
        text,
    )
    if legal:
        text = text[:legal.start()]
    text = re.sub(r"\s*것은\s*$", "", text)
    text = re.sub(r"\s*-\s*$", "", text)
    return normalize_space(text).strip(" ,.-")


def validate_outcome_mapping(config: dict[str, Any]) -> None:
    mapping = config.get("qa", {}).get("outcome_mapping", {})
    review_outcomes = set(mapping.get("review_required", []))
    no_flag_outcomes = set(mapping.get("no_flag", []))
    if not review_outcomes or not no_flag_outcomes:
        raise ValueError("outcome_mapping의 review_required/no_flag 목록은 비어 있을 수 없습니다.")
    overlap = review_outcomes & no_flag_outcomes
    if overlap:
        raise ValueError(f"outcome_mapping 중복: {sorted(overlap)}")


def review_required_from_outcome(outcome: str, config: dict[str, Any]) -> bool:
    mapping = config["qa"]["outcome_mapping"]
    if outcome in set(mapping["review_required"]):
        return True
    if outcome in set(mapping["no_flag"]):
        return False
    raise ValueError(f"outcome_mapping에 없는 공식 결과: {outcome}")


def article_ids(rule_ids: list[str], units: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({units[rule_id]["article_rule_id"] for rule_id in rule_ids})


def evidence_structure(
    rule_ids: list[str],
    units: dict[str, dict[str, Any]],
) -> str:
    pages = article_ids(rule_ids, units)
    return "single_page" if len(pages) <= 1 else "multi_page"


SUBITEM_LABELS = {
    chr(ord("A") + index): f"{label}목"
    for index, label in enumerate("가나다라마바사아자차카타파하")
}


def rule_reference(rule_id: str, units: dict[str, dict[str, Any]]) -> dict[str, str]:
    article_id = units[rule_id]["article_rule_id"]
    article = units[article_id]
    labels = [article["article"]]
    suffix = rule_id[len(article_id):].lstrip("_")
    for token in filter(None, suffix.split("_")):
        if token.startswith("P") and token[1:].isdigit():
            labels.append(f"제{int(token[1:])}항")
        elif token.startswith("I") and token[1:].isdigit():
            labels.append(f"제{int(token[1:])}호")
        elif token in SUBITEM_LABELS:
            labels.append(SUBITEM_LABELS[token])
    return {
        "rule_id": rule_id,
        "regulation": article["document_name"],
        "article": " ".join(labels),
    }


def dedup_fingerprint(artifact_seed: str, rule_ids: list[str], outcome: str) -> str:
    material = re.sub(r"\s+", "", artifact_seed)
    return compact_hash("|".join([material, ",".join(sorted(rule_ids)), outcome]), 24)


def qa_id_for_group(group: list[dict[str, Any]]) -> str:
    provenance_keys = sorted({
        f"{row['source_id']}::{row['decision_id']}"
        for row in group
    })
    return f"QA_{compact_hash('|'.join(provenance_keys), 12).upper()}"


def run_static_check(
    decision_chunks_path: Path,
    config: dict[str, Any],
    corpus: list[dict[str, Any]],
) -> None:
    """LLM 호출 없이 생성 전제를 검증한다: prompt 로딩·placeholder 치환·Gold rule join."""
    from .llm_generation import load_prompt, render_prompt

    cases = load_jsonl(decision_chunks_path)
    units = flatten_rule_units(corpus)
    project_root = ROOT
    prompt_config = config["generation"]
    artifact_template, _ = load_prompt(project_root / prompt_config["artifact_prompt"])
    rationale_template, _ = load_prompt(
        project_root / prompt_config["rationale_prompt"]
    )
    failures: list[str] = []
    multi_rule = 0
    for case in cases:
        qa_id = case.get("qa_id", "?")
        try:
            if not re.fullmatch(r"QA_[A-F0-9]{12}", qa_id):
                raise ValueError(f"잘못된 qa_id 형식: {qa_id}")
            representative = case["chunks"][0]
            if not representative["raw_text"]:
                raise ValueError("raw_text가 비어 있습니다.")
            gold_rules = [
                {
                    **rule_reference(rule_id, units),
                    "rule_text": rule_context_text(rule_id, units),
                }
                for rule_id in case["gold_rule_ids"]
            ]
            if len(case["gold_rule_ids"]) > 1:
                multi_rule += 1
            render_prompt(
                artifact_template,
                {
                    "DOMAIN": case["domain"],
                    "BROADCAST_DATE": representative.get("broadcast_date") or "미상",
                    "RAW_TEXT": representative["raw_text"],
                },
            )
            render_prompt(
                rationale_template,
                {
                    "ARTIFACT": "(정적 검증 자리표시자 — 실제로는 ① artifact 호출 결과)",
                    "RAW_TEXT": representative["raw_text"],
                    "GOLD_RULES": json.dumps(gold_rules, ensure_ascii=False, indent=2),
                },
            )
        except Exception as exc:  # 검증 모드에서는 사례별로 모아 보고한다.
            failures.append(f"{qa_id}: {exc}")
    if len({case["qa_id"] for case in cases}) != len(cases):
        failures.append("중복 qa_id가 있습니다.")
    if failures:
        raise SystemExit(
            f"static check FAIL ({len(failures)}건):\n" + "\n".join(failures[:20])
        )
    print(
        f"static check PASS: {len(cases)} cases ({multi_rule} multi-rule), "
        "prompt 렌더링·placeholder 치환·Gold rule join 이상 없음"
    )


def validate(
    qa_rows: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    units = set(flatten_rule_units(corpus))
    source_ids = {item["source_id"] for item in sources}
    invalid_rule_ids = sorted({
        rule["rule_id"]
        for row in qa_rows
        for rule in row["gold"]["applicable_regulations"]
        if rule["rule_id"] not in units
    })
    invalid_source_ids = sorted({
        source["source_id"]
        for row in qa_rows
        for source in row["provenance"]["sources"]
        if source["source_id"] not in source_ids
    })
    empty_artifacts = [row["qa_id"] for row in qa_rows if not row["artifact"]["text"]]
    empty_rationales = [
        row["qa_id"]
        for row in qa_rows
        if any(
            not rule.get("rationale")
            for rule in row["gold"]["applicable_regulations"]
        )
    ]
    duplicate_qa_ids = len({row["qa_id"] for row in qa_rows}) != len(qa_rows)
    invalid_qa_ids = sorted({
        row["qa_id"]
        for row in qa_rows
        if not re.fullmatch(r"QA_[A-F0-9]{12}", row["qa_id"])
    })
    artifact_leakage_flags = [
        row["qa_id"]
        for row in qa_rows
        if re.search(r"관련\s*심의\s*규정|심의규정에\s*위반|제\d+조", row["artifact"]["text"])
    ]
    invalid_review_required = sorted({
        row["qa_id"]
        for row in qa_rows
        if type(row["gold"].get("review_required")) is not bool
    })
    invalid_overall_answer = sorted({
        row["qa_id"]
        for row in qa_rows
        if (
            len(row["gold"]["applicable_regulations"]) == 1
            and row["gold"].get("overall_answer") is not None
        ) or (
            len(row["gold"]["applicable_regulations"]) > 1
            and not row["gold"].get("overall_answer")
        )
    })
    ok = not (
        invalid_rule_ids
        or invalid_source_ids
        or empty_artifacts
        or empty_rationales
        or duplicate_qa_ids
        or invalid_qa_ids
        or invalid_review_required
        or invalid_overall_answer
    )
    return {
        "qa_count": len(qa_rows),
        "source_count": len(sources),
        "invalid_rule_ids": invalid_rule_ids,
        "invalid_source_ids": invalid_source_ids,
        "empty_artifacts": empty_artifacts,
        "empty_rationales": empty_rationales,
        "duplicate_qa_ids": duplicate_qa_ids,
        "invalid_qa_ids": invalid_qa_ids,
        "invalid_review_required": invalid_review_required,
        "invalid_overall_answer": invalid_overall_answer,
        "artifact_leakage_flags": artifact_leakage_flags,
        "ok": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "legal_corpus.jsonl",
    )
    parser.add_argument(
        "--decision-chunks",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "decision_chunks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "synthetic_qa.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "synthetic_qa_report.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="LLM 호출 없이 프롬프트·placeholder·Gold rule join을 정적으로 검증한다",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--pricing-version")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "generation_cache.jsonl",
    )
    parser.add_argument(
        "--calls",
        type=Path,
        default=ROOT / "outputs" / "synthetic_data" / "generation_calls.jsonl",
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    config = load_config(SYNTHETIC_CONFIG)
    corpus = load_jsonl(args.corpus)

    if args.check:
        run_static_check(args.decision_chunks, config, corpus)
        return

    sources = source_inventory(args.input_dir, args.source_config)

    if args.generate:
        required = {
            "--model": args.model,
            "--input-price-per-million": args.input_price_per_million,
            "--output-price-per-million": args.output_price_per_million,
            "--pricing-version": args.pricing_version,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "--generate에는 다음 인자가 필요합니다: " + ", ".join(missing)
            )
        from .llm_generation import generate_qa_rows

        prompt_config = config["generation"]
        project_root = ROOT
        qa_rows, generation = generate_qa_rows(
            cases=load_jsonl(args.decision_chunks),
            corpus=corpus,
            artifact_prompt_path=project_root / prompt_config["artifact_prompt"],
            rationale_prompt_path=project_root / prompt_config["rationale_prompt"],
            model=args.model,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            pricing_version=args.pricing_version,
            cache_path=args.cache,
            calls_path=args.calls,
            rule_reference=rule_reference,
            limit=args.limit,
        )
        validation = validate(qa_rows, corpus, sources)
        report = {
            "schema_version": config["dataset"]["schema_version"],
            "input_case_count": len(load_jsonl(args.decision_chunks)),
            "output_qa_count": len(qa_rows),
            "partial_generation": args.limit is not None,
            "generation": generation,
            "source_count": len(sources),
            "sources": sources,
            "validation": validation,
            "build_status": "draft_requires_human_review",
        }
        if args.validate and not validation["ok"]:
            raise SystemExit(f"QA 검증 실패: {validation}")
        write_jsonl(args.output, qa_rows)
        write_json(args.report, report)
        print(f"synthetic_qa: {len(qa_rows)} LLM-generated cases -> {args.output}")
        print(
            f"model: {args.model}, calls: {generation['new_call_count']}, "
            f"estimated cost: ${generation['estimated_cost_usd']:.6f}"
        )
        print(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")
        return

    parser.error(
        "deterministic 생성 경로는 제거되었습니다. "
        "--check(호출 없는 정적 검증) 또는 --generate(LLM 생성, model/가격 인자 필요)를 사용하세요."
    )


if __name__ == "__main__":
    main()
