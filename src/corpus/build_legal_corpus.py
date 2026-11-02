from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import (
    ROOT,
    load_config,
    normalize_space,
    render_article_text,
    write_json,
    write_jsonl,
)


ARTICLE_RE = re.compile(r"^제(?P<article>\d+)조(?:의(?P<branch>\d+))?(?:\((?P<heading>[^)]*)\))?\s*(?P<rest>.*)$")
CHAPTER_RE = re.compile(r"^제\d+장\s*.*$")
SECTION_RE = re.compile(r"^제\d+절\s*.*$")
APPENDIX_RE = re.compile(r"^\d{8}\s+\d+\s+부칙\s*$")
REVISION_NOTE_RE = re.compile(r"^\[(?:전문개정|본조신설|본조삭제|제목개정|개정|삭제)[^]]*\]\s*$")
ITEM_RE = re.compile(r"^(?P<num>\d+)\.\s*(?P<text>.*)$")
SUBITEM_RE = re.compile(r"^(?P<label>[가-하])\.\s*(?P<text>.*)$")
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
# 일부 원문 TXT에 제1항 마커가 유사 문자(ⓛ)로 들어온 경우를 정규화한다.
CIRCLED_ALIASES = {"ⓛ": "①"}
CIRCLED_RE = re.compile(f"([{CIRCLED}{''.join(CIRCLED_ALIASES)}])")
KOREAN_ITEM_LABELS = list("가나다라마바사아자차카타파하")


@dataclass
class DraftNode:
    rule_id: str
    label: str
    text_parts: list[str] = field(default_factory=list)
    children: list["DraftNode"] = field(default_factory=list)

    def append(self, text: str) -> None:
        text = text.strip()
        if text:
            self.text_parts.append(text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "text": normalize_space(" ".join(self.text_parts)),
            "children": [child.as_dict() for child in self.children],
        }


def parse_header(raw: str) -> dict[str, str | None]:
    lines = raw.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else ""
    meta = lines[1] if len(lines) > 1 else ""
    effective = re.search(r"시행일:\s*(\d{8})", meta)
    serial = re.search(r"일련번호:\s*(\d+)", meta)
    collected = next((re.search(r"수집일:\s*([0-9-]+)", line) for line in lines[:6] if "수집일" in line), None)
    long_meta = next((line for line in lines if line and not line.startswith("#")), "")
    rule_number = None
    if effective:
        match = re.search(rf"{effective.group(1)}\s+(\d+)\s+(?:일부개정|전부개정|제정)", long_meta)
        if match:
            rule_number = f"제{match.group(1)}호"
    date = effective.group(1) if effective else None
    return {
        "document_name": title,
        "effective_date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else None,
        "serial_number": serial.group(1) if serial else None,
        "rule_number": rule_number,
        "collected_date": collected.group(1) if collected else None,
    }


def extract_operative_lines(raw: str) -> list[str]:
    lines = raw.splitlines()
    start_index = None
    first_prefix = ""
    for index, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        chapter_match = re.search(r"제\d+장\s*.*$", line)
        article_match = re.search(r"제\d+조(?:의\d+)?(?:\([^)]*\))?", line)
        if chapter_match:
            start_index = index
            first_prefix = chapter_match.group(0)
            break
        if article_match:
            start_index = index
            first_prefix = line[article_match.start():]
            break
    if start_index is None:
        raise ValueError("현행 본문 시작점을 찾지 못했습니다.")

    operative = [first_prefix]
    for line in lines[start_index + 1:]:
        stripped = line.strip()
        if APPENDIX_RE.match(stripped):
            break
        if not stripped or REVISION_NOTE_RE.match(stripped):
            continue
        operative.append(stripped)
    return operative


def split_paragraph_markers(lines: list[str]) -> list[str]:
    output: list[str] = []
    for line in lines:
        if not CIRCLED_RE.search(line):
            output.append(line)
            continue
        parts = CIRCLED_RE.split(line)
        prefix = parts[0].strip()
        if prefix:
            output.append(prefix)
        for index in range(1, len(parts), 2):
            marker = CIRCLED_ALIASES.get(parts[index], parts[index])
            text = parts[index + 1].strip() if index + 1 < len(parts) else ""
            output.append(marker + " " + text)
    return output


def article_rule_id(document_id: str, article: str, branch: str | None) -> str:
    suffix = f"ART{article}" + (f"_{branch}" if branch else "")
    return f"{document_id}_{suffix}"


def parse_sub_items(article_id: str, body_lines: list[str]) -> tuple[str, list[dict[str, Any]]]:
    roots: list[DraftNode] = []
    article_lead: list[str] = []
    current_paragraph: DraftNode | None = None
    current_item: DraftNode | None = None
    current_subitem: DraftNode | None = None

    for line in split_paragraph_markers(body_lines):
        line = line.strip()
        if not line:
            continue
        if line[0] in CIRCLED:
            number = CIRCLED.index(line[0]) + 1
            current_paragraph = DraftNode(f"{article_id}_P{number}", f"제{number}항")
            current_paragraph.append(line[1:].strip())
            roots.append(current_paragraph)
            current_item = current_subitem = None
            continue
        item_match = ITEM_RE.match(line)
        if item_match:
            number = int(item_match.group("num"))
            parent_id = current_paragraph.rule_id if current_paragraph else article_id
            current_item = DraftNode(f"{parent_id}_I{number}", f"제{number}호")
            current_item.append(item_match.group("text"))
            if current_paragraph:
                current_paragraph.children.append(current_item)
            else:
                roots.append(current_item)
            current_subitem = None
            continue
        subitem_match = SUBITEM_RE.match(line)
        if subitem_match:
            label = subitem_match.group("label")
            letter = chr(ord("A") + KOREAN_ITEM_LABELS.index(label)) if label in KOREAN_ITEM_LABELS else label
            parent = current_item or current_paragraph
            parent_id = parent.rule_id if parent else article_id
            current_subitem = DraftNode(f"{parent_id}_{letter}", f"{label}목")
            current_subitem.append(subitem_match.group("text"))
            if parent:
                parent.children.append(current_subitem)
            else:
                roots.append(current_subitem)
            continue

        target = current_subitem or current_item or current_paragraph
        if target:
            target.append(line)
        else:
            article_lead.append(line)

    return normalize_space(" ".join(article_lead)), [node.as_dict() for node in roots]


def parse_document(path: Path, document_id: str, authority_level: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    metadata = parse_header(raw)
    operative = extract_operative_lines(raw)
    chapter: str | None = None
    section: str | None = None
    article_chunks: list[tuple[str, str | None, str | None, list[str], str | None, str | None]] = []
    current: tuple[str, str | None, str | None, list[str], str | None, str | None] | None = None

    for line in operative:
        if CHAPTER_RE.match(line):
            chapter = normalize_space(line)
            section = None
            continue
        if SECTION_RE.match(line):
            section = normalize_space(line)
            continue
        match = ARTICLE_RE.match(line)
        if match:
            if current:
                article_chunks.append(current)
            current = (
                match.group("article"), match.group("branch"), match.group("heading"),
                [match.group("rest")] if match.group("rest") else [], chapter, section,
            )
        elif current:
            current[3].append(line)
    if current:
        article_chunks.append(current)

    records: list[dict[str, Any]] = []
    for article, branch, heading, body_lines, article_chapter, article_section in article_chunks:
        rid = article_rule_id(document_id, article, branch)
        article_label = f"제{article}조" + (f"의{branch}" if branch else "")
        direct_text, sub_items = parse_sub_items(rid, body_lines)
        title = f"{article_label}({heading})" if heading else article_label
        hierarchy = [x for x in [article_chapter, article_section, normalize_space(title)] if x]
        records.append({
            "rule_id": rid,
            "document_id": document_id,
            "document_name": metadata["document_name"],
            "authority_level": authority_level,
            "article": article_label,
            "heading": heading,
            "hierarchy_path": hierarchy,
            # article과 모든 child는 text를 자기 node의 직접 본문으로만 사용한다.
            # 전체 조문은 render_article_text()로 필요할 때 생성한다.
            "text": direct_text,
            "sub_items": sub_items,
            "source": {
                "format": "txt",
                "source_file": path.name,
                "effective_date": metadata["effective_date"],
                "rule_number": metadata["rule_number"],
                "serial_number": metadata["serial_number"],
                "body_scope": "operative_body_only",
            },
        })

    report = {
        "document_id": document_id,
        "document_name": metadata["document_name"],
        "source_file": path.name,
        "article_count": len(records),
        "effective_date": metadata["effective_date"],
        "rule_number": metadata["rule_number"],
        "operative_line_count": len(operative),
    }
    return records, report


def validate(
    records: list[dict[str, Any]],
    raw_by_document: dict[str, str],
) -> dict[str, Any]:
    all_ids: list[str] = []
    invalid_parent_links: list[str] = []

    def visit(node: dict[str, Any], parent_id: str) -> None:
        rid = node["rule_id"]
        all_ids.append(rid)
        if not rid.startswith(parent_id + "_"):
            invalid_parent_links.append(f"{parent_id}->{rid}")
        for child in node.get("children", []):
            visit(child, rid)

    for record in records:
        all_ids.append(record["rule_id"])
        for child in record.get("sub_items", []):
            visit(child, record["rule_id"])
    duplicates = sorted({rid for rid in all_ids if all_ids.count(rid) > 1})
    empty_articles = [
        r["rule_id"] for r in records if not render_article_text(r).strip()
    ]
    amendment_leaks = [
        r["rule_id"] for r in records
        if "일부개정규칙" in render_article_text(r)
        or "◇ 개정이유" in render_article_text(r)
    ]
    # round-trip 대조: 마커를 복원해 재조립한 전체 조문이 raw TXT의 연속
    # 부분문자열(공백 무시)로 존재해야 한다. 본문 누락·병합·순서와
    # 마커 렌더링 오류를 잡는다. 모든 parent 배정의 의미까지 독립적으로
    # 보증하는 검사는 아니므로 복잡한 계층은 별도 표본 검수한다.
    squash = lambda text: re.sub(
        r"\s+", "", text.translate(str.maketrans(CIRCLED_ALIASES))
    )
    raw_squashed = {
        document_id: squash(raw) for document_id, raw in raw_by_document.items()
    }
    roundtrip_mismatches = [
        r["rule_id"] for r in records
        if squash(render_article_text(r)) not in raw_squashed[r["document_id"]]
    ]
    return {
        "article_count": len(records),
        "rule_unit_count": len(all_ids),
        "duplicate_rule_ids": duplicates,
        "invalid_parent_links": invalid_parent_links,
        "empty_articles": empty_articles,
        "amendment_leaks": amendment_leaks,
        "roundtrip_mismatches": roundtrip_mismatches,
        "ok": not (
            duplicates
            or invalid_parent_links
            or empty_articles
            or amendment_leaks
            or roundtrip_mismatches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "regulations")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "legal_corpus.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "corpus" / "legal_corpus_report.json",
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    config = load_config()
    records: list[dict[str, Any]] = []
    document_reports: list[dict[str, Any]] = []
    raw_by_document: dict[str, str] = {}
    for document_id, document in config["documents"].items():
        path = args.input_dir / document["file"]
        parsed, report = parse_document(path, document_id, document["authority_level"])
        records.extend(parsed)
        document_reports.append(report)
        raw_by_document[document_id] = path.read_text(encoding="utf-8")

    validation = validate(records, raw_by_document)
    if args.validate and not validation["ok"]:
        raise SystemExit(f"법령 corpus 검증 실패: {validation}")
    write_jsonl(args.output, records)
    write_json(args.report, {
        "schema_version": "legal_corpus_v2",
        "documents": document_reports,
        "validation": validation,
    })
    print(f"legal_corpus: {len(records)} articles -> {args.output}")
    print(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
