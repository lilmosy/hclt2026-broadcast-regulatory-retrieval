"""조문 record → sources/articles/<rule_id>.md 렌더링.

canonical JSONL(legal_corpus.jsonl)의 조문 하나를 heading 계층으로 펼친다.
조 = `#`, 항·호·목은 깊이에 따라 `##`·`###`·`####`. 각 heading에 rule_id를 붙여
LLM이 fact 출처를 항·호 단위로 인용할 수 있게 한다.

LLM이 COMPILE 때 받는 조문 텍스트와 agent가 wiki_read로 읽는 파일이
글자까지 같도록, 이 함수 하나가 둘 다 담당한다.
"""
from __future__ import annotations

import re
from typing import Any

from ..common import normalize_space

RULE_ID_IN_HEADING_RE = re.compile(r"\[([A-Z][A-Z0-9_]+)\]\s*$")


def article_title(record: dict[str, Any]) -> str:
    title = f"{record['document_name']} {record['article']}"
    if record.get("heading"):
        title += f"({record['heading']})"
    return title


def render_article_markdown(record: dict[str, Any]) -> str:
    lines: list[str] = [f"# {article_title(record)} [{record['rule_id']}]"]
    source = record.get("source", {})
    meta = [record["document_id"], " > ".join(record.get("hierarchy_path", []))]
    if source.get("effective_date"):
        meta.append(f"시행 {source['effective_date']}")
    lines.append("> " + " · ".join(part for part in meta if part))
    if record.get("text"):
        lines.append("")
        lines.append(normalize_space(record["text"]))

    def visit(node: dict[str, Any], depth: int) -> None:
        level = min(depth + 1, 6)
        lines.append("")
        lines.append(f"{'#' * level} {node.get('label', '')} [{node['rule_id']}]".rstrip())
        if node.get("text"):
            lines.append(normalize_space(node["text"]))
        for child in node.get("children", []):
            visit(child, depth + 1)

    for child in record.get("sub_items", []):
        visit(child, 1)
    return "\n".join(lines).rstrip() + "\n"


def rule_ids_in_markdown(markdown: str) -> list[str]:
    """heading에 박힌 rule_id를 문서 순서대로 돌려준다 (검증용)."""
    found: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            match = RULE_ID_IN_HEADING_RE.search(line)
            if match:
                found.append(match.group(1))
    return found


def texts_in_record(record: dict[str, Any]) -> list[str]:
    """record의 모든 본문 텍스트를 문서 순서대로 (round-trip 검증용)."""
    out: list[str] = []
    if record.get("text"):
        out.append(normalize_space(record["text"]))

    def visit(node: dict[str, Any]) -> None:
        if node.get("text"):
            out.append(normalize_space(node["text"]))
        for child in node.get("children", []):
            visit(child)

    for child in record.get("sub_items", []):
        visit(child)
    return out
