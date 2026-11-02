from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "corpus.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def render_article_text(record: dict[str, Any]) -> str:
    """구조화된 article record를 사람이 읽는 전체 조문으로 렌더링한다.

    canonical JSONL은 각 node의 직접 본문만 저장한다. 전체 조문 문자열은
    중복 저장하지 않고 필요할 때 이 함수로 결정론적으로 만든다.
    canonical JSONL에는 같은 법문을 평탄화해 중복 저장하지 않는다.
    """
    title = record["article"]
    if record.get("heading"):
        title += f"({record['heading']})"
    parts = [title]
    if record.get("text"):
        parts.append(record["text"])

    circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

    def source_marker(label: str) -> str:
        paragraph = re.fullmatch(r"제(\d+)항", label)
        if paragraph:
            number = int(paragraph.group(1))
            return circled[number - 1] if 1 <= number <= len(circled) else label
        item = re.fullmatch(r"제(\d+)호", label)
        if item:
            return f"{int(item.group(1))}."
        subitem = re.fullmatch(r"([가-하])목", label)
        if subitem:
            return f"{subitem.group(1)}."
        return label

    def visit(node: dict[str, Any]) -> None:
        label = source_marker(node.get("label", ""))
        text = node.get("text", "")
        if label or text:
            parts.append(normalize_space(f"{label} {text}"))
        for child in node.get("children", []):
            visit(child)

    for child in record.get("sub_items", []):
        visit(child)
    return normalize_space(" ".join(parts))


def compact_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_rule_units(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}

    def visit(
        node: dict[str, Any],
        document_id: str,
        article_id: str,
        parent_rule_id: str,
        label_path: list[str],
    ) -> None:
        rid = node["rule_id"]
        current_path = [*label_path, node.get("label", rid)]
        units[rid] = {
            **node,
            "document_id": document_id,
            "article_rule_id": article_id,
            "parent_rule_id": parent_rule_id,
            "label_path": current_path,
        }
        for child in node.get("children", []):
            visit(child, document_id, article_id, rid, current_path)

    for record in records:
        units[record["rule_id"]] = {
            **record,
            "text": record.get("text", ""),
            "full_text": render_article_text(record),
            "article_rule_id": record["rule_id"],
            "parent_rule_id": None,
            "label_path": [record["article"]],
        }
        for child in record.get("sub_items", []):
            visit(
                child,
                record["document_id"],
                record["rule_id"],
                record["rule_id"],
                [record["article"]],
            )
    return units


def rule_context_text(
    rule_id: str,
    units: dict[str, dict[str, Any]],
) -> str:
    """Gold rule의 조문 공통문과 target까지의 계층 문맥을 조립한다."""
    if rule_id not in units:
        raise KeyError(f"corpus에 없는 rule_id: {rule_id}")
    target = units[rule_id]
    article = units[target["article_rule_id"]]
    citation = f"{article['document_name']} {article['article']}"
    if article.get("heading"):
        citation += f"({article['heading']})"
    if rule_id == article["rule_id"]:
        return f"{citation}\n{article['full_text']}"

    path: list[dict[str, Any]] = []
    cursor = target
    while cursor.get("parent_rule_id"):
        path.append(cursor)
        cursor = units[cursor["parent_rule_id"]]
    path.reverse()

    lines = [citation]
    if article.get("text"):
        lines.append(article["text"])
    for node in path:
        lines.append(normalize_space(f"{node.get('label', '')} {node.get('text', '')}"))
    return "\n".join(line for line in lines if line)
