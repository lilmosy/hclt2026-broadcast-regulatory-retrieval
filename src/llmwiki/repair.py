"""주기적 LLM 수리 (논문 Layer 2, LLMPERIODICFIX).

content 검증에서 걸린 페이지만 대상으로, 인용 조문 원문을 근거로 다시 쓰게 하고 결과를
`wiki.replace`로 통째 반영한다. 새 페이지 생성은 허용하지 않는다.
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import RepairOutput
from .validators import cited_articles_markdown

REPAIRABLE_TYPES = {"unsupported_fact", "cross_page_contradiction", "incomplete_page"}


def _llm_repair_batch(
    *,
    client: Any,
    wiki: Any,
    errors: list[dict[str, str]],
    units: dict[str, Any],
    articles_by_id: dict[str, Any],
    template: str,
    system: str,
    source_id: str,
) -> dict[str, Any]:
    targets = [
        e for e in errors if e["type"] in REPAIRABLE_TYPES and e.get("page") in wiki.pages
    ]
    titles = list(dict.fromkeys(e["page"] for e in targets))
    if not titles:
        return {"targets": [], "repaired": [], "ignored": []}
    prompt = (
        template.replace("{ERRORS}", json.dumps(targets, ensure_ascii=False, indent=1))
        .replace("{PAGES}", json.dumps([wiki.page_for_prompt(t) for t in titles], ensure_ascii=False, indent=1))
        .replace("{ARTICLES}", cited_articles_markdown(wiki, titles, units, articles_by_id) or "(인용 원문 없음)")
        .replace("{INDEX}", "\n".join(wiki.index_lines()))
    )
    out = client.call(
        stage="repair", source_id=source_id, system=system, user=prompt, schema=RepairOutput
    )
    repaired, ignored = [], []
    for page in out.pages:
        data = page.model_dump()
        title = data["title"].strip()
        if title in titles and wiki.replace(title, data, source_id):
            repaired.append(title)
        else:
            ignored.append(title)
    return {"targets": titles, "repaired": repaired, "ignored": ignored, "raw_pages": [p.model_dump() for p in out.pages]}


def llm_repair(*, errors, batch_size=4, **kwargs):
    """Bound output size; omissions remain explicit in targets - repaired."""
    titles = list(dict.fromkeys(e.get("page") for e in errors
                              if e["type"] in REPAIRABLE_TYPES and e.get("page") in kwargs["wiki"].pages))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    combined = {"targets": [], "repaired": [], "ignored": [], "raw_pages": []}
    for offset in range(0, len(titles), batch_size):
        subset = set(titles[offset:offset + batch_size])
        result = _llm_repair_batch(errors=[e for e in errors if e.get("page") in subset], **kwargs)
        for key in combined:
            combined[key].extend(result.get(key, []))
    return combined
