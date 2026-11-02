"""구조 검증(결정론, API 0콜)과 코드 자동수정 (논문 Algorithm 1의 STRUCTURALVALIDATE·CODEAUTOFIX).

오류 유형:
- dangling_link   : 링크 대상 제목이 Wiki에 없음
- malformed_ref   : fact 또는 페이지 sources의 rule_id가 corpus에 없음
- incomplete_page : summary가 비었거나 fact가 없음
- article_page    : 개체가 아니라 조문 자체를 페이지로 만듦 (개체 단위 원칙 위반)
- empty_emission  : 조문 하나에서 페이지가 하나도 안 나옴 (z.llm_wiki에서 추가한 8번째 유형)

dangling_link는 뒤 조문에서 해소될 수 있으므로 즉시 지우지 않고 --finalize 때만 자동수정한다.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .articles import render_article_markdown
from .schemas import ContentValidateOutput

ARTICLE_TITLE_RE = re.compile(r"제\s*\d+조(?:의\s*\d+)?")


def structural_validate(
    wiki: Any, titles: list[str], unit_ids: set[str]
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for title in titles:
        page = wiki.pages.get(title)
        if page is None:
            continue
        if ARTICLE_TITLE_RE.search(title):
            errors.append({"type": "article_page", "page": title, "detail": title})
        for link in page["links"]:
            if link["title"] not in wiki.pages:
                errors.append({"type": "dangling_link", "page": title, "detail": link["title"]})
        bad_refs = [s for s in page["sources"] if s not in unit_ids]
        for fact in page["facts"]:
            bad_refs += [s for s in fact["sources"] if s not in unit_ids and s not in bad_refs]
        for ref in dict.fromkeys(bad_refs):
            errors.append({"type": "malformed_ref", "page": title, "detail": ref})
        if not page["summary"].strip():
            errors.append({"type": "incomplete_page", "page": title, "detail": "summary 없음"})
        if not page["facts"]:
            errors.append({"type": "incomplete_page", "page": title, "detail": "fact 없음"})
    return errors


def code_autofix(wiki: Any, errors: list[dict[str, str]]) -> int:
    """dangling link와 없는 rule_id 인용을 제거한다. 수정 건수를 돌려준다."""
    fixed = 0
    for error in errors:
        page = wiki.pages.get(error["page"])
        if page is None:
            continue
        if error["type"] == "dangling_link":
            before = len(page["links"])
            page["links"] = [l for l in page["links"] if l["title"] != error["detail"]]
            fixed += before - len(page["links"])
        elif error["type"] == "malformed_ref":
            ref = error["detail"]
            if ref in page["sources"]:
                page["sources"].remove(ref)
                fixed += 1
            for fact in page["facts"]:
                if ref in fact["sources"]:
                    fact["sources"].remove(ref)
                    fixed += 1
    return fixed


# ---- 2단계: LLM content 검증 (논문 CONTENTVALIDATE, 근거는 digest가 아니라 조문 원문) ----
CONTENT_ERROR_TYPES = {"unsupported_fact", "cross_page_contradiction"}


def cited_articles_markdown(
    wiki: Any, titles: list[str], units: dict[str, Any], articles_by_id: dict[str, Any]
) -> str:
    article_ids: list[str] = []
    for title in titles:
        for rid in wiki.pages[title]["sources"]:
            unit = units.get(rid)
            if unit and unit["article_rule_id"] not in article_ids:
                article_ids.append(unit["article_rule_id"])
    return "\n\n".join(render_article_markdown(articles_by_id[a]) for a in article_ids if a in articles_by_id)


def content_validate(
    *,
    client: Any,
    wiki: Any,
    titles: list[str],
    units: dict[str, Any],
    articles_by_id: dict[str, Any],
    template: str,
    system: str,
    source_id: str,
) -> list[dict[str, str]]:
    titles = [t for t in dict.fromkeys(titles) if t in wiki.pages]
    if not titles:
        return []
    pages_json = json.dumps([wiki.page_for_prompt(t) for t in titles], ensure_ascii=False, indent=1)
    prompt = template.replace("{PAGES}", pages_json).replace(
        "{ARTICLES}", cited_articles_markdown(wiki, titles, units, articles_by_id) or "(인용 원문 없음)"
    )
    out = client.call(
        stage="content_validate", source_id=source_id, system=system, user=prompt,
        schema=ContentValidateOutput,
    )
    return [
        {"type": e.type, "page": e.page.strip(), "detail": e.detail.strip()[:500]}
        for e in out.errors
        if e.type in CONTENT_ERROR_TYPES and e.page.strip() in titles
    ]
