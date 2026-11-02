"""LLM-Wiki 저장소.

- 컴파일 상태는 `_manifest.json`(outputs/llmwiki/compile/)에 있고, Wiki 트리의 .md는 여기서 렌더링한다. QA는 .md만 읽는다.
- 페이지 identity는 제목(title)이다. type은 디렉터리 이름이며 첫 값을 유지한다.
- 병합은 누적이다(fact·alias·tag·link·source 합집합). LLM이 fact를 빼먹었는지는
  emissions에 남긴 raw 출력으로 나중에 대조한다.
- 링크는 LLM이 낸 것만 manifest에 둔다. 역방향 링크(Referenced By)는 렌더링 때 계산한다.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..common import normalize_space, write_json
from .articles import article_title, render_article_markdown

_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def normalize_title(title: str) -> str:
    return normalize_space(title)


def normalize_link_title(title: str) -> str:
    """LLM이 index 표기('상품/보험상품')를 그대로 링크에 쓰면 디렉터리 접두를 뗀다."""
    title = normalize_space(title)
    if "/" in title:
        head, tail = title.split("/", 1)
        if head.strip() and tail.strip():
            return tail.strip()
    return title


def safe_filename(title: str) -> str:
    name = _UNSAFE_FILENAME_RE.sub("-", title).strip(" .-")
    return name or "page"


def page_path(page: dict[str, Any]) -> str:
    return f"{safe_filename(page['type'])}/{safe_filename(page['title'])}"


class Wiki:
    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}
        self.emissions: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {}

    # ---- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Wiki":
        wiki = cls()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            wiki.pages = data.get("pages", {})
            wiki.emissions = data.get("emissions", [])
            wiki.meta = data.get("meta", {})
        return wiki

    def save(self, path: Path) -> None:
        write_json(
            path, {"meta": self.meta, "pages": self.pages, "emissions": self.emissions}
        )

    def done_rule_ids(self) -> set[str]:
        return {e["rule_id"] for e in self.emissions}

    # ---- mutation ----------------------------------------------------------
    def upsert(self, page_out: dict[str, Any], rule_id: str) -> tuple[str, str]:
        """LLM이 낸 페이지 하나를 병합한다. ('created'|'updated', title)."""
        title = normalize_link_title(page_out["title"])  # 'type/제목'으로 오면 접두 제거
        incoming_type = normalize_title(page_out.get("type") or "") or "기타"
        existing = self.pages.get(title)
        if existing is None:
            self.pages[title] = {
                "title": title,
                "type": incoming_type,
                "aliases": [],
                "tags": [],
                "summary": "",
                "facts": [],
                "links": [],
                "sources": [],
                "created_by": rule_id,
                "updated_by": [],
                "type_history": [],
            }
            existing = self.pages[title]
            status = "created"
        else:
            status = "updated"
            existing["updated_by"].append(rule_id)
            if incoming_type != existing["type"]:
                existing["type_history"].append({"rule_id": rule_id, "type": incoming_type})

        if page_out.get("summary"):
            existing["summary"] = normalize_space(page_out["summary"])
        for field in ("aliases", "tags"):
            for value in page_out.get(field, []):
                value = normalize_space(value)
                if value and value != title and value not in existing[field]:
                    existing[field].append(value)
        known_facts = {normalize_space(f["text"]): f for f in existing["facts"]}
        for fact in page_out.get("facts", []):
            text = normalize_space(fact.get("text", ""))
            if not text:
                continue
            sources = [normalize_space(s) for s in fact.get("sources", []) if s.strip()]
            if text in known_facts:
                for s in sources:
                    if s not in known_facts[text]["sources"]:
                        known_facts[text]["sources"].append(s)
            else:
                entry = {"text": text, "sources": sources}
                existing["facts"].append(entry)
                known_facts[text] = entry
        known_links = {l["title"]: l for l in existing["links"]}
        for link in page_out.get("links", []):
            target = normalize_link_title(link.get("title", ""))
            if not target or target == title:
                continue
            relation = normalize_space(link.get("relation", ""))
            if target in known_links:
                if relation and not known_links[target]["relation"]:
                    known_links[target]["relation"] = relation
            else:
                entry = {"title": target, "relation": relation}
                existing["links"].append(entry)
                known_links[target] = entry
        for s in page_out.get("sources", []):
            s = normalize_space(s)
            if s and s not in existing["sources"]:
                existing["sources"].append(s)
        for fact in existing["facts"]:
            for s in fact["sources"]:
                if s not in existing["sources"]:
                    existing["sources"].append(s)
        return status, title

    def replace(self, title: str, page_out: dict[str, Any], repair_id: str) -> bool:
        """LLM 수리 결과로 페이지 내용을 통째로 바꾼다 (근거 없는 fact를 지울 수 있어야 하므로).

        title·type·created_by는 유지하고, sources는 기존과 합집합으로 남긴다.
        """
        existing = self.pages.get(title)
        if existing is None:
            return False
        existing["summary"] = normalize_space(page_out.get("summary", "")) or existing["summary"]
        existing["aliases"] = list(
            dict.fromkeys(normalize_space(a) for a in page_out.get("aliases", []) if a.strip() and normalize_space(a) != title)
        )
        existing["tags"] = list(dict.fromkeys(normalize_space(t) for t in page_out.get("tags", []) if t.strip()))
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in page_out.get("facts", []):
            text = normalize_space(fact.get("text", ""))
            if text and text not in seen:
                seen.add(text)
                facts.append({"text": text, "sources": [normalize_space(s) for s in fact.get("sources", []) if s.strip()]})
        existing["facts"] = facts
        links: list[dict[str, str]] = []
        seen_links: set[str] = set()
        for link in page_out.get("links", []):
            target = normalize_link_title(link.get("title", ""))
            if target and target != title and target not in seen_links:
                seen_links.add(target)
                links.append({"title": target, "relation": normalize_space(link.get("relation", ""))})
        existing["links"] = links
        for s in [normalize_space(s) for s in page_out.get("sources", [])] + [s for f in facts for s in f["sources"]]:
            if s and s not in existing["sources"]:
                existing["sources"].append(s)
        existing.setdefault("repaired_by", []).append(repair_id)
        return True

    # ---- prompt views ------------------------------------------------------
    def index_lines(self) -> list[str]:
        lines = []
        for title, p in sorted(self.pages.items(), key=lambda kv: (kv[1]["type"], kv[0])):
            aliases = ", ".join(p["aliases"][:3])
            tags = " ".join("#" + t.replace(" ", "-") for t in p["tags"][:4])
            summary = p["summary"][:80]
            alias_part = f"; 별칭: {aliases}" if aliases else ""
            lines.append(f"- {p['type']}/{title} ({summary}{alias_part}) {tags}".rstrip())
        return lines

    def page_for_prompt(self, title: str) -> dict[str, Any]:
        p = self.pages[title]
        return {
            "title": p["title"],
            "type": p["type"],
            "aliases": p["aliases"],
            "tags": p["tags"],
            "summary": p["summary"],
            "facts": p["facts"],
            "links": p["links"],
            "sources": p["sources"],
        }

    # ---- rendering ---------------------------------------------------------
    def backlinks(self) -> dict[str, list[tuple[str, str]]]:
        out: dict[str, list[tuple[str, str]]] = {}
        for title, p in self.pages.items():
            for link in p["links"]:
                if link["title"] in self.pages:
                    out.setdefault(link["title"], []).append((title, link["relation"]))
        return out

    def render_page(
        self, title: str, articles_by_id: dict[str, dict[str, Any]], units: dict[str, dict[str, Any]]
    ) -> str:
        p = self.pages[title]
        lines = [
            "---",
            f"type: {p['type']}",
            f"aliases: [{', '.join(p['aliases'])}]",
            f"tags: [{', '.join(p['tags'])}]",
            f"created_by: {p['created_by']}",
            "---",
            f"# {p['title']}",
            f"> {p['summary']}",
            "",
            "## Key Facts",
        ]
        if p["facts"]:
            for fact in p["facts"]:
                cite = ", ".join(fact["sources"]) if fact["sources"] else "출처 없음"
                lines.append(f"- {fact['text']} [{cite}]")
        else:
            lines.append("- (없음)")
        lines += ["", "## Related Pages"]
        if p["links"]:
            for link in p["links"]:
                rel = f" -- {link['relation']}" if link["relation"] else ""
                if link["title"] in self.pages:
                    lines.append(f"- [[{page_path(self.pages[link['title']])}]]{rel}")
                else:
                    lines.append(f"- (미생성 페이지) {link['title']}{rel}")
        else:
            lines.append("- (없음)")
        incoming = self.backlinks().get(title, [])
        if incoming:
            lines += ["", "## Referenced By"]
            for src_title, relation in incoming:
                rel = f" -- {relation}" if relation else ""
                lines.append(f"- [[{page_path(self.pages[src_title])}]]{rel}")
        lines += ["", "## Related Sources"]
        seen_articles: list[str] = []
        for rid in p["sources"]:
            unit = units.get(rid)
            article_id = unit["article_rule_id"] if unit else rid
            if article_id in seen_articles:
                continue
            seen_articles.append(article_id)
            article = articles_by_id.get(article_id)
            label = article_title(article) if article else "(corpus에 없는 출처)"
            lines.append(f"- [[sources/articles/{article_id}]] -- {label}")
        if not seen_articles:
            lines.append("- (없음)")
        return "\n".join(lines) + "\n"

    def render_all(
        self,
        root: Path,
        articles: list[dict[str, Any]],
        units: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        articles_by_id = {a["rule_id"]: a for a in articles}
        # 이전 렌더링의 페이지 디렉터리만 지운다 (_index.md가 있는 폴더). sources·runs는 유지.
        for child in root.iterdir():
            if child.is_dir() and (child / "_index.md").is_file():
                shutil.rmtree(child)
        by_type: dict[str, list[str]] = {}
        for title, p in self.pages.items():
            by_type.setdefault(p["type"], []).append(title)
        for type_name, titles in by_type.items():
            directory = root / safe_filename(type_name)
            directory.mkdir(parents=True, exist_ok=True)
            index = [f"# {type_name}", f"> {len(titles)} pages", ""]
            for title in sorted(titles):
                p = self.pages[title]
                (directory / f"{safe_filename(title)}.md").write_text(
                    self.render_page(title, articles_by_id, units), encoding="utf-8"
                )
                aliases = ", ".join(p["aliases"][:3])
                tags = " ".join("#" + t.replace(" ", "-") for t in p["tags"][:4])
                alias_part = f" ({aliases})" if aliases else ""
                index.append(f"- [[{page_path(p)}]]{alias_part} -- {p['summary']} {tags}".rstrip())
            (directory / "_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

        documents = sorted({a["document_name"] for a in articles})
        global_index = [
            "# Wiki Directory Overview",
            f"> {', '.join(documents)} {len(articles)}개 조문을 LLM이 개체 단위 페이지 "
            f"{len(self.pages)}개로 컴파일한 Wiki. 디렉터리의 _index.md에서 페이지 목록을 보고 "
            "페이지를 읽은 뒤, Related Sources로 조문 원문에 내려간다.",
            "",
            "## Directory Catalog",
        ]
        for type_name in sorted(by_type):
            global_index.append(
                f"- {safe_filename(type_name)}/ -- {len(by_type[type_name])} pages "
                f"(목록: `{safe_filename(type_name)}/_index.md`)"
            )
        global_index.append("- sources/articles/ -- 조문 원문 (rule_id 별 1파일)")
        (root / "index.md").write_text("\n".join(global_index) + "\n", encoding="utf-8")

        sources_dir = root / "sources" / "articles"
        sources_dir.mkdir(parents=True, exist_ok=True)
        for article in articles:
            (sources_dir / f"{article['rule_id']}.md").write_text(
                render_article_markdown(article), encoding="utf-8"
            )
        return {"pages": len(self.pages), "types": {k: len(v) for k, v in by_type.items()}}
