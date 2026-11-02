"""Markdown Wiki store for query time. Reads only the rendered Wiki tree.

The Wiki is the file tree (index.md, <type>/_index.md, <type>/<title>.md,
sources/articles/<rule_id>.md, error_book.yaml). _manifest.json and the
canonical corpus are compile-time files and are never read here.

wiki_search follows the paper's description (LLM-Wiki §3.2): structured
signals (title, aliases, tags, summary) first, page content as fallback.
The paper does not specify the matcher; plain BM25 per stage is our choice.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

from ...common import file_sha256

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_LIST = re.compile(r"^\[(.*)\]$")
_RULE_HEADING = re.compile(r"^#+ .*\[([A-Z_]+ART[0-9_A-Z]*)\]\s*$", re.M)
_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_ARTICLE = re.compile(r"^([A-Z_]+ART\d+(?:_\d+)?)")   # 제24조의3 = ART24_3; 항·호는 _P/_I


def tokens(text: str) -> list[str]:
    words = re.findall(r"[가-힣]+|[a-z0-9_]+", unicodedata.normalize("NFC", text).lower())
    return words + [word[i:i + 2] for word in words if re.fullmatch(r"[가-힣]+", word)
                    for i in range(len(word) - 1)]


def _front_list(value: str) -> list[str]:
    m = _LIST.match(value.strip())
    if not m:
        return [value.strip()] if value.strip() else []
    return [v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip()]


def parse_page(text: str) -> dict:
    """Frontmatter + title + summary + facts + links from a rendered page."""
    front = {}
    m = _FRONT.match(text)
    body = text
    if m:
        body = text[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                front[key.strip()] = value.strip()
    title = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), "")
    summary = next((l[2:].strip() for l in body.splitlines() if l.startswith("> ")), "")
    facts, in_facts = [], False
    for line in body.splitlines():
        if line.startswith("## "):
            in_facts = line.strip() == "## Key Facts"
        elif in_facts and line.startswith("- "):
            facts.append(line[2:].strip())
    return {"title": title, "type": front.get("type", ""), "aliases": _front_list(front.get("aliases", "")),
            "tags": _front_list(front.get("tags", "")), "summary": summary, "facts": facts,
            "links": [l.split("|")[0].strip() for l in _LINK.findall(body)]}


class WikiStore:
    def __init__(self, root: Path, search_k: int = 10, read_max: int = 3):
        self.root = Path(root)
        if not (self.root / "index.md").is_file():
            raise FileNotFoundError(f"Wiki index.md not found under {self.root}")
        self.search_k, self.read_max = search_k, read_max
        self.files: dict[str, Path] = {}
        self.pages: dict[str, dict] = {}
        self.rule_ids: set[str] = set()
        for path in sorted(self.root.rglob("*.md")):
            rel = unicodedata.normalize("NFC", path.relative_to(self.root).as_posix()[:-3])
            self.files[rel] = path
            if rel.startswith("sources/"):
                self.rule_ids |= set(_RULE_HEADING.findall(path.read_text(encoding="utf-8")))
            elif rel != "index" and not rel.endswith("/_index"):
                self.pages[rel] = parse_page(path.read_text(encoding="utf-8"))
        self.article_of = {rid: _ARTICLE.match(rid).group(1) for rid in self.rule_ids}
        rows = list(self.pages.items())
        self._paths = [p for p, _ in rows]
        self._meta = BM25Okapi([tokens(" ".join([pg["title"], *pg["aliases"], *pg["tags"], pg["summary"]])) or ["_"]
                                for _, pg in rows])
        self._body = BM25Okapi([tokens("\n".join(pg["facts"])) or ["_"] for _, pg in rows])
        digest = hashlib.sha256()
        for rel in sorted(self.files):
            digest.update(rel.encode() + b"\0" + file_sha256(self.files[rel]).encode() + b"\n")
        self.hashes = {"wiki": digest.hexdigest()}

    def pages_citing(self, rule_id: str) -> list[str]:
        """Pages whose Key Facts cite this exact rule ID (evaluation only, never shown to the agent)."""
        pattern = re.compile(r"\[[^\]]*" + re.escape(rule_id) + r"(?![_0-9])[^\]]*\]")
        return [p for p, pg in self.pages.items() if any(pattern.search(f) for f in pg["facts"])]

    # ---- tools ------------------------------------------------------------
    def search(self, query: str, k: int | None = None):
        k = k or self.search_k
        ts = tokens(query)
        if not ts:
            return {"results": []}
        picked: list[str] = []
        for index in (self._meta, self._body):          # structured signals first, content as fallback
            scores = index.get_scores(ts)
            ranked = sorted(((s, p) for s, p in zip(scores, self._paths) if s > 0), key=lambda x: (-x[0], x[1]))
            for _, p in ranked:
                if p not in picked:
                    picked.append(p)
                if len(picked) >= k:
                    break
            if len(picked) >= k:
                break
        return {"results": [{"path": p, "title": self.pages[p]["title"], "summary": self.pages[p]["summary"]}
                            for p in picked]}

    def resolve(self, raw: str) -> str | None:
        path = unicodedata.normalize("NFC", raw.strip().strip("[]"))
        if path.endswith(".md"):
            path = path[:-3]
        path = path.strip("/")
        return path if path in self.files else None

    def read(self, paths: list[str]):
        if not 1 <= len(paths) <= self.read_max:
            raise ValueError(f"read requires 1..{self.read_max} paths")
        results = []
        for raw in paths:
            rel = self.resolve(raw)
            if rel is None:
                results.append({"path": raw, "error": "unknown_path; use the exact path from index or search results"})
                continue
            kind = "index" if rel == "index" or rel.endswith("/_index") else ("source" if rel.startswith("sources/") else "page")
            content = self.files[rel].read_text(encoding="utf-8")
            item = {"path": rel, "kind": kind, "content": content}
            if kind == "index":   # pages listed here are candidates the agent saw (used only for trace)
                item["entries"] = [l for l in _LINK.findall(content) if l in self.files]
            results.append(item)
        return {"results": results}
