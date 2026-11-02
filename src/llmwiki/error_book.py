"""Error Book (논문 §3.3): 발견 → 원인 귀속 → constraint 생성 → 다음 COMPILE 프롬프트에 주입 → 검증·종결.

`error_book.yaml`에 유형별 entry(원인, constraint, 열림/닫힘, 횟수)와 발생 로그를 남긴다.
constraint는 자연어 한 문장이며 열려 있는 것만 프롬프트에 붙는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# type -> (root cause, injected constraint, layer)
TAXONOMY: dict[str, tuple[str, str, str]] = {
    "dangling_link": (
        "Wiki에 없는 페이지로 링크를 걸었다.",
        "links의 대상은 '현재 페이지 목록'에 있거나 이번 출력에서 함께 만드는 페이지여야 한다. "
        "없는 페이지로 링크하지 마라.",
        "structural",
    ),
    "malformed_ref": (
        "조문에 없는 rule_id를 출처로 적었다.",
        "sources에는 조문 텍스트의 [ ] 안에 있는 식별자만 쓴다. 식별자를 만들어내거나 변형하지 마라.",
        "structural",
    ),
    "incomplete_page": (
        "summary나 fact가 없는 페이지를 냈다.",
        "모든 페이지는 한 줄 summary와 최소 하나의 fact를 가져야 한다.",
        "structural",
    ),
    "article_page": (
        "개체가 아니라 조문 자체를 페이지로 만들었다.",
        "'제N조' 같은 조문 단위 페이지를 만들지 마라. 조문이 규율하는 개체(대상·상품·행위·주체·개념) "
        "페이지에 fact로 나눠 담고, 조문은 sources로만 가리킨다.",
        "structural",
    ),
    "empty_emission": (
        "조문에서 페이지가 하나도 나오지 않았다.",
        "모든 조문에서 최소 하나의 페이지를 만들거나 갱신하라. 규율 대상이 불분명하면 "
        "조문이 정의하거나 언급하는 개념 페이지를 만든다.",
        "structural",
    ),
    # 2단계(LLM content 검증)에서 쓸 유형. 1단계에서는 발생하지 않는다.
    "unsupported_fact": (
        "인용한 조문에 근거가 없는 fact를 적었다.",
        "fact는 인용한 rule_id의 조문 텍스트로 뒷받침되는 내용만 쓴다.",
        "content",
    ),
    "cross_page_contradiction": (
        "관련 페이지끼리 같은 속성에 다른 값을 적었다.",
        "기존 페이지를 갱신할 때 정의·수치·시간대가 충돌하면 조문 출처를 밝혀 구분하고, "
        "근거 없이 두 값을 나란히 두지 마라.",
        "content",
    ),
}


class ErrorBook:
    def __init__(self, path: Path, silent_types: frozenset[str] = frozenset()) -> None:
        """silent_types: 오류로 세기는 하되 constraint를 프롬프트에 주입하지 않는 유형.

        '그 지시가 없을 때의 경향'을 보려는 경우, 검출 자체가
        constraint로 되돌아오면 변수가 오염되므로 관찰만 한다.
        """
        self.path = path
        self.silent_types = silent_types
        self.entries: dict[str, dict[str, Any]] = {}
        self.log: list[dict[str, Any]] = []
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self.entries = {e["type"]: e for e in data.get("entries", [])}
            self.log = data.get("occurrences", [])

    def discover(self, errors: list[dict[str, str]], rule_id: str) -> None:
        for error in errors:
            self.log.append({"rule_id": rule_id, **error})
            kind = error["type"]
            if kind not in self.entries:
                cause, constraint, layer = TAXONOMY.get(
                    kind, ("unknown", "이 오류 패턴을 피하라.", "structural")
                )
                self.entries[kind] = {
                    "type": kind,
                    "root_cause": cause,
                    "constraint": constraint,
                    "layer": layer,
                    "status": "open",
                    "count": 0,
                    "first_rule_id": rule_id,
                    "last_rule_id": rule_id,
                }
            self.entries[kind]["status"] = "open"
            self.entries[kind]["count"] += 1
            self.entries[kind]["last_rule_id"] = rule_id

    def active_constraints(self) -> list[str]:
        return [
            e["constraint"]
            for e in self.entries.values()
            if e["status"] == "open" and e["type"] not in self.silent_types
        ]

    def verify_and_close(
        self, remaining_errors: list[dict[str, str]], *, layers: set[str] | None = None
    ) -> None:
        remaining = {e["type"] for e in remaining_errors}
        for entry in self.entries.values():
            if layers is not None and entry.get("layer") not in layers:
                continue
            entry["status"] = "open" if entry["type"] in remaining else "closed"

    def counts(self) -> dict[str, int]:
        return {k: v["count"] for k, v in self.entries.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                {"entries": list(self.entries.values()), "occurrences": self.log},
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
