from __future__ import annotations

from collections import Counter
from statistics import mean


def score_prediction(predicted: list[str], gold: list[str], article_map: dict[str, str]):
    """Exact rule-unit scoring of the ordered prediction list against the gold set."""
    pred = list(dict.fromkeys(predicted))
    truth = set(gold)
    if not truth:
        raise ValueError("evidence benchmark item has no gold rules")
    hits = len(set(pred) & truth)
    recall = hits / len(truth)
    precision = hits / len(pred) if pred else 0.
    pred_articles = list(dict.fromkeys(article_map[r] for r in pred if r in article_map))
    gold_articles = {article_map[r] for r in truth}
    return {"hit_at_1": int(bool(pred) and pred[0] in truth),
            "hit_at_5": int(bool(set(pred[:5]) & truth)),
            "hit_at_10": int(bool(set(pred[:10]) & truth)),
            "set_recall": recall, "set_precision": precision,
            "set_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "article_hit_at_1": int(bool(pred_articles) and pred_articles[0] in gold_articles),
            "gold_count": len(truth), "prediction_count": len(pred)}


def page_reach(read_pages: list[str], gold_pages: dict[str, list[str]]):
    """Whether the agent read a page carrying each gold rule (any / all)."""
    read = set(read_pages)
    reached = [bool(set(pages) & read) for pages in gold_pages.values()]
    return {"gold_page_any": int(any(reached)), "gold_page_all": int(all(reached))}


def failure_type(predicted: list[str], gold: list[str], reach: dict, article_map: dict[str, str]) -> str:
    """One label per case: where along search → read → choose the case was lost."""
    pred, truth = list(dict.fromkeys(predicted)), set(gold)
    if pred and pred[0] in truth:
        return "정답"
    if not reach.get("gold_page_any"):
        return "미도달"
    if not pred:
        return "도달·답 없음"
    if article_map.get(pred[0]) in {article_map.get(g) for g in truth}:
        return "도달·같은 조 다른 항호"
    if truth & set(pred):
        return "도달·다른 조 1순위"
    return "도달·gold 목록에 없음"


METRICS = ("hit_at_1", "hit_at_5", "hit_at_10", "set_recall", "set_precision", "set_f1", "article_hit_at_1")


def aggregate(rows, expected_total, article_map: dict[str, str] | None = None):
    def group(items):
        out = {"n": len(items)}
        for k in METRICS:
            out[k] = mean(r["metrics"][k] for r in items) if items else None
        for k in ("gold_page_any", "gold_page_all"):
            out[k] = mean(r["reach"][k] for r in items) if items and "reach" in items[0] else None
        out["mean_searches"] = mean(r["searches"] for r in items) if items else None
        return out
    result = {"completed": len(rows), "expected_total": expected_total,
              "all": group(rows), "single_gold": group([r for r in rows if r["metrics"]["gold_count"] == 1]),
              "multi_gold": group([r for r in rows if r["metrics"]["gold_count"] > 1]),
              "terminations": dict(Counter(r["termination"] for r in rows)),
              "routes": dict(Counter(r.get("route") for r in rows)),
              "failure_types": dict(Counter(failure_type(r["predicted_rule_ids"], r["gold_rule_ids"], r.get("reach", {}), article_map)
                                           for r in rows)) if article_map is not None and rows else {},
              "cases_researching": sum(1 for r in rows if r["searches"] > 1),
              "cases_following_links": sum(1 for r in rows if r.get("followed_links")),
              "cases_reading_sources": sum(1 for r in rows if r.get("read_sources")),
              "by_domain": {d: group([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows})},
              "mean_tool_calls": mean(r["tool_calls"] for r in rows) if rows else None,
              "mean_latency_seconds": mean(r["elapsed_seconds"] for r in rows) if rows else None}
    # Failed/no-answer cases are included with empty predictions, never discarded.
    return result


NOTES_MARK = "<!-- notes: 아래는 사람이 쓴 해석. evaluate가 다시 써도 이 줄 아래는 보존된다 -->"

DEFINITIONS = [
    ("hit@1", "예측 목록의 1순위가 gold 규정(항·호 단위 정확 일치) 중 하나인 비율. 대표 지표",
     "gold {A, B}, 예측 [C, A, D, B] → 1순위 C는 gold 아님 → 0. 예측 [A, C] → 1"),
    ("hit@5 / hit@10", "gold 규정이 예측 목록 5위 / 10위 안에 있는 비율. 목록은 최대 10개",
     "gold {A, B}, 예측 [C, A, D, B] → A가 2위에 있음 → hit@5 = 1"),
    ("조 단위 hit@1", "항·호를 조로 올려서 본 hit@1. hit@1과의 차이 = 조는 맞고 항·호를 틀린 비율",
     "gold 25조 2항 4호, 1순위 25조 6항 → hit@1 = 0이지만 조 단위 hit@1 = 1"),
    ("set_recall", "gold 규정 중 예측 목록 어딘가에 들어온 비율(순위 무관). 빠짐없이 찾았나",
     "gold {A, B}, 예측 [C, A, D] → A만 찾음 → 1/2 = 0.5"),
    ("set_precision", "예측 목록 중 gold인 비율. 군더더기가 적은가",
     "gold {A, B}, 예측 [C, A, D] → 3개 중 gold 1개 → 1/3 = 0.33"),
    ("set_f1", "set_recall과 set_precision의 조화평균",
     "recall 0.5, precision 0.33 → 2·0.5·0.33/(0.5+0.33) = 0.40"),
    ("정답 페이지 도달(1개 이상)", "gold 규정 ID가 Key Facts에 적힌 Wiki 페이지를 하나라도 읽은 비율. 탐색이 됐는가",
     "gold A는 페이지 P에, B는 페이지 Q에 있음. 에이전트가 P만 읽음 → 1"),
    ("정답 페이지 전부 도달", "gold 규정마다 그 페이지를 읽은 비율. 복수 gold에서 두 번째 쟁점까지 찾았는가",
     "위와 같은 상황(P만 읽음) → 0. P와 Q를 다 읽어야 1"),
    ("평균 검색", "사례당 wiki_search 호출 수",
     "검색 → 읽기 → 검색 → 읽기 → 답 이면 2"),
    ("단일 / 복수 gold", "gold 규정이 1개인 사례 / 2개 이상인 사례",
     "에어베드(5조 3항 하나) = 단일, 여론조사(16조 1항과 6항) = 복수"),
]

FAILURE_ORDER = ["정답", "미도달", "도달·다른 조 1순위", "도달·같은 조 다른 항호", "도달·gold 목록에 없음", "도달·답 없음"]
FAILURE_DESC = {"정답": "1순위가 gold",
                "미도달": "정답 페이지를 한 번도 읽지 않음 (검색·탐색 실패)",
                "도달·다른 조 1순위": "정답 페이지를 읽었지만 다른 조문을 1순위로 냄. gold는 목록 뒤에 있음",
                "도달·같은 조 다른 항호": "정답 페이지를 읽고 같은 조까지 갔지만 다른 항·호를 1순위로 냄",
                "도달·gold 목록에 없음": "정답 페이지를 읽었지만 gold가 목록에 아예 없음",
                "도달·답 없음": "정답 페이지를 읽었지만 답을 내지 못함"}


def render_markdown(agg: dict, run_info: dict, previous: str = "") -> str:
    """Single metrics.md per run. Auto part is rewritten; text after NOTES_MARK in `previous` is kept."""
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    num = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    cols = [("hit@1", "hit_at_1"), ("hit@5", "hit_at_5"), ("hit@10", "hit_at_10"),
            ("조 단위 hit@1", "article_hit_at_1"), ("set_recall", "set_recall"),
            ("set_precision", "set_precision"), ("set_f1", "set_f1"),
            ("정답 페이지 도달(1개 이상)", "gold_page_any"), ("정답 페이지 전부 도달", "gold_page_all")]
    lines = [f"# {run_info.get('run', 'run')} 성능 지표", "",
             f"- 완료 {agg['completed']} / {agg['expected_total']}건 · 모델 {run_info.get('model')} · reasoning {run_info.get('reasoning_effort')} · 검색 k={run_info.get('search_k')} · route={run_info.get('route')}",
             "- gold = 방심위 의결문이 실제 인용한 규정 ID(항·호 단위). 정답 페이지 = 그 ID가 Key Facts에 적힌 Wiki 페이지.", "",
             "## 지표 뜻", ""]
    lines += ["| 지표 | 뜻 | 예 |", "|---|---|---|"] + [f"| **{k}** | {v} | {ex} |" for k, v, ex in DEFINITIONS]
    lines += ["", "## 결과", "",
              "| 구분 | n | " + " | ".join(c for c, _ in cols) + " | 평균 검색 |",
              "|---|---|" + "---|" * (len(cols) + 1)]
    for label, key in (("전체", "all"), ("단일 gold", "single_gold"), ("복수 gold", "multi_gold")):
        g = agg[key]
        lines.append(f"| {label} | {g['n']} | " + " | ".join(pct(g[k]) for _, k in cols) + f" | {num(g['mean_searches'])} |")
    lines += ["", "## 도메인별", "", "| 도메인 | n | hit@1 | 조 단위 hit@1 | set_recall | 정답 페이지 도달 |", "|---|---|---|---|---|---|"]
    for d, g in agg["by_domain"].items():
        lines.append(f"| {d} | {g['n']} | {pct(g['hit_at_1'])} | {pct(g['article_hit_at_1'])} | {pct(g['set_recall'])} | {pct(g['gold_page_any'])} |")
    n = max(agg["completed"], 1)
    if agg.get("failure_types"):
        lines += ["", "## 실패 유형 (사례당 하나)", "", "| 유형 | 건수 | 비율 | 뜻 |", "|---|---|---|---|"]
        for k in FAILURE_ORDER:
            if k in agg["failure_types"]:
                v = agg["failure_types"][k]
                lines.append(f"| {k} | {v} | {pct(v / n)} | {FAILURE_DESC[k]} |")
    lines += ["", "## 탐색 통계", "",
              f"- 평균 도구 호출 {num(agg['mean_tool_calls'])}회, 평균 소요 {num(agg['mean_latency_seconds'], 1)}초",
              f"- 경로: " + ", ".join(f"{k} {v}" for k, v in agg["routes"].items()),
              f"- 재검색(검색 2회 이상) {agg['cases_researching']}건 ({pct(agg['cases_researching'] / n)}) · 링크 직접 읽기 {agg['cases_following_links']}건 · 원문 읽기 {agg['cases_reading_sources']}건",
              f"- 종료: " + ", ".join(f"{k} {v}" for k, v in agg["terminations"].items()), ""]
    notes = previous.split(NOTES_MARK, 1)[1] if NOTES_MARK in previous else ""
    return "\n".join(lines) + "\n" + NOTES_MARK + notes
