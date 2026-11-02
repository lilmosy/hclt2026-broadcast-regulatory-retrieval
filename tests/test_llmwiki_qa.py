"""Offline regression tests: Markdown-only store, agent loop, budgets, metrics."""
import json
import tempfile
import unittest
from pathlib import Path

from src.llmwiki.qa.agent import run_agent, validate_answer
from src.llmwiki.qa.metrics import score_prediction, page_reach, aggregate
from src.llmwiki.qa.schemas import Answer
from src.llmwiki.qa.store import WikiStore, parse_page
from src.llmwiki.qa.provider import response_input

PAGE = """---
type: 개념
aliases: [검증 절차]
tags: [테스트]
created_by: TEST_ART1
---
# 검증
> 검증의 개념이다.

## Key Facts
- 검증을 위한 첫 번째 사실이다. [TEST_ART1_P1]
- 별도 조건을 적용한다. [TEST_ART1_P2]

## Related Pages
- [[개념/이웃]] -- 관련 개념

## Related Sources
- [[sources/articles/TEST_ART1]] -- 테스트규정 제1조(검증)
"""
NEIGHBOR = "---\ntype: 개념\naliases: []\ntags: []\n---\n# 이웃\n> 링크로만 도달하는 특수 개념이다.\n\n## Key Facts\n- 이웃 사실. [TEST_ART1_P2]\n"
SOURCE = "# 테스트규정 제1조(검증) [TEST_ART1]\n\n## 제1항 [TEST_ART1_P1]\n첫 번째 원문.\n\n## 제2항 [TEST_ART1_P2]\n두 번째 원문.\n"
SOURCE2 = "# 테스트규정 제1조의2 [TEST_ART1_2]\n\n## 제1호 [TEST_ART1_2_I1]\n조의 바로 아래 호.\n"
INDEX = "# Wiki Directory Overview\n\n## Directory Catalog\n- 개념/ -- 2 pages (목록: `개념/_index.md`)\n"
DIR_INDEX = "# 개념\n> 2 pages\n\n- [[개념/검증]] -- 검증의 개념이다.\n- [[개념/이웃]] -- 특수 개념이다.\n"


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "개념").mkdir()
        (self.root / "sources/articles").mkdir(parents=True)
        (self.root / "index.md").write_text(INDEX)
        (self.root / "개념/_index.md").write_text(DIR_INDEX)
        (self.root / "개념/검증.md").write_text(PAGE)
        (self.root / "개념/이웃.md").write_text(NEIGHBOR)
        (self.root / "개념/기타.md").write_text("---\ntype: 개념\naliases: []\ntags: []\n---\n# 기타\n> 관계없는 페이지.\n\n## Key Facts\n- 무관한 사실. [TEST_ART1]\n")
        (self.root / "sources/articles/TEST_ART1.md").write_text(SOURCE)
        (self.root / "sources/articles/TEST_ART1_2.md").write_text(SOURCE2)
        (self.root / "_manifest.json").write_text(json.dumps({"emissions": [{"secret": "NEVER_SEARCH_THIS"}]}))
        self.store = WikiStore(self.root, search_k=10)

    def tearDown(self):
        self.tmp.cleanup()

    def answer(self, rid="TEST_ART1_P1"):
        return Answer.model_validate({"status": "answered", "review_required": True,
            "applicable_regulations": [{"rule_id": rid, "rationale": "입력 사실과 요건을 대조했다."}],
            "overall_answer": "검토가 필요하다.", "missing_information": []})

    def test_store_reads_markdown_only(self):
        self.assertEqual(self.store.rule_ids, {"TEST_ART1", "TEST_ART1_P1", "TEST_ART1_P2", "TEST_ART1_2", "TEST_ART1_2_I1"})
        self.assertEqual(self.store.article_of["TEST_ART1_P2"], "TEST_ART1")
        self.assertEqual(self.store.article_of["TEST_ART1_2_I1"], "TEST_ART1_2")   # 제1조의2 ≠ 제1조
        page = parse_page(PAGE)
        self.assertEqual(page["aliases"], ["검증 절차"])
        self.assertEqual(len(page["facts"]), 2)
        self.assertEqual(self.store.search("NEVER_SEARCH_THIS")["results"], [])
        out = self.store.read(["../_manifest.json", "index.md", "개념/_index"])
        self.assertIn("error", out["results"][0])
        self.assertEqual(out["results"][1]["kind"], "index")
        self.assertEqual(out["results"][2]["entries"], ["개념/검증", "개념/이웃"])
        self.assertEqual(self.store.read(["sources/articles/TEST_ART1.md"])["results"][0]["kind"], "source")
        self.assertEqual(self.store.read(["개념/검증"])["results"][0]["content"], PAGE)

    def test_search_structured_first_then_content(self):
        # "검증 절차" is an alias (structured) -> 검증 first; "특수" appears only in 이웃's summary
        self.assertEqual(self.store.search("검증 절차")["results"][0]["path"], "개념/검증")
        self.assertEqual(self.store.search("특수")["results"][0]["path"], "개념/이웃")
        # "별도 조건" appears only in a fact of 검증 (content fallback)
        self.assertEqual([r["path"] for r in self.store.search("별도 조건")["results"]], ["개념/검증"])

    def test_validate_answer_minimal_rules(self):
        self.assertTrue(validate_answer(self.answer(), self.store, set()))
        self.assertFalse(validate_answer(self.answer(), self.store, {"개념/검증"}))
        self.assertTrue(validate_answer(self.answer("TEST_ART9"), self.store, {"개념/검증"}))

    def test_responses_retains_reasoning_and_tool_call_ids(self):
        output = [{"type": "reasoning", "id": "r1", "encrypted_content": "opaque"},
                  {"type": "function_call", "call_id": "call1", "name": "wiki_read", "arguments": "{}"}]
        result = response_input([{"role": "user", "content": "q"},
                                 {"role": "assistant", "_response_items": output},
                                 {"role": "tool", "tool_call_id": "call1", "content": "result"}])
        self.assertEqual(result[1:3], output)
        self.assertEqual(result[-1], {"type": "function_call_output", "call_id": "call1", "output": "result"})

    def test_metrics_multi_gold_and_failure_denominator(self):
        amap = {"a": "A", "b": "B", "c": "C"}
        score = score_prediction(["c", "a"], ["a", "b"], amap)
        self.assertEqual(score["hit_at_1"], 0)
        self.assertEqual(score["hit_at_5"], 1)
        self.assertEqual(score["set_recall"], .5)
        failure = score_prediction([], ["a"], amap)
        reach = page_reach(["p1"], {"a": ["p1"], "b": ["p2"]})
        self.assertEqual(reach, {"gold_page_any": 1, "gold_page_all": 0})
        rows = [{"metrics": m, "reach": reach, "domain": "test", "termination": "answered", "tool_calls": 1,
                 "elapsed_seconds": 1, "searches": 1, "route": "search_first", "followed_links": [],
                 "read_sources": []} for m in [score, failure]]
        agg = aggregate(rows, 2)
        self.assertEqual(agg["all"]["hit_at_5"], .5)
        self.assertEqual(agg["all"]["gold_page_any"], 1)
        self.assertEqual(self.store.pages_citing("TEST_ART1_P2"), ["개념/검증", "개념/이웃"])

    def scripted_client(self, steps):
        class FakeClient:
            def __init__(self): self.i = 0; self.forced = []
            def complete(inner, messages, tools, case_id, step, force_tool=None):
                inner.forced.append(force_tool)
                name, args = steps[inner.i]; inner.i += 1
                return {"role": "assistant", "tool_calls": [{"id": str(inner.i), "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
        return FakeClient()

    def test_search_first_trace_and_resume(self):
        client = self.scripted_client([("wiki_search", {"query": "검증"}),
                                       ("wiki_read", {"paths": ["개념/검증"]}),
                                       ("wiki_read", {"paths": ["개념/이웃"]}),
                                       ("submit_answer", self.answer().model_dump())])
        kw = dict(store=self.store, client=client, prompt="p", question="q", artifact={"text": "x"},
                  case_id="q1", checkpoint=self.root / "s.json")
        result = run_agent(**kw)
        self.assertTrue(result["done"])
        self.assertEqual(result["route"], "search_first")
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["followed_links"], ["개념/이웃"])   # 이웃 was never a search candidate
        self.assertEqual(result["read_sources"], [])
        self.assertEqual(run_agent(**kw), result)
        self.assertEqual(client.forced[0], None)

    def test_browse_first_route_and_forced_first_tool(self):
        client = self.scripted_client([("wiki_read", {"paths": ["index.md"]}),
                                       ("wiki_read", {"paths": ["개념/_index"]}),
                                       ("wiki_read", {"paths": ["개념/이웃"]}),
                                       ("submit_answer", self.answer("TEST_ART1_P2").model_dump())])
        result = run_agent(store=self.store, client=client, prompt="p", question="q", artifact={"text": "x"},
                           case_id="q2", checkpoint=self.root / "b.json", route="browse")
        self.assertEqual(client.forced[0], "wiki_read")
        self.assertEqual(result["route"], "browse_first")
        self.assertEqual(result["read_indices"], ["index", "개념/_index"])
        self.assertEqual(result["followed_links"], [])   # listed in the index, so not a link hop

    def test_budget_and_invalid_answer(self):
        class FakeClient:
            def complete(inner, messages, tools, case_id, step, force_tool=None):
                name = "submit_answer" if force_tool == "submit_answer" else "wiki_search"
                args = self.answer().model_dump() if force_tool == "submit_answer" else {"query": "검증"}
                return {"role": "assistant", "tool_calls": [{"id": str(step), "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}]}
        result = run_agent(store=self.store, client=FakeClient(), prompt="p", question="q", artifact={"text": "x"},
                           case_id="q3", checkpoint=self.root / "l.json", max_tools=2, max_revisions=1)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["termination"], "invalid_answer")   # never read a page
        self.assertIsNone(result["answer"])


if __name__ == "__main__":
    unittest.main()
