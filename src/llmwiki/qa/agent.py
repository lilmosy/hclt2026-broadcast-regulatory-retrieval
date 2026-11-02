"""Query-time loop as in LLM-Wiki §3.2: search or browse, read, follow links or
re-search, judge sufficiency, answer. Budget Tmax, patience P, at least one page read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import openai

from ...common import write_json
from .schemas import Answer, SearchArgs, ReadArgs

FIRST_TOOL = {"search": "wiki_search", "browse": "wiki_read"}


def validate_answer(answer: Answer, store, read_pages: set[str]):
    errors = []
    if not read_pages:
        errors.append("Wiki 페이지를 최소 하나 읽어야 한다")
    if answer.status == "answered" and not answer.applicable_regulations:
        errors.append("answered needs at least one regulation")
    if answer.status == "insufficient_evidence" and answer.review_required is False:
        errors.append("insufficient evidence must not imply no_flag")
    ids = [r.rule_id for r in answer.applicable_regulations]
    if len(ids) != len(set(ids)):
        errors.append("duplicate rule IDs")
    for reg in answer.applicable_regulations:
        if reg.rule_id not in store.rule_ids:
            errors.append(f"unknown rule_id: {reg.rule_id}")
        if not reg.rationale.strip():
            errors.append(f"missing rationale: {reg.rule_id}")
    return errors


def run_agent(*, store, client, prompt: str, question: str, artifact: dict,
              case_id: str, checkpoint: Path | None = None, max_tools=15, empty_patience=3,
              max_model_calls=20, max_revisions=2, route="auto"):
    if checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["done"]:
            return state
    else:
        state = {"case_id": case_id, "done": False, "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"question": question, "artifact": artifact}, ensure_ascii=False)}],
            "trace": [], "route": None, "read_indices": [], "read_pages": [], "read_sources": [],
            "followed_links": [], "searches": 0, "candidates": [], "tool_calls": 0,
            "model_calls": 0, "empty_searches": 0, "revisions": 0,
            "answer": None, "termination": None, "elapsed_seconds": 0.0}
    tools = [openai.pydantic_function_tool(SearchArgs, name="wiki_search", description="Wiki 페이지를 검색해 후보 경로·제목·요약을 돌려준다."),
             openai.pydantic_function_tool(ReadArgs, name="wiki_read", description="index.md, <디렉터리>/_index.md, Wiki 페이지, sources/articles 원문을 최대 3개 읽는다."),
             openai.pydantic_function_tool(Answer, name="submit_answer", description="최종 답변을 제출한다. 관련 규정은 관련성 순으로.")]
    started = time.monotonic()

    def save():
        nonlocal started
        now = time.monotonic()
        state["elapsed_seconds"] += now - started
        started = now
        if checkpoint is not None:
            write_json(checkpoint, state)

    def add(key, values):
        state[key] = sorted(set(state[key]) | set(values))

    while state["model_calls"] < max_model_calls and not state["done"]:
        force = state["tool_calls"] >= max_tools or state["empty_searches"] >= empty_patience
        force_tool = "submit_answer" if force else (FIRST_TOOL.get(route) if state["tool_calls"] == 0 else None)
        message = client.complete(state["messages"], tools, case_id, state["model_calls"], force_tool=force_tool)
        state["model_calls"] += 1
        state["messages"].append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "refusal", "_response_items")})
        calls = message.get("tool_calls", [])
        if not calls:
            state["messages"].append({"role": "user", "content": "반드시 도구를 사용하거나 submit_answer로 답변을 제출하라."})
            state["trace"].append({"kind": "no_tool_output", "refusal": message.get("refusal")})
            save()
            continue
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
                if name == "submit_answer":
                    answer = Answer.model_validate(args)
                    problems = validate_answer(answer, store, set(state["read_pages"]))
                    if problems:
                        state["revisions"] += 1
                        result = {"accepted": False, "errors": problems}
                        if state["revisions"] > max_revisions:
                            state["done"] = True
                            state["termination"] = "invalid_answer"
                            state["rejected_answer"] = args
                    else:
                        state["answer"] = answer.model_dump()
                        state["done"] = True
                        state["termination"] = "budget_exhausted" if force else answer.status
                        result = {"accepted": True}
                elif name in ("wiki_search", "wiki_read"):
                    if force:
                        result = {"error": "retrieval budget/patience exhausted; submit evidence collected so far"}
                    else:
                        state["tool_calls"] += 1
                        if state["route"] is None:
                            state["route"] = "search_first" if name == "wiki_search" else "browse_first"
                        if name == "wiki_search":
                            result = store.search(SearchArgs.model_validate(args).query)
                            state["searches"] += 1
                            state["empty_searches"] = state["empty_searches"] + 1 if not result["results"] else 0
                            add("candidates", [r["path"] for r in result["results"]])
                        else:
                            result = store.read(ReadArgs.model_validate(args).paths)
                            for item in result["results"]:
                                kind = item.get("kind")
                                if kind == "index":
                                    add("read_indices", [item["path"]])
                                    add("candidates", item.get("entries", []))
                                elif kind == "source":
                                    add("read_sources", [item["path"]])
                                elif kind == "page":
                                    if item["path"] not in state["candidates"] and state["read_pages"]:
                                        add("followed_links", [item["path"]])
                                    add("read_pages", [item["path"]])
                else:
                    result = {"error": "unknown tool"}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"error": str(exc)}
            state["trace"].append({"step": state["model_calls"], "tool": name,
                                   "arguments": call["function"]["arguments"], "result": result})
            state["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                      "content": json.dumps(result, ensure_ascii=False)})
        save()
    if not state["done"]:
        state["done"] = True
        state["termination"] = "model_call_limit"
        save()
    return state
