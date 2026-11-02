"""Sequential QA evaluation over the Markdown Wiki. Gold is used only AFTER the agent returns.

python -m src.llmwiki.qa.evaluate --check
python -m src.llmwiki.qa.evaluate --run-name smoke --case-id QA_F46C1A7BBDF7 --confirm
python -m src.llmwiki.qa.evaluate --run-name full --confirm

Input: data/benchmark_400.jsonl (paper evaluation set). The agent sees only the scenario
(`artifact`) and the LLM-Wiki question in configs/paper.json; `gold_rule_id` is used for scoring.

Run directory: evaluation.jsonl (first line = run info and hashes, then one line per finished case
with answer, score and full trace), metrics.md (aggregate table, rewritten at the end of every
invocation) and api_cache/. Resume skips cases already in the file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from ...common import ROOT, file_sha256, load_jsonl, write_jsonl
from ..compile import load_config
from .agent import run_agent
from .metrics import aggregate, page_reach, render_markdown, score_prediction
from .provider import ToolClient
from .store import WikiStore

QA_PATH = ROOT / "data/benchmark_400.jsonl"
PAPER_CONFIG = ROOT / "configs/paper.json"
PROMPT_PATH = ROOT / "prompts/llmwiki/qa/qa.md"
OUTPUT_DIR = ROOT / "outputs/llmwiki/qa"
LIMITS = {"max_tools": 15, "empty_patience": 3, "max_model_calls": 20, "max_revisions": 2}


def load_benchmark(path=QA_PATH):
    """Public rows {qa_id, domain, artifact, gold_rule_id} → internal {qa_id, domain, text, gold}."""
    return [{"qa_id": r["qa_id"], "domain": r["domain"], "text": r["artifact"], "gold": [r["gold_rule_id"]]}
            for r in load_jsonl(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--run-name", default="qa")
    parser.add_argument("--route", choices=["auto", "search", "browse"], default="auto",
                        help="첫 도구를 강제한다 (확인용). 논문 기본은 auto")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="medium",
                        help="논문 설정은 medium")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    if args.model != "gpt-5.6-luna":
        raise ValueError("This experiment is frozen to gpt-5.6-luna")
    config = load_config()
    wiki_dir = ROOT / config["output_dir"]
    if not (wiki_dir / "index.md").is_file():
        raise SystemExit(f"Wiki가 없습니다: {wiki_dir}\n먼저 python -m src.llmwiki.compile --generate --confirm 으로 컴파일하세요.")
    store = WikiStore(wiki_dir, search_k=int(config.get("qa_search_k", 10)))
    question = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))["llmwiki"]["question"]
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    dataset = load_benchmark()
    if len({q["qa_id"] for q in dataset}) != len(dataset):
        raise ValueError("duplicate QA IDs")
    for q in dataset:
        if not q["text"].strip():
            raise ValueError("empty artifact")
        if any(r not in store.rule_ids for r in q["gold"]):
            raise ValueError("gold rule ID missing from sources/articles")
    wanted = set(args.case_id)
    selected = [q for q in dataset if not wanted or q["qa_id"] in wanted]
    if wanted - {q["qa_id"] for q in dataset}:
        raise ValueError("unknown case id")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:args.limit]
    if args.check or not args.confirm:
        print(json.dumps({"dataset": len(dataset), "selected": len(selected), "pages": len(store.pages),
                          "rule_ids": len(store.rule_ids), "model": args.model, "route": args.route,
                          "smoke_search": store.search("방송 표현", 3)}, ensure_ascii=False, indent=2))
        return
    directory = OUTPUT_DIR / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This run is already active; not starting a duplicate")
    code_paths = sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "src/common.py"]
    code_hash = hashlib.sha256("\n".join(p.name + ":" + file_sha256(p) for p in code_paths).encode()).hexdigest()
    run_info = {"run": args.run_name, "model": args.model, **store.hashes, "dataset": file_sha256(QA_PATH),
                "prompt": file_sha256(PROMPT_PATH), "question": question, "code": code_hash, "route": args.route,
                "search_k": store.search_k, **LIMITS, "api": "responses", "reasoning_effort": args.reasoning,
                "metric": "hit@1 = first-ranked regulation is a gold rule unit (exact 항·호)"}
    out = directory / "evaluation.jsonl"
    rows = load_jsonl(out) if out.exists() else []
    if rows:
        if rows[0] != run_info:
            raise ValueError("Inputs/config/code changed since this run started; use a new --run-name")
        done = {r["qa_id"] for r in rows[1:]}
    else:
        write_jsonl(out, [run_info])
        done = set()
    client = ToolClient(args.model, directory / "api_cache", reasoning_effort=args.reasoning)
    gold_of = {q["qa_id"]: q["gold"] for q in dataset}

    def append(row):
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        for q in selected:
            case_id = q["qa_id"]
            if case_id in done:
                continue
            print(f"case {case_id}: running", flush=True)
            artifact = {"domain": q["domain"], "text": q["text"]}   # never the gold
            state = run_agent(store=store, client=client, prompt=prompt, question=question, artifact=artifact,
                              case_id=case_id, route=args.route, **LIMITS)
            pred = [r["rule_id"] for r in (state["answer"] or {}).get("applicable_regulations", [])]
            gold = gold_of[case_id]
            gold_pages = {g: store.pages_citing(g) for g in gold}
            row = {"qa_id": case_id, "domain": artifact["domain"], "answer": state["answer"],
                   "predicted_rule_ids": pred, "gold_rule_ids": gold, "gold_pages": gold_pages,
                   "metrics": score_prediction(pred, gold, store.article_of),
                   "reach": page_reach(state["read_pages"], gold_pages),
                   "termination": state["termination"], "route": state["route"],
                   "tool_calls": state["tool_calls"], "searches": state["searches"], "model_calls": state["model_calls"],
                   "elapsed_seconds": state["elapsed_seconds"], "read_indices": state["read_indices"],
                   "read_pages": state["read_pages"], "read_sources": state["read_sources"],
                   "followed_links": state["followed_links"], "trace": state["trace"]}
            append(row)
            done.add(case_id)
            print(f"completed {len(done)}/{len(selected)}; tools={state['tool_calls']}; route={state['route']}; {state['termination']}", flush=True)
        scored = [r for r in load_jsonl(out)[1:] if r["qa_id"] in {q["qa_id"] for q in selected}]
        report = directory / "metrics.md"
        previous = report.read_text(encoding="utf-8") if report.exists() else ""
        report.write_text(render_markdown(aggregate(scored, len(selected), store.article_of), run_info, previous), encoding="utf-8")
        print(f"metrics written to {report}", flush=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
