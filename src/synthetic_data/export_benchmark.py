"""Export the paper evaluation set from the full synthetic dataset.

python -m src.synthetic_data.export_benchmark                # writes data/benchmark_400.jsonl, data/sources.jsonl
python -m src.synthetic_data.export_benchmark --verify       # compares with data/manifest.json, writes nothing

Selection: cases with exactly one gold regulation (len(gold.applicable_regulations) == 1),
in the original order. Rationales, overall answers and official outcomes are dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from ..common import ROOT, file_sha256, load_jsonl
from .build_synthetic_data import BOARD_URL, SOURCE_CONFIG

SYNTHETIC_QA = ROOT / "outputs" / "synthetic_data" / "synthetic_qa.jsonl"
BENCHMARK = ROOT / "data" / "benchmark_400.jsonl"
SOURCES = ROOT / "data" / "sources.jsonl"
MANIFEST = ROOT / "data" / "manifest.json"


def export(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ntt = {s["source_id"]: s["ntt_id"] for s in yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))["sources"]}
    benchmark, sources = [], []
    for row in rows:
        regs = row["gold"]["applicable_regulations"]
        if len(regs) != 1:
            continue
        benchmark.append({"qa_id": row["qa_id"], "domain": row["artifact"]["domain"],
                          "artifact": row["artifact"]["text"], "gold_rule_id": regs[0]["rule_id"]})
        sources.append({"qa_id": row["qa_id"], "decision_ids": row["provenance"]["decision_ids"],
                        "sources": [{**s, "board_url": BOARD_URL.format(ntt_id=ntt[s["source_id"]])}
                                    for s in row["provenance"]["sources"]]})
    return benchmark, sources


def dump(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=SYNTHETIC_QA)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    benchmark, sources = export(load_jsonl(args.input))
    payload = {BENCHMARK: dump(benchmark), SOURCES: dump(sources)}
    if not args.verify:
        for path, data in payload.items():
            path.write_bytes(data)
        print(f"{len(benchmark)} cases -> {BENCHMARK.relative_to(ROOT)}, {SOURCES.relative_to(ROOT)}")
        return

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "input": (file_sha256(args.input), manifest["source_sha256"]["real/data/processed/synthetic_data/synthetic_qa.jsonl"]),
        "case_count": (len(benchmark), manifest["case_count"]),
        "ordered_qa_ids": (hashlib.sha256("".join(r["qa_id"] + "\n" for r in benchmark).encode()).hexdigest(),
                           manifest["ordered_qa_ids_sha256"]),
        **{path.relative_to(ROOT).as_posix(): (hashlib.sha256(data).hexdigest(),
                                                manifest["files_sha256"][path.relative_to(ROOT).as_posix()])
           for path, data in payload.items()},
    }
    for name, (got, want) in checks.items():
        print(f"{'OK  ' if got == want else 'DIFF'} {name}")
    if any(got != want for got, want in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
