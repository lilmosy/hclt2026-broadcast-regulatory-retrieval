# HCLT 2026: Broadcasting Regulatory Retrieval

Official research materials for **Evaluating Agentic Regulatory Retrieval Using a Synthetic Benchmark from Broadcasting Violations**.

**방송심의 위반 사례 기반 합성 벤치마크를 활용한 에이전트 규정 검색 평가**

Soyoung Cho · Yoonseok Heo — Sogang University, HCLT 2026

이 저장소는 논문의 주 평가에 사용한 **단일 정답 400건**, 검색 대상 규정 원문, 제작안 생성 및 LLM-Wiki 프롬프트를 공개합니다.

## Release status

This initial release contains the paper's 400-case evaluation subset, the fixed regulatory corpus, source references, available prompt templates, and documented experiment settings. Evaluation code, baseline selector prompts, full system implementations, and prediction files will be added in subsequent updates. This release is not yet a complete reproduction package.

## Data

- [Benchmark: 400 cases](data/benchmark_400.jsonl)
- [Data documentation](data/README.md)
- [Regulatory corpus: 226 articles / 1,040 rule units](data/legal_corpus.jsonl)
- [Original regulatory TXT snapshots](data/regulations/)
- [Decision references and page numbers](data/sources.jsonl)
- [File checksums and export provenance](data/manifest.json)

The paper describes 532 synthetic cases in total. This repository initially releases the **400 cases with exactly one gold regulation** used in its main evaluation. The remaining 132 multi-answer cases are outside this release. This selection is not a train/test split. The retrieval corpus retains all 1,040 rule units.

| Domain | Evaluation cases |
|---|---:|
| Broadcasting programs | 187 |
| Broadcasting advertisements | 78 |
| Product-sales broadcasts | 135 |
| **Total** | **400** |

## Task and prompts

Given a pre-broadcast production scenario and a method-specific review question, return up to ten unique regulation IDs in ranked order. Evaluation uses exact rule-ID matching; returning a parent article does not count as a correct fine-grained answer.

- [Prompt templates and their roles](prompts/README.md)
- [Paper settings and method-specific questions](configs/paper.json)

Only the scenario and the appropriate question are model inputs. Gold IDs and decision provenance are evaluation metadata and must not be included in retrieval or model input. Baselines and LLM-Wiki use different questions and reasoning settings, as documented in the paper.

## Results reported in the paper

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| BM25 + Luna | 11.25 | 15.50 | 16.75 |
| DenseRAG + Luna | 26.75 | 57.00 | 65.00 |
| GraphRAG-Local + reranking + Luna | 11.50 | 30.50 | 37.00 |
| LLM-Wiki | 38.25 | 63.75 | 66.75 |

Values are percentages over 400 cases. These are system-level comparisons with different reasoning settings and exploration budgets. The table transcribes the manuscript; executable reproduction instructions will accompany the evaluation-code release.

## Load the benchmark

```python
import json
from pathlib import Path

cases = [json.loads(line) for line in
         Path("data/benchmark_400.jsonl").read_text(encoding="utf-8").splitlines()]
settings = json.loads(Path("configs/paper.json").read_text(encoding="utf-8"))
question = settings["llmwiki"]["question"]
scenario = cases[0]["artifact"]
# Keep cases[0]["gold_rule_id"] out of the model input.
```

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Proceedings details and the paper link will be added when available.

## Sources and reuse

Regulatory texts and decision references originate from public materials of the Korea Communications Standards Commission. Scenarios are LLM-generated reconstructions of recorded broadcasting content. See the data documentation for construction and validation scope. A reuse license for the research materials has not yet been specified; this initial release does not assign a blanket license to third-party source materials.
