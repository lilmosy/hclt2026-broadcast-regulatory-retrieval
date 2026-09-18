# Paper evaluation data

## Scope

This release contains the 400 single-gold cases used in the HCLT 2026 paper. They are selected from the existing 532-case synthetic dataset by `len(gold.applicable_regulations) == 1`, retaining the original case order, IDs, scenario text and gold rule IDs. No new generation or relabeling was performed for this export.

The subset contains 187 broadcasting-program, 78 broadcasting-advertisement and 135 product-sales cases. Gold units are 89 articles, 228 paragraphs, 66 items and 17 subitems. The selection is not a training/test partition.

## Files and schema

`benchmark_400.jsonl` contains one JSON object per case:

| Field | Meaning |
|---|---|
| `qa_id` | Original stable case identifier |
| `domain` | `broadcast_program`, `broadcast_ad` or `product_sales` |
| `artifact` | Pre-broadcast scenario; original text preserved |
| `gold_rule_id` | The single gold regulation ID used for exact-match evaluation |

Questions are method-specific and provided in `../configs/paper.json`. Join `sources.jsonl` by `qa_id` to obtain original decision IDs and source references (`source_id`, PDF `pages`, official `board_url`). These references are not model inputs. Official source links are preserved from the original records; their current availability has not been rechecked in this release.

`legal_corpus.jsonl` is a byte-identical copy of the experiment's fixed corpus. It has 226 article-level records and nested `sub_items`, covering 1,040 rule units in total. It is not a flat 1,040-line file. `regulations/` preserves the three original TXT snapshots. These are experiment snapshots, not a claim about the latest regulations.

## Construction and validation

Official decision records were used to identify the reference regulations. An LLM rewrote recorded content as pre-broadcast scenarios, with instructions to preserve content facts and remove decision outcomes, regulatory citations and retrospective judgments. The generation prompt and its system message are included in `../prompts/`.

The source pipeline records automated validation and case filtering. Source records carry the internal status `auto_draft`; this field is omitted from the retrieval-focused public schema. Publication does not change the underlying review status or certify expert review of every case. Gold references reflect the case construction process and should not be interpreted as an exhaustive list of every potentially relevant provision.

`review_required`, generated rationales, overall answers and official outcomes are omitted because the paper's main evaluation measures regulation retrieval. The original working dataset remains unchanged.

## Version integrity

`manifest.json` records source file hashes, the ordered evaluation-ID hash, and checksums for the public payload. The public export has a new schema and therefore a different file hash from the original dataset. Future revisions should preserve this distinction and describe changes in release notes.
