# Released prompt templates

## Scenario generation

`artifact_generation.md` is a byte-identical copy of the original `BINGO_artifact_rewrite_v2.md`. Only the filename was changed. Fill `{DOMAIN}`, `{BROADCAST_DATE}` and `{RAW_TEXT}` with the original generation inputs. `RAW_TEXT` refers to an official decision-row extract, not the synthesized scenario.

The system message is stored under `artifact_generation` in `system_messages.json`. The generation implementation called `chat.completions.parse` with model `gpt-5.6-luna` and structured output containing one string field, `artifact`, with additional fields forbidden. Sampling parameters were not explicitly passed in that call. The initial release provides source references rather than all original generation inputs or the generation pipeline.

## LLM-Wiki

The following templates preserve the original file contents:

- `llmwiki/compile/select_v1.md`: select existing pages for an incoming article.
- `llmwiki/compile/compile_v1.md`: create or update pages.
- `llmwiki/compile/content_validate_v1.md`: check compiled content against source articles.
- `llmwiki/compile/repair_v1.md`: repair pages using validation feedback.
- `llmwiki/qa/qa.md`: query-time regulatory retrieval agent instructions.

Compilation system messages are in `system_messages.json`; query-time instructions are in `llmwiki/qa/qa.md`. The complete tool schemas and execution implementation are not part of this initial release. Use `../configs/paper.json` for the actual method-specific questions and paper settings. Baseline selector prompts will be added with the corresponding code.
