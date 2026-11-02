"""LLM-Wiki 컴파일 CLI (논문 Algorithm 1의 1단계 구현).

조문마다: SELECT(갱신할 기존 페이지 고르기) → COMPILE(페이지 JSON 쓰기) → 병합 →
구조 검증 → Error Book → manifest 저장. 끝나면 Markdown 렌더링과 리포트.

실행 예 (저장소 루트에서):
  python -m src.llmwiki.compile --check                      # API 0콜: 렌더링·프롬프트 조립 검증
  python -m src.llmwiki.compile --generate --document AD_REVIEW --confirm
  python -m src.llmwiki.compile --generate --confirm          # 3규정 전체 (완료분은 resume)
  python -m src.llmwiki.compile --render                      # manifest에서 .md만 다시 찍기
  python -m src.llmwiki.compile --finalize                    # dangling link 자동수정 + Error Book 종결
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..common import ROOT, compact_hash, flatten_rule_units, load_jsonl, write_json
from .articles import render_article_markdown, rule_ids_in_markdown, texts_in_record
from .error_book import ErrorBook
from .repair import llm_repair
from .schemas import CompileOutput, SelectOutput
from .validators import code_autofix, content_validate, structural_validate
from .wiki import Wiki

DEFAULT_CONFIG_PATH = ROOT / "configs" / "llmwiki.yaml"
CORPUS_PATH = ROOT / "data" / "legal_corpus.jsonl"
PLACEHOLDER_RE = re.compile(r"\{[A-Z_]+\}")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dir(config: dict[str, Any]) -> Path:
    """컴파일 상태(_manifest.json)·API cache·실행 로그가 있는 곳. Wiki 트리와 분리한다."""
    path = ROOT / config.get("run_dir", "outputs/llmwiki/compile")
    path.mkdir(parents=True, exist_ok=True)
    return path


def silent_types(config: dict[str, Any]) -> frozenset[str]:
    return frozenset(config.get("error_book_silent_types") or ())


def read_prompt(relative: str) -> tuple[str, str]:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"프롬프트가 비어 있습니다: {path}")
    return text, compact_hash(text, 16)


def render_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    unresolved = PLACEHOLDER_RE.findall(rendered)
    if unresolved:
        raise ValueError(f"미해소 placeholder: {unresolved}")
    return rendered


def build_select_prompt(template: str, wiki: Wiki, article_md: str, k: int) -> str:
    return render_prompt(
        template,
        {
            "MAX_SELECT": str(k),
            "INDEX": "\n".join(wiki.index_lines()) or "(아직 페이지 없음)",
            "ARTICLE": article_md.strip(),
        },
    )


def build_compile_prompt(
    template: str, wiki: Wiki, article_md: str, selected: list[str], constraints: list[str]
) -> str:
    selected_json = (
        json.dumps([wiki.page_for_prompt(t) for t in selected], ensure_ascii=False, indent=1)
        if selected
        else "(없음 — 모두 새 페이지로 만든다)"
    )
    return render_prompt(
        template,
        {
            "CONSTRAINTS": "\n".join(f"- {c}" for c in constraints) or "- (아직 없음)",
            "INDEX": "\n".join(wiki.index_lines()) or "(아직 페이지 없음)",
            "SELECTED_PAGES": selected_json,
            "ARTICLE": article_md.strip(),
        },
    )


def select_articles(
    articles: list[dict[str, Any]], documents: list[str], limit: int | None
) -> list[dict[str, Any]]:
    chosen = [a for a in articles if not documents or a["document_id"] in documents]
    return chosen[:limit] if limit else chosen


# ---------------------------------------------------------------------------
def run_check(config: dict[str, Any], articles: list[dict[str, Any]], units: dict[str, Any]) -> int:
    """API 0콜. 226건 렌더링이 rule_id·본문을 빠짐없이 담는지, 프롬프트가 조립되는지 확인."""
    problems: list[str] = []
    total_chars = 0
    for article in articles:
        md = render_article_markdown(article)
        total_chars += len(md)
        expected_ids = [rid for rid, u in units.items() if u["article_rule_id"] == article["rule_id"]]
        got_ids = rule_ids_in_markdown(md)
        if sorted(got_ids) != sorted(expected_ids) or len(got_ids) != len(set(got_ids)):
            problems.append(f"{article['rule_id']}: rule_id 불일치 {len(got_ids)} vs {len(expected_ids)}")
        for text in texts_in_record(article):
            if text not in md:
                problems.append(f"{article['rule_id']}: 본문 누락 '{text[:30]}…'")
    select_tpl, _ = read_prompt(config["select_prompt"])
    compile_tpl, _ = read_prompt(config["compile_prompt"])
    wiki = Wiki()
    sample = articles[0]
    md = render_article_markdown(sample)
    select_prompt = build_select_prompt(select_tpl, wiki, md, config["select_max"])
    compile_prompt = build_compile_prompt(compile_tpl, wiki, md, [], [])
    print(f"조문 {len(articles)}건, rule unit {len(units)}개, 렌더링 총 {total_chars:,}자")
    print(f"문제 {len(problems)}건")
    for p in problems[:20]:
        print("  -", p)
    print(f"\n샘플 조문 markdown ({sample['rule_id']}):\n{md}")
    print(f"SELECT 프롬프트 {len(select_prompt):,}자 / COMPILE 프롬프트 {len(compile_prompt):,}자 (빈 Wiki 기준)")
    print(f"\n예상 호출 수: SELECT ≤{len(articles) - 1} + COMPILE {len(articles)} = 최대 {2 * len(articles) - 1}콜")
    return 1 if problems else 0


def run_generate(
    config: dict[str, Any],
    articles: list[dict[str, Any]],
    all_articles: list[dict[str, Any]],
    units: dict[str, Any],
    out_dir: Path,
    model: str,
) -> None:
    from .llm import LunaClient

    run_dir = build_dir(config)
    manifest_path = run_dir / "_manifest.json"
    wiki = Wiki.load(manifest_path)
    error_book = ErrorBook(out_dir / "error_book.yaml", silent_types(config))
    unit_ids = set(units)
    select_tpl, select_ver = read_prompt(config["select_prompt"])
    compile_tpl, compile_ver = read_prompt(config["compile_prompt"])
    validate_tpl, validate_ver = read_prompt(config["content_validate_prompt"])
    repair_tpl, repair_ver = read_prompt(config["repair_prompt"])
    validate_every = int(config.get("content_validate_every", 5))
    repair_every = int(config.get("repair_every_validations", 10))
    articles_by_id = {a["rule_id"]: a for a in all_articles}
    client = LunaClient(
        model=model,
        cache_dir=run_dir / "api_cache",
        input_price_per_million=config.get("input_price_per_million"),
        output_price_per_million=config.get("output_price_per_million"),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = run_dir / f"compile_log_{stamp}.json"
    wiki.meta.setdefault("model", model)
    wiki.meta["prompts"] = {
        "select": select_ver, "compile": compile_ver,
        "content_validate": validate_ver, "repair": repair_ver,
    }
    state = wiki.meta.setdefault(
        "stage2",
        {"pending_validation": [], "pending_repair": [], "since_validation": 0,
         "validation_count": 0, "validations": [], "repairs": []},
    )
    done = wiki.done_rule_ids()
    todo = [a for a in articles if a["rule_id"] not in done]
    print(f"완료 {len(done)}건, 이번 대상 {len(todo)}건, 모델 {model}")

    def flush(status: str) -> None:
        wiki.save(manifest_path)
        error_book.save()
        write_json(
            run_path,
            {
                "run_id": run_path.stem,
                "status": status,
                "model": model,
                "prompts": wiki.meta["prompts"],
                "totals": client.totals(),
                "calls": client.calls,
            },
        )

    try:
        for i, article in enumerate(todo, 1):
            rid = article["rule_id"]
            md = render_article_markdown(article)
            index_size = len(wiki.pages)
            selected: list[str] = []
            if wiki.pages:
                sel = client.call(
                    stage="select",
                    source_id=rid,
                    system=config["select_system"],
                    user=build_select_prompt(select_tpl, wiki, md, config["select_max"]),
                    schema=SelectOutput,
                )
                seen: set[str] = set()
                for title in sel.update:
                    title = title.split("/", 1)[-1].strip() if "/" in title else title.strip()
                    if title in wiki.pages and title not in seen:
                        selected.append(title)
                        seen.add(title)
                selected = selected[: config["select_max"]]
            constraints = error_book.active_constraints()
            out = client.call(
                stage="compile",
                source_id=rid,
                system=config["compile_system"],
                user=build_compile_prompt(compile_tpl, wiki, md, selected, constraints),
                schema=CompileOutput,
            )
            raw_pages = [p.model_dump() for p in out.pages]
            created, updated = [], []
            for page in raw_pages:
                status, title = wiki.upsert(page, rid)
                (created if status == "created" else updated).append(title)
            touched = created + updated
            errors = structural_validate(wiki, touched, unit_ids)
            if not touched:
                errors.append({"type": "empty_emission", "page": rid, "detail": "페이지 0개"})
            error_book.discover(errors, rid)
            wiki.emissions.append(
                {
                    "rule_id": rid,
                    "document_id": article["document_id"],
                    "index_size": index_size,
                    "selected": selected,
                    "constraints_injected": constraints,
                    "created": created,
                    "updated": updated,
                    "errors": errors,
                    "raw_pages": raw_pages,
                }
            )
            state["pending_validation"] += [t for t in touched if t not in state["pending_validation"]]
            state["pending_repair"] += [e for e in errors if e["type"] == "incomplete_page"]
            state["since_validation"] += 1
            print(
                f"[{i}/{len(todo)}] {rid}: select={len(selected)} "
                f"created={len(created)} updated={len(updated)} errors={len(errors)}"
            )
            is_last = i == len(todo)
            if state["since_validation"] >= validate_every or is_last:
                run_stage2(
                    client=client, wiki=wiki, error_book=error_book, state=state, units=units,
                    articles_by_id=articles_by_id, config=config, validate_tpl=validate_tpl,
                    repair_tpl=repair_tpl, repair_every=repair_every, rid=rid, is_last=is_last,
                )
            flush("running")
    except KeyboardInterrupt:
        flush("interrupted")
        print("중단됨. manifest 저장 완료 — 같은 명령으로 이어서 실행할 수 있습니다.")
        sys.exit(130)
    except Exception:
        flush("failed")
        raise
    flush("finished")
    finish(wiki, error_book, all_articles, units, out_dir, client.totals())


def run_stage2(
    *, client, wiki, error_book, state, units, articles_by_id, config,
    validate_tpl, repair_tpl, repair_every, rid, is_last,
) -> None:
    """batch content 검증(논문 Algorithm 1 line 5) + 주기적 LLM 수리(line 14~16)."""
    if state["pending_validation"]:
        found = content_validate(
            client=client, wiki=wiki, titles=state["pending_validation"], units=units,
            articles_by_id=articles_by_id, template=validate_tpl,
            system=config["content_validate_system"], source_id=f"validate@{rid}",
        )
        error_book.discover(found, rid)
        state["validations"].append(
            {"after_rule_id": rid, "titles": list(state["pending_validation"]), "errors": found}
        )
        state["pending_repair"] += found
        state["validation_count"] += 1
        print(f"    content 검증: 페이지 {len(state['pending_validation'])}개 → 오류 {len(found)}건")
    state["pending_validation"] = []
    state["since_validation"] = 0
    due = state["validation_count"] > 0 and state["validation_count"] % repair_every == 0
    if state["pending_repair"] and (due or is_last):
        result = llm_repair(
            client=client, wiki=wiki, errors=state["pending_repair"], units=units,
            articles_by_id=articles_by_id, template=repair_tpl,
            system=config["repair_system"], source_id=f"repair@{rid}",
        )
        post: list[dict[str, str]] = []
        if result["repaired"]:
            post = content_validate(
                client=client, wiki=wiki, titles=result["repaired"], units=units,
                articles_by_id=articles_by_id, template=validate_tpl,
                system=config["content_validate_system"], source_id=f"postcheck@{rid}",
            )
            error_book.discover(post, rid)
        state["repairs"].append(
            {"after_rule_id": rid, "input_errors": list(state["pending_repair"]), **result,
             "post_check_errors": list(post)}
        )
        print(f"    LLM 수리: 대상 {len(result['targets'])} → 수리 {len(result['repaired'])}, 재검증 오류 {len(post)}건")
        state["pending_repair"] = [
            e for e in state["pending_repair"] if e.get("page") not in result["repaired"]
        ] + post


def finish(
    wiki: Wiki,
    error_book: ErrorBook,
    all_articles: list[dict[str, Any]],
    units: dict[str, Any],
    out_dir: Path,
    totals: dict[str, Any],
) -> None:
    render = wiki.render_all(out_dir, all_articles, units)
    print(json.dumps({"pages": len(wiki.pages), "error_book": error_book.counts(), "calls": totals},
                     ensure_ascii=False, indent=1))
    print(f"렌더링: {render}")


def run_finalize(all_articles, units, out_dir: Path, config: dict[str, Any]) -> None:
    """dangling link·없는 rule_id 인용 자동수정(논문 Layer 1) 후 Error Book 종결."""
    wiki = Wiki.load(build_dir(config) / "_manifest.json")
    error_book = ErrorBook(out_dir / "error_book.yaml", silent_types(config))
    errors = structural_validate(wiki, list(wiki.pages), set(units))
    fixed = code_autofix(wiki, errors)
    remaining = structural_validate(wiki, list(wiki.pages), set(units))
    error_book.verify_and_close(remaining, layers={"structural"})
    error_book.save()
    wiki.meta["finalized"] = {"fixed": fixed, "remaining": len(remaining)}
    wiki.save(build_dir(config) / "_manifest.json")
    print(f"자동수정 {fixed}건, 남은 구조 오류 {len(remaining)}건")
    finish(wiki, error_book, all_articles, units, out_dir, {})


def main(default_config: Path = DEFAULT_CONFIG_PATH) -> None:
    parser = argparse.ArgumentParser(description="LLM-Wiki 컴파일")
    parser.add_argument("--config", default=str(default_config), help="설정 YAML 경로")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="API 0콜 검증")
    mode.add_argument("--generate", action="store_true", help="LLM 호출로 적재 (완료분은 resume)")
    mode.add_argument("--render", action="store_true", help="manifest에서 .md만 다시 생성")
    mode.add_argument("--finalize", action="store_true", help="dangling link 자동수정 + Error Book 종결")
    parser.add_argument("--document", action="append", default=[], help="document_id 필터 (반복 가능)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 조문만")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None, help="출력 디렉터리 (기본 config.output_dir)")
    parser.add_argument("--confirm", action="store_true", help="--generate 시 실제 API 호출 허용")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    all_articles = load_jsonl(CORPUS_PATH)
    units = flatten_rule_units(all_articles)
    articles = select_articles(all_articles, args.document, args.limit)
    out_dir = Path(args.out) if args.out else ROOT / config["output_dir"]
    model = args.model or config["model"]

    if args.check:
        sys.exit(run_check(config, articles, units))
    if args.render:
        wiki = Wiki.load(build_dir(config) / "_manifest.json")
        finish(wiki, ErrorBook(out_dir / "error_book.yaml", silent_types(config)),
               all_articles, units, out_dir, {})
        return
    if args.finalize:
        run_finalize(all_articles, units, out_dir, config)
        return
    if not args.confirm:
        done = Wiki.load(build_dir(config) / "_manifest.json").done_rule_ids()
        todo = [a for a in articles if a["rule_id"] not in done]
        print(
            f"실행 계획: 조문 {len(todo)}건 (완료 {len(done)}건 제외), 최대 {2 * len(todo)}콜, "
            f"모델 {model}, 출력 {out_dir}\n--confirm 을 붙이면 실제로 호출합니다."
        )
        return
    run_generate(config, articles, all_articles, units, out_dir, model)


if __name__ == "__main__":
    main()
