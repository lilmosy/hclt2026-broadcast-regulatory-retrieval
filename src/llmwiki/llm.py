"""luna(OpenAI 호환) structured-output 클라이언트 + 입력 해시 cache + 호출 로그.

같은 (model, schema, system, user)면 cache에서 꺼내고 API를 부르지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from ..common import ROOT, write_json

T = TypeVar("T", bound=BaseModel)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LunaClient:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.calls: list[dict[str, Any]] = []
        load_dotenv(ROOT / ".env", override=False)
        import openai

        self._openai = openai
        self._client = openai.OpenAI()

    def _cache_key(self, schema_name: str, system: str, user: str) -> str:
        blob = "\n".join([self.model, schema_name, system, user])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def call(
        self,
        *,
        stage: str,
        source_id: str,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        key = self._cache_key(schema.__name__, system, user)
        cache_path = self.cache_dir / f"{key}.json"
        started = time.perf_counter()
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = schema.model_validate(cached["output"])
            self.calls.append(
                {
                    "stage": stage,
                    "source_id": source_id,
                    "model": self.model,
                    "cached": True,
                    "timestamp": utc_now(),
                    "input_tokens": cached.get("input_tokens", 0),
                    "output_tokens": cached.get("output_tokens", 0),
                    "estimated_cost_usd": 0.0,
                }
            )
            return parsed

        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=schema,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("structured output이 비어 있습니다.")
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost = None
                if self.input_price is not None and self.output_price is not None:
                    cost = (
                        input_tokens * self.input_price + output_tokens * self.output_price
                    ) / 1_000_000
                write_json(
                    cache_path,
                    {
                        "stage": stage,
                        "source_id": source_id,
                        "model": self.model,
                        "input_sha256": key,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "output": parsed.model_dump(),
                    },
                )
                self.calls.append(
                    {
                        "stage": stage,
                        "source_id": source_id,
                        "model": self.model,
                        "cached": False,
                        "attempt": attempt,
                        "timestamp": utc_now(),
                        "latency_seconds": round(time.perf_counter() - started, 3),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost_usd": round(cost, 8) if cost is not None else None,
                    }
                )
                return parsed
            except self._openai.RateLimitError as exc:
                last_error = exc
                time.sleep(15 * attempt)
            except (self._openai.APIConnectionError, self._openai.APITimeoutError) as exc:
                last_error = exc
                time.sleep(5 * attempt)
        raise RuntimeError(f"{source_id}/{stage}: luna 호출 실패: {last_error}") from last_error

    def totals(self) -> dict[str, Any]:
        live = [c for c in self.calls if not c["cached"]]
        cost_values = [c["estimated_cost_usd"] for c in live if c.get("estimated_cost_usd") is not None]
        return {
            "calls": len(self.calls),
            "live_calls": len(live),
            "cached_calls": len(self.calls) - len(live),
            "input_tokens": sum(c["input_tokens"] for c in live),
            "output_tokens": sum(c["output_tokens"] for c in live),
            "estimated_cost_usd": round(sum(cost_values), 6) if cost_values else None,
        }
