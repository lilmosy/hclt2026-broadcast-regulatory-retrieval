"""Native Responses tool calls, stateless reasoning continuity and durable cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

from ...common import ROOT, write_json
from ..llm import utc_now


class APIUnavailable(RuntimeError):
    pass


def response_input(messages):
    items = []
    for message in messages:
        if message.get("_response_items") is not None:
            # Include reasoning/encrypted state along with every function call.
            items.extend(message["_response_items"])
        elif message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
        else:
            items.append({"role": message["role"], "content": message.get("content") or ""})
    return items


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


class ToolClient:
    def __init__(self, model, directory, max_output_tokens=5000, reasoning_effort="low"):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_output_tokens = max_output_tokens
        load_dotenv(ROOT / ".env", override=False)
        self.client = openai.OpenAI(timeout=120, max_retries=0)
        self.events = []

    def complete(self, messages, tools, case_id, step, force_tool=None):
        request = {"model": self.model, "input": response_input(messages),
                   "tools": [{"type": "function", **t["function"]} for t in tools],
                   "tool_choice": {"type": "function", "name": force_tool} if force_tool else "required",
                   "parallel_tool_calls": False, "max_output_tokens": self.max_output_tokens,
                   "reasoning": {"effort": self.reasoning_effort}, "store": False,
                   "include": ["reasoning.encrypted_content"]}
        key = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + ".json")
        if path.is_file():
            result = json.loads(path.read_text())
            self.events.append({"key": key, "cached": True, "usage": result["usage"]})
            return result["message"]
        for attempt in range(3):
            started = time.monotonic()
            try:
                response = self.client.responses.create(**request)
                output = [item.model_dump(exclude_none=True) for item in response.output]
                message = {"role": "assistant", "_response_items": output,
                    "tool_calls": [{"id": item["call_id"], "type": "function", "function": {
                        "name": item["name"], "arguments": item["arguments"]}}
                        for item in output if item["type"] == "function_call"]}
                usage = response.usage.model_dump() if response.usage else {}
                usage["prompt_tokens"] = usage.get("input_tokens", 0)
                usage["completion_tokens"] = usage.get("output_tokens", 0)
                record = {"timestamp": utc_now(), "case_id": case_id, "step": step, "key": key,
                          "model": self.model, "attempt": attempt + 1, "cached": False,
                          "latency_seconds": round(time.monotonic() - started, 3),
                          "usage": usage, "finish_reason": response.status}
                append_jsonl(self.directory / "calls.jsonl", record)
                write_json(path, {**record, "message": message})
                self.events.append(record)
                return message
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                record = {"timestamp": utc_now(), "case_id": case_id, "step": step, "key": key,
                          "attempt": attempt + 1, "error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
                append_jsonl(self.directory / "calls.jsonl", record)
                if getattr(exc, "code", None) in ("insufficient_quota", "billing_hard_limit_reached") or attempt == 2:
                    raise APIUnavailable(f"{type(exc).__name__}; saved checkpoint/cache") from exc
                time.sleep(10 * (attempt + 1))
            except openai.APIStatusError as exc:
                append_jsonl(self.directory / "calls.jsonl", {"timestamp": utc_now(),
                    "case_id": case_id, "step": step, "key": key,
                    "error_type": type(exc).__name__, "status_code": exc.status_code})
                raise
