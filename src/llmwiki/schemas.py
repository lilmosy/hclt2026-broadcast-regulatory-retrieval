"""LLM structured output 스키마 (OpenAI parse용 pydantic).

strict mode 호환을 위해 모든 필드는 필수이고 extra는 금지한다.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectOutput(StrictModel):
    update: list[str] = Field(description="갱신할 기존 페이지 제목. 목록의 제목을 그대로 쓴다.")


class FactOut(StrictModel):
    text: str
    sources: list[str] = Field(description="이 fact가 적힌 가장 구체적인 단위의 rule_id")


class LinkOut(StrictModel):
    title: str = Field(description="연결할 페이지 제목")
    relation: str = Field(description="왜 이 페이지로 가야 하는지 한 구절")


class PageOut(StrictModel):
    title: str
    type: str = Field(description="페이지가 들어갈 디렉터리 이름 (짧은 한국어 명사)")
    aliases: list[str]
    tags: list[str]
    summary: str = Field(description="한 줄 요약")
    facts: list[FactOut]
    links: list[LinkOut]
    sources: list[str] = Field(description="이 페이지의 근거가 되는 rule_id 전체")


class CompileOutput(StrictModel):
    pages: list[PageOut]


# ---- 2단계: content 검증과 LLM 수정 ------------------------------------------
from typing import Literal  # noqa: E402


class ContentError(StrictModel):
    type: Literal["unsupported_fact", "cross_page_contradiction"]
    page: str = Field(description="페이지 제목 그대로")
    detail: str = Field(description="어느 fact·relation이 왜 문제인지 구체적으로")


class ContentValidateOutput(StrictModel):
    errors: list[ContentError]


class RepairOutput(StrictModel):
    pages: list[PageOut]
