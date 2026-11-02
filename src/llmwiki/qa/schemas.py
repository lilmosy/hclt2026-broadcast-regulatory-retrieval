from typing import Literal

from pydantic import Field

from ..schemas import StrictModel


class SearchArgs(StrictModel):
    query: str = Field(description="짧고 구체적인 검색어")


class ReadArgs(StrictModel):
    paths: list[str] = Field(min_length=1, max_length=3, description="index.md, <디렉터리>/_index.md, 페이지, sources/articles 경로")


class Regulation(StrictModel):
    rule_id: str = Field(description="Wiki fact 또는 원문에 표시된 조·항·호·목 ID")
    rationale: str = Field(description="제작안의 사실과 해당 조항의 요건을 연결한 설명")


class Answer(StrictModel):
    status: Literal["answered", "insufficient_evidence"]
    review_required: bool | None
    applicable_regulations: list[Regulation] = Field(
        description="관련성이 가장 높은 규정부터 내림차순", max_length=10)
    overall_answer: str
    missing_information: list[str]
