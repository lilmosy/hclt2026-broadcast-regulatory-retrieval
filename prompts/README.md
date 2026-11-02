# 프롬프트

모두 논문 실험에 사용한 템플릿 원본이며 파일명만 바꿨습니다.

## 벤치마크 생성

| 파일 | 단계 | 치환 변수 |
|---|---|---|
| `artifact_generation.md` | 공식 의결 행 → 방송 전 제작안 | `{DOMAIN}`, `{BROADCAST_DATE}`, `{RAW_TEXT}` |
| `rationale_generation.md` | 제작안 + 정답 규정 원문 → 규정별 해설 (400건 공개본에는 미포함) | 파일 참조 |

`RAW_TEXT`는 합성된 제작안이 아니라 공식 의결문에서 뽑은 행입니다. 제작안 생성의 system 메시지는 `system_messages.json`에, 해설 생성의 system 메시지는 `src/synthetic_data/llm_generation.py`에 있습니다. 생성은 `gpt-5.6-luna`, `chat.completions.parse`, 구조화 출력을 사용했고 샘플링 파라미터는 따로 지정하지 않았습니다.

## LLM-Wiki

| 파일 | 역할 |
|---|---|
| `llmwiki/compile/select_v1.md` | 새 조문이 들어올 때 갱신할 기존 페이지를 최대 5개 선택 |
| `llmwiki/compile/compile_v1.md` | 개체 단위 페이지 생성·갱신 |
| `llmwiki/compile/content_validate_v1.md` | 페이지 내용을 인용 조문 원문과 대조 |
| `llmwiki/compile/repair_v1.md` | 검증에서 걸린 페이지 수정 |
| `llmwiki/qa/qa.md` | 질의 시 에이전트 지시문 |

컴파일 system 메시지는 `system_messages.json`과 `configs/llmwiki.yaml`에, 도구 정의는 `src/llmwiki/qa/agent.py`에 있습니다. 방법별 질문은 `configs/paper.json`에 있습니다.
