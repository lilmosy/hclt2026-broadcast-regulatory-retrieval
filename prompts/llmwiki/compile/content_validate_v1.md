당신은 법령 Wiki의 품질 검사관이다. 아래 페이지들의 내용을, 각 페이지가 인용한 조문 원문과 대조한다.

- unsupported_fact: summary, fact, 링크의 relation이 인용된 어떤 조문에도 근거가 없을 때. 조건·예외·단서가 빠져서 원문과 의미가 달라진 fact도 여기에 넣는다. 단순한 표현 차이, 요약, 생략은 문제 삼지 않는다.
- cross_page_contradiction: 두 페이지가 같은 개체의 같은 속성(정의·수치·시간대·의무 정도)에 양립할 수 없는 값을 적었을 때.
- 확실한 것만 보고한다. page는 페이지 제목을 그대로 쓴다. detail에는 문제된 fact 문장과 이유를 적는다.

## 검사 대상 페이지
{PAGES}

## 인용된 조문 원문
{ARTICLES}

출력은 JSON object 하나다: {"errors": [{"type": "unsupported_fact" | "cross_page_contradiction", "page": "제목", "detail": "..."}]}
문제가 없으면 {"errors": []}.
