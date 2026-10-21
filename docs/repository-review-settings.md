# 저장소별 리뷰 지시와 댓글 양식

GitHub App은 리뷰 요청을 처리할 때 **대상 저장소 기본 브랜치의 SHA를 한 번 고정**하고,
그 커밋의 `.review-bot/review-settings.json`을 읽는다. PR head나 fork의 설정은 읽지 않는다.
파일이 없으면 내장 설정을 사용한다. 변경 사항은 다음 요청부터 적용하며 열린 PR을 자동 재리뷰하지 않는다.

```json
{
  "schema_version": 1,
  "review_instructions": {
    "architecture": "측정값 자체보다 변경된 책임과 의존관계의 실제 영향을 설명한다.",
    "intent": "PR 목적과 구현의 연결을 짧게 설명하고, 확인 질문과 결함을 구분한다."
  },
  "presentation": {
    "title_a": "PMD/CPD·아키텍처 리뷰",
    "title_b": "의도 리뷰"
  },
  "comment_templates": {
    "architecture": "{{summary}}\n\n{{static_analysis}}\n\n{{llm_review}}",
    "intent": "{{summary}}\n\n{{llm_review}}\n\n{{perspectives}}"
  }
}
```

필드는 선택 사항이다. `review_instructions`의 `architecture`·`intent`는 리뷰 판단과 설명에만 적용한다.
TEV의 선택·분할, 사실 확인·재작성 지시, 응답 스키마·근거 검증·실행 권한은 앱에서 관리한다.

템플릿은 선언된 섹션을 각각 정확히 한 번 배치하는 문자열이다. Python/Jinja 실행이나 외부 파일·URL 읽기를 하지 않는다.
architecture는 `summary`, `static_analysis`, `llm_review`, intent는 `summary`, `llm_review`, `perspectives`가 필요하다.
제목·식별 마커·검토 범위·미완료 안내는 프로그램이 유지한다. 미완료 결과에는 사용자 템플릿을 적용하지 않는다.
표시 문구는 `title_a`, `title_b`, `static_section`, `llm_section`, `details_label`, `footer_notice`를 지원한다.

형식·슬롯·인코딩 등의 오류나 조회 실패가 있으면 설치·저장소별로 보존한 마지막 정상 번들로 복구한다.
이력이 없으면 내장 설정을 사용한다. 오류와 적용 버전은 실행 기록에 남기고 댓글에도 짧게 알린다.
정상적인 파일 삭제는 내장 기본값으로 돌아가는 설정 변경이다.

작업 중에는 같은 설정 번들을 유지한다. 판단 지시의 해시와 표시용 해시를 분리하여,
표시만 변경된 요청은 고정 원문·원본 결과·ID를 그대로 재검증하고 댓글을 다시 렌더링한다.
판단 지시가 바뀌면 해당 리뷰를 새로 생성한다. TEV 지시와 입력이 같으면 검증된 공통 계획을 재사용할 수 있다.

실행 예산은 운영 설정의 `execution.max_model_calls`(기본 256), `execution.max_total_tokens`(기본 8,388,608)과
기존 작업 마감·공유 admission을 따른다. 설정 파일의 크기 검사는 모델 요청의 실제 토큰 한도 검사를 대신하지 않는다.

이 문서는 구현 계약을 설명한다. 실제 모델·게시·설치 검증 결과는 해당 소스 버전의 이슈/PR 기록과 별도로 확인한다.
