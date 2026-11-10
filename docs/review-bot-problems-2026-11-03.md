# 리뷰봇 문제사항 (2026-11-03)

## 요약

현재 리뷰봇은 변경 파일과 참고 문맥 파일을 구분하지 않고, AST 기반 의존성 탐색도 입력된 파일 묶음 안에서만 제한적으로 수행한다. 이 방식으로 실행한 LGU-CTO/seulgi-server PR #1085 리뷰는 후속 원문 확인 단계에서 모델 입력 한도를 넘겨 실패했으며, 완료된 리뷰 결과가 아니다.

## 검증 범위와 결과

- 검토한 리뷰봇 체크아웃의 Git tree는 로컬 `origin/main`과 일치했다.
- PR #1085의 base는 `98b01470e235528bc808ae58c6df144e4ad8f94a`, head는 `c71581f4dc97fb6868dc2c843f85f9348d528153`였고, 확인 시 PR은 열려 있었다.
- Mac에서 리뷰 엔진의 `intent` 경로를 실행했다. 이 수동 실행은 GitHub CLI 사용자 인증으로 PR을 읽었고, GitHub App 웹훅·worker 경로와 아키텍처 A 리뷰는 실행하지 않았다.
- TEV는 `intent_change`와 `means_fit` 관점을 선택했다. 127개 구문 단위가 4페이지·8그룹으로 계획됐다. 총 22회 planning 호출과 9회 review 호출, 약 23분이 소요됐다.
- 최종 상태는 `failed/native_request_exceeds_model_limits`다. source recheck 요청의 입력 측정값이 319,795토큰으로 설정된 229,466토큰 한도를 넘었다. `findings=0`, `questions=0`은 정상 완료된 무지적 판정이 아니다.
- 전체 pytest 결과는 `3648 passed, 4 failed, 1 warning`이다. 실행 시간은 350.47초였다.
- 경고는 Starlette `TestClient`의 `httpx` 사용이 deprecated됐다는 안내로, 네 테스트 실패 원인은 아니며 의존성 갱신 시 확인할 항목이다.
- 리뷰 댓글은 게시하지 않았고 소스 코드는 수정하지 않았다. 실제 GitHub App 웹훅 전달과 운영 worker 실행은 이 검증에 포함되지 않았다.

## 확인된 문제

### 1. 리뷰 대상과 참고 문맥이 같은 경로 목록에 들어간다

worker는 Java 관련 PR의 빌드 manifest를 포함해 분석 자료를 수집한다. 이어 수집된 모든 파일 경로를 `related_paths`로 넘기고, `build_intent_task_input`은 변경 파일과 `related_paths`의 합집합을 `reviewed_paths`로 저장한다. [worker.py](../src/review_bot/runtime/worker.py#L503) [intent_context.py](../src/review_bot/application/intent_context.py#L476)

#1085의 변경 파일은 9개였지만 실제 입력의 리뷰 경로는 10개였고, 변경되지 않은 `build.gradle`이 포함됐다. 빌드 manifest는 Java 모듈의 Gradle 여부를 판단하는 문맥 자료여야 한다. 문맥과 변경 대상을 분리하지 않으면 미변경 파일이 리뷰 대상으로 보이고, 범위와 입력량도 불필요하게 늘어난다.

**필요한 조치:** 변경된 파일·심볼을 리뷰 대상으로 유지하고, manifest와 의존 파일은 별도의 참고 문맥으로 전달하는 입력 계약이 필요하다.

### 2. AST 탐색은 저장소 전체의 의존 코드를 찾지 않는다

현재 `intent_source_scope`는 Python·Java Tree-sitter 구문 트리에서 선언과 식별자 이름을 수집한다. 이름으로 후보를 찾으며, 같은 버전의 고정 입력 corpus 안에서 유일한 후보만 따라간다. 코드는 이를 컴파일러가 해석한 의존성이 아니라 “고정 corpus 안의 구문 후보”로 명시한다. 깊이 5, 심볼 128개, 파일 32개의 제한도 둔다. [intent_source_scope.py](../src/review_bot/application/intent_source_scope.py#L1) [탐색 알고리즘과 제한](../src/review_bot/application/intent_source_scope.py#L122)

#1085 실행의 corpus에는 변경 파일과 `build.gradle`만 있었다. 변경 코드가 참조하는 저장소 내 다른 Java 구현은 입력에 없었고, 실행 scope에는 `dependency_expansion_limit` 경계가 기록됐다. 따라서 현재 결과로 “기존 의존 코드를 찾아 종합적으로 검토했다”고 말할 수 없다.

**필요한 조치:** 변경 대상을 유지하면서 저장소에서 의존 문맥을 별도로 수집하고, 찾지 못했거나 제한에 걸린 범위를 결과에 명시해야 한다. 식별자 이름 일치만으로 컴파일러 수준 의존성 탐색을 대신한다고 표현해서는 안 된다.

### 3. 분할 후 최종 재확인 입력이 모델 한도를 넘는다

#1085에서는 계획과 분할 리뷰가 진행됐지만, 후속 source recheck 요청이 229,466토큰 입력 한도를 넘어 최종 상태가 실패했다. 여러 분할 요청이 완료돼도 마지막 확인 요청을 만들 수 없으면 전체 리뷰를 완료로 취급할 수 없다.

**필요한 조치:** 최종 종합과 source recheck까지 포함해 실제 모델 요청 크기를 사전에 측정하고, 재확인 요청도 제한된 범위로 나눌 수 있어야 한다. 요청이 한도를 넘으면 무지적 결과로 바꾸지 말고 실패 또는 부분 검토로 남겨야 한다.

### 4. 전체 테스트 스위트가 현재 통과하지 않는다

네 실패는 모두 `test_partition_global_review.py`에서 model-facing 재검토 입력의 인용 ID가 기존 `Q…` 값 그대로일 것이라고 기대한 assertion이다. 현재 구현은 모델 경계에서 ID를 `R…` 형식 별칭으로 바꾸고 응답을 애플리케이션 검증 전에 원래 ID로 복원한다. 별칭 전용 단위 테스트는 통과했지만, 네 기존 검증은 새 wire 계약을 반영하지 않아 전체 suite가 실패한다. [실패 assertion 예시](../tests/unit/test_partition_global_review.py#L69) [별칭 적용](../src/review_bot/application/partition_review.py#L782) [응답 복원](../src/review_bot/application/partition_review.py#L921)

**필요한 조치:** 테스트는 모델 wire의 별칭 사용과 내부 원본 ID 복원을 각각 검증하도록 갱신하고, 전체 suite를 다시 통과시켜야 한다. 현재는 테스트 게이트가 빨간 상태다.

### 5. 자동 PR 감지는 GitHub App 웹훅 운영에 의존한다

현재 서비스는 `/webhook` 수신기와 별도 worker를 사용하고, GitHub App 인증으로 작업을 처리한다. 코드에서 `gh pr list`를 주기적으로 조회하는 폴링 경로는 확인되지 않았다. 따라서 GitHub App을 대상 조직 저장소에 설치하고 이벤트를 웹훅으로 전달하는 운영 조건이 막히면, 이 저장소에는 이를 대신하는 내장 자동 감지 경로가 없다. [웹훅 운영 문서](intent-api-deployment.md#서명-웹훅-운영) [webhook 서비스](../src/review_bot/runtime/service.py#L221) [worker 서비스](../src/review_bot/runtime/service.py#L236)

**필요한 조치:** 조직에서 App 설치 권한을 확보하거나, 계정 토큰 기반 폴링 입력 경로를 별도 설계해야 한다. 현재 `gh` CLI 인증만으로 자동 리뷰가 동작한다고 볼 수 없다.

## 결론

현재 리뷰봇은 #1085의 유효한 최종 리뷰를 만들지 못했다. 특히 변경 대상과 문맥의 혼합, 고정 입력 corpus 밖 의존성 미탐색, source recheck 입력 한도 초과가 실제 실행에서 확인됐다. 전체 테스트 스위트도 네 건 실패했다. 이 문제들을 해소하고 다시 실행하기 전에는 리뷰 완료나 저장소 전체 의존성 검토를 주장할 수 없다.
