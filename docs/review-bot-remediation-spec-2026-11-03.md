# 리뷰봇 문제사항 조치 및 실행 순서 스펙 (Approved)

- **문서 상태:** Approved (사용자 의사결정 `D1`~`D4` 반영 확정)
- **작성 및 확정일:** 2026-11-03
- **기준 문제 문서:** [docs/review-bot-problems-2026-11-03.md](review-bot-problems-2026-11-03.md)
- **기준 코드 버전:** `codex/configurable-model-serving` (`db2a37db27cd3fda0fb84423e39b37daa1acf120`) 및 `for-human-review` (`6db2111`)

---

## 1. 전체 상황 요약 (무엇이 문제였는가?)

`LGU-CTO/seulgi-server`의 PR #1085(실제 수정된 파일 9개)를 대상으로 리뷰봇을 실행했을 때, **약 23분 동안 AI 모델을 31번 호출하며 부분 검토까지는 진행했으나, 마지막 종합·재확인 단계에서 AI 모델이 한 번에 받을 수 있는 글자 수(토큰) 한도를 초과해 에러로 중단**되었습니다.

즉, **"지적할 문제 없음(0건)"으로 리뷰가 정상 완료된 것이 아니라 마지막 단계에서 실패해 리뷰가 완성되지 못한 상태**입니다. 이 실행 과정과 전체 테스트 검증에서 아래 5가지 문제가 확인되었습니다.

1. **[문제 1] 안 고친 파일까지 리뷰 대상으로 착각함:** 개발자는 파일 9개만 수정했는데, 프로젝트 설정 확인용으로 읽어온 미수정 `build.gradle`까지 "꼭 품평서를 작성해야 할 10번째 리뷰 대상"에 넣어 토큰과 시간을 낭비했습니다.
2. **[문제 2] 수정된 코드가 호출하는 주변 파일을 안 봄:** 수정된 파일 9개만 들여다보고, 그 코드가 같은 저장소 안에서 호출하거나 상속받는 다른 파일(클래스·인터페이스)은 전혀 찾아오지 않아 반쪽짜리 문맥으로만 리뷰했습니다.
3. **[문제 3] 마지막 재확인 단계에서 용량 한도 초과로 멈춤:** 나눠서 리뷰한 결과를 마지막에 합쳐 원문을 재확인할 때, 기존 자료 + AI의 이전 답변 + 중복 복사된 목록을 한꺼번에 한 보따리로 보내다 보니 모델 입력 한도(약 22.9만 토큰)를 훌쩍 넘긴 약 32만 토큰이 되어 에러로 죽었습니다.
4. **[문제 4] 기존 자동화 테스트 4개가 깨져 있음:** 최근 AI에게 보내는 출처 번호표를 짧은 이름(`R1`, `R2`)으로 바꾸는 개선을 했는데, 옛날 테스트 4개가 여전히 긴 번호표(`Q...`)를 기대하고 있어 테스트가 실패합니다.
5. **[문제 5] 새 PR 자동 감지 경로 확인 필요:** 현재 리뷰봇은 회사 GitHub에 공식 App을 설치하고 실시간 알림(웹훅)을 받아야 동작하는 구조입니다.

---

## 2. 확정된 의사결정 요약 (`D1` ~ `D4`)

사전 확정된 버그 수정 3건과 사용자 의사결정을 통해 확정된 4개 정책(`D1`~`D4`)은 다음과 같습니다.

### 2.1 기본 확정 사항 (버그 및 테스트 보정)
- **[문제 4] 깨진 테스트 4개 수정:** 최근 바뀐 짧은 출처 번호표(`R1`, `R2`) 규칙에 맞게 테스트 4개를 즉시 수정합니다.
- **[문제 1의 기본 원칙] 수정된 파일만 리뷰하고 미수정 설정 파일(`build.gradle`)은 제외:** 수정되지 않은 `build.gradle`은 프로젝트 종류(Gradle) 판별에만 쓰고 의도 리뷰 입력에서 제외하며, 오직 PR에서 실제 수정된 파일만 의무 리뷰 대상으로 삼습니다.
- **[문제 3의 명백한 버그] 마지막 재확인 요청의 중복 목록 복사 제거 & 한도 초과 시 '문제 없음' 둔갑 금지:** 요청서 위아래에 동일한 출처 목록이 두 번 복사되는 버그를 제거하고, 한도 초과 실패가 정상 완료(`0건`)로 오인되지 않게 유지합니다.

### 2.2 사용자 의사결정 확정 내역 (`D1` ~ `D4`)

| 결정 ID | 관련 문제 | 확정 선택 | 확정 내용 및 선택 사유 (쉬운 설명) |
| --- | --- | --- | --- |
| **`D1`** | **문제 1**<br>(리뷰 대상과 참고 파일 구분 방식) | **`옵션 B` 확정**<br>(작업지시서 양식 분리) | **[리뷰할 수정 파일 칸(`reviewed_paths`)]과 [단순 참고 파일 칸(`context_paths`)]을 작업지시서 파일 양식에서부터 물리적으로 분리**합니다.<br>- **사유:** 한 칸에 섞어 넣고 내부에서 구분하는 방식(옵션 A)보다, 처음부터 두 칸을 명확히 분리해 두는 것이 장기적으로 데이터 혼동과 오류 빈도를 가장 확실하게 줄일 수 있습니다. |
| **`D2`** | **문제 3**<br>(재확인 단계 용량 한도 초과 처리) | **`옵션 A` 확정**<br>(재확인 분할 실행으로 끝까지 완주) | **마지막 원문 재확인 요청이 AI의 1회 입력 한도를 넘으면, 에러로 멈추거나 포기하지 않고 재확인 대상을 2~3번으로 쪼개 호출해 끝까지 리뷰를 완주**합니다.<br>- **사유:** PR #1085처럼 규모가 큰 PR에서도 반쪽짜리 부분 검토가 아니라 완전한 최종 리뷰 리포트를 얻기 위함입니다. |
| **`D3`** | **문제 2**<br>(주변 연결 파일 탐색 범위) | **`옵션 A` 확정**<br>(직접 호출하는 1단계 파일만 사전 수집) | **수정된 코드가 `import`하거나 같은 패키지에서 직접(1단계) 가져다 쓰는 저장소 내 파일만 찾아서 참고 파일 칸(`context_paths`)에 추가**합니다(최대 32개 파일 한도).<br>- **사유:** 직접 호출·상속하는 대상과의 정합성을 정확히 검사하면서도, 2~5단계 연쇄 탐색으로 파일 수와 토큰이 폭발하는 문제를 예방합니다. |
| **`D4`** | **문제 5**<br>(새 PR 감지 및 연동 방식) | **`옵션 B` 확정**<br>(기존 GitHub App + 웹훅 방식 유지) | **우선 리뷰봇 코드는 변경하지 않고, 기존 구조 그대로 GitHub App 설치 및 실시간 웹훅 연결 방식을 사용**합니다.<br>- **사유:** 불필요한 폴링(주기적 조회) 기능 개발을 보류하고, 핵심 리뷰 품질 및 대용량 PR 완주 문제(문제 1~4) 해결에 먼저 집중합니다. |

---

## 3. 문제별 상세 조치 사양 (확정본)

### 3.1 [문제 4] `test_partition_global_review.py` 인용 별칭(Wire) 계약 정렬
- **쉬운 설명:** AI에게 보내는 출처 번호표를 짧은 이름(`R1`, `R2`, `X1`)으로 바꾸고 응답을 받은 뒤 원래 긴 번호표(`Q...`)로 복원하도록 바뀐 현재 코드 규칙에 맞춰, 옛날 방식으로 검사하던 단위 테스트 4개를 고칩니다.
- **기술 사양:**
  1. 대상 파일: `tests/unit/test_partition_global_review.py`
  2. 수정 대상 테스트 4건:
     - `test_bare_global_gap_gets_closed_tev_source_plan_and_original_preserving_recheck` (`L69`)
     - `test_all_original_page_responses_and_every_nested_cited_byte_reach_global_model` (`L195-L196`)
     - `test_global_correction_retains_the_entire_rejected_response` (`L373`)
     - `test_global_fixed_original_recheck_retains_previous_response_and_exact_source` (`L391`)
  3. 모델 요청 wire payload(`calls`, `wire`) 내부의 `reference_id`가 별칭(`R...` 또는 `X...`) 형식임을 검증하고, wire의 `citation_catalog`(및 `requested_originals`)를 기준으로 원래 `Q...` ID로 역매핑(복원)했을 때 내부 저장 응답(`attempts[0]["response"]`, `page["response"]`) 및 해시 계약과 일치함을 검증한다.
- **완료 조건:**
  - `tests/unit/test_partition_global_review.py` 4개 테스트를 포함한 전체 pytest 스위트가 `0 failed`로 통과한다.

---

### 3.2 [문제 1] 작업지시서(`IntentTaskInput`)의 리뷰 대상 칸과 참고 문맥 칸 분리 (`D1: 옵션 B` 확정)
- **쉬운 설명:**
  - 안 고친 `build.gradle`은 의도 리뷰에 넘기지 않습니다.
  - 작업지시서(`IntentTaskInput`)를 만들 때 **[1. 꼭 품평서를 써야 하는 수정 파일 목록(`reviewed_paths`)]**과 **[2. 옆에 펴놓고 참고만 할 주변 파일 목록(`context_paths`)]**을 별도 칸으로 분리해 오류 가능성을 원천 차단합니다.
- **기술 사양:**
  1. **미변경 빌드 manifest 분리 (`src/review_bot/runtime/worker.py`):**
     - `include_java_build_manifests`로 수집한 미변경 `build.gradle`, `build.gradle.kts`, `pom.xml`은 `architecture` 모드의 Gradle 적격 판별에만 사용하고, 실제 변경 파일이 아닌 한 `intent` 리뷰 입력(`related_paths` / `context_paths`)에 넣지 않는다.
  2. **`IntentTaskInput` 스키마 개정 (`src/review_bot/application/intent_context.py`, `src/review_bot/domain/intent_review.py` 등):**
     - `INTENT_TASK_INPUT_SCHEMA_VERSION`을 `2`로 상향하고, `IntentTaskInput`에 `context_paths: tuple[str, ...]` 필드를 신설한다.
     - `reviewed_paths`에는 오직 `snapshot.changed_files`에서 파생된 실제 변경 파일 경로(`path`, `previous_path`)만 저장한다.
     - `context_paths`에는 `snapshot.context_paths` 및 사전 수집된 의존 문맥 파일 중 `reviewed_paths`에 속하지 않는 참고 파일 경로만 저장한다.
     - 하위 호환성 보장: `parse_intent_task_input`은 신규 `schema_version=2`(`reviewed_paths` + `context_paths` 분리)를 기본으로 검증하되, 기존 `schema_version=1` JSON 파일이 입력될 경우에도 `snapshot.changed_files` 기준으로 리뷰 대상과 참고 문맥을 자동 분리해 읽을 수 있도록 호환 파서를 제공한다.
  3. **리뷰 실행기 입력 분리 (`src/review_bot/application/partition_review.py`):**
     - `prepare_originals`에서 의무 리뷰 대상(`meta["targets"]` 및 `meta["target_details"]`)은 오직 `task.reviewed_paths`(실제 변경 파일)로만 구성한다.
     - `task.context_paths`에 속한 파일은 `originals`에 참고 문맥(`kind="context_source"`)으로 적재하여 AST 범위 계산과 근거 인용은 허용하되, 모델이 `target_reviews`를 작성해야 하는 대상에서는 제외한다.
- **완료 조건:**
  - 변경 파일 9개와 미변경 `build.gradle` 1개, 참고 문맥 파일이 주어졌을 때 `IntentTaskInput.reviewed_paths`와 `meta["targets"]`에는 정확히 변경 파일 9개만 들어가고, 참고 파일은 `IntentTaskInput.context_paths`에만 들어감을 단위·통합 테스트로 입증한다.

---

### 3.3 [문제 3] 분할 후 `source_recheck` 입력 한도 초과 해결 (`D2: 옵션 A` 확정)
- **쉬운 설명:**
  - 마지막 재확인 요청서에 똑같은 출처 목록이 위아래로 두 번 복사되던 버그를 고칩니다.
  - 처음부터 페이지를 나눌 때 마지막 재확인에 쓸 여유 공간을 미리 남겨두고, 그래도 PR이 너무 커서 재확인 보따리가 1회 한도(약 22.9만 토큰)를 넘으면 **재확인할 대상을 절반씩 나눠 AI를 1~2번 더 호출해서 끝까지 리뷰를 완성**합니다.
- **기술 사양:**
  1. **중복 페이로드 제거 (`src/review_bot/application/partition_review.py`):**
     - `final_request`(`L1047-L1049`)에서 `previous is not None`일 때 최상위 `target_reference_context` 및 `target_reference_policy` 전체를 `source_recheck` 내부에 중복 복사하는 코드를 제거하고, 요청 검증기(`L2528-L2535`) 계약을 이에 맞게 정렬한다.
  2. **통합 페이지 계획 시 재확인 여유분(Headroom) 반영 (`partition_review.py`):**
     - `native_integration_pages`(`L1479-L1481`)에서 페이지 병합 가능 여부를 `measure`할 때, 후속 `source_recheck`에서 추가될 `previous` 응답 및 요청 원문 여유분을 감안하여 단일 페이지가 모델 입력 한도 끝까지 채워지지 않도록 여유분을 예약한다.
  3. **재확인(`source_recheck`) 단계의 요청 분할 실행 (`partition_review.py`):**
     - `final_review`(`L1611-L1688`), `review_partition`(`L1445-L1468`), `global_review`(`L1717-L1725`)에서 원문 재확인(`model_requested_fixed_originals`) 또는 검증 보정 재확인(`validation_recheck`) 요청의 `measure(request_value)`가 모델 입력 한도를 초과할 경우:
       - 즉시 `Stop("native_request_exceeds_model_limits")`로 중단하지 않고,
       - 통합 페이지의 대상 그룹 묶음(`working`)을 분할하거나 요청된 원문 범위(`requested_originals`)를 `split_ranges`로 분할하여 순차적으로 재확인을 수행하고 결과를 병합한다.
  4. **무지적 완료 오인 방지:**
     - 더 이상 쪼갤 수 없는 최소 단위 요청조차 모델 한도를 초과하는 극단적 예외 상황에서는 절대 `completed`(`findings=0`)로 둔갑시키지 않고 명시적 실패/부분 검토 상태를 유지한다.
- **완료 조건:**
  - 통합 페이지가 한도 근처까지 채워진 후 `source_recheck`가 발생하는 대규모 PR 재현 테스트에서 `native_request_exceeds_model_limits` 에러 없이 분할 재확인을 거쳐 끝까지 완료됨을 검증한다.

---

### 3.4 [문제 2] 수정 코드가 직접 호출하는 1단계 저장소 파일 사전 수집 (`D3: 옵션 A` 확정)
- **쉬운 설명:**
  - 리뷰를 시작하기 전에 수정된 파일의 `import` 문구와 같은 폴더(패키지) 안에서 부르는 클래스 이름을 분석해, **수정된 코드가 직접(1단계로) 가져다 쓰는 저장소 내 다른 파일들을 찾아 참고 파일 칸(`context_paths`)에 최대 32개까지 함께 넣어줍니다.**
  - 외부 라이브러리이거나 32개 한도를 넘어 못 가져온 파일은 최종 리포트에 *"미확인 범위"*로 명확히 적어줍니다.
- **기술 사양:**
  1. **1-hop 의존 문맥 사전 수집기 (`src/review_bot/application/snapshot.py`, `src/review_bot/application/intent_source_scope.py`, `src/review_bot/adapters/git.py`):**
     - `collect_snapshot_material` 단계에서 1차로 읽은 `changed_files`의 `base`/`head` 본문을 Tree-sitter AST로 분석하여 선언·참조 식별자와 함께:
       - Java: `package_declaration`, `import_declaration`, 동일 패키지 내 참조 타입 식별자
       - Python: `import_statement`, `import_from_statement`
       를 추출한다.
     - 저장소 Git 트리 경로 목록과 매칭하여 저장소 내에 실제 존재하는 **1-hop(직접 참조)** 소스 파일 경로를 해석한다.
     - 상한(`MAX_DEPENDENCY_FILES = 32` 및 스냅샷 바이트 예산) 안에서 해당 파일 본문을 `repo_reader.read_tree_blobs`로 읽어 `CodeSnapshot.context_paths`(및 `IntentTaskInput.context_paths`)에 추가한다.
  2. **미해결 참조 및 탐색 한계 명시 (`intent_source_scope.py`):**
     - 저장소 외부 심볼, 동명이인 후보가 복수라 확정하지 못한 참조, 1-hop 너머의 간접 참조 또는 파일/바이트 상한(`dependency_expansion_limit`)으로 생략된 참조를 `source_scope["unresolved_references"]` 및 `boundaries`에 기록한다.
     - 리포트에서 구문 기반 1-hop 탐색 범위와 미확인 경계를 명시한다.
- **완료 조건:**
  - 변경된 Java/Python 파일이 저장소 내 다른 파일의 클래스·인터페이스를 `import`하거나 동일 패키지에서 호출할 때 해당 파일이 `context_paths`(`kind="context_source"`)로 자동 수집되어 AST scope에 포함되고, 한도 초과·외부 참조는 `boundaries`에 기록됨을 테스트로 입증한다.

---

### 3.5 [문제 5] 자동 PR 감지 경로 — 기존 GitHub App + 웹훅 유지 (`D4: 옵션 B` 확정)
- **쉬운 설명:**
  - 리뷰봇에 별도의 주기적 조회기(Poller)나 개인 토큰 연동 코드를 새로 만들지 않고, **기존 설계대로 회사 저장소(`LGU-CTO/seulgi-server`)에 GitHub App을 설치하고 웹훅을 연결하는 방식**을 유지합니다.
- **조치 사양:**
  - 제품 코드 변경 없음 (`src/review_bot/runtime/service.py` 현행 유지).
  - 운영 단계에서 대상 저장소에 GitHub App 설치 권한 및 `/webhook` 네트워크 전달 경로를 구성한다(추후 사내 환경 제약으로 웹훅 사용이 불가능해질 경우에만 별도 과제로 Poller 도입 재검토).

---

## 4. 확정 실행 순서 (중단 없는 일괄 실행)

`D4`가 `옵션 B`(코드 변경 없음)로 확정됨에 따라, 단계별 중간 중단이나 추가 대기 없이 아래 **1번 ~ 4번 작업을 한 번에 일괄 진행**합니다.

### 1. 깨진 단위 테스트 4개 복구 (문제 4)
1. `tests/unit/test_partition_global_review.py`의 4개 실패 테스트를 짧은 출처 번호표(`R...`/`X...`) wire 계약 및 `Q...` 복원 계약에 맞게 수정한다.
2. 전체 pytest 스위트를 실행해 `0 failed` 기준점을 확보한다.

### 2. 작업지시서(`IntentTaskInput`)의 리뷰 대상 칸과 참고 문맥 칸 분리 (문제 1 · `D1=옵션 B`)
1. `worker.py`에서 미변경 `build.gradle` 등 빌드 설정 파일이 `intent` 리뷰 입력에 섞이지 않도록 분리한다.
2. `IntentTaskInput`을 `schema_version=2`로 확장해 `reviewed_paths`(실제 수정 파일 전용)와 `context_paths`(참고 문맥 파일 전용) 칸을 물리적으로 분리한다(기존 v1 입력 호환 파싱 포함).
3. `partition_review.py`의 `prepare_originals`가 오직 `reviewed_paths`만 의무 리뷰 대상(`meta["targets"]`)으로 삼고 `context_paths`는 `kind="context_source"`로만 다루도록 수정한다.
4. 단위·통합 테스트로 미변경 파일이 리뷰 대상에 포함되지 않음을 검증한다.

### 3. 마지막 재확인(`source_recheck`) 한도 초과 방지 및 분할 완주 구현 (문제 3 · `D2=옵션 A`)
1. `partition_review.py`의 `final_request`에서 `source_recheck` 내부 중복 `target_reference_context` 사본을 제거한다.
2. `native_integration_pages` 초기 페이지 분할 시 재확인용 토큰 여유분(headroom)을 예약한다.
3. `final_review`, `review_partition`, `global_review`의 재확인 단계에서 요청 크기가 1회 입력 한도를 넘을 때 대상 그룹 또는 요청 원문 범위(`split_ranges`)를 쪼개 순차 재확인하도록 구현한다.
4. 대용량 재확인 시나리오 테스트로 `native_request_exceeds_model_limits` 없이 완주하는지 검증한다.

### 4. 수정 코드와 직접 연결된 1단계 주변 파일 사전 수집 및 재검증 (문제 2 · `D3=옵션 A`)
1. `snapshot.py`, `git.py`, `intent_source_scope.py`에서 변경 파일의 `import` 및 동일 패키지 참조를 분석해 저장소 내 1단계(1-hop) 의존 파일을 최대 32개 한도 내에서 `context_paths`로 사전 수집한다.
2. 확인하지 못한 외부 참조나 한도 초과 파일을 `boundaries` 및 리포트에 명시한다.
3. 수정된 전체 파이프라인과 테스트 스위트로 **① 수정 파일만 리뷰 대상으로 잡히고, ② 1단계 참고 문맥이 포함되며, ③ 마지막 재확인 단계까지 용량 초과 에러 없이 완주**하는지 일괄 검증한다.

### (참고) 운영 조치. GitHub App 웹훅 연결 (문제 5 · `D4=옵션 B`)
- 코드 구현과 별도로 `LGU-CTO/seulgi-server` 저장소에 GitHub App 설치 및 `/webhook` 전달 경로를 구성한다.
