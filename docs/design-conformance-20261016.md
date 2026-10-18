# 설계 준수 감사 기록

기준: `feat/llamacpp-concurrency`, `e7b1988110027148dc3cbf6049980fc0bd06966f` (2026-10-16).
판정: **부분 준수**. 실행·형식·인용 검증 성공은 요구사항 충족이나 의도 리뷰 품질의 보증이 아니다. 이후 승인한 1회 의도 리뷰의 계약은 아래 후속 기록과 별도 실행 문서에 구분한다.

## 범위와 증거 수준

- 앞선 읽기 전용 감사의 결과를 보존한다. 이 문서를 작성하며 테스트·모델·대상 코드를 다시 실행하지 않았다.
- 최신 사용자 합의, 외부 설계문서, 구현, 관련 테스트 및 보존된 PR #5 결과를 대조했다. 외부 스펙과 비공개 평가 자료는 이 저장소의 커밋에 포함되지 않는다.
- 아래 행 번호는 기준 커밋의 위치다. 이후 역할 기반 파일명 변경은 동작 수정이나 아래 항목의 해결을 뜻하지 않는다.
- **확인된 불일치**는 정적으로 확인한 구현 행동이다. **재현 필요**는 실행 검증이 필요한 위험이다. 실제 장애·비밀 유출·악용이 있었다고 주장하지 않는다.
- 모든 불일치가 발견됐다는 보장은 없다. 기존 2,946개 자동 테스트의 통과는 의미적 품질 감사나 전체 설계 준수 인증이 아니다.

## 확인된 불일치 — 미수정

### B-01 · 분석 방법론 계획이 파일 묶기로 축소됨

- 요구: Tev1이 추적할 동작 경로, 대조할 관계와 분석 방법을 계획하고 후속 리뷰에 전달한다.
- 구현: 파일 묶음·묶은 이유 중심의 응답 계약이다. 전체 입력이 들어가면 한 묶음을 요구하며, 묶은 이유도 후속 분석 행동으로 전달하지 않는다.
- 영향: PR #5에서는 계획 호출 성공이 방법론 계획의 유효성을 입증하지 못했다.
- 근거: [코드 우선 리뷰](../src/review_bot/application/code_first_intent_review.py#L309-L349), [계획 클라이언트](../src/review_bot/adapters/intent_planner.py), [구성 테스트](../tests/unit/test_code_intent_composition.py).
- 종료 조건: 계획이 구체적인 분석 행동을 정하고 실제 후속 요청에 반영되는 반례 테스트와 실행 증거가 필요하다.

### B-02 · 잠정 관계 해석을 후속 단계가 정정하지 못함

- 요구: 초기 관찰은 잠정적이며, 후속 원문 검토가 잘못된 해석을 근거와 함께 정정할 수 있어야 한다.
- 구현: 앞서 검증된 `consistent`·`conflict` 판정을 뒤집는 응답을 거부한다. 형식·인용 검증이 의미적 해석을 고정하는 결과가 된다.
- 영향: PR #5의 초기 잘못된 `conflict`가 후속 평가에 남았다.
- 근거: [코드 우선 리뷰](../src/review_bot/application/code_first_intent_review.py), [현재 관계 판정 테스트](../tests/unit/test_code_first_intent_review.py#L292-L317).
- 종료 조건: 정당한 해석 정정과 근거 없는 변경을 구분하고, 이전 해석과 정정 근거를 함께 보존해야 한다.

### B-03 · 목적 재구성·최종 평가에서 원문 재확인 경로가 없음

- 요구: 종합 중 발견한 맥락 부족을 고정 범위 안의 원문으로 확인하거나 명확히 유보한다.
- 구현: 재확인 경로는 관찰 직후에만 있다. 목적 재구성·최종 평가에서 새 확인 필요성을 발견해도 원문 요청으로 연결할 수 없다.
- 근거: [재확인 및 종합 경로](../src/review_bot/application/code_first_intent_review.py#L853-L910).
- 종료 조건: 허용된 고정 범위·예산 안의 후속 확인 또는 명시적인 부분 검토 처리가 필요하다. Tev1 재계획 루프나 무제한 탐색을 요구하는 것은 아니다.

### B-04 · 누락 정보와 완료 판정이 불일치함

- 요구: 미확인 자료·관계를 단계 사이에 보존하고, 실행 완료·범위 충분·지적 없음을 구분한다.
- 구현: 앞 단계의 누락 정보를 모두 집계하지 않아, 필요한 맥락이 없다는 기록과 최종 완료 판정이 함께 남을 수 있다.
- 근거: [완료 판정](../src/review_bot/application/code_first_intent_review.py#L853-L910), [부분 검토 테스트](../tests/unit/test_code_first_intent_review.py#L320-L365).
- 종료 조건: 계획·관찰·재확인·목적·평가의 미해결 상태를 누적하고 최종 충분성 판정에 반영해야 한다.

### B-05 · 사용자용 의도 설명과 언어 설정이 출력에 반영되지 않음

- 요구: 변경 이유와 구현의 연결을 읽기 쉽게 설명하고 `output.language`를 따른다.
- 구현: 코드 우선 댓글은 영어 문구와 내부 분석 기록 중심이다. 기본 언어 `ko`와 불일치한다.
- 근거: [댓글 생성기](../src/review_bot/adapters/markdown/code_intent_formatter.py#L37-L94), [출력 테스트](../tests/unit/test_code_intent_formatter.py).
- 종료 조건: 의도 → 구현 방식 → 검토 판단 → 필요한 질문 순의 사용자용 설명, 언어 설정 및 내부 상세 기록 분리가 필요하다.

### A-01 · 응답의 중첩 필드와 근거 연결 검증이 불충분함

- 요구: 필수 필드, 허용 신호·변화 분류와 근거가 맞는 응답만 수용한다.
- 구현: 누락 필드를 기본값으로 채우는 경로가 있어 `findings: [{}]` 같은 응답을 완료 처리할 수 있다.
- 근거: [아키텍처 리뷰 검증](../src/review_bot/application/architecture_review.py#L441-L571), [기존 테스트](../tests/unit/test_context_and_architecture_review.py#L221-L278).
- 종료 조건: 필수·추가 필드와 실제 선택된 신호/변화/인용 관계의 로컬 검증 및 malformed 응답 반례가 필요하다.

### A-02 · 모델 예외 원문이 결과와 댓글로 전달됨

- 요구: 원시 예외나 비밀 값을 결과·댓글에 기록하지 않는다.
- 구현: 모델 예외의 `str(exc)`가 `validation_errors`에 저장되고 댓글의 한계 항목으로 출력될 수 있다.
- 근거: [예외 처리](../src/review_bot/application/architecture_review.py#L725-L733), [댓글 출력](../src/review_bot/adapters/markdown/formatter.py#L257-L265).
- 종료 조건: 안전한 오류 코드·이유로 변환한다. 토큰·URL·내부 경로를 포함한 합성 예외로 비노출을 검증해야 한다. 실제 비밀 유출이 확인된 것은 아니다.

### A-03 · Java 소스 없음이 비교 단계에서 분석 실패로 바뀜

- 요구: 정상적인 분석 대상 없음과 분석 실패를 구분한다.
- 구현: 분석기의 `no_java_sources` 결과에는 `source_hash=None`이지만 비교기는 빈 인벤토리의 해시를 요구한다.
- 근거: [분석기](../src/review_bot/adapters/pmd/runner.py#L541-L560), [비교기](../src/review_bot/application/delta.py#L145-L148), [분석기 테스트](../tests/unit/test_pmd_runner.py#L174-L199).
- 종료 조건: 분석기 → 비교기의 실제 결과 연결에서 정상적인 대상 없음 상태가 유지돼야 한다.

## 실행 재현·외부 계약 확인이 필요한 후보 — 미해결

| ID | 정적으로 확인한 위험·한계 | 근거와 필요한 검증 |
| --- | --- | --- |
| R-01 | 이전 작업의 heartbeat가 2초 join 제한 후에도 SQLite 대기 중이면, 다음 작업과 공유하는 `_lease_lost`를 뒤늦게 변경할 가능성 | [Worker](../src/review_bot/runtime/worker.py#L171-L215), [join](../src/review_bot/runtime/worker.py#L456-L462), [DB 대기](../src/review_bot/adapters/sqlite/store.py#L45-L68). DB lock으로 지연된 이전 heartbeat와 다음 작업의 겹침을 재현해야 한다. 실제 경합은 미재현이다. |
| R-02 | 설치 토큰 응답의 저장소 바인딩 필드를 존재할 때만 검사 | [App 인증](../src/review_bot/adapters/github/auth.py#L200-L218). 요청 자체는 저장소 ID와 최소 권한으로 제한돼 있다. 응답 필드 생략의 공식 API 의미와 부정 사례를 확인해야 하며, 실제 과권한 토큰 발급을 확인한 것은 아니다. |
| R-03 | Git 수집에서 생략한 symlink/gitlink 경로가 수집 스냅샷의 `complete=True`와 공존할 수 있음 | [Git 읽기](../src/review_bot/adapters/git.py#L193-L194), [범위 수집](../src/review_bot/application/intent_context.py#L102-L118). 후속 B 입력 검사가 차단할 수 있지만 수집 기록 자체의 완전성 표시와 누락 이유를 연결 검증해야 한다. |
| R-04 | 댓글 쓰기 직후 최신 상태 조회가 실패하면 receipt는 실패지만 댓글에는 경고가 남지 않을 수 있음 | [게시 후 검사](../src/review_bot/application/publication.py#L301-L306), [실패 처리](../src/review_bot/application/publication.py#L398-L405), [기존 테스트](../tests/unit/test_guarded_publication.py#L227-L244). 최신 상태를 확인하지 못한 게시물의 표시·복구 계약을 정해야 한다. 네트워크 장애 중 경고 쓰기도 항상 가능하다는 보장은 없다. |

## 품질 검증 공백

- PR #5의 인용은 고정 원문과 일치했지만, 해결된 과거 문제를 현재 결함으로 분류했다. 프로그램의 `completed/validated`는 이 오탐을 검출하지 못했다.
- 의도는 변경 항목의 반복이 아니라 **문제 상황 → 원인 → 수정 방식 → 유지하려는 동작**의 연결로 평가해야 한다. 추론한 목적을 개발자의 확인된 생각이나 확정 요구사항으로 승격해서는 안 된다.
- 확인한 테스트는 미리 작성된 모델 응답의 연결·형식·인용·실패 보존에 강하다. 정상 수정, 지적 없는 결과, base 문제와 head 결함의 구분, 목적 설명의 깊이와 유용한 질문을 별도 품질 기준으로 검증해야 한다.
- 사용한 모델만을 원인으로 단정하지 않는다. 새 호출 단계 추가는 별도 결정이다.

## 유지할 영역과 별도 보류

- 서명 수신, 설치 적격, 고정 Git 원문, A/B 독립 실행, SQLite 대기열·동시성, 게시 fence와 B 기본 게시 차단에는 실제 연결 근거가 있다. 위 문제를 전체 시스템이 껍데기라는 뜻으로 확대하지 않는다.
- Ubuntu 원격 슬롯 적용·운영, 불확실한 생성 슬롯 자동 복구, 보존 자료 자동 정리, 조직 배포, PR #36의 큰 입력 평가는 별도 미완료다. 기존에 명시한 보류를 새 축소 구현으로 재분류하지 않는다.
- `seulgi-server` CI 이관·대체, PAT 필수 지원, 무제한 관련 파일 탐색은 현재 요구가 아니다.

## 이번 후속 변경의 경계

- 감사 기록을 README·milestones에서 연결하고 Python 모듈을 책임 기반으로 정리한다.
- 목표 이름은 아키텍처 리뷰, 설명 기반 의도 리뷰, 코드 우선 의도 리뷰를 구분한다. 관련 모델·직접 대응 테스트·fixture도 같은 용어를 사용한다.
- A/B 표시는 기존 CLI·설정·직렬화·marker·산출물 호환 표기로 유지한다. 파일명 변경으로 위 문제를 해결했다고 표시하지 않는다.
- 후속 동작 수정의 완료는 해당 항목의 반례 재현, 수정, 관련 회귀와 요구 → 실제 출력 대조 증거를 남긴 뒤에만 기록한다.

### 역할 기반 파일명 변경 — 적용 완료

| 현재 Python 경로 | 책임 기반 목표 경로 |
| --- | --- |
| `application/review_a.py` | `application/architecture_review.py` |
| `application/review_b.py` | `application/declared_intent_review.py` |
| `application/review_b_code.py` | `application/code_first_intent_review.py` |
| `domain/models_b.py` | `domain/intent_models.py` |
| `domain/models_b_code.py` | `domain/code_first_intent_models.py` |
| `tests/unit/test_context_and_review_a.py` | `tests/unit/test_context_and_architecture_review.py` |
| `tests/unit/test_review_b.py` | `tests/unit/test_declared_intent_review.py` |
| `tests/unit/test_review_b_code.py` | `tests/unit/test_code_first_intent_review.py` |
| `tests/fixtures/review_b/` | `tests/fixtures/intent_review/` |
| `tests/fixtures/review_b_code.py` | `tests/fixtures/code_first_intent_review.py` |

IDE 확인: 이전 전용 Rename 시도는 Python 모듈을 찾지 못했다. 열린 루트 프로젝트의 유효 content root는 저장소 루트였고, 별도로 연 기능 worktree의 자동 감지는 build/review-a/.../java 생성물을 Java 모듈로 잡아 src/와 tests/의 Python roots를 구성하지 않았다. Python 지원은 설치되어 있었지만, 이 상태와 이전 Rename 오류의 직접 인과관계는 검증되지 않았다.

적용 상태: 후속 사용자 지시에 따라 10개 지정 경로를 Git-aware shell rename으로 옮기고, source/test import와 fixture 참조를 새 경로로 갱신했다. CLI·설정 식별자, 호출 계약, schema/prompt 버전, A/B 산출물·marker·게시 정책은 변경하지 않았다. 전체 회귀 rtk proxy uv run --locked --offline pytest -q --tb=short → **2946 passed, 232.70s**, 기존 Starlette deprecation warning 1건. 아래 감사 결함은 해결 상태로 바꾸지 않았다.

## 후속 합의: 초기 관점 선택과 1회 리뷰

2026-10-16 사용자는 B-01의 개선 목표를 다시 정의했다. Tev1의 검토 계획 자유 생성을 확장하는 대신, 승인된 8개 의도 리뷰 관점의 필요성을 고정 선택지에서 판정하게 한다. PR 본문과 변경을 함께 읽어 목적·구현·기대 결과의 연결을 검토하는 것이 중심이다.

[1회 리뷰 계약](intent-review-single-run.md)에 초기 선택지, PR 전체의 복수 선택, 주 리뷰 LLM의 선택 보정, 기존 원문의 재해석에 따른 판단 정정, 후속 원문 재확인, 필요한 누락만 부분 검토로 처리하는 원칙, 의도 중심의 사용자 설명을 기록했다.

새 perspectives-v1 경로를 [실행 코어](../src/review_bot/application/intent_perspective_review.py), [카탈로그](../src/review_bot/domain/intent_perspectives.py), [출력기](../src/review_bot/adapters/markdown/intent_perspective_formatter.py)에 구현했다. 검증 근거는 [단위 회귀](../tests/unit/test_intent_perspective_review.py), [일회 CLI·어댑터 연결](../tests/integration/test_intent_perspective_cli.py)과 [milestones의 이번 실행 결과](../milestones.md)를 따른다. 앞선 2946개 테스트나 브레인스토밍 문서 검사를 이번 동작 변경의 회귀 결과로 재사용하지 않는다.

이번 최종 전체 회귀 `rtk proxy uv run --locked --offline pytest -q --tb=short`는 **3002 passed, 1 warning in 241.40s**, exit 0이다. 기존 Starlette TestClient deprecation 경고 1건이며 테스트 삭제나 검증 완화 없이 실행했다.

사용자는 고정 입력 1회 리뷰를 우선하고 최신성 보강을 후순위로 지정했다. R-04는 미해결·후순위로 남긴다. 기존 code-first-v2의 감사 시점 동작, A-01~A-03, R-01~R-03 전체를 해결했다고 표시하지 않는다. 새 경로의 구조 검증이나 회귀 통과도 실제 모델의 의미적 리뷰 품질을 보증하지 않는다.

me707 사전 점검은 온라인 peer 확인 이후에도 리뷰 서버의 생성 슬롯 1개 계약 검증 실패와 선택 역할 메타데이터 시간 초과로 통과하지 못했다. 실제 PR #5 입력의 일회 CLI도 `not_run/token_count_unavailable`, exit 1이며 생성 호출은 0회다. 비공개 실패 결과와 게시 미시도 상태를 보존했고 기존 서버 설정과 검증 조건을 바꾸지 않았다. 실모델 의도 리뷰와 앞선 PR #5 오탐의 개선 여부는 미검증이다.

## 후속 정정: me707 서버별 2슬롯 (2026-10-17)

사용자는 me707의 Bonsai와 Tev1이 각각 2슬롯이어야 한다고 정정했다. 앞선 1슬롯 검사 실패는 이 요구와 맞지 않는 봇의 제약이었다. me707 클라이언트는 정확한 정수 2를 검사하고 MacBook의 별도 1슬롯 계약과 봇의 공유 HTTP 상한 2개는 보존한다. 슬롯당 컨텍스트는 Bonsai 262144, Tev1 131072이며 입력·작업 토큰 한도는 확대하지 않는다.

실제 관리자와 실행 인자를 확인한 뒤 기존 2슬롯 설정의 `tev1-llama.service`를 기동했다. Bonsai는 기존 프로세스를 유지했다. 두 역할 모두 모델·2슬롯·슬롯당 컨텍스트·템플릿 검증을 통과했다. Tev1의 자동 시작 설정은 변경하지 않았다.

실제 첫 선택 호출에서 Tev1은 JSON을 반환했으나 요구한 choice 필드와 달라 기존 검증이 거부했다. 같은 원문·질문·선택지·스키마에 OpenAI의 `response_format.json_schema` 형식을 적용한 대조 호출은 유효한 choice와 정확한 native 사용량을 반환했다. 이 근거로 공통 통신의 요청 형식을 보정하고 MacBook에 중복된 변환을 제거했다. 응답을 재작성하거나 검증을 완화하지 않았으며 초기 실패와 사용량 불명 기록도 보존한다. 최종 회귀와 실PR 결과는 milestones의 이번 후속 기록을 따른다.

최종 전체 회귀는 **3002 passed, 1 warning in 241.87s**, exit 0이다. 실제 PR #5에서 Tev1 선택 8회는 모두 통과했지만 Bonsai 첫 리뷰 호출이 240초 제한에서 종료돼 CLI는 `failed/usage_unavailable`, exit 1이다. 최종 리뷰와 의미적 품질 개선은 미완료다. 실패 결과·사용량 불명·공유 admission의 uncertain 1건을 보존했으며, 두 모델의 슬롯이 유휴라는 표본만으로 불확실 기록을 해제하지 않았다. 앞선 1슬롯 제약은 이번 사용자 정정으로 교체됐고 R-04는 여전히 후순위다.

## 후속 구현: 상시 API·워커와 컨테이너 (2026-10-18)

사용자 승인으로 `perspectives-v1`을 기존 서명 웹훅 워커에 연결하고, 같은 SQLite 대기열에 고정 입력을 접수하는 인증 API를 추가했다. [API·운영 안내](intent-api-deployment.md), [FastAPI 접수·조회](../src/review_bot/runtime/intent_api.py), [워커](../src/review_bot/runtime/worker.py), [Compose 구성](../compose.yaml)을 따른다. API는 접수 후 202를 반환하고 리뷰는 워커가 처리한다. 컨테이너가 CLI 하위 프로세스로 리뷰를 실행하지 않는다. 새 관점 리뷰는 비공개 산출물만 허용하며 기존 A·declared-v1·code-first-v2의 게시 계약은 유지한다.

기능 커밋은 `28a1b1a359f8356a44ce413bf0b171c022e58422`다. 이번 전체 자동 회귀는 **3016 passed, 6 warnings in 77.93s**, exit 0이다. 6개 테스트 프로세스에서 기존 Starlette deprecation이 각각 보고됐다. 서명 수신·새 리뷰 연결, API 인증·입력·재시작 후 영속 전달·결과 조회·실패 보존, 기존 lease·generation fence를 확인했으며 테스트 삭제·검증 완화는 없다. 실모델 API·배포 결과는 [milestones](../milestones.md)의 이번 직접 증거와 구분해 기록한다. API와 컨테이너 추가로 감사의 품질·보안·최신성 결함을 해결했다고 표시하지 않는다.

실제 컨테이너 기동에서 드러난 tmpfs YAML 옵션 분리와 root/0600 네트워크 파일의 비특권 읽기 실패를 `169d16b`·`ba6c2d1`에서 보정했다. 최종 회귀는 **3023 passed, 6 warnings in 68.17s**, exit 0이다. me707의 web/worker는 비특권 UID·읽기 전용 루트로 실행되고 Mac의 health·작업·결과 조회가 모두 HTTP 200이다. 이미지에 설치된 Python 파일 62개가 기능 worktree와 일치한다.

실PR #5는 한 번 접수됐고 재구성 중 대기열에 보존된 뒤 attempts=1로 실행됐다. Tev1 선택 8회는 통과했지만 Bonsai 첫 리뷰가 `model_error`로 끝나 전체 결과는 `failed/usage_unavailable`이다. 서버의 동일 task는 4096토큰을 342.91초에 처리했으며, 완성된 최종 리뷰는 없다. 응답 원문의 finish_reason과 중간 추론 필드는 이번 실패 자료에 보존되지 않아 거부된 정확한 응답 필드는 확정하지 않는다. 기존 어댑터는 비정상 종료 및 중간 추론 응답을 거부한다. 이 연결 계약과 출력 예산은 별도 후속 대상으로 남기고 검증을 완화해 성공으로 바꾸지 않았다. 최종 admission은 0건이고 두 서버의 각 2슬롯은 유휴다. R-04와 기존 리뷰 품질·검증·보안 결함의 상태는 그대로다.
