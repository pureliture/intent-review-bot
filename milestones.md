# GitHub App 설치형 자동 리뷰 구현

## Goal과 승인

- goal: App 설치 및 저장소 선택 후 PR 이벤트가 A 리뷰와 단일 댓글 생성 및 갱신으로 이어지는 실행 경로를 구현하고 자동화된 통합 테스트로 검증한다.
- 계약: 사용자가 명시적으로 선택한 `agentic-execution` (2026-10-11).
- 사용자 확인: goal 승인, 전체 설계 승인, 런타임 설계 승인 및 최신 코드 구현 허락을 각각 확인했다.
- 요구사항: [requirements-a.md](../specs/20261002-github-review-bot/requirements-a.md), 이번 세션의 D1 및 D2 승인.
- 전체 설계: [design.md](../specs/20261002-github-review-bot/design.md).
- 런타임 설계: [runtime-a-draft.md](../specs/20261002-github-review-bot/runtime-a-draft.md).
- 순차 실행 계획: [.junie/plans/revise-spec-github-app-install-ux.md](.junie/plans/revise-spec-github-app-install-ux.md), 완료된 문서 개정 뒤 후속 구현.

## 유지할 경계

- 설치 기반 적격. 운영자 allowlist는 선택적 추가 제한이고, 빈 목록은 설치된 저장소 전체를 허용한다. 설치 정보 없는 PR 이벤트는 적격이 아니다.
- `github_app`의 게시 기본값은 `comment`, offline/CLI 기본값은 `artifact_only`. 운영자의 명시적 `artifact_only`는 GitHub 쓰기를 하지 않는다.
- 모델 profile 미설정은 기존 `not_run` 동작을 유지한다. Ollama 연결 재구현 및 새 공급자 선택은 범위 밖이다.
- 실제 App 설치, 외부 댓글 게시, 배포, 커밋 및 push는 수행하지 않는다.
- 정책, 데이터 취급, 신뢰, 보안 및 공개 계약 변경은 사용자 승인 없이 결정하지 않는다.

## 수직 슬라이스

| 슬라이스 | 상태 | 관찰 가능한 결과 |
| --- | --- | --- |
| 설치 저장소 첫 PR | 완료 | 서명된 `opened` 이벤트가 SQLite에 보존되고 Worker의 실제 A 파이프라인 결과로 댓글 생성 HTTP 요청까지 연결된다 |
| 후속 PR 및 게시 하향 | 완료 | PR 변경은 같은 댓글을 갱신하고 `artifact_only`는 쓰기를 하지 않으며 종료된 작업은 게시되지 않는다 |
| 실행 명령 연결 | 완료 | receiver와 Worker 명령이 구성요소를 실행하며 위 설치형 흐름의 회귀 검증을 통과한다 |

활성 슬라이스는 없다. 코드 구현 goal은 자동화 검증 범위에서 완료했다. 실제 운영 완료를 뜻하지 않는다.

## 직접 증거

- 최종 전체 검사: `rtk proxy uv run --locked pytest -q --tb=short` → 1766 passed (186.20s). Starlette deprecation warning 1개.
- 명령 연결 74개 및 기존 CLI 회귀 37개 통과. `version`, `webhook --help`, `worker --help` 종료코드 0.
- 전체 검사에서 발견한 9개 실패를 수정했다. 관련 검사군 86 passed (21.31s), 전체 재검사로 확인했다. 게시·경합·정리 검사는 유지했다.

- 기준 검사: `rtk proxy uv run --locked pytest -q tests/unit/test_config.py tests/unit/test_config_loader.py tests/unit/test_webhook_and_worker.py tests/unit/test_github_client.py tests/integration/test_webhook_hmac.py`.
- 결과: 155 passed, 1 Starlette deprecation warning, 10.42초. 기존 동작의 기준 검사이며 설치형 goal 완료 증거는 아니다.
- 관찰된 코드 차이: `ReviewBotConfig.validate_github_app_mode`는 빈 allowlist를 거부한다. 수신기에서는 빈 allowlist일 때 설치 정보 없는 PR도 통과할 수 있다. 두 변경을 함께 검증해야 한다.
- 변경 후 설정, 수신기, SQLite, Worker 및 설정 보안 관련 검사 225개 통과 (6.62초).
- 저장소 준비 검사 6개 통과 (1.12초): 실제 bare Git 읽기, 고정 대상과 SHA 검증, 일시적 인증 header, 저장된 토큰 없음.
- 인증 구현 전 4개 실패로 누락을 재현했고 구현 후 인증 및 HTTP client 검사 190개 통과 (2.96초). 실제 RSA 서명과 mock HTTP를 사용했다.

## 결정 및 변경 기록

- 후속 슬라이스 증거: 설치 flow와 intake 통합 검사 29 passed (48.46s). 게시/Worker/SQLite 회귀 검사 200 passed (28.62s).
- 같은 입력은 설치, 성공 기록, App 작성자 및 실제 본문 hash를 모두 확인한 뒤 재사용한다. 관련 없는 라벨은 분석과 댓글 쓰기를 반복하지 않는다.
- 내부 매핑 결정: 기존 `not_attempted`를 `artifact_only`에 사용한다. 미시도를 실패나 미확정으로 표기하는 대안보다 실제 동작을 정확히 보존한다. 새 enum은 없고 결과 파일이 없으면 성공하지 않는다. Worker와 기존 receipt 저장소가 소유한다.
- `draft_comments`는 기존 설정을 소비하며 새 정책을 선택하지 않는다. HTTP 쓰기 응답 미확정은 기존 `unknown`으로 보존하고 자동 재시도하지 않는다. 상태와 민감 원문 미노출을 검사했다.
- 설치 삭제/정지/저장소 제거는 서명 검증 후 해당 설치의 작업만 원자적으로 취소한다. 비공개 보존 자료 삭제 및 보존 기간은 변경하지 않는다.
- 책임 재사용: 기존 generation/lease fence와 SQLite queue. 새 polling, scheduler, registry 또는 분산 서비스는 없다. 이전 receipt는 `unknown`/run 연결 없음으로 이관돼 자동 재사용하지 않는다.
- 다음: 시작 명령과 private 상태 경로, rootless 도구 준비 조건을 연결하고 전체 회귀 검사한다.

- 첫 PR 직접 증거: `rtk proxy uv run --locked pytest -q --tb=short tests/integration/test_github_app_flow.py` → 4 passed (18.56s). 실제 SQLite, Git 객체 준비, PMD/CPD와 A 파이프라인을 사용했다. GitHub는 mock HTTP다.
- 비공개 파일 및 접수/소스 경계 검사 25 passed (1.70s). App bot 신원 및 인증 검사 252 passed (4.35s). Rootless 어댑터 검사 128 passed (2.31s), 실제 Docker는 실행하지 않았다.
- 책임 추가: 설치별 token/provider, private Git cache 및 작업별 factory, atomic 결과 파일. 기존 SQLite/Worker/A 코어/게시 코드는 재사용했다. 삭제한 런타임 책임은 없다.
- 호환 경로: 정적 token 및 고정 Worker 어댑터는 offline/테스트에 남는다. 서비스에서는 검증된 App factory만 사용한다. CLI decoder 재사용은 공유 codec 정리 전까지 유지한다. 기존 Actions cutover는 범위 밖이다.
- 테스트: 기존 필수 allowlist 가정을 D1로 교체했고 접수 적격 검사를 강화했다. 제거한 테스트는 없다. 실제 Ubuntu/image/UID, GitHub 설치 및 모델 품질은 미검증이다.
- 설계와 증거: [런타임 구현 기록](../specs/20261002-github-review-bot/runtime-a-draft.md#18-구현-진행-기록).

## 초기 준비 메모 (이력)

- `artifact_only`의 publication 상태 매핑은 승인 문서에 명시된 남은 결정이다. 임의의 매핑을 추가하지 않는다.
- 토큰 발급 및 설치 이벤트 처리의 구현 세부는 조사 중이다. 기존 비밀 유지와 권한 경계를 완화하지 않는다.
- 설계 변경 결정 및 그 동작 증거: 아직 없음.

## 완료 상태 및 미검증 범위

- 명령 조합의 소유자는 `runtime/service.py`다. `runtime/cli.py`가 호출한다. private 상태, 단일 Worker 및 digest-pinned rootless 분석을 요구한다. Host/rootful fallback은 없다.
- 기존 SQLite queue, lease/generation fence, A 코어, 게시 및 모델 경로를 재사용했다. 새 책임은 설치 token, Git 준비 및 private 결과 파일이다. 삭제한 책임과 테스트는 없다.
- 정적 token과 고정 어댑터는 offline/테스트 호환에 남는다. CLI decoder 재사용은 공유 codec 정리 전까지 남는다. 서비스는 App factory만 사용한다.
- 실제 GitHub 설치/token/댓글, Ubuntu/image/동일 UID bind 접근 및 모델 품질은 미검증이다. 신뢰할 preloaded image와 private 운영 설정은 별도 준비해야 한다.
- TTL/data-cleanup은 미구현이다. 자료 삭제/보존 기간은 사용자 정책 승인 전까지 변경하지 않는다. Actions cutover, 공개, 배포, `commit`, `push`는 수행하지 않았다.
- 다음 행동: 실제 운영 검증에는 별도 허락이 필요하므로 여기서 멈춘다. 구현 및 자동화 검증 기록은 위 완료 상태를 따른다.