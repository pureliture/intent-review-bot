# 의도 리뷰 API와 컨테이너 운영

## 실행 구조

Linux/me707에서 같은 이미지의 `web`과 `worker`를 별도 컨테이너로 실행한다. FastAPI는 입력을 SQLite에 저장하고 `202`를 반환한다. 워커는 기존 SQLite claim·lease·heartbeat·모델 admission을 사용해 리뷰를 실행한다. HTTP 연결이나 SSH 세션을 유지할 필요가 없다. API가 CLI 하위 프로세스를 실행하지 않는다.

`intent.pipeline: perspectives-v1`은 기존 웹훅 워커에서도 선택할 수 있다. 새 관점 리뷰는 `publish.mode: artifact_only`, `b_comments_enabled: false`를 요구하며 외부 댓글을 게시하지 않는다. 기존 declared-v1/code-first-v2 기본값과 게시 계약은 유지한다.

## 고정 입력 API

초기 운영 설정은 [deploy/intent-api.yaml](../deploy/intent-api.yaml)이다. `execution.mode: offline`은 GitHub 수집·게시를 구성하지 않는다는 뜻이며, 모델 프로필 `me707-llamacpp`는 실제 Bonsai와 Tev1을 호출한다. 테스트용 모의 모델로 바꾸는 설정이 아니다.

다음 API에는 `Authorization: Bearer <REVIEW_BOT_API_TOKEN>`이 필요하다.

| 요청 | 동작 |
| --- | --- |
| `GET /health` | 인증 없는 프로세스 생존 확인. 모델·워커·리뷰 성공을 뜻하지 않는다. |
| `POST /reviews/intent?language=ko` | 기존 CLI `--input`과 같은 전체 `IntentTaskInput` JSON을 검증·저장하고 작업 ID와 조회 URL을 반환한다. `en`도 지원한다. |
| `GET /reviews/intent/{review_id}` | queued/running/succeeded/failed 상태, 단계, 시도 수와 시각을 조회한다. |
| `GET /reviews/intent/{review_id}/result` | 저장된 schema-3 결과를 조회한다. 미완료는 409, 결과 파일 없음은 404다. 실패·부분 검토의 결과도 보존한다. |

입력 상한은 기존 `MAX_INPUT_BYTES`와 `service.webhook_max_bytes` 중 작은 값이다. 잘못된 원문·hash·schema는 422, 과대 입력은 413, 인증 실패는 401, 저장 실패는 503이다. `202`는 영속 접수 확인이며 리뷰 완료가 아니다. 같은 POST를 다시 보내면 별도 리뷰가 생기므로 받은 ID로 상태를 조회한다. 연결이 끊겨 접수 여부가 불명인 경우 재전송을 성공 복구로 간주하지 않는다.

고정 입력은 SQLite의 기존 `runs` 대기열에 `fixed_intent` 종류로 저장한다. 별도 작업 스케줄러나 프로세스 내부 임시 대기열은 없다. 웹훅의 PR 세대 병합·취소와 고정 입력을 구분하고 전체 동시 lease 상한 2를 공유한다. 운영 중 평가도 API에 제출해 같은 노드의 공유 model admission으로 관리한다. 결과는 상태 디렉터리의 `artifacts/fixed/{review_id}/attempt-{N}/B/`에 저장한다. 회수된 lease의 새 시도는 새 디렉터리를 사용하므로 이전 결과를 덮어쓰지 않는다. lease 회수는 모델 호출의 exactly-once 보장이 아니며 기존 불확실 admission은 자동 해제하지 않는다.

## Compose 준비

이 구성은 **B 의도 리뷰용 Linux host networking**을 사용한다. 기존 me707 모델 서비스를 그대로 호출하며 모델 컨테이너나 GPU 설정을 새로 만들지 않는다. A 분석용 rootless Docker의 이미지·UID·격리 준비는 별도 기존 계약이다. 이 이미지에 Docker socket을 연결해 A 분석 권한을 부여하지 않는다.

운영자가 다음 경로를 준비한다. 비밀·설정·상태는 저장소와 이미지 밖에 둔다.

- `REVIEW_BOT_CONFIG`: 운영 설정 파일의 절대 경로. 초기 예제를 복사해 사용한다.
- `REVIEW_BOT_RUNTIME_ENV`: mode 0600인 런타임 env 파일. `REVIEW_BOT_API_TOKEN`, `ME707_BONSAI_BASE_URL`, `ME707_TEV1_BASE_URL`을 저장한다. 토큰은 운영 환경에서 임의 생성한다.
- `REVIEW_BOT_STATE_PATH`: 컨테이너 실행 UID가 소유한 mode 0700 상태 디렉터리.
- `REVIEW_BOT_MODEL_STATE_PATH`: 모든 봇 워커가 공유하는 기존 모델 admission 디렉터리. mode 0700을 유지하고 불확실 기록을 우회하려고 새 경로로 바꾸지 않는다. 호스트를 옮길 때는 기존 발신자와 기록을 먼저 확인한다.
- `REVIEW_BOT_UID`, `REVIEW_BOT_GID`: 상태 소유자 ID. 기본값은 1000이다.
- `REVIEW_BOT_NETWORK_CONFIG_DIR`: 실행 UID가 소유한 mode 0700 디렉터리. 아래 준비 명령이 읽을 수 있는 비밀 없는 hosts/resolver 파일을 생성한다. 모델 서버가 실제 바인딩된 주소를 전달하며 호스트 resolver 내용을 복사한다.
- 선택적으로 `REVIEW_BOT_BIND_ADDRESS`, `REVIEW_BOT_PORT`: 기본 `127.0.0.1:18080`. me707에서 Tailscale 전용 주소를 명시하면 사설 API로 접근할 수 있다. GitHub webhook에는 별도의 운영 HTTPS 전달 경로가 필요하다.

런타임 env의 모델 URL은 기존 허용 별칭 `http://ubuntu-ms-7e07:11435/v1`, `http://ubuntu-ms-7e07:11437/v1`을 사용한다. 두 서버는 각각 정확히 2슬롯이어야 한다. Docker의 이름 해석이 해당 호스트를 찾을 수 있어야 하며 전체 Tailscale DNS 이름을 임의의 새 endpoint로 대체하지 않는다.

위 Compose 변수는 비공개 `compose.env` 파일에 저장하고 다음과 같이 실행한다.

```sh
rtk proxy python3 deploy/prepare-network.py \
  --directory /private/path/network --model-address 'LIVE_ME707_IP'
rtk proxy docker compose --env-file /private/path/compose.env config --quiet
rtk proxy docker compose --env-file /private/path/compose.env build web
rtk proxy docker compose --env-file /private/path/compose.env up -d --no-build
```

`web`과 `worker`는 같은 state/admission bind mount를 사용한다. 읽을 수 있는 비밀 없는 hosts/resolver 파일을 명시적인 읽기 전용 bind mount로 제공한다. me707에서 관측된 Docker 생성 파일의 root/0600 권한으로 비특권 프로세스의 이름 해석이 실패하는 문제를 피하며, 호스트의 DNS·방화벽 설정은 변경하지 않는다. 이미지에는 명시적으로 허용한 package·lock 파일만 복사한다. 소스 checkout, `.git`, `.junie`, `.idea`, 키, 모델, 테스트 입력과 생성 결과는 build context에서 제외한다. 컨테이너는 비특권 UID·읽기 전용 루트로 실행한다.

`LIVE_ME707_IP`는 실제 주소로 바꾼다. 네트워크 파일을 다시 생성한 뒤 기존 컨테이너에 반영하려면 진행 중인 리뷰를 마친 후 `up -d --no-build --force-recreate`를 사용한다. 파일은 atomic replace로 갱신하므로 기존 bind mount의 inode는 자동 교체되지 않는다.

현재 초기 설정은 Bonsai2 출력 상한 32,678토큰에 맞춰 호출 제한 3,600초, 전체 리뷰 작업 제한 14,400초를 사용한다. 초안·최종·선택적 재검토의 최대 세 번의 Bonsai 호출과 Tev 선택 시간을 고려한 값이다. 이전 PR #5에서는 봇의 옛 상한 4,096토큰을 342.91초에 생성한 뒤 최종 응답 검증에 실패했다. 그 단일 표본의 속도를 적용하면 32,678토큰 생성은 약 45분이지만, 입력·동시 실행에 따라 달라지며 완료를 보장하지 않는다.

Bonsai와 Tev1은 각각 슬롯당 131,072토큰, 2슬롯을 사용한다. 서버의 전체 context 인자는 각각 262,144다. 사용자가 승인한 KV 배치는 두 서버 모두 GPU, K/V는 Q4_0이며 가중치 형식은 Bonsai PQ2_0·Tev1 Q8_0이다. Bonsai 프로필은 출력을 예약해 입력을 최대 98,394토큰으로 제한한다. API의 기존 작업 누적 토큰 예산 131,072와 최종 JSON 바이트·스키마 검증도 적용된다. 출력 토큰 상한은 추론과 최종 출력을 위한 생성 예산이며 JSON 크기 제한과 다르다. 완성된 응답의 `reasoning_content` 거부를 포함한 검증 규칙은 별도 조건이므로, 예산 보정만으로 이전 실패가 해결됐다고 판정하지 않는다. MacBook LFM과 Tev 계획 역할의 출력 상한은 각각 4,096토큰이다. GPU 할당·실제 처리 속도·Tev1 선택 품질은 기동 후 별도로 확인한다.

## 서명 웹훅 운영

기존 GitHub App 운영 설정에서 `execution.mode/backend: github_app`과 필요한 App ID·서명 secret·키를 준비하면 같은 web 서비스가 `POST /webhook`을 제공한다. 키 파일은 `compose.github-app.yaml`을 추가해 읽기 전용으로 연결한다.

```sh
rtk proxy docker compose --env-file /private/path/compose.env \
  -f compose.yaml -f compose.github-app.yaml up -d --no-build
```

`REVIEW_BOT_APP_KEY_FILE`은 호스트의 비공개 PEM 절대 경로다. 컨테이너에는 `/run/secrets/github-app.pem`으로 연결한다. 기존 GitHub App 환경 변수와 파일 권한 계약을 따른다. `REVIEW_BOT_API_TOKEN`도 명시하면 같은 서버에 고정 입력 API를 추가한다. 이 경우에도 `perspectives-v1`과 비공개 출력 설정이 필요하다. 고정 입력 전용 offline 모드에는 `/webhook` 경로가 없다.

## API 검증

컨테이너 기동 → `/health` → 인증 거부 → 유효한 고정 입력 POST의 202와 작업 ID → 상태 조회 → 결과 조회를 구분해 확인한다. 성공 판정에는 `status=completed`, 원문과 일치하는 hash·SHA, 실제 모델 ID와 사용량, 비공개 결과 파일, 외부 게시 미시도를 함께 확인한다. 모의 모델 회귀와 실모델 결과를 분리해 [milestones](../milestones.md)에 기록한다.

2026-10-18 me707의 web/worker 기동과 Mac에서의 health·상태·결과 GET 200을 확인했다. 실제 PR #5는 단일 POST로 접수돼 컨테이너 재구성 후에도 보존됐고, 실제 Tev1 선택 8회 후 Bonsai 단계에서 `failed/usage_unavailable`로 끝났다. 서버는 4096토큰을 생성했지만 최종 리뷰는 없으며, 이 테스트를 리뷰 성공으로 표시하지 않는다. 현재 어댑터의 정상 종료·중간 추론 응답 거부·출력 한도와 Bonsai 기본 추론의 정합성은 후속 과제다. 외부 게시 설정은 활성화하지 않았다.

## 2026-10-18 GPU KV 전환 실측

사용자 승인 후 기존 두 systemd user unit을 백업하고, 두 모델 모두 슬롯당 131,072토큰·2슬롯·Q4_0 K/V GPU 캐시로 기동했다. Tev를 먼저 Q8 KV에서 Q4 KV로 전환한 다음 Bonsai의 CPU KV를 GPU로 옮겼다. 가중치 파일과 별칭, 사설 바인딩, Bonsai의 기존 mean-centering 파일 및 attention-rotation 설정은 보존했다.

같은 합성 입력 9,085토큰·생성 128토큰으로 Bonsai 서버의 생성 속도를 비교했다. 변경 전 12.216 TPS, 변경 후 82.231 TPS로 약 6.73배였다. 두 요청 모두 `cached_tokens=0`이다. 단일 입력·짧은 생성의 구성 전후 비교이며, 장시간 생성이나 동시 부하의 성능을 보장하지 않는다. GPU 여유는 기동 직후 2,363MiB, 생성 검사 후 2,325MiB였다.

Tev는 봇과 같은 `enable_thinking: false`, 최대 16토큰, 닫힌 A/B/C JSON 스키마로 고정 선택 문제 3건을 모두 통과했다. 이는 기본 구조화 선택 확인이며 실제 PR의 관점 선택 품질이 동등하다는 검증은 아니다. 최초 진단 요청은 추론 비활성 옵션을 누락해 64토큰에서 잘렸으며 그 실패 기록도 보존했다. Tev 선택 클라이언트의 프로필 예산은 기존 4,096토큰으로 유지해 Bonsai의 32,678토큰 예산을 상속하지 않는다. 이 전환으로 이전 실PR 최종 리뷰 실패가 해결됐다고 판정하지 않는다.
