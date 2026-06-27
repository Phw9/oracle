# Oracle Report 분산 추론(Distributed Inference) 가이드

본 문서는 여러 디바이스(예: Raspberry Pi, CUDA GPU 데스크톱, WSL 등)가 동일 네트워크 내에서 사주 및 관상 분석 요청을 분담하여 처리하는 **동적 큐 기반 분산 추론 및 하이브리드 로드밸런싱 아키텍처**의 작동 원리, 실행 방법, 장애 복구 메커니즘을 상세히 설명합니다.

---

## 1. 아키텍처 개요

단일 임베디드 보드(예: Raspberry Pi)에서 로컬 LLM을 실행해 전체 사주/관상 리포트를 생성하면 추론 시간이 길어질 수 있습니다. Oracle Report는 이를 해결하기 위해 **동적 작업 큐(Dynamic Task Queue) 기반 분산 처리 아키텍처**를 제공합니다.

```
                    ┌──────────────┐
                    │   사용자 웹   │
                    └──────┬───────┘
                           │ 1. 입력 & 촬영 시작
                           ▼
                    ┌──────────────┐
                    │    마스터    │ (예: Raspberry Pi 4 - 웹 서버 및 명식 계산)
                    └──────┬───────┘
                           │ 2. 작업 큐 적재 (메타데이터 + 6개 카테고리 태스크)
                           ▼
                    ┌──────────────┐
                    │  작업 큐  │ (queue.Queue)
                    └─┬──────────┬─┘
       3-1. 선착순      │          │ 3-2. 선착순
       작업 가져감      ▼          ▼      작업 가져감
                ┌──────────┐  ┌──────────┐
                │ 슬레이브1 │  │ 슬레이브2 │ (예: CUDA 지원 PC 또는 WSL 기기)
                └──────────┘  └──────────┘
```

### 1.1. 동작 원리 및 역할 정의
1. **마스터 노드(Master)**: 사용자 입력을 받아 사주 명식(만세력)과 얼굴 이미지 품질을 연산하고, 전체 프롬프트를 메타데이터 생성용과 6개 카테고리별 개별 프롬프트로 쪼개어 공용 **작업 큐(Task Queue)**에 넣습니다.
2. **슬레이브 노드(Slave)**: 마스터가 분배한 카테고리별 개별 프롬프트를 수신하여 자체 로컬 LLM 엔진을 통해 비동기로 추론을 수행하고 결과를 반환합니다.
3. **하이브리드 노드(Hybrid)**: 평소에는 독자적인 마스터(웹 UI 서버) 역할을 하지만, 본인의 연산 자원이 유휴 상태(`idle`)일 때는 네트워크 내 다른 기기의 슬레이브 역할을 수행하는 대칭형 협력(P2P) 노드입니다.

---

## 2. 분산 처리 CLI 옵션 및 환경 변수

`./run.sh` 실행 시 CLI 인자로 전달하거나, `.env` 파일의 환경 변수를 통해 제어합니다.

| CLI 옵션 | 환경 변수명 | 값 형식 | 설명 |
| :--- | :--- | :--- | :--- |
| `--distributed-role` | `ORACLE_DISTRIBUTED_ROLE` | `master`, `slave`, `hybrid` | 현재 노드의 역할을 지정합니다. |
| `--distributed-split` | `ORACLE_DISTRIBUTED_SPLIT` | `1` (CLI 입력 시 자동 활성화) | 프롬프트를 쪼개어 분산 처리할지 여부입니다. |
| `--distributed-warmup`| `ORACLE_DISTRIBUTED_WARMUP` | `1` (CLI 입력 시 자동 활성화) | 기동 시 백그라운드로 웜업용 더미 추론을 실행하여 각 노드의 LLM KV 캐시를 미리 생성합니다. |
| `--reasoning` | `ORACLE_REASONING` | `1` (CLI 입력 시 자동 활성화) | 사고모드(Reasoning Mode)를 켭니다. 켜지면 최대 답변 한도가 자동으로 `8192` 토큰으로 크게 늘어납니다. |
| `--master-addr` | `ORACLE_MASTER_ADDR` | `http://<IP>:<Port>` | 마스터 노드의 웹 서비스 주소입니다. |
| `--slave-addrs` | `ORACLE_SLAVE_ADDRS` | `http://<IP1>:<Port>,http://<IP2>:<Port>` | 작업을 분담할 슬레이브들의 웹 API 주소 목록입니다. |
| `--debug` | `ORACLE_APP_DEBUG` | `1` (CLI 입력 시 자동 활성화) | 디버그 로그를 활성화하여 전송/수신된 상세 프롬프트 내용을 STDOUT에 기록합니다. |

---

## 3. 세부 실행 시나리오 (3개 노드 혼재 예시)

공유기 내부망(IP 대역 `192.168.0.x`)에 다음과 같은 3개의 장비가 연결되어 작동하는 시나리오입니다.
* **마스터 Node (Pi 4)**: `192.168.0.5` (카메라 및 메인 웹서버 구동)
* **하이브리드 Node (WSL)**: `192.168.0.13` (CUDA GPU 지원 데스크톱)
* **슬레이브 Node (Pi 5)**: `192.168.0.11` (보조 임베디드 보드)

### Step 1. 슬레이브 및 하이브리드 노드 기동
```bash
# 슬레이브 Pi 5 (192.168.0.11)에서 실행
./run.sh --distributed-role slave --master-addr http://192.168.0.5:8501

# 하이브리드 WSL PC (192.168.0.13)에서 실행 (본인도 마스터 웹서버를 돌리되, 유휴 시 A의 연산을 보조)
./run.sh --distributed-role hybrid --master-addr http://192.168.0.5:8501
```

### Step 2. 마스터 노드 기동
```bash
# 마스터 Pi 4 (192.168.0.5)에서 실행 (원격 슬레이브와 하이브리드, 그리고 본인 로컬을 작업 대상으로 설정)
./run.sh --distributed-role master --distributed-split --distributed-warmup --slave-addrs http://192.168.0.13:8501,http://192.168.0.11:8501,http://localhost:8501
```

---

## 4. 부하분산 및 품질-속도 제어 메커니즘 (Load Balancing)

### 4.1. 캐시 안전 자가 벤치마크 (Cache-Safe Auto-Profiling)
기기별 실시간 연산 속도(TPS)를 실측하되, **진짜 사용자의 실제 요청을 처리하기 위해 보존 중인 추론 엔진의 프롬프트 캐시(KV Cache) 슬롯을 더럽히지 않도록(Evict/Evaporation 방지)** 설계되었습니다.

* **동작 원리**:
  1. 기동 시점 또는 첫 상태 요청 시 로컬 LLM을 상대로 1회성 초단기 자가 측정을 실행합니다.
  2. 시스템 프롬프트(`prefix`)를 완전히 생략한 아주 콤팩트한 5토큰짜리 단일 `user` 더미 프롬프트(`"1"`)만 전송합니다.
  3. llama.cpp API 호출 페이로드에 **`"cache_prompt": false`**를 명시하여 추론 엔진의 캐시 테이블 수정을 원천적으로 차단합니다.
  4. 측정된 지연 시간과 반환 토큰 수(`completion_tokens`)를 통해 기기의 순수 연산 속도(TPS)를 도출하고 메모리에 캐싱합니다.
  * *폴백(Fallback)*: 만약 로컬 LLM 서버 미기동 등의 이유로 벤치마크가 실패하면, 기본 속도 `1.0 TPS`(기본 성능 스코어 `2.0`)를 임시 부여해 서버 중단을 막고 실시간 요청 수행 시의 TPS 데이터를 통해 자연스럽게 보정되도록 유도합니다.

### 4.2. 연산 체급 점수 (Compute Score) 정의
단순한 물리 속도(TPS)에 모델의 파라미터 크기(Billion 단위)를 곱하여, **더 똑똑하고 무거운 모델을 돌리는 기기일수록 더 높은 체급 점수**를 부여받도록 설계했습니다.

$$\text{Compute Score} = \text{TPS (초당 토큰 수)} \times \text{Model Parameter Size (B)}$$

* **모델 크기(Billion) 유추**:
  * 슬레이브는 본인이 돌리고 있는 모델 파일의 이름(예: `gemma-4-E2B-it`, `gemma-3-1b-it` 등)을 분석하여 파라미터 크기를 자동으로 판별합니다.
  * 예: 파일명에 `1b` 포함 시 1.0, `2b` 포함 시 2.0, `e2b/gemma-4` 포함 시 9.0 등
* **예시 비교**:
  * **WSL2 노드 (Gemma 4 9B)**: TPS는 15 Token/sec로 측정되지만, 체급 점수는 $15 \times 9 = \mathbf{135}$점.
  * **라즈베리파이 노드 (Gemma 3 1.5B)**: TPS는 30 Token/sec로 더 빠르지만, 체급 점수는 $30 \times 1.5 = \mathbf{45}$점.
* 마스터 노드는 각 기기의 헬스 체크 API `/api/distributed/status` 로부터 이 `compute_score`를 상시 수집하여 가중치 맵을 업데이트합니다.

### 4.3. 중요도 기반 태스크 라우팅 (Task-to-Model Routing)
사주 해석 카테고리 중 문장력과 추론 능력이 극도로 중요한 핵심 카테고리는 고품질 모델을 탑재한 노드가 전담하도록 제약 조건을 부여합니다.

* **필터링 규칙**:
  * **핵심 카테고리**: `종합 형국`, `타고난 성향과 심리 패턴`, `총평 및 인생의 조언`
  * **라우팅 제한**: `compute_score`가 **`20.0`** 미만인 기기(저성능 기기 또는 경량 모델 탑재 노드)가 태스크 큐에서 핵심 카테고리 작업을 가져갔을 경우, 작업을 즉시 큐에 반납(`task_queue.put(task)`)하고 대기합니다.
  * **데드락(Lock) 방지**: 네트워크 내에 `compute_score >= 20.0`인 고성능 노드가 단 하나도 활성화되어 있지 않을 경우(예: 라즈베리파이들로만 분산망이 구성된 경우)에는 저스펙 기기도 막힘없이 모든 작업을 평소처럼 처리합니다.

---

## 5. 결함 감지 및 Busy 상태 자율 회피 (Failover & Backoff)

### 5.1. 태스크 재시도 및 오프라인 격리
* **태스크 재시도**: 요청 도중 통신 장애나 연산 에러가 발생하면 해당 태스크를 다시 큐에 집어넣어 **최대 3회까지 재시도**를 수행하며, 건강한 다른 기기가 작업을 대신 가져가 완성하도록 보장합니다.
* **오프라인 기기 격리**: 특정 기기가 완전히 다운되었을 경우, 해당 기기에서 연속으로 3회 에러가 나면 마스터는 해당 워커 스레드를 **강제 종료(Offline 처리)**하여 더 이상 죽은 기기가 큐에서 작업을 꺼내 실패하는 낭비를 차단합니다.
* **로컬 루프백 우회(Bypass)**: 마스터 기기가 자기 자신(`localhost` / `127.0.0.1` / 마스터 IP)을 슬레이브 주소로 할당받아 연산할 때는 HTTP 통신을 하지 않고, 내부적으로 **로컬 `LlamaCppChatClient` 호출을 다이렉트로 매핑**하여 네트워크 통신 오버헤드와 소켓 고갈로 인한 스레드 데드락을 원천 차단합니다.

### 5.2. Busy 상태 자율 회피 및 재조회 메커니즘
하이브리드 디바이스 혹은 슬레이브 기기가 다른 연산을 처리 중이어서 바쁜 경우(`busy` 상태), 시스템의 병목을 방지하기 위해 다음과 같은 동적 우회 처리가 수행됩니다.

1. **태스크 즉시 반납**:
   * 전담 워커 스레드가 큐에서 작업을 하나 인출한 후 타겟 슬레이브의 상태가 `busy`임을 확인하면, 해당 작업을 쥐고 대기하지 않고 **즉시 공용 태스크 큐로 다시 밀어 넣습니다 (`task_queue.put(task)`)**.
   * 이로 인해, 마스터 자신이나 다른 한가한 하이브리드 노드들이 대기 시간 없이 즉시 해당 작업을 가져가 처리(대신 추론)할 수 있는 기회가 제공됩니다.
2. **백오프 대기 (Backoff Delay)**:
   * 작업을 반납한 워커 스레드는 슬레이브 기기가 한가해질 때까지 연속적인 호출을 보내지 않고, **정확히 1.0초간 휴식(`time.sleep(1.0)`)**하며 슬레이브 기기에 가해지는 오버헤드를 방지합니다.
3. **재조회 주기**:
   * 1.0초간의 대기 시간이 끝나면 워커 스레드는 다시 큐에서 다음 대기 작업을 가져와 슬레이브의 상태를 재조회(Polling)하여 통신을 재조율합니다.

---

## 6. 핵심 구현 소스 코드 레퍼런스

### 6.1. 캐시 안전 벤치마크 및 체급 점수 산출 (`llm.py`)
```python
# LlamaCppChatClient 내부
def get_or_measure_tps(self) -> float:
    with self._tps_lock:
        if self._measured_tps is not None:
            return self._measured_tps

    # 캐시를 절대 건드리지 않도록 cache_prompt: False 지정 및 시스템 프롬프트 미포함
    payload = {
        "model": self._config.model,
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 5,
        "temperature": 0.1,
        "stream": False,
        "cache_prompt": False
    }
    # ... http post로 호출 후 TPS 측정 및 캐싱
```

### 6.2. 작업 기여도 라우팅 및 자율 반납 (`workflow.py`)
```python
# worker_loop 내부 태스크 인출 및 검사부
compute_score = status_data.get("compute_score", 5.0)

# 핵심 카테고리 검사
is_core_category = cat in ("종합 형국", "타고난 성향과 심리 패턴", "총평 및 인생의 조언")
if is_core_category and compute_score < 20.0:
    # 네트워크상에 핵심 카테고리를 대신 처리할 고성능 노드가 있는지 검사
    other_high_perf_exists = any(
        meta.get("compute_score", 0.0) >= 20.0 
        for meta in scheduler.slave_metadata.values()
    )
    if other_high_perf_exists:
        task_queue.put(task) # 큐에 반납
        task_queue.task_done()
        time.sleep(0.5)
        continue
```
