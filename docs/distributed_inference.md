# Oracle Report 분산 추론(Distributed Inference) 가이드

이 문서는 여러 디바이스(예: Raspberry Pi, CUDA GPU 데스크톱 등)가 동일 네트워크 내에서 사주 및 관상 분석 요청을 분담하여 처리하는 **카테고리별 프롬프트 분할 분산 추론**의 원리와 사용 방법, 그리고 구체적인 실행 예시를 설명합니다.

---

## 1. 아키텍처 개요

단일 임베디드 보드(예: Raspberry Pi)에서 로컬 LLM을 실행해 전체 사주/관상 리포트를 생성하면 추론 시간이 길어질 수 있습니다. Oracle Report는 이를 해결하기 위해 **프롬프트 분할 분산 처리 아키텍처**를 제공합니다.

```
                  ┌──────────────┐
                  │   사용자 웹   │
                  └──────┬───────┘
                         │ 1. 입력 & 촬영 시작
                         ▼
                  ┌──────────────┐
                  │    마스터    │ (예: Raspberry Pi 4 - 웹 서버 및 명식 계산)
                  └─┬──────────┬─┘
                    │          │
    2-1. [사주 종합]│          │ 2-2. [올해 운세]
         분할 요청  │          │      분할 요청
                    ▼          ▼
             ┌──────────┐  ┌──────────┐
             │ 슬레이브1 │  │ 슬레이브2 │ (예: CUDA 지원 PC 또는 Raspberry Pi 5)
             └──────────┘  └──────────┘
```

### 동작 원리
1. **마스터 노드(Master)**는 사용자 입력을 받아 사주 명식(만세력)과 얼굴 이미지 품질을 연산합니다.
2. 마스터 노드는 전체 프롬프트를 **메타데이터 생성용**과 **카테고리 블록별 개별 프롬프트**로 쪼갭니다.
3. 마스터는 등록된 **슬레이브 노드들(Slaves)**에 개별 카테고리 분석 작업을 멀티스레딩(`ThreadPoolExecutor`)으로 **동시에 병렬 요청**합니다.
4. 각 슬레이브는 할당받은 단 하나의 카테고리만 로컬 LLM으로 빠르게 연산하여 단일 JSON 블록을 반환합니다.
5. 마스터가 취합된 결과를 최종 JSON 형태로 결합하고 템플릿에 맞추어 하나의 HTML 리포트를 완성합니다.

---

## 2. 분산 처리 CLI 옵션 및 환경 변수

`./run.sh` 실행 시 CLI 인자로 전달하거나, `.env` 파일 또는 환경 변수로 설정하여 분산 모드를 제어할 수 있습니다.

| CLI 옵션 | 환경 변수명 | 값 형식 | 설명 |
| :--- | :--- | :--- | :--- |
| `--distributed-role` | `ORACLE_DISTRIBUTED_ROLE` | `master` 또는 `slave` | 현재 노드의 역할을 설정합니다. |
| `--distributed-split` | `ORACLE_DISTRIBUTED_SPLIT` | `1` (CLI 입력 시 자동 활성화) | 프롬프트를 쪼개어 분산 처리할지 여부를 결정합니다. |
| `--master-addr` | `ORACLE_MASTER_ADDR` | `http://<IP>:<Port>` | 마스터 노드의 웹 서비스 주소를 설정합니다. |
| `--slave-addrs` | `ORACLE_SLAVE_ADDRS` | `http://<IP1>:<Port>,http://<IP2>:<Port>` | (마스터 전용) 작업을 분담할 슬레이브들의 웹 API 주소 목록입니다. |

---

## 3. 세부 실행 가이드 (3개 노드 예시)

공유기 내부망(IP 대역 `192.168.0.x`)에 다음과 같은 3개의 장비가 연결되어 있다고 가정합니다.
* **마스터 Node**: `192.168.0.5` (Raspberry Pi 4, 카메라 연결 장비)
* **슬레이브 Node 1**: `192.168.0.10` (CUDA GPU PC, 빠른 추론 가능)
* **슬레이브 Node 2**: `192.168.0.11` (Raspberry Pi 5)

### [Step 1] 슬레이브 노드 기동
각 슬레이브 디바이스에서 마스터의 분할 추론 API 요청을 받아 처리할 수 있도록 슬레이브 모드로 기동합니다. (슬레이브 내부의 llama.cpp 로컬 LLM 서버도 함께 띄워집니다.)

```bash
# 슬레이브 1번 PC (192.168.0.10)에서 실행
./run.sh --distributed-role slave --master-addr http://192.168.0.5:8501

# 슬레이브 2번 Pi 5 (192.168.0.11)에서 실행
./run.sh --distributed-role slave --master-addr http://192.168.0.5:8501
```

### [Step 2] 마스터 노드 기동
마스터 장비인 Raspberry Pi 4에서 슬레이브 목록을 기재하고 분산 병렬 연산을 켜서 실행합니다.

```bash
# 마스터 Pi 4 (192.168.0.5)에서 실행
./run.sh --distributed-role master --distributed-split --slave-addrs http://192.168.0.10:8501,http://192.168.0.11:8501
```

---

## 4. 확장성 및 로드 밸런싱 구조

현재 분산 추론 스케줄링은 `DistributedTaskScheduler` 클래스를 통해 구현되어 있습니다.

```python
class DistributedTaskScheduler:
    def __init__(self, slave_addrs: list[str]) -> None:
        self.slave_addrs = slave_addrs
        # 향후 CUDA 지원 등 슬레이브 기기의 하드웨어 정보를 수집하여
        # 가중치 기반 분산을 지원할 수 있도록 메타데이터 공간을 확보함
        self.slave_metadata = {addr: {"cuda": False, "weight": 1.0} for addr in slave_addrs}
        self._next_index = 0

    def select_slave(self, task_name: str) -> str:
        # 현재는 라운드 로빈 방식으로 순차 할당합니다.
        ...
```

### 부하 분산(Load Balancing) 확장 포인트
* **하드웨어 가중치 스케줄링**: 슬레이브의 CUDA 지원 유무 또는 평균 응답 지연 시간(Latency)을 측정한 뒤, 성능이 뛰어난 GPU 슬레이브(예: 슬레이브 1)에 더 많은 카테고리 분석 태스크를 몰아주는 가중치 큐(Weight Queue) 스케줄링 알고리즘을 `select_slave` 내부에 통합할 수 있습니다.
* **폴백 안전장치(Failover)**: 슬레이브 요청 중 하나가 네트워크 지연이나 전원 차단으로 실패하더라도 마스터 노드가 실패한 태스크를 직접 감지해 마스터의 로컬 LLM으로 대체 연산을 지시하거나 다른 가용한 슬레이브에 재할당할 수 있도록 설계되어 있습니다.
