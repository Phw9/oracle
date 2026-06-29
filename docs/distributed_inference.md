# Oracle Report 분산 추론 가이드

이 문서는 `release` 브랜치의 분산 추론 동작을 설명합니다.

## 기준

- UI, 개인 사주 리포트, 궁합 리포트의 입력/출력 흐름은 `main` 브랜치 구현을 따릅니다.
- `ORACLE_DISTRIBUTED_SPLIT=1`이어도 `saju_reading`과 `saju_reading_couple`은 분산 split으로 처리하지 않습니다.
- 분산 split은 관상 LLM 분석 프롬프트에만 적용합니다.
- 관상 rulebase 모드(`ORACLE_FACE_ANALYSIS_MODE=2`)에서는 관상 LLM과 분산 추론을 사용하지 않습니다.

## 지원 범위

분산 split 대상 프롬프트:

- `personal_face_analysis`
- `face_analysis_copule`

분산 처리 흐름:

1. 마스터 노드는 관상 분석 프롬프트를 메타데이터 작업과 카테고리별 작업으로 나눕니다.
2. 각 작업은 `ORACLE_SLAVE_ADDRS`에 등록된 워커 또는 로컬 워커에서 실행됩니다.
3. 결과 JSON을 `face_blocks` 또는 `pair_blocks`로 다시 조합합니다.

지원 API:

- `POST /api/distributed/generate`
- `GET /api/distributed/status`

## 환경 변수

| 변수 | 값 | 설명 |
| --- | --- | --- |
| `ORACLE_DISTRIBUTED_ROLE` | `master`, `slave`, `hybrid` | 현재 노드 역할 |
| `ORACLE_DISTRIBUTED_SPLIT` | `1` 또는 `0` | 관상 프롬프트 split 활성화 |
| `ORACLE_DISTRIBUTED_WARMUP` | `1` 또는 `0` | 관상 프롬프트 웜업 활성화 |
| `ORACLE_SLAVE_ADDRS` | URL 목록 | 쉼표로 구분한 워커 주소 |
| `ORACLE_MASTER_ADDR` | URL | 마스터 주소 |

## 실행 예

마스터:

```bash
./run.sh --distributed-role master --distributed-split --slave-addrs http://192.168.0.13:8501,http://localhost:8501
```

슬레이브:

```bash
./run.sh --distributed-role slave --master-addr http://192.168.0.5:8501
```

하이브리드:

```bash
./run.sh --distributed-role hybrid --distributed-split --slave-addrs http://localhost:8501
```

## 보존 규칙

릴리즈 브랜치에서 사주/궁합 결과 품질과 UI는 `main` 결과물을 우선합니다. 분산 기능을 켜더라도 사주/궁합 LLM 프롬프트는 기존 단일 프롬프트 경로로 생성됩니다.
