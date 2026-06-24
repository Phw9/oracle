# Prompt Editing Guide

프롬프트 담당자는 기본적으로 `configs/prompts.json`만 수정하면 됩니다.

## 수정 파일

- `face_prompt`: 사진/관상 보조 입력 블록
- `face_analysis`: 관상 분석 LLM에 들어가는 프롬프트
- `report`: 예전 단일 리포트 CLI에서 쓰는 종합 프롬프트
- `personal_final`: 개인 리포트 최종 LLM에 들어가는 프롬프트
- `compatibility_final`: 두 사람 궁합 최종 LLM에 들어가는 프롬프트

각 템플릿은 JSON 문자열 배열입니다. 배열의 각 항목은 한 줄로 합쳐집니다.

## 자리표시자

템플릿 안의 `${name}`, `${saju_text}`, `${face_analysis}` 같은 값은 실행 시 코드가 채웁니다. 자리표시자는 삭제하거나 이름을 바꾸면 실행 중 오류가 납니다.

자주 쓰는 자리표시자:

- `${name}`: 이름
- `${gender}`: 입력 성별
- `${birth_datetime}`: 생년월일시
- `${birth_time_text}`: 출생 시간 입력 여부
- `${timezone}`: 시간대
- `${quality_text}`: 캡처 품질 요약
- `${saju_text}`: 사주/만세력 결과 블록
- `${face_analysis}`: 관상 분석 결과
- `${recommendation_text}`: 추천 얼굴 정보
- `${mode}`: 궁합 모드

## 확인 명령

프롬프트만 확인:

```bash
./run.sh prompt face-analysis --name tester --birth-date 1995-03-15 --birth-time 14:30 --gender male
```

사주/만세력 입력 블록만 확인:

```bash
./run.sh prompt saju-reading --name tester --birth-date 1995-03-15 --birth-time 14:30 --gender male
```

개인 최종 프롬프트 확인:

```bash
./run.sh prompt personal-final \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시"
```

LLM 결과만 확인하려면 `prompt` 대신 `prompt-run`을 사용합니다.

```bash
./run.sh prompt-run face-analysis \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --image runs/session-001/capture.jpg
```

## 별도 템플릿 파일 사용

기본 파일 대신 다른 JSON 파일을 쓰려면 `ORACLE_PROMPTS_PATH`를 지정합니다.

```bash
ORACLE_PROMPTS_PATH=configs/prompts.local.json ./run.sh prompt face-analysis \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```
