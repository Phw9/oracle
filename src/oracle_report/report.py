from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_report.config import LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import BirthProfile, ReportArtifact
from oracle_report.physiognomy import (
    FaceReadingInput,
    build_face_prompt,
    format_face_quality,
)
from oracle_report.saju.engine import SajuReading, build_saju_reading, format_saju_reading


@dataclass(frozen=True)
class ReportRequest:
    birth_profile: BirthProfile
    face_input: FaceReadingInput
    output_path: Path | None


def generate_report(request: ReportRequest, llm_config: LlmConfig) -> ReportArtifact:
    saju_reading = build_saju_reading(request.birth_profile.birth_datetime)
    prompt = build_report_prompt(
        birth_profile=request.birth_profile,
        saju_reading=saju_reading,
        face_input=request.face_input,
    )
    markdown = LlamaCppChatClient(llm_config).generate(
        prompt=prompt,
        image_path=request.face_input.image_path,
    )
    output_path = request.output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    result = ReportArtifact(markdown=markdown, output_path=output_path)
    return result


def build_report_prompt(
    birth_profile: BirthProfile,
    saju_reading: SajuReading,
    face_input: FaceReadingInput,
) -> str:
    saju_text = format_saju_reading(saju_reading)
    face_prompt = build_face_prompt(face_input)
    result = f"""
당신은 사주 룰엔진 결과와 사진 보조 입력을 종합해 한국어 리포트를 작성합니다.

[작성 원칙]
- 결과는 참고용 엔터테인먼트 리포트입니다.
- 결정론적으로 단정하지 말고 "경향", "잘 맞는다", "보완하면 좋다"처럼 표현합니다.
- 의학, 투자, 법률, 채용, 합격 여부처럼 중대한 결정을 예측하지 않습니다.
- 얼굴 사진으로 신원, 나이, 성별, 민족, 건강, 직업, 경제력을 추정하지 않습니다.
- 사주 룰엔진 결과를 1차 근거로 쓰고, 관상은 사진에서 보이는 인상 보조 설명으로만 씁니다.
- 출력은 Markdown으로만 작성합니다.

[대상]
- 이름: {birth_profile.name}
- 생년월일시: {birth_profile.birth_datetime.isoformat(sep=" ")}
- 시간대: {birth_profile.timezone}

{saju_text}

{face_prompt}

[출력 포맷]
# {birth_profile.name} 님 종합 운세 리포트
## 한 줄 요약
## 사주 명식
## 오행 밸런스
## 일간 중심 성향
## 관상 보조 풀이
## 종합 흐름
## 강점
## 주의할 점
## 생활 조언
## 참고 문구
""".strip()
    return result


def build_face_analysis_prompt(
    birth_profile: BirthProfile,
    face_input: FaceReadingInput,
) -> str:
    quality_text = format_face_quality(face_input.quality)
    time_text = _birth_time_text(birth_profile)
    result = f"""
당신은 사진에서 보이는 비식별 얼굴 요소를 엔터테인먼트 관상 리포트용 보조 정보로 정리합니다.

[중요 원칙]
- 얼굴 사진으로 신원, 나이, 성별, 민족, 건강, 직업, 경제력, 실제 성격을 단정하지 않습니다.
- 외모 점수, 매력 순위, 우열 비교를 만들지 않습니다.
- 눈, 눈썹, 윤곽, 표정처럼 보이는 요소만 "경향"과 "인상"으로 표현합니다.
- 출력은 아래 형식만 사용합니다.

[대상]
- 이름: {birth_profile.name}
- 입력 성별: {birth_profile.gender or "미입력"}
- 생년월일시: {birth_profile.birth_datetime.isoformat(sep=" ")}
- 태어난 시간: {time_text}

[캡처 품질]
{quality_text}

[출력 형식]
## 관상정보
- 얼굴 인상 태그:
- 눈/눈썹 관찰:
- 윤곽/표정 관찰:
- 리포트에 넣을 보조 해석:
- 주의 문구:
""".strip()
    return result


def build_personal_final_prompt(
    birth_profile: BirthProfile,
    saju_text: str,
    face_analysis: str,
    recommendation_text: str,
) -> str:
    time_text = _birth_time_text(birth_profile)
    result = f"""
당신은 사주팔자와 관상 보조 정보를 종합해 한국어 개인 리포트를 작성합니다.

[작성 원칙]
- 결과는 참고용 엔터테인먼트 리포트입니다.
- 결정론적으로 단정하지 말고 "경향", "잘 맞는다", "보완하면 좋다"처럼 표현합니다.
- 의학, 투자, 법률, 채용, 합격 여부처럼 중대한 결정을 예측하지 않습니다.
- 얼굴 사진으로 신원, 나이, 성별, 민족, 건강, 직업, 경제력을 추정하지 않습니다.
- 사주 룰엔진 결과를 1차 근거로 쓰고, 관상은 보조 설명으로만 씁니다.
- 출력은 Markdown으로만 작성합니다.

[대상 개인정보]
- 이름: {birth_profile.name}
- 입력 성별: {birth_profile.gender or "미입력"}
- 생년월일시: {birth_profile.birth_datetime.isoformat(sep=" ")}
- 태어난 시간: {time_text}
- 시간대: {birth_profile.timezone}

[사주팔자/만세력 정보]
{saju_text}

[관상정보]
{face_analysis}

[내 관상과 궁합 좋은 얼굴 추천 후보]
{recommendation_text}

[출력 포맷]
# {birth_profile.name} 님 Oracle 종합 리포트
## 한 줄 요약
## 사주팔자 핵심
## 관상 보조 풀이
## 종합 성향
## 강점
## 주의할 점
## 생활 조언
## 내 관상과 궁합 좋은 이성 얼굴 추천
## 참고 문구
""".strip()
    return result


def build_compatibility_final_prompt(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_saju_text: str,
    right_saju_text: str,
    face_analysis: str,
) -> str:
    result = f"""
당신은 두 사람의 사주팔자와 관상 보조 정보를 종합해 한국어 궁합 리포트를 작성합니다.

[작성 원칙]
- 결과는 참고용 엔터테인먼트 리포트입니다.
- 관계를 단정하거나 실제 미래를 예언하지 않습니다.
- 얼굴 사진으로 신원, 나이, 성별, 민족, 건강, 직업, 경제력을 추정하지 않습니다.
- 사주 룰엔진 결과를 1차 근거로 쓰고, 관상은 보조 설명으로만 씁니다.
- 궁합 모드는 반드시 "{mode}" 관점으로 작성합니다.
- 출력은 Markdown으로만 작성합니다.

[첫 번째 사람]
- 이름: {left_profile.name}
- 입력 성별: {left_profile.gender or "미입력"}
- 생년월일시: {left_profile.birth_datetime.isoformat(sep=" ")}
- 태어난 시간: {_birth_time_text(left_profile)}

[두 번째 사람]
- 이름: {right_profile.name}
- 입력 성별: {right_profile.gender or "미입력"}
- 생년월일시: {right_profile.birth_datetime.isoformat(sep=" ")}
- 태어난 시간: {_birth_time_text(right_profile)}

[첫 번째 사람 사주팔자/만세력 정보]
{left_saju_text}

[두 번째 사람 사주팔자/만세력 정보]
{right_saju_text}

[두 사람 관상정보]
{face_analysis}

[출력 포맷]
# {left_profile.name} 님과 {right_profile.name} 님의 {mode} 궁합 리포트
## 한 줄 요약
## 사주 기반 상호 보완점
## 관상 보조 인상 궁합
## {mode} 관계의 강점
## {mode} 관계에서 주의할 점
## 관계를 좋게 만드는 행동 제안
## 참고 문구
""".strip()
    return result


def _birth_time_text(profile: BirthProfile) -> str:
    result = "입력됨"
    if not profile.birth_time_known:
        result = "미입력, 정오 기준으로 사주를 보조 계산함"
    return result
