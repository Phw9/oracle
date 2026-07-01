from __future__ import annotations

from datetime import datetime

from oracle_report.compatibility_score import build_compatibility_score_payload
from oracle_report.models import BirthProfile
from oracle_report.saju.repository import ManseRepository
from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch


def test_compatibility_score_payload_contains_rule_based_scores() -> None:
    repository = ManseRepository()
    left = repository.lookup(
        BirthProfile(
            name="left",
            birth_datetime=datetime(1995, 3, 15, 14, 30),
            gender="남성",
        ),
    )
    right = repository.lookup(
        BirthProfile(
            name="right",
            birth_datetime=datetime(1997, 5, 20, 12, 30),
            gender="여성",
        ),
    )

    payload = build_compatibility_score_payload(
        "연인",
        left,
        right,
        _matches(
            {
                "third_balance_error": 0.03,
                "eye_width_ratio": 0.23,
                "mouth_width_ratio": 0.43,
                "nose_width_ratio": 0.18,
                "jaw_width_ratio": 0.70,
            },
        ),
        _matches(
            {
                "third_balance_error": 0.05,
                "eye_width_ratio": 0.20,
                "mouth_width_ratio": 0.37,
                "nose_width_ratio": 0.16,
                "jaw_width_ratio": 0.66,
            },
        ),
    )

    assert 65 <= payload["compatibility_score"] <= 96
    assert 60 <= payload["compatibility_saju_score"] <= 96
    assert 60 <= payload["compatibility_face_score"] <= 96
    assert 0 <= payload["compatibility_mode_bonus"] <= 10
    assert payload["compatibility_score_label"]
    summary = str(payload["compatibility_score_summary"])
    assert any(keyword in summary for keyword in ("로맨스", "설렘", "온기", "매력"))


def test_compatibility_score_summary_depends_on_mode() -> None:
    repository = ManseRepository()
    left = repository.lookup(
        BirthProfile(
            name="left",
            birth_datetime=datetime(1995, 3, 15, 14, 30),
            gender="남성",
        ),
    )
    right = repository.lookup(
        BirthProfile(
            name="right",
            birth_datetime=datetime(1997, 5, 20, 12, 30),
            gender="여성",
        ),
    )
    left_matches = _matches({"third_balance_error": 0.02, "mouth_balance_delta": 0.01})
    right_matches = _matches({"third_balance_error": 0.04, "mouth_balance_delta": 0.02})

    friend_payload = build_compatibility_score_payload(
        "친구",
        left,
        right,
        left_matches,
        right_matches,
    )
    coworker_payload = build_compatibility_score_payload(
        "직장동료",
        left,
        right,
        left_matches,
        right_matches,
    )

    assert "친구" in str(friend_payload["compatibility_score_summary"])
    assert "업무" in str(coworker_payload["compatibility_score_summary"]) or "협업" in str(
        coworker_payload["compatibility_score_summary"],
    )
    assert (
        friend_payload["compatibility_score_summary"]
        != coworker_payload["compatibility_score_summary"]
    )


def _matches(values: dict[str, float]) -> tuple[PhysiognomyRuleMatch, ...]:
    result = tuple(
        PhysiognomyRuleMatch(
            rule_id=f"{metric}-rule",
            metric=metric,
            title=metric,
            basis="",
            tag="",
            observation="",
            interpretation="",
            value=value,
        )
        for metric, value in values.items()
    )
    return result
