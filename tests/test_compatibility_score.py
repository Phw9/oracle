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

    assert 62 <= payload["compatibility_score"] <= 98
    assert 55 <= payload["compatibility_saju_score"] <= 98
    assert 55 <= payload["compatibility_face_score"] <= 98
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


def test_compatibility_scores_vary_across_different_inputs() -> None:
    repository = ManseRepository()
    profiles = (
        BirthProfile(
            name="a",
            birth_datetime=datetime(1992, 1, 10, 0, 30),
            gender="남성",
        ),
        BirthProfile(
            name="b",
            birth_datetime=datetime(1995, 7, 20, 12, 30),
            gender="여성",
        ),
        BirthProfile(
            name="c",
            birth_datetime=datetime(1999, 11, 3, 18, 30),
            gender="남성",
        ),
        BirthProfile(
            name="d",
            birth_datetime=datetime(2001, 4, 12, 8, 30),
            gender="여성",
        ),
    )
    manse = tuple(repository.lookup(profile) for profile in profiles)
    face_sets = (
        _matches(
            {
                "third_balance_error": 0.02,
                "eye_width_ratio": 0.24,
                "mouth_width_ratio": 0.44,
                "nose_width_ratio": 0.19,
            },
        ),
        _matches(
            {
                "third_balance_error": 0.09,
                "eye_width_ratio": 0.18,
                "mouth_width_ratio": 0.33,
                "nose_width_ratio": 0.15,
            },
        ),
        _matches(
            {
                "third_balance_error": 0.04,
                "eye_width_ratio": 0.21,
                "mouth_width_ratio": 0.39,
                "nose_width_ratio": 0.23,
            },
        ),
        _matches(
            {
                "third_balance_error": 0.11,
                "eye_width_ratio": 0.17,
                "mouth_width_ratio": 0.47,
                "nose_width_ratio": 0.17,
            },
        ),
    )

    payloads = (
        build_compatibility_score_payload("연인", manse[0], manse[1], face_sets[0], face_sets[1]),
        build_compatibility_score_payload("친구", manse[0], manse[2], face_sets[0], face_sets[2]),
        build_compatibility_score_payload("직장동료", manse[1], manse[3], face_sets[1], face_sets[3]),
        build_compatibility_score_payload("연인", manse[2], manse[3], face_sets[2], face_sets[3]),
    )
    scores = {payload["compatibility_score"] for payload in payloads}
    summaries = {payload["compatibility_score_summary"] for payload in payloads}

    assert len(scores) >= 3
    assert len(summaries) >= 3


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
