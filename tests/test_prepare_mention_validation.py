"""Tests for mention-validation sampling and scoring."""

import pandas as pd
import pytest

from src.prepare_mention_validation import (
    build_validation_sample,
    calculate_validation_metrics,
    confirm_all_correct,
)


def sample_events() -> pd.DataFrame:
    rows = []
    for index, (person, alias, chapter) in enumerate(
        [
            ("刘备", "刘备", 1),
            ("刘备", "玄德", 2),
            ("刘备", "玄德", 3),
            ("曹操", "曹操", 1),
            ("曹操", "孟德", 4),
            ("曹操", "曹操", 5),
        ],
        start=1,
    ):
        rows.append(
            {
                "mention_id": f"MENTION{index:06d}",
                "chapter_number": chapter,
                "chapter_title": f"第{chapter}回",
                "canonical_name": person,
                "matched_alias": alias,
                "global_start": index * 10,
                "context": f"context {index}",
            }
        )
    return pd.DataFrame(rows)


def test_sample_covers_every_character_and_alias_diversity() -> None:
    sample = build_validation_sample(sample_events(), samples_per_character=2)

    assert len(sample) == 4
    assert set(sample["canonical_name"]) == {"刘备", "曹操"}
    assert set(sample.loc[sample["canonical_name"] == "刘备", "matched_alias"]) == {
        "刘备",
        "玄德",
    }
    assert sample["human_is_correct"].eq("").all()


def test_metrics_do_not_invent_precision_before_review() -> None:
    sample = build_validation_sample(sample_events(), samples_per_character=1)
    metrics = calculate_validation_metrics(sample)

    assert metrics["reviewed"] == 0
    assert metrics["precision"] is None


def test_metrics_score_only_yes_and_no() -> None:
    sample = pd.DataFrame(
        {"human_is_correct": ["yes", "no", "uncertain", ""]}
    )
    metrics = calculate_validation_metrics(sample)

    assert metrics["reviewed"] == 3
    assert metrics["scorable"] == 2
    assert metrics["correct"] == 1
    assert metrics["precision"] == 0.5


def test_invalid_decision_is_rejected() -> None:
    sample = pd.DataFrame({"human_is_correct": ["maybe"]})

    with pytest.raises(ValueError, match="Invalid human_is_correct"):
        calculate_validation_metrics(sample)


def test_explicit_batch_confirmation_records_audit_note() -> None:
    sample = build_validation_sample(sample_events(), samples_per_character=1)
    confirmed = confirm_all_correct(sample, "Confirmed by the researcher.")

    assert confirmed["human_is_correct"].eq("yes").all()
    assert confirmed["human_notes"].eq("Confirmed by the researcher.").all()
    assert calculate_validation_metrics(confirmed)["precision"] == 1.0


def test_batch_confirmation_requires_a_note() -> None:
    sample = build_validation_sample(sample_events(), samples_per_character=1)

    with pytest.raises(ValueError, match="non-empty confirmation note"):
        confirm_all_correct(sample, "")
