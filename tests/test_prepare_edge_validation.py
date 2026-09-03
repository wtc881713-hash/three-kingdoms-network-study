"""Tests for stratified edge-validation sampling and scoring."""

import pandas as pd
import pytest

from src.prepare_edge_validation import (
    build_edge_validation_sample,
    calculate_edge_validation_metrics,
)


def edge_rows(count: int = 90) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": f"人物{index:03d}",
                "target": f"人物{index + 1:03d}",
                "weight": count - index,
                "chapter_count": 1,
                "chapters": "1",
                "sample_evidence": f"evidence {index} || extra {index}",
                "relation_definition": "same_body_paragraph",
            }
            for index in range(count)
        ]
    )


def test_sample_has_equal_tiers_and_unique_edges() -> None:
    sample = build_edge_validation_sample(edge_rows(), samples_per_tier=5)

    assert len(sample) == 15
    assert sample["strength_tier"].value_counts().to_dict() == {
        "strong": 5,
        "medium": 5,
        "weak": 5,
    }
    assert not sample.duplicated(["source", "target"]).any()
    assert sample["evidence_checked"].str.contains("extra").sum() == 0


def test_metrics_wait_for_human_review() -> None:
    sample = build_edge_validation_sample(edge_rows(), samples_per_tier=2)
    metrics = calculate_edge_validation_metrics(sample)

    assert metrics["reviewed"] == 0
    assert metrics["precision"] is None


def test_metrics_require_both_checks_for_correct_edge() -> None:
    sample = pd.DataFrame(
        {
            "human_same_paragraph": ["yes", "yes", "no", "uncertain"],
            "human_both_characters": ["yes", "no", "yes", "yes"],
            "interaction_type": ["direct", "indirect", "unclear", ""],
        }
    )
    metrics = calculate_edge_validation_metrics(sample)

    assert metrics["reviewed"] == 4
    assert metrics["scorable"] == 3
    assert metrics["correct"] == 1
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["direct"] == 1
    assert metrics["indirect"] == 1


def test_invalid_review_value_is_rejected() -> None:
    sample = pd.DataFrame(
        {
            "human_same_paragraph": ["maybe"],
            "human_both_characters": ["yes"],
            "interaction_type": ["direct"],
        }
    )

    with pytest.raises(ValueError, match="Invalid yes/no decisions"):
        calculate_edge_validation_metrics(sample)
