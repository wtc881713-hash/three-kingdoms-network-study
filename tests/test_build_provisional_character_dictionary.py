"""Tests for provisional canonical character aggregation."""

import pandas as pd

from src.build_provisional_character_dictionary import build_dictionary


def test_aliases_are_aggregated_under_one_character() -> None:
    review = pd.DataFrame(
        [
            {
                "candidate_name": "孔明",
                "frequency": 12,
                "proposed_is_character": "yes",
                "proposed_canonical_name": "诸葛亮",
            },
            {
                "candidate_name": "诸葛亮",
                "frequency": 10,
                "proposed_is_character": "yes",
                "proposed_canonical_name": "诸葛亮",
            },
        ]
    )
    events = pd.DataFrame(
        columns=[
            "candidate_name",
            "proposed_canonical_name",
            "resolution_status",
        ]
    )
    result = build_dictionary(review, events)
    assert result.loc[0, "canonical_name"] == "诸葛亮"
    assert result.loc[0, "frequency"] == 22
    assert result.loc[0, "source_aliases"] == "孔明;诸葛亮"


def test_resolved_ambiguous_events_are_counted_individually() -> None:
    review = pd.DataFrame(
        [
            {
                "candidate_name": "鲁肃",
                "frequency": 10,
                "proposed_is_character": "yes",
                "proposed_canonical_name": "鲁肃",
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "candidate_name": "肃",
                "proposed_canonical_name": "鲁肃",
                "resolution_status": "proposed",
            },
            {
                "candidate_name": "肃",
                "proposed_canonical_name": "",
                "resolution_status": "false_positive",
            },
        ]
    )
    result = build_dictionary(review, events)
    assert result.loc[0, "frequency"] == 11
    assert result.loc[0, "resolved_ambiguous_frequency"] == 1


def test_canonical_character_below_threshold_is_removed() -> None:
    review = pd.DataFrame(
        columns=[
            "candidate_name",
            "frequency",
            "proposed_is_character",
            "proposed_canonical_name",
        ]
    )
    events = pd.DataFrame(
        [
            {
                "candidate_name": "肃",
                "proposed_canonical_name": "王肃",
                "resolution_status": "proposed",
            }
        ]
    )
    result = build_dictionary(review, events, minimum_frequency=10)
    assert result.empty
