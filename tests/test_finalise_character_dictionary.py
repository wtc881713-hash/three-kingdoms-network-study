"""Tests for final character-dictionary creation."""

import pandas as pd
import pytest

from src.finalise_character_dictionary import build_final_dictionary


def dictionary_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "character_id": "CHAR0001",
                "canonical_name": "诸葛亮",
                "all_aliases": "亮;孔明;诸葛亮",
                "conflicting_aliases": "",
            },
            {
                "character_id": "CHAR0002",
                "canonical_name": "徐晃",
                "all_aliases": "公明;徐晃",
                "conflicting_aliases": "公明",
            },
        ]
    )


def validation(decisions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"human_is_correct": decisions})


def test_final_dictionary_excludes_short_and_conflicting_aliases() -> None:
    final = build_final_dictionary(dictionary_rows(), validation(["yes", "yes"]))

    zhuge = final.loc[final["canonical_name"] == "诸葛亮"].iloc[0]
    xu_huang = final.loc[final["canonical_name"] == "徐晃"].iloc[0]
    assert zhuge["usable_aliases"] == "孔明;诸葛亮"
    assert xu_huang["usable_aliases"] == "徐晃"
    assert xu_huang["excluded_ambiguous_aliases"] == "公明"
    assert set(final["validation_status"]) == {"human_validated"}


def test_finalisation_rejects_incomplete_review() -> None:
    with pytest.raises(ValueError, match="complete"):
        build_final_dictionary(dictionary_rows(), validation(["yes", ""]))


def test_finalisation_rejects_incorrect_sample() -> None:
    with pytest.raises(ValueError, match="incorrect"):
        build_final_dictionary(dictionary_rows(), validation(["yes", "no"]))
