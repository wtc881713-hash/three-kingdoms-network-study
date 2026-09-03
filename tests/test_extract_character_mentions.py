"""Tests for conservative full-text character mention extraction."""

import pandas as pd

from src.extract_character_mentions import (
    build_alias_index,
    extract_mentions,
    summarise_mentions,
)


def sample_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "character_id": "CHAR0001",
                "canonical_name": "诸葛亮",
                "frequency": 12,
                "all_aliases": "亮;孔明;诸葛亮",
                "conflicting_aliases": "",
            },
            {
                "character_id": "CHAR0002",
                "canonical_name": "徐晃",
                "frequency": 10,
                "all_aliases": "公明;徐晃",
                "conflicting_aliases": "公明",
            },
        ]
    )


def test_alias_index_excludes_short_and_conflicting_aliases() -> None:
    index = build_alias_index(sample_dictionary())

    assert index == {"孔明": "诸葛亮", "诸葛亮": "诸葛亮", "徐晃": "徐晃"}


def test_extract_mentions_uses_longest_non_overlapping_match() -> None:
    text = (
        "第一回 标题\n诸葛亮见孔明。\n"
        "第二回 标题\n徐晃见公明。\n"
    )
    events = extract_mentions(text, build_alias_index(sample_dictionary()))

    assert list(events["canonical_name"]) == ["诸葛亮", "诸葛亮", "徐晃"]
    assert list(events["matched_alias"]) == ["诸葛亮", "孔明", "徐晃"]
    assert list(events["chapter_number"]) == [1, 1, 2]


def test_summary_separates_raw_mentions_from_extraction_events() -> None:
    text = "第一回 标题\n诸葛亮见孔明。\n"
    dictionary = sample_dictionary()
    events = extract_mentions(text, build_alias_index(dictionary))

    summary = summarise_mentions(events, dictionary)
    zhuge = summary.loc[summary["canonical_name"] == "诸葛亮"].iloc[0]
    xu_huang = summary.loc[summary["canonical_name"] == "徐晃"].iloc[0]

    assert zhuge["raw_mention_frequency"] == 2
    assert zhuge["chapter_coverage"] == 1
    assert zhuge["extraction_event_frequency"] == 12
    assert xu_huang["raw_mention_frequency"] == 0
