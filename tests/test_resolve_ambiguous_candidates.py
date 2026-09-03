"""Tests for event-level ambiguous candidate resolution."""

from src.extract_character_candidates import CandidateEvent
from src.resolve_ambiguous_candidates import (
    build_event_review,
    chapter_for_position,
    propose_resolution,
)


def test_su_is_split_by_narrative_stage() -> None:
    assert propose_resolution("肃", 3)[0] == "李肃"
    assert propose_resolution("肃", 42)[0] == "鲁肃"
    assert propose_resolution("肃", 110)[0] == "王肃"


def test_liang_is_not_globally_mapped() -> None:
    assert propose_resolution("亮", 66)[0] == "诸葛亮"
    assert propose_resolution("亮", 113)[0] == "孙亮"


def test_you_is_split_between_xun_you_and_xu_you() -> None:
    assert propose_resolution("攸", 23)[0] == "荀攸"
    assert propose_resolution("攸", 30)[0] == "许攸"


def test_false_positive_is_preserved_for_audit() -> None:
    canonical, status, _ = propose_resolution("张", 1)
    assert canonical == ""
    assert status == "false_positive"


def test_jin_splits_people_from_reporting_verb() -> None:
    assert propose_resolution("进", 2)[0] == "何进"
    assert propose_resolution("进", 12)[0] == "乐进"
    assert propose_resolution("进", 92)[1] == "false_positive"


def test_fan_is_split_by_chapter() -> None:
    assert propose_resolution("范", 15)[0] == "吕范"
    assert propose_resolution("范", 52)[0] == "赵范"
    assert propose_resolution("范", 107)[0] == "桓范"


def test_chapter_lookup_uses_event_position() -> None:
    chapters = [("第一回", 10), ("第二回", 100)]
    assert chapter_for_position(50, chapters) == (1, "第一回")
    assert chapter_for_position(120, chapters) == (2, "第二回")


def test_event_review_contains_human_review_columns() -> None:
    event = CandidateEvent("亮", "speech", "亮曰", 120)
    chapters = [(f"Chapter {index}", index) for index in range(1, 67)]
    dataframe = build_event_review(
        [event],
        chapters,
        {"亮"},
    )
    assert dataframe.loc[0, "proposed_canonical_name"] == "诸葛亮"
    assert dataframe.loc[0, "human_status"] == ""


def test_unresolved_title_word_is_a_false_positive() -> None:
    event = CandidateEvent("定", "chapter_title", "定三分隆中决策", 10)
    dataframe = build_event_review([event], [("第一回", 0)], {"定"})
    assert dataframe.loc[0, "resolution_status"] == "false_positive"
    assert dataframe.loc[0, "resolution_rule"] == "title_lexical_substring"
