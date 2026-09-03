"""Tests for the rule-based character candidate extraction pipeline."""

from pathlib import Path

from src.extract_character_candidates import (
    ALIAS_COLUMNS,
    CANDIDATE_COLUMNS,
    AliasPair,
    build_alias_dataframe,
    build_candidate_dataframe,
    check_text_integrity,
    clean_candidate,
    deduplicate_alias_pairs,
    detect_chapters,
    extract_formal_introductions,
    extract_chapter_title_candidates,
    extract_speech_candidates,
    save_outputs,
)


def event_names(text: str) -> list[str]:
    """Return names extracted from a short speech sample."""
    return [event.candidate_name for event in extract_speech_candidates(text)]


def test_extracts_xuande_from_speech() -> None:
    assert event_names("玄德曰：“此言甚善。”") == ["玄德"]


def test_extracts_kongming_without_reporting_verb() -> None:
    assert event_names("孔明笑曰：“主公勿忧。”") == ["孔明"]


def test_extracts_cao_cao_and_mengde_from_introduction() -> None:
    events, aliases = extract_formal_introductions("姓曹，名操，字孟德。")
    assert {event.candidate_name for event in events} == {"曹操", "孟德"}
    assert ("曹操", "孟德") in {
        (pair.canonical_name, pair.alias) for pair in aliases
    }


def test_extracts_zhuge_liang_and_kongming_from_introduction() -> None:
    events, aliases = extract_formal_introductions("姓诸葛，名亮，字孔明。")
    assert {event.candidate_name for event in events} == {"诸葛亮", "孔明"}
    assert ("诸葛亮", "孔明") in {
        (pair.canonical_name, pair.alias) for pair in aliases
    }


def test_removes_obvious_stopword() -> None:
    assert clean_candidate("众人") is None


def test_preserves_supported_one_character_name() -> None:
    assert event_names("操曰：“可速进兵。”") == ["操"]
    assert clean_candidate("操") == "操"


def test_does_not_extract_reporting_word_as_name() -> None:
    assert event_names("忽见一人大笑曰：“何故如此？”") == []


def test_removes_poetic_narrator_phrase() -> None:
    assert clean_candidate("后人有诗") is None


def test_detects_replacement_character() -> None:
    result = check_text_integrity("第一回 标题\n内容�\n第一百二十回 结尾")
    assert result.replacement_count == 1
    assert result.status == "WARNING"


def test_detects_ascii_question_marks_as_suspicious() -> None:
    result = check_text_integrity("第一回 标题\n乱码???\n第一百二十回 结尾")
    assert result.suspicious_count == 3
    assert result.status == "WARNING"


def test_detects_chapter_titles() -> None:
    text = "第一回 甲\n正文\n第二回 乙\n正文"
    assert [title for title, _ in detect_chapters(text)] == [
        "第一回 甲",
        "第二回 乙",
    ]


def test_chapter_title_uses_longest_non_overlapping_candidate() -> None:
    events = extract_chapter_title_candidates(
        "第一回 诸葛亮出山",
        ["亮", "诸葛亮"],
    )
    assert [event.candidate_name for event in events] == ["诸葛亮"]


def test_removes_duplicate_alias_pairs() -> None:
    pair = AliasPair("曹操", "孟德", "courtesy_name", "姓曹名操字孟德")
    assert deduplicate_alias_pairs([pair, pair]) == [pair]


def test_csv_column_order_without_real_outputs(tmp_path: Path) -> None:
    speech_event = extract_speech_candidates("玄德曰：“善。”")
    candidate_dataframe = build_candidate_dataframe(speech_event)
    alias_dataframe = build_alias_dataframe([])
    candidate_file = tmp_path / "candidates.csv"
    alias_file = tmp_path / "aliases.csv"

    save_outputs(
        candidate_dataframe,
        alias_dataframe,
        candidate_file,
        alias_file,
    )

    assert list(candidate_dataframe.columns) == CANDIDATE_COLUMNS
    assert list(alias_dataframe.columns) == ALIAS_COLUMNS
    assert candidate_file.read_bytes().startswith(b"\xef\xbb\xbf")
    assert alias_file.read_bytes().startswith(b"\xef\xbb\xbf")
