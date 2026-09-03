"""Tests for Project Gutenberg corpus preparation."""

from src.prepare_reference_corpus import (
    extract_ebook_body,
    integer_to_chinese,
    unwrap_hard_wrapped_lines,
)


def test_standard_chinese_chapter_numbers() -> None:
    assert integer_to_chinese(1) == "一"
    assert integer_to_chinese(10) == "十"
    assert integer_to_chinese(21) == "二十一"
    assert integer_to_chinese(100) == "一百"
    assert integer_to_chinese(105) == "一百零五"
    assert integer_to_chinese(110) == "一百一十"
    assert integer_to_chinese(120) == "一百二十"


def test_extracts_only_ebook_body() -> None:
    text = (
        "Header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "Producer\n\n"
        "第一回：标题\n正文\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "Footer"
    )
    assert extract_ebook_body(text) == "第一回：标题\n正文"


def test_unwraps_lines_and_preserves_chapter_heading() -> None:
    text = "第一回 标题\n\n第一行\n第二行\n\n下一段"
    assert unwrap_hard_wrapped_lines(text) == "第一回 标题\n\n第一行第二行\n\n下一段"
