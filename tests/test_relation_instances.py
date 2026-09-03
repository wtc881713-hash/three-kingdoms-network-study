"""Tests for passage-level relation instance construction."""

import pandas as pd

from src.annotation.build_relation_instances import (
    insert_entity_markers,
    split_sentences_with_offsets,
)


def test_sentence_split_preserves_offsets() -> None:
    text = "刘备问计。诸葛亮回答！二人遂行。"
    sentences = split_sentences_with_offsets(text)

    assert [item[2] for item in sentences] == ["刘备问计。", "诸葛亮回答！", "二人遂行。"]
    assert text[sentences[1][0] : sentences[1][1]] == "诸葛亮回答！"


def test_entity_markers_are_inserted() -> None:
    marked = insert_entity_markers("刘备问诸葛亮。", "刘备", "诸葛亮")

    assert "[CHAR_A] 刘备 [/CHAR_A]" in marked
    assert "[CHAR_B] 诸葛亮 [/CHAR_B]" in marked
