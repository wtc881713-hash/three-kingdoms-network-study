"""Tests for visible character markers in edge-validation evidence."""

import pandas as pd
import pytest

from src.highlight_edge_validation_evidence import (
    build_highlighted_sample,
    extract_paragraph_id,
    mark_forms,
)


def test_extract_paragraph_id() -> None:
    assert extract_paragraph_id("Chapter 1, P001-004: text") == "P001-004"


def test_mark_forms_labels_actual_aliases() -> None:
    marked = mark_forms("玄德问孔明，孔明答。", {"玄德"}, {"孔明"})

    assert marked == "【SOURCE:玄德】问【TARGET:孔明】，【TARGET:孔明】答。"


def test_build_highlighted_sample_uses_paragraph_mentions() -> None:
    sample = pd.DataFrame(
        [
            {
                "edge_validation_id": "EDGEVAL001",
                "strength_tier": "strong",
                "source": "刘备",
                "target": "诸葛亮",
                "weight": 5,
                "chapter_count": 2,
                "chapters": "1;2",
                "relation_definition": "same_body_paragraph",
                "evidence_checked": "Chapter 1, P001-004: 玄德问孔明。",
                "human_same_paragraph": "",
                "human_both_characters": "",
                "interaction_type": "",
                "human_notes": "",
            }
        ]
    )
    mentions = pd.DataFrame(
        [
            {"paragraph_id": "P001-004", "canonical_name": "刘备", "matched_alias": "玄德"},
            {"paragraph_id": "P001-004", "canonical_name": "诸葛亮", "matched_alias": "孔明"},
        ]
    )

    highlighted = build_highlighted_sample(sample, mentions)

    assert highlighted.loc[0, "source_forms_in_evidence"] == "玄德"
    assert highlighted.loc[0, "target_forms_in_evidence"] == "孔明"
    assert "【SOURCE:玄德】" in highlighted.loc[0, "highlighted_evidence"]
    assert "【TARGET:孔明】" in highlighted.loc[0, "highlighted_evidence"]


def test_missing_forms_are_rejected() -> None:
    sample = pd.DataFrame(
        [
            {
                "edge_validation_id": "EDGEVAL001",
                "strength_tier": "strong",
                "source": "刘备",
                "target": "诸葛亮",
                "weight": 5,
                "chapter_count": 1,
                "chapters": "1",
                "relation_definition": "same_body_paragraph",
                "evidence_checked": "Chapter 1, P001-004: 玄德独行。",
                "human_same_paragraph": "",
                "human_both_characters": "",
                "interaction_type": "",
                "human_notes": "",
            }
        ]
    )
    mentions = pd.DataFrame(
        [{"paragraph_id": "P001-004", "canonical_name": "刘备", "matched_alias": "玄德"}]
    )

    with pytest.raises(ValueError, match="Missing forms"):
        build_highlighted_sample(sample, mentions)
