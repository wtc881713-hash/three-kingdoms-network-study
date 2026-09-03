"""Tests for formal alias-pair review."""

import pandas as pd

from src.review_formal_alias_pairs import build_review, review_pair


def make_row(name: str, alias: str) -> dict[str, str]:
    """Create a minimal extracted alias row."""
    return {
        "canonical_name": name,
        "alias": alias,
        "alias_type": "courtesy_name",
        "evidence_snippet": "evidence",
    }


def test_normalises_sun_qian_name() -> None:
    reviewed = review_pair(make_row("孙干", "公祐"))
    assert reviewed["canonical_name"] == "孙乾"
    assert reviewed["review_decision"] == "provisional_normalised"


def test_corrects_qin_mi_courtesy_name() -> None:
    reviewed = review_pair(make_row("秦宓", "子"))
    assert reviewed["alias"] == "子敕"
    assert reviewed["reference_url"]


def test_marks_edition_variant_without_overwriting_it() -> None:
    reviewed = review_pair(make_row("关羽", "寿长"))
    assert reviewed["alias"] == "寿长"
    assert reviewed["review_decision"] == "provisional_edition_variant"


def test_adds_ji_ping_common_name() -> None:
    dataframe = pd.DataFrame([make_row("吉太", "称平")])
    review = build_review(dataframe)
    assert (
        (review["canonical_name"] == "吉太")
        & (review["alias"] == "吉平")
        & (review["alias_type"] == "common_name")
    ).any()
