"""Tests for provisional candidate review preparation."""

import pandas as pd

from src.prepare_candidate_review import classify_candidate, prepare_review


def test_stable_alias_is_mapped() -> None:
    decision, canonical, basis = classify_candidate("孔明")
    assert (decision, canonical, basis) == ("yes", "诸葛亮", "stable_alias")


def test_false_positive_is_rejected() -> None:
    decision, canonical, basis = classify_candidate("书略")
    assert decision == "no"
    assert canonical == ""
    assert basis == "lexical_false_positive"


def test_ambiguous_alias_is_not_forced() -> None:
    decision, canonical, _ = classify_candidate("昭")
    assert decision == "uncertain"
    assert canonical == ""


def test_review_preserves_multiple_evidence_snippets() -> None:
    candidates = pd.DataFrame(
        [{"candidate_name": "孔明", "frequency": 12}]
    )
    review = prepare_review(candidates, {"孔明": ["one", "two", "three"]})
    assert review.loc[0, "evidence_1"] == "one"
    assert review.loc[0, "evidence_3"] == "three"
    assert review.loc[0, "review_status"] == "provisional"
