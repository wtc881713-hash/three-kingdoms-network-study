"""Tests for candidate frequency filtering."""

from pathlib import Path

import pandas as pd

from src.filter_character_candidates import (
    filter_candidates,
    read_candidates,
    save_candidates,
)


def sample_dataframe() -> pd.DataFrame:
    """Return a small candidate table for threshold tests."""
    return pd.DataFrame(
        [
            {"candidate_id": "CAND0001", "candidate_name": "孔明", "frequency": 12},
            {"candidate_id": "CAND0002", "candidate_name": "曹操", "frequency": 10},
            {"candidate_id": "CAND0003", "candidate_name": "路人", "frequency": 9},
        ]
    )


def test_threshold_is_inclusive() -> None:
    filtered = filter_candidates(sample_dataframe(), 10)
    assert filtered["candidate_name"].tolist() == ["孔明", "曹操"]


def test_candidates_below_threshold_are_removed() -> None:
    filtered = filter_candidates(sample_dataframe(), 10)
    assert "路人" not in set(filtered["candidate_name"])


def test_rejects_invalid_threshold() -> None:
    try:
        filter_candidates(sample_dataframe(), 0)
    except ValueError as error:
        assert "at least 1" in str(error)
    else:
        raise AssertionError("Expected ValueError for an invalid threshold.")


def test_saved_filter_uses_utf8_bom(tmp_path: Path) -> None:
    output_file = tmp_path / "filtered.csv"
    save_candidates(filter_candidates(sample_dataframe(), 10), output_file)
    assert output_file.read_bytes().startswith(b"\xef\xbb\xbf")


def test_read_candidates_preserves_chinese(tmp_path: Path) -> None:
    input_file = tmp_path / "candidates.csv"
    save_candidates(sample_dataframe(), input_file)
    loaded = read_candidates(input_file)
    assert loaded["candidate_name"].tolist()[0] == "孔明"
