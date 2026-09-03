import pandas as pd
import pytest

from src.model.prepare_splits import allocation_for_label, create_splits, validate_splits


def make_reviewed_data() -> pd.DataFrame:
    labels = ["a"] * 8 + ["b"] * 5 + ["c"] * 4 + ["d"] * 2 + ["e"]
    return pd.DataFrame(
        {
            "instance_id": [f"I{i:02d}" for i in range(20)],
            "character_a": [f"A{i}" for i in range(20)],
            "character_b": [f"B{i}" for i in range(20)],
            "passage": [f"passage {i}" for i in range(20)],
            "primary_relation": labels,
            "annotation_status": ["reviewed"] * 20,
        }
    )


def test_allocation_protects_rare_labels():
    assert allocation_for_label(8) == (4, 2, 2)
    assert allocation_for_label(5) == (3, 1, 1)
    assert allocation_for_label(4) == (2, 1, 1)
    assert allocation_for_label(2) == (2, 0, 0)
    assert allocation_for_label(1) == (1, 0, 0)


def test_create_splits_is_deterministic_and_has_expected_sizes():
    data = make_reviewed_data()
    first = create_splits(data)
    second = create_splits(data)
    assert {name: len(frame) for name, frame in first.items()} == {
        "train": 12, "validation": 4, "test": 4
    }
    assert first["train"]["instance_id"].tolist() == second["train"]["instance_id"].tolist()
    assert set(first["train"]["primary_relation"]) == {"a", "b", "c", "d", "e"}


def test_validate_splits_rejects_pair_leakage():
    data = make_reviewed_data()
    splits = create_splits(data)
    splits["test"].loc[0, ["character_a", "character_b"]] = splits["train"].loc[
        0, ["character_a", "character_b"]
    ].values
    with pytest.raises(ValueError, match="Character-pair leakage"):
        validate_splits(splits)
