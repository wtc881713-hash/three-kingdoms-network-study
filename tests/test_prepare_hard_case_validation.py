import pandas as pd
import pytest

from src.prepare_hard_case_validation import select_cases


def test_select_cases_preserves_requested_order_and_reasons() -> None:
    source = pd.DataFrame(
        {"validation_id": ["A", "B", "C"], "value": [1, 2, 3]}
    )

    selected = select_cases(source, {"C": "hard c", "A": "hard a"})

    assert selected["validation_id"].tolist() == ["C", "A"]
    assert selected["difficulty_reason"].tolist() == ["hard c", "hard a"]


def test_select_cases_rejects_missing_ids() -> None:
    source = pd.DataFrame({"validation_id": ["A"]})

    with pytest.raises(ValueError, match="Missing validation IDs"):
        select_cases(source, {"B": "missing"})
