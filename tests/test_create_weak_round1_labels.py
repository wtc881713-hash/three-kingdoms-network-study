import pandas as pd

from src.model.create_weak_round1_labels import create_weak_labels


def test_weak_labels_are_explicit_and_low_weight():
    data = pd.DataFrame({"suggested_relation": ["kinship", "cooperation"], "difficulty_score": [0.0, 12.0]})
    output = create_weak_labels(data)
    assert output["weak_label"].tolist() == ["kinship", "cooperation"]
    assert set(output["label_source"]) == {"weak_rule"}
    assert output["sample_weight"].between(0.30, 0.60).all()
    assert set(output["weak_label_status"]) == {"automatic_unreviewed"}
