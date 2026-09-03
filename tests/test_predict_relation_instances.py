import pandas as pd

from src.model.predict_relation_instances import aggregate_pairs


def test_aggregate_pairs_is_undirected_and_ranks_labels():
    data = pd.DataFrame(
        {
            "instance_id": ["i1", "i2", "i3"],
            "character_a": ["刘备", "诸葛亮", "刘备"],
            "character_b": ["诸葛亮", "刘备", "诸葛亮"],
            "predicted_label": ["cooperation", "cooperation", "hierarchy_loyalty"],
            "predicted_probability": [0.8, 0.6, 0.9],
        }
    )
    output = aggregate_pairs(data)
    assert len(output) == 2
    assert output.iloc[0]["pair_label_rank"] == 1
    assert output.iloc[0]["predicted_label"] == "cooperation"
