import numpy as np

from src.model.train_frozen_classifier import choose_configuration, evaluate_predictions


def test_evaluate_predictions_uses_all_observed_labels():
    metrics = evaluate_predictions(
        np.array(["a", "b"]), np.array(["a", "a"]), ["a", "b", "c"]
    )
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_f1_all_observed_labels"] < 0.5


def test_choose_configuration_prioritises_macro_f1_then_accuracy():
    rows = [
        {"c_value": 0.1, "class_weight": "none", "macro_f1_all_observed_labels": 0.2, "accuracy": 0.8},
        {"c_value": 1.0, "class_weight": "balanced", "macro_f1_all_observed_labels": 0.3, "accuracy": 0.5},
    ]
    assert choose_configuration(rows)["c_value"] == 1.0
