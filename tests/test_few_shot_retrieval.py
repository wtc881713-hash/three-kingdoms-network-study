import numpy as np
import pandas as pd
import pytest

from src.model.few_shot_retrieval import (
    build_model_text,
    create_model_aids,
    leave_one_out_predictions,
    nearest_indices,
)


def test_build_model_text_contains_entities_and_passage():
    row = pd.Series({"character_a": "刘备", "character_b": "诸葛亮", "passage": "二人议事。"})
    text = build_model_text(row)
    assert "[CHAR_A] 刘备 [/CHAR_A]" in text
    assert "[CHAR_B] 诸葛亮 [/CHAR_B]" in text
    assert "二人议事。" in text


def test_nearest_indices_orders_normalised_cosine_similarity():
    references = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    queries = np.array([[0.9, 0.1]])
    indices, scores = nearest_indices(queries, references, top_k=2)
    assert indices.tolist() == [[0, 1]]
    assert scores[0, 0] > scores[0, 1]
    with pytest.raises(ValueError):
        nearest_indices(queries, references, top_k=4)


def test_leave_one_out_never_retrieves_the_same_row():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = ["a", "a", "b"]
    assert leave_one_out_predictions(vectors, labels) == ["a", "a", "a"]


def test_model_aids_do_not_fill_human_annotation_columns():
    demonstrations = pd.DataFrame(
        {
            "instance_id": ["D1", "D2", "D3"],
            "primary_relation": ["cooperation", "cooperation", "hostility_conflict"],
        }
    )
    candidates = pd.DataFrame(
        {
            "instance_id": ["C1"],
            "primary_relation": [""],
            "annotation_status": ["pending"],
        }
    )
    demo_vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    candidate_vectors = np.array([[1.0, 0.0]])
    output = create_model_aids(
        demonstrations, candidates, demo_vectors, candidate_vectors, top_k=2
    )
    assert output.loc[0, "primary_relation"] == ""
    assert output.loc[0, "annotation_status"] == "pending"
    assert output.loc[0, "few_shot_prediction"] == "cooperation"
    assert output.loc[0, "few_shot_status"] == "suggestion_only_not_ground_truth"
