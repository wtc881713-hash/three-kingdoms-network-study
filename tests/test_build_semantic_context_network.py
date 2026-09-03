import numpy as np
import pandas as pd

from src.build_semantic_context_network import (
    aggregate_character_vectors,
    build_character_contexts,
    build_mutual_knn_edges,
    centre_context_vectors,
    extract_context_snippet,
    select_evenly,
)


def test_select_evenly_keeps_boundaries() -> None:
    rows = [{"value": value} for value in range(10)]

    selected = select_evenly(rows, 4)

    assert selected[0]["value"] == 0
    assert selected[-1]["value"] == 9
    assert len(selected) == 4


def test_extract_context_snippet_centres_alias() -> None:
    snippet = extract_context_snippet("abcdefgh玄德ijklmnop", ["玄德"], window=3)

    assert snippet == "fgh玄德ijk"


def test_build_character_contexts_has_one_row_per_character_paragraph() -> None:
    mentions = pd.DataFrame(
        [
            {
                "canonical_name": "刘备",
                "paragraph_id": "P001-001",
                "chapter_number": 1,
                "chapter_title": "Chapter 1",
                "paragraph_text": "玄德曰。玄德行。",
                "matched_alias": "玄德",
            },
            {
                "canonical_name": "刘备",
                "paragraph_id": "P001-001",
                "chapter_number": 1,
                "chapter_title": "Chapter 1",
                "paragraph_text": "玄德曰。玄德行。",
                "matched_alias": "玄德",
            },
        ]
    )

    contexts = build_character_contexts(mentions)

    assert len(contexts) == 1
    assert contexts.loc[0, "canonical_name"] == "刘备"


def test_aggregate_vectors_normalises_character_means() -> None:
    contexts = pd.DataFrame({"canonical_name": ["A", "A", "B"]})
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    characters, aggregated, counts = aggregate_character_vectors(contexts, vectors)

    assert characters == ["A", "B"]
    assert np.allclose(np.linalg.norm(aggregated, axis=1), 1.0)
    assert counts == {"A": 2, "B": 1}


def test_centre_context_vectors_removes_global_mean_and_normalises() -> None:
    vectors = np.array([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])

    centred = centre_context_vectors(vectors)

    assert np.allclose(np.linalg.norm(centred, axis=1), 1.0)


def test_mutual_knn_edges_require_reciprocal_neighbourhood() -> None:
    characters = ["A", "B", "C"]
    vectors = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    edges = build_mutual_knn_edges(
        characters, vectors, neighbours=1, minimum_similarity=0.5
    )

    assert len(edges) == 1
    assert set(edges.loc[0, ["source", "target"]]) == {"A", "B"}
