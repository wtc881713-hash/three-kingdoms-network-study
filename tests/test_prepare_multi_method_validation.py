import pandas as pd

from src.prepare_multi_method_validation import (
    balanced_dialogue_sample,
    balanced_semantic_sample,
    similarity_tiers,
)


def test_dialogue_sample_balances_rules() -> None:
    rows = []
    for rule in ("explicit_named_target", "adjacent_named_turns"):
        for index in range(40):
            rows.append(
                {
                    "event_id": f"E{rule}{index}", "extraction_rule": rule,
                    "chapter_number": 1, "paragraph_id": f"P{index:03d}",
                    "event_start": index, "source": "A", "source_alias": "A",
                    "target": "B", "target_alias": "B", "paragraph_text": "text",
                }
            )
    sample = balanced_dialogue_sample(pd.DataFrame(rows))

    assert len(sample) == 60
    assert sample["extraction_rule"].value_counts().to_dict() == {
        "adjacent_named_turns": 30,
        "explicit_named_target": 30,
    }


def test_similarity_tiers_cover_all_rank_thirds() -> None:
    edges = pd.DataFrame({"similarity": [0.9 - index * 0.01 for index in range(9)]})

    tiers = similarity_tiers(edges)

    assert tiers.value_counts().to_dict() == {"high": 3, "medium": 3, "low": 3}


def test_semantic_sample_has_twenty_per_tier() -> None:
    edges = []
    for index in range(90):
        edges.append(
            {
                "source": f"A{index}", "target": f"B{index}",
                "similarity": 0.99 - index * 0.001,
                "source_chapter_number": 1, "source_representative_context": "a",
                "target_chapter_number": 2, "target_representative_context": "b",
            }
        )
    sample = balanced_semantic_sample(pd.DataFrame(edges))

    assert len(sample) == 60
    assert sample["similarity_tier"].value_counts().to_dict() == {
        "high": 20, "medium": 20, "low": 20
    }
