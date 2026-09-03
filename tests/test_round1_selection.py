"""Tests for Round 1 candidate selection."""

import pandas as pd

from src.annotation.select_annotation_candidates import score_instances
from src.annotation.select_round1_candidates import (
    RELATION_TARGETS,
    exclude_pilot,
    select_round1,
)


def synthetic_pool() -> pd.DataFrame:
    relations = list(RELATION_TARGETS)
    rows = []
    index = 0
    for chapter in range(1, 121):
        for relation in relations:
            index += 1
            rows.append(
                {
                    "instance_id": f"REL{index:06d}",
                    "chapter_id": chapter,
                    "passage_start": index * 100,
                    "passage_end": index * 100 + 60,
                    "character_a": f"甲{index}",
                    "character_b": f"乙{index}",
                    "surface_a": f"甲{index}",
                    "surface_b": f"乙{index}",
                    "passage": f"甲{index}与乙{index}共同议事。",
                    "candidate_source": "validated_same_body_paragraph",
                    "character_count": 2,
                    "sentence_span": 1,
                    "character_distance": 10,
                    "uses_alias": 0,
                    "difficulty_score": float(index % 12),
                    "difficulty_reasons": "synthetic",
                    "suggested_relation": relation,
                    "model_confidence": "",
                    "narrative_stage": ["early", "middle_early", "middle_late", "late"][(chapter - 1) // 30],
                }
            )
    return pd.DataFrame(rows)


def test_round1_is_exact_balanced_and_unique() -> None:
    batch = select_round1(synthetic_pool())

    assert len(batch) == 60
    assert batch["chapter_id"].nunique() == 60
    assert batch[["character_a", "character_b"]].drop_duplicates().shape[0] == 60
    assert batch["suggested_relation"].value_counts().to_dict() == RELATION_TARGETS
    stages = pd.cut(
        batch["chapter_id"], [0, 30, 60, 90, 120],
        labels=["early", "middle_early", "middle_late", "late"],
    )
    assert set(stages.value_counts()) == {15}
    assert set(batch["annotation_status"]) == {"pending"}


def test_round1_excludes_pilot_pairs_and_passages() -> None:
    pool = synthetic_pool().head(10).copy()
    pilot = pool.iloc[[0]][["character_a", "character_b", "passage"]].copy()
    remaining = exclude_pilot(pool, pilot)

    assert len(remaining) == 9
    assert pool.iloc[0]["instance_id"] not in set(remaining["instance_id"])
