"""Tests for difficult and diverse pilot selection."""

import pandas as pd
import pytest

from src.annotation.select_annotation_candidates import (
    score_instance,
    select_diverse_batch,
)


def make_rows() -> pd.DataFrame:
    rows = []
    for index in range(1, 33):
        chapter = ((index - 1) * 4) % 120 + 1
        stage = ["early", "middle_early", "middle_late", "late"][(chapter - 1) // 30]
        rows.append(
            {
                "instance_id": f"REL{index:06d}",
                "chapter_id": chapter,
                "passage_start": index * 100,
                "passage_end": index * 100 + 50,
                "character_a": f"人物甲{index}",
                "character_b": f"人物乙{index}",
                "surface_a": f"人物甲{index}",
                "surface_b": f"人物乙{index}",
                "passage": f"人物甲{index}暗设计诱人物乙{index}，却又假意相助。" + "其后众人议论。" * index,
                "candidate_source": "validated_same_body_paragraph",
                "difficulty_score": float(index % 10),
                "difficulty_reasons": "political_or_strategic_action",
                "suggested_relation": "deception_manipulation",
                "model_confidence": "",
                "narrative_stage": stage,
            }
        )
    return pd.DataFrame(rows)


def test_difficulty_score_records_interpretable_reasons() -> None:
    row = pd.Series(
        {
            "passage": "刘备暗使人佯攻曹操，却密令关羽相助。" + "其后议论。" * 40,
            "sentence_span": 3,
            "character_distance": 150,
            "character_count": 4,
            "uses_alias": 1,
        }
    )
    score, reasons = score_instance(row)

    assert score > 5
    assert "characters_far_apart" in reasons
    assert "mixed_relation_signals" in reasons


def test_pilot_selection_is_exact_and_diverse() -> None:
    batch = select_diverse_batch(make_rows())

    assert len(batch) == 20
    assert batch["instance_id"].is_unique
    assert batch["chapter_id"].nunique() == 20
    assert batch[["character_a", "character_b"]].drop_duplicates().shape[0] == 20
    assert set(batch["annotation_status"]) == {"pending"}
    stages = pd.cut(
        batch["chapter_id"], [0, 30, 60, 90, 120],
        labels=["early", "middle_early", "middle_late", "late"],
    )
    assert stages.value_counts().to_dict() == {
        "early": 5, "middle_early": 5, "middle_late": 5, "late": 5,
    }


def test_pilot_size_is_fixed() -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        select_diverse_batch(make_rows(), batch_size=10)
