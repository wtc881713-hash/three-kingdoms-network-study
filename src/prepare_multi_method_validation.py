"""Prepare balanced human-review samples for dialogue and semantic networks."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DIALOGUE_EVENTS_FILE = ROOT / "outputs" / "dialogue" / "named_speech" / "dialogue_events.csv"
SEMANTIC_EDGES_FILE = ROOT / "outputs" / "semantic" / "multilingual_minilm" / "edges.csv"
OUTPUT_DIR = ROOT / "outputs" / "validation" / "multi_method"
RANDOM_SEED = 42


def balanced_dialogue_sample(events: pd.DataFrame) -> pd.DataFrame:
    """Select 30 explicit-target and 30 adjacent-turn dialogue events."""
    samples = []
    for rule, group in events.groupby("extraction_rule", sort=True):
        count = min(30, len(group))
        samples.append(group.sample(n=count, random_state=RANDOM_SEED))
    sample = pd.concat(samples, ignore_index=True).sort_values(
        ["extraction_rule", "chapter_number", "paragraph_id", "event_start"],
        kind="stable",
    ).reset_index(drop=True)
    sample.insert(0, "validation_id", [f"DVAL{index:03d}" for index in range(1, len(sample) + 1)])
    sample["human_source_correct"] = ""
    sample["human_target_correct"] = ""
    sample["human_is_direct_exchange"] = ""
    sample["human_notes"] = ""
    columns = [
        "validation_id", "event_id", "extraction_rule", "chapter_number",
        "paragraph_id", "source", "source_alias", "target", "target_alias",
        "paragraph_text", "human_source_correct", "human_target_correct",
        "human_is_direct_exchange", "human_notes",
    ]
    return sample[columns]


def similarity_tiers(edges: pd.DataFrame) -> pd.Series:
    """Assign deterministic high, medium, and low similarity rank thirds."""
    ranks = edges["similarity"].rank(method="first", ascending=False)
    tier_size = int(np.ceil(len(edges) / 3))
    return ranks.map(
        lambda rank: "high" if rank <= tier_size else ("medium" if rank <= tier_size * 2 else "low")
    )


def balanced_semantic_sample(edges: pd.DataFrame) -> pd.DataFrame:
    """Select 20 semantic edges from each similarity-rank third."""
    working = edges.copy()
    working["similarity_tier"] = similarity_tiers(working)
    samples = []
    for tier in ("high", "medium", "low"):
        group = working.loc[working["similarity_tier"] == tier]
        samples.append(group.sample(n=min(20, len(group)), random_state=RANDOM_SEED))
    sample = pd.concat(samples, ignore_index=True).sort_values(
        ["similarity_tier", "similarity"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    sample.insert(0, "validation_id", [f"SVAL{index:03d}" for index in range(1, len(sample) + 1)])
    sample["human_similarity_meaningful"] = ""
    sample["human_relation_type"] = ""
    sample["human_notes"] = ""
    columns = [
        "validation_id", "similarity_tier", "source", "target", "similarity",
        "source_chapter_number", "source_representative_context",
        "target_chapter_number", "target_representative_context",
        "human_similarity_meaningful", "human_relation_type", "human_notes",
    ]
    return sample[columns]


def main() -> None:
    """Generate both non-destructive validation samples."""
    dialogue = pd.read_csv(DIALOGUE_EVENTS_FILE, encoding="utf-8-sig", keep_default_na=False)
    semantic = pd.read_csv(SEMANTIC_EDGES_FILE, encoding="utf-8-sig", keep_default_na=False)
    dialogue_sample = balanced_dialogue_sample(dialogue)
    semantic_sample = balanced_semantic_sample(semantic)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dialogue_sample.to_csv(
        OUTPUT_DIR / "dialogue_validation_sample.csv", index=False, encoding="utf-8-sig"
    )
    semantic_sample.to_csv(
        OUTPUT_DIR / "semantic_validation_sample.csv", index=False, encoding="utf-8-sig"
    )
    workbook_payload = {
        "dialogue": {
            "columns": dialogue_sample.columns.tolist(),
            "rows": dialogue_sample.astype(object).where(pd.notna(dialogue_sample), "").values.tolist(),
        },
        "semantic": {
            "columns": semantic_sample.columns.tolist(),
            "rows": semantic_sample.astype(object).where(pd.notna(semantic_sample), "").values.tolist(),
        },
    }
    (OUTPUT_DIR / "validation_workbook_data.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Dialogue validation rows: {len(dialogue_sample)}")
    print(dialogue_sample["extraction_rule"].value_counts().to_string())
    print(f"Semantic validation rows: {len(semantic_sample)}")
    print(semantic_sample["similarity_tier"].value_counts().to_string())
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
