"""Select a diverse 60-example Round 1 relation-annotation batch."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from src.annotation.select_annotation_candidates import (
        RELATION_COLUMNS,
        normalise_for_duplicate_check,
        score_instances,
    )
except ModuleNotFoundError:
    from select_annotation_candidates import (
        RELATION_COLUMNS,
        normalise_for_duplicate_check,
        score_instances,
    )


ROOT = Path(__file__).resolve().parents[2]
POOL_FILE = ROOT / "data" / "annotation" / "relation_instance_pool.csv"
PILOT_FILE = ROOT / "data" / "annotation" / "annotation_batch_01_reviewed_v2.csv"
OUTPUT_FILE = ROOT / "data" / "annotation" / "annotation_batch_02.csv"
REPORT_FILE = ROOT / "outputs" / "reports" / "round1_selection_report.md"
ROUND1_SIZE = 60
RELATION_TARGETS = {
    "cooperation": 8,
    "hierarchy_loyalty": 8,
    "kinship": 8,
    "friendship_brotherhood": 4,
    "hostility_conflict": 8,
    "deception_manipulation": 8,
    "affection_romance": 6,
    "no_clear_relation": 10,
}


def pair_key(row: pd.Series) -> tuple[str, str]:
    """Return a stable undirected character-pair key."""
    return tuple(sorted((str(row["character_a"]), str(row["character_b"]))))


def exclude_pilot(scored: pd.DataFrame, pilot: pd.DataFrame) -> pd.DataFrame:
    """Exclude all passages and character pairs already used in Round 0."""
    pilot_pairs = {pair_key(row) for _, row in pilot.iterrows()}
    pilot_passages = {
        normalise_for_duplicate_check(str(text)) for text in pilot["passage"]
    }
    mask = scored.apply(
        lambda row: pair_key(row) not in pilot_pairs
        and normalise_for_duplicate_check(str(row["passage"])) not in pilot_passages,
        axis=1,
    )
    return scored.loc[mask].copy()


def select_round1(scored: pd.DataFrame, batch_size: int = ROUND1_SIZE) -> pd.DataFrame:
    """Select exact relation and narrative-stage quotas with unique pairs."""
    if batch_size != ROUND1_SIZE:
        raise ValueError("Round 1 must contain exactly 60 examples.")
    if sum(RELATION_TARGETS.values()) != batch_size:
        raise ValueError("Relation targets must sum to the Round 1 batch size.")

    selected: list[pd.Series] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_passages: set[str] = set()
    chapter_counts: Counter[int] = Counter()
    stage_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    anchor_count = 0

    def choose(relation: str, prefer_anchor: bool) -> pd.Series | None:
        candidates = scored.loc[scored["suggested_relation"] == relation].copy()
        candidates["stage_load"] = candidates["narrative_stage"].map(stage_counts).fillna(0)
        if prefer_anchor:
            candidates = candidates.loc[candidates["difficulty_score"] <= 4.0].sort_values(
                ["stage_load", "difficulty_score", "chapter_id", "instance_id"],
                ascending=[True, True, True, True], kind="stable",
            )
        else:
            candidates = candidates.sort_values(
                ["stage_load", "difficulty_score", "chapter_id", "instance_id"],
                ascending=[True, False, True, True], kind="stable",
            )
        for _, row in candidates.iterrows():
            pair = pair_key(row)
            passage = normalise_for_duplicate_check(str(row["passage"]))
            stage = str(row["narrative_stage"])
            chapter = int(row["chapter_id"])
            if pair in seen_pairs or passage in seen_passages:
                continue
            if stage_counts[stage] >= 15 or chapter_counts[chapter] >= 1:
                continue
            return row
        return None

    slot = 0
    while len(selected) < batch_size:
        progress = False
        for relation, target in RELATION_TARGETS.items():
            if relation_counts[relation] >= target:
                continue
            prefer_anchor = slot % 5 == 0
            row = choose(relation, prefer_anchor)
            if row is None and prefer_anchor:
                row = choose(relation, False)
            if row is None:
                continue
            pair = pair_key(row)
            passage = normalise_for_duplicate_check(str(row["passage"]))
            chapter = int(row["chapter_id"])
            stage = str(row["narrative_stage"])
            selected.append(row)
            seen_pairs.add(pair)
            seen_passages.add(passage)
            chapter_counts[chapter] += 1
            stage_counts[stage] += 1
            relation_counts[relation] += 1
            if float(row["difficulty_score"]) <= 4.0:
                anchor_count += 1
            slot += 1
            progress = True
        if not progress:
            raise ValueError(
                f"Unable to satisfy Round 1 quotas. Selected {len(selected)}; "
                f"relations={dict(relation_counts)}; stages={dict(stage_counts)}"
            )

    batch = pd.DataFrame(selected).sort_values(
        ["chapter_id", "passage_start", "character_a", "character_b"], kind="stable"
    ).reset_index(drop=True)
    batch["instance_id"] = [f"ROUND1_{index:03d}" for index in range(1, batch_size + 1)]
    for column in (
        "primary_relation", "secondary_relation", "relation_direction",
        "relation_polarity", "relation_explicitness", "relation_temporality",
        "evidence_text", "annotator_confidence", "annotator_notes",
    ):
        batch[column] = ""
    batch["annotation_status"] = "pending"
    batch.attrs["anchor_count"] = anchor_count
    return batch[RELATION_COLUMNS]


def create_report(pool: pd.DataFrame, batch: pd.DataFrame) -> str:
    """Create an auditable Round 1 selection report."""
    stages = pd.cut(
        batch["chapter_id"].astype(int), [0, 30, 60, 90, 120],
        labels=["early", "middle_early", "middle_late", "late"],
    ).value_counts().sort_index().to_dict()
    suggestions = batch["suggested_relation"].value_counts().sort_index().to_dict()
    difficult = int((batch["difficulty_score"].astype(float) > 4.0).sum())
    anchors = len(batch) - difficult
    return f"""# Round 1 Annotation Selection Report

- Eligible candidate passages after Round 0 exclusion: {len(pool):,}
- Selected examples: {len(batch)}
- Unique chapters: {batch['chapter_id'].nunique()}
- Unique character pairs: {batch[['character_a', 'character_b']].drop_duplicates().shape[0]}
- Narrative-stage distribution: {stages}
- Rule-suggestion distribution: {suggestions}
- High-difficulty examples: {difficult}
- Clearer anchor examples: {anchors}
- Selection constraints: no Round 0 pair or passage reuse; one example per chapter;
  one example per character pair; 15 examples per narrative stage; fixed relation
  suggestion targets; no near-duplicate passage keys.

The suggested labels are weak rule-based prompts, not ground truth. The
researcher must make an independent passage-based decision.
"""


def main() -> None:
    """Generate and save the 60-example Round 1 batch."""
    pool = pd.read_csv(POOL_FILE, encoding="utf-8-sig", keep_default_na=False)
    pilot = pd.read_csv(PILOT_FILE, encoding="utf-8-sig", keep_default_na=False)
    scored = exclude_pilot(score_instances(pool), pilot)
    batch = select_round1(scored)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    REPORT_FILE.write_text(create_report(scored, batch), encoding="utf-8")
    print(f"Eligible candidates: {len(scored):,}")
    print(f"Round 1 examples: {len(batch)}")
    print(f"Chapters represented: {batch['chapter_id'].nunique()}")
    print(f"Character pairs represented: {batch[['character_a', 'character_b']].drop_duplicates().shape[0]}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
