"""Score difficult relation instances and select a diverse pilot batch."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "annotation" / "relation_instance_pool.csv"
OUTPUT_FILE = ROOT / "data" / "annotation" / "annotation_batch_01.csv"
REPORT_FILE = ROOT / "outputs" / "reports" / "pilot_selection_report.md"
CONTRAST_CUES = ("却", "反", "暗", "假", "佯", "密", "诈")
STRATEGY_CUES = ("计", "谋", "诱", "伏", "使", "令", "劝", "议", "约", "盟", "降", "叛")
POSITIVE_CUES = ("助", "救", "从", "拜", "敬", "爱", "亲", "义", "忠", "保")
NEGATIVE_CUES = ("杀", "攻", "拒", "怒", "恨", "叛", "欺", "擒", "斩", "敌")
SPEECH_CUES = ("曰", "问", "答", "告", "谓", "道")
PRONOUN_CUES = ("其", "彼", "此人", "之", "乃", "遂")
RELATION_COLUMNS = [
    "instance_id", "chapter_id", "passage_start", "passage_end",
    "character_a", "character_b", "surface_a", "surface_b", "passage",
    "candidate_source", "difficulty_score", "difficulty_reasons",
    "suggested_relation", "model_confidence", "primary_relation",
    "secondary_relation", "relation_direction", "relation_polarity",
    "relation_explicitness", "relation_temporality", "evidence_text",
    "annotator_confidence", "annotator_notes", "annotation_status",
]


def contains_any(text: str, cues: tuple[str, ...]) -> bool:
    """Return whether a passage contains any configured cue."""
    return any(cue in text for cue in cues)


def suggest_relation(text: str) -> str:
    """Provide a transparent rule-based suggestion, never a ground-truth label."""
    if any(cue in text for cue in ("婚", "嫁", "娶", "夫人", "美人", "姻", "妻")):
        return "affection_romance"
    if any(cue in text for cue in ("结义", "兄弟", "故交", "故友", "同窗", "为友")):
        return "friendship_brotherhood"
    if any(
        cue in text
        for cue in (
            "父亲", "母亲", "兄长", "胞弟", "叔父", "叔叔", "侄儿", "侄子",
            "儿子", "女儿", "长子", "次子", "幼子", "之子", "其子", "亲生",
            "生母", "养子", "义子", "岳父", "翁婿",
        )
    ):
        return "kinship"
    if contains_any(text, NEGATIVE_CUES) and contains_any(text, STRATEGY_CUES):
        return "deception_manipulation"
    if contains_any(text, NEGATIVE_CUES):
        return "hostility_conflict"
    if any(cue in text for cue in ("主公", "臣", "将军", "丞相", "军师", "拜")):
        return "hierarchy_loyalty"
    if contains_any(text, POSITIVE_CUES):
        return "cooperation"
    return "no_clear_relation"


def score_instance(row: pd.Series) -> tuple[float, str]:
    """Calculate an interpretable difficulty score and its reasons."""
    text = str(row["passage"])
    score = 0.0
    reasons: list[str] = []
    if int(row["sentence_span"]) >= 3:
        score += 2.0
        reasons.append("multi_sentence_relation")
    if int(row["character_distance"]) >= 120:
        score += 1.5
        reasons.append("characters_far_apart")
    if int(row["character_count"]) >= 4:
        score += min(2.0, 0.5 * (int(row["character_count"]) - 2))
        reasons.append("multiple_characters")
    if int(row["uses_alias"]):
        score += 1.0
        reasons.append("alias_or_title")
    if contains_any(text, PRONOUN_CUES):
        score += 0.75
        reasons.append("pronoun_or_omitted_subject")
    if contains_any(text, CONTRAST_CUES):
        score += 1.0
        reasons.append("contrast_or_hidden_action")
    if contains_any(text, STRATEGY_CUES):
        score += 1.0
        reasons.append("political_or_strategic_action")
    if contains_any(text, POSITIVE_CUES) and contains_any(text, NEGATIVE_CUES):
        score += 1.5
        reasons.append("mixed_relation_signals")
    if text.count("曰") >= 2 or sum(text.count(cue) for cue in SPEECH_CUES) >= 3:
        score += 1.0
        reasons.append("dialogue_speaker_uncertainty")
    if len(text) >= 300:
        score += 0.75
        reasons.append("long_context")
    if not reasons:
        reasons.append("clear_anchor_candidate")
    return round(score, 3), ";".join(reasons)


def score_instances(instances: pd.DataFrame) -> pd.DataFrame:
    """Add difficulty and rule-suggestion fields to the instance pool."""
    scored = instances.copy()
    values = scored.apply(score_instance, axis=1)
    scored["difficulty_score"] = [value[0] for value in values]
    scored["difficulty_reasons"] = [value[1] for value in values]
    scored["suggested_relation"] = scored["passage"].map(suggest_relation)
    scored["model_confidence"] = ""
    scored["narrative_stage"] = pd.cut(
        scored["chapter_id"].astype(int),
        bins=[0, 30, 60, 90, 120],
        labels=["early", "middle_early", "middle_late", "late"],
    ).astype(str)
    return scored


def normalise_for_duplicate_check(text: str) -> str:
    """Normalise a passage for conservative duplicate filtering."""
    return re.sub(r"\W+", "", text)[:180]


def select_diverse_batch(scored: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
    """Select hard, chapter-diverse, pair-diverse examples across four stages."""
    if batch_size != 20:
        raise ValueError("The pilot batch must contain exactly 20 examples.")
    ranked = scored.sort_values(
        ["difficulty_score", "chapter_id", "instance_id"],
        ascending=[False, True, True], kind="stable",
    ).copy()
    selected: list[pd.Series] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_chapters: set[int] = set()
    seen_passages: set[str] = set()

    def add_from(frame: pd.DataFrame, count: int) -> None:
        target = len(selected) + count
        for _, row in frame.iterrows():
            if len(selected) >= target:
                return
            pair = tuple(sorted((str(row["character_a"]), str(row["character_b"]))))
            chapter = int(row["chapter_id"])
            passage_key = normalise_for_duplicate_check(str(row["passage"]))
            if pair in seen_pairs or chapter in seen_chapters or passage_key in seen_passages:
                continue
            selected.append(row)
            seen_pairs.add(pair)
            seen_chapters.add(chapter)
            seen_passages.add(passage_key)

    for stage in ("early", "middle_early", "middle_late", "late"):
        stage_ranked = ranked.loc[ranked["narrative_stage"] == stage]
        add_from(stage_ranked, 4)
        stage_anchors = scored.loc[
            (scored["narrative_stage"] == stage)
            & scored["difficulty_score"].between(1.0, 4.0)
        ].sort_values(
            ["difficulty_score", "chapter_id"], ascending=[True, True], kind="stable"
        )
        add_from(stage_anchors, 1)

    add_from(ranked, 20 - len(selected))
    if len(selected) != batch_size:
        raise ValueError(f"Could select only {len(selected)} diverse pilot examples.")
    batch = pd.DataFrame(selected).sort_values("chapter_id", kind="stable").reset_index(drop=True)
    batch["instance_id"] = [f"PILOT{i:03d}" for i in range(1, batch_size + 1)]
    for column in (
        "primary_relation", "secondary_relation", "relation_direction",
        "relation_polarity", "relation_explicitness", "relation_temporality",
        "evidence_text", "annotator_confidence", "annotator_notes",
    ):
        batch[column] = ""
    batch["annotation_status"] = "pending"
    return batch[RELATION_COLUMNS]


def create_report(pool: pd.DataFrame, batch: pd.DataFrame) -> str:
    """Create a concise audit report for the pilot selection."""
    chapter_list = ", ".join(map(str, batch["chapter_id"].astype(int).tolist()))
    reasons = (
        batch["difficulty_reasons"].str.split(";").explode().value_counts().to_dict()
    )
    reason_lines = "\n".join(f"- {name}: {count}" for name, count in reasons.items())
    case_lines = "\n".join(
        f"- {row.instance_id}: Chapter {row.chapter_id}, {row.character_a}–{row.character_b}; "
        f"score {row.difficulty_score}; {row.difficulty_reasons}."
        for row in batch.itertuples()
    )
    return f"""# Pilot Annotation Selection Report

## Summary

- Candidate passages: {len(pool):,}
- Selected pilot examples: {len(batch)}
- Chapters represented: {batch['chapter_id'].nunique()}
- Character pairs represented: {batch[['character_a', 'character_b']].drop_duplicates().shape[0]}
- Chapter distribution: {chapter_list}
- Selection method: interpretable difficulty scoring followed by narrative-stage, chapter, character-pair, and passage diversity constraints.
- Transformer uncertainty was not used because no relation classifier has been trained yet.

## Existing model and environment

- Checkpoint: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Local cache: `.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`
- Architecture: encoder-only BERT/MiniLM, 12 layers, 384 hidden dimensions, 512-token maximum positions.
- Runtime: Python 3.11.9, PyTorch 2.13.0+cpu, no CUDA device detected.
- Decision: reuse this checkpoint later; do not replace it and do not train from scratch.

## Files created

- `src/annotation/build_relation_instances.py`
- `src/annotation/select_annotation_candidates.py`
- `config/relation_labels.yaml`
- `docs/annotation_guidelines.md`
- `data/annotation/relation_instance_pool.csv`
- `data/annotation/annotation_batch_01.csv`
- `outputs/few_shot_pilot/annotation_batch_01.xlsx`
- `outputs/reports/pilot_selection_report.md`
- `tests/test_relation_instances.py`
- `tests/test_select_annotation_candidates.py`

## Main difficulty reasons

{reason_lines}

## Why each passage is useful

{case_lines}

## Researcher decisions needed before training

- Confirm whether the nine-label ontology is understandable in practice.
- Record ambiguous or overlapping labels in `annotator_notes`.
- Do not use general knowledge of the novel unless nearby context is required and documented.
"""


def main() -> None:
    """Score the candidate pool and export exactly 20 pilot examples."""
    instances = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", keep_default_na=False)
    scored = score_instances(instances)
    batch = select_diverse_batch(scored)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    REPORT_FILE.write_text(create_report(scored, batch), encoding="utf-8")
    print(f"Candidate passages: {len(scored):,}")
    print(f"Pilot examples: {len(batch)}")
    print(f"Chapters represented: {batch['chapter_id'].nunique()}")
    print(f"Character pairs represented: {batch[['character_a', 'character_b']].drop_duplicates().shape[0]}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
