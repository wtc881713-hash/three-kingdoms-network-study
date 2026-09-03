"""Prepare and score a stratified validation sample of co-occurrence edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EDGE_FILE = ROOT / "outputs" / "cooccurrence" / "paragraph" / "edges.csv"
SAMPLE_FILE = (
    ROOT
    / "outputs"
    / "cooccurrence"
    / "paragraph"
    / "edge_validation_sample.csv"
)
REPORT_FILE = (
    ROOT
    / "outputs"
    / "cooccurrence"
    / "paragraph"
    / "edge_validation_report.txt"
)

VALID_BINARY = {"", "yes", "no", "uncertain"}
VALID_INTERACTIONS = {"", "direct", "indirect", "unclear"}


def evenly_spaced_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic rows spread across a ranked stratum."""
    if count < 1:
        raise ValueError("Sample count must be at least 1.")
    if len(frame) < count:
        raise ValueError(f"Stratum has {len(frame)} rows but {count} are required.")
    if count == 1:
        return frame.iloc[[len(frame) // 2]]
    indexes = [round(index * (len(frame) - 1) / (count - 1)) for index in range(count)]
    return frame.iloc[indexes]


def build_edge_validation_sample(
    edges: pd.DataFrame,
    samples_per_tier: int = 20,
) -> pd.DataFrame:
    """Sample equal numbers from strong, medium, and weak rank thirds."""
    required = {
        "source",
        "target",
        "weight",
        "chapter_count",
        "chapters",
        "sample_evidence",
        "relation_definition",
    }
    if missing := required - set(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(missing)}")
    if len(edges) < samples_per_tier * 3:
        raise ValueError("At least three complete sampling tiers are required.")

    ranked = edges.sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    tiers = {}
    for tier_index, tier_name in enumerate(("strong", "medium", "weak")):
        start = round(tier_index * len(ranked) / 3)
        end = round((tier_index + 1) * len(ranked) / 3)
        tier = ranked.iloc[start:end].copy()
        tier.insert(0, "strength_tier", tier_name)
        tiers[tier_name] = evenly_spaced_sample(tier, samples_per_tier)

    sample = pd.concat(tiers.values(), ignore_index=True)
    sample.insert(
        0,
        "edge_validation_id",
        [f"EDGEVAL{index:03d}" for index in range(1, len(sample) + 1)],
    )
    sample["evidence_checked"] = sample["sample_evidence"].map(
        lambda value: str(value).split(" || ", maxsplit=1)[0]
    )
    sample["human_same_paragraph"] = ""
    sample["human_both_characters"] = ""
    sample["interaction_type"] = ""
    sample["human_notes"] = ""
    return sample[
        [
            "edge_validation_id",
            "strength_tier",
            "source",
            "target",
            "weight",
            "chapter_count",
            "chapters",
            "relation_definition",
            "evidence_checked",
            "human_same_paragraph",
            "human_both_characters",
            "interaction_type",
            "human_notes",
        ]
    ]


def calculate_edge_validation_metrics(sample: pd.DataFrame) -> dict[str, object]:
    """Calculate relation precision and interaction-type distributions."""
    same = sample["human_same_paragraph"].astype(str).str.strip().str.lower()
    people = sample["human_both_characters"].astype(str).str.strip().str.lower()
    interaction = sample["interaction_type"].astype(str).str.strip().str.lower()
    invalid_binary = sorted((set(same) | set(people)) - VALID_BINARY)
    invalid_interactions = sorted(set(interaction) - VALID_INTERACTIONS)
    if invalid_binary:
        raise ValueError(f"Invalid yes/no decisions: {invalid_binary}")
    if invalid_interactions:
        raise ValueError(f"Invalid interaction types: {invalid_interactions}")

    reviewed = same.isin({"yes", "no", "uncertain"}) & people.isin(
        {"yes", "no", "uncertain"}
    )
    scorable = same.isin({"yes", "no"}) & people.isin({"yes", "no"})
    correct = same.eq("yes") & people.eq("yes")
    scorable_count = int(scorable.sum())
    reviewed_interactions = interaction.loc[reviewed & interaction.ne("")]
    return {
        "total_sample": len(sample),
        "reviewed": int(reviewed.sum()),
        "unreviewed": len(sample) - int(reviewed.sum()),
        "scorable": scorable_count,
        "correct": int((correct & scorable).sum()),
        "precision": (
            float((correct & scorable).sum() / scorable_count)
            if scorable_count
            else None
        ),
        "direct": int(reviewed_interactions.eq("direct").sum()),
        "indirect": int(reviewed_interactions.eq("indirect").sum()),
        "unclear": int(reviewed_interactions.eq("unclear").sum()),
    }


def format_report(metrics: dict[str, object]) -> str:
    """Format a report without inventing results before human review."""
    precision = metrics["precision"]
    precision_text = "NOT AVAILABLE" if precision is None else f"{precision:.4f}"
    status = "PENDING HUMAN REVIEW" if metrics["unreviewed"] else "COMPLETE"
    return "\n".join(
        [
            "Paragraph Edge Validation Report",
            "================================",
            f"Status: {status}",
            f"Total sampled edges: {metrics['total_sample']}",
            f"Reviewed edges: {metrics['reviewed']}",
            f"Unreviewed edges: {metrics['unreviewed']}",
            f"Scorable edges: {metrics['scorable']}",
            f"Correct edges: {metrics['correct']}",
            f"Precision: {precision_text}",
            f"Direct interactions: {metrics['direct']}",
            f"Indirect co-narration: {metrics['indirect']}",
            f"Unclear interactions: {metrics['unclear']}",
            "",
            "Precision requires both same-paragraph and character decisions to be yes.",
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse an optional reviewed CSV input path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Reviewed validation CSV to score instead of the blank sample.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a sample once, then score its current human-review state."""
    args = parse_args()
    if args.input is not None:
        sample = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
        action = f"Loaded reviewed results from {args.input}"
    elif SAMPLE_FILE.exists():
        sample = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig", keep_default_na=False)
        action = "Loaded existing sample without overwriting human fields"
    else:
        edges = pd.read_csv(EDGE_FILE, encoding="utf-8-sig", keep_default_na=False)
        sample = build_edge_validation_sample(edges)
        SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(SAMPLE_FILE, index=False, encoding="utf-8-sig")
        action = "Created stratified edge-validation sample"

    report = format_report(calculate_edge_validation_metrics(sample))
    REPORT_FILE.write_text(report + "\n", encoding="utf-8")
    print(action)
    source_file = args.input if args.input is not None else SAMPLE_FILE
    print(f"Sample file: {source_file}")
    print(f"Report file: {REPORT_FILE}")
    print(report)


if __name__ == "__main__":
    main()
