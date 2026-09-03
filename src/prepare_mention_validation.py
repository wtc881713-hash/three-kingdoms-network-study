"""Prepare and score a reproducible human-validation sample of mentions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EVENT_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "character_mention_events.csv"
)
SAMPLE_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "mention_validation_sample.csv"
)
CONFIRMED_SAMPLE_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "mention_validation_sample_confirmed.csv"
)
REPORT_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "mention_validation_report.txt"
)

VALID_DECISIONS = {"yes", "no", "uncertain", ""}


def select_diverse_rows(
    group: pd.DataFrame,
    samples_per_character: int,
) -> pd.DataFrame:
    """Select deterministic rows while preferring alias and chapter diversity."""
    ordered = group.sort_values(
        ["matched_alias", "chapter_number", "global_start"],
        kind="stable",
    )
    selected_indexes: list[int] = []

    # First pass: one event per distinct alias.
    for _, alias_group in ordered.groupby("matched_alias", sort=True):
        middle = alias_group.iloc[len(alias_group) // 2]
        selected_indexes.append(int(middle.name))
        if len(selected_indexes) >= samples_per_character:
            break

    # Second pass: fill remaining slots with events spread across the text.
    remaining = ordered.loc[~ordered.index.isin(selected_indexes)]
    while len(selected_indexes) < samples_per_character and not remaining.empty:
        position = round(
            (len(remaining) - 1)
            * len(selected_indexes)
            / max(samples_per_character - 1, 1)
        )
        selected_indexes.append(int(remaining.iloc[position].name))
        remaining = remaining.drop(index=selected_indexes[-1])

    return group.loc[selected_indexes]


def build_validation_sample(
    events: pd.DataFrame,
    samples_per_character: int = 2,
) -> pd.DataFrame:
    """Create a balanced validation sample covering every character."""
    if samples_per_character < 1:
        raise ValueError("Samples per character must be at least 1.")
    required = {
        "mention_id",
        "chapter_number",
        "chapter_title",
        "canonical_name",
        "matched_alias",
        "global_start",
        "context",
    }
    if missing := required - set(events.columns):
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    parts = [
        select_diverse_rows(group, samples_per_character)
        for _, group in events.groupby("canonical_name", sort=True)
    ]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values(
        ["canonical_name", "matched_alias", "chapter_number", "global_start"],
        kind="stable",
    ).reset_index(drop=True)
    sample.insert(
        0,
        "validation_id",
        [f"VALID{index:04d}" for index in range(1, len(sample) + 1)],
    )
    sample["human_is_correct"] = ""
    sample["human_correct_canonical_name"] = ""
    sample["human_notes"] = ""
    return sample[
        [
            "validation_id",
            "mention_id",
            "chapter_number",
            "chapter_title",
            "canonical_name",
            "matched_alias",
            "context",
            "human_is_correct",
            "human_correct_canonical_name",
            "human_notes",
        ]
    ]


def calculate_validation_metrics(sample: pd.DataFrame) -> dict[str, object]:
    """Calculate precision from completed human decisions only."""
    decisions = sample["human_is_correct"].astype(str).str.strip().str.lower()
    invalid = sorted(set(decisions) - VALID_DECISIONS)
    if invalid:
        raise ValueError(
            "Invalid human_is_correct values: "
            f"{invalid}. Use yes, no, uncertain, or blank."
        )
    reviewed = decisions.isin({"yes", "no", "uncertain"})
    scorable = decisions.isin({"yes", "no"})
    correct = decisions.eq("yes")
    reviewed_count = int(reviewed.sum())
    scorable_count = int(scorable.sum())
    return {
        "total_sample": len(sample),
        "reviewed": reviewed_count,
        "unreviewed": len(sample) - reviewed_count,
        "uncertain": int(decisions.eq("uncertain").sum()),
        "scorable": scorable_count,
        "correct": int(correct.sum()),
        "precision": (
            float(correct.sum() / scorable_count)
            if scorable_count
            else None
        ),
    }


def format_report(metrics: dict[str, object]) -> str:
    """Format a transparent validation report."""
    precision = metrics["precision"]
    precision_text = "NOT AVAILABLE" if precision is None else f"{precision:.4f}"
    status = "PENDING HUMAN REVIEW" if metrics["unreviewed"] else "COMPLETE"
    return "\n".join(
        [
            "Character Mention Validation Report",
            "===================================",
            f"Status: {status}",
            f"Total sample rows: {metrics['total_sample']}",
            f"Reviewed rows: {metrics['reviewed']}",
            f"Unreviewed rows: {metrics['unreviewed']}",
            f"Uncertain rows: {metrics['uncertain']}",
            f"Scorable rows: {metrics['scorable']}",
            f"Correct rows: {metrics['correct']}",
            f"Precision: {precision_text}",
            "",
            "Precision excludes blank and uncertain decisions.",
        ]
    )


def confirm_all_correct(sample: pd.DataFrame, confirmation_note: str) -> pd.DataFrame:
    """Record an explicit user confirmation for every sampled row."""
    if not confirmation_note.strip():
        raise ValueError("A non-empty confirmation note is required.")
    confirmed = sample.copy()
    confirmed["human_is_correct"] = "yes"
    confirmed["human_correct_canonical_name"] = ""
    confirmed["human_notes"] = confirmation_note.strip()
    return confirmed


def parse_args() -> argparse.Namespace:
    """Parse command-line options for explicit batch confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-all-correct",
        action="store_true",
        help="Mark every sampled mention as human-confirmed correct.",
    )
    parser.add_argument(
        "--confirmation-note",
        default="",
        help="Audit note required with --confirm-all-correct.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the sample if absent, then calculate its current review status."""
    args = parse_args()
    if SAMPLE_FILE.exists():
        sample = pd.read_csv(
            SAMPLE_FILE,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        action = "Loaded existing sample without overwriting human fields"
    else:
        events = pd.read_csv(
            EVENT_FILE,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        sample = build_validation_sample(events)
        SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(SAMPLE_FILE, index=False, encoding="utf-8-sig")
        action = "Created validation sample"

    if args.confirm_all_correct:
        sample = confirm_all_correct(sample, args.confirmation_note)
        sample.to_csv(CONFIRMED_SAMPLE_FILE, index=False, encoding="utf-8-sig")
        action = "Recorded explicit confirmation for all sampled mentions"

    metrics = calculate_validation_metrics(sample)
    report = format_report(metrics)
    REPORT_FILE.write_text(report + "\n", encoding="utf-8")

    print(action)
    output_sample = CONFIRMED_SAMPLE_FILE if args.confirm_all_correct else SAMPLE_FILE
    print(f"Sample file: {output_sample}")
    print(f"Report file: {REPORT_FILE}")
    print(report)


if __name__ == "__main__":
    main()
