"""Build a provisional canonical character table from reviewed candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_REVIEW_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "candidate_review_frequency_ge_10.csv"
)
AMBIGUOUS_EVENT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "ambiguous_candidate_events.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "provisional_canonical_characters_frequency_ge_10.csv"
)


def build_dictionary(
    candidate_review: pd.DataFrame,
    ambiguous_events: pd.DataFrame,
    minimum_frequency: int = 10,
) -> pd.DataFrame:
    """Aggregate stable candidates and resolved events by canonical person."""
    if minimum_frequency < 1:
        raise ValueError("Minimum frequency must be at least 1.")
    totals: defaultdict[str, int] = defaultdict(int)
    stable_counts: defaultdict[str, int] = defaultdict(int)
    resolved_counts: defaultdict[str, int] = defaultdict(int)
    aliases: defaultdict[str, set[str]] = defaultdict(set)

    stable_rows = candidate_review.loc[
        candidate_review["proposed_is_character"] == "yes"
    ]
    for row in stable_rows.to_dict(orient="records"):
        canonical = str(row["proposed_canonical_name"]).strip()
        candidate = str(row["candidate_name"]).strip()
        if not canonical:
            raise ValueError(f"Missing canonical name for stable candidate: {candidate}")
        frequency = int(row["frequency"])
        totals[canonical] += frequency
        stable_counts[canonical] += frequency
        aliases[canonical].add(candidate)

    resolved_rows = ambiguous_events.loc[
        ambiguous_events["resolution_status"] == "proposed"
    ]
    for row in resolved_rows.to_dict(orient="records"):
        canonical = str(row["proposed_canonical_name"]).strip()
        candidate = str(row["candidate_name"]).strip()
        if not canonical:
            raise ValueError(f"Missing canonical name for resolved event: {candidate}")
        totals[canonical] += 1
        resolved_counts[canonical] += 1
        aliases[canonical].add(candidate)

    rows: list[dict[str, object]] = []
    for canonical, frequency in totals.items():
        rows.append(
            {
                "canonical_name": canonical,
                "frequency": frequency,
                "stable_candidate_frequency": stable_counts[canonical],
                "resolved_ambiguous_frequency": resolved_counts[canonical],
                "source_aliases": ";".join(sorted(aliases[canonical])),
                "review_status": "provisional",
                "notes": (
                    "Built from candidate forms that individually met the "
                    "frequency threshold; human confirmation is still required."
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.loc[
        dataframe["frequency"] >= minimum_frequency
    ].copy()
    dataframe = dataframe.sort_values(
        ["frequency", "canonical_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    dataframe.insert(
        0,
        "character_id",
        [f"CHAR{index:04d}" for index in range(1, len(dataframe) + 1)],
    )
    return dataframe


def main() -> None:
    """Build and save the provisional canonical character table."""
    candidate_review = pd.read_csv(
        CANDIDATE_REVIEW_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    ambiguous_events = pd.read_csv(
        AMBIGUOUS_EVENT_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    dictionary = build_dictionary(candidate_review, ambiguous_events)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    discarded_events = int(
        (
            ambiguous_events["resolution_status"] == "false_positive"
        ).sum()
    )
    print(f"Provisional canonical characters: {len(dictionary)}")
    print("Minimum canonical frequency: 10")
    print(f"Included evidence frequency: {int(dictionary['frequency'].sum())}")
    print(f"Discarded ambiguous false-positive events: {discarded_events}")
    print(f"Output file: {OUTPUT_FILE}")
    print("The output is provisional and requires human confirmation.")


if __name__ == "__main__":
    main()
