"""Create the validated character dictionary used by network builders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from src.prepare_mention_validation import calculate_validation_metrics
except ModuleNotFoundError:
    from prepare_mention_validation import calculate_validation_metrics


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "provisional_character_dictionary_with_aliases.csv"
)
VALIDATION_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "mention_validation_sample_confirmed.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "final_character_dictionary.csv"
)


def build_final_dictionary(
    dictionary: pd.DataFrame,
    validation_sample: pd.DataFrame,
    minimum_alias_length: int = 2,
) -> pd.DataFrame:
    """Finalise characters after complete, successful sample validation."""
    metrics = calculate_validation_metrics(validation_sample)
    if metrics["unreviewed"] != 0 or metrics["uncertain"] != 0:
        raise ValueError("Validation must be complete before finalisation.")
    if metrics["precision"] != 1.0:
        raise ValueError("Resolve incorrect sampled mentions before finalisation.")

    alias_owners: defaultdict[str, set[str]] = defaultdict(set)
    per_character: dict[str, set[str]] = {}
    blocked_by_character: dict[str, set[str]] = {}
    for row in dictionary.to_dict(orient="records"):
        canonical = str(row["canonical_name"]).strip()
        blocked = {
            value.strip()
            for value in str(row.get("conflicting_aliases", "")).split(";")
            if value.strip()
        }
        aliases = {
            value.strip()
            for value in str(row["all_aliases"]).split(";")
            if len(value.strip()) >= minimum_alias_length
            and value.strip() not in blocked
        }
        aliases.add(canonical)
        per_character[canonical] = aliases
        blocked_by_character[canonical] = blocked
        for alias in aliases:
            alias_owners[alias].add(canonical)

    rows = []
    for row in dictionary.to_dict(orient="records"):
        canonical = str(row["canonical_name"]).strip()
        usable = {
            alias
            for alias in per_character[canonical]
            if len(alias_owners[alias]) == 1
        }
        rows.append(
            {
                "character_id": row["character_id"],
                "canonical_name": canonical,
                "raw_mention_frequency": "",
                "usable_aliases": ";".join(sorted(usable)),
                "excluded_ambiguous_aliases": ";".join(
                    sorted(blocked_by_character[canonical])
                ),
                "validation_status": "human_validated",
                "validation_sample_size": int(metrics["total_sample"]),
                "validation_precision": float(metrics["precision"]),
                "validation_date": "2026-07-31",
                "notes": (
                    "One-character and cross-character conflicting aliases "
                    "are excluded from automatic matching."
                ),
            }
        )

    final = pd.DataFrame(rows)
    if final["canonical_name"].duplicated().any():
        raise ValueError("Canonical names must be unique.")
    return final


def main() -> None:
    """Build and save the validated network-input dictionary."""
    dictionary = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", keep_default_na=False)
    validation = pd.read_csv(
        VALIDATION_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    final = build_final_dictionary(dictionary, validation)

    summary_file = (
        ROOT / "data" / "metadata" / "gutenberg" / "character_mention_summary.csv"
    )
    summary = pd.read_csv(summary_file, encoding="utf-8-sig", keep_default_na=False)
    mention_counts = summary.set_index("canonical_name")["raw_mention_frequency"]
    final["raw_mention_frequency"] = final["canonical_name"].map(mention_counts)
    if final["raw_mention_frequency"].isna().any():
        raise ValueError("Every final character must have a mention count.")

    final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Validated characters: {len(final)}")
    print(f"Minimum raw mention frequency: {int(final['raw_mention_frequency'].min())}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
