"""Filter character candidates by a reproducible frequency threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "data" / "metadata" / "gutenberg" / "character_candidates.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "character_candidates_frequency_ge_10.csv"
)
REQUIRED_COLUMNS = {"candidate_id", "candidate_name", "frequency"}


def read_candidates(file_path: Path) -> pd.DataFrame:
    """Read a candidate CSV and validate the required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {file_path}")

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Candidate file is missing required columns: {missing}")

    dataframe["frequency"] = pd.to_numeric(
        dataframe["frequency"],
        errors="raise",
    )
    return dataframe


def filter_candidates(
    dataframe: pd.DataFrame,
    minimum_frequency: int,
) -> pd.DataFrame:
    """Return candidates meeting the inclusive frequency threshold."""
    if minimum_frequency < 1:
        raise ValueError("Minimum frequency must be at least 1.")

    filtered = dataframe.loc[
        dataframe["frequency"] >= minimum_frequency
    ].copy()
    filtered = filtered.sort_values(
        ["frequency", "candidate_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return filtered


def save_candidates(dataframe: pd.DataFrame, output_file: Path) -> None:
    """Save the filtered candidates with a UTF-8 BOM."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and threshold."""
    parser = argparse.ArgumentParser(
        description="Keep candidates whose extraction frequency meets a threshold."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-frequency", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Filter the candidate table and print a concise summary."""
    args = parse_args()
    candidates = read_candidates(args.input)
    filtered = filter_candidates(candidates, args.minimum_frequency)
    save_candidates(filtered, args.output)

    print(f"Input candidates: {len(candidates)}")
    print(f"Minimum frequency: {args.minimum_frequency}")
    print(f"Retained candidates: {len(filtered)}")
    print(f"Discarded from filtered output: {len(candidates) - len(filtered)}")
    print(f"Output file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
