"""Import artifact-tool-extracted workbook values into a versioned CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "outputs" / "few_shot_pilot" / "review_check" / "completed_values.json"
)
DEFAULT_OUTPUT = ROOT / "data" / "annotation" / "annotation_batch_01_reviewed.csv"


def import_value_matrix(input_file: Path) -> pd.DataFrame:
    """Load a rectangular value matrix whose first row contains headers."""
    matrix = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(matrix, list) or len(matrix) < 2:
        raise ValueError("The workbook value matrix is empty or invalid.")
    header = [str(value).strip() for value in matrix[0]]
    if not all(header) or len(header) != len(set(header)):
        raise ValueError("Workbook headers must be non-empty and unique.")
    if any(len(row) != len(header) for row in matrix[1:]):
        raise ValueError("Workbook rows do not match the header width.")
    return pd.DataFrame(matrix[1:], columns=header)


def main() -> None:
    """Create a versioned reviewed CSV without overwriting the pilot source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reviewed = import_value_matrix(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Reviewed rows: {len(reviewed)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
