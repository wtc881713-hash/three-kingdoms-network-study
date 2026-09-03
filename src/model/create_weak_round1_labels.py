"""Create transparent low-weight weak labels for Round 1 without claiming human review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "annotation" / "annotation_batch_02.csv"
OUTPUT = ROOT / "data" / "model" / "weak_round1" / "weak_labels.csv"
REPORT = ROOT / "outputs" / "reports" / "weak_round1_labels.json"


def create_weak_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Copy rule suggestions into a separate, explicitly weak label field."""
    output = data.copy()
    output["weak_label"] = output["suggested_relation"].astype(str)
    output["label_source"] = "weak_rule"
    difficulty = output["difficulty_score"].astype(float)
    output["sample_weight"] = (0.60 - difficulty.clip(0, 12) * 0.025).clip(0.30, 0.60).round(3)
    output["weak_label_status"] = "automatic_unreviewed"
    return output


def main() -> None:
    """Write weak labels and a transparent provenance report."""
    data = pd.read_csv(INPUT, encoding="utf-8-sig", keep_default_na=False)
    output = create_weak_labels(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    report = {
        "rows": len(output),
        "label_source": "weak_rule",
        "human_reviewed": False,
        "label_counts": output["weak_label"].value_counts().sort_index().to_dict(),
        "weight_range": [float(output["sample_weight"].min()), float(output["sample_weight"].max())],
        "warning": "Weak labels may be wrong and must not be reported as human annotations.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Weak labels: {len(output)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
