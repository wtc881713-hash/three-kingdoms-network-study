"""Extract difficult dialogue and semantic cases for researcher review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "outputs" / "validation" / "multi_method"
OUTPUT_DIR = ROOT / "outputs" / "validation" / "hard_cases"

DIALOGUE_CASES = {
    "DVAL001": "A strategic prediction is followed by a later battlefield speech; possible scene transition rather than a direct exchange.",
    "DVAL006": "Both speakers advise Liu Bei in sequence and may not address each other.",
    "DVAL008": "Several advisers speak to Liu Bei in a group discussion; consecutive speakers may have different addressees.",
    "DVAL010": "A long multi-speaker visit contains Liu Bei, Zhuge Jun, and Zhang Fei; adjacency may skip the actual addressee.",
    "DVAL017": "Lu Su links separate conversations with Zhuge Liang and Zhou Yu; possible scene transition.",
    "DVAL022": "Zhuge Liang intervenes between Liu Bei and Wei Yan, so the extracted pair may omit the actual speaker.",
    "DVAL023": "The paragraph moves from Liu Zhang's court to Liu Bei and Pang Tong at another location.",
    "DVAL024": "A long military-planning scene contains several repeated speakers and possible competing addressees.",
    "DVAL027": "A long battle narrative precedes the named exchange; the local dialogue is valid only if the final turn boundary is handled correctly.",
    "DVAL028": "Sima Shi and Sima Zhao both respond to their father and may not speak directly to each other.",
}

SEMANTIC_CASES = {
    "SVAL031": "Low-tier similarity between two advisers; distinguish shared role from meaningful narrative similarity.",
    "SVAL032": "Cross-generation Wei figures with no simple direct relationship; similarity may reflect court language.",
    "SVAL033": "Father-son relation, but representative contexts come from different narrative stages.",
    "SVAL034": "Both are linked to anti-tyrant or medical plots, but the connection may be only thematic vocabulary.",
    "SVAL035": "Political opponents from different camps and periods; shared strategic language may create a false semantic link.",
    "SVAL036": "Warriors from different factions and periods; test whether generic battle language drives similarity.",
    "SVAL037": "Same regional storyline but a complex political relationship; distinguish faction from conflict.",
    "SVAL038": "Two advisers named Chen in the same broad era but different loyalties; possible generic adviser-language similarity.",
    "SVAL039": "Brothers in the Sima family; judge whether family/role similarity is visible in the supplied contexts.",
    "SVAL040": "Wei figures from different generations; similarity is close to the retained threshold.",
}


def select_cases(
    source: pd.DataFrame,
    selected: dict[str, str],
) -> pd.DataFrame:
    """Select ordered validation IDs and attach explicit difficulty reasons."""
    indexed = source.set_index("validation_id", drop=False)
    missing = set(selected) - set(indexed.index)
    if missing:
        raise ValueError(f"Missing validation IDs: {sorted(missing)}")
    rows = indexed.loc[list(selected)].copy().reset_index(drop=True)
    rows.insert(1, "difficulty_reason", rows["validation_id"].map(selected))
    return rows


def main() -> None:
    """Create CSV and JSON sources for the compact review workbook."""
    dialogue = pd.read_csv(
        SOURCE_DIR / "dialogue_validation_sample.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    semantic = pd.read_csv(
        SOURCE_DIR / "semantic_validation_sample.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    dialogue_cases = select_cases(dialogue, DIALOGUE_CASES)
    semantic_cases = select_cases(semantic, SEMANTIC_CASES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dialogue_cases.to_csv(
        OUTPUT_DIR / "dialogue_hard_cases.csv", index=False, encoding="utf-8-sig"
    )
    semantic_cases.to_csv(
        OUTPUT_DIR / "semantic_hard_cases.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "dialogue": {
            "columns": dialogue_cases.columns.tolist(),
            "rows": dialogue_cases.astype(object).where(pd.notna(dialogue_cases), "").values.tolist(),
        },
        "semantic": {
            "columns": semantic_cases.columns.tolist(),
            "rows": semantic_cases.astype(object).where(pd.notna(semantic_cases), "").values.tolist(),
        },
    }
    (OUTPUT_DIR / "hard_case_workbook_data.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Dialogue hard cases: {len(dialogue_cases)}")
    print(f"Semantic hard cases: {len(semantic_cases)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
