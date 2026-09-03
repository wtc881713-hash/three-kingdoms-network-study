"""Create a transparent provisional review of formal alias pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "extracted_alias_pairs.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "reviewed_formal_alias_pairs.csv"
)

CTEXT_QIN_MI_URL = "https://ctext.org/text.pl?if=gb&node=603605"


def review_pair(row: dict[str, object]) -> dict[str, object]:
    """Apply documented corrections and edition-variant notes."""
    source_name = str(row["canonical_name"])
    source_alias = str(row["alias"])
    canonical_name = source_name
    alias = source_alias
    status = "provisional_accept"
    basis = "explicit_formal_introduction"
    notes = ""
    reference_url = ""

    if source_name == "孙干":
        canonical_name = "孙乾"
        status = "provisional_normalised"
        basis = "opencc_context_normalisation"
        notes = "OpenCC converted the personal name 乾 to 干."
    elif source_name == "秦宓" and source_alias == "子":
        alias = "子敕"
        status = "provisional_corrected"
        basis = "external_textual_verification"
        notes = (
            "The Gutenberg text contains a placeholder glyph after 子. "
            "The courtesy name was verified as 子敕."
        )
        reference_url = CTEXT_QIN_MI_URL
    elif source_name == "关羽" and source_alias == "寿长":
        status = "provisional_edition_variant"
        basis = "gutenberg_edition_reading"
        notes = "Other editions commonly read this earlier courtesy name as 长生."
    elif source_name == "彭羕" and source_alias == "永言":
        status = "provisional_edition_variant"
        basis = "gutenberg_edition_reading"
        notes = "Historical sources commonly give 永年; this novel edition reads 永言."
    elif source_name == "董袭" and source_alias == "元代":
        status = "provisional_edition_variant"
        basis = "gutenberg_edition_reading"
        notes = "Some historical sources read 元世; the novel edition reads 元代."

    return {
        "source_canonical_name": source_name,
        "source_alias": source_alias,
        "canonical_name": canonical_name,
        "alias": alias,
        "alias_type": row["alias_type"],
        "review_decision": status,
        "decision_basis": basis,
        "evidence_snippet": row["evidence_snippet"],
        "reference_url": reference_url,
        "notes": notes,
        "human_status": "",
        "human_notes": "",
    }


def build_review(alias_pairs: pd.DataFrame) -> pd.DataFrame:
    """Review extracted pairs and add an explicit common-name relation."""
    rows = [
        review_pair(row)
        for row in alias_pairs.to_dict(orient="records")
    ]

    # The same corpus explicitly states "人皆呼为吉平".
    ji_tai = alias_pairs.loc[
        (alias_pairs["canonical_name"] == "吉太")
        & (alias_pairs["alias"] == "称平")
    ]
    if not ji_tai.empty:
        evidence = str(ji_tai.iloc[0]["evidence_snippet"])
        rows.append(
            {
                "source_canonical_name": "吉太",
                "source_alias": "",
                "canonical_name": "吉太",
                "alias": "吉平",
                "alias_type": "common_name",
                "review_decision": "provisional_added",
                "decision_basis": "explicit_called_name_in_corpus",
                "evidence_snippet": evidence,
                "reference_url": "",
                "notes": "The passage explicitly says 人皆呼为吉平.",
                "human_status": "",
                "human_notes": "",
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.drop_duplicates(
        subset=["canonical_name", "alias", "alias_type"],
        keep="first",
    )
    dataframe = dataframe.sort_values(
        ["canonical_name", "alias", "alias_type"],
        kind="stable",
    ).reset_index(drop=True)
    dataframe.insert(
        0,
        "alias_id",
        [f"ALIAS{index:04d}" for index in range(1, len(dataframe) + 1)],
    )
    return dataframe


def main() -> None:
    """Build and save the provisional formal-alias review."""
    alias_pairs = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    review = build_review(alias_pairs)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    counts = review["review_decision"].value_counts().to_dict()
    print(f"Input alias pairs: {len(alias_pairs)}")
    print(f"Reviewed alias relations: {len(review)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Output file: {OUTPUT_FILE}")
    print("All decisions remain provisional until human confirmation.")


if __name__ == "__main__":
    main()
