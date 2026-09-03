"""Add visible character markers to sampled co-occurrence evidence."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = (
    ROOT / "outputs" / "cooccurrence" / "paragraph" / "edge_validation_sample.csv"
)
PARAGRAPH_MENTION_FILE = (
    ROOT / "outputs" / "cooccurrence" / "paragraph" / "paragraph_mentions.csv"
)
OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "cooccurrence"
    / "paragraph"
    / "edge_validation_sample_highlighted.csv"
)

PARAGRAPH_ID_PATTERN = re.compile(r"\b(P\d{3}-\d{3}):")


def extract_paragraph_id(evidence: str) -> str:
    """Extract the stable paragraph identifier from an evidence string."""
    match = PARAGRAPH_ID_PATTERN.search(str(evidence))
    if not match:
        raise ValueError(f"Evidence has no paragraph ID: {evidence[:80]}")
    return match.group(1)


def mark_forms(
    evidence: str,
    source_forms: set[str],
    target_forms: set[str],
) -> str:
    """Wrap actual source and target forms in distinct searchable markers."""
    owner_by_form = {
        **{form: "SOURCE" for form in source_forms},
        **{form: "TARGET" for form in target_forms},
    }
    if not owner_by_form:
        return evidence
    pattern = re.compile(
        "|".join(
            re.escape(form)
            for form in sorted(owner_by_form, key=lambda value: (-len(value), value))
        )
    )
    return pattern.sub(
        lambda match: f"【{owner_by_form[match.group(0)]}:{match.group(0)}】",
        evidence,
    )


def build_highlighted_sample(
    sample: pd.DataFrame,
    paragraph_mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Add actual alias forms and marked evidence to every sampled edge."""
    mentions = paragraph_mentions.copy()
    lookup = {
        (str(paragraph_id), str(character)): set(group["matched_alias"].astype(str))
        for (paragraph_id, character), group in mentions.groupby(
            ["paragraph_id", "canonical_name"],
            sort=False,
        )
    }

    rows = []
    for row in sample.to_dict(orient="records"):
        paragraph_id = extract_paragraph_id(str(row["evidence_checked"]))
        source = str(row["source"])
        target = str(row["target"])
        source_forms = lookup.get((paragraph_id, source), set())
        target_forms = lookup.get((paragraph_id, target), set())
        if not source_forms or not target_forms:
            raise ValueError(
                f"Missing forms for {source}–{target} in {paragraph_id}."
            )
        enriched = dict(row)
        enriched["source_forms_in_evidence"] = ";".join(sorted(source_forms))
        enriched["target_forms_in_evidence"] = ";".join(sorted(target_forms))
        enriched["highlighted_evidence"] = mark_forms(
            str(row["evidence_checked"]),
            source_forms,
            target_forms,
        )
        rows.append(enriched)

    output = pd.DataFrame(rows)
    ordered_columns = [
        "edge_validation_id",
        "strength_tier",
        "source",
        "target",
        "source_forms_in_evidence",
        "target_forms_in_evidence",
        "weight",
        "chapter_count",
        "chapters",
        "relation_definition",
        "highlighted_evidence",
        "human_same_paragraph",
        "human_both_characters",
        "interaction_type",
        "human_notes",
    ]
    return output[ordered_columns]


def main() -> None:
    """Create the highlighted review CSV without altering the original sample."""
    sample = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig", keep_default_na=False)
    paragraph_mentions = pd.read_csv(
        PARAGRAPH_MENTION_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    highlighted = build_highlighted_sample(sample, paragraph_mentions)
    highlighted.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Highlighted validation rows: {len(highlighted)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
