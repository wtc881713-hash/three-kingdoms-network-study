"""Validate reviewed literary character-relation annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "annotation" / "annotation_batch_01_reviewed.csv"
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "annotation_validation_report.json"
REQUIRED_COLUMNS = [
    "instance_id", "chapter_id", "passage_start", "passage_end",
    "character_a", "character_b", "surface_a", "surface_b", "passage",
    "candidate_source", "difficulty_score", "difficulty_reasons",
    "suggested_relation", "model_confidence", "primary_relation",
    "secondary_relation", "relation_direction", "relation_polarity",
    "relation_explicitness", "relation_temporality", "evidence_text",
    "annotator_confidence", "annotator_notes", "annotation_status",
]
RELATION_LABELS = {
    "cooperation", "hierarchy_loyalty", "kinship",
    "friendship_brotherhood", "hostility_conflict",
    "deception_manipulation", "affection_romance",
    "no_clear_relation", "uncertain",
}
DIRECTION_VALUES = {"A_to_B", "B_to_A", "bidirectional", "unclear"}
POLARITY_VALUES = {"positive", "negative", "mixed", "neutral", "unclear"}
EXPLICITNESS_VALUES = {"explicit", "implicit", "inferred", "unclear"}
TEMPORALITY_VALUES = {"stable", "temporary", "changing", "unclear"}
STATUS_VALUES = {"pending", "reviewed"}


def normalise_text(value: object) -> str:
    """Normalise whitespace and punctuation spacing for evidence comparison."""
    return re.sub(r"\s+", "", str(value)).strip()


def issue(level: str, code: str, rows: list[str], message: str) -> dict[str, object]:
    """Create one stable report issue."""
    return {"level": level, "code": code, "rows": rows, "message": message}


def find_near_duplicates(data: pd.DataFrame, threshold: float = 0.9) -> list[list[str]]:
    """Find highly similar passages with different instance identifiers."""
    passages = [normalise_text(text) for text in data["passage"]]
    pairs = []
    for left in range(len(passages)):
        for right in range(left + 1, len(passages)):
            if SequenceMatcher(None, passages[left], passages[right]).ratio() >= threshold:
                pairs.append([
                    str(data.iloc[left]["instance_id"]),
                    str(data.iloc[right]["instance_id"]),
                ])
    return pairs


def validate_annotations(data: pd.DataFrame) -> dict[str, object]:
    """Validate schema, decisions, evidence, duplicates, and label coverage."""
    issues: list[dict[str, object]] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        issues.append(issue("critical", "missing_columns", [], f"Missing columns: {missing_columns}"))
        return {"status": "BLOCKED", "row_count": len(data), "issues": issues}

    data = data.fillna("").copy()
    identifiers = data["instance_id"].astype(str)
    duplicates = identifiers[identifiers.duplicated(keep=False)].unique().tolist()
    if duplicates:
        issues.append(issue("critical", "duplicate_instance_id", duplicates, "Instance identifiers must be unique."))

    pair_keys = data.apply(
        lambda row: "::".join(sorted((str(row["character_a"]), str(row["character_b"])))), axis=1
    )
    duplicate_pair_rows = data.loc[pair_keys.duplicated(keep=False), "instance_id"].astype(str).tolist()
    if duplicate_pair_rows:
        issues.append(issue("warning", "character_pair_reuse", duplicate_pair_rows, "The same character pair occurs more than once in this batch."))

    for field in ("instance_id", "character_a", "character_b", "surface_a", "surface_b", "passage"):
        rows = data.loc[data[field].astype(str).str.strip().eq(""), "instance_id"].astype(str).tolist()
        if rows:
            issues.append(issue("critical", f"missing_{field}", rows, f"{field} must not be empty."))

    allowed_fields = {
        "primary_relation": RELATION_LABELS,
        "relation_direction": DIRECTION_VALUES,
        "relation_polarity": POLARITY_VALUES,
        "relation_explicitness": EXPLICITNESS_VALUES,
        "relation_temporality": TEMPORALITY_VALUES,
        "annotation_status": STATUS_VALUES,
    }
    for field, allowed in allowed_fields.items():
        invalid = data.loc[~data[field].astype(str).isin(allowed), "instance_id"].astype(str).tolist()
        if invalid:
            issues.append(issue("critical", f"invalid_{field}", invalid, f"Invalid values in {field}."))

    invalid_secondary = data.loc[
        ~data["secondary_relation"].astype(str).isin(RELATION_LABELS | {""}), "instance_id"
    ].astype(str).tolist()
    if invalid_secondary:
        issues.append(issue("critical", "invalid_secondary_relation", invalid_secondary, "Secondary relation must be blank or an allowed relation label."))

    confidence = pd.to_numeric(data["annotator_confidence"], errors="coerce")
    invalid_confidence = data.loc[confidence.isna() | ~confidence.between(1, 5), "instance_id"].astype(str).tolist()
    if invalid_confidence:
        issues.append(issue("critical", "invalid_annotator_confidence", invalid_confidence, "Annotator confidence must be an integer from 1 to 5."))

    pending = data.loc[data["annotation_status"].astype(str) != "reviewed", "instance_id"].astype(str).tolist()
    if pending:
        issues.append(issue("critical", "not_reviewed", pending, "Every pilot row must be marked reviewed."))

    empty_evidence = data.loc[data["evidence_text"].astype(str).str.strip().eq(""), "instance_id"].astype(str).tolist()
    if empty_evidence:
        issues.append(issue("critical", "missing_evidence", empty_evidence, "Every reviewed row requires evidence text."))

    numeric_evidence = data.loc[
        data["evidence_text"].astype(str).str.fullmatch(r"\s*\d+(?:\.\d+)?\s*", na=False),
        "instance_id",
    ].astype(str).tolist()
    if numeric_evidence:
        issues.append(issue("critical", "numeric_only_evidence", numeric_evidence, "Evidence text contains only a number and must be replaced with an exact passage quotation."))

    nonmatching_evidence = []
    for row in data.itertuples():
        evidence = normalise_text(row.evidence_text)
        passage = normalise_text(row.passage)
        if evidence and not evidence.isnumeric() and evidence not in passage:
            nonmatching_evidence.append(str(row.instance_id))
    if nonmatching_evidence:
        issues.append(issue("warning", "evidence_not_exact_substring", nonmatching_evidence, "Evidence is not an exact contiguous passage substring; verify spacing, omissions, or combined quotations."))

    same_labels = data.loc[
        data["secondary_relation"].astype(str).ne("")
        & data["primary_relation"].astype(str).eq(data["secondary_relation"].astype(str)),
        "instance_id",
    ].astype(str).tolist()
    if same_labels:
        issues.append(issue("warning", "duplicate_primary_secondary", same_labels, "Primary and secondary relations are identical; normally leave the secondary relation blank."))

    uncertain_with_secondary = data.loc[
        data["primary_relation"].astype(str).eq("uncertain")
        & ~data["secondary_relation"].astype(str).isin({"", "uncertain"}),
        "instance_id",
    ].astype(str).tolist()
    if uncertain_with_secondary:
        issues.append(issue("warning", "uncertain_primary_with_specific_secondary", uncertain_with_secondary, "A specific secondary label exists while the primary label is uncertain; confirm whether the specific label should be primary."))

    unexpected_model_confidence = data.loc[
        data["model_confidence"].astype(str).str.strip().ne(""), "instance_id"
    ].astype(str).tolist()
    if unexpected_model_confidence:
        issues.append(issue("warning", "unexpected_model_confidence", unexpected_model_confidence, "No classifier exists yet, so model_confidence should remain blank and will be ignored."))

    near_duplicates = find_near_duplicates(data)
    if near_duplicates:
        rows = sorted({item for pair in near_duplicates for item in pair})
        issues.append(issue("critical", "near_duplicate_passages", rows, "Near-duplicate passages must not be treated as independent pilot examples."))

    label_counts = Counter(data["primary_relation"].astype(str))
    covered = sorted(label for label, count in label_counts.items() if count)
    missing_labels = sorted(RELATION_LABELS - set(covered))
    maximum_share = max(label_counts.values(), default=0) / max(len(data), 1)
    if len(data) >= 5 and maximum_share > 0.5:
        issues.append(issue("warning", "label_imbalance", [], "One primary relation label represents more than half of the pilot."))

    critical_count = sum(item["level"] == "critical" for item in issues)
    warning_count = sum(item["level"] == "warning" for item in issues)
    return {
        "status": "BLOCKED" if critical_count else "PASS_WITH_WARNINGS" if warning_count else "PASS",
        "row_count": len(data),
        "reviewed_rows": int((data["annotation_status"].astype(str) == "reviewed").sum()),
        "unique_character_pairs": int(pair_keys.nunique()),
        "primary_label_counts": dict(sorted(label_counts.items())),
        "covered_primary_labels": covered,
        "missing_primary_labels": missing_labels,
        "maximum_primary_label_share": maximum_share,
        "near_duplicate_pairs": near_duplicates,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def main() -> None:
    """Validate one reviewed annotation CSV and save a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    report = validate_annotations(data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Status: {report['status']}")
    print(f"Critical issues: {report.get('critical_issue_count', 0)}")
    print(f"Warnings: {report.get('warning_count', 0)}")
    print(f"Saved: {args.report}")


if __name__ == "__main__":
    main()
