"""Tests for reviewed annotation validation."""

import pandas as pd

from src.annotation.validate_annotations import REQUIRED_COLUMNS, validate_annotations


def valid_row(instance_id: str = "PILOT001") -> dict[str, object]:
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "instance_id": instance_id,
            "chapter_id": 1,
            "passage_start": 0,
            "passage_end": 10,
            "character_a": "刘备",
            "character_b": "关羽",
            "surface_a": "玄德",
            "surface_b": "云长",
            "passage": "玄德与云长同往。",
            "candidate_source": "validated_same_body_paragraph",
            "difficulty_score": 1.0,
            "difficulty_reasons": "alias_or_title",
            "suggested_relation": "cooperation",
            "primary_relation": "cooperation",
            "relation_direction": "bidirectional",
            "relation_polarity": "positive",
            "relation_explicitness": "explicit",
            "relation_temporality": "temporary",
            "evidence_text": "玄德与云长同往",
            "annotator_confidence": 5,
            "annotation_status": "reviewed",
        }
    )
    return row


def test_valid_annotation_passes() -> None:
    report = validate_annotations(pd.DataFrame([valid_row()]))

    assert report["status"] == "PASS"
    assert report["critical_issue_count"] == 0


def test_numeric_evidence_blocks_validation() -> None:
    row = valid_row()
    row["evidence_text"] = "42"
    report = validate_annotations(pd.DataFrame([row]))

    assert report["status"] == "BLOCKED"
    assert any(item["code"] == "numeric_only_evidence" for item in report["issues"])


def test_invalid_label_and_confidence_are_critical() -> None:
    row = valid_row()
    row["primary_relation"] = "ally"
    row["annotator_confidence"] = 9
    report = validate_annotations(pd.DataFrame([row]))

    assert report["status"] == "BLOCKED"
    codes = {item["code"] for item in report["issues"]}
    assert "invalid_primary_relation" in codes
    assert "invalid_annotator_confidence" in codes


def test_duplicate_primary_secondary_is_warning() -> None:
    row = valid_row()
    row["secondary_relation"] = "cooperation"
    report = validate_annotations(pd.DataFrame([row]))

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert any(item["code"] == "duplicate_primary_secondary" for item in report["issues"])
