"""Tests for reviewed-alias enrichment."""

import pandas as pd

from src.enrich_character_dictionary_aliases import enrich_dictionary


def test_enrich_dictionary_adds_reviewed_aliases() -> None:
    characters = pd.DataFrame(
        [
            {
                "canonical_name": "刘备",
                "source_aliases": "先主;玄德",
            }
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "canonical_name": "刘备",
                "alias": "玄德",
                "alias_type": "courtesy_name",
                "review_decision": "provisional_accept",
            }
        ]
    )

    enriched, conflicts = enrich_dictionary(characters, aliases)

    assert enriched.loc[0, "all_aliases"] == "先主;刘备;玄德"
    assert enriched.loc[0, "reviewed_formal_aliases"] == "玄德"
    assert enriched.loc[0, "alias_review_status"] == "provisional"
    assert conflicts.empty


def test_enrich_dictionary_ignores_people_below_threshold() -> None:
    characters = pd.DataFrame(
        [{"canonical_name": "刘备", "source_aliases": "玄德"}]
    )
    aliases = pd.DataFrame(
        [
            {
                "canonical_name": "秦宓",
                "alias": "子敕",
                "alias_type": "courtesy_name",
                "review_decision": "provisional_corrected",
            }
        ]
    )

    enriched, _ = enrich_dictionary(characters, aliases)

    assert "子敕" not in enriched.loc[0, "all_aliases"]


def test_enrich_dictionary_reports_alias_conflicts() -> None:
    characters = pd.DataFrame(
        [
            {"canonical_name": "徐晃", "source_aliases": "徐晃"},
            {"canonical_name": "管辂", "source_aliases": "管辂"},
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "canonical_name": "徐晃",
                "alias": "公明",
                "alias_type": "courtesy_name",
                "review_decision": "provisional_accept",
            },
            {
                "canonical_name": "管辂",
                "alias": "公明",
                "alias_type": "courtesy_name",
                "review_decision": "provisional_accept",
            },
        ]
    )

    enriched, conflicts = enrich_dictionary(characters, aliases)

    assert len(conflicts) == 1
    assert conflicts.loc[0, "alias"] == "公明"
    assert conflicts.loc[0, "canonical_names"] == "徐晃;管辂"
    assert set(enriched["alias_review_status"]) == {"requires_context"}
