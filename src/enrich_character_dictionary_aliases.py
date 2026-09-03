"""Enrich the provisional character dictionary with reviewed formal aliases."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CHARACTER_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "provisional_canonical_characters_frequency_ge_10.csv"
)
ALIAS_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "reviewed_formal_alias_pairs.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "provisional_character_dictionary_with_aliases.csv"
)
CONFLICT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "alias_conflicts.csv"
)


def split_aliases(value: object) -> set[str]:
    """Split a semicolon-separated alias field into normalised values."""
    return {
        alias.strip()
        for alias in str(value).split(";")
        if alias.strip()
    }


def enrich_dictionary(
    characters: pd.DataFrame,
    alias_review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge reviewed aliases and report aliases assigned to multiple people."""
    required_character_columns = {
        "canonical_name",
        "source_aliases",
    }
    required_alias_columns = {
        "canonical_name",
        "alias",
        "alias_type",
        "review_decision",
    }
    if missing := required_character_columns - set(characters.columns):
        raise ValueError(f"Missing character columns: {sorted(missing)}")
    if missing := required_alias_columns - set(alias_review.columns):
        raise ValueError(f"Missing alias columns: {sorted(missing)}")

    enriched = characters.copy()
    canonical_names = set(enriched["canonical_name"].astype(str))
    aliases_by_character: defaultdict[str, set[str]] = defaultdict(set)
    types_by_relation: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for row in enriched.to_dict(orient="records"):
        canonical = str(row["canonical_name"]).strip()
        aliases_by_character[canonical].update(split_aliases(row["source_aliases"]))
        aliases_by_character[canonical].add(canonical)

    accepted_review = alias_review.loc[
        alias_review["review_decision"].astype(str).str.startswith("provisional_")
    ]
    for row in accepted_review.to_dict(orient="records"):
        canonical = str(row["canonical_name"]).strip()
        alias = str(row["alias"]).strip()
        if canonical not in canonical_names or not alias:
            continue
        aliases_by_character[canonical].add(alias)
        types_by_relation[(canonical, alias)].add(str(row["alias_type"]).strip())

    owners: defaultdict[str, set[str]] = defaultdict(set)
    for canonical, aliases in aliases_by_character.items():
        for alias in aliases:
            owners[alias].add(canonical)

    conflict_aliases = {
        alias: canonical_set
        for alias, canonical_set in owners.items()
        if len(canonical_set) > 1
    }
    conflict_rows = [
        {
            "alias": alias,
            "canonical_names": ";".join(sorted(canonical_set)),
            "canonical_count": len(canonical_set),
            "resolution_status": "requires_context",
        }
        for alias, canonical_set in sorted(conflict_aliases.items())
    ]
    conflicts = pd.DataFrame(
        conflict_rows,
        columns=[
            "alias",
            "canonical_names",
            "canonical_count",
            "resolution_status",
        ],
    )

    enriched["all_aliases"] = enriched["canonical_name"].map(
        lambda canonical: ";".join(sorted(aliases_by_character[str(canonical)]))
    )
    enriched["reviewed_formal_aliases"] = enriched["canonical_name"].map(
        lambda canonical: ";".join(
            sorted(
                alias
                for (owner, alias), _types in types_by_relation.items()
                if owner == str(canonical)
            )
        )
    )
    enriched["conflicting_aliases"] = enriched["canonical_name"].map(
        lambda canonical: ";".join(
            sorted(
                alias
                for alias in aliases_by_character[str(canonical)]
                if alias in conflict_aliases
            )
        )
    )
    enriched["alias_review_status"] = enriched["conflicting_aliases"].map(
        lambda value: "requires_context" if value else "provisional"
    )
    return enriched, conflicts


def main() -> None:
    """Create the enriched dictionary and a separate alias-conflict report."""
    characters = pd.read_csv(
        CHARACTER_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    alias_review = pd.read_csv(
        ALIAS_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    enriched, conflicts = enrich_dictionary(characters, alias_review)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    conflicts.to_csv(CONFLICT_FILE, index=False, encoding="utf-8-sig")

    added_relations = sum(
        bool(value) for value in enriched["reviewed_formal_aliases"]
    )
    print(f"Characters retained: {len(enriched)}")
    print(f"Characters with reviewed formal aliases: {added_relations}")
    print(f"Conflicting alias forms: {len(conflicts)}")
    print(f"Dictionary output: {OUTPUT_FILE}")
    print(f"Conflict output: {CONFLICT_FILE}")
    print("The dictionary remains provisional until human confirmation.")


if __name__ == "__main__":
    main()
