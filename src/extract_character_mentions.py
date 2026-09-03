"""Extract conservative full-text character mentions from the derived corpus."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    from src.extract_character_candidates import detect_chapters
except ModuleNotFoundError:
    from extract_character_candidates import detect_chapters


ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = (
    ROOT / "data" / "processed" / "three_kingdoms_gutenberg_simplified.txt"
)
DICTIONARY_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "provisional_character_dictionary_with_aliases.csv"
)
EVENT_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "character_mention_events.csv"
)
SUMMARY_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "character_mention_summary.csv"
)

EVENT_COLUMNS = [
    "mention_id",
    "chapter_number",
    "chapter_title",
    "canonical_name",
    "matched_alias",
    "global_start",
    "chapter_start",
    "context",
]


def build_alias_index(
    dictionary: pd.DataFrame,
    minimum_alias_length: int = 2,
) -> dict[str, str]:
    """Build an unambiguous alias-to-character index.

    One-character aliases and aliases explicitly marked as conflicting are
    excluded. These conservative rules reduce false matches in classical
    Chinese prose.
    """
    if minimum_alias_length < 1:
        raise ValueError("Minimum alias length must be at least 1.")

    owners: defaultdict[str, set[str]] = defaultdict(set)
    blocked: set[str] = set()
    for row in dictionary.to_dict(orient="records"):
        canonical = str(row["canonical_name"]).strip()
        conflicts = {
            value.strip()
            for value in str(row.get("conflicting_aliases", "")).split(";")
            if value.strip()
        }
        blocked.update(conflicts)
        aliases = {
            value.strip()
            for value in str(row["all_aliases"]).split(";")
            if value.strip()
        }
        aliases.add(canonical)
        for alias in aliases:
            if len(alias) >= minimum_alias_length:
                owners[alias].add(canonical)

    return {
        alias: next(iter(canonical_names))
        for alias, canonical_names in owners.items()
        if len(canonical_names) == 1 and alias not in blocked
    }


def chapter_spans(text: str) -> list[tuple[int, str, int, int]]:
    """Return chapter number, title, start, and end offsets."""
    chapters = detect_chapters(text)
    if not chapters:
        raise ValueError("No chapter headings were detected.")
    return [
        (
            number,
            title,
            start,
            chapters[index + 1][1] if index + 1 < len(chapters) else len(text),
        )
        for index, (title, start) in enumerate(chapters)
        for number in [index + 1]
    ]


def extract_mentions(
    text: str,
    alias_index: dict[str, str],
    context_radius: int = 24,
) -> pd.DataFrame:
    """Extract longest, non-overlapping alias matches chapter by chapter."""
    if not alias_index:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    if context_radius < 0:
        raise ValueError("Context radius cannot be negative.")

    alternatives = sorted(alias_index, key=lambda value: (-len(value), value))
    pattern = re.compile("|".join(re.escape(alias) for alias in alternatives))
    rows: list[dict[str, object]] = []

    for chapter_number, chapter_title, start, end in chapter_spans(text):
        chapter_text = text[start:end]
        for match in pattern.finditer(chapter_text):
            alias = match.group(0)
            local_start = match.start()
            context_start = max(0, local_start - context_radius)
            context_end = min(
                len(chapter_text),
                match.end() + context_radius,
            )
            context = re.sub(
                r"\s+",
                " ",
                chapter_text[context_start:context_end],
            ).strip()
            rows.append(
                {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "canonical_name": alias_index[alias],
                    "matched_alias": alias,
                    "global_start": start + local_start,
                    "chapter_start": local_start,
                    "context": context,
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events.insert(
        0,
        "mention_id",
        [f"MENTION{index:06d}" for index in range(1, len(events) + 1)],
    )
    return events[EVENT_COLUMNS]


def summarise_mentions(
    events: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise raw mention counts and chapter coverage by character."""
    event_counts = Counter(events["canonical_name"]) if not events.empty else Counter()
    chapter_counts = (
        events.groupby("canonical_name")["chapter_number"].nunique().to_dict()
        if not events.empty
        else {}
    )
    alias_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in events.to_dict(orient="records"):
        alias_counts[str(row["canonical_name"])][str(row["matched_alias"])] += 1

    rows = []
    for row in dictionary.to_dict(orient="records"):
        canonical = str(row["canonical_name"])
        rows.append(
            {
                "character_id": row["character_id"],
                "canonical_name": canonical,
                "raw_mention_frequency": event_counts[canonical],
                "chapter_coverage": chapter_counts.get(canonical, 0),
                "matched_alias_breakdown": ";".join(
                    f"{alias}:{count}"
                    for alias, count in alias_counts[canonical].most_common()
                ),
                "extraction_event_frequency": int(row["frequency"]),
                "frequency_definition": (
                    "Longest non-overlapping exact alias matches; "
                    "one-character and conflicting aliases excluded."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["raw_mention_frequency", "canonical_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    """Extract mention events and save event-level and summary datasets."""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    dictionary = pd.read_csv(
        DICTIONARY_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    alias_index = build_alias_index(dictionary)
    events = extract_mentions(text, alias_index)
    summary = summarise_mentions(events, dictionary)

    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print(f"Chapters scanned: {len(chapter_spans(text))}")
    print(f"Characters in dictionary: {len(dictionary)}")
    print(f"Usable unambiguous aliases: {len(alias_index)}")
    print(f"Character mention events: {len(events)}")
    print(f"Characters with at least one mention: {(summary['raw_mention_frequency'] > 0).sum()}")
    print(f"Event output: {EVENT_FILE}")
    print(f"Summary output: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
