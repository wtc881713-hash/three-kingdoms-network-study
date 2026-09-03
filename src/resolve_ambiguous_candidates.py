"""Resolve ambiguous short names at the individual evidence-event level."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import pandas as pd

try:
    from src.extract_character_candidates import (
        CandidateEvent,
        collect_candidate_events,
        detect_chapters,
        read_text,
    )
except ModuleNotFoundError:
    from extract_character_candidates import (
        CandidateEvent,
        collect_candidate_events,
        detect_chapters,
        read_text,
    )


ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = (
    ROOT / "data" / "processed" / "three_kingdoms_gutenberg_simplified.txt"
)
REVIEW_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "candidate_review_frequency_ge_10.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "ambiguous_candidate_events.csv"
)


def propose_resolution(candidate: str, chapter: int) -> tuple[str, str, str]:
    """Return canonical name, status, and the transparent rule used."""
    if candidate == "肃":
        if chapter in {3, 9}:
            return "李肃", "proposed", "chapter_3_or_9"
        if 29 <= chapter <= 66:
            return "鲁肃", "proposed", "chapter_29_to_66"
        if chapter == 110:
            return "王肃", "proposed", "chapter_110"
    elif candidate == "昭":
        if chapter == 14:
            return "董昭", "proposed", "chapter_14"
        if 43 <= chapter <= 97:
            return "张昭", "proposed", "chapter_43_to_97"
        if chapter >= 108:
            return "司马昭", "proposed", "chapter_108_onward"
    elif candidate == "承":
        return "董承", "proposed", "single_narrative_identity"
    elif candidate == "平":
        if chapter == 23:
            return "吉平", "proposed", "chapter_23"
        if 65 <= chapter <= 99:
            return "王平", "proposed", "chapter_65_to_99"
    elif candidate == "干":
        return "孙乾", "proposed", "single_narrative_identity"
    elif candidate == "张":
        return "", "false_positive", "surname_or_collective_reference"
    elif candidate == "德":
        if chapter in {58, 70, 74}:
            return "庞德", "proposed", "pang_de_narrative_chapters"
    elif candidate == "亮":
        if chapter == 66:
            return "诸葛亮", "proposed", "chapter_66"
        if chapter >= 108:
            return "孙亮", "proposed", "chapter_108_onward"
    elif candidate == "攸":
        if chapter in {23, 46}:
            return "荀攸", "proposed", "xun_you_narrative_chapters"
        if chapter in {30, 33, 34}:
            return "许攸", "proposed", "xu_you_narrative_chapters"
    elif candidate == "兴":
        if 81 <= chapter <= 94:
            return "关兴", "proposed", "chapter_81_to_94"
    elif candidate == "进":
        if chapter in {2, 3}:
            return "何进", "proposed", "chapter_2_or_3"
        if chapter == 12:
            return "乐进", "proposed", "chapter_12"
        if chapter in {26, 92, 110}:
            return "", "false_positive", "verb_jin_or_jinyan"
    elif candidate == "定":
        if chapter == 28:
            return "关定", "proposed", "chapter_28"
        if chapter == 87:
            return "高定", "proposed", "chapter_87"
    elif candidate == "何":
        return "", "false_positive", "interrogative_or_surname_only"
    elif candidate == "范":
        if chapter in {15, 54, 77}:
            return "吕范", "proposed", "lv_fan_narrative_chapters"
        if chapter == 52:
            return "赵范", "proposed", "chapter_52"
        if chapter == 107:
            return "桓范", "proposed", "chapter_107"

    return "", "unresolved", "requires_passage_review"


def chapter_for_position(
    position: int,
    chapters: list[tuple[str, int]],
) -> tuple[int, str]:
    """Return the one-based chapter number and title for a text position."""
    starts = [start for _, start in chapters]
    index = bisect_right(starts, position) - 1
    if index < 0:
        return 0, "Preamble"
    return index + 1, chapters[index][0]


def build_event_review(
    events: list[CandidateEvent],
    chapters: list[tuple[str, int]],
    ambiguous_names: set[str],
) -> pd.DataFrame:
    """Build a deterministic event-level disambiguation table."""
    rows: list[dict[str, object]] = []
    selected = [
        event
        for event in events
        if event.candidate_name in ambiguous_names
    ]
    selected.sort(key=lambda event: (event.position, event.candidate_name))

    for index, event in enumerate(selected, start=1):
        chapter_number, chapter_title = chapter_for_position(
            event.position,
            chapters,
        )
        canonical_name, status, rule = propose_resolution(
            event.candidate_name,
            chapter_number,
        )
        if status == "unresolved" and event.evidence_type == "chapter_title":
            canonical_name = ""
            status = "false_positive"
            rule = "title_lexical_substring"
        rows.append(
            {
                "event_id": f"AMB{index:04d}",
                "candidate_name": event.candidate_name,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "position": event.position,
                "evidence_type": event.evidence_type,
                "evidence_snippet": event.snippet,
                "proposed_canonical_name": canonical_name,
                "resolution_status": status,
                "resolution_rule": rule,
                "human_canonical_name": "",
                "human_status": "",
                "human_notes": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Generate event-level records for all context-dependent candidates."""
    text = read_text(CORPUS_FILE)
    events, _ = collect_candidate_events(text)
    chapters = detect_chapters(text)
    review = pd.read_csv(
        REVIEW_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    ambiguous_names = set(
        review.loc[
            review["proposed_is_character"] == "uncertain",
            "candidate_name",
        ]
    )

    event_review = build_event_review(events, chapters, ambiguous_names)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    event_review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    counts = event_review["resolution_status"].value_counts().to_dict()
    print(f"Ambiguous candidates: {len(ambiguous_names)}")
    print(f"Ambiguous evidence events: {len(event_review)}")
    print(f"Proposed event resolutions: {counts.get('proposed', 0)}")
    print(f"False-positive events: {counts.get('false_positive', 0)}")
    print(f"Unresolved events: {counts.get('unresolved', 0)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("All proposed resolutions require human confirmation.")


if __name__ == "__main__":
    main()
