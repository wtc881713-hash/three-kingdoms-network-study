"""Build passage-level character-pair instances from validated mentions."""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.build_paragraph_cooccurrence_network import segment_paragraphs
except ModuleNotFoundError:
    from build_paragraph_cooccurrence_network import segment_paragraphs


CORPUS_FILE = ROOT / "data" / "processed" / "three_kingdoms_gutenberg_simplified.txt"
MENTION_FILE = ROOT / "data" / "metadata" / "gutenberg" / "character_mention_events.csv"
OUTPUT_FILE = ROOT / "data" / "annotation" / "relation_instance_pool.csv"
SENTENCE_PATTERN = re.compile(r"[^。！？!?；;]+[。！？!?；;]?", re.DOTALL)


def split_sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Split a paragraph while retaining character offsets."""
    sentences = []
    for match in SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append((match.start(), match.end(), sentence))
    return sentences or [(0, len(text), text)]


def sentence_index(offset: int, sentences: list[tuple[int, int, str]]) -> int:
    """Return the sentence containing a paragraph-relative offset."""
    for index, (start, end, _) in enumerate(sentences):
        if start <= offset < end:
            return index
    return max(0, len(sentences) - 1)


def insert_entity_markers(
    passage: str,
    surface_a: str,
    surface_b: str,
) -> str:
    """Mark the first non-overlapping surface occurrence for both entities."""
    matches = []
    for marker, surface in (("CHAR_A", surface_a), ("CHAR_B", surface_b)):
        match = re.search(re.escape(surface), passage)
        if match:
            matches.append((match.start(), match.end(), marker, surface))
    marked = passage
    for start, end, marker, surface in sorted(matches, reverse=True):
        marked = marked[:start] + f"[{marker}] {surface} [/{marker}]" + marked[end:]
    return marked


def build_relation_instances(
    text: str,
    mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one auditable instance for each pair in each body paragraph."""
    paragraphs = segment_paragraphs(text)
    mentions = mentions.copy()
    mentions["global_start"] = mentions["global_start"].astype(int)
    by_chapter = {
        int(chapter): group.sort_values("global_start", kind="stable")
        for chapter, group in mentions.groupby("chapter_number")
    }
    rows: list[dict[str, object]] = []

    for paragraph in paragraphs:
        chapter_mentions = by_chapter.get(paragraph.chapter_number)
        if chapter_mentions is None:
            continue
        inside = chapter_mentions.loc[
            (chapter_mentions["global_start"] >= paragraph.start)
            & (chapter_mentions["global_start"] < paragraph.end)
        ].copy()
        characters = sorted(inside["canonical_name"].astype(str).unique())
        if len(characters) < 2:
            continue

        sentences = split_sentences_with_offsets(paragraph.text)
        for character_a, character_b in combinations(characters, 2):
            mention_a = inside.loc[inside["canonical_name"] == character_a].iloc[0]
            mention_b = inside.loc[inside["canonical_name"] == character_b].iloc[0]
            relative_a = max(0, int(mention_a["global_start"]) - paragraph.start)
            relative_b = max(0, int(mention_b["global_start"]) - paragraph.start)
            index_a = sentence_index(relative_a, sentences)
            index_b = sentence_index(relative_b, sentences)
            first_index = max(0, min(index_a, index_b) - 1)
            last_index = min(len(sentences) - 1, max(index_a, index_b) + 1)
            local_start = sentences[first_index][0]
            local_end = sentences[last_index][1]
            passage = paragraph.text[local_start:local_end].strip()
            surface_a = str(mention_a["matched_alias"])
            surface_b = str(mention_b["matched_alias"])
            rows.append(
                {
                    "instance_id": "",
                    "chapter_id": paragraph.chapter_number,
                    "paragraph_id": paragraph.paragraph_id,
                    "passage_start": paragraph.start + local_start,
                    "passage_end": paragraph.start + local_end,
                    "character_a": character_a,
                    "character_b": character_b,
                    "surface_a": surface_a,
                    "surface_b": surface_b,
                    "passage": passage,
                    "marked_passage": insert_entity_markers(passage, surface_a, surface_b),
                    "candidate_source": "validated_same_body_paragraph",
                    "character_count": len(characters),
                    "sentence_span": abs(index_a - index_b) + 1,
                    "character_distance": abs(relative_a - relative_b),
                    "uses_alias": int(surface_a != character_a or surface_b != character_b),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["chapter_id", "passage_start", "character_a", "character_b"],
        kind="stable",
    ).reset_index(drop=True)
    result["instance_id"] = [f"REL{i:06d}" for i in range(1, len(result) + 1)]
    return result


def main() -> None:
    """Build and save the relation-instance pool."""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    mentions = pd.read_csv(MENTION_FILE, encoding="utf-8-sig", keep_default_na=False)
    instances = build_relation_instances(text, mentions)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    instances.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Relation instances: {len(instances):,}")
    print(f"Chapters represented: {instances['chapter_id'].nunique()}")
    print(f"Character pairs: {instances[['character_a', 'character_b']].drop_duplicates().shape[0]:,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
