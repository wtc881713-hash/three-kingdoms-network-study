"""Build a weighted paragraph-level character co-occurrence network."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
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
    ROOT / "data" / "metadata" / "gutenberg" / "final_character_dictionary.csv"
)
MENTION_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "character_mention_events.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "cooccurrence" / "paragraph"


@dataclass(frozen=True)
class Paragraph:
    """One body paragraph with stable corpus offsets."""

    paragraph_id: str
    chapter_number: int
    chapter_title: str
    paragraph_number: int
    start: int
    end: int
    text: str


def segment_paragraphs(text: str) -> list[Paragraph]:
    """Segment non-heading body paragraphs using blank-line boundaries."""
    chapters = detect_chapters(text)
    if len(chapters) != 120:
        raise ValueError(f"Expected 120 chapters, found {len(chapters)}.")

    paragraphs: list[Paragraph] = []
    for chapter_index, (title, chapter_start) in enumerate(chapters):
        chapter_end = (
            chapters[chapter_index + 1][1]
            if chapter_index + 1 < len(chapters)
            else len(text)
        )
        chapter_text = text[chapter_start:chapter_end]
        paragraph_number = 0
        for match in re.finditer(r"[^\s].*?(?=\n\s*\n|\Z)", chapter_text, re.DOTALL):
            paragraph_text = match.group(0).strip()
            if not paragraph_text or paragraph_text == title:
                continue
            paragraph_number += 1
            start = chapter_start + match.start()
            end = chapter_start + match.end()
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"P{chapter_index + 1:03d}-{paragraph_number:03d}",
                    chapter_number=chapter_index + 1,
                    chapter_title=title,
                    paragraph_number=paragraph_number,
                    start=start,
                    end=end,
                    text=re.sub(r"\s+", " ", paragraph_text).strip(),
                )
            )
    return paragraphs


def assign_mentions_to_paragraphs(
    mentions: pd.DataFrame,
    paragraphs: list[Paragraph],
) -> pd.DataFrame:
    """Assign mention events to paragraph spans by their global offsets."""
    rows: list[dict[str, object]] = []
    mentions_by_chapter = {
        int(chapter): group.sort_values("global_start", kind="stable")
        for chapter, group in mentions.groupby("chapter_number")
    }
    for paragraph in paragraphs:
        chapter_mentions = mentions_by_chapter.get(paragraph.chapter_number)
        if chapter_mentions is None:
            continue
        inside = chapter_mentions.loc[
            (chapter_mentions["global_start"] >= paragraph.start)
            & (chapter_mentions["global_start"] < paragraph.end)
        ]
        for row in inside.to_dict(orient="records"):
            rows.append(
                {
                    "paragraph_id": paragraph.paragraph_id,
                    "chapter_number": paragraph.chapter_number,
                    "chapter_title": paragraph.chapter_title,
                    "paragraph_number": paragraph.paragraph_number,
                    "paragraph_text": paragraph.text,
                    "mention_id": row["mention_id"],
                    "canonical_name": row["canonical_name"],
                    "matched_alias": row["matched_alias"],
                }
            )
    return pd.DataFrame(rows)


def build_edges(
    paragraph_mentions: pd.DataFrame,
    evidence_limit: int = 3,
) -> pd.DataFrame:
    """Count one undirected co-occurrence per character pair per paragraph."""
    edge_counts: Counter[tuple[str, str]] = Counter()
    chapters: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    evidence: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for _, group in paragraph_mentions.groupby("paragraph_id", sort=True):
        characters = sorted(set(group["canonical_name"].astype(str)))
        if len(characters) < 2:
            continue
        first = group.iloc[0]
        evidence_text = (
            f"Chapter {int(first['chapter_number'])}, "
            f"{first['paragraph_id']}: {first['paragraph_text']}"
        )
        for source, target in combinations(characters, 2):
            pair = (source, target)
            edge_counts[pair] += 1
            chapters[pair].add(int(first["chapter_number"]))
            if len(evidence[pair]) < evidence_limit:
                evidence[pair].append(evidence_text)

    rows = [
        {
            "source": source,
            "target": target,
            "weight": weight,
            "chapter_count": len(chapters[(source, target)]),
            "chapters": ";".join(map(str, sorted(chapters[(source, target)]))),
            "sample_evidence": " || ".join(evidence[(source, target)]),
            "relation_definition": "same_body_paragraph",
        }
        for (source, target), weight in edge_counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def weighted_pagerank(
    graph: nx.Graph,
    damping: float = 0.85,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> dict[str, float]:
    """Calculate weighted PageRank without optional SciPy dependencies."""
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    count = len(nodes)
    scores = {node: 1.0 / count for node in nodes}
    strengths = dict(graph.degree(weight="weight"))

    for _ in range(max_iterations):
        dangling = sum(scores[node] for node in nodes if strengths[node] == 0)
        base = (1.0 - damping) / count + damping * dangling / count
        updated = {node: base for node in nodes}
        for source in nodes:
            if strengths[source] == 0:
                continue
            for target, attributes in graph[source].items():
                share = float(attributes.get("weight", 1.0)) / strengths[source]
                updated[target] += damping * scores[source] * share
        difference = sum(abs(updated[node] - scores[node]) for node in nodes)
        scores = updated
        if difference < tolerance:
            break
    return scores


def analyse_network(
    dictionary: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate node metrics and whole-network statistics."""
    graph = nx.Graph()
    graph.add_nodes_from(dictionary["canonical_name"].astype(str))
    for row in edges.to_dict(orient="records"):
        graph.add_edge(row["source"], row["target"], weight=int(row["weight"]))

    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight=None)
    pagerank = weighted_pagerank(graph)

    communities = list(
        nx.community.louvain_communities(graph, weight="weight", seed=42)
    )
    community_id = {
        character: index
        for index, community in enumerate(communities, start=1)
        for character in community
    }

    node_rows = []
    mention_lookup = dictionary.set_index("canonical_name")[
        "raw_mention_frequency"
    ].to_dict()
    for character in graph.nodes:
        node_rows.append(
            {
                "canonical_name": character,
                "raw_mention_frequency": int(mention_lookup[character]),
                "degree": degree[character],
                "weighted_degree": int(strength[character]),
                "degree_centrality": degree_centrality[character],
                "betweenness_centrality": betweenness[character],
                "pagerank": pagerank[character],
                "community_id": community_id[character],
            }
        )
    nodes = pd.DataFrame(node_rows).sort_values(
        ["weighted_degree", "canonical_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    isolates = list(nx.isolates(graph))
    summary = {
        "relation_definition": "Characters share a body paragraph.",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected_components": nx.number_connected_components(graph),
        "isolates": len(isolates),
        "communities": len(communities),
        "total_edge_weight": int(sum(strength.values()) / 2),
    }
    return nodes, summary


def format_summary(summary: dict[str, object], paragraph_count: int) -> str:
    """Format an auditable plain-text network summary."""
    return "\n".join(
        [
            "Paragraph Co-occurrence Network Summary",
            "=======================================",
            f"Relation definition: {summary['relation_definition']}",
            f"Body paragraphs: {paragraph_count}",
            f"Nodes: {summary['node_count']}",
            f"Edges: {summary['edge_count']}",
            f"Total edge weight: {summary['total_edge_weight']}",
            f"Density: {summary['density']:.6f}",
            f"Connected components: {summary['connected_components']}",
            f"Isolates: {summary['isolates']}",
            f"Louvain communities: {summary['communities']}",
        ]
    )


def main() -> None:
    """Build and save the paragraph-level co-occurrence baseline."""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    dictionary = pd.read_csv(DICTIONARY_FILE, encoding="utf-8-sig", keep_default_na=False)
    mentions = pd.read_csv(MENTION_FILE, encoding="utf-8-sig", keep_default_na=False)
    mentions["global_start"] = mentions["global_start"].astype(int)

    paragraphs = segment_paragraphs(text)
    assignments = assign_mentions_to_paragraphs(mentions, paragraphs)
    edges = build_edges(assignments)
    nodes, summary = analyse_network(dictionary, edges)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(OUTPUT_DIR / "paragraph_mentions.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUTPUT_DIR / "edges.csv", index=False, encoding="utf-8-sig")
    nodes.to_csv(OUTPUT_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
    report = format_summary(summary, len(paragraphs))
    (OUTPUT_DIR / "network_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
