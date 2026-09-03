"""Build a conservative named-speaker dialogue network from the novel."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from src.build_paragraph_cooccurrence_network import weighted_pagerank
except ModuleNotFoundError:
    from build_paragraph_cooccurrence_network import weighted_pagerank


ROOT = Path(__file__).resolve().parent.parent
DICTIONARY_FILE = (
    ROOT / "data" / "metadata" / "gutenberg" / "final_character_dictionary.csv"
)
PARAGRAPH_MENTIONS_FILE = (
    ROOT / "outputs" / "cooccurrence" / "paragraph" / "paragraph_mentions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "dialogue" / "named_speech"

SPEECH_MODIFIERS = (
    "大笑|笑|问|答|怒|惊|叹|喜|喝|骂|哭|厉声|高声|正色|从容|慨然|欣然"
)
TARGET_VERBS = "谓|问|告|对|向|语"


@dataclass(frozen=True)
class SpeechTurn:
    """One speech marker with an explicitly named speaker."""

    speaker: str
    alias: str
    start: int


def load_alias_mapping(dictionary: pd.DataFrame) -> dict[str, str]:
    """Create a unique longest-match alias-to-character mapping."""
    mapping: dict[str, str] = {}
    for row in dictionary.to_dict(orient="records"):
        canonical = str(row["canonical_name"])
        aliases = [canonical]
        aliases.extend(
            alias.strip()
            for alias in str(row.get("usable_aliases", "")).split(";")
            if alias.strip()
        )
        for alias in aliases:
            existing = mapping.get(alias)
            if existing is not None and existing != canonical:
                raise ValueError(f"Alias {alias!r} maps to multiple characters.")
            mapping[alias] = canonical
    return mapping


def alias_pattern(alias_mapping: dict[str, str]) -> str:
    """Build a longest-first escaped alias alternation."""
    aliases = sorted(alias_mapping, key=lambda item: (-len(item), item))
    return "(?:" + "|".join(re.escape(alias) for alias in aliases) + ")"


def extract_speech_turns(
    text: str,
    alias_mapping: dict[str, str],
) -> list[SpeechTurn]:
    """Extract speech turns where a validated alias directly precedes '曰'."""
    aliases = alias_pattern(alias_mapping)
    pattern = re.compile(
        rf"(?P<speaker>{aliases})(?P<modifier>{SPEECH_MODIFIERS})?曰[：:]?[「“]"
    )
    return [
        SpeechTurn(
            speaker=alias_mapping[match.group("speaker")],
            alias=match.group("speaker"),
            start=match.start(),
        )
        for match in pattern.finditer(text)
    ]


def extract_explicit_targets(
    text: str,
    alias_mapping: dict[str, str],
) -> list[dict[str, object]]:
    """Extract constructions such as 'Liu Bei said to Zhuge Liang'."""
    aliases = alias_pattern(alias_mapping)
    pattern = re.compile(
        rf"(?P<source>{aliases})(?P<verb>{TARGET_VERBS})(?P<target>{aliases})"
        rf"(?P<modifier>{SPEECH_MODIFIERS})?曰[：:]?[「“]"
    )
    rows = []
    for match in pattern.finditer(text):
        source_alias = match.group("source")
        target_alias = match.group("target")
        source = alias_mapping[source_alias]
        target = alias_mapping[target_alias]
        if source == target:
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "source_alias": source_alias,
                "target_alias": target_alias,
                "event_start": match.start(),
                "extraction_rule": "explicit_named_target",
            }
        )
    return rows


def extract_adjacent_turns(
    text: str,
    alias_mapping: dict[str, str],
) -> list[dict[str, object]]:
    """Connect consecutive, different named speakers in one paragraph."""
    turns = extract_speech_turns(text, alias_mapping)
    rows = []
    for first, second in zip(turns, turns[1:]):
        if first.speaker == second.speaker:
            continue
        rows.append(
            {
                "source": first.speaker,
                "target": second.speaker,
                "source_alias": first.alias,
                "target_alias": second.alias,
                "event_start": second.start,
                "extraction_rule": "adjacent_named_turns",
            }
        )
    return rows


def extract_dialogue_events(
    paragraphs: pd.DataFrame,
    alias_mapping: dict[str, str],
) -> pd.DataFrame:
    """Extract deduplicated dialogue events with paragraph evidence."""
    rows: list[dict[str, object]] = []
    for paragraph in paragraphs.to_dict(orient="records"):
        text = str(paragraph["paragraph_text"])
        events = extract_explicit_targets(text, alias_mapping)
        events.extend(extract_adjacent_turns(text, alias_mapping))
        seen: set[tuple[str, str, str, int]] = set()
        for event in events:
            key = (
                str(event["source"]),
                str(event["target"]),
                str(event["extraction_rule"]),
                int(event["event_start"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "event_id": "",
                    "paragraph_id": paragraph["paragraph_id"],
                    "chapter_number": int(paragraph["chapter_number"]),
                    "chapter_title": paragraph["chapter_title"],
                    **event,
                    "paragraph_text": text,
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["chapter_number"]),
            str(row["paragraph_id"]),
            int(row["event_start"]),
            str(row["extraction_rule"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["event_id"] = f"DIALOGUE{index:06d}"
    return pd.DataFrame(rows)


def build_edges(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directed dialogue events with evidence and rule counts."""
    counts: Counter[tuple[str, str]] = Counter()
    chapters: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    rule_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    evidence: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in events.to_dict(orient="records"):
        pair = (str(row["source"]), str(row["target"]))
        counts[pair] += 1
        chapters[pair].add(int(row["chapter_number"]))
        rule_counts[pair][str(row["extraction_rule"])] += 1
        if len(evidence[pair]) < 3:
            evidence[pair].append(
                f"Chapter {int(row['chapter_number'])}, {row['paragraph_id']}: "
                f"{row['paragraph_text']}"
            )
    rows = []
    for (source, target), weight in counts.items():
        rules = rule_counts[(source, target)]
        rows.append(
            {
                "source": source,
                "target": target,
                "weight": weight,
                "chapter_count": len(chapters[(source, target)]),
                "explicit_target_events": rules["explicit_named_target"],
                "adjacent_turn_events": rules["adjacent_named_turns"],
                "sample_evidence": " || ".join(evidence[(source, target)]),
                "relation_definition": "named_dialogue_transition",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def analyse_network(
    dictionary: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate directed metrics and undirected structural metrics."""
    directed = nx.DiGraph()
    directed.add_nodes_from(dictionary["canonical_name"].astype(str))
    for row in edges.to_dict(orient="records"):
        directed.add_edge(row["source"], row["target"], weight=int(row["weight"]))
    undirected = nx.Graph()
    undirected.add_nodes_from(directed.nodes)
    for source, target, data in directed.edges(data=True):
        weight = int(data["weight"])
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    pagerank = weighted_pagerank(undirected)
    betweenness = nx.betweenness_centrality(undirected, weight=None)
    active = undirected.subgraph([node for node in undirected if undirected.degree(node) > 0])
    communities = (
        list(nx.community.louvain_communities(active, weight="weight", seed=42))
        if active.number_of_nodes()
        else []
    )
    community_ids = {
        node: index
        for index, community in enumerate(communities, start=1)
        for node in community
    }
    rows = []
    for node in directed.nodes:
        rows.append(
            {
                "canonical_name": node,
                "in_degree": directed.in_degree(node),
                "out_degree": directed.out_degree(node),
                "in_weight": directed.in_degree(node, weight="weight"),
                "out_weight": directed.out_degree(node, weight="weight"),
                "undirected_degree": undirected.degree(node),
                "weighted_degree": undirected.degree(node, weight="weight"),
                "betweenness_centrality": betweenness[node],
                "pagerank": pagerank[node],
                "community_id": community_ids.get(node, 0),
            }
        )
    nodes = pd.DataFrame(rows).sort_values(
        ["weighted_degree", "canonical_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary = {
        "nodes_all": directed.number_of_nodes(),
        "nodes_with_events": active.number_of_nodes(),
        "directed_edges": directed.number_of_edges(),
        "undirected_edges": undirected.number_of_edges(),
        "event_count": int(edges["weight"].sum()) if not edges.empty else 0,
        "explicit_target_events": int(edges["explicit_target_events"].sum()) if not edges.empty else 0,
        "adjacent_turn_events": int(edges["adjacent_turn_events"].sum()) if not edges.empty else 0,
        "communities_active_nodes": len(communities),
    }
    return nodes, summary


def format_summary(summary: dict[str, object]) -> str:
    """Format a plain-text audit summary."""
    return "\n".join(
        [
            "Named-Speech Dialogue Network Summary",
            "=====================================",
            "Relation definition: explicit named target or consecutive different named speakers within one body paragraph.",
            f"Dictionary nodes: {summary['nodes_all']}",
            f"Nodes with dialogue events: {summary['nodes_with_events']}",
            f"Directed edges: {summary['directed_edges']}",
            f"Undirected projection edges: {summary['undirected_edges']}",
            f"Dialogue events: {summary['event_count']}",
            f"Explicit named-target events: {summary['explicit_target_events']}",
            f"Adjacent named-turn events: {summary['adjacent_turn_events']}",
            f"Louvain communities among active nodes: {summary['communities_active_nodes']}",
        ]
    )


def main() -> None:
    """Build and save the conservative named-speech dialogue network."""
    dictionary = pd.read_csv(DICTIONARY_FILE, encoding="utf-8-sig", keep_default_na=False)
    mentions = pd.read_csv(
        PARAGRAPH_MENTIONS_FILE, encoding="utf-8-sig", keep_default_na=False
    )
    paragraphs = mentions[
        ["paragraph_id", "chapter_number", "chapter_title", "paragraph_text"]
    ].drop_duplicates("paragraph_id")
    alias_mapping = load_alias_mapping(dictionary)
    events = extract_dialogue_events(paragraphs, alias_mapping)
    edges = build_edges(events)
    nodes, summary = analyse_network(dictionary, edges)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_DIR / "dialogue_events.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUTPUT_DIR / "edges.csv", index=False, encoding="utf-8-sig")
    nodes.to_csv(OUTPUT_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
    report = format_summary(summary)
    (OUTPUT_DIR / "network_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
