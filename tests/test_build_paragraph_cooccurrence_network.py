"""Tests for paragraph-level co-occurrence construction."""

import pandas as pd

from src.build_paragraph_cooccurrence_network import (
    analyse_network,
    assign_mentions_to_paragraphs,
    build_edges,
    weighted_pagerank,
)
from src.build_paragraph_cooccurrence_network import Paragraph


def test_assign_mentions_uses_paragraph_offsets() -> None:
    paragraphs = [
        Paragraph("P001-001", 1, "第一回 标题", 1, 10, 30, "刘备见关羽。"),
        Paragraph("P001-002", 1, "第一回 标题", 2, 32, 50, "曹操独行。"),
    ]
    mentions = pd.DataFrame(
        [
            {"mention_id": "M1", "chapter_number": 1, "global_start": 12, "canonical_name": "刘备", "matched_alias": "刘备"},
            {"mention_id": "M2", "chapter_number": 1, "global_start": 18, "canonical_name": "关羽", "matched_alias": "关羽"},
            {"mention_id": "M3", "chapter_number": 1, "global_start": 35, "canonical_name": "曹操", "matched_alias": "曹操"},
        ]
    )

    assigned = assign_mentions_to_paragraphs(mentions, paragraphs)

    assert list(assigned["paragraph_id"]) == ["P001-001", "P001-001", "P001-002"]


def test_edge_weight_counts_paragraphs_not_mentions() -> None:
    assignments = pd.DataFrame(
        [
            {"paragraph_id": "P001-001", "chapter_number": 1, "paragraph_text": "text one", "canonical_name": "刘备"},
            {"paragraph_id": "P001-001", "chapter_number": 1, "paragraph_text": "text one", "canonical_name": "刘备"},
            {"paragraph_id": "P001-001", "chapter_number": 1, "paragraph_text": "text one", "canonical_name": "关羽"},
            {"paragraph_id": "P002-001", "chapter_number": 2, "paragraph_text": "text two", "canonical_name": "刘备"},
            {"paragraph_id": "P002-001", "chapter_number": 2, "paragraph_text": "text two", "canonical_name": "关羽"},
        ]
    )

    edges = build_edges(assignments)

    assert len(edges) == 1
    assert edges.loc[0, "weight"] == 2
    assert edges.loc[0, "chapter_count"] == 2
    assert edges.loc[0, "relation_definition"] == "same_body_paragraph"


def test_network_keeps_isolated_dictionary_characters() -> None:
    dictionary = pd.DataFrame(
        [
            {"canonical_name": "刘备", "raw_mention_frequency": 10},
            {"canonical_name": "关羽", "raw_mention_frequency": 10},
            {"canonical_name": "曹操", "raw_mention_frequency": 10},
        ]
    )
    edges = pd.DataFrame([{"source": "刘备", "target": "关羽", "weight": 2}])

    nodes, summary = analyse_network(dictionary, edges)

    assert len(nodes) == 3
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 1
    assert summary["isolates"] == 1


def test_weighted_pagerank_is_normalised_without_scipy() -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge("刘备", "关羽", weight=3)
    graph.add_node("曹操")

    scores = weighted_pagerank(graph)

    assert abs(sum(scores.values()) - 1.0) < 1e-8
    assert scores["刘备"] == scores["关羽"]
    assert scores["曹操"] > 0
