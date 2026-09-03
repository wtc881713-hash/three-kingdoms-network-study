import networkx as nx

from src.compare_network_methods import canonical_edge_set, graph_summary, pairwise_overlap


def test_canonical_edge_set_ignores_endpoint_order() -> None:
    graph = nx.Graph()
    graph.add_edge("B", "A", weight=1)

    assert canonical_edge_set(graph) == {("A", "B")}


def test_pairwise_overlap_calculates_jaccard() -> None:
    first = nx.Graph([("A", "B"), ("B", "C")])
    second = nx.Graph([("A", "B"), ("C", "D")])

    overlap = pairwise_overlap({"first": first, "second": second})

    assert overlap.loc[0, "shared_edges"] == 1
    assert overlap.loc[0, "union_edges"] == 3
    assert overlap.loc[0, "jaccard_similarity"] == 1 / 3


def test_graph_summary_uses_only_active_nodes() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edge("A", "B", weight=2)

    summary = graph_summary("cooccurrence", graph)

    assert summary["active_nodes"] == 2
    assert summary["edges"] == 1
    assert summary["total_edge_weight"] == 2
