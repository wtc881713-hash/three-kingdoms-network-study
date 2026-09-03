import networkx as nx

from src.visualise_cooccurrence_network import (
    filter_core_graph,
    scaled_edge_widths,
    scaled_node_sizes,
)


def sample_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("A", weighted_degree=100, community_id=1)
    graph.add_node("B", weighted_degree=25, community_id=1)
    graph.add_node("C", weighted_degree=4, community_id=2)
    graph.add_edge("A", "B", weight=30)
    graph.add_edge("B", "C", weight=5)
    return graph


def test_filter_core_graph_retains_only_qualifying_edges_and_nodes() -> None:
    filtered = filter_core_graph(sample_graph(), minimum_weight=30)

    assert set(filtered.nodes) == {"A", "B"}
    assert set(filtered.edges) == {("A", "B")}


def test_visual_scales_are_monotonic() -> None:
    graph = sample_graph()

    node_sizes = dict(zip(graph.nodes, scaled_node_sizes(graph)))
    edge_widths = dict(zip(graph.edges, scaled_edge_widths(graph)))

    assert node_sizes["A"] > node_sizes["B"] > node_sizes["C"]
    assert edge_widths[("A", "B")] > edge_widths[("B", "C")]
