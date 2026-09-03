import networkx as nx
import pandas as pd

from src.network_visualization import (
    character_method_rows,
    filter_graph,
    normalised_node_sizes,
    shared_layout,
    undirected_edge_table,
)


def test_dialogue_edges_are_combined_without_direction() -> None:
    edges = pd.DataFrame([
        {"source": "A", "target": "B", "weight": 2},
        {"source": "B", "target": "A", "weight": 3},
    ])
    result = undirected_edge_table("dialogue", edges)
    assert len(result) == 1
    assert result.loc[0, "weight"] == 5


def test_shared_layout_contains_union_of_nodes() -> None:
    first = nx.Graph([("A", "B")])
    second = nx.Graph([("B", "C")])
    assert set(shared_layout({"first": first, "second": second})) == {"A", "B", "C"}


def test_filter_graph_supports_threshold_and_ego_view() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=5)
    graph.add_edge("B", "C", weight=1)
    result = filter_graph(graph, minimum_weight=2, focal_character="A")
    assert set(result.nodes) == {"A", "B"}
    assert result.number_of_edges() == 1


def test_node_sizes_stay_in_expected_range() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=5)
    graph.add_edge("A", "C", weight=1)
    sizes = normalised_node_sizes(graph)
    assert min(sizes.values()) >= 7
    assert max(sizes.values()) <= 18
    assert sizes["A"] > sizes["B"]


def test_character_rows_keep_missing_methods_explicit() -> None:
    first = nx.Graph([("A", "B")])
    second = nx.Graph()
    second.add_node("B")
    rows = character_method_rows("A", {"cooccurrence": first, "dialogue": second})
    assert rows.loc[rows["method"] == "Co-occurrence", "active"].iloc[0]
    assert not rows.loc[rows["method"] == "Dialogue", "active"].iloc[0]


def test_all_downloadable_csv_tables_exist_and_are_readable() -> None:
    paths = [
        "outputs/cooccurrence/paragraph/nodes.csv",
        "outputs/cooccurrence/paragraph/edges.csv",
        "outputs/dialogue/named_speech/nodes.csv",
        "outputs/dialogue/named_speech/edges.csv",
        "outputs/semantic/multilingual_minilm/nodes.csv",
        "outputs/semantic/multilingual_minilm/edges.csv",
        "outputs/comparison/method_summary.csv",
        "outputs/comparison/edge_overlap.csv",
        "outputs/comparison/top_nodes.csv",
    ]
    for path in paths:
        table = pd.read_csv(path, encoding="utf-8-sig")
        assert not table.empty, path
