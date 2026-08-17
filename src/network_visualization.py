"""Shared visualisation utilities for the three character-network methods."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import networkx as nx
import pandas as pd
from pyvis.network import Network


ROOT = Path(__file__).resolve().parent.parent

METHOD_CONFIG = {
    "cooccurrence": {
        "label": "Co-occurrence",
        "colour": "#2563EB",
        "meaning": "Two characters appear in the same paragraph.",
        "warning": "This does not always mean direct contact.",
    },
    "dialogue": {
        "label": "Dialogue",
        "colour": "#DC2626",
        "meaning": "A named character speaks to another named character.",
        "warning": "Some speech links are inferred and may be wrong.",
    },
    "semantic_context": {
        "label": "Semantic context",
        "colour": "#B7791F",
        "meaning": "The text around two characters has a similar meaning.",
        "warning": "This does not prove that the characters meet.",
    },
}

METHOD_PATHS = {
    "cooccurrence": (
        ROOT / "outputs" / "cooccurrence" / "paragraph" / "nodes.csv",
        ROOT / "outputs" / "cooccurrence" / "paragraph" / "edges.csv",
    ),
    "dialogue": (
        ROOT / "outputs" / "dialogue" / "named_speech" / "nodes.csv",
        ROOT / "outputs" / "dialogue" / "named_speech" / "edges.csv",
    ),
    "semantic_context": (
        ROOT / "outputs" / "semantic" / "multilingual_minilm" / "nodes.csv",
        ROOT / "outputs" / "semantic" / "multilingual_minilm" / "edges.csv",
    ),
}


def load_network_tables(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and edge tables for a supported network method."""
    if method not in METHOD_PATHS:
        raise ValueError(f"Unsupported method: {method}")
    nodes_path, edges_path = METHOD_PATHS[method]
    nodes = pd.read_csv(nodes_path, encoding="utf-8-sig", keep_default_na=False)
    edges = pd.read_csv(edges_path, encoding="utf-8-sig", keep_default_na=False)
    return nodes, edges


def undirected_edge_table(method: str, edges: pd.DataFrame) -> pd.DataFrame:
    """Return one weighted row per unordered pair, combining dialogue directions."""
    weight_column = "similarity" if method == "semantic_context" else "weight"
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in edges.to_dict(orient="records"):
        source, target = sorted((str(row["source"]), str(row["target"])))
        key = (source, target)
        weight = float(row.get(weight_column, row.get("weight", 1.0)))
        if key not in rows:
            rows[key] = {"source": source, "target": target, "weight": 0.0}
        rows[key]["weight"] = float(rows[key]["weight"]) + weight
    return pd.DataFrame(rows.values(), columns=["source", "target", "weight"])


def build_method_graph(method: str) -> nx.Graph:
    """Build an undirected graph with node metrics from the generated outputs."""
    nodes, edges = load_network_tables(method)
    graph = nx.Graph(method=method)
    for row in nodes.to_dict(orient="records"):
        name = str(row["canonical_name"])
        graph.add_node(name, **row)
    for row in undirected_edge_table(method, edges).to_dict(orient="records"):
        graph.add_edge(str(row["source"]), str(row["target"]), weight=float(row["weight"]))
    return graph


def shared_layout(graphs: Mapping[str, nx.Graph], seed: int = 42) -> dict[str, tuple[float, float]]:
    """Create one deterministic union-graph layout shared by all method views."""
    union = nx.Graph()
    for graph in graphs.values():
        union.add_nodes_from(graph.nodes)
        union.add_edges_from(graph.edges)
    if not union:
        return {}
    positions = nx.spring_layout(union, seed=seed, k=1.8 / math.sqrt(max(len(union), 1)))
    return {str(node): (float(x), float(y)) for node, (x, y) in positions.items()}


def filter_graph(
    graph: nx.Graph,
    minimum_weight: float = 0.0,
    focal_character: str | None = None,
) -> nx.Graph:
    """Filter edges by weight and optionally return a focal node's ego network."""
    filtered = nx.Graph(method=graph.graph.get("method"))
    filtered.add_nodes_from(graph.nodes(data=True))
    filtered.add_edges_from(
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if float(data.get("weight", 1.0)) >= minimum_weight
    )
    if focal_character and focal_character in filtered:
        return nx.ego_graph(filtered, focal_character, radius=1)
    isolates = list(nx.isolates(filtered))
    filtered.remove_nodes_from(isolates)
    return filtered


def normalised_node_sizes(graph: nx.Graph, minimum: float = 7, maximum: float = 18) -> dict[str, float]:
    """Scale weighted degree consistently within a displayed graph."""
    strengths = {str(node): float(value) for node, value in graph.degree(weight="weight")}
    if not strengths:
        return {}
    low, high = min(strengths.values()), max(strengths.values())
    if high == low:
        return {node: minimum + 2 for node in strengths}
    return {
        node: minimum + (value - low) * (maximum - minimum) / (high - low)
        for node, value in strengths.items()
    }


def network_html(
    method: str,
    graph: nx.Graph,
    positions: Mapping[str, tuple[float, float]],
    focal_character: str | None = None,
    height: str = "540px",
) -> str:
    """Create a self-contained PyVis visual using a shared fixed layout."""
    config = METHOD_CONFIG[method]
    sizes = normalised_node_sizes(graph)
    display_positions = dict(positions)
    if focal_character and focal_character in graph:
        neighbours = sorted(str(node) for node in graph if str(node) != focal_character)
        display_positions = {focal_character: (0.0, 0.0)}
        for index, node in enumerate(neighbours):
            ring = index // 16
            ring_nodes = min(16, len(neighbours) - ring * 16)
            angle = 2 * math.pi * (index % 16) / max(ring_nodes, 1)
            radius = 0.55 + ring * 0.35
            display_positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
    ranked_labels = {
        str(node)
        for node, _ in sorted(
            graph.degree(weight="weight"), key=lambda item: (-float(item[1]), str(item[0]))
        )[:25]
    }
    network = Network(height=height, width="100%", bgcolor="#F8FAFC", font_color="#0F172A")
    for node, data in graph.nodes(data=True):
        x, y = display_positions.get(str(node), (0.0, 0.0))
        strength = graph.degree(node, weight="weight")
        selected = str(node) == focal_character
        show_label = len(graph) <= 45 or str(node) in ranked_labels or selected
        network.add_node(
            str(node),
            label=str(node) if show_label else "",
            title=f"{node}<br>Weighted degree: {strength:.3f}<br>Degree: {graph.degree(node)}",
            x=x * 255,
            y=y * 340,
            fixed=True,
            size=sizes.get(str(node), 9),
            color="#111827" if selected else config["colour"],
            borderWidth=3 if selected else 1,
            font={"size": 11 if selected else 8, "face": "Microsoft YaHei", "strokeWidth": 0},
        )
    weights = [float(data.get("weight", 1.0)) for *_, data in graph.edges(data=True)]
    max_weight = max(weights, default=1.0)
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        network.add_edge(
            str(source),
            str(target),
            value=max(0.5, 4.0 * weight / max_weight),
            title=f"Weight: {weight:.3f}",
            color={"color": config["colour"], "opacity": 0.32},
        )
    network.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "navigationButtons": False, "keyboard": True, "zoomView": True, "dragView": True},
        "edges": {"smooth": {"enabled": True, "type": "continuous"}},
    }))
    return network.generate_html(notebook=False)


def character_method_rows(character: str, graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    """Summarise one character's position and strongest neighbours by method."""
    rows = []
    for method, graph in graphs.items():
        if character not in graph:
            rows.append({"method": METHOD_CONFIG[method]["label"], "active": False, "degree": 0, "weighted_degree": 0.0, "strongest_neighbours": "-"})
            continue
        neighbours = sorted(
            ((str(other), float(graph[character][other].get("weight", 1.0))) for other in graph.neighbors(character)),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        rows.append({
            "method": METHOD_CONFIG[method]["label"],
            "active": graph.degree(character) > 0,
            "degree": int(graph.degree(character)),
            "weighted_degree": float(graph.degree(character, weight="weight")),
            "strongest_neighbours": ", ".join(f"{name} ({weight:.2f})" for name, weight in neighbours) or "-",
        })
    return pd.DataFrame(rows)
