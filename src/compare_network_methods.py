"""Compare co-occurrence, dialogue, and semantic-context character networks."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "comparison"
METHOD_FILES = {
    "cooccurrence": ROOT / "outputs" / "cooccurrence" / "paragraph" / "edges.csv",
    "dialogue": ROOT / "outputs" / "dialogue" / "named_speech" / "edges.csv",
    "semantic_context": ROOT / "outputs" / "semantic" / "multilingual_minilm" / "edges.csv",
}


def load_undirected_graph(edges_file: Path) -> nx.Graph:
    """Load an edge table and combine reciprocal directed pairs if necessary."""
    edges = pd.read_csv(edges_file, encoding="utf-8-sig", keep_default_na=False)
    graph = nx.Graph()
    for row in edges.to_dict(orient="records"):
        source = str(row["source"])
        target = str(row["target"])
        weight = float(row.get("weight", row.get("similarity", 1.0)))
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph


def graph_summary(method: str, graph: nx.Graph) -> dict[str, object]:
    """Calculate comparable whole-network measures on active nodes."""
    active = graph.subgraph([node for node in graph if graph.degree(node) > 0]).copy()
    return {
        "method": method,
        "active_nodes": active.number_of_nodes(),
        "edges": active.number_of_edges(),
        "density": nx.density(active) if active.number_of_nodes() > 1 else 0.0,
        "connected_components": nx.number_connected_components(active) if active else 0,
        "average_clustering": nx.average_clustering(active, weight=None) if active else 0.0,
        "total_edge_weight": sum(
            float(data.get("weight", 1.0)) for *_, data in active.edges(data=True)
        ),
        "relation_definition": {
            "cooccurrence": "same body paragraph",
            "dialogue": "explicit named target or adjacent named speech turns",
            "semantic_context": "thresholded mutual nearest-neighbour context similarity",
        }[method],
    }


def canonical_edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    """Return sorted endpoint pairs for overlap calculations."""
    return {tuple(sorted((str(source), str(target)))) for source, target in graph.edges}


def pairwise_overlap(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Calculate edge intersection, union, and Jaccard similarity by method pair."""
    rows = []
    for first, second in combinations(graphs, 2):
        first_edges = canonical_edge_set(graphs[first])
        second_edges = canonical_edge_set(graphs[second])
        intersection = first_edges & second_edges
        union = first_edges | second_edges
        rows.append(
            {
                "method_1": first,
                "method_2": second,
                "shared_edges": len(intersection),
                "union_edges": len(union),
                "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
                "method_1_coverage": len(intersection) / len(first_edges) if first_edges else 0.0,
                "method_2_coverage": len(intersection) / len(second_edges) if second_edges else 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_nodes(method: str, graph: nx.Graph, limit: int = 10) -> list[dict[str, object]]:
    """Return the highest-strength active nodes for one method."""
    ranked = sorted(
        graph.degree(weight="weight"),
        key=lambda item: (-float(item[1]), str(item[0])),
    )[:limit]
    return [
        {
            "method": method,
            "rank": rank,
            "canonical_name": node,
            "weighted_degree": float(weight),
            "degree": graph.degree(node),
        }
        for rank, (node, weight) in enumerate(ranked, start=1)
    ]


def main() -> None:
    """Save common graph measures, edge overlap, and top-node rankings."""
    graphs = {method: load_undirected_graph(path) for method, path in METHOD_FILES.items()}
    summaries = pd.DataFrame(
        [graph_summary(method, graph) for method, graph in graphs.items()]
    )
    overlaps = pairwise_overlap(graphs)
    rankings = pd.DataFrame(
        [row for method, graph in graphs.items() for row in top_nodes(method, graph)]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(OUTPUT_DIR / "method_summary.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(OUTPUT_DIR / "edge_overlap.csv", index=False, encoding="utf-8-sig")
    rankings.to_csv(OUTPUT_DIR / "top_nodes.csv", index=False, encoding="utf-8-sig")
    print("Method summary")
    print(summaries.to_string(index=False))
    print("\nPairwise edge overlap")
    print(overlaps.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
