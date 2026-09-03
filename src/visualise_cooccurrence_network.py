"""Create reproducible dissertation figures for the co-occurrence network."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "cooccurrence" / "paragraph"
OUTPUT_DIR = INPUT_DIR / "figures"
CORE_WEIGHT_THRESHOLD = 30
LAYOUT_SEED = 42

COMMUNITY_COLOURS = {
    1: "#D55E00",
    2: "#0072B2",
    3: "#009E73",
    4: "#CC79A7",
}


def configure_fonts() -> None:
    """Configure common CJK fonts while retaining portable fallbacks."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_graph(nodes_file: Path, edges_file: Path) -> nx.Graph:
    """Load node metrics and weighted edges into an undirected graph."""
    nodes = pd.read_csv(nodes_file, encoding="utf-8-sig")
    edges = pd.read_csv(edges_file, encoding="utf-8-sig")
    graph = nx.Graph()
    for row in nodes.to_dict(orient="records"):
        name = str(row.pop("canonical_name"))
        graph.add_node(name, **row)
    for row in edges.to_dict(orient="records"):
        source = str(row.pop("source"))
        target = str(row.pop("target"))
        graph.add_edge(source, target, **row)
    return graph


def filter_core_graph(graph: nx.Graph, minimum_weight: int) -> nx.Graph:
    """Retain edges meeting a transparent strength threshold and their nodes."""
    selected = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if int(data["weight"]) >= minimum_weight
    ]
    return graph.edge_subgraph(selected).copy()


def scaled_node_sizes(graph: nx.Graph) -> list[float]:
    """Scale node areas using square-root weighted degree."""
    return [
        120.0 + 34.0 * math.sqrt(float(graph.nodes[node]["weighted_degree"]))
        for node in graph.nodes
    ]


def scaled_edge_widths(graph: nx.Graph) -> list[float]:
    """Scale edge widths logarithmically to control extreme weights."""
    return [0.25 + 0.75 * math.log1p(float(data["weight"])) for *_, data in graph.edges(data=True)]


def calculate_layout(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Calculate a deterministic weighted spring layout."""
    positions = nx.spring_layout(
        graph,
        seed=LAYOUT_SEED,
        weight="weight",
        k=1.7 / math.sqrt(max(graph.number_of_nodes(), 1)),
        iterations=300,
    )
    return {node: (float(x), float(y)) for node, (x, y) in positions.items()}


def draw_network(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    output_stem: str,
    title: str,
    label_limit: int | None,
) -> None:
    """Render one network figure as publication-ready PNG and SVG files."""
    figure, axis = plt.subplots(figsize=(15, 12), constrained_layout=True)
    axis.set_facecolor("#FAFAFA")
    node_colours = [
        COMMUNITY_COLOURS[int(graph.nodes[node]["community_id"])]
        for node in graph.nodes
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        width=scaled_edge_widths(graph),
        edge_color="#6B7280",
        alpha=0.28,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=scaled_node_sizes(graph),
        node_color=node_colours,
        edgecolors="#FFFFFF",
        linewidths=0.8,
        alpha=0.92,
    )

    ranked = sorted(
        graph.nodes,
        key=lambda node: float(graph.nodes[node]["weighted_degree"]),
        reverse=True,
    )
    labelled = ranked if label_limit is None else ranked[:label_limit]
    labels = {node: node for node in labelled}
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=9,
        font_weight="bold",
        font_color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.6},
    )

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"Community {community}",
            markerfacecolor=colour,
            markeredgecolor="white",
            markersize=10,
        )
        for community, colour in COMMUNITY_COLOURS.items()
    ]
    axis.legend(handles=legend, loc="upper left", frameon=False, ncol=2)
    axis.set_title(title, fontsize=17, fontweight="bold", pad=16)
    axis.text(
        0.5,
        -0.015,
        "Node size = weighted degree  |  Edge width = paragraph co-occurrence weight  |  Colour = Louvain community",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
    )
    axis.margins(0.14)
    axis.axis("off")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(
            OUTPUT_DIR / f"{output_stem}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)


def save_layouts(
    full_positions: dict[str, tuple[float, float]],
    core_positions: dict[str, tuple[float, float]],
) -> None:
    """Save coordinates so every figure can be audited and reproduced."""
    rows = []
    for view, positions in (("full", full_positions), ("core", core_positions)):
        rows.extend(
            {"view": view, "canonical_name": node, "x": x, "y": y}
            for node, (x, y) in positions.items()
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "network_layouts.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    """Generate the complete and strong-tie views plus an audit note."""
    configure_fonts()
    graph = load_graph(INPUT_DIR / "nodes.csv", INPUT_DIR / "edges.csv")
    core_graph = filter_core_graph(graph, CORE_WEIGHT_THRESHOLD)
    full_positions = calculate_layout(graph)
    core_positions = calculate_layout(core_graph)

    draw_network(
        graph,
        full_positions,
        "full_character_network",
        "Full Paragraph Co-occurrence Network (83 Characters)",
        label_limit=25,
    )
    draw_network(
        core_graph,
        core_positions,
        "core_character_network_weight_ge_30",
        f"Core Character Network (Edge Weight ≥ {CORE_WEIGHT_THRESHOLD})",
        label_limit=None,
    )
    save_layouts(full_positions, core_positions)

    note = "\n".join(
        [
            "Co-occurrence Network Figure Notes",
            "==================================",
            f"Full network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.",
            f"Core threshold: edge weight >= {CORE_WEIGHT_THRESHOLD} shared body paragraphs.",
            f"Core network: {core_graph.number_of_nodes()} nodes, {core_graph.number_of_edges()} edges.",
            f"Layout: weighted spring layout, seed {LAYOUT_SEED}.",
            "Full-network labels: top 25 characters by weighted degree.",
            "Core-network labels: all retained characters.",
        ]
    )
    (OUTPUT_DIR / "figure_notes.txt").write_text(note + "\n", encoding="utf-8")
    print(note)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
