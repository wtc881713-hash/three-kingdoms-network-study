"""Interactive dissertation artefact for comparing character-network methods."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.network_visualization import (
    METHOD_CONFIG,
    build_method_graph,
    character_method_rows,
    filter_graph,
    network_html,
    shared_layout,
)


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Three Kingdoms Network Comparison",
    page_icon="⚔️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    .method-note {border-left: 5px solid #334155; background: #F8FAFC; color: #0F172A; padding: 0.8rem 1rem; margin: 0.5rem 0 1rem;}
    div[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.8rem; border-radius: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = ROOT / "outputs" / "comparison"
    return (
        pd.read_csv(output / "method_summary.csv", encoding="utf-8-sig"),
        pd.read_csv(output / "edge_overlap.csv", encoding="utf-8-sig"),
        pd.read_csv(output / "top_nodes.csv", encoding="utf-8-sig"),
    )


@st.cache_resource
def load_graphs():
    graphs = {method: build_method_graph(method) for method in METHOD_CONFIG}
    return graphs, shared_layout(graphs)


@st.cache_data
def csv_download(path: str) -> bytes:
    """Return a generated CSV as UTF-8 with BOM for Excel compatibility."""
    table = pd.read_csv(ROOT / path, encoding="utf-8-sig", keep_default_na=False)
    return table.to_csv(index=False).encode("utf-8-sig")


def show_downloads(items: Iterable[tuple[str, str, str]]) -> None:
    """Render consistently named download buttons for generated data tables."""
    for label, relative_path, file_name in items:
        st.download_button(
            label=label,
            data=csv_download(relative_path),
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )


graphs, positions = load_graphs()
summary, overlap, top_nodes = load_comparison_tables()

st.title("Character Networks in Romance of the Three Kingdoms")
st.caption("A Digital Humanities project that compares three network methods")

st.markdown(
    """
    <div class="method-note"><strong>Important:</strong> each network shows a different kind of link.
    Co-occurrence shows shared paragraphs. Dialogue shows speech links. Semantic context shows similar text around characters.</div>
    """,
    unsafe_allow_html=True,
)

overview_tab, networks_tab, character_tab, evidence_tab, project_tab = st.tabs(
    [
        "Method overview",
        "Unified network views",
        "Character comparison",
        "Evidence and limits",
        "Project and data",
    ]
)

with overview_tab:
    st.subheader("Network size and shape")
    columns = st.columns(3)
    for column, method in zip(columns, METHOD_CONFIG):
        row = summary.loc[summary["method"] == method].iloc[0]
        config = METHOD_CONFIG[method]
        with column:
            st.markdown(
                f"<h3 style='color:{config['colour']}; margin-bottom:0.4rem'>{config['label']}</h3>",
                unsafe_allow_html=True,
            )
            first, second = st.columns(2)
            first.metric("Active characters", int(row["active_nodes"]))
            second.metric("Edges", int(row["edges"]))
            first.metric("Density", f"{row['density']:.3f}")
            second.metric("Components", int(row["connected_components"]))
            st.write(config["meaning"])
            st.caption(config["warning"])

    st.subheader("Main measures")
    metric_table = summary[["method", "active_nodes", "edges", "density", "connected_components", "average_clustering"]].copy()
    metric_table["method"] = metric_table["method"].map(lambda value: METHOD_CONFIG[value]["label"])
    st.dataframe(metric_table, hide_index=True, use_container_width=True)

    st.subheader("Shared links between methods")
    overlap_display = overlap.copy()
    overlap_display["comparison"] = overlap_display.apply(
        lambda row: f"{METHOD_CONFIG[row['method_1']]['label']} vs {METHOD_CONFIG[row['method_2']]['label']}", axis=1
    )
    for row in overlap_display.itertuples(index=False):
        similarity = float(row.jaccard_similarity)
        st.progress(
            similarity,
            text=f"{row.comparison}: {similarity:.3f}",
        )
    st.dataframe(
        overlap_display[["comparison", "shared_edges", "union_edges", "jaccard_similarity", "method_1_coverage", "method_2_coverage"]],
        hide_index=True,
        use_container_width=True,
    )

with networks_tab:
    st.subheader("Compare the three networks")
    all_characters = sorted(set().union(*(set(graph.nodes) for graph in graphs.values())))
    focal = st.selectbox("Choose one character, or show all", ["All active characters", *all_characters])
    focal_character = None if focal == "All active characters" else focal
    percentile = st.slider("Link strength filter", 0, 95, 70, 5)
    columns = st.columns(3)
    for column, (method, graph) in zip(columns, graphs.items()):
        weights = pd.Series([float(data.get("weight", 1.0)) for *_, data in graph.edges(data=True)])
        threshold = float(weights.quantile(percentile / 100)) if not weights.empty else 0.0
        displayed = filter_graph(graph, threshold, focal_character)
        config = METHOD_CONFIG[method]
        with column:
            st.markdown(f"### {config['label']}")
            st.caption(f"{displayed.number_of_nodes()} nodes · {displayed.number_of_edges()} edges · threshold {threshold:.3f}")
            components.html(network_html(method, displayed, positions, focal_character), height=570, scrolling=False)
            st.caption(config["warning"])

with character_tab:
    st.subheader("One character in three networks")
    character = st.selectbox("Choose a character", sorted(set().union(*(set(graph.nodes) for graph in graphs.values()))), key="character_compare")
    st.dataframe(character_method_rows(character, graphs), hide_index=True, use_container_width=True)

    columns = st.columns(3)
    for column, (method, graph) in zip(columns, graphs.items()):
        ego = filter_graph(graph, 0.0, character)
        with column:
            st.markdown(f"### {METHOD_CONFIG[method]['label']}")
            if character not in graph or graph.degree(character) == 0:
                st.warning("This character has no links in this network.")
            else:
                components.html(network_html(method, ego, positions, character, height="460px"), height=490, scrolling=False)

    st.subheader("Top characters by total link strength")
    ranking = top_nodes.pivot(index="rank", columns="method", values="canonical_name").rename(columns={key: value["label"] for key, value in METHOD_CONFIG.items()})
    st.dataframe(ranking, use_container_width=True)

with evidence_tab:
    st.subheader("How to read each network")
    st.markdown(
        """
        | Method | What the link means | What it can show | What it cannot prove |
        |---|---|---|---|
        | Co-occurrence | Two characters appear in one paragraph | Shared scenes and text closeness | That they speak or have a close bond |
        | Dialogue | A named speech link is found | Direct or likely speech contact | Every social link in the novel |
        | Semantic context | The text around two characters is similar | Similar themes, roles, conflicts, or settings | That they meet, speak, or know each other |
        """
    )
    st.success("Co-occurrence check: all 60 sample links were correct. Only 16 were direct. 41 were indirect and 3 were unclear.")
    st.warning(
        "The dialogue and semantic networks were not fully checked by a person. "
        "Use them to compare methods. Do not treat every link as a confirmed fact."
    )

with project_tab:
    st.subheader("About this project")
    st.markdown(
        """
        This Master's Digital Humanities project studies character links in
        *Romance of the Three Kingdoms*. It compares three network methods.
        Each method shows a different part of the novel.

        **Research question:** How does the chosen method change the character network?
        How does it change our reading of the story?
        """
    )

    st.subheader("Text source")
    st.markdown(
        """
        - **Source:** Project Gutenberg eBook 23950. It is in the public domain.
        - **Text used:** a Simplified Chinese copy made for this project.
        - **Size:** 120 chapters and 3,575 body paragraphs.
        - **Characters:** 83 characters. Each appears at least 10 times.
        - **Note:** this text is not the same as a modern printed edition.
          The damaged first file was not used for the final results.
        """
    )

    st.subheader("Method settings")
    method_parameters = pd.DataFrame(
        [
            {
                "method": "Co-occurrence",
                "link unit": "Shared paragraph",
                "link value": "Number of shared paragraphs",
                "rule": "Keep all found pairs",
                "check status": "All 60 sample links passed",
            },
            {
                "method": "Dialogue",
                "link unit": "Named speech link",
                "link value": "Number of found speech links",
                "rule": "Named target or next named speaker",
                "check status": "Not fully checked; use for method comparison",
            },
            {
                "method": "Semantic context",
                "link unit": "Similar text around characters",
                "link value": "Cosine similarity",
                "rule": "Mutual 5-NN; score at least 0.45",
                "check status": "Not fully checked; use for method comparison",
            },
        ]
    )
    st.dataframe(method_parameters, hide_index=True, use_container_width=True)
    st.caption(
        "The semantic method uses multilingual MiniLM. It takes up to 90 Chinese characters "
        "before and after each name. It uses no more than 100 text samples for each character."
    )

    st.subheader("Download the data")
    first, second, third = st.columns(3)
    with first:
        st.markdown("#### Co-occurrence")
        show_downloads(
            [
                ("Download nodes", "outputs/cooccurrence/paragraph/nodes.csv", "cooccurrence_nodes.csv"),
                ("Download edges", "outputs/cooccurrence/paragraph/edges.csv", "cooccurrence_edges.csv"),
            ]
        )
    with second:
        st.markdown("#### Dialogue")
        show_downloads(
            [
                ("Download nodes", "outputs/dialogue/named_speech/nodes.csv", "dialogue_nodes.csv"),
                ("Download edges", "outputs/dialogue/named_speech/edges.csv", "dialogue_edges.csv"),
            ]
        )
    with third:
        st.markdown("#### Semantic context")
        show_downloads(
            [
                ("Download nodes", "outputs/semantic/multilingual_minilm/nodes.csv", "semantic_nodes.csv"),
                ("Download edges", "outputs/semantic/multilingual_minilm/edges.csv", "semantic_edges.csv"),
            ]
        )

    st.markdown("#### Cross-method comparison")
    comparison_columns = st.columns(3)
    comparison_downloads = [
        ("Download method summary", "outputs/comparison/method_summary.csv", "method_summary.csv"),
        ("Download edge overlap", "outputs/comparison/edge_overlap.csv", "edge_overlap.csv"),
        ("Download top characters", "outputs/comparison/top_nodes.csv", "top_nodes.csv"),
    ]
    for column, item in zip(comparison_columns, comparison_downloads):
        with column:
            show_downloads([item])

    st.subheader("Limits")
    st.markdown(
        """
        - These tables are research results. They are not records of real social ties.
        - A shared paragraph does not always mean direct contact.
        - The dialogue method may miss speakers without names. It may also choose the wrong listener.
        - A semantic link means the nearby text is similar. It does not mean two characters meet.
        - The dialogue and semantic networks were not fully checked by a person.
          They are used only to compare methods.
        """
    )

    st.subheader("Run the website")
    st.code("python -m streamlit run app.py", language="bash")
    st.caption("The layout uses seed 42. Move the mouse over a node or link to see its value.")
