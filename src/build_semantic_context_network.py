"""Build a character network from multilingual sentence-embedding contexts."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np
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
OUTPUT_DIR = ROOT / "outputs" / "semantic" / "multilingual_minilm"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONTEXT_WINDOW = 90
MAX_CONTEXTS_PER_CHARACTER = 100
MUTUAL_NEIGHBOURS = 5
MINIMUM_SIMILARITY = 0.45
RANDOM_SEED = 42


def select_evenly(rows: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    """Select ordered observations evenly across the narrative."""
    if len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    return [rows[index] for index in indices]


def extract_context_snippet(text: str, aliases: list[str], window: int) -> str:
    """Extract a fixed-width context around the first longest alias match."""
    aliases = sorted(set(aliases), key=lambda alias: (-len(alias), alias))
    matches = [
        match
        for alias in aliases
        for match in re.finditer(re.escape(alias), text)
    ]
    if not matches:
        return text[: window * 2].strip()
    match = min(matches, key=lambda item: item.start())
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return text[start:end].strip()


def build_character_contexts(
    paragraph_mentions: pd.DataFrame,
    maximum_per_character: int = MAX_CONTEXTS_PER_CHARACTER,
    window: int = CONTEXT_WINDOW,
) -> pd.DataFrame:
    """Build one local context per character and body paragraph."""
    rows: list[dict[str, object]] = []
    grouped = paragraph_mentions.groupby(
        ["canonical_name", "paragraph_id"], sort=False
    )
    for (character, paragraph_id), group in grouped:
        first = group.iloc[0]
        aliases = group["matched_alias"].astype(str).tolist()
        rows.append(
            {
                "context_id": "",
                "canonical_name": str(character),
                "paragraph_id": str(paragraph_id),
                "chapter_number": int(first["chapter_number"]),
                "chapter_title": str(first["chapter_title"]),
                "context_text": extract_context_snippet(
                    str(first["paragraph_text"]), aliases, window
                ),
            }
        )

    selected: list[dict[str, object]] = []
    by_character: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_character.setdefault(str(row["canonical_name"]), []).append(row)
    for character in sorted(by_character):
        ordered = sorted(
            by_character[character],
            key=lambda row: (int(row["chapter_number"]), str(row["paragraph_id"])),
        )
        selected.extend(select_evenly(ordered, maximum_per_character))
    for index, row in enumerate(selected, start=1):
        row["context_id"] = f"CONTEXT{index:06d}"
    return pd.DataFrame(selected)


def aggregate_character_vectors(
    contexts: pd.DataFrame,
    context_vectors: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Mean-pool normalised context embeddings for each character."""
    characters = sorted(contexts["canonical_name"].astype(str).unique())
    vectors = []
    counts = {}
    for character in characters:
        indices = np.flatnonzero(
            contexts["canonical_name"].astype(str).to_numpy() == character
        )
        mean_vector = context_vectors[indices].mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm == 0:
            raise ValueError(f"Zero semantic vector for {character}.")
        vectors.append(mean_vector / norm)
        counts[character] = len(indices)
    return characters, np.vstack(vectors), counts


def centre_context_vectors(context_vectors: np.ndarray) -> np.ndarray:
    """Remove the corpus-wide style component and renormalise each context."""
    centred = context_vectors - context_vectors.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Corpus centring produced a zero context vector.")
    return centred / norms


def build_mutual_knn_edges(
    characters: list[str],
    vectors: np.ndarray,
    neighbours: int = MUTUAL_NEIGHBOURS,
    minimum_similarity: float = MINIMUM_SIMILARITY,
) -> pd.DataFrame:
    """Build an undirected graph from thresholded mutual nearest neighbours."""
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    neighbour_sets = []
    for row in similarities:
        count = min(neighbours, len(row) - 1)
        indices = np.argpartition(row, -count)[-count:]
        neighbour_sets.append(set(int(index) for index in indices))

    rows = []
    for source_index, source in enumerate(characters):
        for target_index in sorted(neighbour_sets[source_index]):
            if target_index <= source_index:
                continue
            if source_index not in neighbour_sets[target_index]:
                continue
            similarity = float(similarities[source_index, target_index])
            if similarity < minimum_similarity:
                continue
            rows.append(
                {
                    "source": source,
                    "target": characters[target_index],
                    "similarity": similarity,
                    "weight": similarity,
                    "relation_definition": "mutual_semantic_context_similarity",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["similarity", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def add_representative_contexts(
    edges: pd.DataFrame,
    contexts: pd.DataFrame,
    context_vectors: np.ndarray,
    characters: list[str],
    character_vectors: np.ndarray,
) -> pd.DataFrame:
    """Attach the context nearest each character centroid for inspection."""
    representatives: dict[str, dict[str, object]] = {}
    context_characters = contexts["canonical_name"].astype(str).to_numpy()
    for index, character in enumerate(characters):
        positions = np.flatnonzero(context_characters == character)
        scores = context_vectors[positions] @ character_vectors[index]
        best = int(positions[int(np.argmax(scores))])
        representatives[character] = contexts.iloc[best].to_dict()

    enriched = edges.copy()
    for side in ("source", "target"):
        enriched[f"{side}_context_id"] = enriched[side].map(
            lambda name: representatives[str(name)]["context_id"]
        )
        enriched[f"{side}_chapter_number"] = enriched[side].map(
            lambda name: representatives[str(name)]["chapter_number"]
        )
        enriched[f"{side}_representative_context"] = enriched[side].map(
            lambda name: representatives[str(name)]["context_text"]
        )
    return enriched


def analyse_network(
    dictionary: pd.DataFrame,
    edges: pd.DataFrame,
    context_counts: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate common structural measures for the semantic graph."""
    graph = nx.Graph()
    graph.add_nodes_from(dictionary["canonical_name"].astype(str))
    for row in edges.to_dict(orient="records"):
        graph.add_edge(
            row["source"], row["target"], weight=float(row["similarity"])
        )
    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight=None)
    pagerank = weighted_pagerank(graph)
    active = graph.subgraph([node for node in graph if graph.degree(node) > 0])
    communities = (
        list(nx.community.louvain_communities(active, weight="weight", seed=RANDOM_SEED))
        if active.number_of_nodes()
        else []
    )
    community_ids = {
        node: index
        for index, community in enumerate(communities, start=1)
        for node in community
    }
    rows = [
        {
            "canonical_name": node,
            "context_count": context_counts.get(node, 0),
            "degree": degree[node],
            "weighted_degree": strength[node],
            "betweenness_centrality": betweenness[node],
            "pagerank": pagerank[node],
            "community_id": community_ids.get(node, 0),
        }
        for node in graph.nodes
    ]
    nodes = pd.DataFrame(rows).sort_values(
        ["weighted_degree", "canonical_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary = {
        "dictionary_nodes": graph.number_of_nodes(),
        "active_nodes": active.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density_active": nx.density(active) if active.number_of_nodes() > 1 else 0.0,
        "isolates": nx.number_of_isolates(graph),
        "communities_active": len(communities),
    }
    return nodes, summary


def format_summary(summary: dict[str, object], context_count: int, dimension: int) -> str:
    """Format an auditable semantic-network summary."""
    return "\n".join(
        [
            "Semantic-Context Character Network Summary",
            "==========================================",
            f"Model: {MODEL_NAME}",
            f"Embedding dimension: {dimension}",
            f"Context window: +/- {CONTEXT_WINDOW} characters around a validated alias",
            f"Maximum contexts per character: {MAX_CONTEXTS_PER_CHARACTER}",
            f"Encoded contexts: {context_count}",
            "Corpus centring: global context centroid subtracted before character mean pooling",
            f"Edge rule: mutual {MUTUAL_NEIGHBOURS}-nearest neighbours and cosine similarity >= {MINIMUM_SIMILARITY:.2f}",
            f"Dictionary nodes: {summary['dictionary_nodes']}",
            f"Active nodes: {summary['active_nodes']}",
            f"Edges: {summary['edges']}",
            f"Active-node density: {summary['density_active']:.6f}",
            f"Isolates: {summary['isolates']}",
            f"Louvain communities among active nodes: {summary['communities_active']}",
        ]
    )


def main() -> None:
    """Encode contexts and save the semantic-context character network."""
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    from sentence_transformers import SentenceTransformer

    dictionary = pd.read_csv(DICTIONARY_FILE, encoding="utf-8-sig", keep_default_na=False)
    mentions = pd.read_csv(
        PARAGRAPH_MENTIONS_FILE, encoding="utf-8-sig", keep_default_na=False
    )
    contexts = build_character_contexts(mentions)
    model = SentenceTransformer(MODEL_NAME)
    context_vectors = model.encode(
        contexts["context_text"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    context_vectors = centre_context_vectors(context_vectors)
    characters, character_vectors, context_counts = aggregate_character_vectors(
        contexts, context_vectors
    )
    edges = build_mutual_knn_edges(characters, character_vectors)
    edges = add_representative_contexts(
        edges, contexts, context_vectors, characters, character_vectors
    )
    nodes, summary = analyse_network(dictionary, edges, context_counts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(OUTPUT_DIR / "character_contexts.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUTPUT_DIR / "edges.csv", index=False, encoding="utf-8-sig")
    nodes.to_csv(OUTPUT_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
    np.save(OUTPUT_DIR / "character_vectors.npy", character_vectors)
    pd.DataFrame(
        {"vector_row": range(len(characters)), "canonical_name": characters}
    ).to_csv(OUTPUT_DIR / "character_vector_index.csv", index=False, encoding="utf-8-sig")
    report = format_summary(summary, len(contexts), character_vectors.shape[1])
    (OUTPUT_DIR / "network_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
