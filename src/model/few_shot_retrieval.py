"""Build a frozen-encoder nearest-example baseline from reviewed annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMONSTRATIONS = ROOT / "data" / "annotation" / "annotation_batch_01_reviewed_v2.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "annotation" / "annotation_batch_02.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "few_shot_round1" / "annotation_batch_02_model_aided.csv"
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "few_shot_retrieval_baseline.json"
DEFAULT_MODEL = (
    ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots"
    / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)


def build_model_text(row: pd.Series) -> str:
    """Create one consistent entity-aware text representation."""
    return (
        f"Character A: [CHAR_A] {row['character_a']} [/CHAR_A] [SEP] "
        f"Character B: [CHAR_B] {row['character_b']} [/CHAR_B] [SEP] "
        f"Passage: {row['passage']}"
    )


def nearest_indices(
    query_vectors: np.ndarray,
    reference_vectors: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return descending cosine-similarity indices for normalised vectors."""
    if top_k < 1 or top_k > len(reference_vectors):
        raise ValueError("top_k must be between 1 and the reference-set size.")
    similarities = query_vectors @ reference_vectors.T
    indices = np.argsort(-similarities, axis=1)[:, :top_k]
    scores = np.take_along_axis(similarities, indices, axis=1)
    return indices, scores


def leave_one_out_predictions(vectors: np.ndarray, labels: list[str]) -> list[str]:
    """Predict every demonstration from its nearest different demonstration."""
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    return [labels[index] for index in np.argmax(similarities, axis=1)]


def create_model_aids(
    demonstrations: pd.DataFrame,
    candidates: pd.DataFrame,
    demonstration_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    top_k: int = 3,
) -> pd.DataFrame:
    """Append auditable nearest-example suggestions without filling gold labels."""
    indices, scores = nearest_indices(candidate_vectors, demonstration_vectors, top_k)
    label_support = Counter(demonstrations["primary_relation"].astype(str))
    output = candidates.copy()
    predictions: list[str] = []
    supports: list[int] = []
    priorities: list[str] = []
    neighbour_ids: list[str] = []
    neighbour_labels: list[str] = []
    neighbour_scores: list[str] = []
    for row_indices, row_scores in zip(indices, scores, strict=True):
        nearest = demonstrations.iloc[row_indices]
        predicted = str(nearest.iloc[0]["primary_relation"])
        support = label_support[predicted]
        similarity = float(row_scores[0])
        predictions.append(predicted)
        supports.append(support)
        priorities.append("high" if support < 3 or similarity < 0.50 else "normal")
        neighbour_ids.append("|".join(nearest["instance_id"].astype(str)))
        neighbour_labels.append("|".join(nearest["primary_relation"].astype(str)))
        neighbour_scores.append("|".join(f"{float(value):.4f}" for value in row_scores))
    output["few_shot_prediction"] = predictions
    output["few_shot_label_support"] = supports
    output["nearest_similarity"] = scores[:, 0].round(4)
    output["nearest_demo_ids"] = neighbour_ids
    output["nearest_demo_labels"] = neighbour_labels
    output["nearest_demo_similarities"] = neighbour_scores
    output["review_priority"] = priorities
    output["few_shot_status"] = "suggestion_only_not_ground_truth"
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line paths while keeping project defaults reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, default=DEFAULT_DEMONSTRATIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    """Encode demonstrations, evaluate 1-NN, and suggest labels for Round 1."""
    args = parse_args()
    demonstrations = pd.read_csv(args.demonstrations, encoding="utf-8-sig", keep_default_na=False)
    candidates = pd.read_csv(args.candidates, encoding="utf-8-sig", keep_default_na=False)
    reviewed = demonstrations.loc[demonstrations["annotation_status"] == "reviewed"].copy()
    if len(reviewed) != 20:
        raise ValueError(f"Expected 20 reviewed demonstrations; found {len(reviewed)}.")
    if not args.model.exists():
        raise FileNotFoundError(f"Configured local model not found: {args.model}")

    model = SentenceTransformer(str(args.model), local_files_only=True, device="cpu")
    demonstration_vectors = model.encode(
        reviewed.apply(build_model_text, axis=1).tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_vectors = model.encode(
        candidates.apply(build_model_text, axis=1).tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    labels = reviewed["primary_relation"].astype(str).tolist()
    loo_predictions = leave_one_out_predictions(demonstration_vectors, labels)
    aids = create_model_aids(
        reviewed, candidates, demonstration_vectors, candidate_vectors
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    aids.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = {
        "method": "frozen multilingual MiniLM 1-nearest-demonstration retrieval",
        "model_path": str(args.model),
        "training_performed": False,
        "demonstration_count": len(reviewed),
        "candidate_count": len(candidates),
        "demonstration_label_counts": dict(Counter(labels)),
        "unseen_ontology_labels": sorted(
            {
                "cooperation", "hierarchy_loyalty", "kinship",
                "friendship_brotherhood", "hostility_conflict",
                "deception_manipulation", "affection_romance",
                "no_clear_relation", "uncertain",
            }
            - set(labels)
        ),
        "leave_one_out_accuracy": accuracy_score(labels, loo_predictions),
        "leave_one_out_classification_report": classification_report(
            labels, loo_predictions, output_dict=True, zero_division=0
        ),
        "round1_prediction_counts": dict(Counter(aids["few_shot_prediction"])),
        "high_priority_review_count": int((aids["review_priority"] == "high").sum()),
        "interpretation": (
            "The output is a retrieval aid, not gold annotation. It can only "
            "suggest labels represented among the 20 demonstrations."
        ),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reviewed demonstrations: {len(reviewed)}")
    print(f"Round 1 candidates: {len(candidates)}")
    print(f"Leave-one-out accuracy: {report['leave_one_out_accuracy']:.4f}")
    print(f"High-priority review rows: {report['high_priority_review_count']}")
    print(f"Saved suggestions: {args.output}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
