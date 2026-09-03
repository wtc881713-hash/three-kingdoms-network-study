"""Predict all relation instances with the exploratory weakly supervised model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.model.few_shot_retrieval import DEFAULT_MODEL, build_model_text


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "annotation" / "relation_instance_pool.csv"
MODEL_DIR = ROOT / "outputs" / "model" / "weakly_supervised_nine_label"
OUTPUT_DIR = ROOT / "outputs" / "predictions" / "weakly_supervised_nine_label"


def aggregate_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise passage predictions for each undirected character pair."""
    data = predictions.copy()
    data["pair_a"] = data[["character_a", "character_b"]].min(axis=1)
    data["pair_b"] = data[["character_a", "character_b"]].max(axis=1)
    grouped = (
        data.groupby(["pair_a", "pair_b", "predicted_label"], as_index=False)
        .agg(
            predicted_passages=("instance_id", "count"),
            mean_probability=("predicted_probability", "mean"),
            max_probability=("predicted_probability", "max"),
        )
    )
    grouped["label_score"] = grouped["predicted_passages"] * grouped["mean_probability"]
    grouped = grouped.sort_values(
        ["pair_a", "pair_b", "label_score", "predicted_label"],
        ascending=[True, True, False, True], kind="stable",
    )
    grouped["pair_label_rank"] = grouped.groupby(["pair_a", "pair_b"]).cumcount() + 1
    return grouped


def main() -> None:
    """Encode the complete pool and export passage- and pair-level predictions."""
    data = pd.read_csv(INPUT, encoding="utf-8-sig", keep_default_na=False)
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    encoder = SentenceTransformer(str(DEFAULT_MODEL), local_files_only=True, device="cpu")
    embeddings = encoder.encode(
        data.apply(build_model_text, axis=1).tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    probabilities = classifier.predict_proba(embeddings)
    predicted_indices = probabilities.argmax(axis=1)
    output = data.copy()
    output["predicted_label"] = classifier.classes_[predicted_indices]
    output["predicted_probability"] = probabilities.max(axis=1).round(6)
    output["prediction_status"] = "automatic_weak_model_unreviewed"
    for index, label in enumerate(classifier.classes_):
        output[f"probability__{label}"] = probabilities[:, index].round(6)
    pair_summary = aggregate_pairs(output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "relation_instance_predictions.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(OUTPUT_DIR / "pair_relation_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "status": "automatic_weak_model_unreviewed",
        "instance_predictions": len(output),
        "unique_character_pairs": output[["character_a", "character_b"]].apply(lambda row: tuple(sorted(row)), axis=1).nunique(),
        "prediction_counts": output["predicted_label"].value_counts().sort_index().to_dict(),
        "mean_max_probability": float(output["predicted_probability"].mean()),
        "warning": "Predictions inherit weak-label errors and are not validated literary relations.",
    }
    (OUTPUT_DIR / "prediction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Instance predictions: {len(output):,}")
    print(f"Pair-label summary rows: {len(pair_summary):,}")
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
