"""Train and evaluate an exploratory classifier on frozen MiniLM embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.model.few_shot_retrieval import DEFAULT_MODEL, build_model_text


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "model" / "round0_five_label"
OUTPUT_DIR = ROOT / "outputs" / "model" / "round0_five_label_frozen_minilm"
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHTS = (None, "balanced")
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    class_weight: str | None,
) -> LogisticRegression:
    """Fit a deterministic multinomial-capable linear classification head."""
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=5000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def evaluate_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    all_labels: list[str],
) -> dict[str, float]:
    """Return accuracy and strict macro F1 across all observed labels."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_all_observed_labels": float(
            f1_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)
        ),
    }


def choose_configuration(results: list[dict[str, object]]) -> dict[str, object]:
    """Choose by validation macro F1, accuracy, then simpler regularisation."""
    return max(
        results,
        key=lambda row: (
            float(row["macro_f1_all_observed_labels"]),
            float(row["accuracy"]),
            -abs(np.log10(float(row["c_value"]))),
            row["class_weight"] == "none",
        ),
    )


def prediction_table(
    frame: pd.DataFrame,
    model: LogisticRegression,
    features: np.ndarray,
) -> pd.DataFrame:
    """Create instance-level predictions and probabilities for audit."""
    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    output = frame[
        ["instance_id", "chapter_id", "character_a", "character_b", "primary_relation"]
    ].copy()
    output = output.rename(columns={"primary_relation": "gold_label"})
    output["predicted_label"] = predictions
    output["predicted_probability"] = probabilities.max(axis=1).round(6)
    for index, label in enumerate(model.classes_):
        output[f"probability__{label}"] = probabilities[:, index].round(6)
    return output


def main() -> None:
    """Select on validation data, refit on train plus validation, and test once."""
    manifest_path = SPLIT_DIR / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {name: SPLIT_DIR / f"{name}.csv" for name in ("train", "validation", "test")}
    for name, path in paths.items():
        expected = manifest["splits"][name]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Protected {name} split hash mismatch: {actual} != {expected}")
    data = {
        name: pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        for name, path in paths.items()
    }
    all_labels = sorted(data["train"]["primary_relation"].unique())
    encoder = SentenceTransformer(str(DEFAULT_MODEL), local_files_only=True, device="cpu")
    features = {
        name: encoder.encode(
            frame.apply(build_model_text, axis=1).tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for name, frame in data.items()
    }
    labels = {name: frame["primary_relation"].to_numpy() for name, frame in data.items()}

    validation_results: list[dict[str, object]] = []
    for c_value in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            model = fit_classifier(features["train"], labels["train"], c_value, class_weight)
            predictions = model.predict(features["validation"])
            metrics = evaluate_predictions(labels["validation"], predictions, all_labels)
            validation_results.append(
                {
                    "c_value": c_value,
                    "class_weight": class_weight or "none",
                    **metrics,
                }
            )
    selected = choose_configuration(validation_results)
    selected_weight = None if selected["class_weight"] == "none" else str(selected["class_weight"])

    development_features = np.vstack([features["train"], features["validation"]])
    development_labels = np.concatenate([labels["train"], labels["validation"]])
    final_model = fit_classifier(
        development_features,
        development_labels,
        float(selected["c_value"]),
        selected_weight,
    )

    test_predictions = final_model.predict(features["test"])
    test_metrics = evaluate_predictions(labels["test"], test_predictions, all_labels)
    majority_label = pd.Series(development_labels).value_counts().index[0]
    majority_predictions = np.full(len(labels["test"]), majority_label, dtype=object)
    majority_metrics = evaluate_predictions(labels["test"], majority_predictions, all_labels)
    test_report = classification_report(
        labels["test"],
        test_predictions,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels["test"], test_predictions, labels=all_labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, array in features.items():
        np.save(OUTPUT_DIR / f"{name}_embeddings.npy", array)
    joblib.dump(final_model, OUTPUT_DIR / "classifier.joblib")
    pd.DataFrame(validation_results).to_csv(
        OUTPUT_DIR / "validation_grid.csv", index=False, encoding="utf-8-sig"
    )
    prediction_table(data["test"], final_model, features["test"]).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix, index=all_labels, columns=all_labels).to_csv(
        OUTPUT_DIR / "test_confusion_matrix.csv", encoding="utf-8-sig"
    )

    final_test_hash = sha256_file(paths["test"])
    if final_test_hash != manifest["splits"]["test"]["sha256"]:
        raise RuntimeError("Protected test split changed during training.")
    metrics = {
        "experiment_status": "exploratory_five_label_baseline",
        "training_performed": True,
        "encoder_updated": False,
        "classifier_head": "scikit-learn LogisticRegression",
        "encoder_path": str(DEFAULT_MODEL),
        "random_seed": RANDOM_SEED,
        "split_sizes": {name: len(frame) for name, frame in data.items()},
        "selected_configuration": selected,
        "validation_grid": validation_results,
        "test_evaluations_performed": 1,
        "test_metrics": test_metrics,
        "majority_baseline_label": majority_label,
        "majority_baseline_test_metrics": majority_metrics,
        "test_classification_report": test_report,
        "test_split_sha256_before_and_after": final_test_hash,
        "observed_training_labels": all_labels,
        "missing_ontology_labels": manifest["missing_ontology_labels"],
        "test_labels_present": sorted(set(labels["test"])),
        "test_labels_absent": sorted(set(all_labels) - set(labels["test"])),
        "limitations": [
            "Only 20 reviewed examples were available.",
            "Four ontology labels were absent from all splits.",
            "Two observed labels were too rare to place in validation or test.",
            "The four-example test set cannot provide a stable performance estimate.",
            "The encoder was frozen; only a linear classification head was trained.",
            "The classifier did not outperform the majority-label baseline on the protected test.",
        ],
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = f"""# Exploratory Round 0 Frozen-MiniLM Classifier

- Status: exploratory five-label baseline
- Reviewed examples: 20
- Split: 12 train / 4 validation / 4 protected test
- Encoder: paraphrase-multilingual-MiniLM-L12-v2 (frozen)
- Trained component: logistic-regression classification head
- Selected C: {selected['c_value']}
- Selected class weight: {selected['class_weight']}
- Validation macro F1 across five observed labels: {selected['macro_f1_all_observed_labels']:.4f}
- Protected-test accuracy: {test_metrics['accuracy']:.4f}
- Protected-test macro F1 across five observed labels: {test_metrics['macro_f1_all_observed_labels']:.4f}
- Majority-baseline protected-test accuracy: {majority_metrics['accuracy']:.4f}
- Majority-baseline protected-test macro F1: {majority_metrics['macro_f1_all_observed_labels']:.4f}

This experiment cannot predict ontology labels absent from the 20 reviewed
examples. The protected test has only four examples and excludes two rare
observed labels. These metrics are pipeline checks, not reliable estimates of
general literary-relation performance.
"""
    (OUTPUT_DIR / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print(f"Selected configuration: C={selected['c_value']}, class_weight={selected['class_weight']}")
    print(f"Validation macro F1: {selected['macro_f1_all_observed_labels']:.4f}")
    print(f"Protected-test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Protected-test macro F1: {test_metrics['macro_f1_all_observed_labels']:.4f}")
    print(f"Majority-baseline test accuracy: {majority_metrics['accuracy']:.4f}")
    print(f"Saved experiment: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
