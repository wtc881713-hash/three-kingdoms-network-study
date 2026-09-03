"""Train a nine-label classifier from 20 human and 60 low-weight weak examples."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score

from src.model.few_shot_retrieval import DEFAULT_MODEL, build_model_text
from src.model.train_frozen_classifier import C_VALUES, fit_classifier


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "model" / "round0_five_label"
WEAK_FILE = ROOT / "data" / "model" / "weak_round1" / "weak_labels.csv"
OUTPUT_DIR = ROOT / "outputs" / "model" / "weakly_supervised_nine_label"


def main() -> None:
    """Select on human validation data, then evaluate the human protected test."""
    human_train = pd.read_csv(SPLIT_DIR / "train.csv", encoding="utf-8-sig", keep_default_na=False)
    human_validation = pd.read_csv(SPLIT_DIR / "validation.csv", encoding="utf-8-sig", keep_default_na=False)
    human_test = pd.read_csv(SPLIT_DIR / "test.csv", encoding="utf-8-sig", keep_default_na=False)
    weak = pd.read_csv(WEAK_FILE, encoding="utf-8-sig", keep_default_na=False)
    encoder = SentenceTransformer(str(DEFAULT_MODEL), local_files_only=True, device="cpu")

    def encode(frame: pd.DataFrame) -> np.ndarray:
        return encoder.encode(frame.apply(build_model_text, axis=1).tolist(), normalize_embeddings=True, show_progress_bar=False)

    x_human_train, x_validation, x_test, x_weak = map(
        encode, (human_train, human_validation, human_test, weak)
    )
    x_train = np.vstack([x_human_train, x_weak])
    y_train = np.concatenate([
        human_train["primary_relation"].to_numpy(), weak["weak_label"].to_numpy()
    ])
    weights = np.concatenate([
        np.full(len(human_train), 3.0), weak["sample_weight"].astype(float).to_numpy()
    ])
    y_validation = human_validation["primary_relation"].to_numpy()
    observed_labels = sorted(set(y_train))
    trials = []
    models = []
    for c_value in C_VALUES:
        model = fit_classifier(x_train, y_train, c_value, None)
        model.fit(x_train, y_train, sample_weight=weights)
        predictions = model.predict(x_validation)
        trials.append({
            "c_value": c_value,
            "accuracy": float(accuracy_score(y_validation, predictions)),
            "macro_f1_nine_labels": float(f1_score(y_validation, predictions, labels=observed_labels, average="macro", zero_division=0)),
        })
        models.append(model)
    best_index = max(range(len(trials)), key=lambda i: (trials[i]["macro_f1_nine_labels"], trials[i]["accuracy"], -abs(np.log10(trials[i]["c_value"]))))
    selected = trials[best_index]

    development = pd.concat([human_train, human_validation], ignore_index=True)
    x_development = np.vstack([x_human_train, x_validation])
    x_final = np.vstack([x_development, x_weak])
    y_final = np.concatenate([development["primary_relation"].to_numpy(), weak["weak_label"].to_numpy()])
    final_weights = np.concatenate([np.full(len(development), 3.0), weak["sample_weight"].astype(float).to_numpy()])
    final_model = fit_classifier(x_final, y_final, float(selected["c_value"]), None)
    final_model.fit(x_final, y_final, sample_weight=final_weights)
    test_predictions = final_model.predict(x_test)
    test_accuracy = float(accuracy_score(human_test["primary_relation"], test_predictions))
    test_macro = float(f1_score(human_test["primary_relation"], test_predictions, labels=observed_labels, average="macro", zero_division=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, OUTPUT_DIR / "classifier.joblib")
    prediction_table = human_test[["instance_id", "character_a", "character_b", "primary_relation"]].copy()
    prediction_table = prediction_table.rename(columns={"primary_relation": "gold_label"})
    prediction_table["predicted_label"] = test_predictions
    prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "status": "weakly_supervised_exploratory_model",
        "encoder_updated": False,
        "human_training_examples": len(development),
        "weak_training_examples": len(weak),
        "classes": list(final_model.classes_),
        "selected_validation_result": selected,
        "validation_trials": trials,
        "protected_test_accuracy": test_accuracy,
        "protected_test_macro_f1_nine_labels": test_macro,
        "warning": "The 60 automatic labels are weak supervision, not human ground truth.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Classes: {len(final_model.classes_)}")
    print(f"Selected C: {selected['c_value']}")
    print(f"Protected-test accuracy: {test_accuracy:.4f}")
    print(f"Protected-test macro F1 across nine labels: {test_macro:.4f}")
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
