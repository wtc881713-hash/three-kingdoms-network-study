"""Create deterministic protected splits from the 20 reviewed demonstrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.annotation.select_annotation_candidates import normalise_for_duplicate_check


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / "data" / "annotation" / "annotation_batch_01_reviewed_v2.csv"
OUTPUT_DIR = ROOT / "data" / "model" / "round0_five_label"
SEED = 42


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for an auditable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order_key(instance_id: str, seed: int = SEED) -> str:
    """Create a deterministic pseudo-random ordering key."""
    return hashlib.sha256(f"{seed}:{instance_id}".encode("utf-8")).hexdigest()


def allocation_for_label(count: int) -> tuple[int, int, int]:
    """Allocate train, validation, and test counts while protecting rare labels."""
    if count >= 7:
        return count - 4, 2, 2
    if count >= 4:
        return count - 2, 1, 1
    return count, 0, 0


def pair_key(row: pd.Series) -> tuple[str, str]:
    """Return an undirected canonical character-pair key."""
    return tuple(sorted((str(row["character_a"]), str(row["character_b"]))))


def create_splits(data: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Create deterministic label-aware splits without pair or passage leakage."""
    reviewed = data.loc[data["annotation_status"] == "reviewed"].copy()
    if len(reviewed) != 20:
        raise ValueError(f"Expected 20 reviewed examples; found {len(reviewed)}.")
    reviewed["_order"] = reviewed["instance_id"].map(lambda value: stable_order_key(str(value), seed))
    split_parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for _, group in reviewed.groupby("primary_relation", sort=True):
        ordered = group.sort_values("_order", kind="stable").drop(columns="_order")
        train_count, validation_count, test_count = allocation_for_label(len(ordered))
        split_parts["train"].append(ordered.iloc[:train_count])
        split_parts["validation"].append(
            ordered.iloc[train_count : train_count + validation_count]
        )
        split_parts["test"].append(
            ordered.iloc[train_count + validation_count : train_count + validation_count + test_count]
        )
    splits = {
        name: pd.concat(parts, ignore_index=True).sort_values("instance_id", kind="stable").reset_index(drop=True)
        for name, parts in split_parts.items()
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Reject instance, pair, or normalised-passage leakage across splits."""
    seen_instances: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_passages: set[str] = set()
    for name in ("train", "validation", "test"):
        split = splits[name]
        instances = set(split["instance_id"].astype(str))
        pairs = {pair_key(row) for _, row in split.iterrows()}
        passages = {
            normalise_for_duplicate_check(str(value)) for value in split["passage"]
        }
        if seen_instances & instances:
            raise ValueError(f"Instance leakage detected in {name} split.")
        if seen_pairs & pairs:
            raise ValueError(f"Character-pair leakage detected in {name} split.")
        if seen_passages & passages:
            raise ValueError(f"Passage leakage detected in {name} split.")
        seen_instances.update(instances)
        seen_pairs.update(pairs)
        seen_passages.update(passages)


def main() -> None:
    """Write protected split files and a hash manifest."""
    data = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig", keep_default_na=False)
    splits = create_splits(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for name, split in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        split.to_csv(path, index=False, encoding="utf-8-sig")
        split_paths[name] = path
    all_labels = sorted(data["primary_relation"].unique())
    manifest = {
        "created_date": "2026-08-03",
        "seed": SEED,
        "source_file": str(SOURCE_FILE),
        "source_sha256": sha256_file(SOURCE_FILE),
        "observed_labels": all_labels,
        "missing_ontology_labels": sorted(
            {
                "cooperation", "hierarchy_loyalty", "kinship",
                "friendship_brotherhood", "hostility_conflict",
                "deception_manipulation", "affection_romance",
                "no_clear_relation", "uncertain",
            }
            - set(all_labels)
        ),
        "splits": {
            name: {
                "rows": len(splits[name]),
                "label_counts": splits[name]["primary_relation"].value_counts().sort_index().to_dict(),
                "sha256": sha256_file(path),
            }
            for name, path in split_paths.items()
        },
        "leakage_checks": {
            "instance_overlap": 0,
            "character_pair_overlap": 0,
            "normalised_passage_overlap": 0,
        },
        "test_policy": "Protected during model selection; evaluate once after validation selection.",
        "limitation": "Rare observed labels remain train-only; four ontology labels are absent.",
    }
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("train", "validation", "test"):
        print(f"{name}: {len(splits[name])} rows")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
