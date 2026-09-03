"""Check that the Streamlit release is complete and anonymous."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    ".streamlit/config.toml",
    "src/network_visualization.py",
    "outputs/comparison/method_summary.csv",
    "outputs/comparison/edge_overlap.csv",
    "outputs/comparison/top_nodes.csv",
    "outputs/cooccurrence/paragraph/nodes.csv",
    "outputs/cooccurrence/paragraph/edges.csv",
    "outputs/dialogue/named_speech/nodes.csv",
    "outputs/dialogue/named_speech/edges.csv",
    "outputs/semantic/multilingual_minilm/nodes.csv",
    "outputs/semantic/multilingual_minilm/edges.csv",
]

PUBLIC_TEXT_FILES = [
    "app.py",
    "README.md",
    ".streamlit/config.toml",
    "src/network_visualization.py",
]

PRIVATE_MARKERS = ["C:\\Users\\", "C:/Users/", "E:\\Codex\\attachments", "@kcl.ac.uk"]


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Validate release files and write a checksum manifest."""
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")

    for name in REQUIRED_FILES:
        path = ROOT / name
        if path.suffix.lower() == ".csv" and pd.read_csv(path, encoding="utf-8-sig").empty:
            raise ValueError(f"Release table is empty: {name}")

    findings: list[str] = []
    for name in PUBLIC_TEXT_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker.lower() in text.lower():
                findings.append(f"{name}: {marker}")
    if findings:
        raise ValueError(f"Possible private information found: {findings}")

    manifest = {
        "release_files": [
            {
                "path": name.replace("\\", "/"),
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
            }
            for name in REQUIRED_FILES
        ]
    }
    output = ROOT / "outputs" / "release" / "release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_bytes = sum(item["bytes"] for item in manifest["release_files"])
    print("Deployment check: PASS")
    print(f"Required files: {len(REQUIRED_FILES)}")
    print(f"Release file size: {total_bytes / (1024 * 1024):.2f} MiB")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
