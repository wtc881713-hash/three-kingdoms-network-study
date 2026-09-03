"""Build a small deployment archive from the checked website files."""

from __future__ import annotations

import zipfile
from pathlib import Path

try:
    from tools.check_deployment import REQUIRED_FILES, ROOT
except ModuleNotFoundError:  # Support direct execution from the project root.
    from check_deployment import REQUIRED_FILES, ROOT


OUTPUT = ROOT / "outputs" / "release" / "three-kingdoms-streamlit-release.zip"


def main() -> None:
    """Write the website files to a path-preserving ZIP archive."""
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(ROOT / name, arcname=name.replace("\\", "/"))

    with zipfile.ZipFile(OUTPUT) as archive:
        archived = set(archive.namelist())
    expected = {name.replace("\\", "/") for name in REQUIRED_FILES}
    if archived != expected:
        raise ValueError("Release archive contents do not match the required file list")

    print(f"Release archive: {OUTPUT}")
    print(f"Files: {len(archived)}")
    print(f"Archive size: {OUTPUT.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
