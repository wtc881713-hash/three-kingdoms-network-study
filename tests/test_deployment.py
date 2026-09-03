from pathlib import Path

import pandas as pd

from tools.check_deployment import PRIVATE_MARKERS, PUBLIC_TEXT_FILES, REQUIRED_FILES, ROOT
from tools.build_release_archive import OUTPUT


def test_all_release_files_exist() -> None:
    assert all((ROOT / name).is_file() for name in REQUIRED_FILES)


def test_release_csv_files_are_not_empty() -> None:
    for name in REQUIRED_FILES:
        path = ROOT / name
        if path.suffix.lower() == ".csv":
            assert not pd.read_csv(path, encoding="utf-8-sig").empty, name


def test_public_interface_files_do_not_contain_private_markers() -> None:
    for name in PUBLIC_TEXT_FILES:
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        for marker in PRIVATE_MARKERS:
            assert marker.lower() not in text, f"{name}: {marker}"


def test_web_requirements_do_not_include_large_research_packages() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("torch", "transformers", "sentence-transformers", "jupyter"):
        assert package not in requirements
    assert Path(ROOT / "requirements-research.txt").is_file()


def test_public_app_does_not_require_altair_charts() -> None:
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.bar_chart(" not in app_source
    assert "st.altair_chart(" not in app_source


def test_release_archive_contains_only_required_files() -> None:
    import zipfile

    assert OUTPUT.is_file()
    with zipfile.ZipFile(OUTPUT) as archive:
        archived = set(archive.namelist())
    expected = {name.replace("\\", "/") for name in REQUIRED_FILES}
    assert archived == expected
