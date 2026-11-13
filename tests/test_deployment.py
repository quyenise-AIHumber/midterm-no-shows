"""Verify saved artifacts and deployment configuration."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]


def pinned_version(filename, package):
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()
        if line.startswith(f"{package}=="):
            return line.split("==", 1)[1]

    raise AssertionError(f"Missing pinned {package}")


def test_saved_model_files_exist():
    assert (ROOT / "models/no_show.joblib").is_file()
    assert (ROOT / "models/model_info.json").is_file()


def test_scikit_learn_version_matches():
    info = json.loads((ROOT / "models/model_info.json").read_text(encoding="utf-8"))

    expected = info["scikit_learn_version"]

    assert sklearn.__version__ == expected
    assert pinned_version("requirements.txt", "scikit-learn") == expected


def test_space_configuration():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---")

    frontmatter = readme.split("---", 2)[1]

    assert re.search(r"(?m)^sdk:\s*gradio\s*$", frontmatter)
    assert re.search(
        r"(?m)^app_file:\s*app\.py\s*$",
        frontmatter,
    )

    gradio_match = re.search(
        r"(?m)^sdk_version:\s*(.+)$",
        frontmatter,
    )

    assert gradio_match is not None
    assert gradio_match.group(1).strip() == version("gradio")
    assert (ROOT / "app.py").is_file()


def test_gradio_not_in_runtime_requirements():
    assert pinned_version("requirements-dev.txt", "gradio") == version("gradio")

    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert not re.search(
        r"(?m)^\s*gradio\s*(?:[=<>!~]|$)",
        runtime,
    )
