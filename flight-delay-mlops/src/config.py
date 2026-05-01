"""Configuration loader. Single source of truth = params.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | str | None = None) -> dict[str, Any]:
    """Load the project's params.yaml. Returns a plain dict."""
    p = Path(path) if path else PARAMS_PATH
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)
