"""Small reusable helpers for filesystem and JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import OUTPUT_DIRS


def ensure_directories(paths: Iterable[Path] = OUTPUT_DIRS) -> None:
    """Create output directories used by the pipeline."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def make_json_serializable(value: Any) -> Any:
    """Convert numpy/scikit-learn values into plain JSON-compatible objects."""
    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(data), file, indent=2, ensure_ascii=False)

