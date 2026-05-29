from __future__ import annotations
import json
from pathlib import Path
from src.config import OUTPUT_DIRS

def ensure_directories():
    for d in OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

