"""Index persistence - save/load search index to disk."""
from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"


def save_index(data: dict[str, Any], path: Path | None = None) -> None:
    """Save index data to pickle file."""
    path = path or (INDEX_DIR / "index.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Index saved to %s", path)


def load_index(path: Path | None = None) -> dict[str, Any] | None:
    """Load index data from pickle file. Returns None if not found."""
    path = path or (INDEX_DIR / "index.pkl")
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("Index loaded from %s", path)
        return data
    except Exception as e:
        logger.warning("Failed to load index: %s", e)
        return None
