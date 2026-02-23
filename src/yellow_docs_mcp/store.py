"""Index persistence - save/load search index to disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from yellow_docs_mcp.parser import CodeBlock, DocPage, DocSection
from yellow_docs_mcp.search import Chunk

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"
INDEX_VERSION = 2


def _default_index_path() -> Path:
    return INDEX_DIR / "index.json"


def _embeddings_path(index_path: Path) -> Path:
    if index_path.suffix:
        return index_path.with_suffix(".npz")
    return index_path.parent / f"{index_path.name}.npz"


def _tmp_path(path: Path) -> Path:
    return path.parent / f"{path.name}.tmp"


def _serialize_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path,
        "category": chunk.category,
        "page_title": chunk.page_title,
        "section_title": chunk.section_title,
        "text": chunk.text,
        "code_blocks": chunk.code_blocks,
    }


def _deserialize_chunk(data: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=int(data["chunk_id"]),
        path=str(data["path"]),
        category=str(data["category"]),
        page_title=str(data["page_title"]),
        section_title=str(data["section_title"]),
        text=str(data["text"]),
        code_blocks=list(data.get("code_blocks") or []),
    )


def _serialize_page(page: DocPage) -> dict[str, Any]:
    return {
        "path": page.path,
        "category": page.category,
        "title": page.title,
        "description": page.description,
        "keywords": page.keywords,
        "sections": [
            {
                "title": section.title,
                "level": section.level,
                "text": section.text,
                "code_blocks": [
                    {"language": block.language, "code": block.code}
                    for block in section.code_blocks
                ],
            }
            for section in page.sections
        ],
        "raw_content": page.raw_content,
    }


def _deserialize_page(data: dict[str, Any]) -> DocPage:
    sections = []
    for section_data in data.get("sections", []):
        code_blocks = [
            CodeBlock(language=str(block["language"]), code=str(block["code"]))
            for block in section_data.get("code_blocks", [])
        ]
        sections.append(DocSection(
            title=str(section_data["title"]),
            level=int(section_data["level"]),
            text=str(section_data["text"]),
            code_blocks=code_blocks,
        ))

    return DocPage(
        path=str(data["path"]),
        category=str(data["category"]),
        title=str(data["title"]),
        description=str(data.get("description", "")),
        keywords=list(data.get("keywords", [])),
        sections=sections,
        raw_content=str(data.get("raw_content", "")),
    )


def save_index(data: dict[str, Any], path: Path | None = None) -> None:
    """Save index data to JSON/NPZ files."""
    path = path or _default_index_path()
    embeddings_path = _embeddings_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": INDEX_VERSION,
        "chunks": [_serialize_chunk(chunk) for chunk in data["chunks"]],
        "pages": [_serialize_page(page) for page in data["pages"]],
        "has_embeddings": data.get("embeddings") is not None,
        "enable_vectors": bool(data.get("enable_vectors", True)),
    }

    json_tmp = _tmp_path(path)
    json_tmp.write_text(json.dumps(payload), encoding="utf-8")
    json_tmp.replace(path)

    embeddings = data.get("embeddings")
    if embeddings is not None:
        npz_tmp = _tmp_path(embeddings_path)
        with open(npz_tmp, "wb") as f:
            np.savez_compressed(f, embeddings=np.asarray(embeddings))
        npz_tmp.replace(embeddings_path)
    elif embeddings_path.exists():
        embeddings_path.unlink()

    logger.info("Index saved to %s (+ %s)", path, embeddings_path.name)


def load_index(path: Path | None = None) -> dict[str, Any] | None:
    """Load index data from JSON/NPZ files. Returns None if not found."""
    path = path or _default_index_path()
    embeddings_path = _embeddings_path(path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != INDEX_VERSION:
            logger.warning("Unsupported index version in %s", path)
            return None

        chunks = [_deserialize_chunk(item) for item in payload.get("chunks", [])]
        pages = [_deserialize_page(item) for item in payload.get("pages", [])]

        embeddings = None
        if payload.get("has_embeddings"):
            if not embeddings_path.exists():
                logger.warning("Embeddings file missing: %s", embeddings_path)
            else:
                with np.load(embeddings_path, allow_pickle=False) as npz:
                    embeddings = np.asarray(npz["embeddings"])

        logger.info("Index loaded from %s", path)
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "pages": pages,
            "enable_vectors": bool(payload.get("enable_vectors", True)),
        }
    except Exception as exc:
        logger.warning("Failed to load index: %s", exc)
        return None
