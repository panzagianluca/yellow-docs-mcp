"""Index persistence with versioned JSON/NPZ schema."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from yellow_docs_mcp.parser import CodeBlock, DocPage, DocSection
from yellow_docs_mcp.search import Chunk
from yellow_docs_mcp.telemetry import log_metric

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"
INDEX_VERSION = 3


def _default_index_path() -> Path:
    return INDEX_DIR / "index.json"


def _embeddings_path(index_path: Path) -> Path:
    if index_path.suffix:
        return index_path.with_suffix(".npz")
    return index_path.parent / f"{index_path.name}.npz"


def _tmp_path(path: Path) -> Path:
    return path.parent / f"{path.name}.tmp"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_code_blocks(code_blocks: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for block in code_blocks or []:
        normalized.append(
            {
                "language": str(block.get("language", "text")),
                "code": str(block.get("code", "")),
            }
        )
    return normalized


def _fingerprint_from_chunk_payload(data: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(data.get("source", "docs")).encode("utf-8"))
    hasher.update(str(data.get("path", "")).encode("utf-8"))
    hasher.update(str(data.get("section_title", "")).encode("utf-8"))
    hasher.update(str(data.get("text", "")).encode("utf-8"))
    for block in _normalize_code_blocks(data.get("code_blocks")):
        hasher.update(block["language"].encode("utf-8"))
        hasher.update(block["code"].encode("utf-8"))
    return hasher.hexdigest()


def _serialize_chunk(chunk: Chunk) -> dict[str, Any]:
    fingerprint = chunk.fingerprint or _fingerprint_from_chunk_payload(
        {
            "source": chunk.source or "docs",
            "path": chunk.path,
            "section_title": chunk.section_title,
            "text": chunk.text,
            "code_blocks": chunk.code_blocks,
        }
    )
    return {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path,
        "category": chunk.category,
        "page_title": chunk.page_title,
        "section_title": chunk.section_title,
        "text": chunk.text,
        "code_blocks": _normalize_code_blocks(chunk.code_blocks),
        "source": chunk.source or "docs",
        "fingerprint": fingerprint,
    }


def _deserialize_chunk(data: dict[str, Any]) -> Chunk:
    normalized = {
        "chunk_id": int(data["chunk_id"]),
        "path": str(data["path"]),
        "category": str(data["category"]),
        "page_title": str(data["page_title"]),
        "section_title": str(data["section_title"]),
        "text": str(data["text"]),
        "code_blocks": _normalize_code_blocks(data.get("code_blocks")),
        "source": str(data.get("source", "docs")),
        "fingerprint": str(data.get("fingerprint") or _fingerprint_from_chunk_payload(data)),
    }
    return Chunk(**normalized)


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
        "source": page.source or "docs",
    }


def _deserialize_page(data: dict[str, Any]) -> DocPage:
    sections = []
    for section_data in data.get("sections", []):
        code_blocks = [
            CodeBlock(language=str(block["language"]), code=str(block["code"]))
            for block in section_data.get("code_blocks", [])
        ]
        sections.append(
            DocSection(
                title=str(section_data["title"]),
                level=int(section_data["level"]),
                text=str(section_data["text"]),
                code_blocks=code_blocks,
            )
        )

    return DocPage(
        path=str(data["path"]),
        category=str(data["category"]),
        title=str(data["title"]),
        description=str(data.get("description", "")),
        keywords=list(data.get("keywords", [])),
        sections=sections,
        raw_content=str(data.get("raw_content", "")),
        source=str(data.get("source", "docs")),
    )


def _payload_v3(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": INDEX_VERSION,
        "written_at": _iso_now(),
        "chunks": [_serialize_chunk(chunk) for chunk in data["chunks"]],
        "pages": [_serialize_page(page) for page in data["pages"]],
        "has_embeddings": data.get("embeddings") is not None,
        "enable_vectors": bool(data.get("enable_vectors", True)),
        "metadata": data.get("metadata") or {},
    }


def _migrate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    version = int(payload.get("version", -1))
    if version == INDEX_VERSION:
        return payload, False

    if version == 2:
        migrated = dict(payload)
        migrated["version"] = 3
        migrated["written_at"] = _iso_now()
        migrated["metadata"] = {
            **(payload.get("metadata") or {}),
            "migrated_from": 2,
            "migrated_at": _iso_now(),
        }

        fixed_chunks = []
        for chunk in payload.get("chunks", []):
            chunk_copy = dict(chunk)
            chunk_copy["source"] = chunk_copy.get("source", "docs")
            chunk_copy["fingerprint"] = chunk_copy.get("fingerprint") or _fingerprint_from_chunk_payload(
                chunk_copy
            )
            fixed_chunks.append(chunk_copy)
        migrated["chunks"] = fixed_chunks

        fixed_pages = []
        for page in payload.get("pages", []):
            page_copy = dict(page)
            page_copy["source"] = page_copy.get("source", "docs")
            fixed_pages.append(page_copy)
        migrated["pages"] = fixed_pages

        return migrated, True

    logger.warning("Unsupported index version: %s", version)
    return None, False


def _write_payload_and_embeddings(
    payload: dict[str, Any],
    embeddings: np.ndarray | None,
    path: Path,
) -> None:
    embeddings_path = _embeddings_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_tmp = _tmp_path(path)
    json_tmp.write_text(json.dumps(payload), encoding="utf-8")
    json_tmp.replace(path)

    if embeddings is not None:
        npz_tmp = _tmp_path(embeddings_path)
        with open(npz_tmp, "wb") as f:
            np.savez_compressed(f, embeddings=np.asarray(embeddings))
        npz_tmp.replace(embeddings_path)
    elif embeddings_path.exists():
        embeddings_path.unlink()


def save_index(data: dict[str, Any], path: Path | None = None) -> None:
    """Save index data to versioned JSON/NPZ files."""
    path = path or _default_index_path()
    payload = _payload_v3(data)
    embeddings = data.get("embeddings")
    _write_payload_and_embeddings(payload, embeddings, path)
    log_metric(
        logger,
        "store.save_index",
        path=str(path),
        version=INDEX_VERSION,
        chunk_count=len(payload["chunks"]),
        page_count=len(payload["pages"]),
        has_embeddings=bool(payload["has_embeddings"]),
    )


def load_index(path: Path | None = None) -> dict[str, Any] | None:
    """Load index data from versioned JSON/NPZ files."""
    path = path or _default_index_path()
    embeddings_path = _embeddings_path(path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read index JSON %s: %s", path, exc)
        return None

    migrated_payload, migrated = _migrate_payload(payload)
    if migrated_payload is None:
        return None
    payload = migrated_payload

    embeddings = None
    if payload.get("has_embeddings"):
        if not embeddings_path.exists():
            logger.warning("Embeddings file missing: %s", embeddings_path)
        else:
            try:
                with np.load(embeddings_path, allow_pickle=False) as npz:
                    embeddings = np.asarray(npz["embeddings"])
            except Exception as exc:
                logger.warning("Failed to load embeddings %s: %s", embeddings_path, exc)
                embeddings = None

    try:
        chunks = [_deserialize_chunk(item) for item in payload.get("chunks", [])]
        pages = [_deserialize_page(item) for item in payload.get("pages", [])]
    except Exception as exc:
        logger.warning("Failed to deserialize index payload %s: %s", path, exc)
        return None

    if migrated:
        # Rewrite in latest schema so migration runs once.
        _write_payload_and_embeddings(payload, embeddings, path)
        log_metric(logger, "store.migrated_index", path=str(path), version=INDEX_VERSION)

    log_metric(
        logger,
        "store.load_index",
        path=str(path),
        version=payload.get("version"),
        chunk_count=len(chunks),
        page_count=len(pages),
        has_embeddings=embeddings is not None,
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "pages": pages,
        "enable_vectors": bool(payload.get("enable_vectors", True)),
        "metadata": payload.get("metadata") or {},
    }

