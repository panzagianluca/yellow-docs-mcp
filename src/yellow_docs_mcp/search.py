"""Hybrid search engine combining BM25 keyword search and optional vector similarity."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from yellow_docs_mcp.parser import DocPage
from yellow_docs_mcp.telemetry import log_metric, timed_metric

logger = logging.getLogger(__name__)


def _parse_int_env(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%s, using default %d", name, raw, default)
        return default
    if value < minimum:
        value = minimum
    if maximum is not None:
        value = min(value, maximum)
    return value


def _parse_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%s, using default %.2f", name, raw, default)
        return default
    return max(minimum, value)


@dataclass
class SearchResult:
    """A single search result."""

    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    score: float
    source: str = "docs"
    code_blocks: list[dict[str, str]] | None = None


@dataclass
class Chunk:
    """An indexed document chunk."""

    chunk_id: int
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    code_blocks: list[dict[str, str]]
    source: str = "docs"
    fingerprint: str = ""
    embedding: np.ndarray | None = None


class SearchEngine:
    """Hybrid BM25 + vector search with weighted Reciprocal Rank Fusion (RRF)."""

    def __init__(self, enable_vectors: bool | None = None):
        if enable_vectors is None:
            disable_vectors = os.getenv("YELLOW_DOCS_DISABLE_VECTORS", "").strip().lower()
            enable_vectors = disable_vectors not in {"1", "true", "yes", "on"}
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._pages: list[DocPage] = []
        self._ready = False
        self._enable_vectors = enable_vectors

        self._max_chunk_chars = _parse_int_env("YELLOW_DOCS_MAX_CHUNK_CHARS", 1600, minimum=200)
        self._chunk_overlap_chars = _parse_int_env("YELLOW_DOCS_CHUNK_OVERLAP_CHARS", 200, minimum=0)
        self._rrf_k = _parse_int_env("YELLOW_DOCS_RRF_K", 60, minimum=1)
        self._candidate_pool = _parse_int_env("YELLOW_DOCS_CANDIDATE_POOL", 40, minimum=10, maximum=500)
        self._embedding_batch_size = _parse_int_env("YELLOW_DOCS_EMBED_BATCH_SIZE", 64, minimum=1, maximum=512)
        self._bm25_weight = _parse_float_env("YELLOW_DOCS_BM25_WEIGHT", 1.0, minimum=0.0)
        self._vector_weight = _parse_float_env("YELLOW_DOCS_VECTOR_WEIGHT", 1.0, minimum=0.0)

    def is_ready(self) -> bool:
        return self._ready

    def build_index(self, pages: list[DocPage], previous_index: dict[str, Any] | None = None) -> None:
        """Build a search index from parsed document pages."""
        self._ready = False
        self._pages = pages
        self._chunks = []

        chunk_id = 0
        for page in pages:
            source = page.source or "docs"
            for section in page.sections:
                code_block_dicts = [{"language": cb.language, "code": cb.code} for cb in section.code_blocks]
                text_chunks = self._split_text_chunks(section.text)
                if not text_chunks:
                    text_chunks = [section.text.strip()] if section.text.strip() else [""]
                for part_idx, chunk_text in enumerate(text_chunks):
                    fingerprint = self._fingerprint(
                        source=source,
                        path=page.path,
                        section_title=section.title,
                        part_idx=part_idx,
                        text=chunk_text,
                        code_blocks=code_block_dicts,
                    )
                    self._chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            path=page.path,
                            category=page.category,
                            page_title=page.title,
                            section_title=section.title,
                            text=chunk_text,
                            code_blocks=code_block_dicts,
                            source=source,
                            fingerprint=fingerprint,
                        )
                    )
                    chunk_id += 1

        if not self._chunks:
            logger.warning("No chunks to index")
            return

        tokenized = [self._tokenize(f"{c.page_title} {c.section_title} {c.text}") for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._embeddings = None

        previous_embeddings = self._previous_embeddings(previous_index)
        if self._enable_vectors:
            self._build_embeddings(previous_embeddings)

        self._ready = True
        mode = "hybrid" if self._embeddings is not None else "BM25-only"
        log_metric(
            logger,
            "search.index_built",
            mode=mode,
            page_count=len(pages),
            chunk_count=len(self._chunks),
            max_chunk_chars=self._max_chunk_chars,
            overlap_chars=self._chunk_overlap_chars,
        )

    def _split_text_chunks(self, text: str) -> list[str]:
        clean = text.strip()
        if not clean:
            return []
        if len(clean) <= self._max_chunk_chars:
            return [clean]

        chunks: list[str] = []
        start = 0
        step = max(1, self._max_chunk_chars - self._chunk_overlap_chars)
        while start < len(clean):
            end = min(len(clean), start + self._max_chunk_chars)
            if end < len(clean):
                split_floor = start + int(self._max_chunk_chars * 0.6)
                split_floor = min(split_floor, end)
                newline_cut = clean.rfind("\n", split_floor, end)
                space_cut = clean.rfind(" ", split_floor, end)
                cut = max(newline_cut, space_cut)
                if cut > start:
                    end = cut

            piece = clean[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(clean):
                break

            next_start = end - self._chunk_overlap_chars
            if next_start <= start:
                next_start = start + step
            start = next_start

        return chunks

    @staticmethod
    def _fingerprint(
        source: str,
        path: str,
        section_title: str,
        part_idx: int,
        text: str,
        code_blocks: list[dict[str, str]],
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(source.encode("utf-8"))
        hasher.update(path.encode("utf-8"))
        hasher.update(section_title.encode("utf-8"))
        hasher.update(str(part_idx).encode("utf-8"))
        hasher.update(text.encode("utf-8"))
        for block in code_blocks:
            hasher.update(block.get("language", "").encode("utf-8"))
            hasher.update(block.get("code", "").encode("utf-8"))
        return hasher.hexdigest()

    def _previous_embeddings(self, previous_index: dict[str, Any] | None) -> dict[str, np.ndarray]:
        if not previous_index:
            return {}

        chunks = previous_index.get("chunks") or []
        embeddings = previous_index.get("embeddings")
        if embeddings is None:
            return {}

        count = min(len(chunks), len(embeddings))
        mapping: dict[str, np.ndarray] = {}
        for idx in range(count):
            chunk = chunks[idx]
            fingerprint = getattr(chunk, "fingerprint", "") or self._fingerprint(
                source=getattr(chunk, "source", "docs") or "docs",
                path=getattr(chunk, "path", ""),
                section_title=getattr(chunk, "section_title", ""),
                part_idx=0,
                text=getattr(chunk, "text", ""),
                code_blocks=getattr(chunk, "code_blocks", []) or [],
            )
            mapping[fingerprint] = np.asarray(embeddings[idx])
        return mapping

    def _load_model(self) -> bool:
        """Load the embedding model lazily. Returns False if unavailable."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as exc:  # pragma: no cover - runtime/env dependent
            self._enable_vectors = False
            logger.warning("Embedding model unavailable, using BM25-only search: %s", exc)
            return False

    def _build_embeddings(self, previous_embeddings: dict[str, np.ndarray] | None = None) -> None:
        if not self._chunks:
            self._embeddings = None
            return
        if not self._load_model():
            self._embeddings = None
            return

        previous_embeddings = previous_embeddings or {}
        vectors: list[np.ndarray | None] = [None] * len(self._chunks)
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        reused = 0

        for idx, chunk in enumerate(self._chunks):
            reused_vector = previous_embeddings.get(chunk.fingerprint)
            if reused_vector is not None:
                vectors[idx] = reused_vector
                reused += 1
                continue
            pending_indices.append(idx)
            pending_texts.append(f"{chunk.page_title} - {chunk.section_title}\n{chunk.text}")

        generated = 0
        try:
            for start in range(0, len(pending_texts), self._embedding_batch_size):
                text_batch = pending_texts[start : start + self._embedding_batch_size]
                index_batch = pending_indices[start : start + self._embedding_batch_size]
                if not text_batch:
                    continue
                with timed_metric(logger, "search.embed_batch", size=len(text_batch)):
                    embedded = self._model.encode(
                        text_batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                for inner, vector in enumerate(embedded):
                    vectors[index_batch[inner]] = np.asarray(vector)
                    generated += 1
        except Exception as exc:  # pragma: no cover - runtime/env dependent
            self._embeddings = None
            logger.warning("Failed to generate embeddings, using BM25-only search: %s", exc)
            return

        if any(vector is None for vector in vectors):
            self._embeddings = None
            logger.warning("Embedding generation incomplete, using BM25-only search")
            return

        self._embeddings = np.asarray(vectors, dtype=np.float32)
        log_metric(
            logger,
            "search.embeddings_ready",
            total=len(self._chunks),
            reused=reused,
            generated=generated,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        source: str | None = None,
    ) -> list[SearchResult]:
        """Search with weighted RRF merge of BM25 and vector rankings."""
        if not self._ready or self._bm25 is None:
            return []
        if limit <= 0:
            return []

        candidate_indices = []
        for idx, chunk in enumerate(self._chunks):
            if category and chunk.category != category:
                continue
            if source and chunk.source != source:
                continue
            candidate_indices.append(idx)
        if not candidate_indices:
            return []

        with timed_metric(
            logger,
            "search.query",
            query_len=len(query),
            limit=limit,
            category=category or "all",
            source=source or "all",
            candidates=len(candidate_indices),
        ):
            bm25_scores = self._bm25.get_scores(self._tokenize(query))
            bm25_ranked = sorted(
                [(i, float(bm25_scores[i])) for i in candidate_indices],
                key=lambda x: x[1],
                reverse=True,
            )[: self._candidate_pool]

            vector_ranked: list[tuple[int, float]] = []
            if (
                self._enable_vectors
                and self._embeddings is not None
                and len(self._embeddings) == len(self._chunks)
                and self._load_model()
                and self._vector_weight > 0
            ):
                try:
                    query_emb = self._model.encode(query, convert_to_numpy=True)
                    query_norm = np.linalg.norm(query_emb)
                    if query_norm > 0:
                        similarities = np.dot(self._embeddings, query_emb) / (
                            np.linalg.norm(self._embeddings, axis=1) * query_norm + 1e-8
                        )
                        vector_ranked = sorted(
                            [(i, float(similarities[i])) for i in candidate_indices],
                            key=lambda x: x[1],
                            reverse=True,
                        )[: self._candidate_pool]
                except Exception as exc:  # pragma: no cover - runtime/env dependent
                    logger.warning("Vector scoring failed, using BM25-only ranking: %s", exc)

            rrf_scores: dict[int, float] = {}
            if self._bm25_weight > 0:
                for rank, (idx, _) in enumerate(bm25_ranked):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self._bm25_weight / (
                        self._rrf_k + rank + 1
                    )
            if self._vector_weight > 0:
                for rank, (idx, _) in enumerate(vector_ranked):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self._vector_weight / (
                        self._rrf_k + rank + 1
                    )
            if not rrf_scores:
                return []

            top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:limit]

        results: list[SearchResult] = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    path=chunk.path,
                    category=chunk.category,
                    page_title=chunk.page_title,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    score=rrf_scores[idx],
                    source=chunk.source,
                    code_blocks=chunk.code_blocks if chunk.code_blocks else None,
                )
            )
        return results

    def get_all_pages(self) -> list[DocPage]:
        return self._pages

    def get_page_by_path(self, path: str, source: str | None = None) -> DocPage | None:
        source_hint = source
        path_hint = path
        if source_hint is None and ":" in path:
            maybe_source, maybe_path = path.split(":", 1)
            if maybe_source and maybe_path:
                source_hint = maybe_source
                path_hint = maybe_path

        candidates = [p for p in self._pages if source_hint is None or p.source == source_hint]
        for page in candidates:
            if page.path == path_hint:
                return page
        for page in candidates:
            if page.path.endswith(path_hint):
                return page
        path_lower = path_hint.lower().replace("-", " ").replace("_", " ")
        for page in candidates:
            if path_lower in page.title.lower():
                return page
        return None

    def get_index_data(self) -> dict[str, Any]:
        return {
            "chunks": self._chunks,
            "embeddings": self._embeddings,
            "pages": self._pages,
            "enable_vectors": self._enable_vectors,
            "metadata": {
                "search_config": {
                    "max_chunk_chars": self._max_chunk_chars,
                    "chunk_overlap_chars": self._chunk_overlap_chars,
                    "rrf_k": self._rrf_k,
                    "candidate_pool": self._candidate_pool,
                    "bm25_weight": self._bm25_weight,
                    "vector_weight": self._vector_weight,
                }
            },
        }

    def load_index_data(self, data: dict[str, Any]) -> None:
        self._chunks = data["chunks"]
        self._embeddings = data.get("embeddings")
        self._pages = data["pages"]
        self._enable_vectors = bool(data.get("enable_vectors", self._enable_vectors))

        metadata = data.get("metadata") or {}
        search_config = metadata.get("search_config") or {}
        self._rrf_k = int(search_config.get("rrf_k", self._rrf_k))
        self._candidate_pool = int(search_config.get("candidate_pool", self._candidate_pool))
        self._bm25_weight = float(search_config.get("bm25_weight", self._bm25_weight))
        self._vector_weight = float(search_config.get("vector_weight", self._vector_weight))
        self._max_chunk_chars = int(search_config.get("max_chunk_chars", self._max_chunk_chars))
        self._chunk_overlap_chars = int(
            search_config.get("chunk_overlap_chars", self._chunk_overlap_chars)
        )

        for chunk in self._chunks:
            if not chunk.source:
                chunk.source = "docs"
            if not chunk.fingerprint:
                chunk.fingerprint = self._fingerprint(
                    source=chunk.source,
                    path=chunk.path,
                    section_title=chunk.section_title,
                    part_idx=0,
                    text=chunk.text,
                    code_blocks=chunk.code_blocks,
                )
        for page in self._pages:
            if not page.source:
                page.source = "docs"

        tokenized = [self._tokenize(f"{c.page_title} {c.section_title} {c.text}") for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._ready = True
        mode = "hybrid" if self._embeddings is not None else "BM25-only"
        log_metric(logger, "search.index_loaded", mode=mode, chunk_count=len(self._chunks))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

