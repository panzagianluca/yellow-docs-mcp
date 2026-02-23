"""Hybrid search engine combining BM25 keyword search and optional vector similarity."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from yellow_docs_mcp.parser import DocPage

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    score: float
    code_blocks: list[dict[str, str]] | None = None


@dataclass
class Chunk:
    """An indexed document chunk (one per section)."""
    chunk_id: int
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    code_blocks: list[dict[str, str]]
    embedding: np.ndarray | None = None


class SearchEngine:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)."""

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

    def is_ready(self) -> bool:
        return self._ready

    def build_index(self, pages: list[DocPage]) -> None:
        """Build search index from parsed document pages."""
        self._ready = False
        self._pages = pages
        self._chunks = []
        chunk_id = 0
        for page in pages:
            for section in page.sections:
                code_block_dicts = [
                    {"language": cb.language, "code": cb.code}
                    for cb in section.code_blocks
                ]
                self._chunks.append(Chunk(
                    chunk_id=chunk_id,
                    path=page.path,
                    category=page.category,
                    page_title=page.title,
                    section_title=section.title,
                    text=section.text,
                    code_blocks=code_block_dicts,
                ))
                chunk_id += 1

        if not self._chunks:
            logger.warning("No chunks to index!")
            return

        tokenized = [self._tokenize(c.text + " " + c.section_title + " " + c.page_title) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._embeddings = None
        if self._enable_vectors:
            self._build_embeddings()
        self._ready = True
        mode = "hybrid" if self._embeddings is not None else "BM25-only"
        logger.info("Index built (%s): %d chunks from %d pages", mode, len(self._chunks), len(pages))

    def _load_model(self) -> bool:
        """Load the embedding model lazily. Returns False when unavailable."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._enable_vectors = False
            logger.warning("Embedding model unavailable, using BM25-only search: %s", exc)
            return False

    def _build_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        if not self._chunks:
            self._embeddings = None
            return
        if not self._load_model():
            self._embeddings = None
            return
        try:
            texts = [f"{c.page_title} - {c.section_title}\n{c.text}" for c in self._chunks]
            self._embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            logger.info("Generated %d embeddings", len(self._embeddings))
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._embeddings = None
            logger.warning("Failed to generate embeddings, using BM25-only search: %s", exc)

    def search(self, query: str, limit: int = 5, category: str | None = None) -> list[SearchResult]:
        """Search using hybrid BM25 + vector with RRF merge."""
        if not self._ready or self._bm25 is None:
            return []
        if limit <= 0:
            return []

        if category:
            candidate_indices = [i for i, chunk in enumerate(self._chunks) if chunk.category == category]
        else:
            candidate_indices = list(range(len(self._chunks)))
        if not candidate_indices:
            return []

        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        bm25_ranked = sorted(
            [(i, float(bm25_scores[i])) for i in candidate_indices],
            key=lambda x: x[1], reverse=True,
        )[:20]

        vector_ranked: list[tuple[int, float]] = []
        if (
            self._enable_vectors
            and self._embeddings is not None
            and len(self._embeddings) == len(self._chunks)
            and self._load_model()
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
                        key=lambda x: x[1], reverse=True,
                    )[:20]
            except Exception as exc:  # pragma: no cover - depends on runtime env
                logger.warning("Vector query scoring failed, using BM25-only ranking: %s", exc)

        k = 60
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, (idx, _) in enumerate(vector_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        if not rrf_scores:
            return []

        top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:limit]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(SearchResult(
                path=chunk.path, category=chunk.category,
                page_title=chunk.page_title, section_title=chunk.section_title,
                text=chunk.text, score=rrf_scores[idx],
                code_blocks=chunk.code_blocks if chunk.code_blocks else None,
            ))
        return results

    def get_all_pages(self) -> list[DocPage]:
        return self._pages

    def get_page_by_path(self, path: str) -> DocPage | None:
        for page in self._pages:
            if page.path == path:
                return page
        for page in self._pages:
            if page.path.endswith(path):
                return page
        path_lower = path.lower().replace("-", " ").replace("_", " ")
        for page in self._pages:
            if path_lower in page.title.lower():
                return page
        return None

    def get_index_data(self) -> dict:
        return {
            "chunks": self._chunks,
            "embeddings": self._embeddings,
            "pages": self._pages,
            "enable_vectors": self._enable_vectors,
        }

    def load_index_data(self, data: dict) -> None:
        self._chunks = data["chunks"]
        self._embeddings = data.get("embeddings")
        self._pages = data["pages"]
        self._enable_vectors = bool(data.get("enable_vectors", self._enable_vectors))
        tokenized = [self._tokenize(c.text + " " + c.section_title + " " + c.page_title) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._ready = True
        mode = "hybrid" if self._embeddings is not None else "BM25-only"
        logger.info("Index loaded (%s): %d chunks from %d pages", mode, len(self._chunks), len(self._pages))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
