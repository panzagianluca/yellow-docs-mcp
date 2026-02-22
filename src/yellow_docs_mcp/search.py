"""Hybrid search engine combining BM25 keyword search and vector similarity."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

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
    code_blocks: list[dict] | None = None


@dataclass
class Chunk:
    """An indexed document chunk (one per section)."""
    chunk_id: int
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    code_blocks: list[dict]
    embedding: np.ndarray | None = None


class SearchEngine:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._pages: list[DocPage] = []
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def build_index(self, pages: list[DocPage]) -> None:
        """Build search index from parsed document pages."""
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
        self._build_embeddings()
        self._ready = True
        logger.info("Index built: %d chunks from %d pages", len(self._chunks), len(pages))

    def _build_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{c.page_title} - {c.section_title}\n{c.text}" for c in self._chunks]
        self._embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info("Generated %d embeddings", len(self._embeddings))

    def search(self, query: str, limit: int = 5, category: str | None = None) -> list[SearchResult]:
        """Search using hybrid BM25 + vector with RRF merge."""
        if not self._ready:
            return []

        if category:
            valid_ids = {c.chunk_id for c in self._chunks if c.category == category}
        else:
            valid_ids = {c.chunk_id for c in self._chunks}

        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        bm25_ranked = sorted(
            [(i, s) for i, s in enumerate(bm25_scores) if self._chunks[i].chunk_id in valid_ids],
            key=lambda x: x[1], reverse=True,
        )[:20]

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        query_emb = self._model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        vector_ranked = sorted(
            [(i, float(s)) for i, s in enumerate(similarities) if self._chunks[i].chunk_id in valid_ids],
            key=lambda x: x[1], reverse=True,
        )[:20]

        k = 60
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, (idx, _) in enumerate(vector_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

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
        return {"chunks": self._chunks, "embeddings": self._embeddings, "pages": self._pages}

    def load_index_data(self, data: dict) -> None:
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self._pages = data["pages"]
        tokenized = [self._tokenize(c.text + " " + c.section_title + " " + c.page_title) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._ready = True
        logger.info("Index loaded: %d chunks from %d pages", len(self._chunks), len(self._pages))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
