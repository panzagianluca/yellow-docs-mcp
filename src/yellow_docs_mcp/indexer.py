"""Git repo management and multi-repo document indexing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from yellow_docs_mcp.parser import DEFAULT_EXCLUDE_DIRS, DocPage, parse_docs_directory
from yellow_docs_mcp.telemetry import log_metric, timed_metric

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"
STATE_FILE = "sources_state.json"
DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_MAX_HEADING_LEVEL = 3


@dataclass(frozen=True)
class RepoSource:
    """Repository source configuration."""

    name: str
    url: str
    branch: str | None = None
    doc_paths: tuple[str, ...] = field(
        default_factory=lambda: ("docs", "documentation", "doc", "manuals", "guides")
    )


DEFAULT_REPO_SOURCES: tuple[RepoSource, ...] = (
    RepoSource(name="docs", url="https://github.com/layer-3/docs.git", doc_paths=("docs",)),
    RepoSource(name="clearsync", url="https://github.com/layer-3/clearsync.git"),
    RepoSource(name="docs-gitbook", url="https://github.com/layer-3/docs-gitbook.git"),
    RepoSource(name="nitro", url="https://github.com/layer-3/nitro.git"),
    RepoSource(name="release-process", url="https://github.com/layer-3/release-process.git"),
)


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


def _source_name_from_url(url: str) -> str:
    base = url.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[:-4]
    return base or "repo"


def _sources_from_env() -> list[RepoSource] | None:
    raw = os.getenv("YELLOW_DOCS_REPOS", "").strip()
    if not raw:
        return None

    sources: list[RepoSource] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            name, url = token.split("=", 1)
            source_name = name.strip()
            source_url = url.strip()
        else:
            source_name = _source_name_from_url(token)
            source_url = token
        if not source_url.endswith(".git"):
            source_url = f"{source_url}.git"
        sources.append(RepoSource(name=source_name, url=source_url))
    return sources or None


class RepoManager:
    """Manages local clones of multiple documentation repositories."""

    def __init__(
        self,
        base_dir: Path | None = None,
        sources: Sequence[RepoSource] | None = None,
        max_file_bytes: int | None = None,
        max_heading_level: int | None = None,
    ):
        self.base_dir = base_dir or INDEX_DIR
        self.repos_dir = self.base_dir / "repos"
        self.state_file = self.base_dir / STATE_FILE
        self.exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
        self.max_file_bytes = max_file_bytes or _parse_int_env(
            "YELLOW_DOCS_MAX_FILE_BYTES",
            DEFAULT_MAX_FILE_BYTES,
            minimum=10_000,
        )
        self.max_heading_level = max_heading_level or _parse_int_env(
            "YELLOW_DOCS_MAX_HEADING_LEVEL",
            DEFAULT_MAX_HEADING_LEVEL,
            minimum=1,
            maximum=6,
        )
        self.sources: list[RepoSource] = list(
            sources or _sources_from_env() or DEFAULT_REPO_SOURCES
        )
        self._last_state: dict[str, dict[str, Any]] | None = None

    def sync_repo(self) -> bool:
        """Clone/pull all configured repositories. Returns True if content changed."""
        import git

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        previous_state = self._load_state()
        current_state: dict[str, dict[str, Any]] = {}
        changed = False

        with timed_metric(logger, "repo.sync_all", source_count=len(self.sources)):
            for source in self.sources:
                with timed_metric(logger, "repo.sync_source", source=source.name):
                    repo_dir = self._repo_dir(source)
                    commit_changed = self._sync_source(git, source, repo_dir)
                    roots = self._resolve_doc_roots(source, repo_dir)
                    content_hash = self._compute_content_hash(repo_dir, roots)
                    current_state[source.name] = {
                        "url": source.url,
                        "branch": source.branch or "",
                        "content_hash": content_hash,
                        "doc_roots": [str(root.relative_to(repo_dir)).replace("\\", "/") for root in roots],
                        "head_commit": self._head_commit(repo_dir),
                    }
                    prev = previous_state.get(source.name)
                    if (
                        prev is None
                        or commit_changed
                        or prev.get("content_hash") != content_hash
                        or prev.get("head_commit") != current_state[source.name]["head_commit"]
                        or prev.get("doc_roots") != current_state[source.name]["doc_roots"]
                    ):
                        changed = True

            if set(previous_state.keys()) != set(current_state.keys()):
                changed = True

        self._last_state = current_state
        log_metric(logger, "repo.sync_result", changed=changed, source_count=len(self.sources))
        return changed

    def needs_reindex(self) -> bool:
        """Check if local repository content changed since last index build."""
        previous_state = self._load_state()
        if not previous_state:
            return True

        current_state = self._collect_local_state()
        if set(previous_state.keys()) != set(current_state.keys()):
            return True

        for name, prev in previous_state.items():
            curr = current_state.get(name)
            if curr is None:
                return True
            if prev.get("content_hash") != curr.get("content_hash"):
                return True
            if prev.get("doc_roots") != curr.get("doc_roots"):
                return True
        return False

    def save_hash(self) -> None:
        """Persist source state after successful indexing."""
        state = self._last_state or self._collect_local_state()
        self._save_state(state)

    def parse_all_docs(self) -> list[DocPage]:
        """Parse all markdown documents across configured repositories."""
        pages: list[DocPage] = []
        for source in self.sources:
            repo_dir = self._repo_dir(source)
            if not repo_dir.exists():
                continue

            roots = self._resolve_doc_roots(source, repo_dir)
            if not roots:
                continue

            multiple_roots = len(roots) > 1
            for root in roots:
                root_rel = str(root.relative_to(repo_dir)).replace("\\", "/")
                path_prefix = ""
                if multiple_roots:
                    path_prefix = root_rel
                elif root != repo_dir and root_rel not in ("", "docs"):
                    path_prefix = root_rel
                pages.extend(
                    parse_docs_directory(
                        root,
                        source=source.name,
                        path_prefix=path_prefix,
                        max_heading_level=self.max_heading_level,
                        exclude_dirs=self.exclude_dirs,
                        max_file_bytes=self.max_file_bytes,
                    )
                )

        pages.sort(key=lambda p: (p.source, p.path))
        log_metric(logger, "repo.parse_all_docs", page_count=len(pages))
        return pages

    def _repo_dir(self, source: RepoSource) -> Path:
        return self.repos_dir / source.name

    def _sync_source(self, git_module: Any, source: RepoSource, repo_dir: Path) -> bool:
        if not repo_dir.exists():
            logger.info("Cloning %s (%s)...", source.name, source.url)
            clone_kwargs: dict[str, Any] = {"depth": 1}
            if source.branch:
                clone_kwargs["branch"] = source.branch
            git_module.Repo.clone_from(source.url, str(repo_dir), **clone_kwargs)
            return True

        repo = git_module.Repo(str(repo_dir))
        old_head = self._head_commit(repo_dir)
        origin = repo.remotes.origin

        if source.branch and not repo.head.is_detached and repo.active_branch.name != source.branch:
            repo.git.checkout(source.branch)

        if repo.head.is_detached:
            origin.pull()
        else:
            origin.pull(repo.active_branch.name)

        new_head = self._head_commit(repo_dir)
        return old_head != new_head

    def _resolve_doc_roots(self, source: RepoSource, repo_dir: Path) -> list[Path]:
        candidates: list[Path] = []
        for rel in source.doc_paths:
            candidate = repo_dir / rel
            if candidate.exists() and candidate.is_dir() and self._contains_markdown(candidate):
                candidates.append(candidate)

        if not candidates:
            for rel in ("docs", "documentation", "doc", "manuals", "guides"):
                candidate = repo_dir / rel
                if candidate.exists() and candidate.is_dir() and self._contains_markdown(candidate):
                    candidates.append(candidate)

        if not candidates and repo_dir.exists() and self._contains_markdown(repo_dir):
            candidates.append(repo_dir)

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _contains_markdown(self, root: Path) -> bool:
        for _ in self._iter_markdown_files(root):
            return True
        return False

    def _iter_markdown_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for dir_path, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            path_obj = Path(dir_path)
            for filename in filenames:
                if not filename.endswith((".md", ".mdx")):
                    continue
                full_path = path_obj / filename
                if full_path.stat().st_size > self.max_file_bytes:
                    continue
                files.append(full_path)
        return sorted(files)

    def _collect_local_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for source in self.sources:
            repo_dir = self._repo_dir(source)
            if not repo_dir.exists():
                continue
            roots = self._resolve_doc_roots(source, repo_dir)
            state[source.name] = {
                "url": source.url,
                "branch": source.branch or "",
                "content_hash": self._compute_content_hash(repo_dir, roots),
                "doc_roots": [str(root.relative_to(repo_dir)).replace("\\", "/") for root in roots],
                "head_commit": self._head_commit(repo_dir),
            }
        return state

    def _compute_content_hash(self, repo_dir: Path, roots: list[Path]) -> str:
        hasher = hashlib.sha256()
        for root in sorted(roots, key=lambda p: str(p)):
            for file_path in self._iter_markdown_files(root):
                rel = str(file_path.relative_to(repo_dir)).replace("\\", "/")
                hasher.update(rel.encode("utf-8"))
                hasher.update(file_path.read_bytes())
        return hasher.hexdigest()

    def _head_commit(self, repo_dir: Path) -> str:
        try:
            import git

            repo = git.Repo(str(repo_dir))
            return repo.head.commit.hexsha
        except Exception:
            return ""

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read state file %s: %s", self.state_file, exc)
            return {}

    def _save_state(self, state: dict[str, dict[str, Any]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_file)
