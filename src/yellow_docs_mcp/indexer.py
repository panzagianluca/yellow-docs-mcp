"""Git repo management and document indexing."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from yellow_docs_mcp.parser import DocPage, parse_docs_directory

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"
REPO_URL = "https://github.com/layer-3/docs.git"
BRANCH: str | None = None


class RepoManager:
    """Manages the local clone of the docs repo."""

    def __init__(
        self,
        base_dir: Path | None = None,
        repo_url: str = REPO_URL,
        branch: str | None = BRANCH,
    ):
        self.base_dir = base_dir or INDEX_DIR
        self.repo_url = repo_url
        self.branch = branch
        self.repo_dir = self.base_dir / "repo"
        self.docs_dir = self.repo_dir / "docs"
        self.hash_file = self.base_dir / "content_hash.txt"

    def sync_repo(self) -> bool:
        """Clone or pull the docs repo. Returns True if content changed."""
        import git
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.repo_dir.exists():
            logger.info("Cloning %s...", self.repo_url)
            clone_kwargs = {"depth": 1}
            if self.branch:
                clone_kwargs["branch"] = self.branch
            git.Repo.clone_from(self.repo_url, str(self.repo_dir), **clone_kwargs)
            return True
        else:
            logger.info("Pulling latest changes...")
            repo = git.Repo(str(self.repo_dir))
            origin = repo.remotes.origin
            if self.branch and not repo.head.is_detached:
                if repo.active_branch.name != self.branch:
                    repo.git.checkout(self.branch)
            old_hash = self._compute_content_hash(self.docs_dir)
            if repo.head.is_detached:
                origin.pull()
            else:
                origin.pull(repo.active_branch.name)
            new_hash = self._compute_content_hash(self.docs_dir)
            changed = old_hash != new_hash
            if changed:
                logger.info("Docs content changed, re-indexing needed.")
            else:
                logger.info("No content changes detected.")
            return changed

    def needs_reindex(self) -> bool:
        """Check if docs have changed since last index build."""
        if not self.docs_dir.exists():
            return True
        if not self.hash_file.exists():
            return True
        stored_hash = self.hash_file.read_text().strip()
        current_hash = self._compute_content_hash(self.docs_dir)
        return stored_hash != current_hash

    def save_hash(self) -> None:
        """Save current content hash after successful indexing."""
        current_hash = self._compute_content_hash(self.docs_dir)
        self.hash_file.write_text(current_hash)

    def parse_all_docs(self) -> list[DocPage]:
        """Parse all documents in the docs directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        return parse_docs_directory(self.docs_dir)

    def _compute_content_hash(self, docs_dir: Path) -> str:
        """Compute a hash of all doc file contents for change detection."""
        hasher = hashlib.sha256()
        files = sorted(docs_dir.rglob("*"))
        for f in files:
            if f.is_file() and f.suffix in (".md", ".mdx"):
                hasher.update(f.read_bytes())
                hasher.update(str(f.relative_to(docs_dir)).encode())
        return hasher.hexdigest()
