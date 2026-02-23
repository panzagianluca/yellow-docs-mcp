"""Parse MDX/MD documents into structured sections."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeBlock:
    """A fenced code block extracted from a document."""
    language: str
    code: str


@dataclass
class DocSection:
    """A section of a document (split by headings)."""
    title: str
    level: int
    text: str
    code_blocks: list[CodeBlock] = field(default_factory=list)


@dataclass
class DocPage:
    """A parsed document page."""
    path: str
    category: str
    title: str
    description: str
    keywords: list[str]
    sections: list[DocSection]
    raw_content: str
    source: str = "docs"


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".next",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
}


def _extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from document content."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    import yaml
    fm_text = content[3:end].strip()
    body = content[end + 3:].strip()
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def _strip_imports(content: str) -> str:
    """Remove JSX import statements."""
    lines = content.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and ("from " in stripped or "require(" in stripped):
            if "'" in stripped or '"' in stripped:
                continue
        filtered.append(line)
    return "\n".join(filtered)


def _strip_tooltip_tags(content: str) -> str:
    """Replace <Tooltip ...>text</Tooltip> with just text."""
    pattern = r'<Tooltip[^>]*>(.*?)</Tooltip>'
    return re.sub(pattern, r'\1', content, flags=re.DOTALL)


def _strip_tabs_components(content: str) -> str:
    """Strip Tabs/TabItem JSX wrappers, keep inner content."""
    content = re.sub(r'<Tabs[^>]*>', '', content)
    content = re.sub(r'</Tabs>', '', content)
    content = re.sub(r'<TabItem[^>]*>', '', content)
    content = re.sub(r'</TabItem>', '', content)
    return content


def _extract_code_blocks(text: str) -> tuple[str, list[CodeBlock]]:
    """Extract fenced code blocks from text, return cleaned text and blocks."""
    blocks = []
    pattern = r'```(\w*)\n(.*?)```'
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        blocks.append(CodeBlock(language=lang, code=code))
        return ""
    cleaned = re.sub(pattern, replacer, text, flags=re.DOTALL)
    return cleaned.strip(), blocks


def _extract_admonition_text(text: str) -> str:
    """Convert :::type content ::: to plain text, keeping content."""
    lines = text.split("\n")
    result = []
    in_admonition = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":::") and not in_admonition:
            in_admonition = True
            continue
        elif stripped == ":::" and in_admonition:
            in_admonition = False
            continue
        result.append(line)
    return "\n".join(result)


def _split_sections(content: str, max_heading_level: int = 3) -> list[DocSection]:
    """Split content into sections by headings."""
    max_heading_level = min(max(max_heading_level, 1), 6)
    heading_pattern = rf'^(#{{1,{max_heading_level}}})\s+(.+)$'
    lines = content.split("\n")
    sections: list[DocSection] = []
    current_title = ""
    current_level = 1
    current_lines: list[str] = []

    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            if current_lines or current_title:
                raw_text = "\n".join(current_lines).strip()
                text, code_blocks = _extract_code_blocks(raw_text)
                text = _extract_admonition_text(text)
                text = text.strip()
                if text or code_blocks:
                    sections.append(DocSection(
                        title=current_title,
                        level=current_level,
                        text=text,
                        code_blocks=code_blocks,
                    ))
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_title:
        raw_text = "\n".join(current_lines).strip()
        text, code_blocks = _extract_code_blocks(raw_text)
        text = _extract_admonition_text(text)
        text = text.strip()
        if text or code_blocks:
            sections.append(DocSection(
                title=current_title,
                level=current_level,
                text=text,
                code_blocks=code_blocks,
            ))
    return sections


def _title_from_content(content: str) -> str:
    """Extract title from first # heading if no frontmatter title."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def parse_document(
    content: str,
    path: str,
    source: str = "docs",
    max_heading_level: int = 3,
) -> DocPage:
    """Parse a MDX/MD document into structured sections."""
    frontmatter, body = _extract_frontmatter(content)
    body = _strip_imports(body)
    body = _strip_tooltip_tags(body)
    body = _strip_tabs_components(body)
    title = frontmatter.get("title", _title_from_content(body))
    description = frontmatter.get("description", "")
    keywords = frontmatter.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    category = path.split("/")[0] if "/" in path else ""
    sections = _split_sections(body, max_heading_level=max_heading_level)
    return DocPage(
        path=path, category=category, title=title, description=description,
        keywords=keywords, sections=sections, raw_content=content, source=source,
    )


def _iter_doc_files(
    docs_dir: Path,
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
    results: list[Path] = []
    for root, dirs, files in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        root_path = Path(root)
        for filename in files:
            if not filename.endswith((".md", ".mdx")):
                continue
            results.append(root_path / filename)
    return sorted(results)


def parse_docs_directory(
    docs_dir: Path,
    source: str = "docs",
    path_prefix: str = "",
    max_heading_level: int = 3,
    exclude_dirs: set[str] | None = None,
    max_file_bytes: int = 2_000_000,
) -> list[DocPage]:
    """Parse all MDX/MD files in a docs directory."""
    pages = []
    for filepath in _iter_doc_files(docs_dir, exclude_dirs=exclude_dirs):
        if filepath.stat().st_size > max_file_bytes:
            continue
        rel_path = str(filepath.relative_to(docs_dir)).replace("\\", "/")
        if path_prefix:
            rel_path = f"{path_prefix.strip('/')}/{rel_path}"
        content = filepath.read_text(encoding="utf-8")
        page = parse_document(
            content,
            rel_path,
            source=source,
            max_heading_level=max_heading_level,
        )
        pages.append(page)
    return pages
