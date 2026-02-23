"""Tests for JSON/NPZ index persistence."""
import json

import numpy as np

from yellow_docs_mcp.parser import CodeBlock, DocPage, DocSection
from yellow_docs_mcp.search import Chunk
from yellow_docs_mcp.store import load_index, save_index


def _sample_index_data() -> dict:
    return {
        "chunks": [
            Chunk(
                chunk_id=0,
                path="protocol/off-chain/channel-methods.mdx",
                category="protocol",
                page_title="Channel Management Methods",
                section_title="create_channel",
                text="Creates a new payment channel.",
                code_blocks=[{"language": "json", "code": '{"req":[1,"create_channel"]}'}],
            )
        ],
        "embeddings": np.array([[0.1, 0.2]], dtype=np.float32),
        "pages": [
            DocPage(
                path="protocol/off-chain/channel-methods.mdx",
                category="protocol",
                title="Channel Management Methods",
                description="RPC methods",
                keywords=["rpc", "channels"],
                sections=[
                    DocSection(
                        title="create_channel",
                        level=2,
                        text="Creates a new payment channel.",
                        code_blocks=[CodeBlock(language="json", code='{"req":[1,"create_channel"]}')],
                    )
                ],
                raw_content="## create_channel",
            )
        ],
        "enable_vectors": True,
    }


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "index.json"
    save_index(_sample_index_data(), path)

    loaded = load_index(path)
    assert loaded is not None
    assert len(loaded["chunks"]) == 1
    assert loaded["chunks"][0].path == "protocol/off-chain/channel-methods.mdx"
    assert len(loaded["pages"]) == 1
    assert loaded["pages"][0].title == "Channel Management Methods"
    assert loaded["embeddings"] is not None
    assert loaded["embeddings"].shape == (1, 2)


def test_load_missing_index(tmp_path):
    assert load_index(tmp_path / "missing.json") is None


def test_load_corrupted_index(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not-valid-json")
    assert load_index(path) is None


def test_migrate_v2_payload(tmp_path):
    path = tmp_path / "index.json"
    npz_path = tmp_path / "index.npz"
    payload_v2 = {
        "version": 2,
        "chunks": [
            {
                "chunk_id": 0,
                "path": "protocol/off-chain/channel-methods.mdx",
                "category": "protocol",
                "page_title": "Channel Management Methods",
                "section_title": "create_channel",
                "text": "Creates a new payment channel.",
                "code_blocks": [{"language": "json", "code": '{"req":[1,"create_channel"]}'}],
            }
        ],
        "pages": [
            {
                "path": "protocol/off-chain/channel-methods.mdx",
                "category": "protocol",
                "title": "Channel Management Methods",
                "description": "",
                "keywords": [],
                "sections": [
                    {
                        "title": "create_channel",
                        "level": 2,
                        "text": "Creates a new payment channel.",
                        "code_blocks": [{"language": "json", "code": '{"req":[1,"create_channel"]}'}],
                    }
                ],
                "raw_content": "## create_channel",
            }
        ],
        "has_embeddings": True,
        "enable_vectors": True,
    }
    path.write_text(json.dumps(payload_v2), encoding="utf-8")
    with open(npz_path, "wb") as f:
        np.savez_compressed(f, embeddings=np.array([[0.1, 0.2]], dtype=np.float32))

    loaded = load_index(path)
    assert loaded is not None
    assert loaded["chunks"][0].source == "docs"
    assert loaded["chunks"][0].fingerprint
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    assert rewritten["version"] == 3
