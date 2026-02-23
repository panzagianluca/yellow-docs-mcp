"""Tests for JSON/NPZ index persistence."""
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
