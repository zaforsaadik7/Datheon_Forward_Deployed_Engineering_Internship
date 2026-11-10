import hashlib
import math
from pathlib import Path
from typing import List, Optional, Dict
import pytest
import pymupdf
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.etl.models import DocumentChunk
from src.etl.vector_store import VectorStore


class FakeEmbedding(EmbeddingFunction[Documents]):
    """
    Deterministic 16-dimension embedding function for testing.
    Maps known texts to explicit vectors or hashes text to unit vectors.
    """
    def __init__(self, known_vectors: Optional[Dict[str, List[float]]] = None):
        self.known_vectors = known_vectors or {}

    @staticmethod
    def name() -> str:
        return "fake-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbedding":
        return FakeEmbedding()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            if text in self.known_vectors:
                embeddings.append(self.known_vectors[text])
            else:
                # Deterministic 16-dim normalized vector
                hasher = hashlib.md5(text.encode("utf-8")).digest()
                raw = [float(b) for b in hasher[:16]]
                norm = math.sqrt(sum(x * x for x in raw)) or 1.0
                embeddings.append([x / norm for x in raw])
        return embeddings


@pytest.fixture
def fake_embedder():
    return FakeEmbedding()


@pytest.fixture
def tmp_store(tmp_path: Path):
    embedder = FakeEmbedding()
    store = VectorStore(
        path=tmp_path / "test_chroma",
        collection_name="test_collection",
        embedding_function=embedder,
    )
    return store


@pytest.fixture
def chunk_factory():
    def _create(**overrides):
        defaults = {
            "chunk_id": "test_chk_1",
            "document_id": "doc1.txt",
            "text": "Sample chunk content",
            "department": "engineering",
            "clearance_tier": 2,
            "document_date": "2026-01-01",
            "source_file": "doc1.txt",
            "file_type": "txt",
            "chunk_index": 0,
            "page_number": None,
            "row_id": None,
            "section_title": None,
            "token_count": 5,
        }
        defaults.update(overrides)
        return DocumentChunk(**defaults)
    return _create


@pytest.fixture
def make_pdf():
    """Helper fixture to create PDF files with headers and footers."""
    def _make_pdf(
        path: Path,
        pages: List[str],
        header: Optional[str] = None,
        footer_template: Optional[str] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        total_pages = len(pages)

        for page_idx, page_text in enumerate(pages, start=1):
            page = doc.new_page(width=595, height=842)  # A4

            # Draw header
            if header:
                page.insert_text((50, 40), header, fontsize=10)

            # Draw body text with insert_textbox
            page.insert_textbox(pymupdf.Rect(50, 60, 545, 780), page_text, fontsize=11)

            # Draw footer
            if footer_template:
                footer_str = footer_template.format(n=page_idx, total=total_pages)
                page.insert_text((50, 810), footer_str, fontsize=10)

        doc.save(str(path))
        doc.close()
        return path

    return _make_pdf
