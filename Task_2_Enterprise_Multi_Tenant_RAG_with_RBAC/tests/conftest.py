"""Shared pytest fixtures, domain-agnostic feature-hashing FakeEmbedding, and test helpers."""
from pathlib import Path
from typing import Callable, List
import fitz
import pytest

from src.config import DEFAULT_TENANT
from src.etl.models import DocumentChunk
from src.etl.vector_store import VectorStore
from src.rag.cli import DeterministicHashEmbedding

FakeEmbedding = DeterministicHashEmbedding


@pytest.fixture
def fake_embedder() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def tmp_store(tmp_path: Path, fake_embedder: FakeEmbedding) -> VectorStore:
    return VectorStore(
        path=tmp_path / "chroma_test",
        collection_name="test_docs",
        embedding_function=fake_embedder,
    )


@pytest.fixture
def chunk_factory() -> Callable[..., DocumentChunk]:
    counter = {"n": 0}

    def _make(
        *,
        chunk_id: str | None = None,
        document_id: str = "doc.txt",
        text: str = "Sample chunk content for testing vector operations.",
        department: str = "engineering",
        clearance_tier: int = 2,
        document_date: str = "2026-10-20",
        source_file: str = "doc.txt",
        file_type: str = "txt",
        chunk_index: int = 0,
        tenant_id: str = DEFAULT_TENANT,
        page_number: int | None = None,
        row_id: int | None = None,
        section_title: str | None = None,
        token_count: int = 12,
    ) -> DocumentChunk:
        counter["n"] += 1
        cid = chunk_id or f"chunk_{counter['n']}"
        return DocumentChunk(
            chunk_id=cid,
            document_id=document_id,
            text=text,
            department=department,
            clearance_tier=clearance_tier,
            document_date=document_date,
            source_file=source_file,
            file_type=file_type,
            chunk_index=chunk_index,
            tenant_id=tenant_id,
            page_number=page_number,
            row_id=row_id,
            section_title=section_title,
            token_count=token_count,
        )

    return _make


@pytest.fixture
def make_pdf() -> Callable[[Path, List[str]], Path]:
    def _create(path: Path, pages: List[str]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        for page_text in pages:
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 550, 780), page_text, fontsize=11)
        doc.save(str(path))
        doc.close()
        return path

    return _create
