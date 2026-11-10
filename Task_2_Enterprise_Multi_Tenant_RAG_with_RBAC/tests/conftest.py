"""Shared pytest fixtures, keyword-aware deterministic FakeEmbedding, and test helpers."""
import hashlib
import math
import re
from pathlib import Path
from typing import Callable, List
import fitz
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.config import DEFAULT_TENANT
from src.etl.models import DocumentChunk
from src.etl.vector_store import VectorStore

# 16 semantic topic buckets so fast tests with FakeEmbedding still route domain queries
# to the matching sample documents without loading external neural weights.
TOPIC_BUCKETS: tuple[tuple[str, ...], ...] = (
    ("fido2", "mfa", "bitlocker", "filevault", "crowdstrike", "mdm", "workstations"),
    ("password", "16", "characters", "hygiene", "classification", "lockout"),
    ("soc", "incident", "phishing", "soc-alerts", "4911", "triage"),
    ("supplier", "apex", "meridian", "vanguard", "sla", "99.95", "99.99", "sup-101", "sup-102"),
    ("vpc", "peering", "cidr", "10.42.0.0/16", "vpc-prod-compute", "tgw", "transit"),
    ("nat", "egress", "bgp", "64512", "bfd", "failover", "10.42.16.0/20"),
    ("salary", "compensation", "l3", "l4", "l5", "l6", "m3", "staff", "185000", "210000", "240000"),
    ("budget", "q3", "fin-q3-001", "fin-q3-002", "snowflake", "450000", "480000", "-30000", "variance"),
    ("roadmap", "capex", "14.5m", "14,500,000", "ebitda", "7.2m", "7,200,000", "340", "fy2026"),
    ("titan", "valuation", "184,500,000", "184.5m", "buyout", "92,000,000", "retention", "25,000,000"),
    ("escrow", "indemnity", "18,450,000", "morgan", "covenant", "22,000,000", "arr", "non-compete"),
    ("aegis-x", "royalty", "3,500,000", "4.5%", "900,000", "11,402,891", "quantum", "license"),
)


class FakeEmbedding(EmbeddingFunction[Documents]):
    """
    Deterministic 32-dim unit-vector embedding function for fast, offline unit tests.
    Combines keyword topic activation (first 12 dims) with a low-amplitude SHA-256 hash
    signature (remaining 20 dims) so both exact-match and domain-query tests behave
    predictably.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "fake_embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbedding":
        return FakeEmbedding()

    def __call__(self, input: Documents) -> Embeddings:
        results: Embeddings = []
        for text in input:
            lowered = text.lower()
            tokens = set(re.findall(r"[a-z0-9_.\-$%]+", lowered))
            vec: List[float] = []

            matched_any_bucket = False
            for bucket in TOPIC_BUCKETS:
                hits = sum(1 for kw in bucket if kw in lowered or kw in tokens)
                if hits > 0:
                    matched_any_bucket = True
                vec.append(float(hits) * 3.0)

            digest = hashlib.sha256(text.encode("utf-8")).digest()
            hash_scale = 0.15 if matched_any_bucket else 1.0
            for i in range(20):
                val = ((digest[i] / 255.0) * 2.0 - 1.0) * hash_scale
                vec.append(val)

            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results


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

