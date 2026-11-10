from pathlib import Path
import pytest
from src.etl.vector_store import VectorStore
from tests.conftest import FakeEmbedding


def test_collection_uses_cosine_distance(tmp_path: Path):
    # Vector [1, 1, 0, ...] vs Query [1, 0, 0, ...]
    v_doc = [1.0, 1.0] + [0.0] * 14
    v_query = [1.0, 0.0] + [0.0] * 14

    embedder = FakeEmbedding(known_vectors={"doc_text": v_doc, "query_text": v_query})
    store = VectorStore(
        path=tmp_path / "chroma",
        collection_name="cosine_test",
        embedding_function=embedder,
    )

    store.collection.add(
        ids=["doc1"],
        documents=["doc_text"],
        embeddings=[v_doc],
        metadatas=[{"clearance_tier": 1, "department": "engineering"}]
    )

    hits = store.query_by_tier_and_department(
        query_text="query_text",
        user_tier=4,
        user_departments=None,
        top_k=1
    )
    assert len(hits) == 1
    # Cosine distance = 1 - 1/sqrt(2) = 0.2929
    assert abs(hits[0].distance - 0.2929) <= 0.005


def test_upsert_is_idempotent(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", text="text 1", clearance_tier=2)
    tmp_store.upsert_chunks([c1])
    assert tmp_store.count() == 1

    # Re-upsert with modified tier
    c1_updated = chunk_factory(chunk_id="c1", text="text 1", clearance_tier=3)
    tmp_store.upsert_chunks([c1_updated])
    assert tmp_store.count() == 1

    meta = tmp_store.collection.get(ids=["c1"])["metadatas"][0]
    assert meta["clearance_tier"] == 3


def test_batch_size_clamped_to_chroma_limit(tmp_store):
    max_batch = tmp_store.client.get_max_batch_size()
    assert tmp_store.effective_batch_size(10**9) == max_batch
    assert tmp_store.effective_batch_size(100) == 100


def test_metadata_has_no_none_values(tmp_store, chunk_factory):
    c = chunk_factory(chunk_id="c1", page_number=None, row_id=None)
    tmp_store.upsert_chunks([c])
    stored_meta = tmp_store.collection.get(ids=["c1"])["metadatas"][0]
    assert "page_number" not in stored_meta
    assert "row_id" not in stored_meta
    assert None not in stored_meta.values()


def test_tier_filter_excludes_higher_tiers(tmp_store, chunk_factory):
    chunks = [
        chunk_factory(chunk_id=f"c{t}", text="content", clearance_tier=t, department="engineering")
        for t in [1, 2, 3, 4]
    ]
    tmp_store.upsert_chunks(chunks)
    hits = tmp_store.query_by_tier_and_department(
        query_text="content",
        user_tier=2,
        user_departments=None,
        top_k=100
    )
    assert len(hits) == 2
    returned_tiers = {h.metadata["clearance_tier"] for h in hits}
    assert returned_tiers == {1, 2}


def test_department_filter_is_case_insensitive(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", text="q3 spend", department="finance", clearance_tier=2)
    c2 = chunk_factory(chunk_id="c2", text="vpc route", department="engineering", clearance_tier=2)
    tmp_store.upsert_chunks([c1, c2])

    hits = tmp_store.query_by_tier_and_department(
        query_text="spend",
        user_tier=3,
        user_departments=["Finance"],
        top_k=10
    )
    assert len(hits) == 1
    assert hits[0].metadata["department"] == "finance"


def test_combined_tier_and_department_filter(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", department="engineering", clearance_tier=2)
    c2 = chunk_factory(chunk_id="c2", department="engineering", clearance_tier=3)
    c3 = chunk_factory(chunk_id="c3", department="hr", clearance_tier=1)
    c4 = chunk_factory(chunk_id="c4", department="finance", clearance_tier=1)
    tmp_store.upsert_chunks([c1, c2, c3, c4])

    hits = tmp_store.query_by_tier_and_department(
        query_text="sample",
        user_tier=2,
        user_departments=["engineering", "hr"],
        top_k=10
    )
    hit_ids = {h.chunk_id for h in hits}
    assert hit_ids == {"c1", "c3"}


def test_empty_department_list_returns_nothing(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", department="engineering", clearance_tier=1)
    tmp_store.upsert_chunks([c1])

    hits = tmp_store.query_by_tier_and_department(
        query_text="sample",
        user_tier=4,
        user_departments=[],
        top_k=10
    )
    assert hits == []


def test_none_departments_is_tier_only(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", department="engineering", clearance_tier=2)
    c2 = chunk_factory(chunk_id="c2", department="legal", clearance_tier=2)
    tmp_store.upsert_chunks([c1, c2])

    hits = tmp_store.query_by_tier_and_department(
        query_text="sample",
        user_tier=2,
        user_departments=None,
        top_k=10
    )
    assert len(hits) == 2


def test_invalid_tier_raises(tmp_store):
    with pytest.raises(ValueError):
        tmp_store.query_by_tier_and_department("q", user_tier=0, user_departments=None)
    with pytest.raises(ValueError):
        tmp_store.query_by_tier_and_department("q", user_tier=5, user_departments=None)
    with pytest.raises(ValueError):
        tmp_store.query_by_tier_and_department("q", user_tier="2", user_departments=None)


def test_query_on_empty_collection_returns_empty_list(tmp_store):
    hits = tmp_store.query_by_tier_and_department("any", user_tier=2, user_departments=None)
    assert hits == []


def test_top_k_larger_than_count_is_safe(tmp_store, chunk_factory):
    tmp_store.upsert_chunks([chunk_factory(chunk_id="c1", clearance_tier=1)])
    hits = tmp_store.query_by_tier_and_department("any", user_tier=2, user_departments=None, top_k=500)
    assert len(hits) == 1


def test_delete_document_removes_only_that_document(tmp_store, chunk_factory):
    c1 = chunk_factory(chunk_id="c1", document_id="docA", clearance_tier=1)
    c2 = chunk_factory(chunk_id="c2", document_id="docA", clearance_tier=1)
    c3 = chunk_factory(chunk_id="c3", document_id="docB", clearance_tier=1)
    tmp_store.upsert_chunks([c1, c2, c3])
    assert tmp_store.count() == 3

    removed = tmp_store.delete_document("docA")
    assert removed == 2
    assert tmp_store.count() == 1
    assert tmp_store.collection.get(ids=["c3"])["ids"] == ["c3"]

