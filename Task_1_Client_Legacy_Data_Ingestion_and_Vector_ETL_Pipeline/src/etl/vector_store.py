"""VectorStore implementation with verified cosine distance and access control."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from .models import DocumentChunk, to_chroma_metadata
from ..config import COSINE_COLLECTION_KWARGS, TIERS


@dataclass
class QueryHit:
    """A search result with its similarity distance and metadata."""
    chunk_id: str
    text: str
    distance: float
    metadata: Dict[str, Any]


@dataclass
class UpsertTimings:
    """Breakdown of embedding computation time versus database write time."""
    embed_s: float
    write_s: float


class VectorStore:
    """
    ChromaDB wrapper enforcing:
    - Verified cosine distance metric
    - Batched upsert with separate embedding and write timings
    - Document deletion for idempotent re-ingestion
    - Access-controlled querying pre-filtered by tier and department
    """

    def __init__(
        self,
        path: Path,
        collection_name: str = "enterprise_docs",
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.path))
        if embedding_function is None:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_function = embedding_function

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            **COSINE_COLLECTION_KWARGS,
        )

    def effective_batch_size(self, requested: int) -> int:
        """Clamp requested batch size to ChromaDB's max supported batch size."""
        max_batch = self.client.get_max_batch_size()
        return min(requested, max_batch)

    def upsert_chunks(
        self, chunks: List[DocumentChunk], batch_size: int = 256
    ) -> UpsertTimings:
        """
        Embed each batch itself, timed separately, then upsert precomputed embeddings.
        """
        if not chunks:
            return UpsertTimings(embed_s=0.0, write_s=0.0)

        actual_batch = self.effective_batch_size(batch_size)
        total_embed_s = 0.0
        total_write_s = 0.0

        for i in range(0, len(chunks), actual_batch):
            batch = chunks[i : i + actual_batch]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [to_chroma_metadata(c) for c in batch]

            # 1. Timed embedding
            t0 = time.perf_counter()
            embeddings = self.embedding_function(texts)
            total_embed_s += time.perf_counter() - t0

            # 2. Timed write
            t1 = time.perf_counter()
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_write_s += time.perf_counter() - t1

        return UpsertTimings(embed_s=total_embed_s, write_s=total_write_s)

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks associated with a document_id.
        Returns the number of chunks removed.
        """
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        return 0

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            **COSINE_COLLECTION_KWARGS,
        )

    def query_by_tier_and_department(
        self,
        query_text: str,
        user_tier: int,
        user_departments: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[QueryHit]:
        """
        Execute semantic search with pre-filtering access control:
        - user_tier outside 1 to 4 raises ValueError
        - user_departments=None means tier-only filtering
        - user_departments=[] returns []
        - Department names lowercased before query
        """
        if not isinstance(user_tier, int) or isinstance(user_tier, bool) or user_tier not in TIERS:
            raise ValueError(f"Invalid clearance tier '{user_tier}'. Must be an integer in {TIERS}.")

        if user_departments is not None and len(user_departments) == 0:
            return []

        coll_count = self.count()
        if coll_count == 0:
            return []

        tier_filter: Dict[str, Any] = {"clearance_tier": {"$lte": user_tier}}

        if user_departments is not None:
            lowered_depts = [d.lower() for d in user_departments]
            where: Dict[str, Any] = {
                "$and": [
                    tier_filter,
                    {"department": {"$in": lowered_depts}},
                ]
            }
        else:
            where = tier_filter

        n_results = min(top_k, coll_count)
        raw = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

        hits: List[QueryHit] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return hits

        ids = raw["ids"][0]
        docs = raw["documents"][0] if raw.get("documents") else []
        dists = raw["distances"][0] if raw.get("distances") else []
        metas = raw["metadatas"][0] if raw.get("metadatas") else []

        for idx, cid in enumerate(ids):
            hits.append(
                QueryHit(
                    chunk_id=cid,
                    text=docs[idx] if idx < len(docs) else "",
                    distance=dists[idx] if idx < len(dists) else 0.0,
                    metadata=metas[idx] if idx < len(metas) else {},
                )
            )

        return hits
