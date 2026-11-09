"""ChromaDB Vector Store interface with HNSW cosine similarity and metadata filtering."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from .models import DocumentChunk
from ..config import settings


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector storage, cosine similarity indexing,
    batch upserts, and RBAC-filtered retrieval.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

    def get_or_create_collection(
        self,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        hnsw_space: str = settings.HNSW_SPACE,
    ):
        """
        Retrieve or create a collection explicitly indexed with HNSW cosine similarity.
        Valid spaces: 'cosine', 'l2', 'ip'.
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": hnsw_space}
        )

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        batch_size: int = settings.BATCH_SIZE,
    ) -> int:
        """
        Batch upsert DocumentChunks into ChromaDB with deterministic IDs and flat metadata.
        Idempotent: Re-indexing updates existing records without duplication.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata.to_chroma_dict() for c in batch]

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_upserted += len(batch)

        return total_upserted

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
    ) -> List[Dict[str, Any]]:
        """Execute a raw similarity query with optional ChromaDB metadata where filter."""
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )
        return self._format_results(results)

    def query_with_rbac(
        self,
        query_text: str,
        user_tier: int,
        user_departments: Optional[List[str]] = None,
        top_k: int = 5,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search enforcing strict Role-Based Access Control:
        1. Clearance tier restriction: document.clearance_tier <= user_tier
        2. Department restriction: document.department in user_departments (if specified)
        """
        where_clauses: List[Dict[str, Any]] = [
            {"clearance_tier": {"$lte": int(user_tier)}}
        ]

        # Department filtering if user is not unrestricted admin/executive
        if user_departments and "All" not in user_departments and "*" not in user_departments:
            if len(user_departments) == 1:
                where_clauses.append({"department": {"$eq": user_departments[0]}})
            else:
                where_clauses.append({"department": {"$in": user_departments}})

        # Combine into ChromaDB compound where filter if multiple clauses
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        else:
            where_filter = {"$and": where_clauses}

        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )
        return self._format_results(results)

    def count(self, collection_name: str = settings.DEFAULT_COLLECTION_NAME) -> int:
        """Return the number of vectors stored in the collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def reset_collection(self, collection_name: str = settings.DEFAULT_COLLECTION_NAME):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        return self.get_or_create_collection(collection_name)

    @staticmethod
    def _format_results(raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format raw ChromaDB query output into structured result dictionaries."""
        formatted = []
        if not raw_results or not raw_results.get("ids"):
            return formatted

        ids_list = raw_results["ids"][0]
        docs_list = raw_results["documents"][0] if raw_results.get("documents") else []
        meta_list = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
        dist_list = raw_results["distances"][0] if raw_results.get("distances") else []

        for idx, chunk_id in enumerate(ids_list):
            doc_text = docs_list[idx] if idx < len(docs_list) else ""
            metadata = meta_list[idx] if idx < len(meta_list) else {}
            distance = dist_list[idx] if idx < len(dist_list) else 0.0

            # Cosine distance in Chroma: similarity = 1 - cosine_distance
            similarity = round(max(0.0, 1.0 - distance), 4)

            formatted.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "metadata": metadata,
                "distance": round(distance, 4),
                "similarity_score": similarity,
            })

        return formatted
