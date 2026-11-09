"""Semantic and recursive boundary chunker with metadata propagation."""
import hashlib
import re
from typing import List
from .models import RawDocument, DocumentChunk, DocumentMetadata
from ..config import settings


class SemanticChunker:
    """
    Splits documents into overlapping chunks using natural semantic boundaries
    (paragraphs, sentences) while guaranteeing metadata propagation and deterministic IDs.
    """

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: RawDocument) -> List[DocumentChunk]:
        """Split a single RawDocument into multiple DocumentChunks."""
        text = document.content.strip()
        if not text:
            return []

        # If document is already small enough, treat as a single chunk
        if len(text) <= self.chunk_size:
            chunk_id = self._generate_chunk_id(document.document_id, 0, text)
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=text,
                    metadata=document.metadata,
                    token_count=max(1, len(text) // 4),
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        # Recursive splitting into paragraphs, then sentences
        raw_chunks = self._recursive_split(text)
        merged_chunks = self._sliding_window_merge(raw_chunks)

        total = len(merged_chunks)
        chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(merged_chunks):
            chunk_id = self._generate_chunk_id(document.document_id, idx, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    metadata=document.metadata,
                    token_count=max(1, len(chunk_text) // 4),
                    chunk_index=idx,
                    total_chunks=total,
                )
            )

        return chunks

    def chunk_documents(self, documents: List[RawDocument]) -> List[DocumentChunk]:
        """Process a list of RawDocuments into a consolidated list of DocumentChunks."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _recursive_split(self, text: str) -> List[str]:
        """Split text hierarchically by paragraphs, then sentences."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for p in paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                # Split large paragraph by sentence
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    if len(s_clean) <= self.chunk_size:
                        units.append(s_clean)
                    else:
                        # Split very long sentences by words
                        words = s_clean.split(" ")
                        cur_w = []
                        cur_len = 0
                        for w in words:
                            if cur_len + len(w) + 1 > self.chunk_size and cur_w:
                                units.append(" ".join(cur_w))
                                cur_w = [w]
                                cur_len = len(w)
                            else:
                                cur_w.append(w)
                                cur_len += len(w) + 1
                        if cur_w:
                            units.append(" ".join(cur_w))
        return units

    def _sliding_window_merge(self, units: List[str]) -> List[str]:
        """Merge textual units into target chunk sizes with overlap."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for unit in units:
            unit_len = len(unit)
            if current_length + unit_len + 1 <= self.chunk_size:
                current_chunk.append(unit)
                current_length += unit_len + 1
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)

                    # Compute overlap from trailing units
                    overlap_units: List[str] = []
                    overlap_len = 0
                    for u in reversed(current_chunk):
                        if overlap_len + len(u) + 1 <= self.chunk_overlap:
                            overlap_units.insert(0, u)
                            overlap_len += len(u) + 1
                        else:
                            break
                    current_chunk = overlap_units + [unit]
                    current_length = sum(len(u) + 1 for u in current_chunk)
                else:
                    chunks.append(unit)
                    current_chunk = []
                    current_length = 0

        if current_chunk:
            final_text = " ".join(current_chunk).strip()
            if len(final_text) >= self.min_chunk_size:
                chunks.append(final_text)

        return chunks if chunks else units

    @staticmethod
    def _generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
        """Create deterministic, collision-resistant hash ID."""
        hasher = hashlib.sha256()
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(str(chunk_index).encode("utf-8"))
        hasher.update(text[:64].encode("utf-8"))
        return f"{doc_id}_c{chunk_index}_{hasher.hexdigest()[:8]}"
