"""Automated Vector ETL Pipeline orchestrator."""
import time
from pathlib import Path
from typing import List, Dict, Type, Optional
from .models import RawDocument, DocumentChunk, IngestionResult
from .extractors import (
    BaseExtractor,
    PDFExtractor,
    CSVExtractor,
    MarkdownExtractor,
    TextExtractor,
    JSONExtractor,
)
from .chunker import SemanticChunker
from .vector_store import VectorStoreManager
from ..config import settings


class VectorETLPipeline:
    """
    Orchestrates the entire ETL workflow:
    1. Multi-format document discovery and extraction
    2. Semantic chunking with recursive boundary splitting
    3. Metadata normalization and propagation
    4. Persistent ChromaDB cosine vector indexing
    """

    EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {
        ".pdf": PDFExtractor,
        ".csv": CSVExtractor,
        ".md": MarkdownExtractor,
        ".markdown": MarkdownExtractor,
        ".txt": TextExtractor,
        ".json": JSONExtractor,
    }

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        chunker: Optional[SemanticChunker] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.chunker = chunker or SemanticChunker()

    def run(
        self,
        data_dir: Path,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        reset_existing: bool = False,
    ) -> IngestionResult:
        """Execute end-to-end extraction, chunking, and vector ingestion."""
        start_time = time.perf_counter()
        data_dir = Path(data_dir)

        if not data_dir.exists() or not data_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {data_dir}")

        if reset_existing:
            self.vector_store.reset_collection(collection_name)

        # 1. Discover supported files
        files_to_process: List[Path] = []
        for ext in self.EXTRACTOR_REGISTRY.keys():
            files_to_process.extend(data_dir.glob(f"*{ext}"))
            files_to_process.extend(data_dir.glob(f"*{ext.upper()}"))

        files_to_process = sorted(list(set(files_to_process)))
        errors: List[str] = []
        all_raw_docs: List[RawDocument] = []

        # 2. Extract multi-format documents
        for file_path in files_to_process:
            ext = file_path.suffix.lower()
            extractor_cls = self.EXTRACTOR_REGISTRY.get(ext)
            if not extractor_cls:
                continue

            try:
                extractor = extractor_cls()
                docs = extractor.extract(file_path)
                all_raw_docs.extend(docs)
            except Exception as e:
                err_msg = f"Failed extracting '{file_path.name}': {str(e)}"
                errors.append(err_msg)

        # 3. Semantic Chunking
        all_chunks: List[DocumentChunk] = []
        try:
            all_chunks = self.chunker.chunk_documents(all_raw_docs)
        except Exception as e:
            errors.append(f"Chunking stage failure: {str(e)}")

        # 4. Batch Vector Ingestion
        total_upserted = 0
        if all_chunks:
            try:
                total_upserted = self.vector_store.upsert_chunks(
                    chunks=all_chunks,
                    collection_name=collection_name
                )
            except Exception as e:
                errors.append(f"Vector store upsert failure: {str(e)}")

        duration = time.perf_counter() - start_time
        throughput = (total_upserted / duration) if duration > 0 else 0.0

        return IngestionResult(
            total_files_discovered=len(files_to_process),
            total_documents_extracted=len(all_raw_docs),
            total_chunks_created=total_upserted,
            collection_name=collection_name,
            duration_seconds=round(duration, 4),
            throughput_chunks_per_sec=round(throughput, 2),
            errors=errors,
        )
