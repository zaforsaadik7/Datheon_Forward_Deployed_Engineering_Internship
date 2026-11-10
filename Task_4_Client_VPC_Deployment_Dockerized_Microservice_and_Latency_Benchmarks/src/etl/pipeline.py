"""Directory ingestion pipeline with per-file failure isolation and timings."""
import time
from pathlib import Path
from typing import List, Optional

from .manifest import load_manifest, resolve_metadata
from .extractors import get_extractor, UnsupportedFileType
from .chunker import build_chunks
from .vector_store import VectorStore
from .models import FailedDocument, IngestReport


def discover_files(data_dir: Path) -> List[Path]:
    """
    Recursively discover files in data_dir, skipping manifest.yaml and hidden files.
    Returns a sorted list of Paths. Unsupported files are kept to report at discover stage.
    """
    data_dir = Path(data_dir)
    discovered: List[Path] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip manifest.yaml
        if path.name.lower() == "manifest.yaml":
            continue
        # Skip hidden files and hidden directories
        parts = path.relative_to(data_dir).parts
        if any(part.startswith(".") for part in parts):
            continue
        discovered.append(path)

    return discovered


def ingest_directory(
    data_dir: Path,
    store: VectorStore,
    *,
    manifest_path: Optional[Path] = None,
    chunk_size: int = 900,
    overlap: int = 120,
    batch_size: int = 256,
) -> IngestReport:
    """
    Ingest all documents from data_dir into store:
    - Resolves metadata via manifest or fail-closed filename fallback
    - Extracts format-specific segments
    - Builds semantic chunks
    - Deletes old document chunks before indexing new ones
    - Isolates per-file failures without aborting the batch
    - Records stage-by-stage timings
    """
    start_time = time.perf_counter()
    data_dir = Path(data_dir)

    # 1. Load manifest if present
    if manifest_path is None:
        default_manifest = data_dir / "manifest.yaml"
        if default_manifest.exists():
            manifest = load_manifest(default_manifest)
        else:
            manifest = {}
    else:
        manifest = load_manifest(manifest_path)

    files = discover_files(data_dir)

    documents_ok = 0
    documents_failed: List[FailedDocument] = []
    chunks_written = 0
    bytes_read = 0

    extract_s = 0.0
    chunk_s = 0.0
    embed_s = 0.0
    write_s = 0.0

    for file_path in files:
        rel_posix = file_path.relative_to(data_dir).as_posix()
        try:
            mtime = file_path.stat().st_mtime
            file_bytes = file_path.stat().st_size
        except Exception as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="discover", error=str(e))
            )
            continue

        # Stage 1: Extractor discovery
        try:
            extractor = get_extractor(file_path)
        except UnsupportedFileType as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="discover", error=str(e))
            )
            continue

        # Stage 2: Metadata resolution
        try:
            entry = resolve_metadata(rel_posix, manifest, mtime)
        except Exception as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="metadata", error=str(e))
            )
            continue

        # Stage 3: Extraction
        t0 = time.perf_counter()
        try:
            segments = extractor.extract(file_path)
        except Exception as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="extract", error=str(e))
            )
            continue
        extract_s += time.perf_counter() - t0

        # Stage 4: Semantic Chunking
        t1 = time.perf_counter()
        try:
            chunks = build_chunks(
                segments,
                document_id=rel_posix,
                source_file=file_path.name,
                file_type=extractor.file_type,
                entry=entry,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except Exception as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="extract", error=str(e))
            )
            continue
        chunk_s += time.perf_counter() - t1

        # Stage 5: Delete old chunks and upsert new chunks
        try:
            store.delete_document(rel_posix)
            timings = store.upsert_chunks(chunks, batch_size=batch_size)
            embed_s += timings.embed_s
            write_s += timings.write_s
            chunks_written += len(chunks)
            bytes_read += file_bytes
            documents_ok += 1
        except Exception as e:
            documents_failed.append(
                FailedDocument(file=rel_posix, stage="index", error=str(e))
            )
            continue

    total_s = time.perf_counter() - start_time

    return IngestReport(
        documents_ok=documents_ok,
        documents_failed=documents_failed,
        chunks_written=chunks_written,
        bytes_read=bytes_read,
        extract_s=extract_s,
        chunk_s=chunk_s,
        embed_s=embed_s,
        write_s=write_s,
        total_s=total_s,
    )

