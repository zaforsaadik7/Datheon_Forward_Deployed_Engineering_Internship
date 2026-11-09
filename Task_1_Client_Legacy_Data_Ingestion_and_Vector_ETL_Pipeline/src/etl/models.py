"""Pydantic data models for ETL extraction, chunking, and benchmarking."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Normalized metadata attached to every document and chunk."""
    department: str = Field(..., description="Corporate department, e.g. Engineering, Finance, Legal")
    clearance_tier: int = Field(..., ge=1, le=4, description="Clearance level: 1=Public, 2=Internal, 3=Confidential, 4=Secret")
    document_date: str = Field(..., description="ISO 8601 date, e.g. '2026-10-15'")
    source_file: str = Field(..., description="Filename of the source document")
    file_type: str = Field(..., description="File format extension: pdf, csv, md, txt, json")
    page_number: Optional[int] = Field(None, description="Page number for multi-page documents (PDFs)")
    row_id: Optional[int] = Field(None, description="Row index for tabular data (CSVs)")
    section_title: Optional[str] = Field(None, description="Markdown/document section heading")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Custom domain-specific attributes")

    def to_chroma_dict(self) -> Dict[str, Any]:
        """Convert metadata to flat dictionary compatible with ChromaDB constraints."""
        flat = {
            "department": self.department,
            "clearance_tier": int(self.clearance_tier),
            "document_date": self.document_date,
            "source_file": self.source_file,
            "file_type": self.file_type,
        }
        if self.page_number is not None:
            flat["page_number"] = int(self.page_number)
        if self.row_id is not None:
            flat["row_id"] = int(self.row_id)
        if self.section_title is not None:
            flat["section_title"] = str(self.section_title)
        for k, v in self.extra.items():
            if isinstance(v, (str, int, float, bool)):
                flat[f"extra_{k}"] = v
            else:
                flat[f"extra_{k}"] = str(v)
        return flat


class RawDocument(BaseModel):
    """Intermediate extracted representation of a raw client document."""
    document_id: str
    content: str
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    """A semantically bounded chunk with complete operational metadata."""
    chunk_id: str
    document_id: str
    text: str
    metadata: DocumentMetadata
    token_count: int
    chunk_index: int
    total_chunks: int = 1


class IngestionResult(BaseModel):
    """Summary of an ETL pipeline execution."""
    total_files_discovered: int
    total_documents_extracted: int
    total_chunks_created: int
    collection_name: str
    duration_seconds: float
    throughput_chunks_per_sec: float
    errors: List[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """Performance metrics for batch ETL ingestion and vector queries."""
    num_documents: int
    num_chunks: int
    total_tokens: int
    extraction_time_ms: float
    chunking_time_ms: float
    indexing_time_ms: float
    total_time_ms: float
    throughput_docs_per_sec: float
    throughput_chunks_per_sec: float
    query_latency_ms: Dict[str, float]  # mean, p50, p95, p99

