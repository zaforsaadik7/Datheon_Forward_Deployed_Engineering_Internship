"""Pydantic data models for vector ETL segments, chunks, and metadata."""
from datetime import date
from typing import Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, field_validator
from ..config import DEPARTMENTS, TIERS


class Segment(BaseModel):
    """A citation-preserving unit of extracted text from a source document."""
    text: str
    page_number: Optional[int] = None
    row_id: Optional[int] = None
    section_title: Optional[str] = None


class ManifestEntry(BaseModel):
    """Metadata resolved for a document from manifest.yaml or filename fallback."""
    department: str
    clearance_tier: int
    document_date: str
    metadata_source: Literal["manifest", "filename"]

    @field_validator("department")
    @classmethod
    def validate_dept(cls, v: str) -> str:
        dept = v.lower()
        if dept not in DEPARTMENTS:
            raise ValueError(f"Unknown department '{v}'. Must be one of {sorted(list(DEPARTMENTS))}")
        return dept

    @field_validator("clearance_tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        if v not in TIERS:
            raise ValueError(f"Invalid clearance tier {v}. Must be in {TIERS}")
        return v

    @field_validator("document_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except Exception:
            raise ValueError(f"Invalid ISO date '{v}'. Expected YYYY-MM-DD format.")
        return v


class DocumentChunk(BaseModel):
    """A semantically bounded chunk with complete operational metadata."""
    chunk_id: str
    document_id: str
    text: str
    department: str
    clearance_tier: int
    document_date: str
    source_file: str
    file_type: str
    chunk_index: int
    page_number: Optional[int] = None
    row_id: Optional[int] = None
    section_title: Optional[str] = None
    token_count: Optional[int] = None

    @field_validator("department")
    @classmethod
    def validate_dept(cls, v: str) -> str:
        dept = v.lower()
        if dept not in DEPARTMENTS:
            raise ValueError(f"Unknown department '{v}'. Must be one of {sorted(list(DEPARTMENTS))}")
        return dept

    @field_validator("clearance_tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        if v not in TIERS:
            raise ValueError(f"Invalid clearance tier {v}. Must be in {TIERS}")
        return v

    @field_validator("document_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except Exception:
            raise ValueError(f"Invalid ISO date '{v}'. Expected YYYY-MM-DD format.")
        return v


def to_chroma_metadata(chunk: DocumentChunk) -> Dict[str, Union[str, int, float, bool]]:
    """
    Convert a DocumentChunk to ChromaDB-safe flat metadata:
    - Excludes 'text'
    - Omits every key whose value is None (ChromaDB rejects None values)
    """
    raw_dict = chunk.model_dump()
    raw_dict.pop("text", None)

    chroma_meta: Dict[str, Union[str, int, float, bool]] = {}
    for k, v in raw_dict.items():
        if v is not None:
            if isinstance(v, (str, int, float, bool)):
                chroma_meta[k] = v
            else:
                chroma_meta[k] = str(v)

    return chroma_meta
