"""Recursive semantic text chunker and chunk assembly."""
import hashlib
from typing import List, Optional
from .models import Segment, ManifestEntry, DocumentChunk

SEPARATORS = ["\n\n", "\n", ". ", " "]


def estimate_tokens(text: str) -> int:
    """Token count heuristic: max(1, len(text) // 4)."""
    return max(1, len(text) // 4)


def _split_into_pieces(text: str, sep_idx: int, max_piece_len: int) -> List[str]:
    """Recursively split text using the hierarchy of separators."""
    if len(text) <= max_piece_len:
        return [text]

    if sep_idx >= len(SEPARATORS):
        # Hard cut with stride max_piece_len
        return [text[i : i + max_piece_len] for i in range(0, len(text), max_piece_len)]

    sep = SEPARATORS[sep_idx]
    if sep not in text:
        return _split_into_pieces(text, sep_idx + 1, max_piece_len)

    raw_parts = text.split(sep)
    pieces: List[str] = []

    for i, part in enumerate(raw_parts):
        # Re-attach separator except for the last part
        p_with_sep = part + sep if i < len(raw_parts) - 1 else part
        if not p_with_sep:
            continue
        if len(p_with_sep) <= max_piece_len:
            pieces.append(p_with_sep)
        else:
            pieces.extend(_split_into_pieces(p_with_sep, sep_idx + 1, max_piece_len))

    return pieces


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Split text recursively on paragraphs, lines, sentences, or words.
    Guarantees that every chunk is <= chunk_size characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    if len(stripped) <= chunk_size:
        return [stripped]

    # Target window budget for packing pieces
    budget = chunk_size - overlap
    pieces = _split_into_pieces(stripped, 0, budget)

    # Greedily pack pieces into windows
    windows: List[str] = []
    cur_pieces: List[str] = []
    cur_len = 0

    for piece in pieces:
        p_len = len(piece)
        if cur_len + p_len <= budget:
            cur_pieces.append(piece)
            cur_len += p_len
        else:
            if cur_pieces:
                windows.append("".join(cur_pieces))
            cur_pieces = [piece]
            cur_len = p_len

    if cur_pieces:
        windows.append("".join(cur_pieces))

    # Overlap prefixing
    chunks: List[str] = []
    for idx, window in enumerate(windows):
        if idx == 0:
            chunks.append(window.strip())
        else:
            prev_window = windows[idx - 1]
            overlap_text = prev_window[-overlap:]
            # Trim forward to whitespace boundary if possible
            first_space = overlap_text.find(" ")
            if first_space != -1 and first_space < len(overlap_text) - 1:
                overlap_prefix = overlap_text[first_space + 1 :]
            else:
                overlap_prefix = overlap_text

            combined = (overlap_prefix + " " + window).strip()
            # Safety clamp: if combined exceeds chunk_size, truncate to chunk_size
            if len(combined) > chunk_size:
                combined = combined[:chunk_size]
            chunks.append(combined)

    return chunks


def build_chunks(
    segments: List[Segment],
    *,
    document_id: str,
    source_file: str,
    file_type: str,
    entry: ManifestEntry,
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Assemble DocumentChunk objects from extracted Segments.
    Segments are never merged across each other so that citations (page, row, sec) remain exact.
    """
    all_chunks: List[DocumentChunk] = []
    chunk_idx = 0

    for seg in segments:
        text_chunks = chunk_text(seg.text, chunk_size=chunk_size, overlap=overlap)
        for t in text_chunks:
            chunk_id = hashlib.sha256(f"{document_id}|{chunk_idx}".encode("utf-8")).hexdigest()
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=t,
                    department=entry.department,
                    clearance_tier=entry.clearance_tier,
                    document_date=entry.document_date,
                    source_file=source_file,
                    file_type=file_type,
                    chunk_index=chunk_idx,
                    page_number=seg.page_number,
                    row_id=seg.row_id,
                    section_title=seg.section_title,
                    token_count=estimate_tokens(t),
                )
            )
            chunk_idx += 1

    return all_chunks
