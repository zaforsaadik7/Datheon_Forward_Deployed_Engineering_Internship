"""Plain text extractor with robust encoding detection."""
from pathlib import Path
from typing import ClassVar, List
from .base import BaseExtractor, ExtractionError
from ..models import Segment


def read_text_file(path: Path) -> str:
    """
    Read text file trying utf-8-sig first, falling back to cp1252.
    Normalizes CRLF and CR to LF, and rstrips each line.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()

    try:
        content = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content = raw_bytes.decode("cp1252")
        except UnicodeDecodeError as e:
            raise ExtractionError(f"Failed to decode text file '{path.name}': {e}")

    # Normalize newlines
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    # rstrip each line
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


class TextExtractor(BaseExtractor):
    """Extracts segments from plaintext files (.txt, .log)."""
    file_type: ClassVar[str] = "txt"

    def extract(self, path: Path) -> List[Segment]:
        text = read_text_file(path)
        if not text:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")
        return [Segment(text=text)]
