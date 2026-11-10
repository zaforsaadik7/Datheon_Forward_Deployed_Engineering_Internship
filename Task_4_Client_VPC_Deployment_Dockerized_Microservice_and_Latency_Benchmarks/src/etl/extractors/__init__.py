"""Extractors package and central registry."""
from pathlib import Path
from typing import Dict, Type
from .base import BaseExtractor, ExtractionError, UnsupportedFileType
from .text_extractor import TextExtractor, read_text_file
from .json_extractor import JsonExtractor
from .markdown_extractor import MarkdownExtractor
from .csv_extractor import CsvExtractor
from .pdf_extractor import PdfExtractor

_EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {
    ".pdf": PdfExtractor,
    ".csv": CsvExtractor,
    ".md": MarkdownExtractor,
    ".markdown": MarkdownExtractor,
    ".json": JsonExtractor,
    ".txt": TextExtractor,
}


def get_extractor(path: Path) -> BaseExtractor:
    """
    Return an instantiated extractor for the file suffix (case-insensitive).
    Raises UnsupportedFileType if suffix is not recognized.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    extractor_cls = _EXTRACTOR_REGISTRY.get(suffix)
    if not extractor_cls:
        raise UnsupportedFileType(
            f"Unsupported file type '{suffix}' for file '{path.name}'. "
            f"Supported extensions: {sorted(list(_EXTRACTOR_REGISTRY.keys()))}"
        )
    return extractor_cls()


__all__ = [
    "BaseExtractor",
    "ExtractionError",
    "UnsupportedFileType",
    "TextExtractor",
    "read_text_file",
    "JsonExtractor",
    "MarkdownExtractor",
    "CsvExtractor",
    "PdfExtractor",
    "get_extractor",
]
