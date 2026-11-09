"""Abstract base extractor and exception hierarchy."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List
from ..models import Segment


class ExtractionError(Exception):
    """Raised when text extraction fails from a document."""
    pass


class UnsupportedFileType(ExtractionError):
    """Raised when an unrecognized file extension is encountered."""
    pass


class BaseExtractor(ABC):
    """Abstract base class for all file format extractors."""
    file_type: ClassVar[str]

    @abstractmethod
    def extract(self, path: Path) -> List[Segment]:
        """Extract text segments with citation data from the file."""
        pass
