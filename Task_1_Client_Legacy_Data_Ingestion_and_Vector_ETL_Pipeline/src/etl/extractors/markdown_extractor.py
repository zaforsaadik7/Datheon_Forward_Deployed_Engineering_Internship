"""Markdown extractor with breadcrumb tracking and code fence protection."""
import re
from pathlib import Path
from typing import ClassVar, List, Tuple, Optional
from .base import BaseExtractor, ExtractionError
from .text_extractor import read_text_file
from ..models import Segment

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownExtractor(BaseExtractor):
    """Extracts Markdown sections with hierarchical breadcrumb titles."""
    file_type: ClassVar[str] = "markdown"

    def extract(self, path: Path) -> List[Segment]:
        raw_text = read_text_file(path)
        if not raw_text:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")

        lines = raw_text.split("\n")
        in_code_fence = False

        sections: List[Tuple[Optional[str], List[str]]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []
        header_stack: List[Tuple[int, str]] = []

        for line in lines:
            stripped = line.strip()
            # Toggle code fence
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence
                current_lines.append(line)
                continue

            if in_code_fence:
                current_lines.append(line)
                continue

            match = HEADER_RE.match(line)
            if match:
                # Flush previous section if it has non-empty body
                body_text = "\n".join(current_lines).strip()
                if body_text:
                    sections.append((current_title, current_lines))
                current_lines = []

                level = len(match.group(1))
                title = match.group(2).strip()

                # Pop headings of equal or deeper level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()

                header_stack.append((level, title))
                current_title = " > ".join(t for _, t in header_stack)
            else:
                current_lines.append(line)

        # Flush final section
        body_text = "\n".join(current_lines).strip()
        if body_text:
            sections.append((current_title, current_lines))

        segments: List[Segment] = []
        for sec_title, sec_lines in sections:
            body = "\n".join(sec_lines).strip()
            if not body:
                continue

            if sec_title:
                full_text = f"{sec_title}\n\n{body}"
            else:
                full_text = body

            segments.append(Segment(text=full_text, section_title=sec_title))

        if not segments:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")

        return segments
