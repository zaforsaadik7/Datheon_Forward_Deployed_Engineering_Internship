"""Manifest loading and fail-closed metadata resolution."""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict
import yaml

from .models import ManifestEntry
from ..config import DEPARTMENTS, DEFAULT_TIER

logger = logging.getLogger(__name__)


class MetadataError(Exception):
    """Raised when metadata resolution fails or violates validation rules."""
    pass


# Filename fallback regex: ^(?P<dept>[a-z]+)_.*?(?:_tier(?P<tier>[1-4]))?\.\w+$
FILENAME_PATTERN = re.compile(
    r"^(?P<dept>[a-zA-Z]+)_.*?(?:_tier(?P<tier>[1-4]))?\.[a-zA-Z0-9]+$"
)


def load_manifest(path: Path) -> Dict[str, ManifestEntry]:
    """
    Load manifest.yaml and return a dictionary of ManifestEntry keyed by POSIX relative path.
    Wraps validation and YAML parsing errors in MetadataError.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise MetadataError(f"Failed to parse manifest at '{path}': {e}")

    docs_list = data.get("documents", [])
    if not isinstance(docs_list, list):
        raise MetadataError(f"Malformed manifest '{path}': 'documents' must be a list.")

    manifest: Dict[str, ManifestEntry] = {}
    for doc in docs_list:
        file_name = doc.get("file")
        if not file_name:
            raise MetadataError(f"Manifest entry missing 'file' attribute: {doc}")

        try:
            entry = ManifestEntry(
                department=doc["department"],
                clearance_tier=doc["clearance_tier"],
                document_date=str(doc["document_date"]),
                metadata_source="manifest",
            )
            posix_path = Path(file_name).as_posix()
            manifest[posix_path] = entry
        except Exception as e:
            raise MetadataError(f"Invalid metadata for file '{file_name}': {e}")

    return manifest


def resolve_metadata(
    rel_path: str,
    manifest: Dict[str, ManifestEntry],
    mtime: float,
) -> ManifestEntry:
    """
    Resolve metadata for a file path.
    1. Check manifest (exact relative path only).
    2. Fallback to filename convention with fail-closed security:
       - Missing tier defaults to DEFAULT_TIER (4).
       - Unrecognized department raises MetadataError.
    """
    posix_path = Path(rel_path).as_posix()
    basename = Path(rel_path).name

    # 1. Manifest lookup by full relative path only. Matching on the bare
    # filename would let a file in another folder inherit someone else's tier.
    if posix_path in manifest:
        return manifest[posix_path]

    # 2. Filename fallback regex
    match = FILENAME_PATTERN.match(basename)
    if match:
        raw_dept = match.group("dept").lower()
        if raw_dept in DEPARTMENTS:
            tier_str = match.group("tier")
            if tier_str:
                tier = int(tier_str)
            else:
                tier = DEFAULT_TIER
                logger.warning(
                    f"File '{rel_path}' lacks explicit clearance tier; failing closed to Tier {DEFAULT_TIER}."
                )

            doc_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            logger.warning(
                f"Resolved metadata for '{rel_path}' from filename fallback: dept={raw_dept}, tier={tier}."
            )
            return ManifestEntry(
                department=raw_dept,
                clearance_tier=tier,
                document_date=doc_date,
                metadata_source="filename",
            )

    raise MetadataError(
        f"Could not resolve metadata for '{rel_path}'. "
        f"File must be listed in manifest.yaml or match '<department>_*[_tierN].ext' with a valid department."
    )

