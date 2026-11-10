"""Pluggable Generator protocol, overlap-deduplicating ExtractiveSynthesizer, and citation tag renderer."""
import re
from typing import Dict, List, Protocol, Sequence, Set, Tuple
from .models import Citation, DraftAnswer, DraftStatement, RankedChunk
from .rbac import build_citation_from_metadata
from .reranker import _tokenize

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
HEADING_RE = re.compile(
    r"^(?:#+\s+.*|section\s+\d+[:.]\s*[^.]{1,70}|\d+\.\s+[A-Z][A-Za-z\s&,]{1,65}|strictly\s+confidential.*)$",
    re.IGNORECASE,
)


class Generator(Protocol):
    """Pluggable answer generator interface producing marker-indexed DraftAnswers."""

    def generate(self, query: str, ranked_chunks: Sequence[RankedChunk]) -> DraftAnswer:
        ...


def _split_chunk_units(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []

    # Keep pipe-delimited CSV/JSON record rows intact as a single unit
    if " | " in stripped and stripped.count("|") >= 2:
        return [re.sub(r"\s+", " ", stripped)]

    raw_parts = SENTENCE_SPLIT_RE.split(stripped)
    units: List[str] = []
    for part in raw_parts:
        clean = re.sub(r"\s+", " ", part).strip()
        if not clean:
            continue
        if HEADING_RE.match(clean) and "." not in clean[3:]:
            continue
        if len(clean) < 15:
            continue
        units.append(clean)

    if not units:
        units.append(re.sub(r"\s+", " ", stripped))
    return units


def _is_near_duplicate(tokens_a: Set[str], seen_token_sets: List[Set[str]]) -> bool:
    if not tokens_a:
        return True
    for tokens_b in seen_token_sets:
        if not tokens_b:
            continue
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union > 0 and (inter / float(union)) >= 0.85:
            return True
    return False


class ExtractiveSynthesizer:
    """
    Deterministic extractive answer synthesizer.
    Extracts and deduplicates the highest-relevance sentences or structured records from
    authorized RankedChunks and tags each statement with its 1-based chunk index.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: str, ranked_chunks: Sequence[RankedChunk]) -> DraftAnswer:
        if not ranked_chunks or self.max_sentences <= 0:
            return DraftAnswer(statements=())

        q_tokens = set(_tokenize(query))
        candidates: List[Tuple[float, int, int, str, Set[str]]] = []

        for chunk_zero_idx, chunk in enumerate(ranked_chunks):
            chunk_one_idx = chunk_zero_idx + 1
            units = _split_chunk_units(chunk.text)
            for unit_pos, unit in enumerate(units):
                u_tokens = set(_tokenize(unit))
                if not u_tokens:
                    continue
                if q_tokens:
                    overlap = len(q_tokens & u_tokens) / float(len(q_tokens))
                else:
                    overlap = 0.0

                # Reward units carrying concrete figures or technical identifiers
                has_digit_or_code = any(ch.isdigit() or ch in "$%" for ch in unit)
                specificity_bonus = 0.12 if (has_digit_or_code and overlap > 0.0) else 0.0
                score = (0.62 * overlap) + (0.38 * chunk.rerank_score) + specificity_bonus
                candidates.append((score, chunk_one_idx, unit_pos, unit, u_tokens))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        selected: List[Tuple[int, int, str]] = []
        seen_sets: List[Set[str]] = []

        for score, chunk_one_idx, unit_pos, unit_text, u_tokens in candidates:
            if _is_near_duplicate(u_tokens, seen_sets):
                continue
            seen_sets.append(u_tokens)
            selected.append((chunk_one_idx, unit_pos, unit_text))
            if len(selected) >= self.max_sentences:
                break

        statements = tuple(
            DraftStatement(text=text, cited_indices=(chunk_idx,))
            for chunk_idx, _, text in selected
        )
        return DraftAnswer(statements=statements)


def render_cited_answer(
    draft: DraftAnswer,
    ranked_chunks: Sequence[RankedChunk],
) -> Tuple[str, Tuple[Citation, ...]]:
    """
    Render a verified DraftAnswer into prose with canonical locator tags
    ('[source_file, p. N]', '[source_file, row N]', '[source_file, §section]',
    or '[source_file, chunk N]') and return the deduplicated Citations.
    """
    if not draft.statements or not ranked_chunks:
        return "", ()

    citation_by_idx: Dict[int, Citation] = {}
    ordered_citations: List[Citation] = []
    rendered_parts: List[str] = []

    for stmt in draft.statements:
        tags: List[str] = []
        for idx in stmt.cited_indices:
            if 1 <= idx <= len(ranked_chunks):
                if idx not in citation_by_idx:
                    chunk = ranked_chunks[idx - 1]
                    cit = build_citation_from_metadata(
                        citation_index=idx,
                        chunk_id=chunk.chunk_id,
                        metadata=chunk.metadata,
                        rerank_score=chunk.rerank_score,
                    )
                    citation_by_idx[idx] = cit
                    ordered_citations.append(cit)
                tag = citation_by_idx[idx].format_tag()
                if tag not in tags:
                    tags.append(tag)

        clean_stmt = re.sub(r"\[\d+\]", "", stmt.text).strip()
        if tags:
            rendered_parts.append(f"{clean_stmt} {' '.join(tags)}")
        else:
            rendered_parts.append(clean_stmt)

    return " ".join(rendered_parts), tuple(ordered_citations)
