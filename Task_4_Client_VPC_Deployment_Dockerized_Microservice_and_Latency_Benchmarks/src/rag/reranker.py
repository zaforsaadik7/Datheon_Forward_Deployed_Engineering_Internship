"""Cross-encoder neural reranker (ONNX Runtime) and candidate-scoped lexical BM25/phrase reranker."""
import math
import re
from typing import Dict, List, Protocol, Sequence, Set
from ..etl.vector_store import QueryHit
from .models import RankedChunk

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it", "its", "many",
    "much", "must", "of", "on", "or", "our", "please", "should", "tell", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-_./][a-z0-9]+)*", re.IGNORECASE)


def _stem_token(tok: str) -> str:
    if len(tok) >= 5 and tok.isalpha() and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _tokenize(text: str) -> List[str]:
    raw = [_stem_token(t.lower()) for t in TOKEN_RE.findall(text)]
    return [t for t in raw if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _is_specific_identifier(token: str) -> bool:
    return (any(ch in token for ch in "-_./") and len(token) >= 4) or (
        any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
    )


class Reranker(Protocol):
    """Pluggable second-stage reranker protocol operating exclusively on authorized candidates."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[QueryHit],
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> List[RankedChunk]:
        ...


class LexicalReranker:
    """
    Deterministic, zero-IO lexical + cosine reranker.
    Computes BM25 IDF statistics strictly over the authorized candidate list passed to rerank()
    so restricted corpus statistics can never leak to lower-clearance principals.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _score_candidates(self, query: str, candidates: Sequence[QueryHit]) -> List[float]:
        if not candidates:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return [0.0 for _ in candidates]

        q_unique = list(dict.fromkeys(q_tokens))
        q_specific_ids = [qt for qt in q_unique if _is_specific_identifier(qt)]
        q_bigrams = {
            f"{q_tokens[i]} {q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)
        }

        doc_tokens_list: List[List[str]] = [_tokenize(c.text) for c in candidates]
        n_docs = len(candidates)
        avgdl = max(1.0, sum(len(dt) for dt in doc_tokens_list) / float(n_docs))

        # Candidate-scoped document frequency (strictly over `candidates`)
        df: Dict[str, int] = {}
        for qt in q_unique:
            df[qt] = sum(1 for dt in doc_tokens_list if qt in dt)

        raw_bm25: List[float] = []
        coverages: List[float] = []
        phrase_bonuses: List[float] = []
        matched_counts: List[int] = []
        matched_specific_counts: List[int] = []

        for idx, hit in enumerate(candidates):
            dt = doc_tokens_list[idx]
            dl = len(dt)
            tf_map: Dict[str, int] = {}
            for tok in dt:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            bm25_val = 0.0
            matched_q = 0
            matched_spec = 0
            code_id_bonus = 0.0

            for qt in q_unique:
                tf = tf_map.get(qt, 0)
                if tf > 0:
                    matched_q += 1
                    if _is_specific_identifier(qt):
                        matched_spec += 1
                        code_id_bonus += 0.22
                    idf = math.log(1.0 + (n_docs - df[qt] + 0.5) / (df[qt] + 0.5))
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    bm25_val += idf * ((tf * (self.k1 + 1.0)) / max(1e-9, denom))

            raw_bm25.append(bm25_val)
            cov = matched_q / float(len(q_unique))
            coverages.append(cov)
            matched_counts.append(matched_q)
            matched_specific_counts.append(matched_spec)

            doc_bigrams = {f"{dt[i]} {dt[i + 1]}" for i in range(len(dt) - 1)}
            bg_hits = len(q_bigrams & doc_bigrams)
            p_bonus = min(0.45, bg_hits * 0.18 + code_id_bonus)
            phrase_bonuses.append(p_bonus)

        max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
        scores: List[float] = []

        for idx, hit in enumerate(candidates):
            vec_sim = max(0.0, min(1.0, 1.0 - float(hit.distance)))
            cov = coverages[idx]
            matched_q = matched_counts[idx]
            matched_spec = matched_specific_counts[idx]

            # If the query specifies concrete identifiers (like FIN-Q3-002 or vpc-prod-compute)
            # and the candidate matches NONE of them, or if overall query token coverage is
            # below threshold, cap below min_score so off-topic chunks are rejected.
            if q_specific_ids and matched_spec == 0 and cov < 0.50:
                scores.append(round(0.15 * vec_sim, 6))
                continue
            if cov == 0.0 or (len(q_unique) >= 3 and matched_q <= 1 and matched_spec == 0):
                scores.append(round(0.15 * vec_sim, 6))
                continue

            bm25_norm = (raw_bm25[idx] / max_bm25) if max_bm25 > 0.0 else cov
            bm25_grounded = bm25_norm * min(1.0, cov * 1.5)
            lexical_strength = min(
                1.0,
                0.50 * max(bm25_grounded, cov) + 0.50 * min(1.0, cov + phrase_bonuses[idx]),
            )
            combined = min(1.0, 0.25 * vec_sim + 0.75 * lexical_strength)
            scores.append(round(combined, 6))

        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[QueryHit],
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> List[RankedChunk]:
        if not candidates or top_k <= 0:
            return []

        scores = self._score_candidates(query, candidates)
        ranked: List[RankedChunk] = []

        for hit, score in zip(candidates, scores):
            if score < min_score:
                continue
            vec_sim = max(0.0, min(1.0, 1.0 - float(hit.distance)))
            ranked.append(
                RankedChunk(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    vector_distance=float(hit.distance),
                    vector_similarity=round(vec_sim, 6),
                    rerank_score=score,
                    metadata=dict(hit.metadata),
                )
            )

        ranked.sort(key=lambda c: (-c.rerank_score, c.vector_distance, c.chunk_id))
        return ranked[:top_k]


class OnnxCrossEncoder:
    """
    Real transformer cross-encoder reranker backed by ONNX Runtime and HuggingFace tokenizers.
    Jointly encodes (query, chunk_text) pairs through Xenova/ms-marco-MiniLM-L-6-v2
    (quantized ONNX model) and combines cross-attention logits with candidate-scoped lexical
    grounding.
    """

    DEFAULT_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
    DEFAULT_ONNX_FILE = "onnx/model_quantized.onnx"

    def __init__(
        self,
        model_repo: str = DEFAULT_REPO,
        onnx_filename: str = DEFAULT_ONNX_FILE,
        max_length: int = 512,
    ) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_repo = model_repo
        self.max_length = max_length
        self._lexical = LexicalReranker()

        tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=model_repo, filename=onnx_filename)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def rerank(
        self,
        query: str,
        candidates: Sequence[QueryHit],
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> List[RankedChunk]:
        if not candidates or top_k <= 0:
            return []

        import numpy as np

        pairs = [(query, hit.text) for hit in candidates]
        encodings = self.tokenizer.encode_batch(pairs)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self.session.run(None, feed)
        logits = np.asarray(outputs[0]).reshape(-1)
        lex_scores = self._lexical._score_candidates(query, candidates)

        ranked: List[RankedChunk] = []
        for idx, hit in enumerate(candidates):
            lex_score = lex_scores[idx]
            if lex_score < 0.15:
                # Candidate lacks lexical grounding for query terms/identifiers
                continue
            logit = float(logits[idx])
            ce_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            score = round(0.70 * ce_prob + 0.30 * lex_score, 6)
            if score < min_score:
                continue
            vec_sim = max(0.0, min(1.0, 1.0 - float(hit.distance)))
            ranked.append(
                RankedChunk(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    vector_distance=float(hit.distance),
                    vector_similarity=round(vec_sim, 6),
                    rerank_score=score,
                    metadata=dict(hit.metadata),
                )
            )

        ranked.sort(key=lambda c: (-c.rerank_score, c.vector_distance, c.chunk_id))
        return ranked[:top_k]
