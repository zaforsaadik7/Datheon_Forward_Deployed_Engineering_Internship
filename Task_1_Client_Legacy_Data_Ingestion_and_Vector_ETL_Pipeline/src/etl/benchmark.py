"""Batch ingestion and filtered-query benchmark harness."""
import hashlib
import json
import math
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import chromadb
import psutil
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from rich.console import Console
from rich.table import Table

from .pipeline import ingest_directory
from .synthetic import VOCAB, make_synthetic_corpus
from .vector_store import VectorStore


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile by linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(ordered[int(k)])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


class HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic 64-dim embedder so benchmarks and CLI tests run without a model download."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "hash-embedding-64"

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HashEmbedding":
        return HashEmbedding()

    def __call__(self, input: Documents) -> Embeddings:
        out: Embeddings = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest() * 2  # 64 bytes
            raw = [b - 127.5 for b in digest]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / norm for x in raw])
        return out


class PeakRssSampler:
    """Context manager that polls this process's RSS every 20 ms and records the peak."""

    def __init__(self, interval_s: float = 0.02):
        self._interval = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def __enter__(self) -> "PeakRssSampler":
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)


SCENARIOS = [
    {"name": "no_filter", "user_tier": 4, "departments": None},
    {"name": "tier_only", "user_tier": 2, "departments": None},
    {"name": "tier_one_dept", "user_tier": 2, "departments": ["engineering"]},
    {"name": "tier_three_depts", "user_tier": 3, "departments": ["engineering", "finance", "hr"]},
]


def _query_phrases() -> List[str]:
    phrases: List[str] = []
    for dept in sorted(VOCAB):
        words = VOCAB[dept]
        for i in range(0, 4):
            phrases.append(" ".join(words[i * 3:(i * 3) + 3]))
    return phrases[:20] if len(phrases) >= 20 else phrases


def _make_store(path: Path, embedder: str) -> VectorStore:
    fn = HashEmbedding() if embedder == "fake" else None
    return VectorStore(path, embedding_function=fn)


def _measure_queries(store: VectorStore, scale: int, n: int) -> List[Dict[str, Any]]:
    phrases = _query_phrases()
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        tier, depts = scenario["user_tier"], scenario["departments"]
        allowed = {d.lower() for d in depts} if depts else None
        for i in range(10):  # warm-up, not recorded
            store.query_by_tier_and_department(phrases[i % len(phrases)], tier, depts, top_k=5)
        latencies: List[float] = []
        violations = 0
        for i in range(n):
            t0 = time.perf_counter()
            hits = store.query_by_tier_and_department(phrases[i % len(phrases)], tier, depts, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            for h in hits:
                if h.metadata["clearance_tier"] > tier:
                    violations += 1
                elif allowed is not None and h.metadata["department"] not in allowed:
                    violations += 1
        rows.append({
            "scale_docs": scale,
            "scenario": scenario["name"],
            "n": n,
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "filter_violations": violations,
        })
    return rows


def run_benchmark(
    work_dir: Path,
    scales: Sequence[int] = (50, 200, 1000),
    batch_sizes: Sequence[int] = (32, 128, 512),
    queries_per_scenario: int = 200,
    embedder: Literal["default", "fake"] = "default",
    seed: int = 42,
) -> Dict[str, Any]:
    """Ingest synthetic corpora at several scales and batch sizes, then time filtered queries."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "chromadb": chromadb.__version__,
            "embedder": embedder,
            "seed": seed,
            "chunk_size": 900,
            "overlap": 120,
        },
        "ingestion": [],
        "queries": [],
    }

    for scale in scales:
        corpus = work_dir / f"corpus_{scale}"
        make_synthetic_corpus(corpus, scale, seed=seed)
        query_store: Optional[VectorStore] = None

        for bs in batch_sizes:
            store = _make_store(work_dir / f"db_{scale}_{bs}", embedder)
            store.reset()
            with PeakRssSampler() as rss:
                result = ingest_directory(corpus, store, batch_size=bs)
            total = max(result.total_s, 1e-9)
            report["ingestion"].append({
                "scale_docs": scale,
                "batch_size": bs,
                "documents": result.documents_ok,
                "chunks": result.chunks_written,
                "bytes": result.bytes_read,
                "extract_s": result.extract_s,
                "chunk_s": result.chunk_s,
                "embed_s": result.embed_s,
                "write_s": result.write_s,
                "total_s": result.total_s,
                "docs_per_s": result.documents_ok / total,
                "chunks_per_s": result.chunks_written / total,
                "mb_per_s": (result.bytes_read / (1024 * 1024)) / total,
                "rss_peak_mb": rss.peak_mb,
                "failed_documents": len(result.documents_failed),
            })
            if query_store is None:
                query_store = store

        assert query_store is not None
        report["queries"].extend(_measure_queries(query_store, scale, queries_per_scenario))

    return report


def write_report(report: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def render_report(report: Dict[str, Any], console: Optional[Console] = None) -> None:
    console = console or Console()
    ing = Table(title="Ingestion")
    for col in ("docs", "batch", "chunks", "extract s", "chunk s", "embed s", "write s", "chunks/s", "MB/s", "peak RSS MB"):
        ing.add_column(col, justify="right")
    for r in report["ingestion"]:
        ing.add_row(
            str(r["scale_docs"]), str(r["batch_size"]), str(r["chunks"]),
            f"{r['extract_s']:.2f}", f"{r['chunk_s']:.2f}", f"{r['embed_s']:.2f}", f"{r['write_s']:.2f}",
            f"{r['chunks_per_s']:.1f}", f"{r['mb_per_s']:.2f}", f"{r['rss_peak_mb']:.0f}",
        )
    console.print(ing)

    q = Table(title="Query latency (ms)")
    for col in ("docs", "scenario", "n", "P50", "P95", "P99", "violations"):
        q.add_column(col, justify="right")
    for r in report["queries"]:
        q.add_row(
            str(r["scale_docs"]), r["scenario"], str(r["n"]),
            f"{r['p50_ms']:.1f}", f"{r['p95_ms']:.1f}", f"{r['p99_ms']:.1f}", str(r["filter_violations"]),
        )
    console.print(q)
