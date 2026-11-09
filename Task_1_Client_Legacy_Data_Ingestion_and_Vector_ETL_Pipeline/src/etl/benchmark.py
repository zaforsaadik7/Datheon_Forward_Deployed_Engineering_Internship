"""Batch Ingestion and Query Latency Benchmarking Suite."""
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rich.console import Console
from rich.table import Table

from .models import BenchmarkResult
from .pipeline import VectorETLPipeline
from .vector_store import VectorStoreManager
from ..config import settings

console = Console()


class ETLBenchmarkRunner:
    """
    Executes rigorous performance profiling across all stages of the vector ETL pipeline:
    - Extraction latency by format
    - Semantic chunking throughput
    - Vector database cosine index write speed
    - End-to-end query latency with RBAC metadata filtering (P50, P95, P99)
    """

    def __init__(self, data_dir: Path, collection_name: str = "benchmark_collection"):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.vector_store = VectorStoreManager()
        self.pipeline = VectorETLPipeline(vector_store=self.vector_store)

    def run_benchmark(self, query_iterations: int = 30) -> BenchmarkResult:
        """Run full profiling suite and return BenchmarkResult."""
        console.print("[bold cyan]🚀 Starting Datheon Vector ETL Pipeline Benchmark...[/bold cyan]")

        # 1. Measure Document Extraction
        t0 = time.perf_counter()
        raw_docs = []
        files = list(self.data_dir.glob("*.*"))
        for f in files:
            ext = f.suffix.lower()
            if ext in self.pipeline.EXTRACTOR_REGISTRY:
                extractor = self.pipeline.EXTRACTOR_REGISTRY[ext]()
                raw_docs.extend(extractor.extract(f))
        extraction_time_ms = (time.perf_counter() - t0) * 1000

        # 2. Measure Semantic Chunking
        t1 = time.perf_counter()
        chunks = self.pipeline.chunker.chunk_documents(raw_docs)
        chunking_time_ms = (time.perf_counter() - t1) * 1000

        total_tokens = sum(c.token_count for c in chunks)

        # 3. Measure Vector Store Indexing
        self.vector_store.reset_collection(self.collection_name)
        t2 = time.perf_counter()
        self.vector_store.upsert_chunks(chunks, collection_name=self.collection_name)
        indexing_time_ms = (time.perf_counter() - t2) * 1000

        total_time_ms = extraction_time_ms + chunking_time_ms + indexing_time_ms
        total_time_sec = total_time_ms / 1000.0

        throughput_docs = len(raw_docs) / total_time_sec if total_time_sec > 0 else 0
        throughput_chunks = len(chunks) / total_time_sec if total_time_sec > 0 else 0

        # 4. Measure Query Latency with Metadata Filters
        sample_queries = [
            ("What is the acceptable use policy for enterprise devices?", 1, ["IT/Security"]),
            ("What are the VPC peering CIDR blocks?", 2, ["Engineering"]),
            ("What was the approved Q3 budget for GPU compute?", 3, ["Finance"]),
            ("What are the cap table terms for Project Titan?", 4, ["Legal", "Executive"]),
            ("What are the salary bands for senior forward deployed engineers?", 2, ["HR"]),
        ]

        query_latencies = []
        for q_idx in range(query_iterations):
            q_text, tier, depts = sample_queries[q_idx % len(sample_queries)]
            t_start = time.perf_counter()
            self.vector_store.query_with_rbac(
                query_text=q_text,
                user_tier=tier,
                user_departments=depts,
                top_k=3,
                collection_name=self.collection_name,
            )
            lat_ms = (time.perf_counter() - t_start) * 1000
            query_latencies.append(lat_ms)

        lat_arr = np.array(query_latencies)
        query_stats = {
            "mean": round(float(np.mean(lat_arr)), 2),
            "p50": round(float(np.percentile(lat_arr, 50)), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
            "p99": round(float(np.percentile(lat_arr, 99)), 2),
        }

        result = BenchmarkResult(
            num_documents=len(raw_docs),
            num_chunks=len(chunks),
            total_tokens=total_tokens,
            extraction_time_ms=round(extraction_time_ms, 2),
            chunking_time_ms=round(chunking_time_ms, 2),
            indexing_time_ms=round(indexing_time_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            throughput_docs_per_sec=round(throughput_docs, 2),
            throughput_chunks_per_sec=round(throughput_chunks, 2),
            query_latency_ms=query_stats,
        )

        self._display_results(result)
        return result

    def _display_results(self, result: BenchmarkResult):
        """Render results in a clean rich terminal table."""
        table = Table(title="📊 Vector ETL Ingestion & Latency Benchmark Results", style="bold green")
        table.add_column("Metric / Phase", style="cyan", no_wrap=True)
        table.add_column("Observed Value", style="bold yellow")
        table.add_column("Target SLA", style="magenta")

        table.add_row("Raw Extracted Documents", str(result.num_documents), "-")
        table.add_row("Total Semantic Chunks", str(result.num_chunks), "-")
        table.add_row("Indexed Token Count", f"{result.total_tokens:,}", "-")
        table.add_row("Extraction Phase Latency", f"{result.extraction_time_ms} ms", "< 500 ms")
        table.add_row("Semantic Chunking Latency", f"{result.chunking_time_ms} ms", "< 200 ms")
        table.add_row("Vector Upsert Latency", f"{result.indexing_time_ms} ms", "< 1000 ms")
        table.add_row("Total Pipeline Runtime", f"{result.total_time_ms} ms", "< 2000 ms")
        table.add_row("Throughput (Chunks/sec)", f"{result.throughput_chunks_per_sec} chunks/s", "> 50 chunks/s")
        table.add_row("Query Latency (P50 Median)", f"{result.query_latency_ms['p50']} ms", "< 50 ms")
        table.add_row("Query Latency (P95 Tail)", f"{result.query_latency_ms['p95']} ms", "< 100 ms")
        table.add_row("Query Latency (P99 Tail)", f"{result.query_latency_ms['p99']} ms", "< 150 ms")

        console.print(table)

    def save_json_report(self, result: BenchmarkResult, output_path: Path):
        """Save benchmark statistics to a JSON artifact."""
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"[green]✓ Benchmark report successfully written to {output_path}[/green]")
