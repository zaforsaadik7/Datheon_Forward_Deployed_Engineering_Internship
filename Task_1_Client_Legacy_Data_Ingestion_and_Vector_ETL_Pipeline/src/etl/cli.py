"""Command-Line Interface for the Datheon Vector ETL Pipeline."""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .pipeline import VectorETLPipeline
from .vector_store import VectorStoreManager
from .benchmark import ETLBenchmarkRunner
from ..config import settings

console = Console()


def handle_ingest(args):
    """Handle `ingest` subcommand."""
    data_dir = Path(args.data_dir)
    console.print(f"[bold cyan]📥 Ingesting legacy data from:[/bold cyan] {data_dir}")
    console.print(f"[cyan]Target Collection:[/cyan] {args.collection} (Reset: {args.reset})")

    pipeline = VectorETLPipeline()
    result = pipeline.run(
        data_dir=data_dir,
        collection_name=args.collection,
        reset_existing=args.reset
    )

    table = Table(title="✨ Ingestion Summary", style="bold blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Discovered Source Files", str(result.total_files_discovered))
    table.add_row("Extracted Documents", str(result.total_documents_extracted))
    table.add_row("Created Semantic Chunks", str(result.total_chunks_created))
    table.add_row("Target Collection", result.collection_name)
    table.add_row("Duration", f"{result.duration_seconds} s")
    table.add_row("Throughput", f"{result.throughput_chunks_per_sec} chunks/s")

    console.print(table)

    if result.errors:
        console.print("[bold red]Warnings / Errors during ingestion:[/bold red]")
        for err in result.errors:
            console.print(f"[red]• {err}[/red]")


def handle_query(args):
    """Handle `query` subcommand."""
    store = VectorStoreManager()
    console.print(f"[bold cyan]🔍 Semantic Search Query:[/bold cyan] '{args.query}'")
    console.print(f"[dim]User Clearance Tier: {args.tier} | Departments: {args.department or 'All'}[/dim]\n")

    depts = [d.strip() for d in args.department.split(",")] if args.department else None

    results = store.query_with_rbac(
        query_text=args.query,
        user_tier=args.tier,
        user_departments=depts,
        top_k=args.top_k,
        collection_name=args.collection
    )

    if not results:
        console.print("[yellow]No matching documents found within your clearance boundaries.[/yellow]")
        return

    table = Table(title=f"🔎 Query Results (Top {len(results)})", style="bold green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Sim.", style="magenta", width=7)
    table.add_column("Source & Metadata", style="cyan", width=32)
    table.add_column("Content Snippet", style="white")

    for idx, r in enumerate(results, start=1):
        m = r["metadata"]
        src = f"{m.get('source_file', 'unknown')}\nTier: {m.get('clearance_tier')} | Dept: {m.get('department')}"
        if m.get("page_number"):
            src += f"\nPage: {m.get('page_number')}"
        elif m.get("row_id"):
            src += f"\nRow: {m.get('row_id')}"
        elif m.get("section_title"):
            src += f"\nSec: {m.get('section_title')[:18]}"

        snippet = r["text"].replace("\n", " ")[:160] + ("..." if len(r["text"]) > 160 else "")
        table.add_row(str(idx), f"{r['similarity_score']:.3f}", src, snippet)

    console.print(table)


def handle_benchmark(args):
    """Handle `benchmark` subcommand."""
    data_dir = Path(args.data_dir)
    runner = ETLBenchmarkRunner(data_dir=data_dir, collection_name="benchmark_suite")
    results = runner.run_benchmark(query_iterations=args.iterations)

    if args.output:
        runner.save_json_report(results, Path(args.output))


def handle_inspect(args):
    """Inspect collection statistics and sample chunks."""
    store = VectorStoreManager()
    total = store.count(collection_name=args.collection)
    console.print(f"[bold cyan]Collection:[/bold cyan] {args.collection} (Total Indexed Vectors: {total})")

    if total == 0:
        console.print("[yellow]Collection is empty. Run `python -m src.etl.cli ingest` first.[/yellow]")
        return

    coll = store.get_or_create_collection(args.collection)
    sample = coll.peek(limit=args.limit)

    table = Table(title=f"📋 Sample Vectors in '{args.collection}'", style="bold blue")
    table.add_column("Chunk ID", style="dim")
    table.add_column("Tier", style="yellow")
    table.add_column("Dept", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Text Excerpt", style="white")

    for idx, cid in enumerate(sample["ids"]):
        meta = sample["metadatas"][idx]
        doc = sample["documents"][idx].replace("\n", " ")[:90] + "..."
        table.add_row(
            cid[:22],
            str(meta.get("clearance_tier")),
            str(meta.get("department")),
            str(meta.get("source_file")),
            doc
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Datheon Forward Deployed Engineering - Task 1 Vector ETL CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest
    ingest_p = subparsers.add_parser("ingest", help="Run ETL pipeline on corporate documents")
    ingest_p.add_argument("--data-dir", default=str(settings.SAMPLE_DOCS_DIR), help="Directory of source documents")
    ingest_p.add_argument("--collection", default=settings.DEFAULT_COLLECTION_NAME, help="Target ChromaDB collection")
    ingest_p.add_argument("--reset", action="store_true", help="Reset collection before indexing")

    # Query
    query_p = subparsers.add_parser("query", help="Semantic similarity search with RBAC metadata filter")
    query_p.add_argument("--query", required=True, help="Search query string")
    query_p.add_argument("--tier", type=int, default=4, help="User clearance level (1 to 4)")
    query_p.add_argument("--department", default=None, help="User department or comma-separated list")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_p.add_argument("--collection", default=settings.DEFAULT_COLLECTION_NAME, help="Target ChromaDB collection")

    # Benchmark
    bench_p = subparsers.add_parser("benchmark", help="Profile ingestion throughput & query latencies")
    bench_p.add_argument("--data-dir", default=str(settings.SAMPLE_DOCS_DIR), help="Directory of source documents")
    bench_p.add_argument("--iterations", type=int, default=25, help="Number of query benchmark rounds")
    bench_p.add_argument("--output", default="benchmark_results.json", help="Path to save JSON benchmark report")

    # Inspect
    inspect_p = subparsers.add_parser("inspect", help="Inspect collection counts and sample vectors")
    inspect_p.add_argument("--collection", default=settings.DEFAULT_COLLECTION_NAME, help="Collection name")
    inspect_p.add_argument("--limit", type=int, default=5, help="Number of samples to display")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "inspect":
        handle_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
