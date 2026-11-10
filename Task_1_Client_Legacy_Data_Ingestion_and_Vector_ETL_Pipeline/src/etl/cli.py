"""Command-line interface: ingest, query, benchmark and reset."""
import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from ..config import get_settings
from .benchmark import HashEmbedding, render_report, run_benchmark, write_report
from .pipeline import ingest_directory
from .vector_store import VectorStore

console = Console()


def _store(db_path: str, embedder: str) -> VectorStore:
    fn = HashEmbedding() if embedder == "fake" else None
    return VectorStore(Path(db_path), embedding_function=fn)


def _ints(csv: str) -> List[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def cmd_ingest(args: argparse.Namespace) -> int:
    store = _store(args.db_path, args.embedder)
    manifest = Path(args.manifest) if args.manifest else None
    report = ingest_directory(
        Path(args.data_dir), store,
        manifest_path=manifest,
        chunk_size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size,
    )
    summary = Table(title="Ingestion summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Documents ok", str(report.documents_ok))
    summary.add_row("Documents failed", str(len(report.documents_failed)))
    summary.add_row("Chunks written", str(report.chunks_written))
    summary.add_row("Extract / chunk (s)", f"{report.extract_s:.2f} / {report.chunk_s:.2f}")
    summary.add_row("Embed / write (s)", f"{report.embed_s:.2f} / {report.write_s:.2f}")
    summary.add_row("Total (s)", f"{report.total_s:.2f}")
    console.print(summary)

    if report.documents_failed:
        failed = Table(title="Failed documents")
        for col in ("File", "Stage", "Error"):
            failed.add_column(col)
        for f in report.documents_failed:
            failed.add_row(f.file, f.stage, f.error)
        console.print(failed)
        return 1
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    store = _store(args.db_path, args.embedder)
    try:
        hits = store.query_by_tier_and_department(
            args.text, args.tier, args.dept, top_k=args.top_k
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not hits:
        print("No matching documents within your clearance.")
        return 0
    for i, h in enumerate(hits, start=1):
        m = h.metadata
        where = (
            f"page {m['page_number']}" if "page_number" in m
            else f"row {m['row_id']}" if "row_id" in m
            else m.get("section_title", "")
        )
        snippet = h.text.replace("\n", " ")[:120]
        print(f"{i}. [{h.distance:.3f}] {m['source_file']} (tier {m['clearance_tier']}, "
              f"{m['department']}) {where}: {snippet}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        report = run_benchmark(
            Path(work),
            scales=_ints(args.scales),
            batch_sizes=_ints(args.batch_sizes),
            queries_per_scenario=args.queries,
            embedder=args.embedder,
        )
    render_report(report, console)
    write_report(report, Path(args.out))
    console.print(f"Report written to {args.out}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    _store(args.db_path, args.embedder).reset()
    print(f"Collection reset at {args.db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="python -m src.etl.cli", description="Vector ETL pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db-path", default=str(settings.chroma_path))
        p.add_argument("--embedder", choices=["default", "fake"], default="default")

    p = sub.add_parser("ingest", help="Ingest a directory of documents")
    p.add_argument("--data-dir", default=str(settings.data_dir / "sample_documents"))
    p.add_argument("--manifest", default=None)
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--batch-size", type=int, default=256)
    common(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="Search with tier and department access control")
    p.add_argument("text")
    p.add_argument("--tier", type=int, required=True)
    p.add_argument("--dept", action="append", default=None, help="repeat for several departments")
    p.add_argument("--top-k", type=int, default=5)
    common(p)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("benchmark", help="Run the ingestion and query benchmark")
    p.add_argument("--scales", default="50,200,1000")
    p.add_argument("--batch-sizes", default="32,128,512")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--out", default="reports/benchmark.json")
    p.add_argument("--embedder", choices=["default", "fake"], default="default")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("reset", help="Drop and recreate the collection")
    common(p)
    p.set_defaults(func=cmd_reset)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
