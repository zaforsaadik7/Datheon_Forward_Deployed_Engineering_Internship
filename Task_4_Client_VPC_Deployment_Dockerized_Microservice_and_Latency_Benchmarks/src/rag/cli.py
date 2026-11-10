"""Command-line interface for Enterprise Multi-Tenant RAG (ingest, ask, eval, personas, serve)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..etl.pipeline import ingest_directory
from ..etl.vector_store import VectorStore
from .engine import MultiTenantRAGEngine
from .eval import run_rbac_evaluation
from .rbac import PERSONAS, load_api_key_registry
from .reranker import LexicalReranker, OnnxCrossEncoder, Reranker, _tokenize

console = Console()


class DeterministicHashEmbedding(EmbeddingFunction[Documents]):
    """
    Domain-agnostic 256-dim signed feature-hashing embedding function (Weinberger et al.)
    for offline unit testing. Uses general token and character 4-gram SHA-256 projections
    with zero corpus-specific or query-specific vocabulary lists.
    """

    DIM = 256

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "deterministic_hash_embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "DeterministicHashEmbedding":
        return DeterministicHashEmbedding()

    def __call__(self, input: Documents) -> Embeddings:
        results: Embeddings = []
        for text in input:
            vec = [0.0] * self.DIM
            tokens = _tokenize(text)
            features: List[tuple[str, float]] = []
            for tok in tokens:
                features.append((f"w:{tok}", 1.0))
                if len(tok) >= 4:
                    for i in range(len(tok) - 3):
                        features.append((f"g:{tok[i:i+4]}", 0.35))

            if not features:
                features.append((f"raw:{text.strip().lower()}", 1.0))

            for feat, weight in features:
                digest = hashlib.sha256(feat.encode("utf-8")).digest()
                for slot in range(0, 8, 2):
                    idx = digest[slot] % self.DIM
                    sign = 1.0 if (digest[slot + 1] & 1) == 0 else -1.0
                    vec[idx] += sign * weight

            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results


def _build_reranker(kind: str) -> Reranker:
    if kind == "onnx":
        return OnnxCrossEncoder()
    return LexicalReranker()


def build_parser() -> argparse.ArgumentParser:
    s = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m src.rag.cli",
        description="Enterprise Multi-Tenant RAG with RBAC CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. personas
    sub.add_parser("personas", help="Display the 6 RBAC personas and their configured API keys")

    # 2. ingest
    p_ing = sub.add_parser("ingest", help="Ingest enterprise corpus into ChromaDB")
    p_ing.add_argument("--data-dir", type=Path, default=s.data_dir)
    p_ing.add_argument("--chroma-path", type=Path, default=s.chroma_path)
    p_ing.add_argument("--collection", type=str, default=s.collection_name)
    p_ing.add_argument("--use-fake-embeddings", action="store_true")

    # 3. ask
    p_ask = sub.add_parser("ask", help="Query the Multi-Tenant RAG engine as a persona")
    p_ask.add_argument("--persona", type=str, choices=sorted(PERSONAS.keys()), required=True)
    p_ask.add_argument("--query", type=str, required=True)
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--chroma-path", type=Path, default=s.chroma_path)
    p_ask.add_argument("--collection", type=str, default=s.collection_name)
    p_ask.add_argument("--reranker", choices=["lexical", "onnx"], default="onnx")
    p_ask.add_argument("--use-fake-embeddings", action="store_true")

    # 4. eval
    p_eval = sub.add_parser("eval", help="Run the 19-case RBAC & accuracy evaluation suite")
    p_eval.add_argument("--data-dir", type=Path, default=s.data_dir)
    p_eval.add_argument("--chroma-path", type=Path, default=s.chroma_path)
    p_eval.add_argument("--collection", type=str, default=s.collection_name)
    p_eval.add_argument("--out", type=Path, default=Path("reports/rag_eval.json"))
    p_eval.add_argument("--reranker", choices=["lexical", "onnx"], default="onnx")
    p_eval.add_argument("--use-fake-embeddings", action="store_true")

    # 5. serve
    p_srv = sub.add_parser("serve", help="Run the authenticated FastAPI RAG server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "personas":
        registry = load_api_key_registry()
        key_by_user = {p.user_id: k for k, p in registry.items()}
        table = Table(title="Enterprise Multi-Tenant RBAC Personas")
        table.add_column("Persona ID", style="cyan")
        table.add_column("API Key", style="yellow")
        table.add_column("Tenant", style="magenta")
        table.add_column("Tier", justify="right")
        table.add_column("Departments", style="green")
        table.add_column("Role Title")
        for pid, p in PERSONAS.items():
            table.add_row(
                pid,
                key_by_user.get(pid, "-"),
                p.tenant_id,
                str(p.clearance_tier),
                ", ".join(sorted(p.departments)),
                p.role_title,
            )
        console.print(table)
        return 0

    if args.command == "ingest":
        emb = DeterministicHashEmbedding() if args.use_fake_embeddings else None
        store = VectorStore(
            path=args.chroma_path,
            collection_name=args.collection,
            embedding_function=emb,
        )
        rep = ingest_directory(args.data_dir, store)
        console.print(
            f"[bold green]Ingested {rep.documents_ok} documents[/bold green] "
            f"({rep.chunks_written} chunks, {len(rep.documents_failed)} failed) in {rep.total_s:.2f}s"
        )
        return 0 if not rep.documents_failed else 1

    if args.command == "ask":
        emb = DeterministicHashEmbedding() if args.use_fake_embeddings else None
        store = VectorStore(
            path=args.chroma_path,
            collection_name=args.collection,
            embedding_function=emb,
        )
        engine = MultiTenantRAGEngine(
            store=store,
            reranker=_build_reranker(args.reranker),
        )
        principal = PERSONAS[args.persona]
        resp = engine.answer(query=args.query, principal=principal, top_k=args.top_k)
        console.print(f"[bold cyan]Persona:[/bold cyan] {principal.user_id} (Tier {principal.clearance_tier}, {sorted(principal.departments)})")
        console.print(f"[bold cyan]Rewritten Query:[/bold cyan] {resp.rewritten_query}")
        console.print(f"[bold cyan]Abstained:[/bold cyan] {resp.abstained}")
        console.print(f"[bold green]Answer:[/bold green] {resp.answer}")
        if resp.citations:
            console.print("[bold yellow]Citations:[/bold yellow] " + ", ".join(c.format_tag() for c in resp.citations))
        return 0

    if args.command == "eval":
        emb = DeterministicHashEmbedding() if args.use_fake_embeddings else None
        store = VectorStore(
            path=args.chroma_path,
            collection_name=args.collection,
            embedding_function=emb,
        )
        if store.count() == 0:
            ingest_directory(args.data_dir, store)

        engine = MultiTenantRAGEngine(
            store=store,
            reranker=_build_reranker(args.reranker),
        )
        manifest_path = Path(args.data_dir) / "manifest.yaml"
        report = run_rbac_evaluation(engine, manifest_path=manifest_path)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        summary = report["summary"]
        table = Table(title="RBAC & Ground-Truth Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold green")
        table.add_row("Total Cases", str(summary["total_cases"]))
        table.add_row("RBAC Leaks", str(summary["rbac_leaks"]))
        table.add_row("Abstention Accuracy", f"{summary['abstention_accuracy'] * 100:.1f}%")
        table.add_row("Answer Accuracy", f"{summary['answer_accuracy'] * 100:.1f}%")
        table.add_row("Citation Locator Accuracy", f"{summary['citation_locator_accuracy'] * 100:.1f}%")
        table.add_row("Guardrail Pass Rate", f"{summary['guardrail_pass_rate'] * 100:.1f}%")
        table.add_row("Latency P50 / P95 (ms)", f"{summary['latency_ms']['p50']:.1f} / {summary['latency_ms']['p95']:.1f}")
        console.print(table)
        return 0 if summary["rbac_leaks"] == 0 else 2

    if args.command == "serve":
        import uvicorn
        from ..api.app import create_app

        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
