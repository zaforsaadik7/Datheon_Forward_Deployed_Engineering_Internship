"""Deterministic synthetic corpus generator used to benchmark ingestion at scale."""
import json
import random
from pathlib import Path
from typing import Dict, List

import yaml

from ..config import DEPARTMENTS

# Small per-department vocabularies so retrieval queries have something to find.
VOCAB: Dict[str, List[str]] = {
    "engineering": ["vpc", "subnet", "peering", "latency", "kubernetes", "cluster", "deployment", "runbook",
                    "routing", "firewall", "autoscaling", "observability", "rollback", "pipeline"],
    "finance": ["budget", "invoice", "vendor", "audit", "forecast", "accrual", "ledger", "payment",
                "variance", "procurement", "quarter", "reconciliation", "spend", "approval"],
    "legal": ["agreement", "license", "clause", "liability", "indemnity", "covenant", "jurisdiction",
              "termination", "confidentiality", "royalty", "arbitration", "warranty", "assignment"],
    "hr": ["salary", "band", "bonus", "benefits", "onboarding", "performance", "leave", "headcount",
           "compensation", "review", "promotion", "recruiting", "policy"],
    "executive": ["roadmap", "strategy", "revenue", "forecast", "board", "initiative", "market",
                  "milestone", "investment", "expansion", "competitor", "objective", "priority"],
    "operations": ["supplier", "logistics", "sla", "inventory", "shipment", "compliance", "security",
                   "catalog", "warehouse", "maintenance", "vendor", "capacity", "incident"],
}
FILLER = ["the", "of", "for", "and", "with", "across", "during", "within", "under", "per", "each", "new"]
KINDS = ("md", "csv", "txt", "json")


def _sentence(rng: random.Random, words: List[str]) -> str:
    n = rng.randint(8, 16)
    toks = [rng.choice(words) if rng.random() < 0.6 else rng.choice(FILLER) for _ in range(n)]
    return " ".join(toks).capitalize() + "."


def _paragraph(rng: random.Random, words: List[str]) -> str:
    return " ".join(_sentence(rng, words) for _ in range(rng.randint(3, 6)))


def _markdown(rng: random.Random, words: List[str], idx: int) -> str:
    parts = [f"# Synthetic runbook {idx}"]
    for s in range(rng.randint(4, 7)):
        parts.append(f"## Section {s + 1}: {rng.choice(words)} {rng.choice(words)}")
        parts.append(_paragraph(rng, words))
        parts.append(_paragraph(rng, words))
    return "\n\n".join(parts) + "\n"


def _csv(rng: random.Random, words: List[str], dept: str) -> str:
    rows = ["id,department,item,description,amount"]
    for i in range(50):
        rows.append(f"R-{i:03d},{dept},{rng.choice(words)},{_sentence(rng, words)},{rng.randint(100, 99999)}")
    return "\n".join(rows) + "\n"


def _txt(rng: random.Random, words: List[str], idx: int) -> str:
    parts = [f"MEMORANDUM {idx}"]
    parts.extend(_paragraph(rng, words) for _ in range(rng.randint(4, 8)))
    return "\n\n".join(parts) + "\n"


def _json(rng: random.Random, words: List[str], dept: str) -> str:
    records = [
        {"id": f"REC-{i:03d}", "name": " ".join(rng.choices(words, k=2)), "notes": _sentence(rng, words)}
        for i in range(20)
    ]
    return json.dumps({"catalog": {"department": dept}, "records": records}, indent=2) + "\n"


def make_synthetic_corpus(out_dir: Path, n_docs: int, seed: int = 42) -> Path:
    """
    Write n_docs documents plus manifest.yaml into out_dir and return the manifest path.
    Uses a private random.Random(seed), so global RNG state is untouched and runs repeat exactly.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depts = sorted(DEPARTMENTS)
    entries = []

    for i in range(n_docs):
        kind = KINDS[i % len(KINDS)]
        dept = depts[rng.randrange(len(depts))]
        # the first four documents cover tiers 1-4 so every corpus of 4+ docs has all tiers
        tier = i % 4 + 1 if i < 4 else rng.randint(1, 4)
        words = VOCAB[dept]
        name = f"{dept}_doc{i:05d}_tier{tier}.{kind}"

        if kind == "md":
            body = _markdown(rng, words, i)
        elif kind == "csv":
            body = _csv(rng, words, dept)
        elif kind == "txt":
            body = _txt(rng, words, i)
        else:
            body = _json(rng, words, dept)
        (out_dir / name).write_text(body, encoding="utf-8", newline="\n")

        entries.append({
            "file": name,
            "department": dept,
            "clearance_tier": tier,
            "document_date": f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}",
        })

    manifest_path = out_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump({"documents": entries}, sort_keys=False), encoding="utf-8"
    )
    return manifest_path
