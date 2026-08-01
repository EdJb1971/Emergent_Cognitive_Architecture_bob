"""Measures retrieval quality of a Chroma collection against a fixed query set.

Each collection is queried with the provider that produced its vectors, so a
Gemini baseline and a local candidate are compared like for like. Run this before
switching EMBEDDING_PROVIDER; a migration without a measured delta is a guess.

Usage:
    python -m src.tools.eval_retrieval --build-template --sample 60
    python -m src.tools.eval_retrieval --collection cognitive_cycles
    python -m src.tools.eval_retrieval --seeded \
        --fixture tests/fixtures/memory_retrieval_seeded.json
    python -m src.tools.eval_retrieval --collection cognitive_cycles \
        --compare cognitive_cycles__ollama_embeddinggemma_latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb

from src.core.config import settings
from src.providers.base import read_collection_identity
from src.providers.execution_scheduler import ModelExecutionScheduler
from src.providers.factory import build_embedding_provider, build_embedding_provider_for_identity
from src.tools.reembed import open_client

logger = logging.getLogger(__name__)

FIXTURE_VERSION = 1
SEEDED_FIXTURE_VERSION = 2
SUPPORTED_FIXTURE_VERSIONS = {FIXTURE_VERSION, SEEDED_FIXTURE_VERSION}
DEFAULT_FIXTURE = Path("tests/fixtures/memory_retrieval.json")


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for position, record_id in enumerate(retrieved, start=1):
        if record_id in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary relevance; every listed id counts equally."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, record_id in enumerate(retrieved[:k])
        if record_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


@dataclass
class QueryDiagnostic:
    query_id: str
    recall: float
    mrr: float
    ndcg: float
    retrieved: List[str]
    relevant: List[str]

    def summary(self) -> str:
        return (
            f"{self.query_id}: recall={self.recall:.3f} "
            f"MRR={self.mrr:.3f} NDCG={self.ndcg:.3f}"
        )


@dataclass
class EvaluationResult:
    collection: str
    identity: str
    k: int
    queries: int
    recall: float
    mrr: float
    ndcg: float
    misses: List[str]
    weak_queries: List[QueryDiagnostic]

    def summary(self) -> str:
        return (
            f"{self.collection} [{self.identity}] over {self.queries} queries at k={self.k}: "
            f"recall@k={self.recall:.3f} MRR={self.mrr:.3f} NDCG@k={self.ndcg:.3f}"
        )


def load_fixture(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No fixture at {path}. Run with --build-template to generate one from the current database."
        )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    version = fixture.get("version")
    if version not in SUPPORTED_FIXTURE_VERSIONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_FIXTURE_VERSIONS))
        raise ValueError(f"Fixture version {version} is not supported; expected one of {supported}.")
    queries = [q for q in fixture.get("queries", []) if q.get("query") and q.get("relevant_ids")]
    if not queries:
        raise ValueError(
            f"{path} contains no usable entries. Each query needs 'query' text and at least one 'relevant_ids' value."
        )
    fixture["queries"] = queries
    if version == SEEDED_FIXTURE_VERSION:
        records = fixture.get("records") or []
        record_ids = [record.get("id") for record in records if record.get("id") and record.get("document")]
        if not record_ids:
            raise ValueError(f"{path} contains no usable seeded records.")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"{path} contains duplicate seeded record ids.")
        known_record_ids = set(record_ids)
        unknown_ids = sorted(
            {
                relevant_id
                for query in queries
                for relevant_id in query["relevant_ids"]
                if relevant_id not in known_record_ids
            }
        )
        if unknown_ids:
            raise ValueError(
                f"{path} references relevant ids absent from records: {', '.join(unknown_ids)}"
            )
        fixture["records"] = [
            record for record in records if record.get("id") and record.get("document")
        ]
    return fixture


async def seed_fixture_collection(
    client: Any,
    collection_name: str,
    records: Sequence[Dict[str, Any]],
    scheduler: ModelExecutionScheduler,
) -> str:
    """Create an exact ephemeral collection from a versioned synthetic fixture."""
    if not collection_name.startswith("retrieval_eval_"):
        raise ValueError("Seeded collection names must start with 'retrieval_eval_'.")

    provider = build_embedding_provider(scheduler)
    identity = await provider.verify()
    metadata = identity.as_collection_metadata()
    metadata.update({"fixture_seeded": True, "fixture_record_count": len(records)})
    collection = client.create_collection(name=collection_name, metadata=metadata)

    ids = [str(record["id"]) for record in records]
    documents = [str(record["document"]) for record in records]
    metadatas = []
    for record in records:
        record_metadata = dict(record.get("metadata") or {})
        record_metadata["fixture_record"] = True
        metadatas.append(record_metadata)

    embeddings = await provider.embed_batch(documents)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    if collection.count() != len(records):
        raise ValueError(
            f"Seeded collection '{collection_name}' contains {collection.count()} records; "
            f"expected {len(records)}."
        )
    return identity.describe()


async def evaluate_collection(
    client: Any,
    collection_name: str,
    queries: Sequence[Dict[str, Any]],
    scheduler: ModelExecutionScheduler,
    k: int = 5,
) -> EvaluationResult:
    collection = client.get_collection(name=collection_name)
    identity = read_collection_identity(collection.metadata)
    if identity is None:
        raise ValueError(
            f"Collection '{collection_name}' has no embedding identity, so it cannot be queried reliably."
        )
    provider = build_embedding_provider_for_identity(identity, scheduler)

    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    gains: List[float] = []
    misses: List[str] = []
    weak_queries: List[QueryDiagnostic] = []

    query_embeddings = await provider.embed_batch([entry["query"] for entry in queries])
    if len(query_embeddings) != len(queries):
        raise ValueError(
            f"Embedding provider returned {len(query_embeddings)} query vectors for "
            f"{len(queries)} queries."
        )

    for entry, embedding in zip(queries, query_embeddings):
        response = collection.query(query_embeddings=[embedding], n_results=k)
        retrieved = (response.get("ids") or [[]])[0]
        relevant = entry["relevant_ids"]

        recall = recall_at_k(retrieved, relevant, k)
        recalls.append(recall)
        rank = reciprocal_rank(retrieved, relevant)
        reciprocal_ranks.append(rank)
        gain = ndcg_at_k(retrieved, relevant, k)
        gains.append(gain)
        if rank == 0.0:
            misses.append(entry.get("id") or entry["query"][:60])
        if recall < 1.0 - 1e-9 or rank < 1.0 - 1e-9 or gain < 1.0 - 1e-9:
            weak_queries.append(
                QueryDiagnostic(
                    query_id=entry.get("id") or entry["query"][:60],
                    recall=recall,
                    mrr=rank,
                    ndcg=gain,
                    retrieved=list(retrieved),
                    relevant=list(relevant),
                )
            )

    count = len(queries)
    return EvaluationResult(
        collection=collection_name,
        identity=identity.describe(),
        k=k,
        queries=count,
        recall=sum(recalls) / count,
        mrr=sum(reciprocal_ranks) / count,
        ndcg=sum(gains) / count,
        misses=misses,
        weak_queries=weak_queries,
    )


def print_diagnostics(result: EvaluationResult) -> None:
    if not result.weak_queries:
        return
    print(f"Queries below perfect relevance ({len(result.weak_queries)}):")
    for diagnostic in result.weak_queries:
        print(f"  {diagnostic.summary()}")


def build_template(client: Any, collection_name: str, sample: int, path: Path) -> int:
    """Emit a fixture skeleton seeded from real records; queries still need human review."""
    collection = client.get_collection(name=collection_name)
    total = collection.count()
    page = collection.get(include=["documents", "metadatas"], limit=min(total, max(sample * 4, sample)))
    ids = page.get("ids") or []
    documents = page.get("documents") or []

    population = [(i, d) for i, d in zip(ids, documents) if d]
    chosen = random.sample(population, min(sample, len(population)))

    fixture = {
        "version": FIXTURE_VERSION,
        "source_collection": collection_name,
        "notes": (
            "Replace each 'query' with how you would actually ask for this memory, and confirm "
            "'relevant_ids'. A seeded query copied verbatim from the document measures nothing."
        ),
        "queries": [
            {
                "id": f"q{index + 1}",
                "query": document[:200].replace("\n", " ").strip(),
                "relevant_ids": [record_id],
                "tags": [],
                "reviewed": False,
            }
            for index, (record_id, document) in enumerate(chosen)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return len(fixture["queries"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure retrieval quality for a Chroma collection.")
    parser.add_argument("--collection", help="Baseline collection to evaluate; defaults to the fixture source collection or cognitive_cycles.")
    parser.add_argument("--compare", help="Candidate collection to evaluate against the baseline.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--build-template", action="store_true", help="Generate a fixture skeleton and exit.")
    parser.add_argument("--sample", type=int, default=60, help="Records to seed into the template.")
    parser.add_argument(
        "--seeded",
        action="store_true",
        help="Seed and evaluate an isolated ephemeral collection from a version 2 fixture.",
    )
    return parser


async def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.build_template:
        client = open_client()
        collection_name = args.collection or "cognitive_cycles"
        written = build_template(client, collection_name, args.sample, args.fixture)
        print(f"Wrote {written} draft queries to {args.fixture}.")
        print("Review every entry before trusting a measurement taken from it.")
        return 0

    try:
        fixture = load_fixture(args.fixture)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    queries = fixture["queries"]
    unreviewed = [q for q in queries if not q.get("reviewed", True)]
    if unreviewed:
        print(f"Warning: {len(unreviewed)} of {len(queries)} queries are still marked reviewed=false.\n")

    scheduler = ModelExecutionScheduler(
        max_interactive=settings.OLLAMA_MAX_INTERACTIVE_REQUESTS,
        max_background=settings.OLLAMA_MAX_BACKGROUND_REQUESTS,
    )

    collection_name = args.collection or fixture.get("source_collection") or "cognitive_cycles"
    if args.seeded:
        if fixture.get("version") != SEEDED_FIXTURE_VERSION:
            print(f"--seeded requires a version {SEEDED_FIXTURE_VERSION} fixture with records.")
            return 1
        if args.compare:
            print("--compare is not supported with the single-provider ephemeral seeded run.")
            return 2
        client = chromadb.EphemeralClient()
        try:
            identity = await seed_fixture_collection(
                client,
                collection_name,
                fixture["records"],
                scheduler,
            )
        except ValueError as error:
            print(error)
            return 1
        print(
            f"Seeded {len(fixture['records'])} records into ephemeral collection "
            f"'{collection_name}' [{identity}]."
        )
    else:
        client = open_client()

    baseline = await evaluate_collection(client, collection_name, queries, scheduler, args.k)
    print(baseline.summary())
    print_diagnostics(baseline)

    if not args.compare:
        return 0

    candidate = await evaluate_collection(client, args.compare, queries, scheduler, args.k)
    print(candidate.summary())
    print_diagnostics(candidate)
    print(
        f"\ndelta recall@k={candidate.recall - baseline.recall:+.3f} "
        f"MRR={candidate.mrr - baseline.mrr:+.3f} "
        f"NDCG@k={candidate.ndcg - baseline.ndcg:+.3f}"
    )
    newly_missed = sorted(set(candidate.misses) - set(baseline.misses))
    if newly_missed:
        print(f"\nQueries the candidate misses but the baseline finds ({len(newly_missed)}):")
        for item in newly_missed:
            print(f"  {item}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
