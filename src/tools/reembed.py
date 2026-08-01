"""Rebuilds a Chroma collection's vectors with the active embedding provider.

Vector spaces are not interchangeable between embedding models, so a switch is a
rebuild into a new collection rather than an in-place update. The source collection
is never modified, which keeps the previous vector space recoverable.

Usage:
    python -m src.tools.reembed --list
    python -m src.tools.reembed --collection cognitive_cycles --dry-run
    python -m src.tools.reembed --collection cognitive_cycles
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.providers.base import EmbeddingModelIdentity, read_collection_identity
from src.providers.execution_scheduler import ModelExecutionScheduler
from src.providers.factory import build_embedding_provider

logger = logging.getLogger(__name__)

MIGRATABLE_COLLECTIONS = ("cognitive_cycles", "conversation_summaries")
MAX_COLLECTION_NAME_LENGTH = 63


@dataclass
class MigrationReport:
    source: str
    target: str
    identity: str
    total: int = 0
    migrated: int = 0
    resumed: int = 0
    skipped_empty: int = 0
    failed: int = 0
    dry_run: bool = False
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        mode = "DRY RUN" if self.dry_run else "MIGRATED"
        return (
            f"[{mode}] {self.source} -> {self.target} as {self.identity}: "
            f"{self.migrated} written, {self.resumed} already present, "
            f"{self.skipped_empty} without documents, {self.failed} failed, {self.total} total"
        )


def target_collection_name(source: str, identity: EmbeddingModelIdentity) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{identity.provider}_{identity.model}".lower()).strip("_")
    name = f"{source}__{slug}"
    if len(name) > MAX_COLLECTION_NAME_LENGTH:
        name = name[:MAX_COLLECTION_NAME_LENGTH].rstrip("_")
    return name


def open_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False, is_persistent=True),
    )


def describe_collections(client: chromadb.ClientAPI) -> List[str]:
    lines = []
    for collection in client.list_collections():
        handle = client.get_collection(name=collection.name)
        identity = read_collection_identity(handle.metadata)
        count = handle.count()
        if identity:
            label = identity.describe()
        else:
            label = "untagged, empty" if count == 0 else "untagged, UNKNOWN vector space"
        lines.append(f"{handle.name}: {count} records, {label}")
    return lines


def _existing_ids(target: Any, ids: Sequence[str]) -> set:
    if not ids:
        return set()
    try:
        found = target.get(ids=list(ids), include=[])
    except Exception:
        return set()
    return set(found.get("ids") or [])


async def reembed_collection(
    client: Any,
    provider: Any,
    source_name: str,
    target_name: Optional[str] = None,
    batch_size: int = 32,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> MigrationReport:
    identity = await provider.verify()
    source = client.get_collection(name=source_name)
    resolved_target = target_name or target_collection_name(source_name, identity)

    stored = read_collection_identity(source.metadata)
    if stored and stored.provider == identity.provider and stored.model == identity.model:
        raise ValueError(
            f"Collection '{source_name}' already holds {identity.describe()} vectors. Nothing to migrate."
        )
    stored_label = stored.describe() if stored else "unknown"

    report = MigrationReport(
        source=source_name,
        target=resolved_target,
        identity=identity.describe(),
        dry_run=dry_run,
    )

    target = None
    if not dry_run:
        metadata = {
            key: value for key, value in identity.as_collection_metadata().items() if value is not None
        }
        metadata["migrated_from"] = source_name
        metadata["migrated_from_identity"] = stored_label
        target = client.get_or_create_collection(name=resolved_target, metadata=metadata)

    offset = 0
    while True:
        page_size = batch_size if limit is None else min(batch_size, limit - report.total)
        if page_size <= 0:
            break
        page = source.get(include=["documents", "metadatas"], limit=page_size, offset=offset)
        ids = page.get("ids") or []
        if not ids:
            break
        offset += len(ids)
        report.total += len(ids)

        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        already = _existing_ids(target, ids) if target is not None else set()

        pending_ids: List[str] = []
        pending_docs: List[str] = []
        pending_metadata: List[Any] = []
        for index, record_id in enumerate(ids):
            document = documents[index] if index < len(documents) else None
            if not document:
                report.skipped_empty += 1
                continue
            if record_id in already:
                report.resumed += 1
                continue
            pending_ids.append(record_id)
            pending_docs.append(document)
            pending_metadata.append(metadatas[index] if index < len(metadatas) else None)

        if not pending_ids:
            continue
        if dry_run:
            report.migrated += len(pending_ids)
            continue

        try:
            vectors = await provider.embed_batch(pending_docs)
            target.upsert(
                ids=pending_ids,
                embeddings=vectors,
                documents=pending_docs,
                metadatas=pending_metadata,
            )
            report.migrated += len(pending_ids)
            logger.info("Re-embedded %s/%s records from '%s'.", report.migrated, report.total, source_name)
        except Exception as error:
            report.failed += len(pending_ids)
            report.errors.append(f"batch at offset {offset - len(ids)}: {error}")
            logger.error("Failed to re-embed batch at offset %s: %s", offset - len(ids), error)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild Chroma vectors with the active embedding provider.")
    parser.add_argument("--collection", action="append", help="Source collection; repeatable. Defaults to the known memory collections.")
    parser.add_argument("--target", help="Explicit target name. Only valid with a single --collection.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Stop after this many source records; useful for a trial run.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without embedding or writing.")
    parser.add_argument("--list", action="store_true", help="Show collections with their embedding identity and exit.")
    return parser


async def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = open_client()

    if args.list:
        for line in describe_collections(client):
            print(line)
        return 0

    sources = args.collection or [
        name for name in MIGRATABLE_COLLECTIONS
        if name in {c.name for c in client.list_collections()}
    ]
    if not sources:
        print("No matching collections found.")
        return 1
    if args.target and len(sources) > 1:
        print("--target requires exactly one --collection.")
        return 2

    provider = build_embedding_provider(
        ModelExecutionScheduler(
            max_interactive=settings.OLLAMA_MAX_INTERACTIVE_REQUESTS,
            max_background=settings.OLLAMA_MAX_BACKGROUND_REQUESTS,
        )
    )

    exit_code = 0
    for source_name in sources:
        try:
            report = await reembed_collection(
                client=client,
                provider=provider,
                source_name=source_name,
                target_name=args.target,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        except Exception as error:
            print(f"{source_name}: {error}")
            exit_code = 1
            continue
        print(report.summary())
        for message in report.errors:
            print(f"  error: {message}")
        if report.failed:
            exit_code = 1

    if exit_code == 0 and not args.dry_run:
        print("\nSource collections were not modified. Point EMBEDDING_PROVIDER at the new")
        print("vector space only after retrieval quality has been compared.")
    return exit_code


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
