"""Binds ChromaDB collections to the embedding model that produced their vectors."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.providers.base import (
    EmbeddingIdentityMismatch,
    EmbeddingModelIdentity,
    check_embedding_identity,
    read_collection_identity,
)

logger = logging.getLogger(__name__)


def resolve_embedding_identity(llm_service: Any) -> Optional[EmbeddingModelIdentity]:
    """Read the active embedding identity from whichever provider is wired in."""
    identity = getattr(llm_service, "identity", None)
    return identity if isinstance(identity, EmbeddingModelIdentity) else None


def _collection_is_empty(collection: Any) -> bool:
    try:
        return collection.count() == 0
    except Exception:  # a collection we cannot count is treated as populated
        return False


def apply_embedding_identity(
    collection: Any,
    identity: Optional[EmbeddingModelIdentity],
    *,
    enforced: bool = True,
) -> None:
    """Stamp identity onto a new collection, or refuse to write into a foreign vector space.

    Dimension is deliberately not used as the compatibility test: different providers
    frequently share one (Gemini embedding-001 and embeddinggemma are both 768).
    """
    if identity is None:
        return

    name = getattr(collection, "name", "<unknown>")
    stored_metadata = getattr(collection, "metadata", None)
    stored = read_collection_identity(stored_metadata)

    if stored is None:
        if not _collection_is_empty(collection):
            message = (
                f"Collection '{name}' holds vectors but carries no embedding identity, so the "
                f"model that produced them is unknown."
            )
            if enforced:
                raise EmbeddingIdentityMismatch(name, "unknown", identity.describe())
            logger.warning("%s Continuing because identity enforcement is disabled.", message)
            return

        metadata = {
            key: value
            for key, value in identity.as_collection_metadata().items()
            if value is not None
        }
        merged = {**(stored_metadata or {}), **metadata}
        try:
            collection.modify(metadata=merged)
        except Exception as error:
            logger.warning("Could not stamp embedding identity on '%s': %s", name, error)
            return
        logger.info("Stamped collection '%s' with embedding identity %s.", name, identity.describe())
        return

    mismatch = check_embedding_identity(name, stored_metadata, identity)
    if mismatch is None:
        return
    if enforced:
        raise mismatch
    logger.warning("%s Continuing because identity enforcement is disabled.", mismatch)
