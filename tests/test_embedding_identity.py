from unittest.mock import MagicMock

import pytest

from src.providers.base import EmbeddingIdentityMismatch, EmbeddingModelIdentity
from src.services.embedding_identity import apply_embedding_identity, resolve_embedding_identity

GEMINI = EmbeddingModelIdentity(provider="gemini", model="models/embedding-001", vector_dimension=768)
OLLAMA = EmbeddingModelIdentity(provider="ollama", model="embeddinggemma:latest", vector_dimension=768)


def _collection(name: str, metadata=None, count: int = 0):
    collection = MagicMock()
    collection.name = name
    collection.metadata = metadata
    collection.count.return_value = count
    return collection


def test_resolve_embedding_identity_reads_active_provider():
    provider = MagicMock()
    provider.identity = OLLAMA
    assert resolve_embedding_identity(provider) == OLLAMA


def test_resolve_embedding_identity_returns_none_for_untagged_provider():
    assert resolve_embedding_identity(object()) is None


def test_empty_collection_is_stamped_with_active_identity():
    collection = _collection("cognitive_cycles")

    apply_embedding_identity(collection, OLLAMA)

    metadata = collection.modify.call_args.kwargs["metadata"]
    assert metadata["embedding_provider"] == "ollama"
    assert metadata["embedding_model"] == "embeddinggemma:latest"
    assert metadata["embedding_dimension"] == 768


def test_matching_identity_is_accepted():
    collection = _collection("cognitive_cycles", metadata=OLLAMA.as_collection_metadata(), count=10)

    apply_embedding_identity(collection, OLLAMA)

    collection.modify.assert_not_called()


def test_same_dimension_different_model_is_rejected():
    """768 dimensions is shared by both models, so dimension alone cannot be the guard."""
    collection = _collection("cognitive_cycles", metadata=GEMINI.as_collection_metadata(), count=10)

    with pytest.raises(EmbeddingIdentityMismatch):
        apply_embedding_identity(collection, OLLAMA)


def test_populated_legacy_collection_is_rejected_for_a_new_provider():
    collection = _collection("cognitive_cycles", metadata=None, count=10)

    with pytest.raises(EmbeddingIdentityMismatch):
        apply_embedding_identity(collection, OLLAMA)


def test_populated_legacy_collection_is_accepted_for_the_legacy_provider():
    collection = _collection("cognitive_cycles", metadata=None, count=10)

    apply_embedding_identity(collection, GEMINI)

    collection.modify.assert_not_called()


def test_enforcement_can_be_disabled_for_recovery():
    collection = _collection("cognitive_cycles", metadata=GEMINI.as_collection_metadata(), count=10)

    apply_embedding_identity(collection, OLLAMA, enforced=False)


def test_unknown_identity_is_a_no_op():
    collection = _collection("cognitive_cycles")

    apply_embedding_identity(collection, None)

    collection.modify.assert_not_called()
