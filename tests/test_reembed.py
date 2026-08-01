from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.base import EmbeddingModelIdentity
from src.tools.reembed import reembed_collection, target_collection_name

OLLAMA = EmbeddingModelIdentity(provider="ollama", model="embeddinggemma:latest", vector_dimension=768)


def _provider(identity=OLLAMA):
    provider = MagicMock()
    provider.verify = AsyncMock(return_value=identity)
    provider.embed_batch = AsyncMock(side_effect=lambda texts: [[0.5] * 768 for _ in texts])
    return provider


def _source(records, metadata=None):
    collection = MagicMock()
    collection.name = "cognitive_cycles"
    collection.metadata = metadata

    def get(include=None, limit=None, offset=0, ids=None):
        page = records[offset : offset + (limit or len(records))]
        return {
            "ids": [r["id"] for r in page],
            "documents": [r["document"] for r in page],
            "metadatas": [r["metadata"] for r in page],
        }

    collection.get.side_effect = get
    return collection


def _client(source, target_existing_ids=()):
    target = MagicMock()
    target.get.return_value = {"ids": list(target_existing_ids)}
    client = MagicMock()
    client.get_collection.return_value = source
    client.get_or_create_collection.return_value = target
    return client, target


def _records(count, start=0):
    return [
        {"id": f"id-{i}", "document": f"document {i}", "metadata": {"user_id": "u"}}
        for i in range(start, start + count)
    ]


def test_target_name_encodes_provider_and_model():
    assert target_collection_name("cognitive_cycles", OLLAMA) == "cognitive_cycles__ollama_embeddinggemma_latest"


def test_target_name_stays_within_chroma_limit():
    long_identity = EmbeddingModelIdentity(provider="ollama", model="a" * 90)
    assert len(target_collection_name("cognitive_cycles", long_identity)) <= 63


@pytest.mark.asyncio
async def test_migration_writes_new_vectors_without_touching_source():
    source = _source(_records(5))
    client, target = _client(source)

    report = await reembed_collection(client, _provider(), "cognitive_cycles", batch_size=2)

    assert report.migrated == 5
    assert report.total == 5
    assert report.failed == 0
    source.upsert.assert_not_called()
    source.modify.assert_not_called()
    assert target.upsert.call_count == 3


@pytest.mark.asyncio
async def test_target_is_stamped_with_active_identity_and_provenance():
    client, _ = _client(_source(_records(1)))

    await reembed_collection(client, _provider(), "cognitive_cycles")

    metadata = client.get_or_create_collection.call_args.kwargs["metadata"]
    assert metadata["embedding_provider"] == "ollama"
    assert metadata["embedding_model"] == "embeddinggemma:latest"
    assert metadata["migrated_from"] == "cognitive_cycles"
    assert metadata["migrated_from_identity"] == "unknown"


@pytest.mark.asyncio
async def test_migration_resumes_by_skipping_records_already_in_target():
    source = _source(_records(4))
    client, target = _client(source, target_existing_ids=["id-0", "id-1"])

    report = await reembed_collection(client, _provider(), "cognitive_cycles", batch_size=4)

    assert report.resumed == 2
    assert report.migrated == 2


@pytest.mark.asyncio
async def test_records_without_documents_are_reported_not_embedded():
    records = _records(2)
    records[0]["document"] = ""
    client, _ = _client(_source(records))
    provider = _provider()

    report = await reembed_collection(client, provider, "cognitive_cycles")

    assert report.skipped_empty == 1
    assert report.migrated == 1
    provider.embed_batch.assert_awaited_once_with(["document 1"])


@pytest.mark.asyncio
async def test_dry_run_never_creates_or_writes_a_target():
    client, _ = _client(_source(_records(3)))
    provider = _provider()

    report = await reembed_collection(client, provider, "cognitive_cycles", dry_run=True)

    assert report.migrated == 3
    assert report.dry_run is True
    client.get_or_create_collection.assert_not_called()
    provider.embed_batch.assert_not_called()


@pytest.mark.asyncio
async def test_migrating_into_the_same_vector_space_is_refused():
    source = _source(_records(1), metadata=OLLAMA.as_collection_metadata())
    client, _ = _client(source)

    with pytest.raises(ValueError, match="Nothing to migrate"):
        await reembed_collection(client, _provider(), "cognitive_cycles")


@pytest.mark.asyncio
async def test_batch_failure_is_counted_and_does_not_abort_the_run():
    source = _source(_records(4))
    client, _ = _client(source)
    provider = _provider()
    provider.embed_batch = AsyncMock(side_effect=[RuntimeError("ollama down"), [[0.5] * 768, [0.5] * 768]])

    report = await reembed_collection(client, provider, "cognitive_cycles", batch_size=2)

    assert report.failed == 2
    assert report.migrated == 2
    assert "ollama down" in report.errors[0]


@pytest.mark.asyncio
async def test_limit_stops_early_for_a_trial_run():
    client, _ = _client(_source(_records(10)))

    report = await reembed_collection(client, _provider(), "cognitive_cycles", batch_size=3, limit=4)

    assert report.total == 4
