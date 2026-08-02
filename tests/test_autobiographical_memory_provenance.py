from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import chromadb
import pytest

from src.models.core_models import CognitiveCycle
from src.providers.base import EmbeddingModelIdentity
from src.services.autobiographical_memory_system import AutobiographicalMemorySystem


@pytest.mark.asyncio
async def test_consolidated_memories_use_explicit_embeddings_stable_ids_and_provenance():
    identity = EmbeddingModelIdentity(
        provider="ollama", model="embeddinggemma", vector_dimension=3
    )
    embeddings = SimpleNamespace(
        identity=identity,
        generate_embedding=AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )
    system = AutobiographicalMemorySystem(embedding_service=embeddings)
    system.episodic_collection = MagicMock()
    system.semantic_collection = MagicMock()
    user_id = uuid4()
    cycle = CognitiveCycle(
        user_id=user_id,
        user_input="I prefer diagrams",
        final_response="Understood",
    )

    first = await system.create_episodic_memory(
        cycle,
        "The user expressed a durable preference.",
        0.9,
        consolidation_job_id="job-1",
        generation_provider="ollama",
        generation_model="qwen",
    )
    retry = await system.create_episodic_memory(
        cycle,
        "The user expressed a durable preference.",
        0.9,
        consolidation_job_id="job-2",
        generation_provider="ollama",
        generation_model="qwen",
    )
    concept = await system.extract_semantic_memory(
        user_id=str(user_id),
        concept_name="prefers_diagrams",
        description="The user prefers visual explanations.",
        category="user_preference",
        source_episodes=[first.episode_id],
        source_cycle_ids=[str(cycle.cycle_id)],
        consolidation_job_id="job-2",
        generation_provider="ollama",
        generation_model="qwen",
    )

    assert first.episode_id == retry.episode_id
    assert concept.source_episode_ids == [first.episode_id]
    assert concept.source_cycle_ids == [str(cycle.cycle_id)]
    assert concept.embedding_provider == "ollama"
    assert concept.embedding_model == "embeddinggemma"
    assert system.episodic_collection.upsert.call_args.kwargs["embeddings"] == [[0.1, 0.2, 0.3]]
    semantic_call = system.semantic_collection.upsert.call_args.kwargs
    assert semantic_call["embeddings"] == [[0.1, 0.2, 0.3]]
    assert semantic_call["metadatas"][0]["consolidation_job_id"] == "job-2"


def test_chroma_filter_builder_uses_explicit_and_for_multiple_conditions():
    where = AutobiographicalMemorySystem._where_all(
        [{"user_id": "u"}, {"confidence": {"$gte": 0.5}}]
    )

    assert where == {
        "$and": [{"user_id": "u"}, {"confidence": {"$gte": 0.5}}]
    }


@pytest.mark.asyncio
async def test_semantic_memory_round_trip_uses_persistent_collection_contract(monkeypatch):
    identity = EmbeddingModelIdentity(
        provider="test", model="fixed", vector_dimension=3
    )
    embeddings = SimpleNamespace(
        identity=identity,
        generate_embedding=AsyncMock(return_value=[1.0, 0.0, 0.0]),
    )
    system = AutobiographicalMemorySystem(embedding_service=embeddings)
    monkeypatch.setattr(
        "src.services.autobiographical_memory_system.app_settings.CHROMA_COLLECTION_EPISODIC",
        f"test_episodic_{uuid4().hex}",
    )
    monkeypatch.setattr(
        "src.services.autobiographical_memory_system.app_settings.CHROMA_COLLECTION_SEMANTIC",
        f"test_semantic_{uuid4().hex}",
    )
    await system.connect(chromadb.EphemeralClient())
    user_id = str(uuid4())
    cycle_id = str(uuid4())
    stored = await system.extract_semantic_memory(
        user_id=user_id,
        concept_name="prefers_diagrams",
        description="The user prefers visual explanations.",
        category="user_preference",
        source_episodes=["episode-1"],
        source_cycle_ids=[cycle_id],
        consolidation_job_id="job-1",
        generation_provider="ollama",
        generation_model="qwen",
    )

    retrieved = await system.query_semantic_memories(
        user_id=user_id,
        query="How should I explain this?",
        category="user_preference",
        min_confidence=0.5,
    )

    assert [item.concept_id for item in retrieved] == [stored.concept_id]
    assert retrieved[0].source_cycle_ids == [cycle_id]
    assert retrieved[0].consolidation_job_id == "job-1"
    assert retrieved[0].embedding_provider == "test"
