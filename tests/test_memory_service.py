from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle, MemoryQueryRequest
from src.services.memory_service import MemoryService


@pytest.fixture
def mock_llm_service():
    service = AsyncMock()
    service.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return service


@pytest.fixture
def memory_service(mock_llm_service):
    service = MemoryService(llm_service=mock_llm_service)
    service.cycles_collection = MagicMock()
    service.summary_manager.update_summary = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_upsert_cycle_stores_cycle_in_stm_ltm_and_summary(memory_service, mock_llm_service):
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="Remember this",
        final_response="I will remember it.",
    )

    stored = await memory_service.upsert_cycle(cycle)

    assert stored is True
    assert cycle.user_input_embedding == [0.1, 0.2, 0.3]
    assert cycle.final_response_embedding == [0.1, 0.2, 0.3]
    assert memory_service._stm_cache[cycle.user_id].recent_cycles == [cycle]
    memory_service.summary_manager.update_summary.assert_awaited_once_with(cycle.user_id, cycle)
    memory_service.cycles_collection.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_query_memory_merges_ranked_stm_and_ltm_results(memory_service, mock_llm_service):
    user_id = uuid4()
    stm_cycle = CognitiveCycle(
        user_id=user_id,
        session_id=uuid4(),
        user_input="Local memory about project architecture",
        user_input_embedding=[1.0, 0.0, 0.0],
    )
    await memory_service.add_cycle(stm_cycle)

    ltm_cycle = CognitiveCycle(
        user_id=user_id,
        session_id=uuid4(),
        user_input="Persisted memory about project architecture",
    )
    memory_service.cycles_collection.query.return_value = {
        "metadatas": [[{"json_data": ltm_cycle.model_dump_json()}]],
        "distances": [[0.2]],
    }
    mock_llm_service.generate_embedding.return_value = [1.0, 0.0, 0.0]

    results = await memory_service.query_memory(
        MemoryQueryRequest(user_id=user_id, query_text="project architecture")
    )

    assert [cycle.cycle_id for cycle in results] == [stm_cycle.cycle_id, ltm_cycle.cycle_id]
    memory_service.cycles_collection.query.assert_called_once()


@pytest.mark.asyncio
async def test_query_memory_rejects_user_id_override_in_metadata_filters(memory_service):
    with pytest.raises(Exception, match="may not include user_id"):
        await memory_service.query_memory(
            MemoryQueryRequest(
                user_id=uuid4(),
                query_text="anything",
                metadata_filters={"user_id": "another-user"},
            )
        )
