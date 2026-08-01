"""Integration-shaped tests for the current MemoryService query contract."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle, MemoryQueryRequest
from src.models.memory_models import ConversationSummary
from src.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_query_memory_reads_ltm_and_records_access_stats():
    user_id = uuid4()
    llm_service = AsyncMock()
    llm_service.generate_embedding.return_value = [1.0, 0.0, 0.0]
    service = MemoryService(llm_service=llm_service)
    service.summary_manager.get_or_create_summary = AsyncMock(
        return_value=ConversationSummary(user_id=user_id)
    )
    cycle = CognitiveCycle(
        user_id=user_id,
        session_id=uuid4(),
        user_input="Persisted memory",
    )
    service.cycles_collection = MagicMock()
    service.cycles_collection.query.return_value = {
        "metadatas": [[{"json_data": cycle.model_dump_json()}]],
        "distances": [[0.2]],
    }

    result = await service.query_memory(
        MemoryQueryRequest(user_id=user_id, query_text="Persisted memory")
    )

    assert [item.cycle_id for item in result] == [cycle.cycle_id]
    assert service._access_stats[user_id].ltm_hits == 1
    service.cycles_collection.query.assert_called_once()
