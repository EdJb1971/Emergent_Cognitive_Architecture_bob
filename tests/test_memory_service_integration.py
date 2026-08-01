"""Current STM integration checks for MemoryService."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_add_cycle_reports_flush_when_stm_budget_is_exceeded():
    llm_service = AsyncMock()
    service = MemoryService(llm_service=llm_service)
    service.STM_TOKEN_BUDGET = 10
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="A cycle that crosses the small test budget",
    )

    stm = MagicMock()
    stm.add_cycle = AsyncMock(return_value=(True, [cycle]))
    service._stm_cache[cycle.user_id] = stm

    should_flush, cycles_to_flush = await service.add_cycle(cycle)

    assert should_flush is True
    assert cycles_to_flush == [cycle]
    stm.add_cycle.assert_awaited_once_with(cycle)
