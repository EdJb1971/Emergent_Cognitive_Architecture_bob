"""Current token-aware STM tests for MemoryService."""
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.models.memory_models import ShortTermMemory
from src.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_add_cycle_creates_a_per_user_token_aware_stm(monkeypatch):
    monkeypatch.setattr(ShortTermMemory, "_count_cycle_tokens", AsyncMock(return_value=1))
    service = MemoryService(llm_service=AsyncMock())
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="Test input",
        final_response="Test response",
    )

    should_flush, cycles_to_flush = await service.add_cycle(cycle)

    stm = service._stm_cache[cycle.user_id]
    assert should_flush is False
    assert cycles_to_flush is None
    assert stm.recent_cycles == [cycle]
    assert stm.token_count == 1
