"""Current ShortTermMemory token-budget behavior tests."""
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.models.memory_models import ShortTermMemory


@pytest.mark.asyncio
async def test_stm_requests_a_summary_when_token_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(ShortTermMemory, "_count_cycle_tokens", AsyncMock(return_value=8))
    user_id = uuid4()
    stm = ShortTermMemory(user_id=user_id, token_budget=10)
    first_cycle = CognitiveCycle(user_id=user_id, session_id=uuid4(), user_input="first")
    second_cycle = CognitiveCycle(user_id=user_id, session_id=uuid4(), user_input="second")

    assert await stm.add_cycle(first_cycle) == (False, None)
    needs_summary, cycles_to_summarize = await stm.add_cycle(second_cycle)

    assert needs_summary is True
    assert cycles_to_summarize == [first_cycle]
