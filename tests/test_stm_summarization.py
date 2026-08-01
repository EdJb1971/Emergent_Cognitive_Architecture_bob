"""Current summary-manager resilience tests."""
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.services.summary_manager import SummaryManager


@pytest.mark.asyncio
async def test_summary_update_degrades_without_interrupting_a_cycle():
    llm_service = AsyncMock()
    llm_service.generate_text.side_effect = RuntimeError("provider unavailable")
    manager = SummaryManager(llm_service=llm_service)
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="Remember this context",
        final_response="I will retain the important parts.",
    )

    summary = await manager.update_summary(cycle.user_id, cycle)

    assert summary.user_id == cycle.user_id
    assert summary.update_count == 0
