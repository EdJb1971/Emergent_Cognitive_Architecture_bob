"""Current SummaryManager behavior tests."""
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.services.summary_manager import SummaryManager


@pytest.mark.asyncio
async def test_summary_update_extracts_current_context():
    llm_service = AsyncMock()
    llm_service.generate_text.return_value = """{
        "user_name": "Ada",
        "ai_name": "Bob",
        "location": "London",
        "new_topics": ["memory"],
        "entities": ["ChromaDB"],
        "context_points": ["Ada is reviewing memory design"],
        "preferences": {"detail": "high"},
        "conversation_state": "deep_discussion"
    }"""
    llm_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
    manager = SummaryManager(llm_service=llm_service)
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="My name is Ada and I live in London. Explain memory.",
        final_response="Here is the memory design.",
    )

    summary = await manager.update_summary(cycle.user_id, cycle)

    assert "memory" in summary.key_topics
    assert "Ada" in summary.entities
    assert "ChromaDB" in summary.entities
    assert summary.conversation_state == "deep_discussion"
    assert summary.embedding == [0.1, 0.2, 0.3]
