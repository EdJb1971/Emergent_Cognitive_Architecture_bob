"""Tests for current CognitiveBrain memory integration."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.models.memory_models import ConversationSummary, MemoryAccessStats
from src.services.cognitive_brain import CognitiveBrain
from src.models.research_models import (
    ResearchClaim,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchSource,
)


@pytest.fixture
def memory_service():
    service = MagicMock()
    service.summary_manager.get_or_create_summary = AsyncMock(
        return_value=ConversationSummary(
            user_id=uuid4(),
            key_topics=["memory", "context"],
            context_points=["Testing memory integration"],
            conversation_state="active",
        )
    )
    service.query_memory = AsyncMock(return_value=[])
    service.get_access_stats = AsyncMock(
        return_value=MemoryAccessStats(stm_hits=1, ltm_hits=0, avg_relevance=0.8)
    )
    service.get_immediate_transcript.return_value = "User: Earlier context"
    return service


@pytest.fixture
def llm_service():
    service = AsyncMock()
    service.generate_text.return_value = """{
        "final_response": "Test response",
        "response_metadata": {
            "response_type": "informational",
            "tone": "neutral",
            "strategies": ["context_integration"],
            "cognitive_moves": ["provide_context"]
        },
        "outcome_signals": {
            "user_satisfaction_potential": 0.8,
            "engagement_potential": 0.7
        }
    }"""
    service.moderate_content.return_value = {"is_safe": True}
    return service


@pytest.mark.asyncio
async def test_response_generation_uses_memory_context(llm_service, memory_service):
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="How does memory work?",
    )
    brain = CognitiveBrain(llm_service=llm_service, memory_service=memory_service)

    response, metadata, signals = await brain.generate_response(cycle)

    prompt = llm_service.generate_text.call_args.kwargs["prompt"]
    assert response == "Test response"
    assert metadata.response_type == "informational"
    assert signals.user_satisfaction_potential == 0.9
    assert "Earlier context" in prompt
    assert "Topics: memory, context" in prompt
    memory_service.summary_manager.get_or_create_summary.assert_awaited_once_with(cycle.user_id)
    memory_service.query_memory.assert_awaited_once()
    memory_service.get_access_stats.assert_awaited_once_with(cycle.user_id)


@pytest.mark.asyncio
async def test_grounded_research_is_bounded_in_prompt_and_sources_are_deterministic(
    llm_service, memory_service
):
    cycle = CognitiveCycle(
        user_id=uuid4(),
        session_id=uuid4(),
        user_input="What is current?",
    )
    packet = ResearchPacket(
        request_id=uuid4(),
        decision_id=uuid4(),
        query=cycle.user_input,
        status=ResearchPacketStatus.COMPLETED,
        provider="grounded-test",
        answer="The reviewed release is current.",
        claims=[
            ResearchClaim(
                text="The reviewed release is current.",
                source_ids=["s1"],
                confidence=0.9,
            )
        ],
        sources=[
            ResearchSource(
                source_id="s1",
                title="Authoritative release page",
                url="https://example.test/release",
            )
        ],
        grounding_verified=True,
    )
    brain = CognitiveBrain(llm_service=llm_service, memory_service=memory_service)

    response, metadata, _signals = await brain.generate_response(
        cycle,
        research_packets=(packet,),
    )

    prompt = llm_service.generate_text.call_args.kwargs["prompt"]
    assert "Verified claim [R1]: The reviewed release is current." in prompt
    assert "https://example.test/release" in response
    assert "[R1]" in response
    assert "grounded_research" in metadata.strategies
