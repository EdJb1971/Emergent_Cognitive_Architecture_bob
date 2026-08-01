from unittest.mock import AsyncMock

import pytest

from src.services.meta_cognitive_monitor import (
    ActionRecommendation,
    GapType,
    MetaCognitiveMonitor,
)


@pytest.mark.asyncio
async def test_generate_uncertainty_response_caps_output_tokens_and_truncates_words():
    memory_service = AsyncMock()
    llm_service = AsyncMock()
    llm_service.generate_text = AsyncMock(
        return_value=" ".join(["word"] * 120)
    )

    monitor = MetaCognitiveMonitor(memory_service=memory_service, llm_service=llm_service)
    response = await monitor.generate_uncertainty_response(
        query="Can you explain a niche topic?",
        gap_type=GapType.TOPIC_UNKNOWN,
        recommendation=ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY,
    )

    assert len(response.split()) <= monitor.max_uncertainty_response_words
    assert llm_service.generate_text.await_args.kwargs["max_output_tokens"] == monitor.max_uncertainty_output_tokens


@pytest.mark.asyncio
async def test_generate_uncertainty_response_uses_fallback_on_generation_error():
    memory_service = AsyncMock()
    llm_service = AsyncMock()
    llm_service.generate_text = AsyncMock(side_effect=RuntimeError("boom"))

    monitor = MetaCognitiveMonitor(memory_service=memory_service, llm_service=llm_service)
    response = await monitor.generate_uncertainty_response(
        query="Need details",
        gap_type=GapType.KNOWLEDGE_SPARSE,
        recommendation=ActionRecommendation.SEARCH_FIRST,
    )

    assert response == "I'd like to search for the most current information on that topic."