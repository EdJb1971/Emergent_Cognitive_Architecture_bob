from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.research_models import (
    EscalationReason,
    ResearchContextPolicy,
    ResearchPacketStatus,
    ResearchRequest,
)
from src.services.gemini_grounded_research_provider import GeminiGroundedResearchProvider


def _client_with(interaction):
    interactions = SimpleNamespace(create=AsyncMock(return_value=interaction))
    return SimpleNamespace(interactions=interactions)


@pytest.mark.asyncio
async def test_gemini_provider_extracts_annotated_claims_and_sources():
    answer = "Python 3.14 is the current feature release."
    interaction = {
        "steps": [
            {
                "type": "google_search_call",
                "arguments": {"queries": ["current Python feature release"]},
            },
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://www.python.org/downloads/",
                                "title": "Python downloads",
                                "start_index": 0,
                                "end_index": len(answer),
                            }
                        ],
                    }
                ],
            },
        ]
    }
    client = _client_with(interaction)
    provider = GeminiGroundedResearchProvider(
        api_key=None,
        model_name="models/gemini-3.5-flash-lite",
        client=client,
    )
    request = ResearchRequest(
        decision_id=uuid4(),
        query="What is the current Python feature release?",
        reasons=[EscalationReason.TIME_SENSITIVE],
    )

    packet = await provider.research(request)

    assert packet.status == ResearchPacketStatus.COMPLETED
    assert packet.grounding_verified is True
    assert packet.answer == answer
    assert packet.sources[0].url == "https://www.python.org/downloads/"
    assert packet.claims[0].source_ids == ["s1"]
    assert packet.search_queries == ["current Python feature release"]
    client.interactions.create.assert_awaited_once_with(
        model="gemini-3.5-flash-lite",
        input=request.query,
        tools=[{"type": "google_search"}],
        timeout=30.0,
    )
    assert request.context_policy == ResearchContextPolicy.QUESTION_ONLY


@pytest.mark.asyncio
async def test_gemini_provider_fails_closed_without_usable_url_annotations():
    client = _client_with(
        {
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "An uncited answer", "annotations": []}],
                }
            ]
        }
    )
    provider = GeminiGroundedResearchProvider(
        api_key=None,
        model_name="gemini-3.5-flash-lite",
        client=client,
    )
    request = ResearchRequest(
        decision_id=uuid4(),
        query="What changed today?",
        reasons=[EscalationReason.TIME_SENSITIVE],
    )

    packet = await provider.research(request)

    assert packet.status == ResearchPacketStatus.FAILED
    assert packet.grounding_verified is False
    assert packet.sources == []
    assert packet.claims == []


def test_gemini_provider_is_unavailable_without_key_or_injected_client():
    provider = GeminiGroundedResearchProvider(
        api_key=None,
        model_name="gemini-3.5-flash-lite",
    )

    assert provider.is_available() is False
