import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.agents.discovery_agent import DiscoveryAgent
from src.models.research_models import ResearchOutcome
from src.services.escalation_policy import EscalationPolicy
from src.services.research_service import DisabledResearchProvider, ResearchService


@pytest.mark.asyncio
async def test_discovery_routes_llm_suggestions_through_policy_using_original_query():
    llm_service = MagicMock()
    llm_service.generate_text = AsyncMock(
        return_value=json.dumps(
            {
                "knowledge_gaps": ["implementation detail"],
                "curiosities_generated": ["What alternatives exist?"],
                "proposed_explorations": ["Compare approaches"],
                "discovery_priority": 4,
                "potential_research_queries": ["latest external implementation research"],
            }
        )
    )
    memory_service = MagicMock()
    memory_service.summary_manager.get_or_create_summary = AsyncMock(
        return_value=SimpleNamespace(summary_text="Private summary that must stay local")
    )
    memory_service.query_memory = AsyncMock(return_value=[])
    research_service = MagicMock(spec=ResearchService)
    safe_service = ResearchService(EscalationPolicy(research_enabled=False), DisabledResearchProvider())
    research_service.consider = AsyncMock(
        return_value=ResearchOutcome(
            decision=safe_service.decide("Explain the implementation.", source="discovery_agent")
        )
    )
    agent = DiscoveryAgent(llm_service, memory_service, research_service)
    user_id = uuid4()

    output = await agent.process_input("Explain the implementation.", user_id=user_id)

    research_service.consider.assert_awaited_once_with(
        user_query="Explain the implementation.",
        candidate_queries=["latest external implementation research"],
        source="discovery_agent",
    )
    assert output.analysis["research"]["decision"]["disposition"] == "not_required"
    assert output.analysis["web_search_results"] == []


@pytest.mark.asyncio
async def test_discovery_accepts_legacy_suggestion_field_but_cannot_bypass_policy():
    llm_service = MagicMock()
    llm_service.generate_text = AsyncMock(
        return_value=json.dumps(
            {
                "knowledge_gaps": [],
                "curiosities_generated": [],
                "proposed_explorations": [],
                "discovery_priority": 1,
                "potential_web_searches": ["latest private suggestion"],
            }
        )
    )
    memory_service = MagicMock()
    memory_service.summary_manager.get_or_create_summary = AsyncMock(
        return_value=SimpleNamespace(summary_text="")
    )
    memory_service.query_memory = AsyncMock(return_value=[])
    research_service = ResearchService(
        EscalationPolicy(research_enabled=True),
        RecordingUnavailableProvider(),
    )
    agent = DiscoveryAgent(llm_service, memory_service, research_service)

    output = await agent.process_input("Explain recursion.", user_id=uuid4())

    assert output.analysis["research"]["decision"]["disposition"] == "not_required"
    assert research_service.provider.calls == 0


class RecordingUnavailableProvider:
    provider_name = "recording"
    model_name = "never-call"

    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    async def research(self, request):
        self.calls += 1
        raise AssertionError("Policy should not invoke provider for a normal user query")
