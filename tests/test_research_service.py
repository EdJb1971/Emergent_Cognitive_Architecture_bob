from typing import Optional

import pytest

from src.models.research_models import (
    EscalationDisposition,
    EscalationReason,
    ResearchClaim,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchRequest,
    ResearchSource,
)
from src.services.escalation_policy import EscalationPolicy
from src.services.research_service import DisabledResearchProvider, ResearchService


class RecordingResearchProvider:
    provider_name = "test-research"
    model_name: Optional[str] = "grounded-test-v1"

    def __init__(self) -> None:
        self.requests = []

    def is_available(self) -> bool:
        return True

    async def research(self, request: ResearchRequest) -> ResearchPacket:
        self.requests.append(request)
        return ResearchPacket(
            request_id=request.request_id,
            decision_id=request.decision_id,
            query=request.query,
            status=ResearchPacketStatus.COMPLETED,
            provider=self.provider_name,
            model=self.model_name,
            claims=[ResearchClaim(text="Reviewed claim", source_ids=["s1"], confidence=0.9)],
            sources=[ResearchSource(source_id="s1", title="Primary source", url="https://example.test")],
            confidence=0.9,
            context_policy=request.context_policy,
            context_chars=len(request.context_summary or ""),
        )


def test_normal_query_needs_no_research_even_when_provider_is_available():
    provider = RecordingResearchProvider()
    service = ResearchService(
        EscalationPolicy(research_enabled=True),
        provider,
    )

    decision = service.decide("Explain how a binary tree works.", source="test")

    assert decision.disposition == EscalationDisposition.NOT_REQUIRED
    assert decision.reasons == []


@pytest.mark.asyncio
async def test_llm_suggested_query_cannot_authorize_research_by_itself():
    provider = RecordingResearchProvider()
    service = ResearchService(EscalationPolicy(research_enabled=True), provider)

    outcome = await service.consider(
        "Explain how a binary tree works.",
        candidate_queries=["latest binary tree research"],
        source="discovery_agent",
    )

    assert outcome.decision.disposition == EscalationDisposition.NOT_REQUIRED
    assert outcome.packets == []
    assert provider.requests == []


@pytest.mark.asyncio
async def test_disabled_research_never_invokes_provider_for_explicit_request():
    provider = RecordingResearchProvider()
    service = ResearchService(EscalationPolicy(research_enabled=False), provider)

    outcome = await service.consider(
        "Please search the web for the latest release.",
        source="discovery_agent",
    )

    assert outcome.decision.disposition == EscalationDisposition.BLOCKED_DISABLED
    assert EscalationReason.EXPLICIT_RESEARCH_REQUEST in outcome.decision.reasons
    assert outcome.packets == []
    assert provider.requests == []


@pytest.mark.asyncio
async def test_local_only_mode_overrides_enabled_research_and_available_provider():
    provider = RecordingResearchProvider()
    service = ResearchService(
        EscalationPolicy(research_enabled=True, local_only=True),
        provider,
    )

    outcome = await service.consider(
        "Browse the internet for today's weather.",
        source="discovery_agent",
    )

    assert outcome.decision.disposition == EscalationDisposition.BLOCKED_LOCAL_ONLY
    assert provider.requests == []


@pytest.mark.asyncio
async def test_enabled_explicit_research_emits_structured_question_only_packet():
    provider = RecordingResearchProvider()
    service = ResearchService(
        EscalationPolicy(research_enabled=True),
        provider,
        max_queries=2,
        max_query_chars=80,
    )

    outcome = await service.consider(
        "Please research the current supported Python versions.",
        candidate_queries=[
            "current supported Python versions",
            "current supported Python versions",
            "latest Python release support schedule",
            "ignored third query",
        ],
        source="discovery_agent",
    )

    assert outcome.decision.disposition == EscalationDisposition.APPROVED
    assert len(outcome.packets) == 2
    assert len(provider.requests) == 2
    assert [request.query for request in provider.requests] == [
        "current supported Python versions",
        "latest Python release support schedule",
    ]
    assert all(request.context_summary is None for request in provider.requests)
    assert all(packet.context_chars == 0 for packet in outcome.packets)
    assert all(packet.latency_ms is not None and packet.latency_ms >= 0 for packet in outcome.packets)
    assert outcome.packets[0].claims[0].source_ids == ["s1"]


def test_policy_detects_local_signals_without_external_classification():
    service = ResearchService(
        EscalationPolicy(research_enabled=True, low_confidence_threshold=0.6),
        DisabledResearchProvider(),
    )

    decision = service.decide(
        "Who is the current director of the named institute?",
        source="meta_cognitive_monitor",
        confidence=0.4,
        named_fact_missing=True,
        metacognitive_gap=True,
    )

    assert decision.disposition == EscalationDisposition.BLOCKED_UNAVAILABLE
    assert decision.reasons == [
        EscalationReason.TIME_SENSITIVE,
        EscalationReason.LOW_CONFIDENCE,
        EscalationReason.NAMED_FACT_MISSING,
        EscalationReason.METACOGNITIVE_GAP,
    ]


def test_searching_local_memory_is_not_misclassified_as_external_research():
    service = ResearchService(EscalationPolicy(research_enabled=True), RecordingResearchProvider())

    decision = service.decide("Search my memory for the dentist appointment.", source="test")

    assert decision.disposition == EscalationDisposition.NOT_REQUIRED
