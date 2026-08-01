"""Fail-closed orchestration boundary for provider-neutral external research."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Protocol, Sequence

from src.models.research_models import (
    EscalationDecision,
    ResearchOutcome,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchRequest,
)
from src.services.escalation_policy import EscalationPolicy

logger = logging.getLogger(__name__)


class ResearchProvider(Protocol):
    provider_name: str
    model_name: Optional[str]

    def is_available(self) -> bool:
        ...

    async def research(self, request: ResearchRequest) -> ResearchPacket:
        ...


class DisabledResearchProvider:
    provider_name = "disabled"
    model_name = None

    def is_available(self) -> bool:
        return False

    async def research(self, request: ResearchRequest) -> ResearchPacket:
        raise RuntimeError("The disabled research provider cannot execute requests.")


class ResearchService:
    """Own policy, context minimisation, provider invocation, and audit output."""

    SERVICE_ID = "research_service"

    def __init__(
        self,
        policy: EscalationPolicy,
        provider: Optional[ResearchProvider] = None,
        *,
        max_queries: int = 3,
        max_query_chars: int = 500,
    ) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be at least 1.")
        if max_query_chars < 32:
            raise ValueError("max_query_chars must be at least 32.")
        self.policy = policy
        self.provider = provider or DisabledResearchProvider()
        self.max_queries = max_queries
        self.max_query_chars = max_query_chars

    def decide(
        self,
        query: str,
        *,
        source: str,
        confidence: Optional[float] = None,
        named_fact_missing: bool = False,
        metacognitive_gap: bool = False,
    ) -> EscalationDecision:
        decision = self.policy.evaluate(
            query,
            source=source,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            provider_available=self.provider.is_available(),
            confidence=confidence,
            named_fact_missing=named_fact_missing,
            metacognitive_gap=metacognitive_gap,
        )
        logger.info(
            "RESEARCH_DECISION %s",
            json.dumps(
                {
                    "decision_id": str(decision.decision_id),
                    "source": decision.source,
                    "disposition": decision.disposition.value,
                    "reasons": [reason.value for reason in decision.reasons],
                    "provider": decision.provider,
                    "model": decision.model,
                    "query_chars": decision.query_chars,
                    "estimated_query_tokens": decision.estimated_query_tokens,
                    "context_policy": decision.context_policy.value,
                    "decided_at": decision.decided_at.isoformat(),
                },
                sort_keys=True,
            ),
        )
        return decision

    async def consider(
        self,
        user_query: str,
        *,
        candidate_queries: Sequence[str] = (),
        source: str,
        confidence: Optional[float] = None,
        named_fact_missing: bool = False,
        metacognitive_gap: bool = False,
    ) -> ResearchOutcome:
        decision = self.decide(
            user_query,
            source=source,
            confidence=confidence,
            named_fact_missing=named_fact_missing,
            metacognitive_gap=metacognitive_gap,
        )
        if not decision.approved:
            return ResearchOutcome(decision=decision)

        queries = self._bounded_queries(candidate_queries or (user_query,))
        if not queries:
            queries = self._bounded_queries((user_query,))
        packets = []
        for query in queries:
            request = ResearchRequest(
                decision_id=decision.decision_id,
                query=query,
                reasons=decision.reasons,
                context_summary=None,
            )
            provider_started = time.perf_counter()
            try:
                packet = await self.provider.research(request)
                if not isinstance(packet, ResearchPacket):
                    packet = ResearchPacket.model_validate(packet)
                if packet.request_id != request.request_id or packet.decision_id != decision.decision_id:
                    raise ValueError("Research provider returned mismatched audit identifiers.")
            except Exception as error:
                logger.warning(
                    "Research provider %s failed for request %s: %s",
                    self.provider.provider_name,
                    request.request_id,
                    error,
                )
                packet = ResearchPacket(
                    request_id=request.request_id,
                    decision_id=decision.decision_id,
                    query=request.query,
                    status=ResearchPacketStatus.FAILED,
                    provider=self.provider.provider_name,
                    model=self.provider.model_name,
                    caveats=[f"Research provider failed: {type(error).__name__}"],
                    context_policy=request.context_policy,
                    context_chars=0,
                )
            packet = packet.model_copy(
                update={"latency_ms": (time.perf_counter() - provider_started) * 1000.0}
            )
            packets.append(packet)
            logger.info(
                "RESEARCH_PACKET request_id=%s decision_id=%s status=%s provider=%s "
                "sources=%d claims=%d context_chars=%d latency_ms=%.2f estimated_cost=%s",
                packet.request_id,
                packet.decision_id,
                packet.status.value,
                packet.provider,
                len(packet.sources),
                len(packet.claims),
                packet.context_chars,
                packet.latency_ms,
                packet.estimated_cost,
            )
        return ResearchOutcome(decision=decision, packets=packets)

    def _bounded_queries(self, candidates: Sequence[str]) -> list[str]:
        bounded = []
        seen = set()
        for candidate in candidates:
            query = str(candidate).strip()[: self.max_query_chars]
            key = query.casefold()
            if not query or key in seen:
                continue
            seen.add(key)
            bounded.append(query)
            if len(bounded) >= self.max_queries:
                break
        return bounded
