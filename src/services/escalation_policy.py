"""Deterministic, local policy gate for any external research contact."""

import re
from typing import Optional

from src.models.research_models import (
    EscalationDecision,
    EscalationDisposition,
    EscalationReason,
)


_EXPLICIT_RESEARCH_PATTERNS = (
    re.compile(r"\b(?:search|browse)\s+(?:the\s+)?(?:web|internet|online)\b", re.IGNORECASE),
    re.compile(r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:search|browse|research|look\s+up)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+)?(?:search|browse|research|google|look\s+up)\b", re.IGNORECASE),
    re.compile(r"\blook\s+(?:this|that|it)\s+up\b", re.IGNORECASE),
)

_TIME_SENSITIVE_PATTERN = re.compile(
    r"\b(?:latest|current|currently|today|tonight|now|recent|recently|breaking|"
    r"news|weather|forecast|price|exchange\s+rate|score|standings|schedule|"
    r"release\s+date|current\s+version|as\s+of)\b",
    re.IGNORECASE,
)


class EscalationPolicy:
    """Classify research need without consulting an LLM or external service."""

    def __init__(
        self,
        *,
        research_enabled: bool = False,
        local_only: bool = False,
        low_confidence_threshold: float = 0.55,
    ) -> None:
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        self.research_enabled = research_enabled
        self.local_only = local_only
        self.low_confidence_threshold = low_confidence_threshold

    def evaluate(
        self,
        query: str,
        *,
        source: str,
        provider: str,
        model: Optional[str],
        provider_available: bool,
        confidence: Optional[float] = None,
        named_fact_missing: bool = False,
        metacognitive_gap: bool = False,
    ) -> EscalationDecision:
        normalized_query = query.strip()
        reasons = []

        if self._is_explicit_request(normalized_query):
            reasons.append(EscalationReason.EXPLICIT_RESEARCH_REQUEST)
        if _TIME_SENSITIVE_PATTERN.search(normalized_query):
            reasons.append(EscalationReason.TIME_SENSITIVE)
        if confidence is not None and confidence < self.low_confidence_threshold:
            reasons.append(EscalationReason.LOW_CONFIDENCE)
        if named_fact_missing:
            reasons.append(EscalationReason.NAMED_FACT_MISSING)
        if metacognitive_gap:
            reasons.append(EscalationReason.METACOGNITIVE_GAP)

        if not reasons:
            disposition = EscalationDisposition.NOT_REQUIRED
            rationale = "No configured escalation condition matched."
        elif self.local_only:
            disposition = EscalationDisposition.BLOCKED_LOCAL_ONLY
            rationale = "Research is prohibited while local-only mode is active."
        elif not self.research_enabled:
            disposition = EscalationDisposition.BLOCKED_DISABLED
            rationale = "Research need was detected, but research is disabled."
        elif not provider_available:
            disposition = EscalationDisposition.BLOCKED_UNAVAILABLE
            rationale = "Research is enabled, but no configured provider is available."
        else:
            disposition = EscalationDisposition.APPROVED
            rationale = "Research is allowed by the configured escalation policy."

        return EscalationDecision(
            source=source,
            disposition=disposition,
            reasons=reasons,
            rationale=rationale,
            research_enabled=self.research_enabled,
            local_only=self.local_only,
            provider_available=provider_available,
            provider=provider,
            model=model,
            query_chars=len(normalized_query),
            estimated_query_tokens=(len(normalized_query) + 3) // 4,
        )

    @staticmethod
    def _is_explicit_request(query: str) -> bool:
        if re.search(r"\b(?:memory|notes|conversation|chat history)\b", query, re.IGNORECASE):
            if re.search(r"\b(?:search|look\s+up)\b", query, re.IGNORECASE):
                return False
        return any(pattern.search(query) for pattern in _EXPLICIT_RESEARCH_PATTERNS)

