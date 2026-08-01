"""Coordinates waking and offline inquiry creation without provider access."""

import logging
from datetime import timedelta
from typing import Optional, Sequence
from uuid import UUID

from src.models.research_models import (
    CognitiveEffortAction,
    CognitiveResearchAssessment,
    CognitiveResearchSignals,
    InquiryCandidate,
    InquirySourceType,
    utc_now,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.inquiry_candidate_store import InquiryCandidateStore

logger = logging.getLogger(__name__)


_QUEUE_ACTIONS = {
    CognitiveEffortAction.QUEUE_INQUIRY,
    CognitiveEffortAction.AUTHORIZE_RESEARCH,
}


class InquiryCandidateService:
    """Persist research needs while remaining incapable of executing research."""

    def __init__(
        self,
        store: InquiryCandidateStore,
        drive: CognitiveResearchDrive,
        *,
        enabled: bool = True,
        ttl_days: int = 14,
    ) -> None:
        if ttl_days < 1:
            raise ValueError("Inquiry candidate TTL must be at least one day.")
        self.store = store
        self.drive = drive
        self.enabled = enabled
        self.ttl_days = ttl_days

    async def propose_waking(
        self,
        *,
        user_id: UUID,
        question: str,
        assessment: CognitiveResearchAssessment,
        source_cycle_id: UUID,
        hypothesis: Optional[str] = None,
    ) -> Optional[InquiryCandidate]:
        if not self.enabled or assessment.recommended_action not in _QUEUE_ACTIONS:
            return None
        return await self._enqueue(
            user_id=user_id,
            question=question,
            hypothesis=hypothesis,
            assessment=assessment,
            source_type=InquirySourceType.WAKING,
            source_cycle_ids=[source_cycle_id],
        )

    async def propose_offline(
        self,
        *,
        user_id: UUID,
        question: str,
        signals: CognitiveResearchSignals,
        source_type: InquirySourceType,
        source_cycle_ids: Sequence[UUID] = (),
        source_pattern_ids: Sequence[UUID] = (),
        hypothesis: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Optional[InquiryCandidate]:
        if source_type == InquirySourceType.WAKING:
            raise ValueError("propose_offline requires reflection or dream source_type.")
        if not self.enabled:
            return None
        assessment = self.drive.assess(
            signals,
            source=f"offline_{source_type.value}",
            user_id=str(user_id),
        )
        if assessment.recommended_action not in _QUEUE_ACTIONS:
            return None
        return await self._enqueue(
            user_id=user_id,
            question=question,
            hypothesis=hypothesis,
            assessment=assessment,
            source_type=source_type,
            source_cycle_ids=source_cycle_ids,
            source_pattern_ids=source_pattern_ids,
            metadata=metadata,
        )

    async def _enqueue(
        self,
        *,
        user_id: UUID,
        question: str,
        assessment: CognitiveResearchAssessment,
        source_type: InquirySourceType,
        source_cycle_ids: Sequence[UUID],
        source_pattern_ids: Sequence[UUID] = (),
        hypothesis: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Optional[InquiryCandidate]:
        normalized_question = question.strip()[:1000]
        if not normalized_question:
            return None
        candidate = InquiryCandidate(
            user_id=user_id,
            question=normalized_question,
            hypothesis=hypothesis[:2000] if hypothesis else None,
            source_type=source_type,
            source_cycle_ids=list(source_cycle_ids),
            source_pattern_ids=list(source_pattern_ids),
            assessment=assessment,
            priority=assessment.drive_score,
            expected_information_gain=assessment.signals.expected_information_gain,
            shadow_mode=assessment.shadow_mode,
            expires_at=utc_now() + timedelta(days=self.ttl_days),
            metadata=metadata or {},
        )
        try:
            stored, _created = await self.store.enqueue(candidate)
            return stored
        except Exception as error:
            logger.warning(
                "Failed to persist inquiry candidate %s from %s: %s",
                candidate.inquiry_id,
                source_type.value,
                error,
            )
            return None
