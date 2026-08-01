"""Waking review boundary for queued research inquiries."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from src.models.research_models import (
    CognitiveEffortAction,
    CognitiveResearchAssessment,
    CognitiveResearchSignals,
    InquiryReviewDisposition,
    InquiryStatus,
    ResearchPacketStatus,
    WakingInquiryReviewOutcome,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.research_service import ResearchService


class WakingInquiryService:
    """Reassess locally, obtain approval, and execute through the guarded research seam."""

    def __init__(
        self,
        store: InquiryCandidateStore,
        drive: CognitiveResearchDrive,
        research_service: ResearchService,
        *,
        require_user_approval: bool = True,
    ) -> None:
        self.store = store
        self.drive = drive
        self.research_service = research_service
        self.require_user_approval = require_user_approval

    async def review_candidate(
        self,
        *,
        user_id: UUID,
        inquiry_id: UUID,
        signals: Optional[CognitiveResearchSignals] = None,
        assessment: Optional[CognitiveResearchAssessment] = None,
        user_approved: bool = False,
        local_resolution: Optional[str] = None,
    ) -> WakingInquiryReviewOutcome:
        if (signals is None) == (assessment is None):
            raise ValueError("Provide exactly one of fresh signals or an existing waking assessment.")
        claimed = await self.store.transition(
            inquiry_id,
            user_id,
            InquiryStatus.UNDER_REVIEW,
            resolution="Claimed for waking revalidation.",
        )
        fresh = assessment or self.drive.assess(
            signals,
            source="waking_inquiry_revalidation",
            user_id=str(user_id),
        )

        if local_resolution:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.RESOLVED_LOCALLY,
                resolution=local_resolution[:2000],
            )
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESOLVED_LOCALLY,
                assessment=fresh,
                rationale="Waking cognition supplied a sufficient local resolution.",
            )

        if fresh.recommended_action == CognitiveEffortAction.ROUTINE_LOCAL:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.RESOLVED_LOCALLY,
                resolution="Fresh waking assessment no longer justified external research.",
            )
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESOLVED_LOCALLY,
                assessment=fresh,
                rationale="The research drive fell below the local-effort threshold.",
            )

        if fresh.effective_action != CognitiveEffortAction.AUTHORIZE_RESEARCH:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.QUEUED,
                resolution="Deferred by shadow mode, inhibition, or insufficient evidence.",
            )
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.DEFERRED,
                assessment=fresh,
                rationale="The active cognitive controller did not authorize provider contact.",
            )

        explicit_waking_request = (
            claimed.source_type.value == "waking" and fresh.signals.explicit_user_request
        )
        if self.require_user_approval and not (user_approved or explicit_waking_request):
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.QUEUED,
                resolution="Awaiting explicit user approval for external research.",
            )
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.AWAITING_USER_APPROVAL,
                assessment=fresh,
                rationale="Offline inquiries require waking user approval before leaving the machine.",
            )

        approved = await self.store.record_review(
            inquiry_id,
            user_id,
            fresh,
            InquiryStatus.APPROVED,
            resolution="Approved by waking cognitive governance.",
        )
        try:
            research_outcome = await self.research_service.consider(
                approved.question,
                source="waking_inquiry_review",
                cognitive_assessment=fresh,
                confidence=1.0 - fresh.signals.epistemic_uncertainty,
                named_fact_missing=fresh.signals.metacognitive_gap,
                metacognitive_gap=True,
            )
        except Exception as error:
            candidate = await self.store.transition(
                inquiry_id,
                user_id,
                InquiryStatus.RESEARCH_FAILED,
                resolution=f"Research boundary failed safely ({type(error).__name__}).",
            )
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESEARCH_FAILED,
                assessment=fresh,
                rationale="Research failed safely; no external content was synthesized.",
            )
        completed = [
            packet
            for packet in research_outcome.packets
            if packet.status == ResearchPacketStatus.COMPLETED and packet.grounding_verified
        ]
        if completed:
            candidate = await self.store.transition(
                inquiry_id,
                user_id,
                InquiryStatus.RESEARCHED,
                resolution=f"Grounded research completed ({research_outcome.decision.decision_id}).",
            )
            self.drive.record_research_execution(str(user_id))
            return WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESEARCHED,
                assessment=fresh,
                research_outcome=research_outcome,
                rationale="Grounded, source-validated research completed.",
            )

        candidate = await self.store.transition(
            inquiry_id,
            user_id,
            InquiryStatus.RESEARCH_FAILED,
            resolution=f"Research failed or was blocked ({research_outcome.decision.disposition.value}).",
        )
        return WakingInquiryReviewOutcome(
            candidate=candidate,
            disposition=InquiryReviewDisposition.RESEARCH_FAILED,
            assessment=fresh,
            research_outcome=research_outcome,
            rationale="No valid grounded packet was available; no research content was synthesized.",
        )
