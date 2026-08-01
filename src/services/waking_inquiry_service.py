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
    ResearchLedgerEventType,
    WakingInquiryReviewOutcome,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.research_service import ResearchService
from src.services.research_calibration_ledger import ResearchCalibrationLedger


class WakingInquiryService:
    """Reassess locally, obtain approval, and execute through the guarded research seam."""

    def __init__(
        self,
        store: InquiryCandidateStore,
        drive: CognitiveResearchDrive,
        research_service: ResearchService,
        *,
        require_user_approval: bool = True,
        ledger: Optional[ResearchCalibrationLedger] = None,
    ) -> None:
        self.store = store
        self.drive = drive
        self.research_service = research_service
        self.require_user_approval = require_user_approval
        self.ledger = ledger

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
        if self.ledger:
            await self.ledger.record_assessment(
                fresh,
                user_id=user_id,
                inquiry_id=inquiry_id,
                event_type=ResearchLedgerEventType.WAKING_REVALIDATION,
            )

        if local_resolution:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.RESOLVED_LOCALLY,
                resolution=local_resolution[:2000],
            )
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESOLVED_LOCALLY,
                assessment=fresh,
                rationale="Waking cognition supplied a sufficient local resolution.",
            ))

        if fresh.recommended_action == CognitiveEffortAction.ROUTINE_LOCAL:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.RESOLVED_LOCALLY,
                resolution="Fresh waking assessment no longer justified external research.",
            )
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESOLVED_LOCALLY,
                assessment=fresh,
                rationale="The research drive fell below the local-effort threshold.",
            ))

        if fresh.effective_action != CognitiveEffortAction.AUTHORIZE_RESEARCH:
            candidate = await self.store.record_review(
                inquiry_id,
                user_id,
                fresh,
                InquiryStatus.QUEUED,
                resolution="Deferred by shadow mode, inhibition, or insufficient evidence.",
            )
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.DEFERRED,
                assessment=fresh,
                rationale="The active cognitive controller did not authorize provider contact.",
            ))

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
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.AWAITING_USER_APPROVAL,
                assessment=fresh,
                rationale="Offline inquiries require waking user approval before leaving the machine.",
            ))

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
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESEARCH_FAILED,
                assessment=fresh,
                rationale="Research failed safely; no external content was synthesized.",
            ))
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
            return await self._finalize(user_id, WakingInquiryReviewOutcome(
                candidate=candidate,
                disposition=InquiryReviewDisposition.RESEARCHED,
                assessment=fresh,
                research_outcome=research_outcome,
                rationale="Grounded, source-validated research completed.",
            ))

        candidate = await self.store.transition(
            inquiry_id,
            user_id,
            InquiryStatus.RESEARCH_FAILED,
            resolution=f"Research failed or was blocked ({research_outcome.decision.disposition.value}).",
        )
        return await self._finalize(user_id, WakingInquiryReviewOutcome(
            candidate=candidate,
            disposition=InquiryReviewDisposition.RESEARCH_FAILED,
            assessment=fresh,
            research_outcome=research_outcome,
            rationale="No valid grounded packet was available; no research content was synthesized.",
        ))

    async def _finalize(
        self,
        user_id: UUID,
        outcome: WakingInquiryReviewOutcome,
    ) -> WakingInquiryReviewOutcome:
        if not self.ledger:
            return outcome
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_RESOLVED,
            user_id=user_id,
            inquiry_id=outcome.candidate.inquiry_id,
            assessment_id=outcome.assessment.assessment_id,
            payload={
                "disposition": outcome.disposition.value,
                "status": outcome.candidate.status.value,
                "rationale": outcome.rationale,
                "resolution": outcome.candidate.resolution,
            },
        )
        if outcome.research_outcome:
            decision = outcome.research_outcome.decision
            await self.ledger.append(
                ResearchLedgerEventType.RESEARCH_DECISION,
                user_id=user_id,
                inquiry_id=outcome.candidate.inquiry_id,
                assessment_id=outcome.assessment.assessment_id,
                decision_id=decision.decision_id,
                payload={"decision": decision.model_dump(mode="json")},
            )
            for packet in outcome.research_outcome.packets:
                await self.ledger.record_packet(
                    packet,
                    user_id=user_id,
                    inquiry_id=outcome.candidate.inquiry_id,
                )
        return outcome
