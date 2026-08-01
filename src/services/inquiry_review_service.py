"""Human-facing inquiry review and calibration operations."""

from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

from src.models.research_models import (
    CalibrationLabelRequest,
    InquiryActionRequest,
    InquiryApproveRequest,
    InquiryCandidate,
    InquiryDetail,
    InquiryStatus,
    ResearchLedgerEvent,
    ResearchLedgerEventType,
    SourceQualityFeedbackRequest,
    WakingInquiryReviewOutcome,
)
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.waking_inquiry_service import WakingInquiryService


class InquiryReviewService:
    """Scoped facade for queue actions; all decisions become immutable events."""

    def __init__(
        self,
        store: InquiryCandidateStore,
        waking_service: WakingInquiryService,
        ledger: ResearchCalibrationLedger,
    ) -> None:
        self.store = store
        self.waking_service = waking_service
        self.ledger = ledger

    async def list_inquiries(
        self,
        user_id: UUID,
        *,
        statuses: Optional[Sequence[InquiryStatus]] = None,
        limit: int = 50,
    ) -> list[InquiryCandidate]:
        await self.store.expire_due()
        return await self.store.list_candidates(user_id, statuses=statuses, limit=limit)

    async def inspect(self, user_id: UUID, inquiry_id: UUID) -> InquiryDetail:
        candidate = await self._require_candidate(user_id, inquiry_id)
        events = await self.ledger.list_events(user_id, inquiry_id=inquiry_id, limit=500)
        return InquiryDetail(candidate=candidate, ledger_events=events)

    async def approve(
        self,
        user_id: UUID,
        inquiry_id: UUID,
        request: InquiryApproveRequest,
    ) -> WakingInquiryReviewOutcome:
        candidate = await self._require_candidate(user_id, inquiry_id)
        if candidate.status != InquiryStatus.QUEUED:
            raise ValueError("Only queued inquiries can enter waking approval review.")
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_REQUESTED,
            user_id=user_id,
            inquiry_id=inquiry_id,
            assessment_id=candidate.assessment.assessment_id,
            payload={"action": "approve", "reason": request.reason},
        )
        return await self.waking_service.review_candidate(
            user_id=user_id,
            inquiry_id=inquiry_id,
            signals=request.signals or candidate.assessment.signals,
            user_approved=True,
        )

    async def dismiss(
        self,
        user_id: UUID,
        inquiry_id: UUID,
        request: InquiryActionRequest,
    ) -> InquiryCandidate:
        candidate = await self._require_candidate(user_id, inquiry_id)
        if candidate.status not in {
            InquiryStatus.QUEUED,
            InquiryStatus.UNDER_REVIEW,
            InquiryStatus.APPROVED,
            InquiryStatus.RESEARCH_FAILED,
        }:
            raise ValueError("Only open inquiries can be dismissed.")
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_REQUESTED,
            user_id=user_id,
            inquiry_id=inquiry_id,
            payload={"action": "dismiss", "reason": request.reason},
        )
        dismissed = await self.store.transition(
            inquiry_id,
            user_id,
            InquiryStatus.DISMISSED,
            resolution=request.reason,
        )
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_RESOLVED,
            user_id=user_id,
            inquiry_id=inquiry_id,
            payload={
                "action": "dismiss",
                "status": dismissed.status.value,
                "resolution": dismissed.resolution,
            },
        )
        return dismissed

    async def retry(
        self,
        user_id: UUID,
        inquiry_id: UUID,
        request: InquiryActionRequest,
    ) -> InquiryCandidate:
        candidate = await self._require_candidate(user_id, inquiry_id)
        if candidate.status != InquiryStatus.RESEARCH_FAILED:
            raise ValueError("Only failed research inquiries can be re-queued for retry.")
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_REQUESTED,
            user_id=user_id,
            inquiry_id=inquiry_id,
            payload={"action": "retry", "reason": request.reason},
        )
        retried = await self.store.transition(
            inquiry_id,
            user_id,
            InquiryStatus.QUEUED,
            resolution=request.reason,
        )
        await self.ledger.append(
            ResearchLedgerEventType.REVIEW_RESOLVED,
            user_id=user_id,
            inquiry_id=inquiry_id,
            payload={
                "action": "retry",
                "status": retried.status.value,
                "resolution": retried.resolution,
            },
        )
        return retried

    async def record_source_feedback(
        self,
        user_id: UUID,
        inquiry_id: UUID,
        feedback: SourceQualityFeedbackRequest,
    ) -> ResearchLedgerEvent:
        await self._require_candidate(user_id, inquiry_id)
        return await self.ledger.append_source_feedback(
            user_id=user_id,
            inquiry_id=inquiry_id,
            feedback=feedback,
        )

    async def record_calibration_label(
        self,
        user_id: UUID,
        assessment_id: UUID,
        label: CalibrationLabelRequest,
    ) -> ResearchLedgerEvent:
        return await self.ledger.append_calibration_label(
            user_id=user_id,
            assessment_id=assessment_id,
            label=label,
        )

    async def _require_candidate(self, user_id: UUID, inquiry_id: UUID) -> InquiryCandidate:
        candidate = await self.store.get(inquiry_id, user_id)
        if candidate is None:
            raise KeyError(f"Inquiry candidate {inquiry_id} was not found.")
        return candidate
