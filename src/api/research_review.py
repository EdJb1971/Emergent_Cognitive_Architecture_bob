"""Authenticated waking inquiry review and calibration API."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.dependencies import get_api_key_user_id
from src.models.research_models import (
    CalibrationLabelRequest,
    InquiryActionRequest,
    InquiryApproveRequest,
    InquiryCandidate,
    InquiryDetail,
    InquiryListResponse,
    InquiryStatus,
    ResearchLedgerEvent,
    ResearchLedgerEventType,
    ResearchLedgerResponse,
    SourceQualityFeedbackRequest,
    WakingInquiryReviewOutcome,
)
from src.services.inquiry_review_service import InquiryReviewService
from src.services.research_calibration_ledger import ResearchCalibrationLedger


router = APIRouter(prefix="/api/research", tags=["research-governance"])


def _review_service(request: Request) -> InquiryReviewService:
    service = getattr(request.app.state, "inquiry_review_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inquiry review service is not initialized.",
        )
    return service


def _ledger(request: Request) -> ResearchCalibrationLedger:
    ledger = getattr(request.app.state, "research_calibration_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research calibration ledger is not initialized.",
        )
    return ledger


def _translate_domain_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.args[0])
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Research governance operation failed.",
    )


@router.get("/inquiries", response_model=InquiryListResponse)
async def list_inquiries(
    request: Request,
    statuses: Optional[list[InquiryStatus]] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    user_id: UUID = Depends(get_api_key_user_id),
) -> InquiryListResponse:
    inquiries = await _review_service(request).list_inquiries(
        user_id, statuses=statuses, limit=limit
    )
    return InquiryListResponse(inquiries=inquiries, count=len(inquiries))


@router.get("/inquiries/{inquiry_id}", response_model=InquiryDetail)
async def inspect_inquiry(
    inquiry_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> InquiryDetail:
    try:
        return await _review_service(request).inspect(user_id, inquiry_id)
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.post("/inquiries/{inquiry_id}/approve", response_model=WakingInquiryReviewOutcome)
async def approve_inquiry(
    inquiry_id: UUID,
    body: InquiryApproveRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> WakingInquiryReviewOutcome:
    try:
        return await _review_service(request).approve(user_id, inquiry_id, body)
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.post("/inquiries/{inquiry_id}/dismiss", response_model=InquiryCandidate)
async def dismiss_inquiry(
    inquiry_id: UUID,
    body: InquiryActionRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> InquiryCandidate:
    try:
        return await _review_service(request).dismiss(user_id, inquiry_id, body)
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.post("/inquiries/{inquiry_id}/retry", response_model=InquiryCandidate)
async def retry_inquiry(
    inquiry_id: UUID,
    body: InquiryActionRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> InquiryCandidate:
    try:
        return await _review_service(request).retry(user_id, inquiry_id, body)
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.post(
    "/inquiries/{inquiry_id}/source-feedback",
    response_model=ResearchLedgerEvent,
    status_code=status.HTTP_201_CREATED,
)
async def record_source_feedback(
    inquiry_id: UUID,
    body: SourceQualityFeedbackRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> ResearchLedgerEvent:
    try:
        return await _review_service(request).record_source_feedback(
            user_id, inquiry_id, body
        )
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.get("/ledger", response_model=ResearchLedgerResponse)
async def list_research_ledger(
    request: Request,
    event_types: Optional[list[ResearchLedgerEventType]] = Query(default=None),
    inquiry_id: Optional[UUID] = Query(default=None),
    assessment_id: Optional[UUID] = Query(default=None),
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: UUID = Depends(get_api_key_user_id),
) -> ResearchLedgerResponse:
    events = await _ledger(request).list_events(
        user_id,
        event_types=event_types,
        inquiry_id=inquiry_id,
        assessment_id=assessment_id,
        after_sequence=after_sequence,
        limit=limit,
    )
    return ResearchLedgerResponse(
        events=events,
        count=len(events),
        next_after_sequence=events[-1].sequence if len(events) == limit else None,
    )


@router.post(
    "/calibration/{assessment_id}/labels",
    response_model=ResearchLedgerEvent,
    status_code=status.HTTP_201_CREATED,
)
async def label_calibration_observation(
    assessment_id: UUID,
    body: CalibrationLabelRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> ResearchLedgerEvent:
    try:
        return await _review_service(request).record_calibration_label(
            user_id, assessment_id, body
        )
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.get("/calibration/summary", response_model=dict[str, Any])
async def get_calibration_summary(
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> dict[str, Any]:
    return await _ledger(request).calibration_summary(user_id)
