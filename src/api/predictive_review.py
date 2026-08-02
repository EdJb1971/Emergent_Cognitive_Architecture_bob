"""Authenticated predictive-perception review and calibration API."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.dependencies import get_api_key_user_id
from src.models.predictive_models import (
    PredictiveAssessmentListResponse,
    PredictiveAssessmentReview,
    PredictiveCalibrationLabelRequest,
    PredictiveCalibrationSummary,
    PredictiveLedgerEvent,
    PredictiveLedgerEventType,
    PredictiveLedgerResponse,
    PredictiveReviewStatus,
)
from src.services.predictive_calibration_store import PredictiveCalibrationStore


router = APIRouter(prefix="/api/predictive", tags=["predictive-calibration"])


def _store(request: Request) -> PredictiveCalibrationStore:
    store = getattr(request.app.state, "predictive_calibration_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive calibration store is not initialized.",
        )
    return store


def _translate_domain_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.args[0])
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Predictive calibration operation failed.",
    )


@router.get("/assessments", response_model=PredictiveAssessmentListResponse)
async def list_predictive_assessments(
    request: Request,
    review_status: Optional[PredictiveReviewStatus] = Query(default=None),
    material_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    user_id: UUID = Depends(get_api_key_user_id),
) -> PredictiveAssessmentListResponse:
    assessments = await _store(request).list_assessments(
        user_id,
        review_status=review_status,
        material_only=material_only,
        limit=limit,
    )
    return PredictiveAssessmentListResponse(
        assessments=assessments,
        count=len(assessments),
    )


@router.get(
    "/assessments/{assessment_id}",
    response_model=PredictiveAssessmentReview,
)
async def inspect_predictive_assessment(
    assessment_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> PredictiveAssessmentReview:
    try:
        return await _store(request).get_assessment(user_id, assessment_id)
    except KeyError as error:
        raise _translate_domain_error(error) from error


@router.post(
    "/assessments/{assessment_id}/labels",
    response_model=PredictiveLedgerEvent,
    status_code=status.HTTP_201_CREATED,
)
async def label_predictive_assessment(
    assessment_id: UUID,
    body: PredictiveCalibrationLabelRequest,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> PredictiveLedgerEvent:
    try:
        return await _store(request).append_label(
            user_id=user_id,
            assessment_id=assessment_id,
            label=body,
        )
    except (KeyError, ValueError) as error:
        raise _translate_domain_error(error) from error


@router.get("/calibration/summary", response_model=PredictiveCalibrationSummary)
async def get_predictive_calibration_summary(
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> PredictiveCalibrationSummary:
    return await _store(request).calibration_summary(user_id)


@router.get("/ledger", response_model=PredictiveLedgerResponse)
async def list_predictive_ledger(
    request: Request,
    event_types: Optional[list[PredictiveLedgerEventType]] = Query(default=None),
    assessment_id: Optional[UUID] = Query(default=None),
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: UUID = Depends(get_api_key_user_id),
) -> PredictiveLedgerResponse:
    events = await _store(request).list_events(
        user_id,
        event_types=event_types,
        assessment_id=assessment_id,
        after_sequence=after_sequence,
        limit=limit,
    )
    return PredictiveLedgerResponse(
        events=events,
        count=len(events),
        next_after_sequence=events[-1].sequence if len(events) == limit else None,
    )
