"""Authenticated local identity and clean-start controls."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.dependencies import get_api_key_user_id
from src.models.identity_models import (
    CleanStartRequest,
    CleanStartStatus,
    IdentityProfile,
    IdentityUpdateRequest,
)
from src.services.clean_start_service import CleanStartService
from src.services.identity_service import IdentityConflictError, IdentityService


router = APIRouter(prefix="/api/settings", tags=["settings"])


def _identity(request: Request) -> IdentityService:
    service = getattr(request.app.state, "identity_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Identity settings are not initialized.")
    return service


def _clean_start(request: Request) -> CleanStartService:
    service = getattr(request.app.state, "clean_start_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Clean-start control is not initialized.")
    return service


@router.get("/identity", response_model=IdentityProfile)
async def get_identity(
    request: Request,
    _user_id: UUID = Depends(get_api_key_user_id),
) -> IdentityProfile:
    return _identity(request).current


@router.put("/identity", response_model=IdentityProfile)
async def update_identity(
    body: IdentityUpdateRequest,
    request: Request,
    _user_id: UUID = Depends(get_api_key_user_id),
) -> IdentityProfile:
    try:
        return await _identity(request).update(body)
    except IdentityConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/clean-start", response_model=CleanStartStatus)
async def clean_start_status(
    request: Request,
    _user_id: UUID = Depends(get_api_key_user_id),
) -> CleanStartStatus:
    return _clean_start(request).status()


@router.post("/clean-start", response_model=CleanStartStatus)
async def arm_clean_start(
    body: CleanStartRequest,
    request: Request,
    _user_id: UUID = Depends(get_api_key_user_id),
) -> CleanStartStatus:
    try:
        return _clean_start(request).arm(
            confirmation=body.confirmation,
            preserve_identity=body.preserve_identity,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("/clean-start", response_model=CleanStartStatus)
async def cancel_clean_start(
    request: Request,
    _user_id: UUID = Depends(get_api_key_user_id),
) -> CleanStartStatus:
    return _clean_start(request).cancel()
