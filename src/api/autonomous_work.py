"""Authenticated operator surface for autonomous cognitive work."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from src.dependencies import get_api_key_user_id
from src.models.autonomous_work_models import (
    AutonomousLedgerEvent,
    AutonomousRuntimeState,
    AutonomousRuntimeUpdate,
    AutonomousTaskRecord,
    AutonomousTaskStatus,
    AutonomousTaskType,
)


router = APIRouter(prefix="/api/autonomous-work", tags=["autonomous-work"])


class TaskAction(BaseModel):
    reason: str = Field(..., min_length=3, max_length=500)


def _governor(request: Request):
    service = getattr(request.app.state, "autonomous_work_governor", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Autonomous work governor is unavailable.")
    return service


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=error.args[0])
    if isinstance(error, ValueError):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=500, detail="Autonomous work operation failed.")


@router.get("/runtime", response_model=AutonomousRuntimeState)
async def get_runtime(
    request: Request, user_id: UUID = Depends(get_api_key_user_id)
) -> AutonomousRuntimeState:
    del user_id
    return _governor(request).runtime_state()


@router.put("/runtime", response_model=AutonomousRuntimeState)
async def update_runtime(
    body: AutonomousRuntimeUpdate,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> AutonomousRuntimeState:
    return await _governor(request).update_runtime(user_id, body)


@router.get("/tasks")
async def list_tasks(
    request: Request,
    statuses: Optional[list[AutonomousTaskStatus]] = Query(default=None),
    task_types: Optional[list[AutonomousTaskType]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: UUID = Depends(get_api_key_user_id),
):
    tasks = await _governor(request).store.list_tasks(
        user_id, statuses=statuses, task_types=task_types, limit=limit
    )
    return {"tasks": tasks, "count": len(tasks)}


@router.get("/tasks/{task_id}", response_model=AutonomousTaskRecord)
async def inspect_task(
    task_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> AutonomousTaskRecord:
    record = await _governor(request).store.get_task(task_id)
    if record is None or record.request.user_id != user_id:
        raise HTTPException(status_code=404, detail="autonomous task not found")
    return record


@router.post("/tasks/{task_id}/cancel", response_model=AutonomousTaskRecord)
async def cancel_task(
    task_id: UUID,
    body: TaskAction,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> AutonomousTaskRecord:
    try:
        return await _governor(request).cancel(user_id, task_id, body.reason)
    except (KeyError, ValueError) as error:
        raise _translate(error) from error


@router.post("/tasks/{task_id}/retry", response_model=AutonomousTaskRecord)
async def retry_task(
    task_id: UUID,
    body: TaskAction,
    request: Request,
    user_id: UUID = Depends(get_api_key_user_id),
) -> AutonomousTaskRecord:
    try:
        return await _governor(request).retry(user_id, task_id, body.reason)
    except (KeyError, ValueError) as error:
        raise _translate(error) from error


@router.get("/ledger")
async def list_ledger(
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: UUID = Depends(get_api_key_user_id),
):
    events = await _governor(request).store.list_events(
        user_id, after_sequence=after_sequence, limit=limit
    )
    return {
        "events": events,
        "count": len(events),
        "integrity_verified": await _governor(request).store.verify_integrity(),
    }

