"""Compatibility adapter for legacy callers of the autonomous-work governor."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Optional
from uuid import UUID

from src.models.autonomous_work_models import (
    AutonomousProviderPolicy,
    AutonomousTaskRequest,
    AutonomousTaskType,
)


logger = logging.getLogger(__name__)


class BackgroundTaskQueue:
    """Routes governed work centrally; retains a small fallback for isolated tests."""

    def __init__(self, governor=None):
        self._tasks: set[asyncio.Task] = set()
        self._orchestration_service = None
        self.governor = governor
        logger.info("BackgroundTaskQueue initialized.")

    def set_governor(self, governor) -> None:
        self.governor = governor

    def set_orchestration_service(self, orchestration_service):
        self._orchestration_service = orchestration_service

    def enqueue_task(
        self,
        coro: Coroutine[Any, Any, Any],
        task_name: str = "background_task",
        *,
        task_type: Optional[AutonomousTaskType] = None,
        user_id: Optional[UUID] = None,
        trigger_reason: Optional[str] = None,
        deduplication_key: Optional[str] = None,
        signals: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
    ):
        """Schedule a coroutine, using the governor when contract metadata is supplied."""
        if self.governor and task_type and user_id:
            async def submit():
                request = AutonomousTaskRequest(
                    user_id=user_id,
                    task_type=task_type,
                    trigger_reason=trigger_reason or task_name,
                    signals=signals or {},
                    payload=payload or {},
                    deduplication_key=deduplication_key or task_name,
                    provider_policy=AutonomousProviderPolicy.LOCAL_ONLY,
                )

                async def execute(_request):
                    return await coro

                record = await self.governor.submit(request, executor=execute)
                if record.request.task_id != request.task_id or record.status.value == "rejected":
                    coro.close()

            task = asyncio.create_task(submit(), name=f"governor-submit-{task_name}")
        else:
            task = asyncio.create_task(self._run_and_remove_task(coro, task_name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def enqueue(self, task_name: str, payload: Any):
        """Route legacy DecisionEngine names into the unified task contract."""
        if self.governor:
            mapping = {
                "autonomous:reflection": AutonomousTaskType.REFLECTION,
                "autonomous:discovery": AutonomousTaskType.DISCOVERY,
                "autonomous:self_assess": AutonomousTaskType.SELF_ASSESSMENT,
                "autonomous:curiosity": AutonomousTaskType.CURIOSITY,
            }
            task_type = mapping.get(task_name)
            if not task_type:
                logger.warning("Unknown autonomous task type: %s", task_name)
                return None
            user_id = UUID(str(payload["user_id"]))
            request = AutonomousTaskRequest(
                user_id=user_id,
                task_type=task_type,
                trigger_reason=f"decision policy: {payload.get('policy', task_type.value)}",
                signals=payload.get("signals", {}),
                payload=payload,
                deduplication_key=f"{task_type.value}:{user_id}",
            )
            return await self.governor.submit(request)

        if not self._orchestration_service:
            logger.error("Cannot route task: OrchestrationService not set")
            return None

        async def wrapper():
            user_id = payload.get("user_id")
            if task_name == "autonomous:reflection":
                await self._orchestration_service.trigger_reflection(user_id, 10, "autonomous")
            elif task_name == "autonomous:discovery":
                await self._orchestration_service.trigger_discovery(
                    user_id, "knowledge_gap", str(payload.get("signals", {}))
                )
            elif task_name == "autonomous:self_assess":
                await self._orchestration_service.trigger_reflection(user_id, 20, "self_assessment")
            elif task_name == "autonomous:curiosity":
                await self._orchestration_service.trigger_discovery(
                    user_id, "curiosity_exploration", str(payload.get("signals", {}))
                )

        return self.enqueue_task(wrapper(), task_name=task_name)

    async def _run_and_remove_task(self, coro, task_name):
        try:
            await coro
        except asyncio.CancelledError:
            logger.warning("Background task '%s' was cancelled.", task_name)
        except Exception:
            logger.exception("Background task '%s' failed.", task_name)

    async def shutdown(self):
        tasks = list(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
