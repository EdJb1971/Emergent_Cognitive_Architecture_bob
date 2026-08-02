"""Executive-control layer for bounded, explainable autonomous cognition."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID, uuid4

from src.models.autonomous_work_models import (
    AutonomousEventType,
    AutonomousProviderPolicy,
    AutonomousRuntimeState,
    AutonomousRuntimeUpdate,
    AutonomousTaskPolicy,
    AutonomousTaskRecord,
    AutonomousTaskRequest,
    AutonomousTaskStatus,
    AutonomousTaskType,
    utc_now,
)
from src.services.autonomous_work_store import AutonomousWorkStore


logger = logging.getLogger(__name__)
Handler = Callable[[AutonomousTaskRequest], Awaitable[Any]]
RuntimeCallback = Callable[[AutonomousRuntimeState], Awaitable[None]]


class AutonomousWorkGovernor:
    """Admits and supervises all non-foreground cognitive work under one policy."""

    def __init__(
        self,
        *,
        store: AutonomousWorkStore,
        policies: Dict[AutonomousTaskType, AutonomousTaskPolicy],
        master_enabled: bool = True,
        max_concurrent_global: int = 1,
        provider_is_local: Optional[Callable[[], bool]] = None,
    ) -> None:
        if max_concurrent_global < 1:
            raise ValueError("max_concurrent_global must be positive")
        missing = set(AutonomousTaskType) - set(policies)
        if missing:
            raise ValueError(f"missing autonomous policies: {sorted(item.value for item in missing)}")
        self.store = store
        self.policies = {key: value.model_copy(deep=True) for key, value in policies.items()}
        self.master_enabled = bool(master_enabled)
        self.max_concurrent_global = max_concurrent_global
        self.provider_is_local = provider_is_local or (lambda: True)
        self._handlers: Dict[AutonomousTaskType, Handler] = {}
        self._active: Dict[UUID, asyncio.Task] = {}
        self._requests: Dict[UUID, AutonomousTaskRequest] = {}
        self._running: set[UUID] = set()
        self._cancel_reasons: Dict[UUID, str] = {}
        self._admission_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_global)
        self._started = False
        self._changed_at = utc_now()
        self._runtime_callbacks: list[RuntimeCallback] = []

    async def start(self, system_user_id: UUID) -> None:
        if self._started:
            return
        persisted = await self.store.load_runtime()
        if persisted:
            self.master_enabled = bool(persisted.get("master_enabled", self.master_enabled))
            enabled = persisted.get("category_enabled", {})
            for task_type, value in enabled.items():
                if task_type in AutonomousTaskType._value2member_map_:
                    self.policies[AutonomousTaskType(task_type)].enabled = bool(value)
        self._started = True
        await self.store.append_event(
            AutonomousEventType.GOVERNOR_STARTED,
            user_id=system_user_id,
            payload={
                "master_enabled": self.master_enabled,
                "enabled_categories": [k.value for k, v in self.policies.items() if v.enabled],
                "max_concurrent_global": self.max_concurrent_global,
            },
        )

    async def shutdown(self, system_user_id: UUID) -> None:
        if not self._started:
            return
        tasks = list(self._active.values())
        for task_id, task in list(self._active.items()):
            if not task.done():
                self._cancel_reasons[task_id] = "application shutdown"
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active.clear()
        self._running.clear()
        await self.store.append_event(
            AutonomousEventType.GOVERNOR_STOPPED,
            user_id=system_user_id,
            payload={"shutdown_clean": True},
        )
        self._started = False

    def register_handler(self, task_type: AutonomousTaskType, handler: Handler) -> None:
        self._handlers[task_type] = handler

    def add_runtime_callback(self, callback: RuntimeCallback) -> None:
        self._runtime_callbacks.append(callback)

    def is_enabled(self, task_type: AutonomousTaskType) -> bool:
        return self.master_enabled and self.policies[task_type].enabled

    async def submit(
        self,
        request: AutonomousTaskRequest,
        *,
        executor: Optional[Handler] = None,
        wait: bool = False,
        bypass_cooldown: bool = False,
    ) -> AutonomousTaskRecord:
        if not self._started:
            return await self._reject(request, "governor_not_started")
        policy = self.policies[request.task_type]
        async with self._admission_lock:
            duplicate = await self.store.find_active_duplicate(
                request.user_id, request.task_type, request.deduplication_key
            )
            if duplicate:
                await self.store.append_event(
                    AutonomousEventType.TASK_DUPLICATE,
                    user_id=request.user_id,
                    task_id=duplicate.request.task_id,
                    task_type=request.task_type,
                    payload={
                        "duplicate_request_id": str(request.task_id),
                        "deduplication_key": request.deduplication_key,
                    },
                )
                return duplicate
            rejection = await self._admission_rejection(request, policy, bypass_cooldown)
            if rejection:
                return await self._reject(request, rejection)
            handler = executor or self._handlers.get(request.task_type)
            if handler is None:
                return await self._reject(request, "handler_not_registered")
            record = AutonomousTaskRecord(
                request=request,
                max_attempts=policy.max_retries + 1,
            )
            await self.store.save_task(record)
            await self.store.append_event(
                AutonomousEventType.TASK_QUEUED,
                user_id=request.user_id,
                task_id=request.task_id,
                task_type=request.task_type,
                payload={
                    "trigger_reason": request.trigger_reason,
                    "deduplication_key": request.deduplication_key,
                    "signals": request.signals,
                    "priority": request.priority,
                    "provider_policy": request.provider_policy.value,
                    "max_attempts": record.max_attempts,
                    "timeout_seconds": policy.timeout_seconds,
                },
            )
            task = asyncio.create_task(
                self._execute(record, handler, policy),
                name=f"autonomous-{request.task_type.value}-{request.task_id}",
            )
            self._active[request.task_id] = task
            self._requests[request.task_id] = request
        if wait:
            await asyncio.gather(task, return_exceptions=True)
            return await self.store.get_task(request.task_id) or record
        return record

    async def cancel(self, user_id: UUID, task_id: UUID, reason: str) -> AutonomousTaskRecord:
        record = await self.store.get_task(task_id)
        if record is None or record.request.user_id != user_id:
            raise KeyError("autonomous task not found")
        if record.status not in {AutonomousTaskStatus.QUEUED, AutonomousTaskStatus.RUNNING}:
            raise ValueError(f"task in {record.status.value} state cannot be cancelled")
        task = self._active.get(task_id)
        if task and not task.done():
            self._cancel_reasons[task_id] = f"operator cancellation: {reason}"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return await self.store.get_task(task_id) or record

    async def retry(self, user_id: UUID, task_id: UUID, reason: str) -> AutonomousTaskRecord:
        record = await self.store.get_task(task_id)
        if record is None or record.request.user_id != user_id:
            raise KeyError("autonomous task not found")
        if record.status not in {AutonomousTaskStatus.FAILED, AutonomousTaskStatus.CANCELLED}:
            raise ValueError("only failed or cancelled tasks can be retried")
        request = record.request.model_copy(
            update={
                "task_id": uuid4(),
                "trigger_reason": f"operator retry: {reason}",
                "deduplication_key": f"{record.request.deduplication_key}:retry:{utc_now().isoformat()}",
                "created_at": utc_now(),
            }
        )
        return await self.submit(request, bypass_cooldown=True)

    async def note_activity(
        self, user_id: UUID, *, reason: str = "foreground waking activity"
    ) -> int:
        """Foreground waking cognition preempts interruptible autonomous work."""
        cancelled = 0
        for task_id, task in list(self._active.items()):
            request = self._requests.get(task_id)
            if (
                request
                and request.user_id == user_id
                and self.policies[request.task_type].cancel_on_user_activity
                and not task.done()
            ):
                self._cancel_reasons[task_id] = reason
                task.cancel()
                cancelled += 1
        return cancelled

    def runtime_state(self) -> AutonomousRuntimeState:
        unfinished = {task_id for task_id, task in self._active.items() if not task.done()}
        running = unfinished & self._running
        return AutonomousRuntimeState(
            master_enabled=self.master_enabled,
            max_concurrent_global=self.max_concurrent_global,
            active_count=len(running),
            queued_count=len(unfinished - running),
            policies=self.policies,
            changed_at=self._changed_at,
        )

    async def update_runtime(
        self, user_id: UUID, update: AutonomousRuntimeUpdate
    ) -> AutonomousRuntimeState:
        if update.master_enabled is not None:
            self.master_enabled = update.master_enabled
        for task_type, enabled in update.category_enabled.items():
            self.policies[task_type].enabled = enabled
        self._changed_at = utc_now()
        if not self.master_enabled:
            await self.note_activity(user_id, reason="operator paused autonomous work")
        state = self.runtime_state()
        await self.store.save_runtime(
            {
                "master_enabled": state.master_enabled,
                "category_enabled": {
                    key.value: policy.enabled for key, policy in state.policies.items()
                },
                "changed_at": state.changed_at.isoformat(),
            }
        )
        await self.store.append_event(
            AutonomousEventType.RUNTIME_CHANGED,
            user_id=user_id,
            payload={
                "reason": update.reason,
                "master_enabled": state.master_enabled,
                "category_enabled": {
                    key.value: value.enabled for key, value in state.policies.items()
                },
            },
        )
        for callback in self._runtime_callbacks:
            await callback(state)
        return state

    async def _admission_rejection(self, request, policy, bypass_cooldown):
        if not self.master_enabled:
            return "master_disabled"
        if not policy.enabled:
            return "category_disabled"
        if request.provider_policy != policy.provider_policy:
            return "provider_policy_mismatch"
        if request.provider_policy == AutonomousProviderPolicy.LOCAL_ONLY and not self.provider_is_local():
            return "local_provider_required"
        active_for_user = sum(
            1 for task_id, task in self._active.items()
            if not task.done()
            and self._requests.get(task_id)
            and self._requests[task_id].user_id == request.user_id
            and self._requests[task_id].task_type == request.task_type
        )
        if active_for_user >= policy.max_concurrent_per_user:
            return "per_user_concurrency_limit"
        if await self.store.count_recent(request.user_id, request.task_type) >= policy.max_per_hour:
            return "hourly_rate_limit"
        if not bypass_cooldown and policy.cooldown_seconds:
            completed = await self.store.last_completed_at(request.user_id, request.task_type)
            if completed and utc_now() - completed < timedelta(seconds=policy.cooldown_seconds):
                return "cooldown"
        return None

    async def _reject(self, request, reason):
        record = AutonomousTaskRecord(
            request=request,
            status=AutonomousTaskStatus.REJECTED,
            completed_at=utc_now(),
            rejection_reason=reason,
            max_attempts=self.policies[request.task_type].max_retries + 1,
        )
        await self.store.save_task(record)
        await self.store.append_event(
            AutonomousEventType.TASK_REJECTED,
            user_id=request.user_id,
            task_id=request.task_id,
            task_type=request.task_type,
            payload={"reason": reason, "trigger_reason": request.trigger_reason},
        )
        return record

    async def _execute(self, record, handler, policy):
        request = record.request
        try:
            async with self._semaphore:
                self._running.add(request.task_id)
                for attempt in range(1, record.max_attempts + 1):
                    record.attempt = attempt
                    record.status = AutonomousTaskStatus.RUNNING
                    record.started_at = record.started_at or utc_now()
                    await self.store.save_task(record)
                    await self.store.append_event(
                        AutonomousEventType.TASK_STARTED,
                        user_id=request.user_id, task_id=request.task_id,
                        task_type=request.task_type, payload={"attempt": attempt},
                    )
                    try:
                        result = await asyncio.wait_for(handler(request), timeout=policy.timeout_seconds)
                        record.status = AutonomousTaskStatus.COMPLETED
                        record.completed_at = utc_now()
                        record.result = self._normalise_result(result)
                        record.error = None
                        await self.store.save_task(record)
                        await self.store.append_event(
                            AutonomousEventType.TASK_COMPLETED,
                            user_id=request.user_id, task_id=request.task_id,
                            task_type=request.task_type,
                            payload={"attempt": attempt, "result": record.result},
                        )
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        record.error = str(exc)[:2000]
                        if attempt < record.max_attempts:
                            await self.store.append_event(
                                AutonomousEventType.TASK_RETRYING,
                                user_id=request.user_id, task_id=request.task_id,
                                task_type=request.task_type,
                                payload={"attempt": attempt, "error": record.error},
                            )
                            continue
                        record.status = AutonomousTaskStatus.FAILED
                        record.completed_at = utc_now()
                        await self.store.save_task(record)
                        await self.store.append_event(
                            AutonomousEventType.TASK_FAILED,
                            user_id=request.user_id, task_id=request.task_id,
                            task_type=request.task_type,
                            payload={"attempt": attempt, "error": record.error},
                        )
        except asyncio.CancelledError:
            record.status = AutonomousTaskStatus.CANCELLED
            record.completed_at = utc_now()
            reason = self._cancel_reasons.pop(
                request.task_id, "cancelled by supervisor"
            )
            record.error = f"Cancelled: {reason}"
            await asyncio.shield(self.store.save_task(record))
            await asyncio.shield(self.store.append_event(
                AutonomousEventType.TASK_CANCELLED,
                user_id=request.user_id, task_id=request.task_id,
                task_type=request.task_type, payload={"reason": reason},
            ))
        finally:
            self._active.pop(request.task_id, None)
            self._requests.pop(request.task_id, None)
            self._running.discard(request.task_id)
            self._cancel_reasons.pop(request.task_id, None)

    @staticmethod
    def _normalise_result(result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return {"value": str(result)[:4000]}
