"""Single-owner lifecycle and idle policy for sleep-like consolidation."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from uuid import UUID, uuid4

from src.models.sleep_models import SleepLedgerEventType
from src.services.memory_consolidation_service import MemoryConsolidationService
from src.services.metrics_service import MetricType, MetricsService
from src.services.sleep_cycle_ledger import SleepCycleLedger
from src.models.autonomous_work_models import (
    AutonomousTaskRequest,
    AutonomousTaskStatus,
    AutonomousTaskType,
)


logger = logging.getLogger(__name__)


class SleepCycleCoordinator:
    """Runs one bounded consolidation pipeline only after genuine user idle time."""

    DEFAULT_PIPELINE = (
        "episodic_to_semantic",
        "memory_replay",
        "pattern_extraction",
    )

    def __init__(
        self,
        *,
        consolidation_service: MemoryConsolidationService,
        ledger: SleepCycleLedger,
        user_ids: Iterable[UUID],
        enabled: bool = False,
        idle_seconds: float = 1800.0,
        check_interval_seconds: float = 60.0,
        max_cycles: int = 20,
        require_local_provider: bool = True,
        metrics_service: Optional[MetricsService] = None,
        autonomous_governor: Optional[Any] = None,
    ) -> None:
        if idle_seconds < 0:
            raise ValueError("idle_seconds cannot be negative")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        self.consolidation_service = consolidation_service
        self.ledger = ledger
        self.user_ids = tuple(dict.fromkeys(user_ids))
        self.enabled = bool(enabled)
        self.idle_seconds = float(idle_seconds)
        self.check_interval_seconds = float(check_interval_seconds)
        self.max_cycles = max_cycles
        self.require_local_provider = bool(require_local_provider)
        self.metrics_service = metrics_service
        self.autonomous_governor = autonomous_governor
        started = time.monotonic()
        self._last_activity = {user_id: started for user_id in self.user_ids}
        self._activity_epochs = {user_id: 0 for user_id in self.user_ids}
        self._loop_task: Optional[asyncio.Task] = None
        self._active_tasks: Dict[UUID, asyncio.Task] = {}
        self._stop_event = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._started = False

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def note_activity(self, user_id: UUID, *, at_monotonic: Optional[float] = None) -> None:
        """Wake signal: reset idle time and cancel that user's active sleep work."""
        self._last_activity[user_id] = time.monotonic() if at_monotonic is None else at_monotonic
        self._activity_epochs[user_id] = self._activity_epochs.get(user_id, 0) + 1
        active = self._active_tasks.get(user_id)
        if active and not active.done():
            active.cancel()
            logger.info("User activity cancelled active sleep cycle for %s", user_id)

    async def start(self) -> bool:
        """Start the sole scheduler task; disabled mode deliberately creates none."""
        if not self.enabled:
            logger.info("SleepCycleCoordinator disabled; no scheduler task started.")
            return False
        if self.running:
            return False
        self._stop_event.clear()
        for user_id in self.user_ids:
            await self.ledger.append(
                SleepLedgerEventType.COORDINATOR_STARTED,
                user_id=user_id,
                payload={
                    "idle_seconds": self.idle_seconds,
                    "check_interval_seconds": self.check_interval_seconds,
                    "max_cycles": self.max_cycles,
                    "require_local_provider": self.require_local_provider,
                },
            )
        self._started = True
        self._loop_task = asyncio.create_task(self._run_loop(), name="sleep-cycle-coordinator")
        return True

    async def shutdown(self) -> None:
        """Cancel active inference and stop promptly without leaving orphan tasks."""
        self._stop_event.set()
        active_tasks = list(self._active_tasks.values())
        for task in active_tasks:
            if not task.done():
                task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        self._loop_task = None
        self._active_tasks.clear()
        if self._started:
            for user_id in self.user_ids:
                await self.ledger.append(
                    SleepLedgerEventType.COORDINATOR_STOPPED,
                    user_id=user_id,
                    payload={"shutdown_clean": True},
                )
        self._started = False

    async def set_enabled(self, enabled: bool) -> None:
        """Apply an operator sleep toggle without requiring a process restart."""
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        if enabled:
            self.enabled = True
            await self.start()
        else:
            await self.shutdown()
            self.enabled = False

    async def run_once(
        self,
        user_id: UUID,
        *,
        now_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate gates once and, when admitted, execute one bounded sleep pipeline."""
        if not self.enabled:
            return {"status": "skipped", "reason": "disabled"}
        current = time.monotonic() if now_monotonic is None else now_monotonic
        observed_activity = self._last_activity.get(user_id, current)
        observed_epoch = self._activity_epochs.get(user_id, 0)
        idle_for = max(0.0, current - observed_activity)
        if idle_for < self.idle_seconds:
            return {"status": "skipped", "reason": "not_idle", "idle_seconds": idle_for}
        if self.require_local_provider and not self._provider_is_local():
            return await self._record_skip(user_id, "non_local_provider", idle_for)
        if not await self.consolidation_service.should_consolidate(str(user_id)):
            return {"status": "skipped", "reason": "cooldown", "idle_seconds": idle_for}
        if self._activity_epochs.get(user_id, 0) != observed_epoch:
            return {"status": "skipped", "reason": "user_activity", "idle_seconds": 0.0}
        if self._run_lock.locked() or any(not task.done() for task in self._active_tasks.values()):
            return {"status": "skipped", "reason": "already_running", "idle_seconds": idle_for}

        async with self._run_lock:
            run_id = uuid4()
            if self.autonomous_governor:
                request = AutonomousTaskRequest(
                    user_id=user_id,
                    task_type=AutonomousTaskType.SLEEP,
                    trigger_reason="idle consolidation threshold reached",
                    signals={"idle_seconds": idle_for, "max_cycles": self.max_cycles},
                    payload={"run_id": str(run_id)},
                    deduplication_key=f"sleep:{user_id}",
                    priority=0.65,
                )

                async def execute(_request):
                    result = await self._execute_pipeline(user_id, idle_for, run_id)
                    if result.get("status") == "failed":
                        raise RuntimeError(result.get("error", "sleep pipeline failed"))
                    if result.get("status") == "cancelled":
                        raise asyncio.CancelledError
                    return result

                record = await self.autonomous_governor.submit(
                    request, executor=execute, wait=True
                )
                if record.status == AutonomousTaskStatus.COMPLETED:
                    return record.result
                return {
                    "status": record.status.value,
                    "run_id": str(run_id),
                    "reason": record.rejection_reason or record.error,
                }
            task = asyncio.create_task(
                self._execute_pipeline(user_id, idle_for, run_id),
                name=f"sleep-cycle-{user_id}",
            )
            self._active_tasks[user_id] = task
            try:
                return await task
            except asyncio.CancelledError:
                payload = {"reason": "user_activity_or_shutdown_before_start"}
                await asyncio.shield(
                    self.ledger.append(
                        SleepLedgerEventType.CYCLE_CANCELLED,
                        user_id=user_id,
                        run_id=run_id,
                        payload=payload,
                    )
                )
                await asyncio.shield(
                    self._record_metric(user_id, run_id, "cancelled", payload)
                )
                return {"status": "cancelled", "run_id": str(run_id)}
            finally:
                self._active_tasks.pop(user_id, None)

    async def _execute_pipeline(
        self,
        user_id: UUID,
        idle_for: float,
        run_id: UUID,
    ) -> Dict[str, Any]:
        await self.ledger.append(
            SleepLedgerEventType.CYCLE_STARTED,
            user_id=user_id,
            run_id=run_id,
            payload={"idle_seconds": idle_for, "pipeline": list(self.DEFAULT_PIPELINE)},
        )
        try:
            candidates_by_stage: Dict[str, list[str]] = {}
            salience = None
            for consolidation_type in self.DEFAULT_PIPELINE:
                stage_cycle_ids, stage_salience = (
                    await self.consolidation_service.get_consolidation_candidates(
                        str(user_id),
                        consolidation_type=consolidation_type,
                        limit=self.max_cycles,
                    )
                )
                candidates_by_stage[consolidation_type] = stage_cycle_ids
                if salience is None and stage_salience is not None:
                    salience = stage_salience

            cycle_ids = list(
                dict.fromkeys(
                    cycle_id
                    for stage_ids in candidates_by_stage.values()
                    for cycle_id in stage_ids
                )
            )
            if not cycle_ids:
                return await self._record_skip(
                    user_id,
                    "no_candidates",
                    idle_for,
                    run_id=run_id,
                )

            jobs = []
            for consolidation_type in self.DEFAULT_PIPELINE:
                stage_cycle_ids = candidates_by_stage[consolidation_type]
                if not stage_cycle_ids:
                    continue
                job = await self.consolidation_service.create_consolidation_job(
                    user_id=str(user_id),
                    consolidation_type=consolidation_type,
                    cycle_ids=list(stage_cycle_ids),
                    priority=0.7,
                    run_id=run_id,
                    salience_advisory=salience,
                )
                job = await self.consolidation_service.execute_consolidation_job(
                    job.job_id,
                    run_id=run_id,
                )
                jobs.append(job)
                if job.status != "completed":
                    raise RuntimeError(
                        f"sleep job {job.job_id} ended with status {job.status}"
                    )

            self.consolidation_service.record_consolidation_completed(str(user_id))

            payload = {
                "cycle_count": len(cycle_ids),
                "job_ids": [job.job_id for job in jobs],
                "episodes_created": sum(job.episodes_created for job in jobs),
                "semantic_concepts_extracted": sum(
                    job.semantic_concepts_extracted for job in jobs
                ),
                "patterns_discovered": sum(len(job.patterns_discovered) for job in jobs),
            }
            await self.ledger.append(
                SleepLedgerEventType.CYCLE_COMPLETED,
                user_id=user_id,
                run_id=run_id,
                payload=payload,
            )
            await self._record_metric(user_id, run_id, "completed", payload)
            return {"status": "completed", "run_id": str(run_id), **payload}
        except asyncio.CancelledError:
            await asyncio.shield(
                self.ledger.append(
                    SleepLedgerEventType.CYCLE_CANCELLED,
                    user_id=user_id,
                    run_id=run_id,
                    payload={"reason": "user_activity_or_shutdown"},
                )
            )
            await asyncio.shield(
                self._record_metric(
                    user_id,
                    run_id,
                    "cancelled",
                    {"reason": "user_activity_or_shutdown"},
                )
            )
            return {"status": "cancelled", "run_id": str(run_id)}
        except Exception as exc:
            await self.ledger.append(
                SleepLedgerEventType.CYCLE_FAILED,
                user_id=user_id,
                run_id=run_id,
                payload={"error": str(exc)[:2000]},
            )
            await self._record_metric(
                user_id,
                run_id,
                "failed",
                {"error": str(exc)[:500]},
            )
            return {"status": "failed", "run_id": str(run_id), "error": str(exc)}

    async def _record_skip(
        self,
        user_id: UUID,
        reason: str,
        idle_for: float,
        *,
        run_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        await self.ledger.append(
            SleepLedgerEventType.CYCLE_SKIPPED,
            user_id=user_id,
            run_id=run_id,
            payload={"reason": reason, "idle_seconds": idle_for},
        )
        return {"status": "skipped", "reason": reason, "idle_seconds": idle_for}

    def _provider_is_local(self) -> bool:
        capabilities = getattr(self.consolidation_service.llm_service, "capabilities", None)
        return bool(capabilities and capabilities.is_local is True)

    async def _record_metric(
        self,
        user_id: UUID,
        run_id: UUID,
        status: str,
        payload: Dict[str, Any],
    ) -> None:
        if not self.metrics_service:
            return
        await self.metrics_service.record_metric(
            MetricType.SLEEP_CYCLE,
            {"status": status, "run_id": str(run_id), **payload},
            user_id=str(user_id),
        )

    async def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.check_interval_seconds,
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                for user_id in self.user_ids:
                    if self._stop_event.is_set():
                        break
                    await self.run_once(user_id)
        except asyncio.CancelledError:
            logger.info("SleepCycleCoordinator loop cancelled.")
