from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from src.providers.base import ProviderPurpose


ResultType = TypeVar("ResultType")


@dataclass(frozen=True)
class SchedulerSnapshot:
    active_interactive: int
    active_background: int
    max_interactive: int
    max_background: int


class ModelExecutionScheduler:
    """Bounds local inference so background work cannot crowd out chat cycles."""

    def __init__(self, max_interactive: int = 1, max_background: int = 1):
        self._interactive = asyncio.Semaphore(max_interactive)
        self._background = asyncio.Semaphore(max_background)
        self._active_interactive = 0
        self._active_background = 0
        self._max_interactive = max_interactive
        self._max_background = max_background
        self._lock = asyncio.Lock()

    async def execute(
        self,
        purpose: ProviderPurpose,
        operation: Callable[[], Awaitable[ResultType]],
    ) -> ResultType:
        semaphore = self._background if purpose == ProviderPurpose.BACKGROUND else self._interactive
        active_name = "_active_background" if purpose == ProviderPurpose.BACKGROUND else "_active_interactive"
        async with semaphore:
            async with self._lock:
                setattr(self, active_name, getattr(self, active_name) + 1)
            try:
                return await operation()
            finally:
                async with self._lock:
                    setattr(self, active_name, getattr(self, active_name) - 1)

    async def snapshot(self) -> SchedulerSnapshot:
        async with self._lock:
            return SchedulerSnapshot(
                active_interactive=self._active_interactive,
                active_background=self._active_background,
                max_interactive=self._max_interactive,
                max_background=self._max_background,
            )