import asyncio

import pytest

from src.providers.base import ProviderPurpose
from src.providers.execution_scheduler import ModelExecutionScheduler


@pytest.mark.asyncio
async def test_scheduler_keeps_interactive_and_background_limits_separate():
    scheduler = ModelExecutionScheduler(max_interactive=1, max_background=1)
    interactive_started = asyncio.Event()
    release_interactive = asyncio.Event()
    background_started = asyncio.Event()

    async def interactive_operation():
        interactive_started.set()
        await release_interactive.wait()
        return "interactive"

    async def background_operation():
        background_started.set()
        return "background"

    interactive_task = asyncio.create_task(
        scheduler.execute(ProviderPurpose.INTERACTIVE, interactive_operation)
    )
    await interactive_started.wait()
    background_result = await scheduler.execute(ProviderPurpose.BACKGROUND, background_operation)

    assert background_started.is_set()
    assert background_result == "background"
    snapshot = await scheduler.snapshot()
    assert snapshot.active_interactive == 1
    assert snapshot.active_background == 0

    release_interactive.set()
    assert await interactive_task == "interactive"
