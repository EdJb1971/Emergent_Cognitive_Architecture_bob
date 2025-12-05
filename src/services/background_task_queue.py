import logging
import asyncio
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)

class BackgroundTaskQueue:
    """
    A simple in-memory background task queue using asyncio.create_task.
    """
    def __init__(self):
        self._tasks = set() # To keep strong references to tasks
        logger.info("BackgroundTaskQueue initialized.")

    def enqueue_task(self, coro: Coroutine[Any, Any, Any], task_name: str = "background_task"):
        """
        Enqueues a coroutine to be run as a background task.
        The task is added to a set to prevent it from being garbage collected
        before completion. It removes itself from the set upon completion.
        """
        task = asyncio.create_task(self._run_and_remove_task(coro, task_name))
        self._tasks.add(task)
        logger.info(f"Enqueued background task: {task_name}")

    def __init__(self):
        self._tasks = set()  # To keep strong references to tasks
        self._orchestration_service = None  # Will be set via setter
        logger.info("BackgroundTaskQueue initialized.")

    def set_orchestration_service(self, orchestration_service):
        """Set the OrchestrationService instance for task routing."""
        self._orchestration_service = orchestration_service
        logger.info("OrchestrationService reference set in BackgroundTaskQueue")

    async def enqueue(self, task_name: str, payload: Any):
        """
        Routes autonomous tasks to appropriate OrchestrationService handlers.
        Supported task types:
        - autonomous:reflection -> trigger_reflection
        - autonomous:discovery -> trigger_discovery
        - autonomous:self_assess -> trigger_reflection with special context
        - autonomous:curiosity -> trigger_discovery with exploration context
        """
        if not self._orchestration_service:
            logger.error("Cannot route task: OrchestrationService not set")
            return

        async def _task_wrapper():
            try:
                logger.info(f"Processing autonomous task '{task_name}' with payload: {payload}")
                user_id = payload.get("user_id")
                
                if task_name == "autonomous:reflection":
                    await self._orchestration_service.trigger_reflection(
                        user_id=user_id,
                        num_cycles=10,  # Configurable via settings
                        trigger_type="autonomous"
                    )
                elif task_name == "autonomous:discovery":
                    await self._orchestration_service.trigger_discovery(
                        user_id=user_id,
                        discovery_type="knowledge_gap",
                        context=str(payload.get("signals", {}))
                    )
                elif task_name == "autonomous:self_assess":
                    await self._orchestration_service.trigger_reflection(
                        user_id=user_id,
                        num_cycles=20,  # Deeper reflection for self-assessment
                        trigger_type="self_assessment"
                    )
                elif task_name == "autonomous:curiosity":
                    await self._orchestration_service.trigger_discovery(
                        user_id=user_id,
                        discovery_type="curiosity_exploration",
                        context=str(payload.get("signals", {}))
                    )
                else:
                    logger.warning(f"Unknown autonomous task type: {task_name}")
                
                logger.info(f"Successfully processed autonomous task '{task_name}'")
            except Exception as e:
                logger.exception(f"Failed to process autonomous task '{task_name}': {e}")

        # Schedule the wrapper as a background task
        self.enqueue_task(_task_wrapper(), task_name=task_name)

    async def _run_and_remove_task(self, coro: Coroutine[Any, Any, Any], task_name: str):
        """
        Runs the given coroutine and removes the task from the internal set upon completion.
        """
        try:
            await coro
            logger.info(f"Background task '{task_name}' completed successfully.")
        except asyncio.CancelledError:
            logger.warning(f"Background task '{task_name}' was cancelled.")
        except Exception as e:
            logger.error(f"Background task '{task_name}' failed with an unhandled exception: {e}", exc_info=True)
        finally:
            # Ensure the task is removed from the set even if it fails or is cancelled
            current_task = asyncio.current_task()
            if current_task in self._tasks:
                self._tasks.remove(current_task)

    async def shutdown(self):
        """
        Attempts to gracefully shut down all running background tasks.
        """
        if not self._tasks:
            logger.info("No background tasks to shut down.")
            return
        
        logger.info(f"Attempting to shut down {len(self._tasks)} background tasks...")
        for task in list(self._tasks): # Iterate over a copy to allow modification of the original set
            task.cancel()
        
        # Wait for tasks to complete or be cancelled, with a timeout
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("All background tasks shut down.")
