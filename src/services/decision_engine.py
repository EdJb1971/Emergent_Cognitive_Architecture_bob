"""
Decision Engine

Rule-based decision engine to evaluate signals and enqueue autonomous tasks
(reflection, discovery, self-assessment, curiosity) based on configurable policies.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable

from src.core.config import settings
from src.services.background_task_queue import BackgroundTaskQueue

logger = logging.getLogger(__name__)


@dataclass
class TriggerPolicy:
    name: str
    threshold: Any
    cooldown_seconds: int = 300
    enabled: bool = True
    # evaluation function receives (user_id, signals) and returns bool
    evaluator: Optional[Callable[[str, Dict[str, Any]], bool]] = None


class DecisionEngine:
    """A small, rule-based decision engine.

    Responsibilities:
    - Receive signals (memory stats, agent outcomes)
    - Evaluate configured rules/policies
    - Enqueue tasks to BackgroundTaskQueue when rules fire
    - Enforce per-user cooldowns and de-duplication
    """

    def __init__(self, task_queue: BackgroundTaskQueue):
        self.task_queue = task_queue
        self.policies: Dict[str, TriggerPolicy] = {}
        # per-user last-fired times: {user_id: {policy_name: datetime}}
        self._last_fired: Dict[str, Dict[str, datetime]] = {}
        self._lock = asyncio.Lock()
        self._init_default_policies()
        logger.info("DecisionEngine initialized with default policies.")

    def _init_default_policies(self):
        # Enhanced reflection evaluator using flush_completed and query metrics
        def reflection_eval(user_id: str, signals: Dict[str, Any]) -> bool:
            cycles_trigger = signals.get("cycles_since_reflection", 0) >= int(getattr(settings, "REFLECTION_CYCLE_THRESHOLD", 10))
            # Trigger reflection after memory flushes to analyze consolidated context
            recent_flushes = signals.get("recent_flush_count", 0)
            flush_trigger = recent_flushes > int(getattr(settings, "REFLECTION_FLUSH_THRESHOLD", 2))
            return cycles_trigger or flush_trigger

        def discovery_eval(user_id: str, signals: Dict[str, Any]) -> bool:
            # Enhanced discovery using memory query metrics
            avg_sat = signals.get("avg_user_satisfaction", 1.0)
            unknown_count = signals.get("unknown_intent_count", 0)
            
            # Use query metrics to detect knowledge gaps
            query_metrics = signals.get("query_metrics", {})
            avg_relevance = query_metrics.get("avg_relevance", 1.0)
            miss_rate = query_metrics.get("miss_rate", 0.0)
            
            sat_threshold = float(getattr(settings, "DISCOVERY_SAT_THRESHOLD", 0.4))
            relevance_threshold = float(getattr(settings, "DISCOVERY_RELEVANCE_THRESHOLD", 0.5))
            miss_threshold = float(getattr(settings, "DISCOVERY_MISS_THRESHOLD", 0.3))
            
            return (
                (avg_sat < sat_threshold) or 
                (unknown_count >= int(getattr(settings, "DISCOVERY_UNKNOWN_INTENT_COUNT", 3))) or
                (avg_relevance < relevance_threshold) or
                (miss_rate > miss_threshold)
            )

        def self_assess_eval(user_id: str, signals: Dict[str, Any]) -> bool:
            # Enhanced self-assessment using memory performance metrics
            if signals.get("force_self_assess", False):
                return True
                
            # Trigger self-assessment if memory performance degrades
            query_metrics = signals.get("query_metrics", {})
            avg_latency = query_metrics.get("avg_latency", 0.0)
            error_rate = query_metrics.get("error_rate", 0.0)
            
            latency_threshold = float(getattr(settings, "SELF_ASSESS_LATENCY_THRESHOLD", 2.0))
            error_threshold = float(getattr(settings, "SELF_ASSESS_ERROR_THRESHOLD", 0.1))
            
            return (avg_latency > latency_threshold) or (error_rate > error_threshold)

        def curiosity_eval(user_id: str, signals: Dict[str, Any]) -> bool:
            # Enhanced curiosity using summary coverage and query patterns
            coverage = signals.get("summary_coverage", 1.0)
            coverage_threshold = float(getattr(settings, "CURIOSITY_COVERAGE_THRESHOLD", 0.5))
            
            # Analyze query patterns for potential exploration
            query_metrics = signals.get("query_metrics", {})
            topic_entropy = query_metrics.get("topic_entropy", 0.0)  # Measure of topic diversity
            novel_queries = query_metrics.get("novel_query_rate", 0.0)  # Rate of unique queries
            
            entropy_threshold = float(getattr(settings, "CURIOSITY_ENTROPY_THRESHOLD", 0.3))
            novelty_threshold = float(getattr(settings, "CURIOSITY_NOVELTY_THRESHOLD", 0.2))
            
            return (
                (coverage < coverage_threshold) or
                (topic_entropy < entropy_threshold) or
                (novel_queries < novelty_threshold)
            )

        self.policies["reflection"] = TriggerPolicy(
            name="reflection",
            threshold=None,
            cooldown_seconds=int(getattr(settings, "REFLECTION_COOLDOWN", 600)),
            evaluator=reflection_eval,
        )
        self.policies["discovery"] = TriggerPolicy(
            name="discovery",
            threshold=None,
            cooldown_seconds=int(getattr(settings, "DISCOVERY_COOLDOWN", 900)),
            evaluator=discovery_eval,
        )
        self.policies["self_assess"] = TriggerPolicy(
            name="self_assess",
            threshold=None,
            cooldown_seconds=int(getattr(settings, "SELF_ASSESS_COOLDOWN", 3600)),
            evaluator=self_assess_eval,
        )
        self.policies["curiosity"] = TriggerPolicy(
            name="curiosity",
            threshold=None,
            cooldown_seconds=int(getattr(settings, "CURIOSITY_COOLDOWN", 1200)),
            evaluator=curiosity_eval,
        )

    async def ingest_signals(self, user_id: str, signals: Dict[str, Any]):
        """Main entrypoint for pushing signals into the Decision Engine."""
        async with self._lock:
            for name, policy in self.policies.items():
                if not policy.enabled:
                    continue
                try:
                    should_fire = False
                    if policy.evaluator:
                        should_fire = policy.evaluator(user_id, signals)

                    if should_fire and not self._is_on_cooldown(user_id, name, policy.cooldown_seconds):
                        await self._fire_policy(user_id, name, signals)
                except Exception as e:
                    logger.exception(f"Error evaluating policy {name} for user {user_id}: {e}")

    def _is_on_cooldown(self, user_id: str, policy_name: str, cooldown_seconds: int) -> bool:
        user_map = self._last_fired.get(user_id, {})
        last = user_map.get(policy_name)
        if not last:
            return False
        return (datetime.utcnow() - last) < timedelta(seconds=cooldown_seconds)

    async def _fire_policy(self, user_id: str, policy_name: str, signals: Dict[str, Any]):
        """Enqueue a corresponding background task and record the firing time."""
        # Map policy to a task name / payload
        task_name = f"autonomous:{policy_name}"
        payload = {
            "user_id": user_id,
            "policy": policy_name,
            "triggered_at": datetime.utcnow().isoformat(),
            "signals": signals,
        }

        # De-duplicate via BackgroundTaskQueue (assumed support for idempotent tasks by name+user)
        try:
            await self.task_queue.enqueue(task_name, payload)
            # record last fired
            if user_id not in self._last_fired:
                self._last_fired[user_id] = {}
            self._last_fired[user_id][policy_name] = datetime.utcnow()
            logger.info(f"Fired policy {policy_name} for user {user_id} and enqueued task {task_name}.")
        except Exception as e:
            logger.exception(f"Failed to enqueue autonomous task for policy {policy_name}, user {user_id}: {e}")

    # Administrative helpers
    def enable_policy(self, name: str):
        if name in self.policies:
            self.policies[name].enabled = True

    def disable_policy(self, name: str):
        if name in self.policies:
            self.policies[name].enabled = False

    def set_evaluator(self, name: str, evaluator: Callable[[str, Dict[str, Any]], bool]):
        if name in self.policies:
            self.policies[name].evaluator = evaluator


# Lightweight factory
def create_decision_engine(task_queue: BackgroundTaskQueue) -> DecisionEngine:
    return DecisionEngine(task_queue)
