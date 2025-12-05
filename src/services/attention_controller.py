"""Attention Controller Service

Thalamus/ACC-inspired controller that evaluates quick analysis + working memory
signals and emits directives to bias which agents fire and how much context they
receive. Initial implementation focuses on infrastructure and lightweight
heuristics so the controller can run in shadow mode today and evolve toward the
full Phase 7 design.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple, List

from src.models.agent_models import QuickAnalysis, AttentionDirective, ConflictReport

logger = logging.getLogger(__name__)


class AttentionController:
    """Computes excitatory/inhibitory directives for selective attention."""

    def __init__(self, enabled: bool = False, shadow_mode: bool = True):
        self.enabled = enabled
        # Shadow mode automatically enabled when controller itself is disabled
        self.shadow_mode = shadow_mode or not enabled
        self._last_context_by_user: Dict[str, Dict[str, Any]] = {}
        if not self.enabled:
            logger.info("AttentionController initialized (disabled)")
        elif self.shadow_mode:
            logger.info("AttentionController initialized in shadow mode")
        else:
            logger.info("AttentionController initialized (active control)")

    async def generate_directive(
        self,
        quick_analysis: QuickAnalysis,
        agent_activation: Dict[str, bool],
        working_memory_snapshot: Optional[Dict[str, Any]] = None,
        conflict_report: Optional[ConflictReport] = None,
        user_id: Optional[str] = None,
        stage: str = "pre_stage1",
        update_last_context: bool = False,
    ) -> AttentionDirective:
        """Create a directive describing how routing should be biased."""
        notes = []
        suppress_agents = []
        amplify_agents = []
        agent_bias: Dict[str, float] = {}
        memory_override: Optional[Dict[str, Any]] = None

        previous_context = self._last_context_by_user.get(user_id) if user_id else None
        drift_score, drift_reasons = self._compute_drift(previous_context, working_memory_snapshot)
        drift_detected = drift_score >= 0.5

        # Urgency/emotional heuristics
        if quick_analysis.urgency == "high":
            amplify_agents.extend(["emotional", "planning", "critic"])
            agent_bias.update({"emotional": 0.5, "critic": 0.2})
            notes.append("High urgency → ensure regulation + planning")
            memory_override = {
                "limit": 5,
                "min_relevance": 0.5,
                "description": "AttentionController: broaden context for urgent query",
            }

        # Simple/low urgency tasks: suppress heavy agents to save latency
        if quick_analysis.complexity == "simple" and quick_analysis.urgency == "low":
            suppress_agents.extend(["creative", "discovery", "critic"])
            agent_bias.update({"creative": -0.5, "discovery": -0.5})
            notes.append("Low urgency simple query → suppress exploratory agents")

        # Deep context need? ensure memory stays on
        if quick_analysis.context_need == "deep":
            amplify_agents.append("memory")
            agent_bias["memory"] = 0.4
            notes.append("Deep context requested → boost memory agent")

        # If working memory indicates emotional priority, enforce emotional agent
        if working_memory_snapshot and working_memory_snapshot.get("emotional_priority"):
            amplify_agents.append("emotional")
            agent_bias["emotional"] = max(agent_bias.get("emotional", 0.0), 0.6)
            notes.append("Working memory flagged emotional priority")

        # Conflict heuristics: if prior cycle had low coherence, keep critic on
        if conflict_report and conflict_report.coherence_score < 0.6:
            amplify_agents.append("critic")
            agent_bias["critic"] = max(agent_bias.get("critic", 0.0), 0.5)
            notes.append("Prior conflicts → keep critic engaged")

        if drift_detected:
            amplify_agents.extend(["memory", "planning"])
            agent_bias["memory"] = max(agent_bias.get("memory", 0.0), 0.5)
            agent_bias["planning"] = max(agent_bias.get("planning", 0.0), 0.3)
            if drift_score >= 0.7:
                amplify_agents.append("discovery")
                agent_bias["discovery"] = max(agent_bias.get("discovery", 0.0), 0.3)
            notes.append(
                f"Conversation drift detected ({drift_score:.2f}) → expand context + planning"
            )
        elif working_memory_snapshot and working_memory_snapshot.get("inferred_goals"):
            notes.append("Stable goals detected; maintaining baseline routing")

        # Guarantee baseline behaviors even if heuristics didn't add notes
        if not notes:
            notes.append("Heuristics yielded neutral directive")

        directive = AttentionDirective(
            shadow_mode=self.shadow_mode,
            agent_bias=agent_bias,
            suppress_agents=list(dict.fromkeys(suppress_agents)),
            amplify_agents=list(dict.fromkeys(amplify_agents)),
            memory_override=memory_override,
            notes=notes,
            drift_score=drift_score,
            drift_reasons=drift_reasons,
            stage=stage,
            user_id=user_id,
        )

        if update_last_context and working_memory_snapshot and user_id:
            self._last_context_by_user[user_id] = working_memory_snapshot

        return directive

    def should_apply(self, directive: AttentionDirective) -> bool:
        """Return True when directive should modify routing."""
        return self.enabled and not directive.shadow_mode

    def apply_directive(
        self,
        directive: AttentionDirective,
        agent_activation: Dict[str, bool],
        memory_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply directive to agent activation + memory config maps."""
        updated_activation = dict(agent_activation)

        for agent in directive.suppress_agents:
            updated_activation[agent] = False

        for agent in directive.amplify_agents:
            updated_activation[agent] = True

        for agent, bias in directive.agent_bias.items():
            if bias >= 0.25:
                updated_activation[agent] = True
            elif bias <= -0.25:
                updated_activation[agent] = False

        updated_memory = dict(memory_config)
        if directive.memory_override:
            updated_memory.update(directive.memory_override)

        return {
            "agent_activation": updated_activation,
            "memory_config": updated_memory,
        }

    def _compute_drift(
        self,
        previous_context: Optional[Dict[str, Any]],
        current_context: Optional[Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        if not previous_context or not current_context:
            return 0.0, []

        score = 0.0
        reasons: List[str] = []

        def _jaccard(a: List[str], b: List[str]) -> float:
            set_a = {item.lower() for item in a if isinstance(item, str)}
            set_b = {item.lower() for item in b if isinstance(item, str)}
            if not set_a and not set_b:
                return 1.0
            union = set_a | set_b
            if not union:
                return 1.0
            return len(set_a & set_b) / len(union)

        # Topics drift
        topic_similarity = _jaccard(
            previous_context.get("topics", []),
            current_context.get("topics", []),
        )
        if topic_similarity < 0.3:
            delta = (0.3 - topic_similarity) * 0.8
            score += delta
            reasons.append(
                f"Topic shift detected (similarity={topic_similarity:.2f})"
            )

        # Goals drift
        goal_similarity = _jaccard(
            previous_context.get("inferred_goals", []),
            current_context.get("inferred_goals", []),
        )
        if goal_similarity < 0.4:
            delta = (0.4 - goal_similarity) * 0.6
            score += delta
            reasons.append(
                f"Goal change detected (similarity={goal_similarity:.2f})"
            )

        # Attention focus drift
        focus_similarity = _jaccard(
            previous_context.get("attention_focus", []),
            current_context.get("attention_focus", []),
        )
        if focus_similarity < 0.4:
            delta = (0.4 - focus_similarity) * 0.4
            score += delta
            reasons.append(
                f"Attention focus shift (similarity={focus_similarity:.2f})"
            )

        # Sentiment shift
        prev_sent = previous_context.get("sentiment")
        curr_sent = current_context.get("sentiment")
        if prev_sent and curr_sent and prev_sent != curr_sent:
            score += 0.2
            reasons.append(
                f"Sentiment shift {prev_sent} → {curr_sent}"
            )

        # Bound score to [0,1]
        score = max(0.0, min(1.0, score))
        return score, reasons