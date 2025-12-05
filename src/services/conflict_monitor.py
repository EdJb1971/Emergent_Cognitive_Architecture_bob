"""
Conflict Monitor Service - Coherence Detection & Adaptive Control

Inspired by the Anterior Cingulate Cortex (ACC), which detects conflicts between
competing responses and signals when cognitive control adjustments are needed.

This service identifies inconsistencies in agent outputs and triggers meta-cognitive
adjustments to ensure coherent, integrated responses.
"""

from src.models.agent_models import Conflict, ConflictReport
from src.services.reinforcement_learning_service import (
    ReinforcementLearningService,
    ContextTypes,
    StrategyTypes,
)
from src.models.core_models import AgentOutput, ErrorAnalysis
from typing import List, Optional, Dict, Any
from uuid import UUID


class ConflictMonitor:
    """
    Detects inconsistencies in agent outputs and triggers meta-cognitive adjustment.
    Acts like the ACC in monitoring for conflicts and signaling need for control adjustments.
    """
    
    def __init__(self, rl_service: Optional[ReinforcementLearningService] = None):
        self.conflict_thresholds = {
            "sentiment_coherence_mismatch": 0.3,
            "memory_planning_disconnect": 0.5,
            "creative_critic_divergence": 0.6,
            "emotional_logic_conflict": 0.4
        }
        # Optional reinforcement learning service for strategy selection
        self.rl_service = rl_service
    
    async def detect_conflicts(self, agent_outputs: List[AgentOutput]) -> ConflictReport:
        """
        Analyze agent outputs for conflicts and inconsistencies.
        
        Args:
            agent_outputs: List of outputs from all activated agents
            
        Returns:
            ConflictReport with detected conflicts and recommended actions
        """
        conflicts = []
        
        # Build agent lookup for easier access
        agents_by_id = {output.agent_id: output for output in agent_outputs}
        
        # Check for sentiment-coherence conflicts
        sentiment_conflict = self._check_sentiment_coherence_conflict(agents_by_id)
        if sentiment_conflict:
            conflicts.append(sentiment_conflict)
        
        # Check for memory-planning conflicts
        memory_planning_conflict = self._check_memory_planning_conflict(agents_by_id)
        if memory_planning_conflict:
            conflicts.append(memory_planning_conflict)
        
        # Check for creative-critic conflicts (often productive!)
        creative_critic_conflict = self._check_creative_critic_conflict(agents_by_id)
        if creative_critic_conflict:
            conflicts.append(creative_critic_conflict)
        
        # Check for emotional-logic conflicts
        emotional_logic_conflict = self._check_emotional_logic_conflict(agents_by_id)
        if emotional_logic_conflict:
            conflicts.append(emotional_logic_conflict)
        
        # Check for perception-memory conflicts
        perception_memory_conflict = self._check_perception_memory_conflict(agents_by_id)
        if perception_memory_conflict:
            conflicts.append(perception_memory_conflict)
        
        # Determine if adjustment is needed
        requires_adjustment = any(c.severity == "high" for c in conflicts)
        
        # Calculate overall coherence score
        coherence_score = self._calculate_coherence_score(conflicts, len(agent_outputs))
        
        # If RL service is available, enrich conflicts with RL-selected strategies
        if self.rl_service and conflicts:
            conflict_context_map = {
                "sentiment_coherence_mismatch": (ContextTypes.EMOTIONAL_VS_TECHNICAL, [
                    StrategyTypes.PRIORITIZE_EMOTIONAL,
                    StrategyTypes.PRIORITIZE_TECHNICAL,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.WEIGHTED_SYNTHESIS
                ]),
                "memory_planning_disconnect": (ContextTypes.DETAILED_VS_OVERVIEW, [
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ]),
                "creative_critic_divergence": (ContextTypes.CREATIVE_VS_FACTUAL, [
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ]),
                "emotional_logic_conflict": (ContextTypes.EMPATHY_VS_ACCURACY, [
                    StrategyTypes.PRIORITIZE_EMOTIONAL,
                    StrategyTypes.PRIORITIZE_TECHNICAL,
                    StrategyTypes.EMPATHY_THEN_FACTS,
                    StrategyTypes.FACTS_THEN_EMPATHY,
                    StrategyTypes.BLEND_BOTH
                ]),
                "perception_memory_conflict": (ContextTypes.DETAILED_VS_OVERVIEW, [
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ])
            }
            enriched_conflicts = []
            for c in conflicts:
                mapping = conflict_context_map.get(c.conflict_type)
                if not mapping:
                    enriched_conflicts.append(c)
                    continue
                ctx, strategies = mapping
                try:
                    selected = await self.rl_service.select_strategy(
                        context=ctx,
                        available_strategies=strategies
                    )
                    # Inject RL info into details without changing schema
                    details = c.details or {}
                    details.update({
                        "rl_context": ctx,
                        "candidate_strategies": strategies,
                        "rl_selected_strategy": selected
                    })
                    enriched_conflicts.append(Conflict(
                        conflict_type=c.conflict_type,
                        agents=c.agents,
                        severity=c.severity,
                        resolution_strategy=c.resolution_strategy,
                        details=details
                    ))
                except Exception as e:
                    # Graceful degrade if RL selection fails
                    details = c.details or {}
                    details.update({"rl_error": str(e)})
                    enriched_conflicts.append(Conflict(
                        conflict_type=c.conflict_type,
                        agents=c.agents,
                        severity=c.severity,
                        resolution_strategy=c.resolution_strategy,
                        details=details
                    ))
            conflicts = enriched_conflicts

        return ConflictReport(
            conflicts=conflicts,
            requires_adjustment=requires_adjustment,
            coherence_score=coherence_score
        )
    
    def _find_agent_output(self, agents_by_id: Dict[str, AgentOutput], agent_id: str) -> Optional[AgentOutput]:
        """Helper to safely find agent output."""
        return agents_by_id.get(agent_id)
    
    def _check_sentiment_coherence_conflict(self, agents_by_id: Dict[str, AgentOutput]) -> Optional[Conflict]:
        """
        Check if emotional sentiment conflicts with logical coherence assessment.
        Example: User is positive, but critic finds response incoherent.
        """
        emotional = self._find_agent_output(agents_by_id, "emotional_agent")
        critic = self._find_agent_output(agents_by_id, "critic_agent")
        
        if not emotional or not critic:
            return None
        
        sentiment = emotional.analysis.get("sentiment", "neutral")
        coherence = critic.analysis.get("logical_coherence", "medium")
        
        # Conflict: Positive sentiment but low coherence suggests confusion
        if sentiment in ["positive", "very_positive"] and coherence == "low":
            return Conflict(
                conflict_type="sentiment_coherence_mismatch",
                agents=["emotional_agent", "critic_agent"],
                severity="medium",
                resolution_strategy="Prioritize coherence - user may be confused or sarcastic",
                details={
                    "sentiment": sentiment,
                    "coherence": coherence,
                    "emotional_confidence": emotional.confidence,
                    "critic_confidence": critic.confidence
                }
            )
        
        # Conflict: Negative sentiment with high coherence - user understands but disagrees
        if sentiment in ["negative", "very_negative"] and coherence == "high":
            return Conflict(
                conflict_type="sentiment_coherence_mismatch",
                agents=["emotional_agent", "critic_agent"],
                severity="low",
                resolution_strategy="Acknowledge understanding but address concerns",
                details={
                    "sentiment": sentiment,
                    "coherence": coherence,
                    "note": "User understands clearly but has concerns"
                }
            )
        
        return None
    
    def _check_memory_planning_conflict(self, agents_by_id: Dict[str, AgentOutput]) -> Optional[Conflict]:
        """
        Check if memory agent has strong context but planning agent ignores it.
        """
        memory = self._find_agent_output(agents_by_id, "memory_agent")
        planning = self._find_agent_output(agents_by_id, "planning_agent")
        
        if not memory or not planning:
            return None
        
        memory_confidence = memory.confidence
        retrieved_count = len(memory.analysis.get("retrieved_context", []))
        
        # Check if planning suggests "no context" despite strong memory
        planning_analysis_str = str(planning.analysis).lower()
        suggests_no_context = any(phrase in planning_analysis_str for phrase in [
            "no context", "no prior", "first time", "new conversation", "no history"
        ])
        
        if memory_confidence > 0.8 and retrieved_count > 0 and suggests_no_context:
            return Conflict(
                conflict_type="memory_planning_disconnect",
                agents=["memory_agent", "planning_agent"],
                severity="high",
                resolution_strategy="Re-run planning with explicit memory context injection",
                details={
                    "memory_confidence": memory_confidence,
                    "retrieved_memories": retrieved_count,
                    "planning_suggests_no_context": suggests_no_context
                }
            )
        
        return None
    
    def _check_creative_critic_conflict(self, agents_by_id: Dict[str, AgentOutput]) -> Optional[Conflict]:
        """
        Check for creative-critic divergence.
        Note: This is often productive tension, not always a problem!
        """
        creative = self._find_agent_output(agents_by_id, "creative_agent")
        critic = self._find_agent_output(agents_by_id, "critic_agent")
        
        if not creative or not critic:
            return None
        
        creative_confidence = creative.confidence
        creative_score = creative.analysis.get("creative_score", 0.5)
        coherence = critic.analysis.get("logical_coherence", "medium")
        
        # High creativity but low coherence - balance needed
        if creative_confidence > 0.7 and creative_score > 0.7 and coherence == "low":
            return Conflict(
                conflict_type="creative_critic_divergence",
                agents=["creative_agent", "critic_agent"],
                severity="low",
                resolution_strategy="Balance novelty with clarity - explain creative leaps",
                details={
                    "creative_confidence": creative_confidence,
                    "creative_score": creative_score,
                    "coherence": coherence,
                    "note": "Productive tension - needs integration"
                }
            )
        
        return None
    
    def _check_emotional_logic_conflict(self, agents_by_id: Dict[str, AgentOutput]) -> Optional[Conflict]:
        """
        Check if emotional urgency conflicts with logical action recommendations.
        """
        emotional = self._find_agent_output(agents_by_id, "emotional_agent")
        planning = self._find_agent_output(agents_by_id, "planning_agent")
        
        if not emotional or not planning:
            return None
        
        intensity = emotional.analysis.get("intensity", "low")
        sentiment = emotional.analysis.get("sentiment", "neutral")
        recommended_action = planning.analysis.get("recommended_action", "")
        
        # High emotional intensity but planning suggests "wait" or "defer"
        if intensity == "high" and sentiment in ["negative", "very_negative"]:
            if any(word in str(recommended_action).lower() for word in ["wait", "defer", "later", "postpone"]):
                return Conflict(
                    conflict_type="emotional_logic_conflict",
                    agents=["emotional_agent", "planning_agent"],
                    severity="medium",
                    resolution_strategy="Prioritize emotional support, then suggest logical steps",
                    details={
                        "emotional_intensity": intensity,
                        "sentiment": sentiment,
                        "planning_action": recommended_action
                    }
                )
        
        return None
    
    def _check_perception_memory_conflict(self, agents_by_id: Dict[str, AgentOutput]) -> Optional[Conflict]:
        """
        Check if perception identifies topics that contradict memory context.
        """
        perception = self._find_agent_output(agents_by_id, "perception_agent")
        memory = self._find_agent_output(agents_by_id, "memory_agent")
        
        if not perception or not memory:
            return None
        
        topics = perception.analysis.get("topics", [])
        retrieved_context = memory.analysis.get("retrieved_context", [])
        
        # If perception identifies as "new topic" but memory has high confidence
        context_type = perception.analysis.get("context_type", "")
        if "new" in context_type.lower() and memory.confidence > 0.7 and len(retrieved_context) > 0:
            return Conflict(
                conflict_type="perception_memory_conflict",
                agents=["perception_agent", "memory_agent"],
                severity="low",
                resolution_strategy="Trust memory - may be continuation of previous topic",
                details={
                    "perception_context_type": context_type,
                    "memory_confidence": memory.confidence,
                    "topics": topics,
                    "note": "User may be continuing previous conversation"
                }
            )
        
        return None
    
    def _calculate_coherence_score(self, conflicts: List[Conflict], total_agents: int) -> float:
        """
        Calculate overall coherence score based on conflicts detected.
        
        Returns:
            Coherence score from 0.0 (many conflicts) to 1.0 (no conflicts)
        """
        if not conflicts:
            return 1.0
        
        # Weight conflicts by severity
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5
        }
        
        total_conflict_weight = sum(severity_weights.get(c.severity, 0.2) for c in conflicts)
        
        # Normalize by number of agents (more agents = more opportunity for conflicts)
        normalized_conflict = total_conflict_weight / max(total_agents, 1)
        
        # Convert to coherence score (inverse of conflict)
        coherence = max(0.0, 1.0 - normalized_conflict)
        
        return round(coherence, 3)

    def generate_error_analysis(
        self, 
        cycle_id: UUID,
        agent_outputs: List[AgentOutput], 
        coherence_score: float,
        user_input_summary: str,
        response_summary: str,
        cycle_metadata: Dict[str, Any]
    ) -> Optional[ErrorAnalysis]:
        """
        Generate structured error analysis when coherence is critically low.
        
        Args:
            cycle_id: ID of the cognitive cycle
            agent_outputs: All agent outputs from the cycle
            coherence_score: Calculated coherence score
            user_input_summary: Brief summary of user input
            response_summary: Brief summary of system response
            cycle_metadata: Additional cycle context
            
        Returns:
            ErrorAnalysis object if coherence is critically low, None otherwise
        """
        # Only generate analysis for critically low coherence
        if coherence_score >= 0.5:
            return None
            
        # Get conflicts for detailed analysis
        conflict_report = self.detect_conflicts(agent_outputs)
        
        # Determine primary error category based on conflicts
        primary_category = "agent_conflict_unresolved"
        if conflict_report.conflicts:
            conflict_types = [c.conflict_type for c in conflict_report.conflicts]
            if "sentiment_coherence_mismatch" in conflict_types:
                primary_category = "emotional_mismatch"
            elif "creative_critic_divergence" in conflict_types:
                primary_category = "logical_inconsistency"
            elif "memory_planning_disconnect" in conflict_types:
                primary_category = "context_misinterpretation"
        
        # Extract agent information
        agents_activated = [output.agent_id for output in agent_outputs]
        agent_conflicts = [
            {
                "type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "involved_agents": c.involved_agents
            }
            for c in conflict_report.conflicts
        ]
        
        # Calculate severity based on coherence score
        severity_score = 1.0 - coherence_score  # Lower coherence = higher severity
        
        # Generate skill improvement areas based on conflicts
        skill_improvement_areas = []
        if any(c.conflict_type == "sentiment_coherence_mismatch" for c in conflict_report.conflicts):
            skill_improvement_areas.append("emotional_intelligence")
        if any(c.conflict_type == "creative_critic_divergence" for c in conflict_report.conflicts):
            skill_improvement_areas.append("critical_thinking")
        if any(c.conflict_type == "memory_planning_disconnect" for c in conflict_report.conflicts):
            skill_improvement_areas.append("planning")
        
        # Suggest better agent sequence (prioritize critic when there are conflicts)
        recommended_sequence = agents_activated.copy()
        if "critic" not in recommended_sequence and conflict_report.conflicts:
            recommended_sequence.insert(0, "critic")  # Put critic first for better coherence
        
        return ErrorAnalysis(
            cycle_id=cycle_id,
            failure_type="coherence_failure",
            severity_score=severity_score,
            agents_activated=agents_activated,
            agent_conflicts=agent_conflicts,
            coherence_score=coherence_score,
            expected_outcome=0.7,  # Expected good outcome
            actual_outcome=coherence_score * 0.8,  # Estimate based on coherence
            primary_error_category=primary_category,
            recommended_agent_sequence=recommended_sequence,
            skill_improvement_areas=skill_improvement_areas,
            user_input_summary=user_input_summary,
            response_summary=response_summary,
            cycle_metadata=cycle_metadata
        )
