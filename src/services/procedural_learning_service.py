"""
Procedural Learning Service - Cerebellum-Inspired Skill Refinement

This service implements cerebellar learning mechanisms that track and improve
performance on specific conversation skills through error-based learning and
optimal sequence discovery.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from uuid import UUID

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.services.metrics_service import MetricsService, MetricType
from src.models.core_models import ErrorAnalysis
from src.core.config import settings

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of conversational skills to track and improve"""
    TECHNICAL_EXPLANATION = "technical_explanation"
    EMOTIONAL_SUPPORT = "emotional_support"
    CREATIVE_BRAINSTORMING = "creative_brainstorming"
    PROBLEM_SOLVING = "problem_solving"
    FACTUAL_DELIVERY = "factual_delivery"
    TEACHING_TUTORING = "teaching_tutoring"
    GENERAL_CONVERSATION = "general_conversation"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class SkillPerformance:
    """Tracks performance metrics for a specific skill"""
    skill_category: SkillCategory
    total_attempts: int = 0
    successful_attempts: int = 0
    avg_satisfaction_score: float = 0.0
    avg_confidence_score: float = 0.0
    error_patterns: Dict[str, int] = None  # error_type -> count
    last_updated: datetime = None

    def __post_init__(self):
        if self.error_patterns is None:
            self.error_patterns = {}
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    def update_performance(self, success: bool, satisfaction_score: float, confidence_score: float, error_type: Optional[str] = None):
        """Update performance metrics with new outcome"""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1

        # Update running averages
        self.avg_satisfaction_score = ((self.avg_satisfaction_score * (self.total_attempts - 1)) + satisfaction_score) / self.total_attempts
        self.avg_confidence_score = ((self.avg_confidence_score * (self.total_attempts - 1)) + confidence_score) / self.total_attempts

        # Track error patterns
        if error_type:
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1

        self.last_updated = datetime.utcnow()


@dataclass
class StrategySequence:
    """Represents a learned sequence of agent strategies for a task"""
    task_type: str
    agent_sequence: List[str]  # e.g., ["creative", "planning", "critic"]
    total_uses: int = 0
    successful_uses: int = 0
    avg_outcome_score: float = 0.0
    learned_from_cycles: List[str] = None  # cycle IDs that contributed to learning

    def __post_init__(self):
        if self.learned_from_cycles is None:
            self.learned_from_cycles = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this sequence"""
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses

    def update_outcome(self, success: bool, outcome_score: float, cycle_id: str):
        """Update sequence performance with new outcome"""
        self.total_uses += 1
        if success:
            self.successful_uses += 1

        # Update running average
        self.avg_outcome_score = ((self.avg_outcome_score * (self.total_uses - 1)) + outcome_score) / self.total_uses

        # Track learning source
        if cycle_id not in self.learned_from_cycles:
            self.learned_from_cycles.append(cycle_id)


class ProceduralLearningService:
    """
    Cerebellum-inspired skill refinement and procedural learning.
    Tracks performance on conversation skills and learns optimal strategies.
    """

    def __init__(self, memory_service: MemoryService, llm_service: LLMIntegrationService, metrics_service: Optional[MetricsService] = None):
        """
        Initialize the procedural learning service.

        Args:
            memory_service: For accessing conversation history and patterns
            llm_service: For analyzing performance patterns and generating insights
            metrics_service: For recording learning metrics
        """
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.metrics_service = metrics_service

        # Skill performance tracking
        self.skill_performance: Dict[SkillCategory, SkillPerformance] = {}

        # Learned strategy sequences
        self.strategy_sequences: Dict[str, List[StrategySequence]] = {}  # task_type -> list of sequences

        # Performance thresholds
        self.success_threshold = 0.7  # Consider outcome successful if satisfaction >= 0.7
        self.min_samples_for_learning = 5  # Need at least N samples before considering learned
        self.sequence_learning_threshold = 3  # Need at least N different sequences tried

        logger.info("ProceduralLearningService initialized for skill refinement and sequence learning.")

    async def track_skill_performance(
        self,
        skill_category: SkillCategory,
        outcome_score: float,
        confidence_score: float,
        agent_sequence: List[str],
        cycle_metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> SkillPerformance:
        """
        Track performance of a specific conversational skill.

        Args:
            skill_category: The type of skill being performed
            outcome_score: Overall outcome satisfaction (0.0-1.0)
            confidence_score: System confidence in the response (0.0-1.0)
            agent_sequence: List of agents used in this interaction
            cycle_metadata: Additional metadata from the cognitive cycle
            user_id: Optional user ID for personalized learning

        Returns:
            Updated SkillPerformance object
        """
        try:
            # Get or create skill performance tracker
            if skill_category not in self.skill_performance:
                self.skill_performance[skill_category] = SkillPerformance(skill_category=skill_category)

            skill_perf = self.skill_performance[skill_category]

            # Determine if this was a successful outcome
            success = outcome_score >= self.success_threshold

            # Identify error patterns if unsuccessful
            error_type = None
            if not success:
                error_type = self._identify_error_pattern(cycle_metadata, outcome_score, confidence_score)

            # Update performance metrics
            skill_perf.update_performance(success, outcome_score, confidence_score, error_type)

            # Track strategy sequence performance
            await self._track_strategy_sequence(skill_category, agent_sequence, success, outcome_score, cycle_metadata.get('cycle_id'))

            # Record learning metrics
            if self.metrics_service:
                await self.metrics_service.record_metric(
                    MetricType.LEARNING_EVENT,
                    {
                        "skill_category": skill_category.value,
                        "outcome_score": outcome_score,
                        "confidence_score": confidence_score,
                        "success": success,
                        "error_type": error_type,
                        "agent_sequence": agent_sequence,
                        "performance_improvement": skill_perf.success_rate if skill_perf.total_attempts > 1 else 0.0,
                        "total_attempts": skill_perf.total_attempts
                    },
                    cycle_id=cycle_metadata.get('cycle_id'),
                    user_id=user_id
                )

            logger.info(
                f"Tracked skill performance for {skill_category.value}: "
                f"success_rate={skill_perf.success_rate:.2f}, "
                f"avg_satisfaction={skill_perf.avg_satisfaction_score:.2f}, "
                f"total_attempts={skill_perf.total_attempts}"
            )

            return skill_perf

        except Exception as e:
            logger.warning(f"Failed to track skill performance for {skill_category}: {e}")
            return self.skill_performance.get(skill_category, SkillPerformance(skill_category=skill_category))

    async def learn_from_error(
        self,
        error_analysis: Optional[ErrorAnalysis] = None,
        error_type: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None,
        skill_category: Optional[SkillCategory] = None,
        expected_outcome: Optional[float] = None,
        actual_outcome: Optional[float] = None,
        agent_sequence: Optional[List[str]] = None,
        cycle_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Learn from errors using structured ErrorAnalysis or legacy parameters.

        Args:
            error_analysis: Structured error analysis object (preferred)
            error_type: Legacy error type string
            error_context: Legacy error context dictionary
            skill_category: Legacy skill category
            expected_outcome: Legacy expected outcome
            actual_outcome: Legacy actual outcome
            agent_sequence: Legacy agent sequence
            cycle_metadata: Legacy cycle metadata
            user_id: User ID for personalization

        Returns:
            Dictionary with improvement suggestions, or None if no suggestions
        """
        try:
            # Handle structured ErrorAnalysis (preferred method)
            if error_analysis:
                return await self._learn_from_structured_error(error_analysis, user_id)
            
            # Handle legacy parameters for backward compatibility
            if error_type and error_context:
                return await self._learn_from_legacy_error(
                    error_type, error_context, skill_category, expected_outcome, 
                    actual_outcome, agent_sequence, cycle_metadata, user_id
                )
            
            logger.warning("learn_from_error called without valid parameters")
            return None

        except Exception as e:
            logger.warning(f"Failed to learn from error: {e}")
            return None

    async def _learn_from_structured_error(
        self,
        error_analysis: ErrorAnalysis,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Learn from structured ErrorAnalysis object.
        
        Args:
            error_analysis: Structured error analysis
            user_id: User ID for personalization
            
        Returns:
            Dictionary with improvement suggestions
        """
        try:
            # Determine skill category from error analysis
            skill_category = self._map_error_to_skill_category(error_analysis.primary_error_category)
            
            # Calculate error magnitude
            error_magnitude = error_analysis.expected_outcome - error_analysis.actual_outcome
            if error_magnitude < 0.1:  # Not a significant error
                return None

            # Get skill performance data
            skill_perf = self.skill_performance.get(skill_category)
            if not skill_perf or skill_perf.total_attempts < self.min_samples_for_learning:
                return None

            # Update skill performance with error
            skill_perf.update_performance(
                success=False,
                satisfaction_score=error_analysis.actual_outcome,
                confidence_score=0.3,  # Low confidence for errors
                error_type=error_analysis.failure_type
            )

            # Generate improvement suggestions using LLM
            suggestions = await self._generate_improvement_suggestions_from_analysis(
                error_analysis, skill_perf.error_patterns
            )

            # Update strategy sequences if recommended sequence differs
            if error_analysis.recommended_agent_sequence != error_analysis.agents_activated:
                await self._update_strategy_sequence_from_error(
                    error_analysis, skill_category, user_id
                )

            logger.info(f"Generated structured error-based improvements for {skill_category.value}: {len(suggestions)} suggestions")

            return {
                "skill_category": skill_category.value,
                "error_magnitude": error_magnitude,
                "severity_score": error_analysis.severity_score,
                "current_success_rate": skill_perf.success_rate,
                "improvement_suggestions": suggestions,
                "recommended_agent_sequence": error_analysis.recommended_agent_sequence,
                "skill_improvement_areas": error_analysis.skill_improvement_areas,
                "primary_error_category": error_analysis.primary_error_category
            }

        except Exception as e:
            logger.warning(f"Failed to learn from structured error: {e}")
            return None

    async def _learn_from_legacy_error(
        self,
        error_type: str,
        error_context: Dict[str, Any],
        skill_category: Optional[SkillCategory],
        expected_outcome: Optional[float],
        actual_outcome: Optional[float],
        agent_sequence: Optional[List[str]],
        cycle_metadata: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Legacy error learning method for backward compatibility.
        """
        try:
            # Default values for legacy calls
            skill_category = skill_category or SkillCategory.GENERAL_CONVERSATION
            expected_outcome = expected_outcome or 0.7
            actual_outcome = actual_outcome or 0.3
            agent_sequence = agent_sequence or []
            cycle_metadata = cycle_metadata or {}

            error_magnitude = expected_outcome - actual_outcome
            if error_magnitude < 0.2:  # Not a significant error
                return None

            # Analyze error patterns
            skill_perf = self.skill_performance.get(skill_category)
            if not skill_perf or skill_perf.total_attempts < self.min_samples_for_learning:
                return None

            # Generate improvement suggestions using LLM
            suggestions = await self._generate_improvement_suggestions(
                skill_category, error_magnitude, agent_sequence, skill_perf.error_patterns, cycle_metadata
            )

            logger.info(f"Generated legacy error-based improvements for {skill_category.value}: {len(suggestions)} suggestions")

            return {
                "skill_category": skill_category.value,
                "error_magnitude": error_magnitude,
                "current_success_rate": skill_perf.success_rate,
                "improvement_suggestions": suggestions,
                "recommended_agent_sequence": self._suggest_better_sequence(skill_category, agent_sequence)
            }

        except Exception as e:
            logger.warning(f"Failed to learn from legacy error for {skill_category}: {e}")
            return None

    def _map_error_to_skill_category(self, error_category: str) -> SkillCategory:
        """
        Map error categories to skill categories for learning.
        
        Args:
            error_category: Error category from ErrorAnalysis
            
        Returns:
            Corresponding SkillCategory
        """
        category_mapping = {
            "knowledge_gap_uncovered": SkillCategory.TECHNICAL_EXPLANATION,
            "skill_deficiency": SkillCategory.PROBLEM_SOLVING,
            "logical_inconsistency": SkillCategory.FACTUAL_DELIVERY,
            "context_misinterpretation": SkillCategory.GENERAL_CONVERSATION,
            "agent_coordination_failure": SkillCategory.CONFLICT_RESOLUTION,
            "response_generation_failure": SkillCategory.GENERAL_CONVERSATION,
            "coherence_failure": SkillCategory.CONFLICT_RESOLUTION,
            "meta_cognitive_failure": SkillCategory.TEACHING_TUTORING
        }
        return category_mapping.get(error_category, SkillCategory.GENERAL_CONVERSATION)

    async def _generate_improvement_suggestions_from_analysis(
        self,
        error_analysis: ErrorAnalysis,
        error_patterns: Dict[str, int]
    ) -> List[str]:
        """
        Generate improvement suggestions based on structured error analysis.
        
        Args:
            error_analysis: Structured error analysis
            error_patterns: Historical error patterns
            
        Returns:
            List of improvement suggestions
        """
        try:
            # Create context for LLM
            context = f"""
            Error Analysis:
            - Failure Type: {error_analysis.failure_type}
            - Severity: {error_analysis.severity_score:.2f}
            - Primary Category: {error_analysis.primary_error_category}
            - Agents Used: {', '.join(error_analysis.agents_activated)}
            - Recommended Sequence: {', '.join(error_analysis.recommended_agent_sequence)}
            - Skill Areas for Improvement: {', '.join(error_analysis.skill_improvement_areas)}
            - User Input: {error_analysis.user_input_summary}
            - Response: {error_analysis.response_summary}
            
            Historical Error Patterns: {error_patterns}
            """
            
            prompt = f"""
            Based on this structured error analysis, provide 3-5 specific, actionable suggestions 
            for improving the system's performance. Focus on:
            1. Agent sequence optimization
            2. Skill development areas
            3. Process improvements
            4. Pattern recognition from historical errors
            
            Context: {context}
            
            Provide suggestions as a numbered list.
            """
            
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            # Parse suggestions from response
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/punctuation
                    clean_line = line.lstrip('0123456789.- ').strip()
                    if clean_line:
                        suggestions.append(clean_line)
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.warning(f"Failed to generate improvement suggestions from analysis: {e}")
            return ["Review agent coordination patterns", "Consider alternative agent sequences", "Focus on identified skill improvement areas"]

    async def _update_strategy_sequence_from_error(
        self,
        error_analysis: ErrorAnalysis,
        skill_category: SkillCategory,
        user_id: Optional[str]
    ):
        """
        Update strategy sequences based on error analysis recommendations.
        
        Args:
            error_analysis: Structured error analysis
            skill_category: Mapped skill category
            user_id: User ID for personalization
        """
        try:
            task_type = f"{skill_category.value}_{user_id}" if user_id else skill_category.value
            
            # Initialize task type if not exists
            if task_type not in self.strategy_sequences:
                self.strategy_sequences[task_type] = []
            
            # Find existing sequence or create new one
            existing_sequence = None
            for seq in self.strategy_sequences[task_type]:
                if seq.agent_sequence == error_analysis.agents_activated:
                    existing_sequence = seq
                    break
            
            if not existing_sequence:
                existing_sequence = StrategySequence(
                    task_type=task_type,
                    agent_sequence=error_analysis.agents_activated.copy()
                )
                self.strategy_sequences[task_type].append(existing_sequence)
            
            # Record failure for current sequence
            existing_sequence.update_outcome(
                success=False,
                outcome_score=error_analysis.actual_outcome,
                cycle_id=str(error_analysis.cycle_id)
            )
            
            # Create or update recommended sequence
            recommended_sequence = None
            for seq in self.strategy_sequences[task_type]:
                if seq.agent_sequence == error_analysis.recommended_agent_sequence:
                    recommended_sequence = seq
                    break
            
            if not recommended_sequence:
                recommended_sequence = StrategySequence(
                    task_type=task_type,
                    agent_sequence=error_analysis.recommended_agent_sequence.copy()
                )
                self.strategy_sequences[task_type].append(recommended_sequence)
            
            # Note: Recommended sequence gets updated with success when it performs well
            
            logger.debug(f"Updated strategy sequences for {task_type} based on error analysis")
            
        except Exception as e:
            logger.warning(f"Failed to update strategy sequence from error: {e}")

    async def get_optimal_strategy_sequence(self, task_type: str, available_agents: List[str]) -> Optional[List[str]]:
        """
        Get the learned optimal sequence of agents for a task type.

        Args:
            task_type: The type of task (e.g., "technical_question", "emotional_support")
            available_agents: List of agents currently available

        Returns:
            Optimal agent sequence, or None if no learned sequence available
        """
        try:
            if task_type not in self.strategy_sequences:
                return None

            sequences = self.strategy_sequences[task_type]

            # Filter sequences that only use available agents
            valid_sequences = [
                seq for seq in sequences
                if all(agent in available_agents for agent in seq.agent_sequence) and seq.total_uses >= self.min_samples_for_learning
            ]

            if not valid_sequences:
                return None

            # Return the sequence with highest success rate
            best_sequence = max(valid_sequences, key=lambda s: s.success_rate)

            # Only return if significantly better than random
            if best_sequence.success_rate > 0.6 and best_sequence.total_uses >= 10:
                logger.info(f"Returning learned optimal sequence for {task_type}: {best_sequence.agent_sequence} (success_rate: {best_sequence.success_rate:.2f})")
                return best_sequence.agent_sequence

            return None

        except Exception as e:
            logger.warning(f"Failed to get optimal sequence for {task_type}: {e}")
            return None

    def get_skill_performance_summary(self, skill_category: Optional[SkillCategory] = None) -> Dict[str, Any]:
        """
        Get performance summary for skills.

        Args:
            skill_category: Specific skill to summarize, or None for all skills

        Returns:
            Dictionary with performance summaries
        """
        try:
            if skill_category:
                skills_to_summarize = [skill_category]
            else:
                skills_to_summarize = list(self.skill_performance.keys())

            summary = {}
            for skill in skills_to_summarize:
                if skill in self.skill_performance:
                    perf = self.skill_performance[skill]
                    summary[skill.value] = {
                        "total_attempts": perf.total_attempts,
                        "success_rate": perf.success_rate,
                        "avg_satisfaction": perf.avg_satisfaction_score,
                        "avg_confidence": perf.avg_confidence_score,
                        "common_errors": sorted(perf.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3],
                        "last_updated": perf.last_updated.isoformat() if perf.last_updated else None
                    }

            return summary

        except Exception as e:
            logger.warning(f"Failed to generate skill performance summary: {e}")
            return {}

    async def suggest_skill_improvement(self, skill_category: SkillCategory) -> Optional[Dict[str, Any]]:
        """
        Suggest improvements for a specific skill based on performance patterns.

        Args:
            skill_category: The skill to analyze

        Returns:
            Dictionary with improvement suggestions
        """
        try:
            skill_perf = self.skill_performance.get(skill_category)
            if not skill_perf or skill_perf.total_attempts < self.min_samples_for_learning:
                return None

            # Analyze performance patterns
            improvement_areas = []

            if skill_perf.success_rate < 0.6:
                improvement_areas.append("low_success_rate")
            if skill_perf.avg_satisfaction_score < 0.7:
                improvement_areas.append("low_satisfaction")
            if skill_perf.avg_confidence_score < 0.6:
                improvement_areas.append("low_confidence")

            # Analyze error patterns
            top_errors = sorted(skill_perf.error_patterns.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_errors:
                improvement_areas.extend([f"reduce_{error_type}" for error_type, _ in top_errors])

            if not improvement_areas:
                return None

            # Generate specific suggestions
            suggestions = await self._generate_skill_improvements(skill_category, improvement_areas, skill_perf)

            return {
                "skill_category": skill_category.value,
                "current_performance": {
                    "success_rate": skill_perf.success_rate,
                    "avg_satisfaction": skill_perf.avg_satisfaction_score,
                    "total_attempts": skill_perf.total_attempts
                },
                "improvement_areas": improvement_areas,
                "specific_suggestions": suggestions
            }

        except Exception as e:
            logger.warning(f"Failed to suggest improvements for {skill_category}: {e}")
            return None

    # Helper methods

    def _identify_error_pattern(self, cycle_metadata: Dict[str, Any], outcome_score: float, confidence_score: float) -> Optional[str]:
        """Identify the type of error that occurred"""
        try:
            # Check for common error patterns
            if outcome_score < 0.3:
                return "very_low_satisfaction"
            elif confidence_score > 0.8 and outcome_score < 0.5:
                return "overconfidence"
            elif "conflict" in cycle_metadata.get("final_conflicts", {}):
                return "unresolved_conflict"
            elif cycle_metadata.get("response_metadata", {}).get("response_type") == "error":
                return "system_error"
            elif outcome_score < 0.6:
                return "low_satisfaction"
            else:
                return "general_unsatisfactory"

        except Exception:
            return "unknown_error"

    async def _track_strategy_sequence(
        self,
        skill_category: SkillCategory,
        agent_sequence: List[str],
        success: bool,
        outcome_score: float,
        cycle_id: Optional[str]
    ):
        """Track performance of specific agent sequences"""
        try:
            task_type = skill_category.value

            if task_type not in self.strategy_sequences:
                self.strategy_sequences[task_type] = []

            sequences = self.strategy_sequences[task_type]

            # Find existing sequence or create new one
            sequence_key = tuple(agent_sequence)
            existing_sequence = None

            for seq in sequences:
                if tuple(seq.agent_sequence) == sequence_key:
                    existing_sequence = seq
                    break

            if not existing_sequence:
                existing_sequence = StrategySequence(
                    task_type=task_type,
                    agent_sequence=agent_sequence.copy()
                )
                sequences.append(existing_sequence)

            # Update sequence performance
            existing_sequence.update_outcome(success, outcome_score, cycle_id or "unknown")

        except Exception as e:
            logger.warning(f"Failed to track strategy sequence: {e}")

    def _suggest_better_sequence(self, skill_category: SkillCategory, current_sequence: List[str]) -> Optional[List[str]]:
        """Suggest a better agent sequence based on learned patterns"""
        try:
            task_type = skill_category.value
            sequences = self.strategy_sequences.get(task_type, [])

            if len(sequences) < 2:
                return None

            # Find the best performing sequence
            best_sequence = max(sequences, key=lambda s: s.success_rate)

            # Only suggest if significantly better
            if best_sequence.success_rate > 0.7 and best_sequence.total_uses >= self.min_samples_for_learning:
                return best_sequence.agent_sequence

            return None

        except Exception:
            return None

    async def _generate_improvement_suggestions(
        self,
        skill_category: SkillCategory,
        error_magnitude: float,
        agent_sequence: List[str],
        error_patterns: Dict[str, int],
        cycle_metadata: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions using LLM analysis"""
        try:
            prompt = f"""
            Analyze this conversational skill performance issue and suggest specific improvements:

            Skill Category: {skill_category.value}
            Error Magnitude: {error_magnitude:.2f}
            Agent Sequence Used: {', '.join(agent_sequence)}
            Error Patterns: {json.dumps(error_patterns, indent=2)}
            Cycle Context: {json.dumps(cycle_metadata, indent=2)}

            Provide 2-3 specific, actionable suggestions for improving this skill.
            Focus on agent selection, sequencing, or strategy adjustments.
            Keep suggestions concise and practical.
            """

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.3,  # Lower temperature for more focused suggestions
                max_output_tokens=200
            )

            # Parse suggestions from response
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line[0].isdigit()):
                    # Extract the suggestion text
                    suggestion = line.lstrip('-0123456789. ').strip()
                    if suggestion:
                        suggestions.append(suggestion)

            return suggestions[:3]  # Limit to top 3

        except Exception as e:
            logger.warning(f"Failed to generate improvement suggestions: {e}")
            return ["Review agent selection for this skill type", "Consider different agent sequencing", "Analyze successful examples for patterns"]

    async def _generate_skill_improvements(
        self,
        skill_category: SkillCategory,
        improvement_areas: List[str],
        skill_perf: SkillPerformance
    ) -> List[str]:
        """Generate skill-specific improvement suggestions"""
        try:
            prompt = f"""
            Based on performance analysis, suggest improvements for the {skill_category.value} skill:

            Current Performance:
            - Success Rate: {skill_perf.success_rate:.2f}
            - Average Satisfaction: {skill_perf.avg_satisfaction_score:.2f}
            - Total Attempts: {skill_perf.total_attempts}
            - Common Errors: {json.dumps(skill_perf.error_patterns, indent=2)}

            Areas Needing Improvement: {', '.join(improvement_areas)}

            Provide 2-3 specific suggestions for improving this conversational skill.
            Focus on practical changes to agent usage, response strategies, or interaction patterns.
            """

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.4,
                max_output_tokens=250
            )

            # Parse suggestions
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short lines
                    suggestions.append(line)

            return suggestions[:3]

        except Exception as e:
            logger.warning(f"Failed to generate skill improvements: {e}")
            return ["Increase practice with this skill type", "Study successful examples", "Adjust agent selection strategy"]