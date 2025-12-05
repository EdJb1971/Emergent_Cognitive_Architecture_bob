"""
Meta-Cognitive Monitor Service - "Feeling of Knowing" Implementation

This service implements prefrontal cortex-inspired meta-cognitive monitoring that:
- Detects when Bob should admit he doesn't know something
- Prevents overconfident hallucinations on obscure topics
- Triggers appropriate uncertainty responses
- Decides when to search vs. generate answers
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.core.config import settings
from src.models.core_models import ErrorAnalysis
from uuid import UUID

logger = logging.getLogger(__name__)


class ActionRecommendation(Enum):
    """Recommended actions based on meta-cognitive assessment"""
    ANSWER = "answer"  # Safe to proceed with synthesis
    SEARCH_FIRST = "search_first"  # Trigger web search before answering
    ASK_CLARIFICATION = "ask_clarification"  # Need more information from user
    DECLINE_POLITELY = "decline_politely"  # Admit uncertainty gracefully
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"  # Note uncertainty but still answer


class GapType(Enum):
    """Types of knowledge gaps detected"""
    TOPIC_UNKNOWN = "topic_unknown"  # Never encountered this topic
    KNOWLEDGE_SPARSE = "knowledge_sparse"  # Some knowledge but insufficient
    HIGH_COMPLEXITY = "high_complexity"  # Topic too complex for current knowledge
    CONTRADICTORY_INFO = "contradictory_info"  # Conflicting information in memory
    OUTDATED_INFO = "outdated_info"  # Knowledge exists but may be outdated
    OVERCONFIDENCE_RISK = "overconfidence_risk"  # Agents confident but gap detected


class MetaCognitiveMonitor:
    """
    Monitors confidence boundaries and knowledge gaps.
    Prevents overconfident responses on unknown topics.
    Implements "feeling of knowing" meta-cognition.
    """

    def __init__(self, memory_service: MemoryService, llm_service: LLMIntegrationService):
        """
        Initialize the meta-cognitive monitor.

        Args:
            memory_service: For accessing conversation history and semantic memory
            llm_service: For generating uncertainty responses and gap analysis
        """
        self.memory_service = memory_service
        self.llm_service = llm_service

        # Thresholds for decision making
        self.knowledge_gap_threshold = 0.7  # Above this = significant gap
        self.overconfidence_threshold = 0.6  # Agent confidence vs knowledge coverage ratio
        self.confidence_variance_threshold = 0.5  # Max variance in agent confidences
        self.min_confidence_threshold = 0.4  # Minimum confidence to proceed

        logger.info("MetaCognitiveMonitor initialized with knowledge gap detection.")

    async def assess_answer_appropriateness(
        self,
        query: str,
        agent_outputs: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Tuple[ActionRecommendation, GapType, float, str]:
        """
        Assess whether Bob should attempt to answer this query.

        Args:
            query: The user's query text
            agent_outputs: List of agent output dictionaries
            user_id: Optional user ID for personalized assessment

        Returns:
            Tuple of (recommendation, gap_type, confidence_score, explanation)
        """
        try:
            # Extract agent confidences and analyze variance
            agent_confidences = []
            for output in agent_outputs:
                if 'confidence' in output:
                    agent_confidences.append(output['confidence'])

            if not agent_confidences:
                return ActionRecommendation.DECLINE_POLITELY, GapType.TOPIC_UNKNOWN, 0.0, "No agent confidence data available"

            avg_confidence = sum(agent_confidences) / len(agent_confidences)
            confidence_variance = sum((c - avg_confidence) ** 2 for c in agent_confidences) / len(agent_confidences)

            # Compute knowledge gap score
            knowledge_gap_score, gap_type = await self.compute_knowledge_gap_score(query, user_id)

            # Detect overconfidence
            overconfidence_score = await self.detect_overconfidence(agent_confidences, knowledge_gap_score)

            # Make recommendation based on analysis
            recommendation, explanation = self.recommend_action(
                knowledge_gap_score=knowledge_gap_score,
                avg_confidence=avg_confidence,
                confidence_variance=confidence_variance,
                overconfidence_score=overconfidence_score,
                query=query,
                gap_type=gap_type
            )

            confidence_score = min(avg_confidence, 1.0 - knowledge_gap_score)  # Conservative confidence

            logger.info(
                f"Meta-cognitive assessment: {recommendation.value} "
                f"(gap: {knowledge_gap_score:.2f}, confidence: {avg_confidence:.2f}, "
                f"variance: {confidence_variance:.2f}) - {explanation}"
            )

            return recommendation, gap_type, confidence_score, explanation

        except Exception as e:
            logger.warning(f"Meta-cognitive assessment failed: {e}")
            return ActionRecommendation.ASK_CLARIFICATION, GapType.TOPIC_UNKNOWN, 0.5, f"Assessment error: {str(e)}"

    def generate_error_analysis(
        self,
        cycle_id: UUID,
        recommendation: ActionRecommendation,
        gap_type: GapType,
        confidence_score: float,
        query: str,
        agents_activated: List[str],
        user_input_summary: str,
        response_summary: str,
        cycle_metadata: Dict[str, Any]
    ) -> Optional[ErrorAnalysis]:
        """
        Generate structured error analysis for meta-cognitive failures.
        
        Args:
            cycle_id: ID of the cognitive cycle
            recommendation: Meta-cognitive action recommendation
            gap_type: Type of knowledge gap detected
            confidence_score: Meta-cognitive confidence score
            query: Original user query
            agents_activated: List of agents that were activated
            user_input_summary: Brief summary of user input
            response_summary: Brief summary of system response
            cycle_metadata: Additional cycle context
            
        Returns:
            ErrorAnalysis object for significant meta-cognitive failures, None otherwise
        """
        # Only generate analysis for significant meta-cognitive failures
        significant_failures = [ActionRecommendation.DECLINE_POLITELY, ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY]
        if recommendation not in significant_failures:
            return None
            
        # Determine failure type and severity
        if recommendation == ActionRecommendation.DECLINE_POLITELY:
            failure_type = "meta_cognitive_decline"
            severity_score = 0.8  # High severity - complete refusal
            primary_category = "knowledge_gap_uncovered"
        else:  # ACKNOWLEDGE_UNCERTAINTY
            failure_type = "meta_cognitive_uncertainty"
            severity_score = 0.6  # Medium severity - partial uncertainty
            primary_category = "skill_deficiency"
        
        # Map gap types to error categories
        gap_to_error_map = {
            GapType.TOPIC_UNKNOWN: "knowledge_gap_uncovered",
            GapType.KNOWLEDGE_SPARSE: "knowledge_gap_uncovered", 
            GapType.HIGH_COMPLEXITY: "skill_deficiency",
            GapType.CONTRADICTORY_INFO: "logical_inconsistency",
            GapType.OUTDATED_INFO: "context_misinterpretation",
            GapType.OVERCONFIDENCE_RISK: "skill_deficiency"
        }
        
        primary_category = gap_to_error_map.get(gap_type, "knowledge_gap_uncovered")
        
        # Determine skill improvement areas based on gap type
        skill_improvement_areas = []
        if gap_type in [GapType.TOPIC_UNKNOWN, GapType.KNOWLEDGE_SPARSE]:
            skill_improvement_areas.append("research")
        if gap_type == GapType.HIGH_COMPLEXITY:
            skill_improvement_areas.extend(["critical_thinking", "planning"])
        if gap_type == GapType.CONTRADICTORY_INFO:
            skill_improvement_areas.append("critical_thinking")
        
        # Suggest agent sequence improvements
        recommended_sequence = agents_activated.copy()
        if "discovery" not in recommended_sequence and recommendation == ActionRecommendation.SEARCH_FIRST:
            recommended_sequence.append("discovery")  # Add discovery agent for knowledge gaps
        
        return ErrorAnalysis(
            cycle_id=cycle_id,
            failure_type=failure_type,
            severity_score=severity_score,
            agents_activated=agents_activated,
            coherence_score=None,  # Meta-cognitive failures don't have coherence scores
            meta_cognitive_assessment={
                "recommendation": recommendation.value,
                "gap_type": gap_type.value,
                "confidence_score": confidence_score
            },
            expected_outcome=0.7,  # Expected good outcome
            actual_outcome=confidence_score * 0.5,  # Estimate based on meta-cognitive confidence
            primary_error_category=primary_category,
            recommended_agent_sequence=recommended_sequence,
            skill_improvement_areas=skill_improvement_areas,
            user_input_summary=user_input_summary,
            response_summary=response_summary,
            cycle_metadata=cycle_metadata
        )

    async def compute_knowledge_gap_score(self, query: str, user_id: Optional[str] = None) -> Tuple[float, GapType]:
        """
        Compute how well Bob's knowledge covers this topic.

        Args:
            query: The user's query
            user_id: Optional user ID for personalized knowledge assessment

        Returns:
            Tuple of (gap_score 0.0-1.0, gap_type)
        """
        try:
            # Analyze query characteristics
            query_complexity = self._assess_query_complexity(query)
            query_is_factual = self._is_factual_query(query)
            query_is_personal = self._is_personal_query(query)

            # Check semantic memory coverage
            semantic_coverage = await self._check_semantic_memory_coverage(query, user_id)

            # Check episodic memory relevance
            episodic_relevance = await self._check_episodic_memory_relevance(query, user_id)

            # Check for contradictory information
            contradiction_score = await self._check_contradictory_information(query, user_id)

            # Combine factors into gap score
            base_gap = 1.0 - (semantic_coverage * 0.6 + episodic_relevance * 0.4)

            # Adjust for complexity and contradiction
            if query_complexity > 0.7:
                base_gap = min(1.0, base_gap + 0.3)  # Complex topics increase gap
                gap_type = GapType.HIGH_COMPLEXITY
            elif contradiction_score > 0.5:
                base_gap = min(1.0, base_gap + 0.2)  # Contradictions increase uncertainty
                gap_type = GapType.CONTRADICTORY_INFO
            elif semantic_coverage < 0.2:
                gap_type = GapType.TOPIC_UNKNOWN
            elif semantic_coverage < 0.5:
                gap_type = GapType.KNOWLEDGE_SPARSE
            else:
                gap_type = GapType.TOPIC_UNKNOWN  # Default

            # Personal queries have lower gap (Bob can be more conversational)
            if query_is_personal:
                base_gap *= 0.7

            gap_score = min(1.0, max(0.0, base_gap))

            return gap_score, gap_type

        except Exception as e:
            logger.warning(f"Knowledge gap computation failed: {e}")
            return 0.8, GapType.TOPIC_UNKNOWN  # Conservative default

    async def detect_overconfidence(self, agent_confidences: List[float], knowledge_gap_score: float) -> float:
        """
        Detect when agents are confident but knowledge coverage is poor.

        Args:
            agent_confidences: List of agent confidence scores
            knowledge_gap_score: Knowledge gap score (0.0-1.0)

        Returns:
            Overconfidence score (0.0-1.0, higher = more overconfident)
        """
        if not agent_confidences:
            return 0.0

        avg_confidence = sum(agent_confidences) / len(agent_confidences)

        # Overconfidence = high agent confidence but high knowledge gap
        overconfidence = avg_confidence * knowledge_gap_score

        # Also check for unrealistic confidence levels
        if avg_confidence > 0.9 and knowledge_gap_score > 0.6:
            overconfidence = min(1.0, overconfidence + 0.3)

        return min(1.0, overconfidence)

    def recommend_action(
        self,
        knowledge_gap_score: float,
        avg_confidence: float,
        confidence_variance: float,
        overconfidence_score: float,
        query: str,
        gap_type: GapType
    ) -> Tuple[ActionRecommendation, str]:
        """
        Recommend the appropriate action based on meta-cognitive analysis.

        Args:
            knowledge_gap_score: 0.0-1.0 (higher = bigger gap)
            avg_confidence: Average agent confidence
            confidence_variance: Variance in agent confidences
            overconfidence_score: Overconfidence detection score
            query: Original query text
            gap_type: Type of knowledge gap

        Returns:
            Tuple of (recommendation, explanation)
        """
        query_is_factual = self._is_factual_query(query)

        # High knowledge gap scenarios
        if knowledge_gap_score > self.knowledge_gap_threshold:
            if query_is_factual:
                return ActionRecommendation.SEARCH_FIRST, f"Significant knowledge gap ({knowledge_gap_score:.2f}) on factual topic - should search first"
            else:
                return ActionRecommendation.ASK_CLARIFICATION, f"Knowledge gap ({knowledge_gap_score:.2f}) on complex topic - need clarification"

        # Overconfidence scenarios
        if overconfidence_score > self.overconfidence_threshold:
            return ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY, f"Overconfidence detected ({overconfidence_score:.2f}) - should acknowledge uncertainty"

        # High confidence variance (agents disagree)
        if confidence_variance > self.confidence_variance_threshold:
            return ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY, f"Agent disagreement (variance: {confidence_variance:.2f}) - acknowledge uncertainty"

        # Low overall confidence
        if avg_confidence < self.min_confidence_threshold:
            if knowledge_gap_score > 0.4:
                return ActionRecommendation.DECLINE_POLITELY, f"Low confidence ({avg_confidence:.2f}) and knowledge gap - decline politely"
            else:
                return ActionRecommendation.ASK_CLARIFICATION, f"Low confidence ({avg_confidence:.2f}) - ask for clarification"

        # Safe to answer
        return ActionRecommendation.ANSWER, f"Knowledge adequate (gap: {knowledge_gap_score:.2f}, confidence: {avg_confidence:.2f}) - safe to answer"

    async def generate_uncertainty_response(self, query: str, gap_type: GapType, recommendation: ActionRecommendation) -> str:
        """
        Generate an appropriate uncertainty response.

        Args:
            query: Original query
            gap_type: Type of knowledge gap
            recommendation: Recommended action

        Returns:
            Natural uncertainty response
        """
        try:
            # Use LLM to generate natural uncertainty responses
            prompt = f"""
            Generate a natural, helpful response acknowledging uncertainty about: "{query}"

            Gap type: {gap_type.value}
            Recommended action: {recommendation.value}

            Response should be:
            - Honest about limitations
            - Helpful and engaging
            - Appropriate for the gap type
            - 1-2 sentences maximum

            Examples:
            - "I'm not entirely sure about that topic, but I can look it up for you."
            - "That's outside my current knowledge area, but I'd be happy to search for information."
            - "I have some thoughts but should double-check the details first."
            """

            response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.7,  # Some creativity for natural responses
                max_output_tokens=100
            )

            # Clean up response
            response = response.strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            return response

        except Exception as e:
            logger.warning(f"Uncertainty response generation failed: {e}")
            # Fallback responses
            fallbacks = {
                ActionRecommendation.SEARCH_FIRST: "I'd like to search for the most current information on that topic.",
                ActionRecommendation.ASK_CLARIFICATION: "Could you provide more details about what you're looking for?",
                ActionRecommendation.DECLINE_POLITELY: "That's a bit outside my current knowledge area.",
                ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY: "I'm not 100% certain about that, but here's what I think."
            }
            return fallbacks.get(recommendation, "I'm not entirely sure about that.")

    # Helper methods

    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0.0-1.0)"""
        complexity_indicators = [
            len(query.split()) > 20,  # Long query
            bool(re.search(r'\b(quantum|neural|algorithm|theory|physics|mathematics)\b', query.lower())),  # Technical terms
            query.count('?') > 1,  # Multiple questions
            len([w for w in query.split() if len(w) > 8]) > 2  # Long words
        ]
        return sum(complexity_indicators) / len(complexity_indicators)

    def _is_factual_query(self, query: str) -> bool:
        """Determine if query is seeking factual information"""
        factual_indicators = ['what is', 'how does', 'when did', 'where is', 'who was', 'explain']
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in factual_indicators)

    def _is_personal_query(self, query: str) -> bool:
        """Determine if query is personal/conversational"""
        personal_indicators = ['you think', 'your opinion', 'how are you', 'tell me about']
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in personal_indicators)

    async def _check_semantic_memory_coverage(self, query: str, user_id: Optional[str] = None) -> float:
        """Check how well semantic memory covers the query topic (0.0-1.0)"""
        try:
            from src.models.core_models import MemoryQueryRequest
            from uuid import UUID

            # Use memory service to find relevant memories
            query_request = MemoryQueryRequest(
                user_id=UUID(user_id) if user_id else UUID('00000000-0000-0000-0000-000000000000'),
                query_text=query,
                limit=5,
                min_relevance_score=0.3
            )
            memories = await self.memory_service.query_memory(query_request)

            if not memories:
                return 0.0

            # Simple coverage score based on number and relevance of memories
            coverage = min(1.0, len(memories) / 3.0)  # 3+ memories = full coverage

            # Could be enhanced with embedding similarity scores
            return coverage

        except Exception as e:
            logger.warning(f"Semantic memory coverage check failed: {e}")
            return 0.0

    async def _check_episodic_memory_relevance(self, query: str, user_id: Optional[str] = None) -> float:
        """Check episodic memory relevance (0.0-1.0)"""
        try:
            # Check recent conversations for similar topics
            stm = await self.memory_service.get_stm(user_id)
            if stm:
                recent_cycles = stm.get_recent_cycles(limit=10)
                relevant_cycles = 0

                for cycle in recent_cycles:
                    if cycle.user_input and self._queries_similar(query, cycle.user_input):
                        relevant_cycles += 1

                return min(1.0, relevant_cycles / 3.0)  # 3+ similar recent conversations

            return 0.0

        except Exception as e:
            logger.warning(f"Episodic memory relevance check failed: {e}")
            return 0.0

    async def _check_contradictory_information(self, query: str, user_id: Optional[str] = None) -> float:
        """Check for contradictory information in memory (0.0-1.0)"""
        # This would require more sophisticated analysis
        # For now, return low score (could be enhanced with LLM analysis)
        return 0.0

    def _queries_similar(self, query1: str, query2: str) -> bool:
        """Simple similarity check between queries"""
        # Basic keyword overlap check
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap >= 2  # At least 2 common words