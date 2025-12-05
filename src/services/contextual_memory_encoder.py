"""
Contextual Memory Encoder Service - Rich Memory Bindings

Inspired by the hippocampus, which doesn't just store memories but binds contextual
information (where, when, emotional state) and consolidates them over time.

This service enriches cognitive cycles with rich contextual tags for better recall
and enables memory consolidation based on significance.
"""

from src.models.core_models import CognitiveCycle, ContextualBindings, ConsolidationMetadata, AgentOutput
from datetime import datetime
from typing import List, Dict, Any, Optional
import re


class ContextualMemoryEncoder:
    """
    Encodes memories with rich contextual tags inspired by hippocampal function.
    Creates temporal, emotional, semantic, relational, and cognitive bindings.
    """
    
    def __init__(self):
        self.personal_disclosure_patterns = [
            r'\bmy name is\b', r'\bi live\b', r'\bi work\b', r'\bi feel\b',
            r'\bi\'m worried\b', r'\bi\'m excited\b', r'\bi\'m frustrated\b',
            r'\bhonestly\b', r'\bto be honest\b', r'\bmy family\b', r'\bmy partner\b',
            r'\bmy job\b', r'\bi\'m from\b', r'\bi grew up\b'
        ]
        
        self.insight_patterns = [
            r'\bi understand\b', r'\bi see\b', r'\bthat makes sense\b',
            r'\baha\b', r'\boh i get it\b', r'\bi realize\b', r'\bi learned\b',
            r'\bnow i know\b', r'\bthat explains\b'
        ]
    
    async def encode_cycle(self, cycle: CognitiveCycle, session_start: datetime) -> CognitiveCycle:
        """
        Enrich cognitive cycle with contextual bindings.
        
        Args:
            cycle: The cognitive cycle to enrich
            session_start: When the current session started (for duration calculation)
            
        Returns:
            Enriched cognitive cycle with contextual_bindings and consolidation_metadata
        """
        # Extract contextual bindings
        contextual_bindings = self._extract_contextual_bindings(cycle, session_start)
        
        # Compute consolidation priority
        consolidation_metadata = self._compute_consolidation_metadata(cycle, contextual_bindings)
        
        # Add to cycle metadata
        cycle.metadata["contextual_bindings"] = contextual_bindings.model_dump(mode='json')
        cycle.metadata["consolidation_metadata"] = consolidation_metadata.model_dump(mode='json')
        
        return cycle
    
    def _extract_contextual_bindings(self, cycle: CognitiveCycle, session_start: datetime) -> ContextualBindings:
        """
        Extract rich contextual information from the cycle.
        """
        # Temporal context
        time_of_day = self._get_time_category(cycle.timestamp)
        session_duration = (cycle.timestamp - session_start).total_seconds() / 60.0
        
        # Emotional context
        emotional_valence, emotional_arousal = self._extract_emotional_context(cycle.agent_outputs)
        
        # Semantic context
        topics, entities, intent = self._extract_semantic_context(cycle)
        
        # Relational context
        conversation_depth = self._assess_conversation_depth(cycle)
        rapport_level = self._assess_rapport_level(cycle)
        
        # Cognitive context
        complexity = self._assess_cognitive_complexity(cycle)
        novelty = self._assess_novelty(cycle)
        
        return ContextualBindings(
            time_of_day=time_of_day,
            session_duration_minutes=round(session_duration, 2),
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            topics=topics,
            entities=entities,
            intent=intent,
            conversation_depth=conversation_depth,
            rapport_level=rapport_level,
            complexity=complexity,
            novelty=novelty
        )
    
    def _compute_consolidation_metadata(self, cycle: CognitiveCycle, bindings: ContextualBindings) -> ConsolidationMetadata:
        """
        Compute how important this memory is for long-term consolidation.
        High emotional arousal, novelty, and personal relevance increase priority.
        """
        priority = 0.5  # baseline
        
        # High emotional arousal increases priority
        if bindings.emotional_arousal == "high":
            priority += 0.2
        elif bindings.emotional_arousal == "medium":
            priority += 0.1
        
        # Novel information increases priority
        if bindings.novelty > 0.7:
            priority += 0.15
        elif bindings.novelty > 0.5:
            priority += 0.08
        
        # Deep/personal conversations increase priority
        if bindings.conversation_depth in ["deep", "intimate"]:
            priority += 0.15
        elif bindings.conversation_depth == "moderate":
            priority += 0.08
        
        # Personal disclosures are highly significant
        if self._contains_personal_disclosure(cycle.user_input):
            priority += 0.2
        
        # Insights and breakthroughs are important
        if self._contains_insight(cycle.user_input):
            priority += 0.15
        
        # Complex cognitive work deserves consolidation
        if bindings.complexity == "complex":
            priority += 0.1
        
        return ConsolidationMetadata(
            consolidation_priority=min(1.0, priority),
            replay_count=0,
            last_accessed=cycle.timestamp,
            access_count=0
        )
    
    def _get_time_category(self, timestamp: datetime) -> str:
        """Categorize time of day."""
        hour = timestamp.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _extract_emotional_context(self, agent_outputs: List[AgentOutput]) -> tuple[str, str]:
        """Extract emotional valence and arousal from emotional agent."""
        emotional_output = next((o for o in agent_outputs if o.agent_id == "emotional_agent"), None)
        
        if not emotional_output:
            return "neutral", "low"
        
        sentiment = emotional_output.analysis.get("sentiment", "neutral")
        intensity = emotional_output.analysis.get("intensity", "low")
        
        # Map sentiment to valence
        if sentiment in ["positive", "very_positive"]:
            valence = "positive"
        elif sentiment in ["negative", "very_negative"]:
            valence = "negative"
        elif sentiment == "mixed":
            valence = "mixed"
        else:
            valence = "neutral"
        
        # Map intensity to arousal
        arousal = intensity  # Already in low/medium/high format
        
        return valence, arousal
    
    def _extract_semantic_context(self, cycle: CognitiveCycle) -> tuple[List[str], List[str], str]:
        """Extract topics, entities, and intent from perception agent."""
        perception_output = next((o for o in cycle.agent_outputs if o.agent_id == "perception_agent"), None)
        
        topics = []
        entities = []
        intent = "statement"
        
        if perception_output:
            topics = perception_output.analysis.get("topics", [])
            # Extract entities (capitalized words/phrases)
            keywords = perception_output.analysis.get("keywords", [])
            entities = [k for k in keywords if k and k[0].isupper()]
            intent = perception_output.analysis.get("context_type", "statement")
        
        return topics, entities, intent
    
    def _assess_conversation_depth(self, cycle: CognitiveCycle) -> str:
        """Assess depth of conversation based on content."""
        user_input_lower = cycle.user_input.lower()
        
        # Intimate: Personal disclosures, deep emotions
        if self._contains_personal_disclosure(user_input_lower):
            return "intimate"
        
        # Deep: Complex reasoning, philosophical, insights
        if self._contains_insight(user_input_lower) or len(cycle.user_input.split()) > 50:
            return "deep"
        
        # Moderate: Questions, explanations
        if '?' in cycle.user_input or len(cycle.user_input.split()) > 20:
            return "moderate"
        
        # Superficial: Short statements, greetings
        return "superficial"
    
    def _assess_rapport_level(self, cycle: CognitiveCycle) -> str:
        """
        Assess rapport level based on interaction quality.
        This is a simplified version - ideally would track over time.
        """
        user_input_lower = cycle.user_input.lower()
        
        # Strong rapport indicators
        if any(phrase in user_input_lower for phrase in ['thank you', 'appreciate', 'helpful', 'great', 'love how']):
            return "strong"
        
        # Established rapport indicators
        if any(phrase in user_input_lower for phrase in ['remember', 'last time', 'as we discussed']):
            return "established"
        
        # Developing rapport indicators
        if '?' in cycle.user_input or self._contains_personal_disclosure(user_input_lower):
            return "developing"
        
        # Initial
        return "initial"
    
    def _assess_cognitive_complexity(self, cycle: CognitiveCycle) -> str:
        """Assess cognitive load/complexity of the interaction."""
        # Count activated agents as proxy for complexity
        num_agents = len(cycle.agent_outputs)
        
        # Check for complex reasoning indicators
        planning_output = next((o for o in cycle.agent_outputs if o.agent_id == "planning_agent"), None)
        has_complex_planning = planning_output and len(planning_output.analysis.get("feasible_options", [])) > 2
        
        critic_output = next((o for o in cycle.agent_outputs if o.agent_id == "critic_agent"), None)
        has_contradictions = critic_output and len(critic_output.analysis.get("contradictions_found", [])) > 0
        
        if num_agents >= 6 or has_complex_planning or has_contradictions:
            return "complex"
        elif num_agents >= 4:
            return "moderate"
        else:
            return "simple"
    
    def _assess_novelty(self, cycle: CognitiveCycle) -> float:
        """
        Assess how novel/surprising this interaction is.
        This is simplified - ideally would compare against past interactions.
        """
        novelty = 0.5  # baseline
        
        # Creative agent confidence suggests novelty
        creative_output = next((o for o in cycle.agent_outputs if o.agent_id == "creative_agent"), None)
        if creative_output:
            creative_score = creative_output.analysis.get("creative_score", 0.5)
            novelty = max(novelty, creative_score)
        
        # Discovery agent activation suggests novelty
        discovery_output = next((o for o in cycle.agent_outputs if o.agent_id == "discovery_agent"), None)
        if discovery_output:
            knowledge_gaps = discovery_output.analysis.get("knowledge_gaps", [])
            if len(knowledge_gaps) > 2:
                novelty += 0.2
        
        return min(1.0, novelty)
    
    def _contains_personal_disclosure(self, text: str) -> bool:
        """Check if text contains personal disclosure markers."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.personal_disclosure_patterns)
    
    def _contains_insight(self, text: str) -> bool:
        """Check if text contains insight/breakthrough markers."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.insight_patterns)
