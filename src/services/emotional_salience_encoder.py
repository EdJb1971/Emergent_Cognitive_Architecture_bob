"""
Emotional Salience Encoder - Tags memories with emotional significance.

Inspired by the Amygdala, this service:
- Computes emotional salience scores for cognitive cycles
- Detects personal disclosures and emotionally charged moments
- Prioritizes memories for consolidation and recall
- Enables human-like memory - we remember emotional events better
"""
import logging
from typing import Optional, List

from src.models.core_models import CognitiveCycle, AgentOutput
from src.models.agent_models import EmotionalSalience

logger = logging.getLogger(__name__)


class EmotionalSalienceEncoder:
    """
    Tags memories with emotional significance, like the Amygdala.
    Enables prioritized recall of emotionally salient moments.
    """
    
    def __init__(self):
        """Initialize the emotional salience encoder."""
        logger.info("EmotionalSalienceEncoder initialized.")
    
    def compute_salience(self, cycle: CognitiveCycle) -> EmotionalSalience:
        """
        Compute emotional salience for a cognitive cycle.
        
        Args:
            cycle: The cognitive cycle to score
            
        Returns:
            EmotionalSalience object with scores and metadata
        """
        # Start with baseline
        salience_score = 0.5
        emotional_valence = "neutral"
        emotional_arousal = "low"
        contains_disclosure = False
        novelty_score = 0.5
        
        # Extract emotional output if available
        emotional_output = self._find_emotional_output(cycle.agent_outputs)
        
        if emotional_output:
            sentiment = emotional_output.analysis.get("current_sentiment", "neutral")
            confidence = emotional_output.confidence
            
            # Determine valence
            if sentiment in ["positive", "very_positive"]:
                emotional_valence = "positive"
            elif sentiment in ["negative", "very_negative"]:
                emotional_valence = "negative"
            elif sentiment == "mixed":
                emotional_valence = "mixed"
            
            # Strong emotions increase salience
            if sentiment in ["very_positive", "very_negative"]:
                salience_score += 0.3
                emotional_arousal = "high"
            elif sentiment in ["positive", "negative"]:
                salience_score += 0.15
                emotional_arousal = "medium"
            
            # High confidence emotional reads increase salience
            if confidence > 0.8:
                salience_score += 0.1
                if emotional_arousal == "medium":
                    emotional_arousal = "high"
            
            # Empathy cues suggest emotional significance
            empathy_cues = emotional_output.analysis.get("empathy_cues", [])
            if empathy_cues:
                salience_score += min(0.15, len(empathy_cues) * 0.05)
        
        # Check for personal disclosure
        contains_disclosure = self._contains_personal_disclosure(cycle.user_input)
        if contains_disclosure:
            salience_score += 0.2
            emotional_arousal = "high"  # Personal disclosure is inherently arousing
        
        # Check for breakthrough/insight moments
        if self._contains_insight(cycle.user_input):
            salience_score += 0.15
        
        # Check for relationship-defining moments (naming, role assignment, etc.)
        if self._contains_relationship_moment(cycle.user_input):
            salience_score += 0.2
            emotional_arousal = "high"
        
        # Assess novelty from memory agent
        memory_output = self._find_memory_output(cycle.agent_outputs)
        if memory_output:
            relevance = memory_output.analysis.get("relevance_score", 0.5)
            # Low relevance suggests novelty (nothing similar in memory)
            novelty_score = 1.0 - relevance
            if novelty_score > 0.7:
                salience_score += 0.15
        
        # Compute consolidation priority
        # High salience, high arousal, and novelty all increase consolidation priority
        consolidation_priority = salience_score
        if emotional_arousal == "high":
            consolidation_priority = min(1.0, consolidation_priority + 0.1)
        if novelty_score > 0.7:
            consolidation_priority = min(1.0, consolidation_priority + 0.1)
        
        # Cap salience at 1.0
        salience_score = min(1.0, salience_score)
        
        result = EmotionalSalience(
            salience_score=salience_score,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            contains_personal_disclosure=contains_disclosure,
            novelty_score=novelty_score,
            consolidation_priority=consolidation_priority
        )
        
        logger.debug(
            f"Computed salience for cycle {cycle.cycle_id}: "
            f"score={salience_score:.2f}, valence={emotional_valence}, "
            f"arousal={emotional_arousal}, disclosure={contains_disclosure}"
        )
        
        return result
    
    def _find_emotional_output(self, outputs: List[AgentOutput]) -> Optional[AgentOutput]:
        """Find emotional agent output."""
        for output in outputs:
            if output.agent_id == "emotional_agent":
                return output
        return None
    
    def _find_memory_output(self, outputs: List[AgentOutput]) -> Optional[AgentOutput]:
        """Find memory agent output."""
        for output in outputs:
            if output.agent_id == "memory_agent":
                return output
        return None
    
    def _contains_personal_disclosure(self, text: str) -> bool:
        """
        Check if input contains personal disclosure.
        
        Args:
            text: User input text
            
        Returns:
            True if personal disclosure detected
        """
        disclosure_markers = [
            "my name is",
            "i live",
            "i feel",
            "i'm worried",
            "i'm excited",
            "i'm frustrated",
            "honestly",
            "to be honest",
            "between you and me",
            "i'm scared",
            "i'm afraid",
            "i love",
            "i hate",
            "my family",
            "my partner",
            "my job",
            "i work at",
            "i'm from"
        ]
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in disclosure_markers)
    
    def _contains_insight(self, text: str) -> bool:
        """
        Check if input contains breakthrough/insight moment.
        
        Args:
            text: User input text
            
        Returns:
            True if insight moment detected
        """
        insight_markers = [
            "i understand",
            "i see",
            "that makes sense",
            "aha",
            "oh i get it",
            "now i understand",
            "that's interesting",
            "i never thought",
            "i realize",
            "i learned"
        ]
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in insight_markers)
    
    def _contains_relationship_moment(self, text: str) -> bool:
        """
        Check if input contains relationship-defining moment.
        
        Args:
            text: User input text
            
        Returns:
            True if relationship moment detected
        """
        relationship_markers = [
            "your name is",
            "call you",
            "i'll call you",
            "you are my",
            "you are a",
            "your role is",
            "act as my",
            "thank you for",
            "appreciate you",
            "you've been",
            "you seem",
            "you sound"
        ]
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in relationship_markers)
    
    def should_prioritize_in_retrieval(self, salience: EmotionalSalience) -> bool:
        """
        Determine if a memory should be prioritized in retrieval.
        
        Args:
            salience: The emotional salience object
            
        Returns:
            True if memory should be boosted in retrieval
        """
        # High salience memories get priority
        if salience.salience_score > 0.7:
            return True
        
        # Personal disclosures always prioritized
        if salience.contains_personal_disclosure:
            return True
        
        # High arousal + positive/negative valence
        if salience.emotional_arousal == "high" and salience.emotional_valence in ["positive", "negative"]:
            return True
        
        return False
    
    def get_retrieval_boost(self, salience: EmotionalSalience) -> float:
        """
        Calculate retrieval score boost for salient memories.
        
        Args:
            salience: The emotional salience object
            
        Returns:
            Multiplier for retrieval score (1.0 = no boost, >1.0 = boost)
        """
        if not self.should_prioritize_in_retrieval(salience):
            return 1.0
        
        # Base boost
        boost = 1.0
        
        # Salience-based boost
        if salience.salience_score > 0.8:
            boost += 0.3
        elif salience.salience_score > 0.7:
            boost += 0.2
        
        # Personal disclosure boost
        if salience.contains_personal_disclosure:
            boost += 0.25
        
        # High arousal boost
        if salience.emotional_arousal == "high":
            boost += 0.15
        
        return boost
