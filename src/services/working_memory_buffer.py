"""
Working Memory Buffer - Maintains active task context across agent stages.

Inspired by Prefrontal Cortex (PFC) working memory, this service:
- Extracts key insights from Stage 1 (foundational) agents
- Infers user goals and attention focus
- Provides synthesized context to Stage 2 (analytical) agents
- Enables goal-directed, coherent responses
"""
import logging
from typing import List, Dict, Any, Optional

from src.models.core_models import AgentOutput
from src.models.agent_models import WorkingMemoryContext

logger = logging.getLogger(__name__)


class WorkingMemoryBuffer:
    """
    Maintains active task context in working memory, like the Prefrontal Cortex.
    Synthesizes insights from Stage 1 to guide Stage 2 processing.
    """
    
    def __init__(self):
        """Initialize the working memory buffer."""
        self.context = WorkingMemoryContext()
        logger.debug("WorkingMemoryBuffer initialized.")
    
    def reset(self):
        """Reset the working memory buffer for a new cognitive cycle."""
        self.context = WorkingMemoryContext()
        logger.debug("WorkingMemoryBuffer reset.")
    
    def set_attention_motifs(self, motifs: List[str]):
        """Apply attention motifs provided by higher-level controllers."""
        unique = list(dict.fromkeys(motifs))[:6]
        self.context.attention_motifs = unique
        logger.debug("WM: Attention motifs applied: %s", unique)

    def update_from_stage1(self, stage1_outputs: List[AgentOutput], user_input: str):
        """
        Extract key insights from Stage 1 agents to guide Stage 2.
        
        Args:
            stage1_outputs: List of agent outputs from Stage 1 (perception, emotional, memory)
            user_input: The original user input
        """
        self.context.user_input = user_input
        
        # Extract from perception agent
        perception = self._find_output(stage1_outputs, "perception_agent")
        if perception and perception.confidence > 0.0:
            analysis = perception.analysis
            self.context.topics = analysis.get("topics", [])
            self.context.multimodal = (
                analysis.get("image_present", False) or 
                analysis.get("audio_present", False) or
                analysis.get("image_analysis") is not None or
                analysis.get("audio_analysis") is not None
            )
            logger.debug(f"WM: Extracted topics from perception: {self.context.topics}")
        
        # Extract from emotional agent
        emotional = self._find_output(stage1_outputs, "emotional_agent")
        if emotional and emotional.confidence > 0.0:
            analysis = emotional.analysis
            self.context.sentiment = analysis.get("current_sentiment", "neutral")
            self.context.emotional_priority = emotional.confidence > 0.8
            
            # Extract empathy cues for attention focus
            empathy_cues = analysis.get("empathy_cues", [])
            if empathy_cues:
                self.context.attention_focus.extend(empathy_cues)
            
            logger.debug(
                f"WM: Extracted sentiment: {self.context.sentiment}, "
                f"emotional_priority: {self.context.emotional_priority}"
            )
        
        # Extract from memory agent
        memory = self._find_output(stage1_outputs, "memory_agent")
        if memory and memory.confidence > 0.0:
            analysis = memory.analysis
            self.context.recalled_memories = analysis.get("retrieved_context", [])
            self.context.memory_confidence = analysis.get("relevance_score", 0.0)
            
            # Extract entities from recalled memories for attention focus
            if self.context.memory_confidence > 0.6:
                entities = self._extract_entities_from_memories(self.context.recalled_memories)
                self.context.attention_focus.extend(entities)
            
            logger.debug(
                f"WM: Extracted {len(self.context.recalled_memories)} memories, "
                f"confidence: {self.context.memory_confidence:.2f}"
            )
        
        # Infer user goals based on context
        self.context.inferred_goals = self._infer_goals(user_input, self.context)
        
        logger.info(
            f"WM: Updated from Stage 1 - topics: {self.context.topics}, "
            f"goals: {self.context.inferred_goals}, "
            f"attention: {self.context.attention_focus[:3]}"
        )
    
    def get_enhanced_prompt_context(self) -> str:
        """
        Generate rich context string for Stage 2 agents and Cognitive Brain.
        
        Returns:
            Formatted working memory context
        """
        context_str = "\n--- WORKING MEMORY (Current Task Context) ---\n"
        
        if self.context.topics:
            context_str += f"Active Topics: {', '.join(self.context.topics)}\n"
        
        if self.context.attention_focus:
            # Deduplicate and limit
            unique_focus = list(dict.fromkeys(self.context.attention_focus))[:5]
            context_str += f"Attention Focus: {', '.join(unique_focus)}\n"

        if self.context.attention_motifs:
            context_str += f"Attention Motifs: {', '.join(self.context.attention_motifs)}\n"
        
        if self.context.inferred_goals:
            context_str += f"Inferred Goals: {', '.join(self.context.inferred_goals)}\n"
        
        if self.context.sentiment:
            context_str += f"Emotional State: {self.context.sentiment}"
            if self.context.emotional_priority:
                context_str += " (HIGH PRIORITY)"
            context_str += "\n"
        
        if self.context.recalled_memories:
            context_str += f"Memory Context: {len(self.context.recalled_memories)} relevant memories recalled"
            if self.context.memory_confidence > 0.7:
                context_str += " (HIGH CONFIDENCE)"
            context_str += "\n"
        
        if self.context.multimodal:
            context_str += "Multimodal Input: Present\n"
        
        context_str += "---\n"
        
        return context_str
    
    def _find_output(self, outputs: List[AgentOutput], agent_id: str) -> Optional[AgentOutput]:
        """Find agent output by agent_id."""
        for output in outputs:
            if output.agent_id == agent_id:
                return output
        return None
    
    def _extract_entities_from_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key entities from recalled memories for attention focus.
        
        Args:
            memories: List of recalled memory contexts
            
        Returns:
            List of entity strings
        """
        entities = []
        
        for memory in memories[:3]:  # Check top 3 memories
            # Extract from user_input if present
            if "user_input" in memory:
                user_input = memory["user_input"]
                # Simple capitalized word extraction as entities
                words = user_input.split()
                for word in words:
                    if word and word[0].isupper() and len(word) > 2:
                        # Filter out common words
                        if word.lower() not in ["i", "you", "the", "a", "an", "this", "that"]:
                            entities.append(word)
            
            # Could also extract from metadata if available
            if "metadata" in memory:
                metadata = memory.get("metadata", {})
                if "entities" in metadata:
                    entities.extend(metadata["entities"])
        
        # Deduplicate
        return list(dict.fromkeys(entities))[:5]
    
    def _infer_goals(self, user_input: str, context: WorkingMemoryContext) -> List[str]:
        """
        Infer what the user is trying to accomplish.
        Like PFC goal representation in working memory.
        
        Args:
            user_input: The user's input
            context: Current working memory context
            
        Returns:
            List of inferred goal strings
        """
        goals = []
        user_input_lower = user_input.lower()
        
        # Question detection
        if "?" in user_input or any(q in user_input_lower for q in ["what", "how", "why", "when", "where", "who"]):
            goals.append("answer_question")
        
        # Emotional response needed
        if context.emotional_priority or context.sentiment in ["negative", "very_negative"]:
            goals.append("address_emotional_state")
        
        # Memory continuity
        if context.memory_confidence > 0.7:
            goals.append("maintain_continuity")
        
        # Problem-solving
        if any(word in user_input_lower for word in ["help", "fix", "solve", "problem", "issue", "error"]):
            goals.append("solve_problem")
        
        # Creative exploration
        if any(word in user_input_lower for word in ["idea", "creative", "imagine", "what if", "brainstorm"]):
            goals.append("creative_exploration")
        
        # Clarification needed
        if any(word in user_input_lower for word in ["explain", "clarify", "understand", "confused"]):
            goals.append("provide_clarity")
        
        # Action/task execution
        if any(word in user_input_lower for word in ["do", "create", "make", "build", "implement"]):
            goals.append("execute_task")
        
        # Default: general conversation
        if not goals:
            goals.append("engage_in_dialogue")
        
        return goals
    
    def get_context_for_stage2(self, agent_id: str) -> str:
        """
        Get tailored context for a specific Stage 2 agent.
        
        Args:
            agent_id: The agent requesting context
            
        Returns:
            Tailored context string
        """
        base_context = self.get_enhanced_prompt_context()
        
        # Add agent-specific guidance
        if agent_id == "planning_agent":
            if "solve_problem" in self.context.inferred_goals:
                base_context += "\nFocus: Problem-solving strategies needed.\n"
            elif "execute_task" in self.context.inferred_goals:
                base_context += "\nFocus: Action planning and task breakdown needed.\n"
        
        elif agent_id == "creative_agent":
            if "creative_exploration" in self.context.inferred_goals:
                base_context += "\nFocus: Novel perspectives and creative solutions encouraged.\n"
            elif self.context.topics:
                base_context += f"\nFocus: Creative reframing of topics: {', '.join(self.context.topics[:2])}\n"
        
        elif agent_id == "critic_agent":
            if "provide_clarity" in self.context.inferred_goals:
                base_context += "\nFocus: Identify potential confusion or ambiguity.\n"
            elif self.context.emotional_priority:
                base_context += "\nFocus: Ensure emotional sensitivity in response.\n"
        
        elif agent_id == "discovery_agent":
            if "answer_question" in self.context.inferred_goals:
                base_context += "\nFocus: Identify knowledge gaps that need external research.\n"
        
        return base_context
