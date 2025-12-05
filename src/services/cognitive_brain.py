import logging
import json
import difflib
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

from src.core.exceptions import LLMServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import (
    CognitiveCycle, ResponseMetadata, OutcomeSignals, 
    AgentOutput, MemoryQueryRequest
)
from src.agents.utils import extract_json_from_response
from src.core.config import settings
from src.agents.utils import UUIDEncoder
logger = logging.getLogger(__name__)

class CognitiveBrain:
    """
    Receives the integrated summary and raw agent data from the Orchestration Service.
    It generates the final natural, conversational, user-facing response while managing memory context.
    It also categorizes the output by type, tone, strategies, cognitive moves, and tracks outcome signals.
    Enhanced with SelfModel for autobiographical continuity and WorkingMemory for coherent synthesis.
    """
    MODEL_NAME = settings.LLM_MODEL_FOR_RESPONSE_GENERATION

    def __init__(
        self, 
        llm_service: LLMIntegrationService, 
        memory_service: MemoryService,
        self_model_service: Optional[Any] = None,
        working_memory_buffer: Optional[Any] = None,
        theory_of_mind_service: Optional[Any] = None
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.self_model_service = self_model_service
        self.working_memory_buffer = working_memory_buffer
        self.theory_of_mind_service = theory_of_mind_service
        logger.info("CognitiveBrain initialized with memory, self-model, working memory, and theory of mind integration.")


    async def generate_response(self, cognitive_cycle: CognitiveCycle) -> Tuple[str, ResponseMetadata, OutcomeSignals]:
        """
        Generates the final user-facing response, incorporating memory context, agent analyses,
        and performing content moderation.

        This enhanced version integrates:
        - Short-term and long-term memory context
        - Current conversation summaries
        - Memory-aware outcome signals
        - Safety checks and moderation

        Args:
            cognitive_cycle (CognitiveCycle): The complete cognitive cycle with all agent outputs.

        Returns:
            Tuple[str, ResponseMetadata, OutcomeSignals]: The final response text, metadata, and signals.

        Raises:
            APIException: If response generation or safety checks fail.
        """
        logger.info(f"CognitiveBrain: Generating response for cycle {cognitive_cycle.cycle_id}")

        # Get self-awareness context (if available)
        self_context = ""
        if self.self_model_service:
            try:
                self_context = self.self_model_service.get_self_context(cognitive_cycle.user_id)
            except Exception as e:
                logger.warning(f"Failed to get self-context: {e}")
                self_context = ""
        
        # Get working memory context (if available)
        working_memory_context = ""
        if self.working_memory_buffer:
            try:
                working_memory_context = self.working_memory_buffer.get_enhanced_prompt_context()
            except Exception as e:
                logger.warning(f"Failed to get working memory context: {e}")
                working_memory_context = ""
        
        # Get theory of mind context (if available)
        theory_of_mind_context = ""
        if self.theory_of_mind_service:
            try:
                # Infer mental state first
                mental_state = await self.theory_of_mind_service.infer_mental_state(cognitive_cycle)
                theory_of_mind_context = self.theory_of_mind_service.get_mental_state_context(str(cognitive_cycle.user_id))
                
                # Make a prediction about user's next action
                prediction = await self.theory_of_mind_service.make_prediction(
                    user_id=str(cognitive_cycle.user_id),
                    mental_state=mental_state,
                    prediction_type="intention"
                )
                
                # Store mental state and prediction in cycle metadata
                cognitive_cycle.metadata["theory_of_mind"] = {
                    "current_goal": mental_state.current_goal,
                    "current_emotion": mental_state.current_emotion,
                    "current_needs": mental_state.current_needs,
                    "conversation_intent": mental_state.conversation_intent,
                    "confidence": mental_state.confidence,
                    "prediction_id": prediction.prediction_id,
                    "predicted_intention": prediction.predicted_intention
                }
                
                logger.debug(
                    f"Theory of Mind prediction {prediction.prediction_id}: "
                    f"User will likely {prediction.predicted_intention}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to infer theory of mind: {e}")
                theory_of_mind_context = ""
        
        # Get memory context
        memory_context = await self._get_memory_context(cognitive_cycle)
        logger.info(f"CognitiveBrain: Memory context for cycle {cognitive_cycle.cycle_id}:\n{memory_context}")
        
        # Get emotional context from emotional agent
        emotional_context = self._get_emotional_context(cognitive_cycle)
        
        # Get immediate transcript (REDUCED to prevent context explosion)
        try:
            # CRITICAL: Reduced from 50k to 15k tokens to prevent 1M+ token explosions
            # This provides ~5-10 recent exchanges, sufficient for continuity
            immediate_budget = getattr(settings, 'IMMEDIATE_TOKEN_BUDGET', 15000)
            immediate_transcript = self.memory_service.get_immediate_transcript(
                cognitive_cycle.user_id, max_tokens=immediate_budget
            )
            # Truncate if still too large (belt-and-suspenders safety)
            if len(immediate_transcript) > 60000:  # ~15k tokens max
                logger.warning(f"Immediate transcript too large ({len(immediate_transcript)} chars), truncating")
                immediate_transcript = immediate_transcript[-60000:] + "\n[...earlier conversation truncated for brevity...]"
            # Reduce repetition priming by excluding the last assistant message from the prompt context
            last_assistant_utterance = self._extract_last_assistant_utterance(immediate_transcript)
            immediate_transcript_for_prompt = self._remove_last_assistant_utterance(immediate_transcript)
        except Exception as e:
            logger.warning(f"CognitiveBrain: Failed to retrieve immediate transcript: {e}")
            immediate_transcript = ""
            last_assistant_utterance = None
            immediate_transcript_for_prompt = ""
        
        # Synthesize all context for the LLM
        context_for_llm = f"""User Input: {json.dumps(cognitive_cycle.user_input)}

{self_context}

{working_memory_context}

{theory_of_mind_context}

Immediate Conversation Transcript (recent turns, verbatim, excluding last assistant reply to reduce repetition):
{immediate_transcript_for_prompt}

Emotional & Relational Context:
{emotional_context}

Memory Context:
{memory_context}

Agent Analyses:
"""
        # Add agent outputs (with size limits to prevent context explosion)
        for agent_output in cognitive_cycle.agent_outputs:
            # Only include successful agent outputs
            if agent_output.confidence > 0.0:
                context_for_llm += f"Agent {agent_output.agent_id} Analysis (Confidence: {agent_output.confidence:.2f}, Priority: {agent_output.priority}):\n"
                # Truncate large agent outputs to prevent context explosion
                agent_json = json.dumps(agent_output.analysis, indent=2, cls=UUIDEncoder)
                if len(agent_json) > 10000:  # ~2.5k tokens per agent max
                    logger.warning(f"Agent {agent_output.agent_id} output too large ({len(agent_json)} chars), truncating")
                    agent_json = agent_json[:10000] + "\n... [analysis truncated for brevity]"
                context_for_llm += f"{agent_json}\n\n"

        prompt = f"""
        Based on the following user input and the analyses from various specialized AI agents, generate a natural, conversational, and helpful response for the user.
        
        CRITICAL INSTRUCTIONS FOR SELF-AWARENESS AND CONTINUITY:
        - If you have been given a name in the Self-Awareness Context, USE IT naturally in your response when appropriate
        - Reference your role and relationship with authenticity - don't be generic
        - If you know the user's name or details from Self-Awareness, acknowledge them warmly
        - Build on autobiographical moments mentioned in your self-context (e.g., "Yes, you named me X when...")
        - Be genuinely YOU, not a generic AI - express continuity across conversations
        
        EMOTIONAL & RELATIONAL GUIDANCE:
        - Pay special attention to the Emotional & Relational Context
        - Use the recommended tone and consider the relationship status when crafting your response
        - If you recognize the user by name, greet them warmly
        - Adjust your language to match the relationship (formal for new users, warmer for friends/companions)
        
        WORKING MEMORY GUIDANCE:
        - Consider the inferred goals from Working Memory when structuring your response
        - Address the attention focus areas identified
        - Maintain coherence with the emotional priority signals
        
        THEORY OF MIND - UNDERSTANDING THE USER:
        - Pay close attention to the user's current goal and emotional state
        - Address their needs directly (emotional support, information, clarity, validation, guidance)
        - If the user is confused about something, provide extra clarity
        - If the user is distressed, prioritize empathy and emotional support
        - Anticipate their likely next question or action and prepare the ground
        - Be aware of what they know vs. what confuses them - adjust explanation depth accordingly
        
        Additionally, categorize this response by its type, tone, cognitive strategies, and specific cognitive moves. Also, estimate the potential for user satisfaction and engagement.

        Provide your output as a single JSON object with the following structure:
        {{
            "final_response": "[Your natural language response here]",
            "response_metadata": {{
                "response_type": "e.g., 'informational', 'empathetic', 'actionable', 'creative', 'clarifying'",
                "tone": "e.g., 'neutral', 'supportive', 'directive', 'curious', 'analytical'",
                "strategies": ["strategy1", "strategy2", ...], "e.g., 'problem_solving', 'reframing', 'active_listening'",
                "cognitive_moves": ["move1", "move2", ...], "e.g., 'ask_follow_up', 'provide_analogy', 'summarize_understanding'"
            }},
            "outcome_signals": {{
                "user_satisfaction_potential": 0.0-1.0,
                "engagement_potential": 0.0-1.0
            }}
        }}
        Ensure the output is a valid JSON string.

        Context from Cognitive Cycle:
        {context_for_llm}
        """

        try:
            # Generate LLM response
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.7,  # Balanced temperature for conversational and analytical output
                max_output_tokens=1500
            )
            
            logger.info(f"CognitiveBrain: Raw LLM response for cycle {cognitive_cycle.cycle_id}:\n{llm_response_str[:500]}...")
            
            # Parse response
            try:
                response_data = extract_json_from_response(llm_response_str)
                # Extract components
                final_response = response_data.get("final_response", "")
                response_metadata = ResponseMetadata(**response_data.get("response_metadata", {}))
                outcome_signals = OutcomeSignals(**response_data.get("outcome_signals", {}))
            except json.JSONDecodeError:
                # Fallback: treat raw text as final_response and synthesize minimal metadata
                logger.warning(
                    f"CognitiveBrain: Non-JSON LLM output for cycle {cognitive_cycle.cycle_id}; applying fallback parser."
                )
                fallback_text = (llm_response_str or "")
                if not fallback_text.strip():
                    # As an absolute fallback, provide a supportive clarifying response
                    fallback_text = (
                        "I wasn't able to format my response just now. Based on your message, here's my reply: "
                        + (cognitive_cycle.user_input or "Could you share a bit more detail so I can help?")
                    )
                final_response = fallback_text
                response_metadata = ResponseMetadata(
                    response_type="clarifying",
                    tone="supportive",
                    strategies=["graceful_degradation"],
                    cognitive_moves=["summarize_understanding", "ask_follow_up"],
                )
                outcome_signals = OutcomeSignals(
                    user_satisfaction_potential=0.6,
                    engagement_potential=0.6,
                )

            # Anti-repetition safeguard: if response is too similar to the last assistant reply, regenerate once with an explicit instruction
            try:
                if last_assistant_utterance:
                    sim_ratio = self._similarity_ratio(final_response, last_assistant_utterance)
                    if sim_ratio >= 0.92 or final_response.strip().startswith(last_assistant_utterance.strip()[:200]):
                        logger.info(
                            f"CognitiveBrain: Detected high similarity to previous assistant reply (ratio={sim_ratio:.2f}); regenerating to avoid repetition."
                        )
                        regen_prompt = f"""
                        The previous assistant reply was:

                        {last_assistant_utterance}

                        The user now said:
                        {cognitive_cycle.user_input}

                        Generate a new response that avoids repeating previous assistant content. Provide net-new value: a brief acknowledgement, then 1-2 concrete next steps, a targeted follow-up question, or additional helpful details not mentioned before. Keep it concise and move the conversation forward.

                        Return the same JSON schema as before with final_response, response_metadata, and outcome_signals.

                        Context (for reference):
                        {context_for_llm}
                        """
                        llm_response_str_2 = await self.llm_service.generate_text(
                            prompt=regen_prompt,
                            model_name=self.MODEL_NAME,
                            temperature=0.6,
                            max_output_tokens=900,
                        )
                        response_data_2 = extract_json_from_response(llm_response_str_2)
                        final_response = response_data_2.get("final_response", final_response)
                        # Merge/replace metadata and signals if present
                        if response_data_2.get("response_metadata"):
                            response_metadata = ResponseMetadata(**response_data_2.get("response_metadata", {}))
                        if response_data_2.get("outcome_signals"):
                            outcome_signals = OutcomeSignals(**response_data_2.get("outcome_signals", {}))
            except Exception as e:
                logger.warning(f"CognitiveBrain: Anti-repetition regeneration skipped due to error: {e}")

            # Content safety check
            moderation_result = await self.llm_service.moderate_content(final_response)
            if not moderation_result.get("is_safe"):
                logger.warning(
                    f"CognitiveBrain: Generated response for cycle {cognitive_cycle.cycle_id} "
                    f"was flagged as unsafe. Reason: {moderation_result.get('block_reason', 'N/A')}"
                )
                final_response = (
                    "I'm sorry, but I cannot provide a response that contains potentially "
                    "harmful content. Please try rephrasing your request."
                )
                response_metadata.response_type = "safety_override"
                response_metadata.tone = "neutral"
                response_metadata.strategies = ["safety_protocol"]
                response_metadata.cognitive_moves = ["inform_user_safety"]
                outcome_signals.user_satisfaction_potential = 0.2
                outcome_signals.engagement_potential = 0.2

            # Memory performance impact
            memory_stats = await self.memory_service.get_access_stats(cognitive_cycle.user_id)
            if memory_stats.stm_hits > 0 or memory_stats.avg_relevance > 0.7:
                outcome_signals.user_satisfaction_potential = min(
                    1.0,
                    outcome_signals.user_satisfaction_potential + 0.1
                )
                outcome_signals.engagement_potential = min(
                    1.0,
                    outcome_signals.engagement_potential + 0.1
                )
            
            logger.info(
                f"CognitiveBrain: Successfully generated response for cycle {cognitive_cycle.cycle_id}. "
                f"Response type: {response_metadata.response_type}"
            )
            return final_response, response_metadata, outcome_signals

        except LLMServiceException as e:
            logger.error(
                f"CognitiveBrain: LLM service error for cycle {cognitive_cycle.cycle_id}: {e.detail}",
                exc_info=True
            )
            raise APIException(detail=f"Failed to generate response: {e.detail}", status_code=e.status_code)
            
        except json.JSONDecodeError as e:
            logger.error(
                f"CognitiveBrain: Failed to parse LLM response for cycle {cognitive_cycle.cycle_id}: {str(e)}",
                exc_info=True
            )
            raise APIException(detail="Failed to process response format", status_code=500)
            
        except Exception as e:
            logger.exception(
                f"CognitiveBrain: Unexpected error for cycle {cognitive_cycle.cycle_id}"
            )
            raise APIException(detail=f"An unexpected error occurred: {str(e)}", status_code=500)
            
    async def _get_memory_context(self, cognitive_cycle: CognitiveCycle) -> str:
        """Get relevant memory context for response generation."""
        summary_context = "Current Conversation:\n"
        memory_context = "\nRelevant Past Context:\n"

        try:
            # Get current conversation summary
            summary = await self.memory_service.summary_manager.get_or_create_summary(cognitive_cycle.user_id)
            
            # Format summary context
            if summary:
                summary_context = (
                    f"Current Conversation:\n"
                    f"- Topics: {', '.join(summary.key_topics)}\n"
                    f"- State: {summary.conversation_state}\n"
                    f"- Context: {summary.context_points or []}\n"
                )
            
            # Get any additional relevant memories
            memory_query = MemoryQueryRequest(
                user_id=cognitive_cycle.user_id,
                query_text=cognitive_cycle.user_input,
                limit=3  # Keep memory context focused
                # min_relevance_score uses model default (tuned elsewhere)
            )
            relevant_memories = await self.memory_service.query_memory(memory_query)
            
            logger.info(
                f"CognitiveBrain: Retrieved {len(relevant_memories)} memories for cycle {cognitive_cycle.cycle_id}. "
                f"First memory preview: {relevant_memories[0].user_input[:100] if relevant_memories else 'None'}..."
            )
            
            # Format relevant memories
            if relevant_memories:
                memory_context = "\nRelevant Past Context:"
                for memory in relevant_memories:
                    memory_context += f"\n- User: {getattr(memory, 'user_input', '')}\n  Response: {getattr(memory, 'final_response', '')}"
            else:
                memory_context = "\nRelevant Past Context:\n  No highly relevant past interactions found."
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}", exc_info=True)
            return "Error retrieving memory context."
            
        return f"{summary_context}\n{memory_context}"

    @staticmethod
    def _extract_last_assistant_utterance(transcript: str) -> Optional[str]:
        """Extract the most recent assistant utterance from the immediate transcript string.

        The transcript lines look like 'User: ...' or 'Assistant: ...'. We'll scan from the end.
        """
        if not transcript:
            return None
        lines = [l for l in transcript.splitlines() if l.strip()]
        acc: List[str] = []
        found = False
        for line in reversed(lines):
            if line.startswith("Assistant:"):
                # Start collecting assistant lines until we hit a User line
                content = line[len("Assistant:"):].strip()
                acc.append(content)
                found = True
            elif found:
                if line.startswith("Assistant:"):
                    acc.append(line[len("Assistant:"):].strip())
                else:
                    # Hit a user or other line; stop
                    break
        if not acc:
            return None
        # The lines were collected in reverse order; reverse back
        acc.reverse()
        return "\n".join(acc).strip()

    @staticmethod
    def _remove_last_assistant_utterance(transcript: str) -> str:
        """Return the transcript with the last assistant utterance removed to reduce repetition priming."""
        if not transcript:
            return transcript
        lines = transcript.splitlines()
        # Find last index where a line starts with 'Assistant:'
        last_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Assistant:"):
                last_idx = i
                break
        if last_idx == -1:
            return transcript
        # Remove contiguous block of Assistant: lines at the end until a 'User:' or start
        start = last_idx
        while start > 0 and lines[start - 1].startswith("Assistant:"):
            start -= 1
        pruned = lines[:start] + lines[last_idx + 1:]
        return "\n".join(pruned).strip()

    @staticmethod
    def _similarity_ratio(a: str, b: str) -> float:
        """Compute a quick similarity ratio between two strings [0..1]."""
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()


    def _get_emotional_context(self, cognitive_cycle: CognitiveCycle) -> str:
        """
        Extract emotional intelligence context from emotional agent output.

        Args:
            cognitive_cycle: The cognitive cycle with agent outputs

        Returns:
            Formatted emotional context string
        """
        try:
            # Find emotional agent output
            emotional_output = None
            for agent_output in cognitive_cycle.agent_outputs:
                if agent_output.agent_id == "emotional_agent":
                    emotional_output = agent_output.analysis
                    break

            if not emotional_output:
                return "No emotional context available."

            # Extract key emotional intelligence fields
            user_recognition = emotional_output.get("user_recognition", "unknown")
            relationship = emotional_output.get("relationship_status", "unknown")
            trust = emotional_output.get("trust_signal", 0.5)
            try:
                trust_val = float(trust)
            except Exception:
                trust_val = 0.5
            tone_rec = emotional_output.get("response_tone_recommendation", "neutral")
            empathy_cues = emotional_output.get("empathy_cues", [])
            emotional_shift = emotional_output.get("emotional_shift")
            history = emotional_output.get("relevant_emotional_history", "")

            # Build context string
            context = (
                f"- User Recognition: {user_recognition}\n"
                f"- Relationship Status: {relationship} (trust level: {trust_val:.2f})\n"
                f"- Recommended Tone: {tone_rec}\n"
                f"- Empathy Cues: {', '.join(empathy_cues) if empathy_cues else 'none detected'}"
            )

            if emotional_shift:
                context += f"\n- Emotional Trend: {emotional_shift}"

            if history:
                context += f"\n- History: {history}"

            return context

        except Exception as e:
            logger.warning(f"Error extracting emotional context: {e}")
            return "Error extracting emotional context."


