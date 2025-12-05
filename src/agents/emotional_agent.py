import logging
import json
from typing import Dict, Any, Optional
from uuid import UUID

from src.core.exceptions import LLMServiceException, AgentServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.services.emotional_memory_service import EmotionalMemoryService
from src.models.core_models import MemoryQueryRequest
from src.models.core_models import AgentOutput
from src.models.agent_models import EmotionalAnalysis, EmotionScore, EmotionalAgentOutput
from src.core.config import settings
from src.agents.utils import UUIDEncoder

from src.agents.utils import extract_json_from_response

logger = logging.getLogger(__name__)

class EmotionalAgent:
    """
    Specialized AI agent focused on detecting emotional tone, interpersonal dynamics, and sentiment.
    Enhanced with relational awareness and emotional intelligence through EmotionalMemoryService.
    Outputs structured data including analysis, confidence, priority, and relational context.
    """
    AGENT_ID = "emotional_agent"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(
        self, 
        llm_service: LLMIntegrationService, 
        memory_service: MemoryService,
        emotional_memory_service: EmotionalMemoryService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.emotional_memory = emotional_memory_service
        logger.info(f"{self.AGENT_ID} initialized with memory and emotional intelligence.")

    async def process_input(self, user_input: str, user_id: Optional[UUID] = None) -> AgentOutput:
        """
        Processes user input with emotional intelligence and relational awareness.

        Args:
            user_input (str): The user's input text.
            user_id (Optional[UUID]): The user's UUID for profile tracking.

        Returns:
            AgentOutput: Structured output containing enhanced emotional analysis.

        Raises:
            AgentServiceException: If there's an error during processing.
        """
        # Sanitize user_input to prevent prompt injection
        sanitized_user_input = json.dumps(user_input)
        
        # Get emotional profile
        profile = await self.emotional_memory.get_or_create_profile(user_id) if user_id else None
        
        # Get conversation summary for additional context
        summary = None
        if user_id:
            summary = await self.memory_service.summary_manager.get_or_create_summary(user_id)
        
        # Build emotionally intelligent prompt
        prompt = self._build_emotional_intelligence_prompt(
            user_input=sanitized_user_input,
            profile=profile,
            summary=summary
        )

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.5,  # Moderate temperature for nuanced emotional detection
                max_output_tokens=500
            )
            
            analysis_data = extract_json_from_response(llm_response_str)
            
            # Extract current sentiment for profile update
            current_sentiment = analysis_data.get("current_sentiment", "neutral")
            
            # Update emotional profile if we have one
            if profile and user_id:
                await self.emotional_memory.update_profile(
                    user_id=user_id,
                    current_sentiment=current_sentiment,
                    user_input=user_input,
                    conversation_summary=summary
                )
            
            # Build enhanced output with relational context
            if profile:
                output = EmotionalAgentOutput(
                    current_sentiment=current_sentiment,
                    sentiment_confidence=analysis_data.get("sentiment_confidence", 0.85),
                    user_recognition=self.emotional_memory.get_recognition_status(profile),
                    relationship_status=profile.relationship_type,
                    trust_signal=profile.trust_level,
                    emotional_shift=profile.emotional_trend,
                    empathy_cues=analysis_data.get("empathy_cues", []),
                    response_tone_recommendation=analysis_data.get("tone_recommendation", "neutral"),
                    relevant_emotional_history=self.emotional_memory.summarize_emotional_history(profile),
                    confidence=analysis_data.get("confidence", 0.85),
                    priority=2
                )
            else:
                # Fallback for users without profiles (shouldn't happen in practice)
                output = EmotionalAgentOutput(
                    current_sentiment=current_sentiment,
                    sentiment_confidence=analysis_data.get("sentiment_confidence", 0.85),
                    empathy_cues=analysis_data.get("empathy_cues", []),
                    response_tone_recommendation=analysis_data.get("tone_recommendation", "neutral"),
                    confidence=analysis_data.get("confidence", 0.85)
                )

            logger.info(
                f"{self.AGENT_ID} successfully processed input. "
                f"Sentiment: {output.current_sentiment}, "
                f"Relationship: {output.relationship_status if profile else 'N/A'}"
            )

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=output.model_dump(),
                confidence=output.confidence,
                priority=output.priority,
                raw_output=llm_response_str
            )
        except LLMServiceException as e:
            logger.error(f"{self.AGENT_ID} failed to get LLM response: {e.detail}", exc_info=True)
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"LLM interaction failed: {e.detail}",
                status_code=e.status_code
            )
        except json.JSONDecodeError as e:
            logger.error(f"{self.AGENT_ID} failed to parse LLM response as JSON: {e}. Raw response: {llm_response_str[:200]}...", exc_info=True)
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"Failed to parse LLM response for emotional analysis. Invalid JSON format. Error: {e}",
                status_code=500
            )
        except Exception as e:
            logger.exception(f"{self.AGENT_ID} encountered an unexpected error during processing.")
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"An unexpected error occurred: {e}",
                status_code=500
            )
    
    def _build_emotional_intelligence_prompt(
        self,
        user_input: str,
        profile: Optional[Any],
        summary: Optional[Any]
    ) -> str:
        """
        Build an emotionally intelligent prompt with relational and emotional context.
        
        Args:
            user_input: Sanitized user input (JSON-escaped)
            profile: EmotionalProfile if available
            summary: ConversationSummary if available
            
        Returns:
            Context-aware prompt string
        """
        # Build relational context
        relationship_context = ""
        if profile:
            if profile.user_name:
                relationship_context += f"User's name: {profile.user_name}\n"
            relationship_context += f"Relationship: {profile.relationship_type}\n"
            relationship_context += f"Trust level: {profile.trust_level:.2f}\n"
            relationship_context += f"Interaction count: {profile.interaction_count}\n"
        
        # Build emotional history
        emotional_history = ""
        if profile and profile.recent_sentiments:
            emotional_history += f"Recent emotional trajectory: {' → '.join(profile.recent_sentiments[-3:])}\n"
            emotional_history += f"Emotional trend: {profile.emotional_trend}\n"
        
        # Build conversation context
        conversation_context = ""
        if summary:
            topics = getattr(summary, 'key_topics', [])
            state = getattr(summary, 'conversation_state', 'unknown')
            if topics:
                conversation_context += f"Current topics: {', '.join(topics)}\n"
            conversation_context += f"Conversation state: {state}\n"
        
        prompt = f"""You are an emotionally intelligent agent analyzing a user's input with full relational and historical context.

RELATIONAL CONTEXT:
{relationship_context if relationship_context else "First-time user, no prior relationship."}

EMOTIONAL HISTORY:
{emotional_history if emotional_history else "No emotional history available."}

CONVERSATION CONTEXT:
{conversation_context if conversation_context else "No conversation context available."}

CURRENT USER INPUT:
{user_input}

Analyze this input with emotional intelligence, considering:
1. Current sentiment (positive/negative/neutral/mixed)
2. How they're feeling given our relationship and history
3. Empathy cues (are they excited? concerned? testing me? seeking validation? frustrated?)
4. What tone should I use in response? (warm, supportive, celebratory, concerned, neutral, playful)

Provide analysis as JSON:
{{
    "current_sentiment": "positive/negative/neutral/mixed",
    "sentiment_confidence": 0.0-1.0,
    "empathy_cues": ["specific emotional signals you detect, e.g., 'testing_memory', 'seeking_validation', 'excited', 'concerned'"],
    "tone_recommendation": "warm/supportive/celebratory/concerned/neutral/playful",
    "reasoning": "brief explanation of your emotional read",
    "confidence": 0.0-1.0
}}
"""
        return prompt
