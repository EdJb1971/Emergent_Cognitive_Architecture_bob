import logging
import json
from typing import Dict, Any, Optional

from src.core.exceptions import LLMServiceException, AgentServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import MemoryQueryRequest
from src.models.core_models import AgentOutput
from src.models.agent_models import PerceptionAnalysis
from src.core.config import settings
from src.agents.utils import UUIDEncoder
from src.agents.utils import extract_json_from_response

logger = logging.getLogger(__name__)

class PerceptionAgent:
    """
    Specialized AI agent responsible for analyzing user input to identify patterns, topics, and context type.
    Now extended to acknowledge multimodal inputs (text, visual, audio).
    Outputs structured data including analysis, confidence, and priority.
    """
    AGENT_ID = "perception_agent"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(self, llm_service: LLMIntegrationService, memory_service):
        self.llm_service = llm_service
        self.memory_service = memory_service
        logger.info(f"{self.AGENT_ID} initialized with memory integration.")

    async def process_input(self, user_input: str, user_id: Optional[str] = None, image_base64: Optional[str] = None, audio_base64: Optional[str] = None) -> AgentOutput:
        """
        Processes user input to extract perception data, acknowledging multimodal inputs.

        Args:
            user_input (str): The user's input text.
            image_base64 (Optional[str]): Optional base64 encoded image data.
            audio_base64 (Optional[str]): Optional base64 encoded audio data.

        Returns:
            AgentOutput: Structured output containing perception analysis.

        Raises:
            AgentServiceException: If there's an error during processing.
        """
        # Sanitize user_input to prevent prompt injection
        sanitized_user_input = json.dumps(user_input)

        # Retrieve memory context if user_id is provided
        memory_context = ""
        if user_id is not None:
            # Get summary context
            summary = await self.memory_service.summary_manager.get_or_create_summary(user_id)
            summary_text = summary.summary_text if hasattr(summary, "summary_text") else ""
            from src.models.core_models import MemoryQueryRequest
            query_request = MemoryQueryRequest(
                user_id=user_id,
                query_text=user_input,
                limit=3
            )
            memories = await self.memory_service.query_memory(query_request)
            memory_context = f"\nMemory Context:\nSummary: {summary_text}\nRecent Memories: " + "\n".join([
                f"- {getattr(mem, 'user_input', '')} => {getattr(mem, 'final_response', '')}" for mem in memories
            ])

        # Build a context string for multimodal inputs
        multimodal_context = ""
        if image_base64:
            multimodal_context += """
        The user has provided an image. Analyze the image to identify objects, scenes, and any relevant text.
        """
        if audio_base64:
            multimodal_context += """
        The user has provided an audio clip. Transcribe the audio and analyze its content and emotional tone.
        """

        context_for_perception = f"User Input: {sanitized_user_input}"
        if memory_context:
            context_for_perception += f"\n\n{memory_context}"

        prompt = f"""
        Analyze the following user input and identify its main topics, underlying patterns, and overall context type.
        {multimodal_context}
        Use the provided memory context to inform your analysis.
        Provide your analysis in a JSON object with the following structure:
        {{
            "topics": ["topic1", "topic2", ...],
            "patterns": ["pattern1", "pattern2", ...],
            "context_type": "e.g., 'question', 'statement', 'command', 'narrative', 'request', 'feedback'",
            "keywords": ["keyword1", "keyword2", ...],
            "image_present": {str(bool(image_base64)).lower()},
            "audio_present": {str(bool(audio_base64)).lower()},
            "image_analysis": {{ "description": "...", "objects": [], "text": "..." }},
            "audio_analysis": {{ "transcription": "...", "emotion": "..." }}
        }}
        Ensure the output is a valid JSON string.
        
        User Input: {sanitized_user_input}
        """
        
        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                image_base64=image_base64,
                audio_base64=audio_base64,
                model_name=self.MODEL_NAME,
                temperature=0.3, # Lower temperature for more focused analysis
                max_output_tokens=500
            )            
            # Attempt to parse the JSON response
            analysis_data = extract_json_from_response(llm_response_str)
            perception_analysis = PerceptionAnalysis(**analysis_data)

            logger.info(f"{self.AGENT_ID} successfully processed input. Topics: {perception_analysis.topics[:3]}. Image present: {bool(image_base64)}, Audio present: {bool(audio_base64)}")

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=perception_analysis.model_dump(),
                confidence=0.9, 
                priority=5,     
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
                detail=f"Failed to parse LLM response for perception analysis. Invalid JSON format. Error: {e}",
                status_code=500
            )
        except Exception as e:
            logger.exception(f"{self.AGENT_ID} encountered an unexpected error during processing.")
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"An unexpected error occurred: {e}",
                status_code=500
            )
