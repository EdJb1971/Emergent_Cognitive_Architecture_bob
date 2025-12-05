import logging
import json
from typing import Dict, Any, Optional, List

from src.core.exceptions import LLMServiceException, AgentServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import MemoryQueryRequest
from src.models.core_models import AgentOutput
from src.models.agent_models import PlanningAnalysis
from src.core.config import settings
from src.agents.utils import UUIDEncoder
from src.agents.utils import extract_json_from_response
from src.agents.utils import compact_agent_outputs, estimate_tokens

logger = logging.getLogger(__name__)

class PlanningAgent:
    """
    Specialized AI agent that evaluates possible response options and their feasibility based on the user input
    and other agent insights. Outputs structured data including potential responses, feasibility, confidence, and priority.
    """
    AGENT_ID = "planning_agent"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(self, llm_service: LLMIntegrationService, memory_service):
        self.llm_service = llm_service
        self.memory_service = memory_service
        logger.info(f"{self.AGENT_ID} initialized with memory integration.")

    async def process_input(self, user_input: str, user_id: Optional[str] = None, other_agent_outputs: Optional[List[AgentOutput]] = None) -> AgentOutput:
        """
        Processes user input and the outputs of other agents to generate planning analysis.

        Args:
            user_input (str): The user's input text.
            other_agent_outputs (Optional[List[AgentOutput]]): List of outputs from other agents.

        Returns:
            AgentOutput: Structured output containing planning analysis.

        Raises:
            AgentServiceException: If there's an error during processing.
        """
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

        context_for_planning = f"User Input: {sanitized_user_input}"
        if other_agent_outputs:
            agent_summaries = compact_agent_outputs(other_agent_outputs, per_agent_max_chars=8000, total_max_chars=30000)
            context_for_planning += f"\n\nOther Agent Outputs for Context:\n{agent_summaries}"

        if memory_context:
            context_for_planning += f"\n\n{memory_context}"

        prompt = f"""
        Based on the following context, evaluate possible response options, their feasibility, strategic considerations, and potential risks.
        Use the provided memory context and agent outputs to inform your planning analysis.
        Provide your analysis in a JSON object with the following structure:
        {{
            "feasible_options": ["option1", "option2", ...],
            "strategic_considerations": ["consideration1", "consideration2", ...],
            "risks": ["risk1", "risk2", ...],
            "recommended_action": "Optional recommended action or strategy, or null if none."
        }}
        Ensure the output is a valid JSON string.

        Context for Planning: {context_for_planning}
        """

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.4, # Moderate temperature for balanced planning
                max_output_tokens=700
            )
            
            analysis_data = extract_json_from_response(llm_response_str)
            planning_analysis = PlanningAnalysis(**analysis_data)

            logger.info(f"{self.AGENT_ID} successfully processed input. Options: {planning_analysis.feasible_options[:3]}")

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=planning_analysis.model_dump(),
                confidence=0.8, 
                priority=7,     
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
                detail=f"Failed to parse LLM response for planning analysis. Invalid JSON format. Error: {e}",
                status_code=500
            )
        except Exception as e:
            logger.exception(f"{self.AGENT_ID} encountered an unexpected error during processing.")
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"An unexpected error occurred: {e}",
                status_code=500
            )
