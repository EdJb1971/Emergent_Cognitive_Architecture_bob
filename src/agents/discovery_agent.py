import logging
import json
from typing import Optional, List
from uuid import UUID

from src.core.exceptions import LLMServiceException, AgentServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.models.core_models import AgentOutput, MemoryQueryRequest
from src.models.agent_models import DiscoveryAnalysis
from src.core.config import settings
from src.services.research_service import ResearchService
from src.agents.utils import compact_agent_outputs, extract_json_from_response

logger = logging.getLogger(__name__)

class DiscoveryAgent:
    """
    Specialized AI agent that identifies knowledge gaps, generates curiosities, and proposes explorations
    based on the user input and current context. External research can only occur through
    the deterministic ResearchService policy boundary.
    Outputs structured data including identified gaps, curiosities, confidence, and priority.
    """
    AGENT_ID = "discovery_agent"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(self, llm_service: LLMIntegrationService, memory_service, research_service: ResearchService):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.research_service = research_service
        logger.info(f"{self.AGENT_ID} initialized with memory and governed research integration.")

    async def process_input(self, user_input: str, user_id: Optional[UUID] = None, other_agent_outputs: Optional[List[AgentOutput]] = None) -> AgentOutput:
        """
        Processes user input and the outputs of other agents to identify knowledge gaps and curiosities.
        Research suggestions from the LLM are advisory. The original user query must
        independently satisfy the local escalation policy before any provider is called.

        Args:
            user_input (str): The user's input text.
            user_id (Optional[UUID]): The user initiating the discovery, used for audit correlation.
            other_agent_outputs (Optional[List[AgentOutput]]): List of outputs from other agents.

        Returns:
            AgentOutput: Structured output containing discovery analysis.

        Raises:
            AgentServiceException: If there's an error during processing.
        """
        sanitized_user_input = json.dumps(user_input)

        # Retrieve memory context for the user
        memory_context = ""
        if user_id is not None:
            summary = await self.memory_service.summary_manager.get_or_create_summary(user_id)
            summary_text = summary.summary_text if hasattr(summary, "summary_text") else ""
            query_request = MemoryQueryRequest(
                user_id=user_id,
                query_text=user_input,
                limit=3
            )
            memories = await self.memory_service.query_memory(query_request)
            memory_context = f"\nMemory Context:\nSummary: {summary_text}\nRecent Memories: " + "\n".join([
                f"- {getattr(mem, 'input_text', '')} => {getattr(mem, 'output_text', '')}" for mem in memories
            ])

        context_for_discovery = f"User Input: {sanitized_user_input}"
        if other_agent_outputs:
            agent_summaries = compact_agent_outputs(other_agent_outputs, per_agent_max_chars=8000, total_max_chars=30000)
            context_for_discovery += f"\n\nOther Agent Outputs for Context:\n{agent_summaries}"

        if memory_context:
            context_for_discovery += f"\n\n{memory_context}"

        # First, let the LLM identify potential knowledge gaps and propose initial explorations
        initial_discovery_prompt = f"""
        Analyze the following context to identify knowledge gaps, generate curiosities, and propose initial explorations.
        Use the provided memory context and agent outputs to inform your discovery analysis.
        Suggest concise external research queries only when fresh outside information would materially help.
        
        Provide your analysis in a JSON object with the following structure:
        {{
            "knowledge_gaps": ["gap1", "gap2", ...],
            "curiosities_generated": ["curiosity1", "curiosity2", ...],
            "proposed_explorations": ["exploration1", "exploration2", ...],
            "discovery_priority": 1-10,
            "potential_research_queries": ["research query 1", "research query 2", ...]
        }}
        Ensure the output is a valid JSON string.

        Context for Discovery: {context_for_discovery}
        """

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=initial_discovery_prompt,
                temperature=0.6, # Moderate temperature for balanced discovery
                max_output_tokens=1000,
                response_json=True,
            )
            
            analysis_data = extract_json_from_response(llm_response_str)
            
            potential_research_queries = analysis_data.pop("potential_research_queries", None)
            if potential_research_queries is None:
                potential_research_queries = analysis_data.pop("potential_web_searches", [])
            if not isinstance(potential_research_queries, list):
                potential_research_queries = []

            research_outcome = await self.research_service.consider(
                user_query=user_input,
                candidate_queries=[str(query) for query in potential_research_queries],
                source=self.AGENT_ID,
            )
            analysis_data["research"] = research_outcome.model_dump(mode="json")
            analysis_data["web_search_results"] = []

            discovery_analysis = DiscoveryAnalysis(**analysis_data)

            logger.info(
                "%s successfully processed input. Gaps: %s. Research disposition: %s; packets: %d",
                self.AGENT_ID,
                discovery_analysis.knowledge_gaps[:3],
                research_outcome.decision.disposition.value,
                len(research_outcome.packets),
            )

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=discovery_analysis.model_dump(mode="json"),
                confidence=0.75, 
                priority=3,     
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
                detail=f"Failed to parse LLM response for discovery analysis. Invalid JSON format. Error: {e}",
                status_code=500
            )
        except AgentServiceException: # Re-raise AgentServiceExceptions (e.g., from moderation)
            raise
        except Exception as e:
            logger.exception(f"{self.AGENT_ID} encountered an unexpected error during processing.")
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"An unexpected error occurred: {e}",
                status_code=500
            )
