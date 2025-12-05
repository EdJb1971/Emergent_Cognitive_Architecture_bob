import logging
import json
from typing import Dict, Any, Optional, List
from uuid import UUID

from src.core.exceptions import LLMServiceException, AgentServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import MemoryQueryRequest
from src.models.core_models import AgentOutput
from src.models.agent_models import DiscoveryAnalysis
from src.core.config import settings
from src.services.web_browsing_service import WebBrowsingService
from src.agents.utils import UUIDEncoder
from src.agents.utils import compact_agent_outputs, estimate_tokens

from src.agents.utils import extract_json_from_response

logger = logging.getLogger(__name__)

class DiscoveryAgent:
    """
    Specialized AI agent that identifies knowledge gaps, generates curiosities, and proposes explorations
    based on the user input and current context. Now enhanced to leverage the Web Browsing Service for external information gathering.
    Outputs structured data including identified gaps, curiosities, confidence, and priority.
    """
    AGENT_ID = "discovery_agent"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(self, llm_service: LLMIntegrationService, memory_service, web_browsing_service: WebBrowsingService):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.web_browsing_service = web_browsing_service
        logger.info(f"{self.AGENT_ID} initialized with memory integration.")

    async def process_input(self, user_input: str, user_id: UUID, other_agent_outputs: Optional[List[AgentOutput]] = None) -> AgentOutput:
        """
        Processes user input and the outputs of other agents to identify knowledge gaps and curiosities.
        Now includes logic to trigger web browsing for external information gathering.

        Args:
            user_input (str): The user's input text.
            user_id (UUID): The ID of the user initiating the discovery (for web browsing audit).
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
            from src.models.core_models import MemoryQueryRequest
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
        Consider if any of the proposed explorations would benefit from external web research.
        
        Provide your analysis in a JSON object with the following structure:
        {{
            "knowledge_gaps": ["gap1", "gap2", ...],
            "curiosities_generated": ["curiosity1", "curiosity2", ...],
            "proposed_explorations": ["exploration1", "exploration2", ...],
            "discovery_priority": 1-10,
            "potential_web_searches": ["search query 1", "search query 2", ...] // New field for web search suggestions
        }}
        Ensure the output is a valid JSON string.

        Context for Discovery: {context_for_discovery}
        """

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=initial_discovery_prompt,
                model_name=self.MODEL_NAME,
                temperature=0.6, # Moderate temperature for balanced discovery
                max_output_tokens=1000
            )
            
            analysis_data = extract_json_from_response(llm_response_str)
            
            # Extract potential web searches
            potential_web_searches = analysis_data.pop("potential_web_searches", [])
            
            web_search_results = []
            # Determine if browsing is enabled via the service's capability method or legacy attribute
            browsing_enabled = False
            is_enabled_fn = getattr(self.web_browsing_service, "is_enabled", None)
            if callable(is_enabled_fn):
                try:
                    browsing_enabled = is_enabled_fn()
                except Exception:
                    browsing_enabled = False
            else:
                browsing_enabled = bool(getattr(self.web_browsing_service, "serpapi_api_key", None))

            if potential_web_searches and browsing_enabled:
                logger.info(f"{self.AGENT_ID}: Initiating {len(potential_web_searches)} web searches for user {user_id}.")
                for query in potential_web_searches:
                    # SEC-LLM-003 Fix: Moderate LLM-generated queries before passing to WebBrowsingService
                    moderation_result = await self.llm_service.moderate_content(query)
                    if not moderation_result.get("is_safe"):
                        logger.warning(f"{self.AGENT_ID}: LLM-generated web search query blocked due to safety concerns for user {user_id}: '{query[:50]}...'. Reason: {moderation_result.get('block_reason', 'N/A')}")
                        web_search_results.append({"query": query, "summary": "Web search query blocked due to safety concerns.", "source": "moderation_blocked"})
                        continue # Skip this unsafe query

                    try:
                        # Call the WebBrowsingService
                        result = await self.web_browsing_service.browse_and_scrape(query, user_id)
                        web_search_results.append(result)
                    except APIException as e:
                        logger.warning(f"{self.AGENT_ID}: Web browsing failed for query '{query}': {e.detail}")
                        web_search_results.append({"query": query, "summary": f"Failed to retrieve web content: {e.detail}", "source": "web_browsing_service_error"})
                    except Exception as e:
                        logger.error(f"{self.AGENT_ID}: Unexpected error during web browsing for query '{query}': {e}", exc_info=True)
                        web_search_results.append({"query": query, "summary": f"Unexpected error during web browsing: {e}", "source": "web_browsing_service_error"})
            
            # If web browsing not configured, note it to avoid repeated warnings
            if potential_web_searches and not browsing_enabled:
                logger.info(f"{self.AGENT_ID}: Web browsing is disabled (no SERPAPI_API_KEY). Skipping {len(potential_web_searches)} suggested searches.")

            # Add web search results to the analysis data
            analysis_data["web_search_results"] = web_search_results

            discovery_analysis = DiscoveryAnalysis(**analysis_data)

            logger.info(f"{self.AGENT_ID} successfully processed input. Gaps: {discovery_analysis.knowledge_gaps[:3]}. Web searches performed: {len(web_search_results)}")

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=discovery_analysis.model_dump(),
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
