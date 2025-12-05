import logging
import json
from uuid import UUID
from typing import List, Optional

from src.core.exceptions import LLMServiceException, AgentServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import AgentOutput, MemoryQueryRequest, CognitiveCycle
from src.models.agent_models import MemoryAnalysis
from src.models.memory_models import ShortTermMemory, ConversationSummary
from src.agents.utils import UUIDEncoder, clamp_text, _compact_memory_records
from src.core.config import settings

logger = logging.getLogger(__name__)

class MemoryAgent:
    """
    Enhanced Memory Agent that intelligently utilizes both short-term and long-term memory
    to provide relevant context for the current conversation.
    """
    AGENT_ID = "memory_agent"
    EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
    DEFAULT_LIMIT = 5
    STM_CONFIDENCE_BOOST = 0.1  # Boost confidence when STM hits are found
    
    def __init__(self, llm_service: LLMIntegrationService, memory_service: MemoryService):
        self.llm_service = llm_service
        self.memory_service = memory_service
        logger.info(f"{self.AGENT_ID} initialized with enhanced memory capabilities.")

    def _calculate_memory_confidence(self, 
                                  memories: List[CognitiveCycle], 
                                  stm_hits: int,
                                  avg_relevance: float) -> float:
        """
        Calculate confidence score based on memory retrieval results.
        
        Args:
            memories: List of retrieved memories
            stm_hits: Number of matches from short-term memory
            avg_relevance: Average relevance score of retrieved memories
        
        Returns:
            float: Confidence score between 0 and 1
        """
        if not memories:
            return 0.5  # baseline confidence with no memories
        
        # Base confidence from number of memories and their relevance
        base_confidence = min(0.8, (len(memories) / self.DEFAULT_LIMIT) * avg_relevance)
        
        # Boost confidence if we have recent (STM) matches
        stm_boost = min(self.STM_CONFIDENCE_BOOST * stm_hits, 0.2)
        
        return min(0.95, base_confidence + stm_boost)

    def _analyze_memories(self, 
                        memories: List[CognitiveCycle], 
                        stm_hits: int,
                        summary: Optional[ConversationSummary] = None) -> MemoryAnalysis:
        """
        Analyze retrieved memories and create a structured analysis.
        
        Args:
            memories: List of retrieved memories
            stm_hits: Number of matches from short-term memory
            summary: Current conversation summary if available
        
        Returns:
            MemoryAnalysis: Structured analysis of retrieved memories
        """
        if not memories:
            return MemoryAnalysis(
                retrieved_context=[],
                relevance_score=0.0,
                source_memory_ids=[]
            )

        # Calculate average relevance score
        relevance_scores = [getattr(mem, "score", 0.0) for mem in memories]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Apply recency boost for STM hits
        if stm_hits > 0:
            avg_relevance = min(1.0, avg_relevance * (1.0 + (stm_hits / len(memories)) * 0.2))

        # Enhance analysis with summary context if available
        enhanced_context = []
        if summary:
            # Include current context points
            enhanced_context.extend(
                {"type": "context", "content": point} 
                for point in summary.context_points
            )
            # Include active topics
            enhanced_context.extend(
                {"type": "topic", "content": topic} 
                for topic in summary.key_topics[-3:]  # Last 3 topics
            )
            # Include relevant preferences
            enhanced_context.extend(
                {"type": "preference", "content": f"{k}: {v}"} 
                for k, v in summary.user_preferences.items()
            )

        # Project memories to compact, safe shape to prevent prompt/context explosion
        projected_records = _compact_memory_records(
            [mem.model_dump() for mem in memories],
            max_items=3,  # keep it tight – other agents also retrieve memory context
            max_user_input=300,
            max_final_response=600,
        )

        # Combine compact memory projection with enhanced context
        combined_context = [*projected_records, *enhanced_context]

        return MemoryAnalysis(
            retrieved_context=combined_context,
            relevance_score=avg_relevance,
            source_memory_ids=[str(mem.cycle_id) for mem in memories]
        )

    async def process_input(self, user_input: str, user_id: UUID) -> AgentOutput:
        """
        Enhanced processing that intelligently combines STM and LTM results.

        Args:
            user_input (str): The user's input text.
            user_id (UUID): The ID of the user.

        Returns:
            AgentOutput: Structured output containing memory analysis.

        Raises:
            AgentServiceException: If there's an error during processing.
        """
        try:
            # 1. Generate embedding for the user input
            query_embedding = await self.llm_service.generate_embedding(
                text=user_input,
                model_name=self.EMBEDDING_MODEL_NAME
            )
            logger.debug(f"{self.AGENT_ID} generated embedding for user input.")

            # 2. Query memory with enhanced request
            query_request = MemoryQueryRequest(
                user_id=user_id,
                query_text=user_input,
                query_embedding=query_embedding,
                limit=self.DEFAULT_LIMIT
            )
            
            # Get conversation summary first
            current_summary = await self.memory_service.summary_manager.get_or_create_summary(user_id)
            
            # The enhanced MemoryService will automatically check STM first
            retrieved_memories = await self.memory_service.query_memory(query_request)
            
            # Get memory access stats
            stats = self.memory_service._access_stats.get(user_id)
            stm_hits = stats.stm_hits if stats else 0
            
            # Analyze memories with summary context
            memory_analysis = self._analyze_memories(
                memories=retrieved_memories,
                stm_hits=stm_hits,
                summary=current_summary
            )
            
            # Calculate confidence based on comprehensive factors
            confidence = self._calculate_memory_confidence(
                retrieved_memories,
                stm_hits,
                memory_analysis.relevance_score
            )

            logger.info(
                f"{self.AGENT_ID} processed input successfully. "
                f"Retrieved {len(retrieved_memories)} memories "
                f"(STM hits: {stm_hits}). Confidence: {confidence:.2f}"
            )

            return AgentOutput(
                agent_id=self.AGENT_ID,
                analysis=memory_analysis.model_dump(),
                confidence=confidence,
                priority=8,  # Memory agent maintains high priority
                raw_output=json.dumps(
                    memory_analysis.model_dump().get("retrieved_context", []),
                    cls=UUIDEncoder,
                    default=str
                )
            )
        except LLMServiceException as e:
            logger.error(f"{self.AGENT_ID} failed during LLM embedding generation: {e.detail}", exc_info=True)
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"LLM embedding generation failed: {e.detail}",
                status_code=e.status_code
            )
        except APIException as e: # Catch potential errors from MemoryService placeholder
            logger.error(f"{self.AGENT_ID} failed during memory service interaction: {e.detail}", exc_info=True)
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"Memory service interaction failed: {e.detail}",
                status_code=e.status_code
            )
        except Exception as e:
            logger.exception(f"{self.AGENT_ID} encountered an unexpected error during processing.")
            raise AgentServiceException(
                agent_id=self.AGENT_ID,
                detail=f"An unexpected error occurred: {e}",
                status_code=500
            )
