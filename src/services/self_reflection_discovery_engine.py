import logging
import json
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from src.core.exceptions import APIException, LLMServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import CognitiveCycle, DiscoveredPattern, MemoryQueryRequest
from src.core.config import settings
from src.agents.utils import extract_json_from_response
from src.models.research_models import CognitiveResearchSignals, InquirySourceType
from src.services.inquiry_candidate_service import InquiryCandidateService
from src.models.autonomous_work_models import AutonomousTaskRequest, AutonomousTaskType

logger = logging.getLogger(__name__)

class SelfReflectionAndDiscoveryEngine:
    """
    A background service that orchestrates self-reflection and autonomous discovery tasks.
    It interacts with the Memory Service to analyze past cycles and upsert findings.
    Can optionally integrate with ProactiveEngagementEngine to share insights with users.
    """
    REFLECTION_MODEL_NAME = settings.LLM_MODEL_NAME
    DISCOVERY_MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(
        self, 
        llm_service: LLMIntegrationService, 
        memory_service: MemoryService,
        proactive_engine: Optional[Any] = None,
        inquiry_candidate_service: Optional[InquiryCandidateService] = None,
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.proactive_engine = proactive_engine  # Optional ProactiveEngagementEngine
        self.inquiry_candidate_service = inquiry_candidate_service
        self.autonomous_work_governor = None
        logger.info("SelfReflectionAndDiscoveryEngine initialized.")

    async def _generate_pattern_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a given text using the LLM service.
        """
        try:
            return await self.llm_service.generate_embedding(text=text)
        except LLMServiceException as e:
            logger.error(f"Failed to generate embedding for pattern: {e.detail}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error generating embedding for pattern: {e}", exc_info=True)
            return []

    async def execute_reflection(self, user_id: UUID, num_cycles: int):
        """
        Executes a self-reflection process for a specified number of past cognitive cycles.
        Identifies patterns of success/failure, generates meta-learnings, and knowledge gaps.
        Stores findings and updates cycle statuses.
        """
        logger.info(f"Initiating reflection for user {user_id} on {num_cycles} past cycles.")
        try:
            # 1. Retrieve past cognitive cycles for reflection
            recent_cycles = await self.memory_service.get_recent_cycles_for_reflection(user_id, num_cycles)

            if not recent_cycles:
                logger.info(f"No pending cognitive cycles found for user {user_id} to reflect on.")
                return

            logger.info(f"Retrieved {len(recent_cycles)} cycles for reflection for user {user_id}.")

            # 2. Synthesize context for LLM reflection
            reflection_context = "Past Cognitive Cycles for Reflection:\n\n"
            for cycle in recent_cycles:
                reflection_context += f"Cycle ID: {cycle.cycle_id}\n"
                reflection_context += f"User Input: {json.dumps(cycle.user_input)}\n"
                reflection_context += f"Final Response: {json.dumps(cycle.final_response)}\n"
                reflection_context += f"Agent Outputs Summary: {json.dumps([{{'agent_id': ao.agent_id, 'analysis_summary': ao.analysis.get('summary', str(ao.analysis))}} for ao in cycle.agent_outputs])}\n"
                reflection_context += f"Response Metadata: {json.dumps(cycle.response_metadata.model_dump() if cycle.response_metadata else {})}\n"
                reflection_context += f"Outcome Signals: {json.dumps(cycle.outcome_signals.model_dump() if cycle.outcome_signals else {})}\n"
                reflection_context += "---\n"

            reflection_prompt = f"""
            Analyze the following past cognitive cycles to identify patterns of success and failure, generate meta-learnings, and pinpoint knowledge gaps.
            Provide your analysis as a JSON array of 'DiscoveredPattern' objects. Each object should have:
            - 'pattern_type': 'meta_learning', 'success_pattern', 'failure_pattern', 'knowledge_gap'
            - 'description': A detailed explanation of the pattern or insight.
            - 'source_cycle_ids': A list of UUIDs of the cycles that informed this pattern.
            - 'metadata': Any additional relevant details.

            Example structure for one pattern:
            {{
                "pattern_type": "meta_learning",
                "description": "Empathetic responses lead to higher user engagement in emotional support scenarios.",
                "source_cycle_ids": ["uuid1", "uuid2"],
                "metadata": {{
                    "impact": "high",
                    "recommended_strategy": "prioritize empathy for emotional queries"
                }}
            }}

            Context for Reflection:
           {reflection_context}
            """

            llm_response_str = await self.llm_service.generate_text(
                prompt=reflection_prompt,
                model_name=self.REFLECTION_MODEL_NAME,
                temperature=0.5,
                max_output_tokens=2000
            )

            raw_patterns_data = extract_json_from_response(llm_response_str)
            discovered_patterns: List[DiscoveredPattern] = []

            for pattern_data in raw_patterns_data:
                # Ensure UUIDs are correctly parsed/converted
                pattern_data['user_id'] = str(user_id)
                pattern_data['source_cycle_ids'] = [UUID(cid) for cid in pattern_data.get('source_cycle_ids', [])]
                pattern = DiscoveredPattern(**pattern_data)
                pattern.embedding = await self._generate_pattern_embedding(pattern.description)
                discovered_patterns.append(pattern)

            # 3. Store discovered patterns and optionally generate proactive messages
            for pattern in discovered_patterns:
                await self.memory_service.upsert_pattern(pattern)

                if self.inquiry_candidate_service and self._is_inquiry_pattern(pattern):
                    metadata = pattern.metadata or {}
                    await self.inquiry_candidate_service.propose_offline(
                        user_id=user_id,
                        question=pattern.description,
                        hypothesis=str(metadata.get("hypothesis")) if metadata.get("hypothesis") else None,
                        signals=self._offline_inquiry_signals(pattern),
                        source_type=InquirySourceType.REFLECTION,
                        source_cycle_ids=pattern.source_cycle_ids,
                        source_pattern_ids=[pattern.pattern_id],
                        metadata={"reflection_type": "self_reflection"},
                    )
                
                # Check if pattern is worth sharing proactively
                if self.proactive_engine and self._should_share_pattern(pattern):
                    await self._submit_proactive_message(user_id, pattern)
            
            logger.info(f"Stored {len(discovered_patterns)} new patterns for user {user_id}.")

            # 4. Update reflection status of processed cycles
            for cycle in recent_cycles:
                cycle.reflection_status = "completed"
                await self.memory_service.upsert_cycle(cycle) # Update the cycle in memory
            logger.info(f"Updated reflection status for {len(recent_cycles)} cycles for user {user_id}.")

        except LLMServiceException as e:
            logger.error(f"Reflection failed for user {user_id} due to LLM error: {e.detail}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Reflection failed for user {user_id}: LLM response was not valid JSON. Error: {e}. Raw: {llm_response_str[:500]}...", exc_info=True)
            raise
        except APIException as e:
            logger.error(f"Reflection failed for user {user_id} due to MemoryService error: {e.detail}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"Unexpected error during reflection for user {user_id}: {e}", exc_info=True)
            raise

    async def execute_discovery(self, user_id: UUID, discovery_type: str, context: Optional[str]):
        """
        Executes an autonomous discovery process based on type and context.
        Generates new insights/knowledge and stores them.
        """
        logger.info(f"Initiating discovery of type '{discovery_type}' for user {user_id}.")
        try:
            discovery_prompt = f"""
            Perform an autonomous discovery process of type '{discovery_type}'.
            Context for discovery: {json.dumps(context) if context else 'None'}.

            If discovery_type is 'memory_analysis': Analyze past interactions to find latent connections or anomalies.
            If discovery_type is 'curiosity_exploration': Generate new questions or areas of inquiry based on recent interactions or knowledge gaps.
            If discovery_type is 'self_assessment': Evaluate the system's performance or understanding in a specific area.

            Provide your discovery as a JSON array of 'DiscoveredPattern' objects. Each object should have:
            - 'pattern_type': 'new_knowledge', 'curiosity', 'system_insight', 'refined_strategy'
            - 'description': A detailed explanation of the discovery.
            - 'source_cycle_ids': A list of UUIDs of any relevant cycles (can be empty).
            - 'metadata': Any additional relevant details.

            Ensure the output is a valid JSON string.
            """

            llm_response_str = await self.llm_service.generate_text(
                prompt=discovery_prompt,
                model_name=self.DISCOVERY_MODEL_NAME,
                temperature=0.7, # Higher temperature for more exploratory discovery
                max_output_tokens=1000
            )

            raw_patterns_data = extract_json_from_response(llm_response_str)
            discovered_patterns: List[DiscoveredPattern] = []
            unique_source_cycle_ids = set()

            for pattern_data in raw_patterns_data:
                pattern_data['user_id'] = str(user_id)
                # Convert source_cycle_ids from string to UUID and collect unique IDs
                cycle_ids_for_pattern = [UUID(cid) for cid in pattern_data.get('source_cycle_ids', [])]
                pattern_data['source_cycle_ids'] = cycle_ids_for_pattern
                unique_source_cycle_ids.update(cycle_ids_for_pattern)

                pattern = DiscoveredPattern(**pattern_data)
                pattern.embedding = await self._generate_pattern_embedding(pattern.description)
                discovered_patterns.append(pattern)

            # Store discovered patterns and generate proactive messages
            for pattern in discovered_patterns:
                await self.memory_service.upsert_pattern(pattern)

                if self.inquiry_candidate_service and self._is_inquiry_pattern(pattern):
                    metadata = pattern.metadata or {}
                    await self.inquiry_candidate_service.propose_offline(
                        user_id=user_id,
                        question=pattern.description,
                        hypothesis=str(metadata.get("hypothesis")) if metadata.get("hypothesis") else None,
                        signals=self._offline_inquiry_signals(pattern),
                        source_type=InquirySourceType.REFLECTION,
                        source_cycle_ids=pattern.source_cycle_ids,
                        source_pattern_ids=[pattern.pattern_id],
                        metadata={"discovery_type": discovery_type},
                    )
                
                # Generate proactive message for discoveries worth sharing
                if self.proactive_engine and self._should_share_pattern(pattern):
                    await self._submit_proactive_message(user_id, pattern)
            
            logger.info(f"Stored {len(discovered_patterns)} new discovery patterns for user {user_id} of type '{discovery_type}'.")

            # Update discovery status of relevant cycles
            if unique_source_cycle_ids:
                logger.info(f"Updating discovery status for {len(unique_source_cycle_ids)} cycles for user {user_id}.")
                for cycle_id in unique_source_cycle_ids:
                    cycle_to_update = await self.memory_service.get_cycle_by_id(user_id, cycle_id)
                    if cycle_to_update:
                        cycle_to_update.discovery_status = "completed"
                        await self.memory_service.upsert_cycle(cycle_to_update)
                        logger.debug(f"Updated discovery status for cycle {cycle_id} to 'completed'.")
                    else:
                        logger.warning(f"Could not find cycle {cycle_id} to update discovery status for user {user_id}.")
            else:
                logger.info(f"No specific source cycles identified for discovery type '{discovery_type}' for user {user_id}. No cycle statuses updated.")

        except LLMServiceException as e:
            logger.error(f"Discovery failed for user {user_id} (type: {discovery_type}) due to LLM error: {e.detail}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Discovery failed for user {user_id} (type: {discovery_type}): LLM response was not valid JSON. Error: {e}. Raw: {llm_response_str[:500]}...", exc_info=True)
            raise
        except APIException as e:
            logger.error(f"Discovery failed for user {user_id} (type: {discovery_type}) due to MemoryService error: {e.detail}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"Unexpected error during discovery for user {user_id} (type: {discovery_type}): {e}", exc_info=True)
            raise

    @staticmethod
    def _is_inquiry_pattern(pattern: DiscoveredPattern) -> bool:
        pattern_type = pattern.pattern_type.casefold()
        return "knowledge_gap" in pattern_type or "curiosity" in pattern_type

    @staticmethod
    def _offline_inquiry_signals(pattern: DiscoveredPattern) -> CognitiveResearchSignals:
        metadata = pattern.metadata or {}

        def score(name: str, default: float) -> float:
            try:
                return max(0.0, min(1.0, float(metadata.get(name, default))))
            except (TypeError, ValueError):
                return default

        is_gap = "knowledge_gap" in pattern.pattern_type.casefold()
        return CognitiveResearchSignals(
            epistemic_uncertainty=score("uncertainty", 0.85 if is_gap else 0.70),
            cognitive_conflict=score("conflict", 0.55 if is_gap else 0.35),
            novelty_prediction_error=score("novelty", 0.65),
            temporal_volatility=score("temporal_volatility", 0.15),
            task_stakes=score("salience", 0.55 if is_gap else 0.40),
            persistence_after_local_attempts=score("persistence", 0.35 if is_gap else 0.20),
            expected_information_gain=score("expected_information_gain", 0.80 if is_gap else 0.70),
            privacy_risk=0.05,
            cloud_cost=0.25,
            metacognitive_gap=True,
        )
    
    def _should_share_pattern(self, pattern: DiscoveredPattern) -> bool:
        """
        Determine if a discovered pattern is worth sharing proactively with the user.
        
        Patterns worth sharing:
        - Knowledge gaps (Bob wants to learn)
        - Interesting discoveries (new connections)
        - High-impact meta-learnings
        
        Not worth sharing:
        - System insights (internal performance)
        - Low-confidence patterns
        """
        pattern_type = pattern.pattern_type.lower()
        
        # Always share knowledge gaps (Bob asking for help)
        if "knowledge_gap" in pattern_type:
            return True
        
        # Share curiosity and discovery patterns
        if "curiosity" in pattern_type or "new_knowledge" in pattern_type:
            return True
        
        # Share high-impact meta-learnings
        if "meta_learning" in pattern_type:
            impact = pattern.metadata.get("impact", "").lower()
            if impact in ["high", "significant"]:
                return True
        
        # Don't share internal system insights
        if "system_insight" in pattern_type or "performance" in pattern_type:
            return False
        
        return False

    async def _submit_proactive_message(
        self, user_id: UUID, pattern: DiscoveredPattern
    ) -> None:
        """Proactive generation is an independently governed, opt-in task."""
        if not self.autonomous_work_governor:
            await self.proactive_engine.generate_proactive_message_from_pattern(
                user_id=user_id, pattern=pattern
            )
            return
        try:
            priority = max(0.0, min(1.0, float((pattern.metadata or {}).get("salience", 0.5))))
        except (TypeError, ValueError):
            priority = 0.5
        request = AutonomousTaskRequest(
            user_id=user_id,
            task_type=AutonomousTaskType.PROACTIVE_ENGAGEMENT,
            trigger_reason=f"shareable {pattern.pattern_type} pattern",
            payload={"pattern": pattern.model_dump(mode="json")},
            signals={"pattern_type": pattern.pattern_type},
            deduplication_key=f"proactive:{user_id}:{pattern.pattern_id}",
            priority=priority,
        )

        async def execute(_request):
            return await self.proactive_engine.generate_proactive_message_from_pattern(
                user_id=user_id, pattern=pattern
            )

        await self.autonomous_work_governor.submit(request, executor=execute)
