"""
Memory Consolidation Service - Sleep-like Memory Processing

Inspired by memory consolidation during sleep, this service:
- Runs in the background during idle periods
- Replays and strengthens important memories
- Extracts semantic knowledge from episodic experiences
- Discovers patterns across multiple episodes
- Prioritizes emotionally salient and novel memories

This mimics how the brain consolidates memories during sleep/rest.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from src.models.agent_models import MemoryConsolidationJob, EpisodicMemory, SemanticMemory
from src.models.core_models import CognitiveCycle
from src.services.memory_service import MemoryService
from src.services.autobiographical_memory_system import AutobiographicalMemorySystem
from src.services.llm_integration_service import LLMIntegrationService
from src.models.research_models import CognitiveResearchSignals, InquirySourceType
from src.services.inquiry_candidate_service import InquiryCandidateService
from src.services.salience_network import SalienceNetwork
from src.services.sleep_cycle_ledger import SleepCycleLedger
from src.models.sleep_models import SleepLedgerEventType
from src.providers.base import ProviderPurpose, ProviderRequest
from src.agents.utils import extract_json_from_response

logger = logging.getLogger(__name__)


class MemoryConsolidationService:
    """
    Background service for memory consolidation and replay.
    Mimics sleep-like memory processing in the brain.
    """
    
    def __init__(
        self,
        memory_service: MemoryService,
        autobiographical_system: AutobiographicalMemorySystem,
        llm_service: LLMIntegrationService,
        proactive_engine: Optional[Any] = None,
        inquiry_candidate_service: Optional[InquiryCandidateService] = None,
        salience_network: Optional[SalienceNetwork] = None,
        audit_ledger: Optional[SleepCycleLedger] = None,
        consolidation_interval_minutes: float = 30.0,
    ):
        self.memory_service = memory_service
        self.autobiographical_system = autobiographical_system
        self.llm_service = llm_service
        self.proactive_engine = proactive_engine  # Optional ProactiveEngagementEngine
        self.inquiry_candidate_service = inquiry_candidate_service
        self.salience_network = salience_network
        self.audit_ledger = audit_ledger
        self.consolidation_jobs: Dict[str, MemoryConsolidationJob] = {}
        self.consolidation_interval_minutes = max(0.0, consolidation_interval_minutes)
        self.last_consolidation: Dict[str, datetime] = {}  # user_id -> last consolidation time
        logger.info("MemoryConsolidationService initialized.")
    
    async def should_consolidate(self, user_id: str) -> bool:
        """
        Check if consolidation should run for this user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if consolidation should run
        """
        last_time = self.last_consolidation.get(user_id)
        if not last_time:
            return True  # Never consolidated before
        
        time_since_last = datetime.utcnow() - last_time
        return time_since_last.total_seconds() / 60 >= self.consolidation_interval_minutes

    def record_consolidation_completed(self, user_id: str) -> None:
        """Start the cooldown only after the owning sleep pipeline fully succeeds."""
        self.last_consolidation[user_id] = datetime.utcnow()
    
    async def create_consolidation_job(
        self,
        user_id: str,
        consolidation_type: str = "episodic_to_semantic",
        cycle_ids: Optional[List[str]] = None,
        priority: float = 0.5,
        run_id: Optional[UUID] = None,
        salience_advisory: Optional[Dict[str, Any]] = None,
    ) -> MemoryConsolidationJob:
        """
        Create a new consolidation job.
        
        Args:
            user_id: User identifier
            consolidation_type: Type of consolidation to perform
            cycle_ids: Optional specific cycles to consolidate
            priority: Job priority
            
        Returns:
            MemoryConsolidationJob object
        """
        job_id = str(uuid4())
        
        # If no specific cycles, get recent high-priority cycles
        if not cycle_ids:
            cycle_ids, salience_advisory = (
                await self.get_consolidation_candidates(
                    user_id,
                    consolidation_type=consolidation_type,
                )
            )
        
        job = MemoryConsolidationJob(
            job_id=job_id,
            run_id=str(run_id) if run_id else None,
            user_id=user_id,
            cycle_ids_to_process=cycle_ids,
            consolidation_type=consolidation_type,
            priority=priority,
            status="pending",
            salience_advisory=salience_advisory,
        )
        
        await self._audit_job(SleepLedgerEventType.JOB_CREATED, job, run_id=run_id)
        self.consolidation_jobs[job_id] = job
        logger.info(f"Created consolidation job {job_id} for user {user_id}: {len(cycle_ids)} cycles")
        
        return job
    
    async def _get_consolidation_candidates(self, user_id: str, limit: int = 20) -> List[str]:
        """
        Get cycle IDs that are candidates for consolidation.
        Prioritizes high consolidation_priority cycles from recent history.
        """
        cycle_ids, _ = await self.get_consolidation_candidates(
            user_id,
            consolidation_type="episodic_to_semantic",
            limit=limit,
        )
        return cycle_ids

    async def get_consolidation_candidates(
        self,
        user_id: str,
        consolidation_type: str = "episodic_to_semantic",
        limit: int = 20,
    ) -> tuple[List[str], Optional[Dict[str, Any]]]:
        """Return the unchanged baseline selection plus an optional replay ranking."""
        try:
            # Get recent cycles with high consolidation priority
            cycles = await self.memory_service.get_user_cycles(
                user_id=UUID(user_id),
                limit=limit
            )
            
            # Filter for cycles with consolidation metadata
            candidates = []
            for cycle in cycles:
                consolidation_meta = cycle.metadata.get("consolidation_metadata", {})
                priority = consolidation_meta.get("consolidation_priority", 0.0)
                completed_types = cycle.metadata.get("sleep_consolidation", {})

                if priority > 0.6 and consolidation_type not in completed_types:
                    candidates.append(str(cycle.cycle_id))
            
            salience_advisory = None
            if self.salience_network and self.salience_network.enabled:
                assessment = self.salience_network.assess_memories(
                    cycles,
                    goal_terms=("memory replay", "learning", "unresolved pattern"),
                    top_k=limit,
                )
                salience_advisory = assessment.model_dump(mode="json")
                salience_advisory["baseline_selected_ids"] = list(candidates)

            logger.debug(f"Found {len(candidates)} consolidation candidates for user {user_id}")
            return candidates, salience_advisory
            
        except Exception as e:
            logger.error(f"Error getting consolidation candidates: {e}")
            raise
    
    async def execute_consolidation_job(
        self,
        job_id: str,
        *,
        run_id: Optional[UUID] = None,
    ) -> MemoryConsolidationJob:
        """
        Execute a consolidation job in the background.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Updated MemoryConsolidationJob
        """
        job = self.consolidation_jobs.get(job_id)
        if not job:
            raise KeyError(f"Consolidation job {job_id} not found")
        
        effective_run_id = run_id or (UUID(job.run_id) if job.run_id else None)
        job.status = "processing"
        await self._audit_job(
            SleepLedgerEventType.JOB_STARTED,
            job,
            run_id=effective_run_id,
        )
        logger.info(f"Executing consolidation job {job_id}: {job.consolidation_type}")
        
        try:
            if job.consolidation_type == "episodic_to_semantic":
                await self._consolidate_episodic_to_semantic(job)
            elif job.consolidation_type == "memory_replay":
                await self._replay_memories(job)
            elif job.consolidation_type == "pattern_extraction":
                await self._extract_patterns(job)
            else:
                raise ValueError(f"Unknown consolidation type: {job.consolidation_type}")

            await self._mark_cycles_processed(job)
            
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            if effective_run_id is None:
                self.record_consolidation_completed(job.user_id)
            await self._audit_job(
                SleepLedgerEventType.JOB_COMPLETED,
                job,
                run_id=effective_run_id,
            )
            
            logger.info(
                f"Completed consolidation job {job_id}: "
                f"episodes={job.episodes_created}, semantic={job.semantic_concepts_extracted}, "
                f"patterns={len(job.patterns_discovered)}"
            )

            if self.inquiry_candidate_service and job.patterns_discovered:
                try:
                    await self._queue_dream_inquiries(job)
                except Exception as error:
                    logger.warning(
                        "Dream inquiry handoff failed for consolidation job %s: %s",
                        job.job_id,
                        error,
                    )
            
            # 🎯 After "dreaming", Bob might want to share interesting insights
            if self.proactive_engine and job.patterns_discovered:
                try:
                    from uuid import UUID
                    user_uuid = UUID(job.user_id)
                    
                    # Check if any patterns from consolidation are worth sharing
                    for pattern in job.patterns_discovered:
                        # Randomly select interesting patterns (don't spam all of them)
                        import random
                        if random.random() < 0.3:  # 30% chance per pattern
                            await self.proactive_engine.generate_proactive_message_from_pattern(
                                user_id=user_uuid,
                                pattern=pattern
                            )
                            logger.info(f"Generated proactive message from consolidation pattern for user {job.user_id}")
                            break  # Only queue one message per consolidation cycle
                except Exception as e:
                    logger.warning(f"Failed to generate proactive message from consolidation: {e}")
            
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            await asyncio.shield(
                self._audit_job(
                    SleepLedgerEventType.JOB_CANCELLED,
                    job,
                    run_id=effective_run_id,
                )
            )
            raise
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await self._audit_job(
                SleepLedgerEventType.JOB_FAILED,
                job,
                run_id=effective_run_id,
            )
            logger.error(f"Consolidation job {job_id} failed: {e}", exc_info=True)
        
        return job

    async def _audit_job(
        self,
        event_type: SleepLedgerEventType,
        job: MemoryConsolidationJob,
        *,
        run_id: Optional[UUID],
    ) -> None:
        if not self.audit_ledger:
            return
        await self.audit_ledger.append(
            event_type,
            user_id=UUID(job.user_id),
            run_id=run_id,
            job_id=UUID(job.job_id),
            payload={"job": job.model_dump(mode="json")},
        )

    async def _mark_cycles_processed(self, job: MemoryConsolidationJob) -> None:
        completed_at = datetime.utcnow().isoformat()
        for cycle_id in job.cycle_ids_to_process:
            updated = await self.memory_service.patch_cycle_metadata(
                UUID(job.user_id),
                UUID(cycle_id),
                {
                    "sleep_consolidation": {
                        job.consolidation_type: {
                            "completed_at": completed_at,
                            "job_id": job.job_id,
                            "run_id": job.run_id,
                        }
                    }
                },
            )
            if updated is None:
                raise KeyError(f"Cycle {cycle_id} disappeared during consolidation")

    async def _queue_dream_inquiries(self, job: MemoryConsolidationJob) -> None:
        """Persist unresolved offline discoveries; this path has no research-provider access."""
        from uuid import UUID

        markers = ("unknown", "unclear", "question", "missing", "contradiction", "anomaly", "needs research")
        source_cycle_ids = []
        for cycle_id in job.cycle_ids_to_process:
            try:
                source_cycle_ids.append(UUID(str(cycle_id)))
            except (TypeError, ValueError):
                continue
        for pattern in job.patterns_discovered:
            question = str(pattern).strip()
            if not question or not any(marker in question.casefold() for marker in markers):
                continue
            await self.inquiry_candidate_service.propose_offline(
                user_id=UUID(job.user_id),
                question=question,
                signals=CognitiveResearchSignals(
                    epistemic_uncertainty=0.80,
                    cognitive_conflict=0.70,
                    novelty_prediction_error=0.75,
                    temporal_volatility=0.15,
                    task_stakes=max(0.45, min(0.85, float(job.priority))),
                    persistence_after_local_attempts=0.40,
                    expected_information_gain=0.80,
                    privacy_risk=0.05,
                    cloud_cost=0.25,
                    metacognitive_gap=True,
                ),
                source_type=InquirySourceType.DREAM,
                source_cycle_ids=source_cycle_ids,
                metadata={"consolidation_job_id": job.job_id},
            )
    
    async def _consolidate_episodic_to_semantic(self, job: MemoryConsolidationJob):
        """
        Convert episodic memories into semantic knowledge.
        Example: Multiple episodes of user saying "I like coffee" -> semantic fact "user prefers coffee"
        """
        # Get the cycles to consolidate
        cycles = []
        for cycle_id in job.cycle_ids_to_process:
            cycle = await self.memory_service.get_cycle_by_id(
                UUID(job.user_id),
                UUID(cycle_id),
            )
            if cycle is None:
                raise KeyError(f"Cycle {cycle_id} not found for episodic consolidation")
            cycles.append(cycle)
        
        if not cycles:
            raise ValueError(f"Job {job.job_id} has no cycles to consolidate")
        
        # Create episodic memories from high-significance cycles
        episode_ids_by_cycle: Dict[str, str] = {}
        for cycle in cycles:
            consolidation_priority = cycle.metadata.get("consolidation_metadata", {}).get("consolidation_priority", 0.5)
            
            if consolidation_priority > 0.7:  # High significance
                # Generate rich narrative using LLM
                narrative, provider, model = await self._generate_episode_narrative(cycle)
                
                # Extract emotional tone
                contextual_bindings = cycle.metadata.get("contextual_bindings", {})
                emotional_valence = contextual_bindings.get("emotional_valence", "neutral")
                
                # Extract key insights
                key_insights = await self._extract_insights_from_cycle(cycle)
                
                # Create episodic memory
                episode = await self.autobiographical_system.create_episodic_memory(
                    cycle=cycle,
                    narrative=narrative,
                    significance=consolidation_priority,
                    emotional_tone=emotional_valence,
                    key_insights=key_insights,
                    consolidation_job_id=job.job_id,
                    generation_provider=provider,
                    generation_model=model,
                )
                episode_ids_by_cycle[str(cycle.cycle_id)] = episode.episode_id
                job.episodes_created += 1
        
        # Extract semantic concepts from the episodes
        # Group cycles by topic and extract learned facts
        semantic_concepts = await self._extract_semantic_concepts_from_cycles(
            cycles,
            job,
            episode_ids_by_cycle,
        )
        job.semantic_concepts_extracted = len(semantic_concepts)
    
    async def _generate_episode_narrative(
        self,
        cycle: CognitiveCycle,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Generate a rich narrative description of the episode using LLM."""
        try:
            prompt = f"""
Generate a concise but vivid narrative description of this interaction episode.
Focus on what happened, the emotional tone, and any significant moments.

User Input: {cycle.user_input}

System Response: {cycle.final_response[:200] if cycle.final_response else "No response"}

Context: {cycle.metadata.get('contextual_bindings', {}).get('topics', [])}

Provide a 2-3 sentence narrative in past tense, as if remembering this moment.
"""
            
            narrative, provider, model = await self._generate_background_text(
                prompt=prompt,
                max_output_tokens=150,
                temperature=0.7
            )
            
            return narrative.strip(), provider, model
            
        except Exception as e:
            logger.warning(f"Could not generate narrative, using default: {e}")
            return f"User asked: {cycle.user_input[:100]}...", None, None
    
    async def _extract_insights_from_cycle(self, cycle: CognitiveCycle) -> List[str]:
        """Extract key insights or learnings from the cycle."""
        insights = []
        
        # Check discovery agent output
        for output in cycle.agent_outputs:
            if output.agent_id == "discovery_agent":
                proposed = output.analysis.get("proposed_explorations", [])
                insights.extend(proposed[:2])  # Take top 2
        
        # Check planning agent output
        for output in cycle.agent_outputs:
            if output.agent_id == "planning_agent":
                recommended = output.analysis.get("recommended_action")
                if recommended:
                    insights.append(f"Action: {recommended}")
        
        return insights[:3]  # Max 3 insights
    
    async def _extract_semantic_concepts_from_cycles(
        self,
        cycles: List[CognitiveCycle],
        job: MemoryConsolidationJob,
        episode_ids_by_cycle: Dict[str, str],
    ) -> List[SemanticMemory]:
        """
        Extract semantic concepts (facts, preferences, patterns) from multiple cycles.
        """
        concepts = []
        
        # Group cycles by topic
        topic_groups: Dict[str, List[CognitiveCycle]] = {}
        for cycle in cycles:
            bindings = cycle.metadata.get("contextual_bindings", {})
            topics = bindings.get("topics", [])
            
            for topic in topics:
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(cycle)
        
        # For each topic with multiple occurrences, extract semantic knowledge
        for topic, topic_cycles in topic_groups.items():
            if len(topic_cycles) >= 2:  # Need multiple instances to form a pattern
                # Analyze for user preferences or facts
                concept = await self._analyze_topic_for_semantic_knowledge(
                    topic,
                    topic_cycles,
                    job,
                    episode_ids_by_cycle,
                )
                if concept:
                    concepts.append(concept)
        
        return concepts
    
    async def _analyze_topic_for_semantic_knowledge(
        self,
        topic: str,
        cycles: List[CognitiveCycle],
        job: MemoryConsolidationJob,
        episode_ids_by_cycle: Dict[str, str],
    ) -> Optional[SemanticMemory]:
        """Analyze multiple cycles about a topic to extract semantic knowledge."""
        try:
            # Build summary of cycles
            cycle_summaries = []
            for cycle in cycles[:5]:  # Max 5 cycles
                cycle_summaries.append(f"- {cycle.user_input[:100]}")
            
            prompt = f"""
Analyze these interactions about "{topic}" and extract ONE key fact, preference, or concept that the user has revealed.

Interactions:
{chr(10).join(cycle_summaries)}

Provide:
1. Concept name (short, e.g., "prefers_visual_learning")
2. Description (one sentence)
3. Category (user_preference, user_fact, user_goal, system_capability)

Format as JSON: {{"concept_name": "...", "description": "...", "category": "..."}}
"""
            
            response, provider, model = await self._generate_background_text(
                prompt=prompt,
                max_output_tokens=150,
                temperature=0.3,
                response_json=True,
            )
            
            data = extract_json_from_response(response)
            source_cycle_ids = [str(c.cycle_id) for c in cycles]
            source_episode_ids = [
                episode_ids_by_cycle[cycle_id]
                for cycle_id in source_cycle_ids
                if cycle_id in episode_ids_by_cycle
            ]
            
            concept = await self.autobiographical_system.extract_semantic_memory(
                user_id=job.user_id,
                concept_name=data["concept_name"],
                description=data["description"],
                category=data["category"],
                source_episodes=source_episode_ids,
                source_cycle_ids=source_cycle_ids,
                consolidation_job_id=job.job_id,
                generation_provider=provider,
                generation_model=model,
                confidence=0.7,
            )
            
            return concept
            
        except Exception as e:
            logger.warning(f"Could not extract semantic concept for topic '{topic}': {e}")
            return None
    
    async def _replay_memories(self, job: MemoryConsolidationJob):
        """
        Replay memories to strengthen them (like memory replay during sleep).
        This is mostly metadata updates - marking memories as "replayed" which can boost retrieval.
        """
        logger.info(f"Replaying {len(job.cycle_ids_to_process)} memories for strengthening")
        
        for cycle_id in job.cycle_ids_to_process:
            cycle = await self.memory_service.get_cycle_by_id(
                UUID(job.user_id),
                UUID(cycle_id),
            )
            if cycle is None:
                raise KeyError(f"Cycle {cycle_id} not found for replay")
            consolidation = cycle.metadata.get("consolidation_metadata", {})
            replay_count = int(consolidation.get("replay_count", 0)) + 1
            await self.memory_service.patch_cycle_metadata(
                UUID(job.user_id),
                UUID(cycle_id),
                {
                    "consolidation_metadata": {
                        "replay_count": replay_count,
                        "last_accessed": datetime.utcnow().isoformat(),
                    }
                },
            )
        job.patterns_discovered.append(f"Replayed {len(job.cycle_ids_to_process)} memories")
    
    async def _extract_patterns(self, job: MemoryConsolidationJob):
        """
        Extract patterns across multiple memories.
        Example: User often asks questions in the evening, user prefers detailed explanations, etc.
        """
        cycles = []
        for cycle_id in job.cycle_ids_to_process[:10]:  # Analyze up to 10 cycles
            cycle = await self.memory_service.get_cycle_by_id(
                UUID(job.user_id),
                UUID(cycle_id),
            )
            if cycle is None:
                raise KeyError(f"Cycle {cycle_id} not found for pattern extraction")
            cycles.append(cycle)
        
        if len(cycles) < 3:
            logger.warning("Not enough cycles to extract patterns")
            return
        
        # Analyze patterns using LLM
        try:
            # Build context from cycles
            cycle_contexts = []
            for cycle in cycles:
                bindings = cycle.metadata.get("contextual_bindings", {})
                cycle_contexts.append(
                    f"- Time: {bindings.get('time_of_day', 'unknown')}, "
                    f"Depth: {bindings.get('conversation_depth', 'unknown')}, "
                    f"Topics: {bindings.get('topics', [])}"
                )
            
            prompt = f"""
Analyze these interaction patterns and identify 2-3 behavioral patterns or preferences:

{chr(10).join(cycle_contexts[:8])}

Provide patterns as a JSON array: ["pattern1", "pattern2", ...]
Examples: "tends to ask complex questions in the evening", "prefers deep conversations over small talk"
"""
            
            response, _provider, _model = await self._generate_background_text(
                prompt=prompt,
                max_output_tokens=200,
                temperature=0.5,
                response_json=True,
            )

            patterns = extract_json_from_response(response)
            
            if isinstance(patterns, list):
                job.patterns_discovered.extend(patterns)
                logger.info(f"Discovered {len(patterns)} patterns for user {job.user_id}")
            
        except Exception as e:
            logger.warning(f"Could not extract patterns: {e}")

    async def _generate_background_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
        response_json: bool = False,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Use the local provider's background semaphore when that seam is available."""
        generation_provider = getattr(
            self.llm_service,
            "generation_provider",
            self.llm_service,
        )
        generate = getattr(type(generation_provider), "generate", None)
        if callable(generate):
            result = await generation_provider.generate(
                ProviderRequest(
                    purpose=ProviderPurpose.BACKGROUND,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    response_json=response_json,
                )
            )
            return result.content, result.provider, result.model

        response = await generation_provider.generate_text(
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_json=response_json,
        )
        capabilities = getattr(generation_provider, "capabilities", None)
        return (
            response,
            getattr(capabilities, "provider", None),
            getattr(capabilities, "model", None),
        )
