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
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from src.models.agent_models import MemoryConsolidationJob, EpisodicMemory, SemanticMemory
from src.models.core_models import CognitiveCycle
from src.services.memory_service import MemoryService
from src.services.autobiographical_memory_system import AutobiographicalMemorySystem
from src.services.llm_integration_service import LLMIntegrationService

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
        proactive_engine: Optional[Any] = None
    ):
        self.memory_service = memory_service
        self.autobiographical_system = autobiographical_system
        self.llm_service = llm_service
        self.proactive_engine = proactive_engine  # Optional ProactiveEngagementEngine
        self.consolidation_jobs: Dict[str, MemoryConsolidationJob] = {}
        self.consolidation_interval_minutes = 30  # Run every 30 minutes
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
    
    async def create_consolidation_job(
        self,
        user_id: str,
        consolidation_type: str = "episodic_to_semantic",
        cycle_ids: Optional[List[str]] = None,
        priority: float = 0.5
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
            cycle_ids = await self._get_consolidation_candidates(user_id)
        
        job = MemoryConsolidationJob(
            job_id=job_id,
            user_id=user_id,
            cycle_ids_to_process=cycle_ids,
            consolidation_type=consolidation_type,
            priority=priority,
            status="pending"
        )
        
        self.consolidation_jobs[job_id] = job
        logger.info(f"Created consolidation job {job_id} for user {user_id}: {len(cycle_ids)} cycles")
        
        return job
    
    async def _get_consolidation_candidates(self, user_id: str, limit: int = 20) -> List[str]:
        """
        Get cycle IDs that are candidates for consolidation.
        Prioritizes high consolidation_priority cycles from recent history.
        """
        try:
            # Get recent cycles with high consolidation priority
            from uuid import UUID
            cycles = await self.memory_service.get_user_cycles(
                user_id=UUID(user_id),
                limit=limit
            )
            
            # Filter for cycles with consolidation metadata
            candidates = []
            for cycle in cycles:
                consolidation_meta = cycle.metadata.get("consolidation_metadata", {})
                priority = consolidation_meta.get("consolidation_priority", 0.0)
                
                if priority > 0.6:  # Only consolidate medium-high priority memories
                    candidates.append(str(cycle.cycle_id))
            
            logger.debug(f"Found {len(candidates)} consolidation candidates for user {user_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting consolidation candidates: {e}")
            return []
    
    async def execute_consolidation_job(self, job_id: str) -> MemoryConsolidationJob:
        """
        Execute a consolidation job in the background.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Updated MemoryConsolidationJob
        """
        job = self.consolidation_jobs.get(job_id)
        if not job:
            logger.error(f"Consolidation job {job_id} not found")
            return None
        
        job.status = "processing"
        logger.info(f"Executing consolidation job {job_id}: {job.consolidation_type}")
        
        try:
            if job.consolidation_type == "episodic_to_semantic":
                await self._consolidate_episodic_to_semantic(job)
            elif job.consolidation_type == "memory_replay":
                await self._replay_memories(job)
            elif job.consolidation_type == "pattern_extraction":
                await self._extract_patterns(job)
            else:
                logger.warning(f"Unknown consolidation type: {job.consolidation_type}")
            
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            self.last_consolidation[job.user_id] = datetime.utcnow()
            
            logger.info(
                f"Completed consolidation job {job_id}: "
                f"episodes={job.episodes_created}, semantic={job.semantic_concepts_extracted}, "
                f"patterns={len(job.patterns_discovered)}"
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
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            logger.error(f"Consolidation job {job_id} failed: {e}", exc_info=True)
        
        return job
    
    async def _consolidate_episodic_to_semantic(self, job: MemoryConsolidationJob):
        """
        Convert episodic memories into semantic knowledge.
        Example: Multiple episodes of user saying "I like coffee" -> semantic fact "user prefers coffee"
        """
        from uuid import UUID
        
        # Get the cycles to consolidate
        cycles = []
        for cycle_id in job.cycle_ids_to_process:
            try:
                cycle = await self.memory_service.get_cycle_by_id(UUID(cycle_id))
                if cycle:
                    cycles.append(cycle)
            except Exception as e:
                logger.warning(f"Could not retrieve cycle {cycle_id}: {e}")
        
        if not cycles:
            logger.warning(f"No cycles to consolidate for job {job.job_id}")
            return
        
        # Create episodic memories from high-significance cycles
        for cycle in cycles:
            consolidation_priority = cycle.metadata.get("consolidation_metadata", {}).get("consolidation_priority", 0.5)
            
            if consolidation_priority > 0.7:  # High significance
                # Generate rich narrative using LLM
                narrative = await self._generate_episode_narrative(cycle)
                
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
                    key_insights=key_insights
                )
                
                job.episodes_created += 1
        
        # Extract semantic concepts from the episodes
        # Group cycles by topic and extract learned facts
        semantic_concepts = await self._extract_semantic_concepts_from_cycles(cycles, job.user_id)
        job.semantic_concepts_extracted = len(semantic_concepts)
    
    async def _generate_episode_narrative(self, cycle: CognitiveCycle) -> str:
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
            
            narrative = await self.llm_service.generate_text(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            return narrative.strip()
            
        except Exception as e:
            logger.warning(f"Could not generate narrative, using default: {e}")
            return f"User asked: {cycle.user_input[:100]}..."
    
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
        user_id: str
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
                    topic, topic_cycles, user_id
                )
                if concept:
                    concepts.append(concept)
        
        return concepts
    
    async def _analyze_topic_for_semantic_knowledge(
        self,
        topic: str,
        cycles: List[CognitiveCycle],
        user_id: str
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
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            # Parse JSON response
            import json
            data = json.loads(response.strip())
            
            concept = await self.autobiographical_system.extract_semantic_memory(
                user_id=user_id,
                concept_name=data["concept_name"],
                description=data["description"],
                category=data["category"],
                source_episodes=[str(c.cycle_id) for c in cycles],
                confidence=0.7
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
        
        # In a full implementation, this would:
        # 1. Retrieve the memories
        # 2. Update their "replay_count" metadata
        # 3. Potentially re-embed them with higher importance weights
        
        # For now, just log and update job
        job.patterns_discovered.append(f"Replayed {len(job.cycle_ids_to_process)} memories")
    
    async def _extract_patterns(self, job: MemoryConsolidationJob):
        """
        Extract patterns across multiple memories.
        Example: User often asks questions in the evening, user prefers detailed explanations, etc.
        """
        from uuid import UUID
        
        cycles = []
        for cycle_id in job.cycle_ids_to_process[:10]:  # Analyze up to 10 cycles
            try:
                cycle = await self.memory_service.get_cycle_by_id(UUID(cycle_id))
                if cycle:
                    cycles.append(cycle)
            except Exception as e:
                logger.warning(f"Could not retrieve cycle {cycle_id}: {e}")
        
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
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                max_tokens=200,
                temperature=0.5
            )
            
            import json
            patterns = json.loads(response.strip())
            
            if isinstance(patterns, list):
                job.patterns_discovered.extend(patterns)
                logger.info(f"Discovered {len(patterns)} patterns for user {job.user_id}")
            
        except Exception as e:
            logger.warning(f"Could not extract patterns: {e}")
    
    async def run_background_consolidation_loop(self):
        """
        Background loop that periodically checks for consolidation opportunities.
        This would run as a background task in main.py.
        """
        logger.info("Starting background consolidation loop")
        
        while True:
            try:
                # Check all active users (in a real system, track active users)
                # For now, just wait for explicit triggers
                await asyncio.sleep(60 * self.consolidation_interval_minutes)
                
                logger.debug("Background consolidation check (no active users to process)")
                
            except asyncio.CancelledError:
                logger.info("Background consolidation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying
