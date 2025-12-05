import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime # CQ-003 Fix: Import datetime

from src.core.exceptions import AgentServiceException, APIException
from src.models.core_models import UserRequest, AgentOutput, CognitiveCycle, ResponseMetadata, OutcomeSignals, ErrorAnalysis
from src.models.agent_models import AttentionDirective
from src.agents.perception_agent import PerceptionAgent
from src.agents.emotional_agent import EmotionalAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.creative_agent import CreativeAgent
from src.agents.critic_agent import CriticAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.services.web_browsing_service import WebBrowsingService
from src.services.audio_input_processor import AudioInputProcessor
from src.models.multimodal_models import AudioAnalysis
from src.services.cognitive_brain import CognitiveBrain
from src.services.memory_service import MemoryService
from src.services.background_task_queue import BackgroundTaskQueue
from src.services.self_reflection_discovery_engine import SelfReflectionAndDiscoveryEngine
from src.services.working_memory_buffer import WorkingMemoryBuffer
from src.services.thalamus_gateway import ThalamusGateway
from src.services.attention_controller import AttentionController
from src.services.conflict_monitor import ConflictMonitor
from src.services.reinforcement_learning_service import (
    ReinforcementLearningService,
    ContextTypes,
    StrategyTypes,
)
from src.services.contextual_memory_encoder import ContextualMemoryEncoder
from src.services.emotional_memory_service import EmotionalMemoryService
from src.services.meta_cognitive_monitor import MetaCognitiveMonitor, ActionRecommendation, GapType
from src.services.procedural_learning_service import ProceduralLearningService, SkillCategory
from src.services.metrics_service import MetricsService, MetricType

logger = logging.getLogger(__name__)

class OrchestrationService:
    """
    The core orchestrator of the cognitive cycle (Central Agent).
    It receives user input, dispatches tasks to specialized AI agents in parallel,
    collects and synthesizes their structured outputs, and prepares the data for the Cognitive Brain.
    It then passes the synthesized data to the Cognitive Brain for final response generation
    and stores the complete cycle in the Memory Service. It also triggers background reflection and discovery tasks.
    """
    def __init__(
        self, 
        perception_agent: PerceptionAgent,
        emotional_agent: EmotionalAgent,
        memory_agent: MemoryAgent,
        planning_agent: PlanningAgent,
        creative_agent: CreativeAgent,
        critic_agent: CriticAgent,
        discovery_agent: DiscoveryAgent,
        web_browsing_service: WebBrowsingService,
        cognitive_brain: CognitiveBrain,
        memory_service: MemoryService,
        background_task_queue: BackgroundTaskQueue,
        self_reflection_discovery_engine: SelfReflectionAndDiscoveryEngine,
        working_memory_buffer: Optional[WorkingMemoryBuffer] = None,
        thalamus_gateway: Optional[ThalamusGateway] = None,
        attention_controller: Optional[AttentionController] = None,
        conflict_monitor: Optional[ConflictMonitor] = None,
        contextual_memory_encoder: Optional[ContextualMemoryEncoder] = None,
        audio_input_processor: Optional[AudioInputProcessor] = None,
        rl_service: Optional[ReinforcementLearningService] = None,
        emotional_memory_service: Optional[EmotionalMemoryService] = None,
        meta_cognitive_monitor: Optional[MetaCognitiveMonitor] = None,
        procedural_learning_service: Optional[ProceduralLearningService] = None,
        metrics_service: Optional[MetricsService] = None
    ):
        self.perception_agent = perception_agent
        self.emotional_agent = emotional_agent
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent
        self.creative_agent = creative_agent
        self.critic_agent = critic_agent
        self.discovery_agent = discovery_agent
        self.web_browsing_service = web_browsing_service
        self.audio_input_processor = audio_input_processor
        self.cognitive_brain = cognitive_brain
        self.memory_service = memory_service
        self.background_task_queue = background_task_queue
        self.self_reflection_discovery_engine = self_reflection_discovery_engine
        self.working_memory_buffer = working_memory_buffer or WorkingMemoryBuffer()
        self.thalamus_gateway = thalamus_gateway or ThalamusGateway()
        self.attention_controller = attention_controller
        self.conflict_monitor = conflict_monitor or ConflictMonitor()
        self.contextual_memory_encoder = contextual_memory_encoder or ContextualMemoryEncoder()
        self.rl_service = rl_service  # May be None until RL milestone integrated
        self.emotional_memory_service = emotional_memory_service
        self.meta_cognitive_monitor = meta_cognitive_monitor
        self.procedural_learning_service = procedural_learning_service
        self.metrics_service = metrics_service
        self.session_start = datetime.utcnow()  # Track session start for contextual encoding
        logger.info("OrchestrationService (Central Agent) initialized with all specialized agents, Cognitive Brain, Memory Service, Background Task Queue, Self-Reflection & Discovery Engine, Working Memory Buffer, Thalamus Gateway, Conflict Monitor, Contextual Memory Encoder, Emotional Memory Service, Meta-Cognitive Monitor, optional Attention Controller, and optional Reinforcement Learning Service.")


    async def orchestrate_cycle(self, user_request: UserRequest) -> CognitiveCycle:
        """
        Initiates a full cognitive cycle by dispatching tasks to all specialized agents in parallel,
        collecting their outputs, synthesizing them, generating a final response via Cognitive Brain,
        and storing the complete cycle in Memory Service.

        Args:
            user_request (UserRequest): The incoming user request.

        Returns:
            CognitiveCycle: A complete cognitive cycle object containing all agent outputs and the final response.

        Raises:
            APIException: If a critical error occurs during orchestration.
        """
        logger.info(f"Orchestrating cognitive cycle for user {user_request.user_id}, session {user_request.session_id}")

        # If audio present, attempt transcription first so everyone can use the text
        effective_input_text = user_request.input_text or ""
        audio_analysis_dict: Optional[Dict[str, Any]] = None
        if user_request.audio_base64 and self.audio_input_processor:
            try:
                audio_analysis: AudioAnalysis = await self.audio_input_processor.process_audio(
                    audio_base64=user_request.audio_base64
                )
                audio_analysis_dict = audio_analysis.model_dump()
                if audio_analysis.transcription:
                    if effective_input_text:
                        effective_input_text = f"{effective_input_text}\n\n[Audio transcript]: {audio_analysis.transcription}"
                    else:
                        effective_input_text = audio_analysis.transcription
                logger.info(f"Audio transcription attached to cycle input (first 60 chars): {effective_input_text[:60]}...")
            except Exception as e:
                logger.warning(f"Audio transcription failed or skipped: {e}")

        # If after audio processing there's still no text, provide a safe placeholder to satisfy downstream validators
        used_placeholder = False
        if not effective_input_text or not effective_input_text.strip():
            if user_request.audio_base64:
                effective_input_text = "[Audio received; transcription unavailable]"
                used_placeholder = True
            elif user_request.image_base64:
                effective_input_text = "[Image received; description unavailable]"
                used_placeholder = True

        cognitive_cycle = CognitiveCycle(
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            user_input=effective_input_text,
            agent_outputs=[]
        )

        # Record cognitive cycle start metric after input processing
        if self.metrics_service:
            await self.metrics_service.record_metric(
                MetricType.COGNITIVE_CYCLE,
                {
                    "event": "cycle_started",
                    "user_id": str(user_request.user_id),
                    "session_id": str(user_request.session_id),
                    "has_image": user_request.image_base64 is not None,
                    "has_audio": user_request.audio_base64 is not None,
                    "input_length": len(effective_input_text) if effective_input_text else 0
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )

        # --- Pre-Interaction: Capture emotional profile state for RL reward computation ---
        pre_interaction_profile = None
        if self.emotional_memory_service:
            try:
                pre_interaction_profile = await self.emotional_memory_service.get_or_create_profile(user_request.user_id)
                cognitive_cycle.metadata["pre_interaction_trust"] = pre_interaction_profile.trust_level
                cognitive_cycle.metadata["pre_interaction_sentiment"] = pre_interaction_profile.last_emotion_detected
                logger.debug(f"Pre-interaction emotional state captured: trust={pre_interaction_profile.trust_level:.2f}, sentiment={pre_interaction_profile.last_emotion_detected}")
            except Exception as e:
                logger.warning(f"Failed to capture pre-interaction emotional profile: {e}")

        # --- Pre-Processing: Thalamus Gateway (Selective Attention) ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Thalamus Gateway - Analyzing input for selective agent activation")
        input_routing = await self.thalamus_gateway.route_input(
            user_input=effective_input_text,
            user_id=str(user_request.user_id),
            has_image=user_request.image_base64 is not None,
            has_audio=user_request.audio_base64 is not None
        )
        logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Input routing - {input_routing.quick_analysis}")
        logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Agent activation map - {input_routing.agent_activation}")

        attention_directives = []
        if self.attention_controller:
            working_memory_snapshot = None
            if getattr(self.working_memory_buffer, "context", None):
                working_memory_snapshot = self.working_memory_buffer.context.model_dump(mode="json")

            attention_directive = await self.attention_controller.generate_directive(
                quick_analysis=input_routing.quick_analysis,
                agent_activation=input_routing.agent_activation,
                working_memory_snapshot=working_memory_snapshot,
                 user_id=str(user_request.user_id),
                 stage="pre_stage1",
                 update_last_context=False,
            )
            attention_directive_dump = attention_directive.model_dump(mode="json")
            attention_directives.append(attention_directive_dump)
            directive_applied = False

            motifs = self._derive_attention_motifs(working_memory_snapshot, attention_directive)
            if motifs:
                input_routing.attention_motifs = motifs

            if self.attention_controller.should_apply(attention_directive):
                adjustments = self.attention_controller.apply_directive(
                    attention_directive,
                    input_routing.agent_activation,
                    input_routing.memory_config,
                )
                input_routing.agent_activation = adjustments["agent_activation"]
                input_routing.memory_config = adjustments["memory_config"]
                if hasattr(self.thalamus_gateway, "apply_attention_directive"):
                    self.thalamus_gateway.apply_attention_directive(
                        input_routing,
                        attention_directive,
                        attention_motifs=motifs if motifs else None,
                    )
                directive_applied = True
                logger.debug(
                    "Cycle %s: AttentionController applied adjustments %s",
                    cognitive_cycle.cycle_id,
                    adjustments,
                )
            else:
                logger.debug(
                    "Cycle %s: AttentionController in shadow mode (directive logged only)",
                    cognitive_cycle.cycle_id,
                )

            if self.metrics_service:
                await self.metrics_service.record_metric(
                    MetricType.ATTENTION_DIRECTIVE,
                    {
                        "shadow_mode": attention_directive.shadow_mode,
                        "applied": directive_applied,
                        "notes": attention_directive.notes,
                        "agent_bias": attention_directive.agent_bias,
                        "stage": attention_directive.stage,
                        "drift_score": attention_directive.drift_score,
                        "attention_motifs": motifs,
                    },
                    cycle_id=str(cognitive_cycle.cycle_id),
                    user_id=str(user_request.user_id),
                )
        
        # Store routing info in cycle metadata for analysis
        cognitive_cycle.metadata["thalamus_routing"] = {
            "quick_analysis": input_routing.quick_analysis.model_dump(mode='json'),
            "agent_activation": input_routing.agent_activation,
            "memory_config": input_routing.memory_config
        }
        if attention_directives:
            cognitive_cycle.metadata["attention_directives"] = attention_directives
        if used_placeholder:
            cognitive_cycle.metadata.setdefault("notes", []).append("effective_input_text was placeholder due to missing transcription/description")
        if audio_analysis_dict:
            cognitive_cycle.metadata["audio_analysis"] = audio_analysis_dict

        # --- Stage 1: Foundational Agents (with selective activation) ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Starting Stage 1 - Foundational Agents")
        stage1_agents_and_tasks = []
        
        # Perception is always active
        if input_routing.agent_activation.get("perception", True):
            stage1_agents_and_tasks.append((
                self.perception_agent,
                self.perception_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id,
                    image_base64=user_request.image_base64,
                    audio_base64=user_request.audio_base64
                )
            ))
        
        # Emotional agent (conditionally activated)
        if input_routing.agent_activation.get("emotional", True):
            stage1_agents_and_tasks.append((
                self.emotional_agent,
                self.emotional_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Emotional agent skipped (low priority)")
        
        # Memory agent (conditionally activated with configured depth)
        if input_routing.agent_activation.get("memory", True):
            stage1_agents_and_tasks.append((
                self.memory_agent,
                self.memory_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Memory agent skipped (minimal context need)")
        
        # Record agent activation metrics
        if self.metrics_service:
            activated_agents = [agent.AGENT_ID for agent, _ in stage1_agents_and_tasks]
            await self.metrics_service.record_metric(
                MetricType.AGENT_ACTIVATION,
                {
                    "stage": 1,
                    "agents_activated": activated_agents,
                    "total_agents": len(activated_agents),
                    "routing_decisions": input_routing.agent_activation
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )
        
        stage1_results = await asyncio.gather(*[task for _, task in stage1_agents_and_tasks], return_exceptions=True)

        # Process Stage 1 results
        stage1_outputs = []
        for (agent_instance, _), result in zip(stage1_agents_and_tasks, stage1_results):
            # This logic is duplicated for both stages. Consider refactoring to a helper method.
            agent_id = agent_instance.AGENT_ID
            start_time = time.perf_counter()
            if isinstance(result, AgentOutput):
                stage1_outputs.append(result)
                cognitive_cycle.agent_outputs.append(result)
                logger.debug(f"Orchestration: Stage 1 agent {agent_id} completed successfully.")
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000
                logger.info(f"AGENT_METRIC: {agent_id} | Cycle: {cognitive_cycle.cycle_id} | User: {user_request.user_id} | Status: success | Duration: {duration:.2f}ms | Confidence: {result.confidence}",
                            extra={"agent_id": agent_id, "cycle_id": str(cognitive_cycle.cycle_id), "user_id": str(user_request.user_id), "status": "success", "duration_ms": duration, "confidence": result.confidence})
            elif isinstance(result, Exception):
                logger.error(f"Orchestration: Stage 1 agent {agent_id} failed: {result}", exc_info=True)
                error_detail = str(result.detail) if isinstance(result, AgentServiceException) else str(result)
                cognitive_cycle.agent_outputs.append(AgentOutput(agent_id=agent_id, analysis={"error": error_detail, "status": "failed"}, confidence=0.0, priority=1, raw_output=f"Error: {error_detail}"))
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000
                logger.error(f"AGENT_METRIC: {agent_id} | Cycle: {cognitive_cycle.cycle_id} | User: {user_request.user_id} | Status: failed | Duration: {duration:.2f}ms | Error: {error_detail}",
                             extra={"agent_id": agent_id, "cycle_id": str(cognitive_cycle.cycle_id), "user_id": str(user_request.user_id), "status": "failed", "duration_ms": duration, "error": error_detail})

        # --- Working Memory Update: Extract insights from Stage 1 ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Updating Working Memory from Stage 1 outputs")
        self.working_memory_buffer.reset()
        self.working_memory_buffer.update_from_stage1(stage1_outputs, effective_input_text)
        
        # --- Stage 1.5: Conflict Detection ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Conflict Monitor - Checking Stage 1 for inconsistencies")
        stage1_conflict_report = await self.conflict_monitor.detect_conflicts(stage1_outputs)
        cognitive_cycle.metadata["stage1_conflicts"] = {
            "conflicts": [c.model_dump(mode='json') for c in stage1_conflict_report.conflicts],
            "requires_adjustment": stage1_conflict_report.requires_adjustment,
            "coherence_score": stage1_conflict_report.coherence_score
        }
        
        # Record conflict resolution metrics
        if self.metrics_service:
            await self.metrics_service.record_metric(
                MetricType.CONFLICT_RESOLUTION,
                {
                    "stage": 1,
                    "conflict_count": len(stage1_conflict_report.conflicts),
                    "coherence_score": stage1_conflict_report.coherence_score,
                    "requires_adjustment": stage1_conflict_report.requires_adjustment,
                    "conflict_types": [c.conflict_type for c in stage1_conflict_report.conflicts],
                    "severity_levels": [c.severity for c in stage1_conflict_report.conflicts]
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )
        
        if stage1_conflict_report.conflicts:
            logger.warning(f"Cycle {cognitive_cycle.cycle_id}: Detected {len(stage1_conflict_report.conflicts)} Stage 1 conflicts - Coherence: {stage1_conflict_report.coherence_score:.2f}")
            for conflict in stage1_conflict_report.conflicts:
                logger.debug(f"  - {conflict.conflict_type} ({conflict.severity}): {conflict.resolution_strategy}")

        # Post-Stage1 attention update with drift detection
        if self.attention_controller:
            working_memory_snapshot = self.working_memory_buffer.context.model_dump(mode="json")
            post_stage_directive = await self.attention_controller.generate_directive(
                quick_analysis=input_routing.quick_analysis,
                agent_activation=input_routing.agent_activation,
                working_memory_snapshot=working_memory_snapshot,
                conflict_report=stage1_conflict_report,
                user_id=str(user_request.user_id),
                stage="post_stage1",
                update_last_context=True,
            )
            directive_dump = post_stage_directive.model_dump(mode="json")
            attention_directives.append(directive_dump)
            motifs = self._derive_attention_motifs(working_memory_snapshot, post_stage_directive)
            if motifs:
                input_routing.attention_motifs = motifs
                self.working_memory_buffer.set_attention_motifs(motifs)

            directive_applied = False
            if self.attention_controller.should_apply(post_stage_directive):
                adjustments = self.attention_controller.apply_directive(
                    post_stage_directive,
                    input_routing.agent_activation,
                    input_routing.memory_config,
                )
                input_routing.agent_activation = adjustments["agent_activation"]
                input_routing.memory_config = adjustments["memory_config"]
                if hasattr(self.thalamus_gateway, "apply_attention_directive"):
                    self.thalamus_gateway.apply_attention_directive(
                        input_routing,
                        post_stage_directive,
                        attention_motifs=motifs if motifs else None,
                    )
                directive_applied = True
                logger.debug(
                    "Cycle %s: Post-stage AttentionController adjustments %s",
                    cognitive_cycle.cycle_id,
                    adjustments,
                )
            else:
                logger.debug(
                    "Cycle %s: Post-stage AttentionController in shadow mode",
                    cognitive_cycle.cycle_id,
                )

            if self.metrics_service:
                await self.metrics_service.record_metric(
                    MetricType.ATTENTION_DIRECTIVE,
                    {
                        "shadow_mode": post_stage_directive.shadow_mode,
                        "applied": directive_applied,
                        "notes": post_stage_directive.notes,
                        "agent_bias": post_stage_directive.agent_bias,
                        "stage": post_stage_directive.stage,
                        "drift_score": post_stage_directive.drift_score,
                        "attention_motifs": motifs,
                    },
                    cycle_id=str(cognitive_cycle.cycle_id),
                    user_id=str(user_request.user_id),
                )
        
        # --- Stage 2: Analytical and Creative Agents (with selective activation) ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Starting Stage 2 - Analytical and Creative Agents")
        stage2_agents_and_tasks = []
        
        # Planning agent (conditionally activated)
        if input_routing.agent_activation.get("planning", True):
            stage2_agents_and_tasks.append((
                self.planning_agent,
                self.planning_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id,
                    other_agent_outputs=stage1_outputs
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Planning agent skipped (simple query)")
        
        # Creative agent (conditionally activated)
        if input_routing.agent_activation.get("creative", False):
            stage2_agents_and_tasks.append((
                self.creative_agent,
                self.creative_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id,
                    other_agent_outputs=stage1_outputs
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Creative agent skipped (not needed)")
        
        # Critic agent (conditionally activated)
        if input_routing.agent_activation.get("critic", True):
            stage2_agents_and_tasks.append((
                self.critic_agent,
                self.critic_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id,
                    other_agent_outputs=stage1_outputs
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Critic agent skipped (low complexity)")
        
        # Discovery agent (conditionally activated)
        if input_routing.agent_activation.get("discovery", True):
            stage2_agents_and_tasks.append((
                self.discovery_agent,
                self.discovery_agent.process_input(
                    user_input=effective_input_text,
                    user_id=user_request.user_id,
                    other_agent_outputs=stage1_outputs
                )
            ))
        else:
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Discovery agent skipped (no deep exploration needed)")
        
        # Record Stage 2 agent activation metrics
        if self.metrics_service and stage2_agents_and_tasks:
            activated_agents = [agent.AGENT_ID for agent, _ in stage2_agents_and_tasks]
            await self.metrics_service.record_metric(
                MetricType.AGENT_ACTIVATION,
                {
                    "stage": 2,
                    "agents_activated": activated_agents,
                    "total_agents": len(activated_agents),
                    "routing_decisions": input_routing.agent_activation
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )
        
        stage2_results = await asyncio.gather(*[task for _, task in stage2_agents_and_tasks], return_exceptions=True)

        # Process Stage 2 results
        for (agent_instance, _), result in zip(stage2_agents_and_tasks, stage2_results):
            agent_id = agent_instance.AGENT_ID
            start_time = time.perf_counter()
            if isinstance(result, AgentOutput):
                cognitive_cycle.agent_outputs.append(result)
                logger.debug(f"Orchestration: Stage 2 agent {agent_id} completed successfully.")
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000
                logger.info(f"AGENT_METRIC: {agent_id} | Cycle: {cognitive_cycle.cycle_id} | User: {user_request.user_id} | Status: success | Duration: {duration:.2f}ms | Confidence: {result.confidence}",
                            extra={"agent_id": agent_id, "cycle_id": str(cognitive_cycle.cycle_id), "user_id": str(user_request.user_id), "status": "success", "duration_ms": duration, "confidence": result.confidence})
            elif isinstance(result, Exception):
                logger.error(f"Orchestration: Stage 2 agent {agent_id} failed: {result}", exc_info=True)
                error_detail = str(result.detail) if isinstance(result, AgentServiceException) else str(result)
                cognitive_cycle.agent_outputs.append(AgentOutput(agent_id=agent_id, analysis={"error": error_detail, "status": "failed"}, confidence=0.0, priority=1, raw_output=f"Error: {error_detail}"))
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000
                logger.error(f"AGENT_METRIC: {agent_id} | Cycle: {cognitive_cycle.cycle_id} | User: {user_request.user_id} | Status: failed | Duration: {duration:.2f}ms | Error: {error_detail}",
                             extra={"agent_id": agent_id, "cycle_id": str(cognitive_cycle.cycle_id), "user_id": str(user_request.user_id), "status": "failed", "duration_ms": duration, "error": error_detail})

        # --- Stage 2.5: Final Conflict Check ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Conflict Monitor - Final coherence check")
        all_outputs = cognitive_cycle.agent_outputs
        final_conflict_report = await self.conflict_monitor.detect_conflicts(all_outputs)
        cognitive_cycle.metadata["final_conflicts"] = {
            "conflicts": [c.model_dump(mode='json') for c in final_conflict_report.conflicts],
            "requires_adjustment": final_conflict_report.requires_adjustment,
            "coherence_score": final_conflict_report.coherence_score
        }
        
        if final_conflict_report.conflicts:
            logger.warning(f"Cycle {cognitive_cycle.cycle_id}: Detected {len(final_conflict_report.conflicts)} total conflicts - Coherence: {final_conflict_report.coherence_score:.2f}")
            for conflict in final_conflict_report.conflicts:
                logger.debug(f"  - {conflict.conflict_type} ({conflict.severity}): {conflict.resolution_strategy}")
        else:
            logger.info(f"Cycle {cognitive_cycle.cycle_id}: No conflicts detected - Coherence: {final_conflict_report.coherence_score:.2f}")
        
        # Record final conflict resolution metrics with coherence improvement
        if self.metrics_service:
            # Calculate coherence improvement (final - initial)
            initial_coherence = stage1_conflict_report.coherence_score
            final_coherence = final_conflict_report.coherence_score
            coherence_improvement = final_coherence - initial_coherence
            
            await self.metrics_service.record_metric(
                MetricType.CONFLICT_RESOLUTION,
                {
                    "stage": "final",
                    "conflict_count": len(final_conflict_report.conflicts),
                    "coherence_score": final_conflict_report.coherence_score,
                    "coherence_improvement": coherence_improvement,
                    "requires_adjustment": final_conflict_report.requires_adjustment,
                    "conflict_types": [c.conflict_type for c in final_conflict_report.conflicts],
                    "severity_levels": [c.severity for c in final_conflict_report.conflicts]
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )
        
        # --- RL Strategy Selection (Basal Ganglia) ---
        if self.rl_service and final_conflict_report.conflicts:
            rl_guidance = []

            # Map conflict types to RL contexts + available strategies
            conflict_context_map = {
                "sentiment_coherence_mismatch": (ContextTypes.EMOTIONAL_VS_TECHNICAL, [
                    StrategyTypes.PRIORITIZE_EMOTIONAL,
                    StrategyTypes.PRIORITIZE_TECHNICAL,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.WEIGHTED_SYNTHESIS
                ]),
                "memory_planning_disconnect": (ContextTypes.DETAILED_VS_OVERVIEW, [
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ]),
                "creative_critic_divergence": (ContextTypes.CREATIVE_VS_FACTUAL, [
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ]),
                "emotional_logic_conflict": (ContextTypes.EMPATHY_VS_ACCURACY, [
                    StrategyTypes.PRIORITIZE_EMOTIONAL,
                    StrategyTypes.PRIORITIZE_TECHNICAL,
                    StrategyTypes.EMPATHY_THEN_FACTS,
                    StrategyTypes.FACTS_THEN_EMPATHY,
                    StrategyTypes.BLEND_BOTH
                ]),
                "perception_memory_conflict": (ContextTypes.DETAILED_VS_OVERVIEW, [
                    StrategyTypes.WEIGHTED_SYNTHESIS,
                    StrategyTypes.BLEND_BOTH,
                    StrategyTypes.PRIORITIZE_TECHNICAL
                ])
            }

            for conflict in final_conflict_report.conflicts:
                ctx_tuple = conflict_context_map.get(conflict.conflict_type)
                if not ctx_tuple:
                    continue  # Skip conflicts without RL mapping yet
                rl_context, strategies = ctx_tuple
                try:
                    selected = await self.rl_service.select_strategy(
                        context=rl_context,
                        available_strategies=strategies,
                        user_id=user_request.user_id
                    )
                    rl_guidance.append({
                        "conflict_type": conflict.conflict_type,
                        "rl_context": rl_context,
                        "available_strategies": strategies,
                        "selected_strategy": selected,
                        "resolution_hint": conflict.resolution_strategy
                    })
                except Exception as e:
                    logger.warning(f"RL strategy selection failed for conflict {conflict.conflict_type}: {e}")

            if rl_guidance:
                cognitive_cycle.metadata["rl_strategy_guidance"] = rl_guidance
                logger.info(f"RL selected strategies for {len(rl_guidance)} conflicts in cycle {cognitive_cycle.cycle_id}.")

        logger.info(f"Cognitive cycle orchestrated for user {user_request.user_id}. Collected {len(cognitive_cycle.agent_outputs)} agent outputs.")

        # --- Step 1.75: Meta-Cognitive Assessment (Pre-Response Gate) ---
        meta_cognitive_override = None
        if self.meta_cognitive_monitor:
            try:
                recommendation, gap_type, confidence_score, explanation = await self.meta_cognitive_monitor.assess_answer_appropriateness(
                    query=effective_input_text,
                    agent_outputs=[ao.model_dump() for ao in cognitive_cycle.agent_outputs],
                    user_id=str(user_request.user_id)
                )

                cognitive_cycle.metadata["meta_cognitive_assessment"] = {
                    "recommendation": recommendation.value,
                    "gap_type": gap_type.value,
                    "confidence_score": confidence_score,
                    "explanation": explanation
                }

                # Handle non-answer recommendations
                if recommendation != ActionRecommendation.ANSWER:
                    if recommendation == ActionRecommendation.SEARCH_FIRST:
                        # Trigger web search via DiscoveryAgent
                        logger.info(f"Meta-cognitive: Triggering web search for knowledge gap - {explanation}")
                        # This will be handled by setting a flag for later processing

                    elif recommendation in [ActionRecommendation.ASK_CLARIFICATION, ActionRecommendation.DECLINE_POLITELY]:
                        # Generate uncertainty response
                        uncertainty_response = await self.meta_cognitive_monitor.generate_uncertainty_response(
                            query=effective_input_text,
                            gap_type=gap_type,
                            recommendation=recommendation
                        )
                        meta_cognitive_override = uncertainty_response
                        logger.info(f"Meta-cognitive override: {recommendation.value} - {explanation}")

                    elif recommendation == ActionRecommendation.ACKNOWLEDGE_UNCERTAINTY:
                        # Add uncertainty note but still generate response
                        cognitive_cycle.metadata["acknowledge_uncertainty"] = True
                        logger.info(f"Meta-cognitive: Will acknowledge uncertainty in response - {explanation}")

            except Exception as e:
                logger.warning(f"Meta-cognitive assessment failed: {e}")

        # --- Step 2: Generate final response using Cognitive Brain ---
        try:
            # Use meta-cognitive override if set
            if meta_cognitive_override:
                final_response_text = meta_cognitive_override
                response_metadata = ResponseMetadata(
                    response_type="meta_cognitive",
                    tone="honest",
                    strategies=["uncertainty_acknowledgment"],
                    cognitive_moves=["assess_knowledge_boundaries"]
                )
                outcome_signals = OutcomeSignals(
                    user_satisfaction_potential=0.7,  # Honest uncertainty often appreciated
                    engagement_potential=0.6
                )
            else:
                final_response_text, response_metadata, outcome_signals = await self.cognitive_brain.generate_response(cognitive_cycle)

            cognitive_cycle.final_response = final_response_text
            cognitive_cycle.response_metadata = response_metadata
            cognitive_cycle.outcome_signals = outcome_signals
            logger.info(f"Cognitive Brain generated final response for cycle {cognitive_cycle.cycle_id}.")
        except APIException as e:
            logger.error(f"Orchestration: Cognitive Brain failed to generate response for cycle {cognitive_cycle.cycle_id}: {e.detail}", exc_info=True)
            cognitive_cycle.final_response = "An error occurred while generating the response."
            cognitive_cycle.response_metadata = ResponseMetadata(response_type="error", tone="neutral", strategies=["error_handling"], cognitive_moves=["inform_user"])
            cognitive_cycle.outcome_signals = OutcomeSignals(user_satisfaction_potential=0.1, engagement_potential=0.1)
        except Exception as e:
            logger.critical(f"Orchestration: Unexpected error from Cognitive Brain for cycle {cognitive_cycle.cycle_id}: {e}", exc_info=True)
            cognitive_cycle.final_response = "An unexpected error prevented response generation."
            cognitive_cycle.response_metadata = ResponseMetadata(response_type="error", tone="neutral", strategies=["error_handling"], cognitive_moves=["inform_user"])
            cognitive_cycle.outcome_signals = OutcomeSignals(user_satisfaction_potential=0.0, engagement_potential=0.0)
        
        # --- Step 2.5: Update SelfModel (autobiographical memory) ---
        if hasattr(self, 'self_model_service') and self.cognitive_brain.self_model_service:
            try:
                await self.cognitive_brain.self_model_service.update_from_cycle(cognitive_cycle)
                logger.debug(f"Updated self-model for user {user_request.user_id}")
            except Exception as e:
                logger.warning(f"Failed to update self-model: {e}")
        
        # --- Step 2.6: Procedural Learning - Track Skill Performance ---
        if self.procedural_learning_service:
            try:
                # Track performance for key skills used in this cycle
                skill_categories = []
                
                # Determine skill categories based on agents activated and response characteristics
                if cognitive_cycle.agent_outputs:
                    agent_names = [ao.agent_id for ao in cognitive_cycle.agent_outputs]
                    if "perception" in agent_names:
                        skill_categories.append("perception")
                    if "emotional" in agent_names:
                        skill_categories.append("emotional_intelligence")
                    if "planning" in agent_names:
                        skill_categories.append("planning")
                    if "creative" in agent_names:
                        skill_categories.append("creativity")
                    if "critic" in agent_names:
                        skill_categories.append("critical_thinking")
                    if "discovery" in agent_names:
                        skill_categories.append("research")
                
                # Add response quality skill
                skill_categories.append("response_generation")
                
                # Track performance for each skill category
                for skill_category_str in skill_categories:
                    # Map string skill categories to SkillCategory enum
                    skill_category_mapping = {
                        "perception": SkillCategory.TECHNICAL_EXPLANATION,
                        "emotional_intelligence": SkillCategory.EMOTIONAL_SUPPORT,
                        "planning": SkillCategory.PROBLEM_SOLVING,
                        "creativity": SkillCategory.CREATIVE_BRAINSTORMING,
                        "critical_thinking": SkillCategory.PROBLEM_SOLVING,
                        "research": SkillCategory.TECHNICAL_EXPLANATION,
                        "response_generation": SkillCategory.GENERAL_CONVERSATION
                    }
                    skill_category = skill_category_mapping.get(skill_category_str, SkillCategory.GENERAL_CONVERSATION)
                    
                    performance_score = cognitive_cycle.outcome_signals.user_satisfaction_potential
                    
                    # Adjust score based on meta-cognitive assessment
                    if cognitive_cycle.metadata.get("meta_cognitive_assessment"):
                        assessment = cognitive_cycle.metadata["meta_cognitive_assessment"]
                        if assessment.get("recommendation") == "acknowledge_uncertainty":
                            # Uncertainty acknowledgment is a positive skill
                            performance_score = min(1.0, performance_score + 0.1)
                        elif assessment.get("recommendation") in ["ask_clarification", "decline_politely"]:
                            # These are appropriate responses to knowledge gaps
                            performance_score = min(1.0, performance_score + 0.05)
                    
                    # Calculate average confidence from agent outputs
                    avg_confidence = 0.0
                    if cognitive_cycle.agent_outputs:
                        confidences = [ao.confidence for ao in cognitive_cycle.agent_outputs if ao.confidence > 0.0]
                        if confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                    
                    await self.procedural_learning_service.track_skill_performance(
                        skill_category=skill_category,
                        outcome_score=performance_score,
                        confidence_score=avg_confidence,
                        agent_sequence=[ao.agent_id for ao in cognitive_cycle.agent_outputs],
                        cycle_metadata={
                            "cycle_id": str(cognitive_cycle.cycle_id),
                            "user_input_length": len(cognitive_cycle.user_input),
                            "response_length": len(cognitive_cycle.final_response),
                            "agent_count": len(cognitive_cycle.agent_outputs),
                            "has_conflicts": bool(cognitive_cycle.metadata.get("final_conflicts")),
                            "meta_cognitive_used": bool(cognitive_cycle.metadata.get("meta_cognitive_assessment"))
                        },
                        user_id=str(user_request.user_id)
                    )
                
                # Generate structured error analysis for learning
                error_analyses = []
                
                # Generate error analysis from ConflictMonitor for low coherence cycles
                if final_conflict_report.coherence_score < 0.5:
                    conflict_error_analysis = await self.conflict_monitor.generate_error_analysis(
                        cycle_id=cognitive_cycle.cycle_id,
                        agent_outputs=cognitive_cycle.agent_outputs,
                        coherence_score=final_conflict_report.coherence_score,
                        user_input_summary=cognitive_cycle.user_input[:200],
                        response_summary=cognitive_cycle.final_response[:200] if cognitive_cycle.final_response else "",
                        cycle_metadata=cognitive_cycle.metadata
                    )
                    if conflict_error_analysis:
                        error_analyses.append(conflict_error_analysis)
                
                # Generate error analysis from MetaCognitiveMonitor for significant failures
                if cognitive_cycle.metadata.get("meta_cognitive_assessment"):
                    assessment = cognitive_cycle.metadata["meta_cognitive_assessment"]
                    recommendation = ActionRecommendation(assessment["recommendation"])
                    gap_type = GapType(assessment["gap_type"])
                    confidence_score = assessment["confidence_score"]
                    
                    meta_cognitive_error_analysis = await self.meta_cognitive_monitor.generate_error_analysis(
                        cycle_id=cognitive_cycle.cycle_id,
                        recommendation=recommendation,
                        gap_type=gap_type,
                        confidence_score=confidence_score,
                        query=effective_input_text,
                        agents_activated=[ao.agent_id for ao in cognitive_cycle.agent_outputs],
                        user_input_summary=cognitive_cycle.user_input[:200],
                        response_summary=cognitive_cycle.final_response[:200] if cognitive_cycle.final_response else "",
                        cycle_metadata=cognitive_cycle.metadata
                    )
                    if meta_cognitive_error_analysis:
                        error_analyses.append(meta_cognitive_error_analysis)
                
                # Learn from structured error analyses
                if self.procedural_learning_service:
                    for error_analysis in error_analyses:
                        await self.procedural_learning_service.learn_from_error(
                            error_analysis=error_analysis,
                            user_id=str(user_request.user_id)
                        )
                
                # Legacy error learning for basic response errors (backward compatibility)
                if self.procedural_learning_service and cognitive_cycle.response_metadata.response_type == "error" and not error_analyses:
                    await self.procedural_learning_service.learn_from_error(
                        error_type="response_generation_failure",
                        error_context={
                            "cycle_id": str(cognitive_cycle.cycle_id),
                            "user_input": cognitive_cycle.user_input[:200],  # Truncate for context
                            "error_response": cognitive_cycle.final_response,
                            "agent_outputs_count": len(cognitive_cycle.agent_outputs)
                        },
                        user_id=str(user_request.user_id)
                    )
                
                logger.debug(f"Procedural learning: Tracked performance for {len(skill_categories)} skills in cycle {cognitive_cycle.cycle_id}")
                
            except Exception as e:
                logger.warning(f"Procedural learning tracking failed: {e}")
        
        # --- Step 2.75: Contextual Memory Encoding ---
        logger.info(f"Cycle {cognitive_cycle.cycle_id}: Contextual Memory Encoder - Enriching with contextual bindings")
        try:
            cognitive_cycle = await self.contextual_memory_encoder.encode_cycle(cognitive_cycle, self.session_start)
            consolidation_priority = cognitive_cycle.metadata.get("consolidation_metadata", {}).get("consolidation_priority", 0.5)
            logger.debug(f"Cycle {cognitive_cycle.cycle_id}: Consolidation priority = {consolidation_priority:.2f}")
        except Exception as e:
            logger.warning(f"Failed to encode contextual bindings: {e}")
        
        # --- Step 3: Store complete cognitive cycle in Memory Service ---
        try:
            await self.memory_service.upsert_cycle(cognitive_cycle)
            logger.info(f"Cognitive cycle {cognitive_cycle.cycle_id} successfully stored in Memory Service.")
        except APIException as e:
            logger.error(f"Orchestration: Failed to store cognitive cycle {cognitive_cycle.cycle_id} in Memory Service: {e.detail}", exc_info=True)
            # This is a non-critical failure for the user response, but important for system memory
        except Exception as e:
            logger.critical(f"Orchestration: Unexpected error storing cognitive cycle {cognitive_cycle.cycle_id} in Memory Service: {e}", exc_info=True)
        
        # --- Step 3.5: Theory of Mind Validation ---
        if self.cognitive_brain.theory_of_mind_service:
            try:
                # Get previous cycle for comparison
                previous_cycle = None
                stm = await self.memory_service.get_stm(str(user_request.user_id))
                if stm:
                    cycles = stm.get_recent_cycles()
                    if len(cycles) >= 2:
                        # Second-to-last cycle (before current)
                        previous_cycle = cycles[1]
                
                # Auto-validate previous predictions
                validation_results = await self.cognitive_brain.theory_of_mind_service.auto_validate_predictions(
                    user_id=str(user_request.user_id),
                    current_cycle=cognitive_cycle,
                    previous_cycle=previous_cycle
                )
                
                if validation_results:
                    cognitive_cycle.metadata["theory_of_mind_validation"] = validation_results
                    logger.info(f"Validated {len(validation_results)} theory of mind predictions for user {user_request.user_id}")
                
                # Get validation statistics
                stats = self.cognitive_brain.theory_of_mind_service.get_validation_stats(str(user_request.user_id))
                if stats["validated"] > 0:
                    logger.debug(f"Theory of Mind accuracy for user {user_request.user_id}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['validated']})")
                
            except Exception as e:
                logger.warning(f"Failed to validate theory of mind predictions: {e}")

        # --- RL Reward Update (Using composite reward signals from multiple sources) ---
        if self.rl_service and cognitive_cycle.metadata.get("rl_strategy_guidance") and cognitive_cycle.outcome_signals:
            # Compute composite reward from multiple emotional and behavioral signals
            composite_reward = await self._compute_composite_reward(
                user_request.user_id,
                cognitive_cycle,
                pre_interaction_profile
            )

            for guidance in cognitive_cycle.metadata.get("rl_strategy_guidance", []):
                try:
                    await self.rl_service.update_from_outcome(
                        context=guidance["rl_context"],
                        strategy_used=guidance["selected_strategy"],
                        reward=composite_reward,
                        user_id=user_request.user_id,
                        metadata={
                            "conflict_type": guidance["conflict_type"],
                            "coherence_score": cognitive_cycle.metadata.get("final_conflicts", {}).get("coherence_score"),
                            "resolution_hint": guidance.get("resolution_hint"),
                            "composite_reward_breakdown": cognitive_cycle.metadata.get("reward_breakdown", {})
                        }
                    )
                except Exception as e:
                    logger.warning(f"RL reward update failed for context {guidance.get('rl_context')}: {e}")

        # Record cognitive cycle completion metric
        if self.metrics_service:
            await self.metrics_service.record_metric(
                MetricType.COGNITIVE_CYCLE,
                {
                    "event": "cycle_completed",
                    "user_id": str(user_request.user_id),
                    "session_id": str(user_request.session_id),
                    "total_agents": len(cognitive_cycle.agent_outputs),
                    "response_length": len(cognitive_cycle.final_response) if cognitive_cycle.final_response else 0,
                    "user_satisfaction": cognitive_cycle.outcome_signals.user_satisfaction_potential if cognitive_cycle.outcome_signals else None,
                    "processing_time": (datetime.utcnow() - cognitive_cycle.timestamp).total_seconds(),
                    "conflict_count": len(cognitive_cycle.metadata.get("final_conflicts", {}).get("conflicts", []))
                },
                cycle_id=str(cognitive_cycle.cycle_id),
                user_id=str(user_request.user_id)
            )

        return cognitive_cycle

    def _derive_attention_motifs(
        self,
        working_memory_snapshot: Optional[Dict[str, Any]],
        directive: AttentionDirective
    ) -> List[str]:
        motifs: List[str] = []
        if working_memory_snapshot:
            motifs.extend(working_memory_snapshot.get("attention_focus", [])[:3])
            motifs.extend(working_memory_snapshot.get("topics", [])[:2])
            goals = working_memory_snapshot.get("inferred_goals", [])
            motifs.extend([f"goal:{goal}" for goal in goals[:2]])
        for reason in directive.drift_reasons:
            motifs.append(f"drift:{reason}")
        return list(dict.fromkeys([m for m in motifs if m]))[:6]

    async def _compute_composite_reward(
        self,
        user_id: UUID,
        cognitive_cycle: CognitiveCycle,
        pre_interaction_profile
    ) -> float:
        """
        Compute composite reward signal from multiple emotional and behavioral sources.

        Args:
            user_id: The user's UUID
            cognitive_cycle: The completed cognitive cycle
            pre_interaction_profile: Emotional profile before interaction

        Returns:
            Composite reward value (0.0 to 1.0)
        """
        reward_components = {
            "trust_delta": 0.0,
            "sentiment_shift": 0.0,
            "user_feedback": 0.0,
            "engagement_continuation": 0.0,
            "satisfaction_potential": cognitive_cycle.outcome_signals.user_satisfaction_potential
        }

        try:
            # Get post-interaction emotional profile
            if self.emotional_memory_service:
                post_interaction_profile = await self.emotional_memory_service.get_or_create_profile(user_id)

                # 1. Trust delta (0.3 weight): Improvement in trust level
                if pre_interaction_profile:
                    trust_delta = post_interaction_profile.trust_level - pre_interaction_profile.trust_level
                    reward_components["trust_delta"] = max(0.0, min(1.0, trust_delta + 0.5))  # Normalize to 0-1
                else:
                    reward_components["trust_delta"] = 0.5  # Neutral if no pre-profile

                # 2. Sentiment shift (0.2 weight): Positive sentiment change
                pre_sentiment = cognitive_cycle.metadata.get("pre_interaction_sentiment", "neutral")
                post_sentiment = post_interaction_profile.last_emotion_detected

                sentiment_values = {"positive": 1.0, "neutral": 0.5, "negative": 0.0, "mixed": 0.3}
                pre_value = sentiment_values.get(pre_sentiment, 0.5)
                post_value = sentiment_values.get(post_sentiment, 0.5)
                sentiment_shift = post_value - pre_value
                reward_components["sentiment_shift"] = max(0.0, min(1.0, sentiment_shift + 0.5))  # Normalize to 0-1

            # 3. User feedback (0.3 weight): Explicit positive/negative feedback in response
            user_input_lower = cognitive_cycle.user_input.lower()
            positive_feedback = any(word in user_input_lower for word in [
                "thank you", "thanks", "good job", "well done", "appreciate", "helpful",
                "great", "awesome", "excellent", "perfect", "love it"
            ])
            negative_feedback = any(word in user_input_lower for word in [
                "bad", "terrible", "awful", "horrible", "don't like", "not helpful",
                "confusing", "frustrating", "annoying", "stupid"
            ])

            if positive_feedback:
                reward_components["user_feedback"] = 1.0
            elif negative_feedback:
                reward_components["user_feedback"] = 0.0
            else:
                reward_components["user_feedback"] = 0.5  # Neutral

            # 4. Engagement continuation (0.2 weight): Based on conversation length and follow-up
            input_length = len(cognitive_cycle.user_input.strip())
            has_followup_questions = "?" in cognitive_cycle.user_input
            engagement_score = min(1.0, input_length / 200.0)  # Longer inputs = more engagement
            if has_followup_questions:
                engagement_score = min(1.0, engagement_score + 0.3)
            reward_components["engagement_continuation"] = engagement_score

            # Store breakdown in cycle metadata for analysis
            cognitive_cycle.metadata["reward_breakdown"] = reward_components

            # Weighted composite reward
            weights = {
                "trust_delta": 0.3,
                "sentiment_shift": 0.2,
                "user_feedback": 0.3,
                "engagement_continuation": 0.2
            }

            composite_reward = (
                reward_components["trust_delta"] * weights["trust_delta"] +
                reward_components["sentiment_shift"] * weights["sentiment_shift"] +
                reward_components["user_feedback"] * weights["user_feedback"] +
                reward_components["engagement_continuation"] * weights["engagement_continuation"] +
                reward_components["satisfaction_potential"] * 0.0  # Not weighted in composite, kept for reference
            )

            # Ensure reward is in valid range
            composite_reward = max(0.0, min(1.0, composite_reward))

            logger.debug(
                f"Composite reward for user {user_id}: {composite_reward:.3f} "
                f"(trust: {reward_components['trust_delta']:.2f}, "
                f"sentiment: {reward_components['sentiment_shift']:.2f}, "
                f"feedback: {reward_components['user_feedback']:.2f}, "
                f"engagement: {reward_components['engagement_continuation']:.2f})"
            )

            return composite_reward

        except Exception as e:
            logger.warning(f"Failed to compute composite reward, using satisfaction potential: {e}")
            cognitive_cycle.metadata["reward_breakdown"] = {"error": str(e)}
            return cognitive_cycle.outcome_signals.user_satisfaction_potential

    async def trigger_reflection(self, user_id: UUID, num_cycles: int, trigger_type: str) -> bool:
        """
        Internal API to trigger self-reflection for N past cycles.
        Enqueues a background task for the SelfReflectionAndDiscoveryEngine.
        """
        logger.info(f"OrchestrationService: Enqueuing reflection task for user {user_id}, {num_cycles} cycles, type {trigger_type}.")
        self.background_task_queue.enqueue_task(
            self.self_reflection_discovery_engine.execute_reflection(user_id, num_cycles),
            task_name=f"reflection_task_{user_id}_{datetime.utcnow().timestamp()}"
        )
        return True

    async def trigger_discovery(self, user_id: UUID, discovery_type: str, context: Optional[str]) -> bool:
        """
        Internal API to trigger a specific autonomous discovery type.
        Enqueues a background task for the SelfReflectionAndDiscoveryEngine.
        """
        logger.info(f"OrchestrationService: Enqueuing discovery task for user {user_id}, type {discovery_type}.")
        self.background_task_queue.enqueue_task(
            self.self_reflection_discovery_engine.execute_discovery(user_id, discovery_type, context),
            task_name=f"discovery_task_{user_id}_{discovery_type}_{datetime.utcnow().timestamp()}"
        )
        return True
