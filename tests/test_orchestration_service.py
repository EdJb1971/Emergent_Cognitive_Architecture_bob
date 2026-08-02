import pytest
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.services.orchestration_service import OrchestrationService
from src.services.cognitive_brain import CognitiveBrain
from src.services.memory_service import MemoryService
from src.services.background_task_queue import BackgroundTaskQueue
from src.services.self_reflection_discovery_engine import SelfReflectionAndDiscoveryEngine
from src.services.escalation_policy import EscalationPolicy
from src.services.meta_cognitive_monitor import ActionRecommendation, GapType
from src.services.research_service import DisabledResearchProvider, ResearchService
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.inquiry_candidate_service import InquiryCandidateService
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.emotional_salience_encoder import EmotionalSalienceEncoder
from src.agents.perception_agent import PerceptionAgent
from src.agents.emotional_agent import EmotionalAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.creative_agent import CreativeAgent
from src.agents.critic_agent import CriticAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.models.core_models import UserRequest, AgentOutput, CognitiveCycle, ResponseMetadata, OutcomeSignals
from src.models.research_models import (
    InquiryCandidate,
    InquiryReviewDisposition,
    InquirySourceType,
    InquiryStatus,
    ResearchClaim,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchSource,
)
from src.core.exceptions import AgentServiceException, APIException

@pytest.fixture
def mock_agents():
    agents = {}
    for agent_cls in [PerceptionAgent, EmotionalAgent, MemoryAgent, PlanningAgent, CreativeAgent, CriticAgent, DiscoveryAgent]:
        mock_agent = AsyncMock(spec=agent_cls)
        mock_agent.AGENT_ID = agent_cls.AGENT_ID
        mock_agent.process_input.return_value = AgentOutput(
            agent_id=mock_agent.AGENT_ID,
            analysis={"summary": f"Mock analysis from {mock_agent.AGENT_ID}"},
            confidence=0.9,
            priority=5
        )
        agents[agent_cls.AGENT_ID] = mock_agent
    return agents

@pytest.fixture
def mock_cognitive_brain():
    mock = AsyncMock(spec=CognitiveBrain)
    mock.theory_of_mind_service = None
    mock.generate_response.return_value = (
        "Mock final response",
        ResponseMetadata(response_type="informational", tone="neutral", strategies=[], cognitive_moves=[]),
        OutcomeSignals(user_satisfaction_potential=0.8, engagement_potential=0.7)
    )
    return mock

@pytest.fixture
def mock_memory_service():
    mock = AsyncMock(spec=MemoryService)
    mock.upsert_cycle.return_value = True
    return mock

@pytest.fixture
def mock_background_task_queue():
    mock = MagicMock(spec=BackgroundTaskQueue)
    mock.enqueue_task.return_value = None
    return mock

@pytest.fixture
def mock_self_reflection_discovery_engine():
    mock = AsyncMock(spec=SelfReflectionAndDiscoveryEngine)
    mock.execute_reflection.return_value = None
    mock.execute_discovery.return_value = None
    return mock

@pytest.fixture
def orchestration_service(
    mock_agents,
    mock_cognitive_brain,
    mock_memory_service,
    mock_background_task_queue,
    mock_self_reflection_discovery_engine
):
    return OrchestrationService(
        perception_agent=mock_agents["perception_agent"],
        emotional_agent=mock_agents["emotional_agent"],
        memory_agent=mock_agents["memory_agent"],
        planning_agent=mock_agents["planning_agent"],
        creative_agent=mock_agents["creative_agent"],
        critic_agent=mock_agents["critic_agent"],
        discovery_agent=mock_agents["discovery_agent"],
        research_service=MagicMock(),
        cognitive_brain=mock_cognitive_brain,
        memory_service=mock_memory_service,
        background_task_queue=mock_background_task_queue,
        self_reflection_discovery_engine=mock_self_reflection_discovery_engine,
        emotional_memory_service=None,  # Not needed for basic tests
        rl_service=None,  # Not needed for basic tests
        meta_cognitive_monitor=None  # Not needed for basic tests
    )

@pytest.mark.asyncio
async def test_orchestrate_cycle_success(orchestration_service, mock_agents, mock_cognitive_brain, mock_memory_service):
    user_request = UserRequest(user_id=uuid4(), input_text="test input", session_id=uuid4())
    
    cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)

    assert isinstance(cognitive_cycle, CognitiveCycle)
    assert cognitive_cycle.user_input == user_request.input_text
    assert [output.agent_id for output in cognitive_cycle.agent_outputs] == ["perception_agent"]
    assert cognitive_cycle.final_response == "Mock final response"
    assert cognitive_cycle.response_metadata.response_type == "informational"
    assert cognitive_cycle.outcome_signals.user_satisfaction_potential == 0.8

    mock_agents["perception_agent"].process_input.assert_called_once()

    mock_cognitive_brain.generate_response.assert_called_once_with(cognitive_cycle)
    mock_memory_service.upsert_cycle.assert_called_once_with(cognitive_cycle)

@pytest.mark.asyncio
async def test_orchestrate_cycle_agent_failure(orchestration_service, mock_agents, mock_cognitive_brain, mock_memory_service):
    user_request = UserRequest(user_id=uuid4(), input_text="test input", session_id=uuid4())
    
    mock_agents["perception_agent"].process_input.side_effect = AgentServiceException(
        agent_id="perception_agent", detail="Perception error", status_code=500
    )

    cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)

    assert isinstance(cognitive_cycle, CognitiveCycle)
    assert len(cognitive_cycle.agent_outputs) >= 1
    
    failed_agent_output = next(ao for ao in cognitive_cycle.agent_outputs if ao.agent_id == "perception_agent")
    assert failed_agent_output.analysis["status"] == "failed"
    assert "Perception error" in failed_agent_output.analysis["error"]
    assert failed_agent_output.confidence == 0.0
    assert failed_agent_output.priority == 1

    mock_agents["perception_agent"].process_input.assert_called_once()

    mock_cognitive_brain.generate_response.assert_called_once()
    mock_memory_service.upsert_cycle.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrate_cycle_cognitive_brain_failure(orchestration_service, mock_cognitive_brain, mock_memory_service):
    user_request = UserRequest(user_id=uuid4(), input_text="test input", session_id=uuid4())
    
    mock_cognitive_brain.generate_response.side_effect = APIException(detail="Brain malfunction", status_code=500)

    cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)

    assert cognitive_cycle.final_response == "An error occurred while generating the response."
    assert cognitive_cycle.response_metadata.response_type == "error"
    assert cognitive_cycle.outcome_signals.user_satisfaction_potential == 0.1
    mock_memory_service.upsert_cycle.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrate_cycle_memory_service_failure(orchestration_service, mock_memory_service):
    user_request = UserRequest(user_id=uuid4(), input_text="test input", session_id=uuid4())
    
    mock_memory_service.upsert_cycle.side_effect = APIException(detail="DB write error", status_code=500)

    cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)

    assert cognitive_cycle.final_response == "Mock final response"
    mock_memory_service.upsert_cycle.assert_called_once()


@pytest.mark.asyncio
async def test_orchestration_persists_salience_advisory_and_emotional_encoding(
    orchestration_service,
    mock_agents,
):
    memory_id = str(uuid4())
    mock_agents["memory_agent"].process_input.return_value = AgentOutput(
        agent_id="memory_agent",
        analysis={
            "retrieved_context": [{"cycle_id": memory_id, "user_input": "prior context"}],
            "relevance_score": 0.8,
            "source_memory_ids": [memory_id],
            "salience_advisory": {
                "version": "salience-v1",
                "enabled": True,
                "shadow_mode": True,
                "candidate_count": 1,
                "top_k": 1,
                "pruning_applied": False,
                "candidates": [],
            },
        },
        confidence=0.9,
        priority=8,
    )
    orchestration_service.emotional_salience_encoder = EmotionalSalienceEncoder()

    cycle = await orchestration_service.orchestrate_cycle(
        UserRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            input_text="Do you remember how I feel about this?",
        )
    )

    assert cycle.metadata["salience_advisory"]["version"] == "salience-v1"
    assert cycle.metadata["salience_advisory"]["pruning_applied"] is False
    assert 0.0 <= cycle.metadata["emotional_salience"]["salience_score"] <= 1.0


@pytest.mark.asyncio
async def test_meta_cognitive_search_recommendation_records_governed_decision(orchestration_service):
    monitor = MagicMock()
    monitor.assess_answer_appropriateness = AsyncMock(
        return_value=(
            ActionRecommendation.SEARCH_FIRST,
            GapType.TOPIC_UNKNOWN,
            0.2,
            "The named fact is absent and may have changed.",
        )
    )
    orchestration_service.meta_cognitive_monitor = monitor
    orchestration_service.research_service = ResearchService(
        EscalationPolicy(research_enabled=False, low_confidence_threshold=0.55),
        DisabledResearchProvider(),
    )
    request = UserRequest(
        user_id=uuid4(),
        input_text="Who is the current director of the institute?",
        session_id=uuid4(),
    )

    cycle = await orchestration_service.orchestrate_cycle(request)

    decision = cycle.metadata["research_escalation"]
    assert decision["disposition"] == "blocked_disabled"
    assert decision["source"] == "meta_cognitive_monitor"
    assert decision["reasons"] == [
        "time_sensitive",
        "low_confidence",
        "named_fact_missing",
        "metacognitive_gap",
    ]


@pytest.mark.asyncio
async def test_waking_cycle_records_shadow_research_drive_and_queues_inquiry(orchestration_service):
    monitor = MagicMock()
    monitor.assess_answer_appropriateness = AsyncMock(
        return_value=(
            ActionRecommendation.SEARCH_FIRST,
            GapType.TOPIC_UNKNOWN,
            0.2,
            "Fresh external evidence is required.",
        )
    )
    orchestration_service.meta_cognitive_monitor = monitor
    orchestration_service.research_service = ResearchService(
        EscalationPolicy(research_enabled=False),
        DisabledResearchProvider(),
    )
    orchestration_service.cognitive_research_drive = CognitiveResearchDrive(
        enabled=False,
        shadow_mode=True,
    )
    inquiry_service = MagicMock(spec=InquiryCandidateService)
    inquiry_service.propose_waking = AsyncMock(return_value=None)
    orchestration_service.inquiry_candidate_service = inquiry_service
    ledger = MagicMock(spec=ResearchCalibrationLedger)
    ledger.record_assessment = AsyncMock(
        return_value=SimpleNamespace(event_id=uuid4())
    )
    orchestration_service.research_calibration_ledger = ledger
    request = UserRequest(
        user_id=uuid4(),
        input_text="Please search the web for the latest institute director.",
        session_id=uuid4(),
        metadata={"local_reasoning_attempts": 999},
    )

    cycle = await orchestration_service.orchestrate_cycle(request)

    assessment = cycle.metadata["cognitive_research_drive"]
    assert assessment["recommended_action"] == "authorize_research"
    assert assessment["effective_action"] == "routine_local"
    assert assessment["shadow_mode"] is True
    assert assessment["signals"]["persistence_after_local_attempts"] == 0.0
    inquiry_service.propose_waking.assert_awaited_once()
    ledger.record_assessment.assert_awaited_once()
    assert "research_calibration_event_id" in cycle.metadata


@pytest.mark.asyncio
async def test_authorized_grounded_packet_is_passed_to_cognitive_brain(orchestration_service):
    monitor = MagicMock()
    monitor.assess_answer_appropriateness = AsyncMock(
        return_value=(
            ActionRecommendation.SEARCH_FIRST,
            GapType.TOPIC_UNKNOWN,
            0.2,
            "Fresh external evidence is required.",
        )
    )
    orchestration_service.meta_cognitive_monitor = monitor
    orchestration_service.research_service = ResearchService(
        EscalationPolicy(research_enabled=True),
        DisabledResearchProvider(),
    )
    orchestration_service.cognitive_research_drive = CognitiveResearchDrive(
        enabled=True,
        shadow_mode=False,
    )
    inquiry_service = MagicMock(spec=InquiryCandidateService)

    async def propose(**kwargs):
        assessment = kwargs["assessment"]
        return InquiryCandidate(
            user_id=kwargs["user_id"],
            question=kwargs["question"],
            source_type=InquirySourceType.WAKING,
            source_cycle_ids=[kwargs["source_cycle_id"]],
            assessment=assessment,
            priority=assessment.drive_score,
            expected_information_gain=assessment.signals.expected_information_gain,
            shadow_mode=False,
        )

    inquiry_service.propose_waking = AsyncMock(side_effect=propose)
    orchestration_service.inquiry_candidate_service = inquiry_service
    packet = ResearchPacket(
        request_id=uuid4(),
        decision_id=uuid4(),
        query="Please search the web for the latest institute director.",
        status=ResearchPacketStatus.COMPLETED,
        provider="grounded-test",
        answer="The current director is verified.",
        claims=[
            ResearchClaim(
                text="The current director is verified.",
                source_ids=["s1"],
                confidence=0.9,
            )
        ],
        sources=[ResearchSource(source_id="s1", title="Institute", url="https://example.test")],
        grounding_verified=True,
    )
    waking_service = MagicMock()

    async def review(**kwargs):
        assessment = kwargs["assessment"]
        stored = InquiryCandidate(
            inquiry_id=kwargs["inquiry_id"],
            user_id=kwargs["user_id"],
            question=packet.query,
            source_type=InquirySourceType.WAKING,
            assessment=assessment,
            priority=assessment.drive_score,
            expected_information_gain=assessment.signals.expected_information_gain,
            status=InquiryStatus.RESEARCHED,
            shadow_mode=False,
        )
        return SimpleNamespace(
            candidate=stored,
            disposition=InquiryReviewDisposition.RESEARCHED,
            rationale="Grounded research completed.",
            research_outcome=SimpleNamespace(packets=[packet]),
        )

    waking_service.review_candidate = AsyncMock(side_effect=review)
    orchestration_service.waking_inquiry_service = waking_service
    request = UserRequest(
        user_id=uuid4(),
        input_text=packet.query,
        session_id=uuid4(),
    )

    cycle = await orchestration_service.orchestrate_cycle(request)

    assert cycle.metadata["waking_inquiry_review"]["disposition"] == "researched"
    assert cycle.metadata["research_packets"][0]["grounding_verified"] is True
    orchestration_service.cognitive_brain.generate_response.assert_awaited_once()
    call = orchestration_service.cognitive_brain.generate_response.await_args
    assert call.kwargs["research_packets"] == (packet,)


@pytest.mark.asyncio
async def test_trigger_reflection_enqueues_task(orchestration_service, mock_background_task_queue, mock_self_reflection_discovery_engine):
    user_id = uuid4()
    num_cycles = 5
    trigger_type = "manual"

    result = await orchestration_service.trigger_reflection(user_id, num_cycles, trigger_type)
    assert result is True
    mock_background_task_queue.enqueue_task.assert_called_once()
    args, kwargs = mock_background_task_queue.enqueue_task.call_args
    assert inspect.iscoroutine(args[0])
    args[0].close()
    assert kwargs['task_name'].startswith("reflection_task_")

@pytest.mark.asyncio
async def test_trigger_discovery_enqueues_task(orchestration_service, mock_background_task_queue, mock_self_reflection_discovery_engine):
    user_id = uuid4()
    discovery_type = "memory_analysis"
    context = "recent interactions"

    result = await orchestration_service.trigger_discovery(user_id, discovery_type, context)
    assert result is True
    mock_background_task_queue.enqueue_task.assert_called_once()
    args, kwargs = mock_background_task_queue.enqueue_task.call_args
    assert inspect.iscoroutine(args[0])
    args[0].close()
    assert kwargs['task_name'].startswith("discovery_task_")
