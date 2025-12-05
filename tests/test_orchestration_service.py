import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.services.orchestration_service import OrchestrationService
from src.services.cognitive_brain import CognitiveBrain
from src.services.memory_service import MemoryService
from src.services.background_task_queue import BackgroundTaskQueue
from src.services.self_reflection_discovery_engine import SelfReflectionAndDiscoveryEngine
from src.agents.perception_agent import PerceptionAgent
from src.agents.emotional_agent import EmotionalAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.creative_agent import CreativeAgent
from src.agents.critic_agent import CriticAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.models.core_models import UserRequest, AgentOutput, CognitiveCycle, ResponseMetadata, OutcomeSignals
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
    assert len(cognitive_cycle.agent_outputs) == 7
    assert cognitive_cycle.final_response == "Mock final response"
    assert cognitive_cycle.response_metadata.response_type == "informational"
    assert cognitive_cycle.outcome_signals.user_satisfaction_potential == 0.8

    for agent_id, mock_agent in mock_agents.items():
        mock_agent.process_input.assert_called_once()

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
    assert len(cognitive_cycle.agent_outputs) == 7
    
    failed_agent_output = next(ao for ao in cognitive_cycle.agent_outputs if ao.agent_id == "perception_agent")
    assert failed_agent_output.analysis["status"] == "failed"
    assert "Perception error" in failed_agent_output.analysis["error"]
    assert failed_agent_output.confidence == 0.0
    assert failed_agent_output.priority == 1

    for agent_id, mock_agent in mock_agents.items():
        mock_agent.process_input.assert_called_once()

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
async def test_trigger_reflection_enqueues_task(orchestration_service, mock_background_task_queue, mock_self_reflection_discovery_engine):
    user_id = uuid4()
    num_cycles = 5
    trigger_type = "manual"

    result = await orchestration_service.trigger_reflection(user_id, num_cycles, trigger_type)
    assert result is True
    mock_background_task_queue.enqueue_task.assert_called_once()
    args, kwargs = mock_background_task_queue.enqueue_task.call_args
    assert args[0].__qualname__ == 'SelfReflectionAndDiscoveryEngine.execute_reflection'
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
    assert args[0].__qualname__ == 'SelfReflectionAndDiscoveryEngine.execute_discovery'
    assert kwargs['task_name'].startswith("discovery_task_")
