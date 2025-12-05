import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from src.agents.memory_agent import MemoryAgent
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import CognitiveCycle, MemoryQueryRequest, AgentOutput
from src.models.memory_models import ShortTermMemory, MemoryAccessStats
from src.core.exceptions import AgentServiceException
from src.core.config import settings

@pytest.fixture
def mock_llm_service():
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_embedding.return_value = [0.1] * 384  # Mocked embedding vector
    return service

@pytest.fixture
def mock_memory_service():
    service = AsyncMock(spec=MemoryService)
    service._access_stats = {}
    return service

@pytest.fixture
def memory_agent(mock_llm_service, mock_memory_service):
    return MemoryAgent(llm_service=mock_llm_service, memory_service=mock_memory_service)

@pytest.fixture
def sample_cognitive_cycle():
    return CognitiveCycle(
        cycle_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        timestamp=datetime.utcnow(),
        user_input="Test input",
        final_response="Test response",
        score=0.8  # Add relevance score
    )

@pytest.mark.asyncio
async def test_memory_agent_confidence_calculation(memory_agent):
    """Test the confidence calculation logic with different scenarios."""
    # Test with no memories
    confidence = memory_agent._calculate_memory_confidence([], 0, 0.0)
    assert confidence == 0.5  # baseline confidence

    # Test with memories but no STM hits
    memories = [MagicMock(score=0.8) for _ in range(3)]
    confidence = memory_agent._calculate_memory_confidence(memories, 0, 0.8)
    assert confidence < 0.95  # should be less than max
    assert confidence > 0.5  # but higher than baseline

    # Test with memories and STM hits
    confidence = memory_agent._calculate_memory_confidence(memories, 2, 0.8)
    assert confidence > 0.5  # should be boosted by STM hits

@pytest.mark.asyncio
async def test_memory_agent_memory_analysis(memory_agent, sample_cognitive_cycle):
    """Test the memory analysis functionality."""
    # Test with no memories
    analysis = memory_agent._analyze_memories([], 0)
    assert analysis.relevance_score == 0.0
    assert len(analysis.retrieved_context) == 0

    # Test with memories and STM hits
    memories = [sample_cognitive_cycle]
    analysis = memory_agent._analyze_memories(memories, 1)
    assert analysis.relevance_score > 0.0
    assert len(analysis.retrieved_context) == 1
    assert len(analysis.source_memory_ids) == 1

@pytest.mark.asyncio
async def test_memory_agent_process_input(memory_agent, mock_memory_service, sample_cognitive_cycle):
    """Test the complete process_input flow."""
    user_id = uuid4()
    
    # Setup mock returns
    mock_memory_service.query_memory.return_value = [sample_cognitive_cycle]
    mock_memory_service._access_stats[user_id] = MemoryAccessStats(stm_hits=1)

    # Process input
    result = await memory_agent.process_input("test query", user_id)

    # Verify result structure
    assert isinstance(result, AgentOutput)
    assert result.agent_id == memory_agent.AGENT_ID
    assert result.confidence > 0.5  # Should be higher due to STM hit
    assert result.priority == 8
    
    # Verify memory service interaction
    mock_memory_service.query_memory.assert_called_once()
    called_request = mock_memory_service.query_memory.call_args[0][0]
    assert called_request.user_id == user_id
    assert called_request.query_text == "test query"

@pytest.mark.asyncio
async def test_memory_agent_error_handling(memory_agent, mock_memory_service):
    """Test error handling in the memory agent."""
    user_id = uuid4()

    # Test LLM service error
    memory_agent.llm_service.generate_embedding.side_effect = Exception("LLM Error")
    with pytest.raises(AgentServiceException) as exc_info:
        await memory_agent.process_input("test", user_id)
    assert "error occurred" in str(exc_info.value)

    # Test memory service error
    memory_agent.llm_service.generate_embedding.side_effect = None  # Reset error
    mock_memory_service.query_memory.side_effect = Exception("Memory Error")
    with pytest.raises(AgentServiceException) as exc_info:
        await memory_agent.process_input("test", user_id)
    assert "error occurred" in str(exc_info.value)

@pytest.mark.asyncio
async def test_memory_agent_stm_integration(memory_agent, mock_memory_service, sample_cognitive_cycle):
    """Test integration with short-term memory features."""
    user_id = uuid4()
    
    # Setup mock returns for multiple queries
    mock_memory_service.query_memory.return_value = [sample_cognitive_cycle]
    
    # Simulate increasing STM hits
    for stm_hits in range(3):
        mock_memory_service._access_stats[user_id] = MemoryAccessStats(stm_hits=stm_hits)
        result = await memory_agent.process_input("test query", user_id)
        
        # Verify confidence increases with more STM hits
        if stm_hits > 0:
            assert result.confidence > 0.5  # Should be boosted by STM hits
            
    # Verify memory service was called appropriately
    assert mock_memory_service.query_memory.call_count == 3