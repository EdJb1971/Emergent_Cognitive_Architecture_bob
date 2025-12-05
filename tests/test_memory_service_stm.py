import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.models.core_models import CognitiveCycle, MemoryQueryRequest
from src.models.memory_models import ShortTermMemory, MemoryAccessStats
from src.core.exceptions import ConfigurationError, APIException
from src.core.config import settings

@pytest.fixture
def mock_llm_service():
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_embedding.return_value = [0.1] * 384  # Mocked embedding vector
    return service

@pytest.fixture
def memory_service(mock_llm_service):
    return MemoryService(llm_service=mock_llm_service)

@pytest.fixture
def sample_cognitive_cycle():
    return CognitiveCycle(
        cycle_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        timestamp=datetime.utcnow(),
        user_input="Test input",
        final_response="Test response"
    )

@pytest.mark.asyncio
async def test_stm_initialization(memory_service):
    """Test that STM is properly initialized for a new user."""
    user_id = uuid4()
    stm = memory_service._get_or_create_stm(user_id)
    
    assert isinstance(stm, ShortTermMemory)
    assert stm.user_id == user_id
    assert len(stm.recent_cycles) == 0
    assert stm.max_size == 10

@pytest.mark.asyncio
async def test_stm_cycle_addition(memory_service, sample_cognitive_cycle):
    """Test adding cycles to STM and verify max_size enforcement."""
    user_id = sample_cognitive_cycle.user_id
    stm = memory_service._get_or_create_stm(user_id)
    
    # Add more cycles than max_size
    for i in range(15):
        cycle = sample_cognitive_cycle.model_copy(update={
            'cycle_id': uuid4(),
            'timestamp': datetime.utcnow() + timedelta(minutes=i)
        })
        await memory_service.upsert_cycle(cycle)
    
    assert len(stm.recent_cycles) == stm.max_size
    # Verify newest is first
    assert stm.recent_cycles[0].timestamp > stm.recent_cycles[1].timestamp

@pytest.mark.asyncio
async def test_unified_memory_query(memory_service, sample_cognitive_cycle, mock_llm_service):
    """Test that memory queries check both STM and LTM."""
    user_id = sample_cognitive_cycle.user_id
    
    # Add some cycles to memory
    cycles = []
    for i in range(5):
        cycle = sample_cognitive_cycle.model_copy(update={
            'cycle_id': uuid4(),
            'timestamp': datetime.utcnow() + timedelta(minutes=i),
            'user_input': f"Test input {i}"
        })
        cycles.append(cycle)
        await memory_service.upsert_cycle(cycle)
    
    # Query memory
    query_request = MemoryQueryRequest(
        user_id=user_id,
        query_text="Test input",
        limit=10
    )
    
    results = await memory_service.query_memory(query_request)
    
    # Verify results
    assert len(results) > 0
    assert all(r.user_id == user_id for r in results)
    # Verify ordering (newest first)
    assert all(results[i].timestamp >= results[i+1].timestamp 
              for i in range(len(results)-1))

@pytest.mark.asyncio
async def test_memory_access_stats(memory_service, sample_cognitive_cycle):
    """Test that memory access statistics are properly tracked."""
    user_id = sample_cognitive_cycle.user_id
    
    # Add a cycle and query memory multiple times
    await memory_service.upsert_cycle(sample_cognitive_cycle)
    
    query_request = MemoryQueryRequest(
        user_id=user_id,
        query_text="Test input",
        limit=5
    )
    
    # Perform multiple queries
    for _ in range(3):
        await memory_service.query_memory(query_request)
    
    # Check stats
    stats = memory_service._access_stats[user_id]
    assert isinstance(stats, MemoryAccessStats)
    assert stats.stm_hits > 0  # Should have some STM hits
    assert stats.ltm_queries > 0  # Should have queried LTM

@pytest.mark.asyncio
async def test_error_handling(memory_service):
    """Test error handling in memory operations."""
    # Test missing user_id
    with pytest.raises(APIException) as exc_info:
        await memory_service.query_memory(MemoryQueryRequest(
            user_id=None,
            query_text="Test"
        ))
    assert "user_id is required" in str(exc_info.value)

    # Test missing cycle_id
    with pytest.raises(APIException) as exc_info:
        await memory_service.upsert_cycle(CognitiveCycle(
            cycle_id=None,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.utcnow(),
            user_input="Test"
        ))