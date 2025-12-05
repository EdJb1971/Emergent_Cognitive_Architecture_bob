import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.services.summary_manager import SummaryManager
from src.models.core_models import CognitiveCycle, MemoryQueryRequest
from src.models.memory_models import ShortTermMemory, MemoryAccessStats, ConversationSummary
from src.core.exceptions import ConfigurationError, APIException

@pytest.fixture
def mock_llm_service():
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_embedding.return_value = [0.1] * 384
    return service

@pytest.fixture
def mock_summary_manager():
    manager = AsyncMock(spec=SummaryManager)
    summary = ConversationSummary(
        user_id=uuid4(),
        context_points=["Test context"],
        key_topics=["Test topic"]
    )
    manager.get_relevant_summaries.return_value = [summary]
    return manager

@pytest.fixture
def memory_service(mock_llm_service):
    service = MemoryService(llm_service=mock_llm_service)
    service.summary_manager = mock_summary_manager()
    return service

@pytest.mark.asyncio
async def test_memory_service_initialization(memory_service):
    """Test proper initialization of memory components."""
    assert memory_service._stm_cache == {}
    assert memory_service._access_stats == {}
    assert memory_service.summary_manager is not None

@pytest.mark.asyncio
async def test_query_memory_with_summary(memory_service):
    """Test memory querying with summary integration."""
    user_id = uuid4()
    query_request = MemoryQueryRequest(
        user_id=user_id,
        query_text="test query"
    )
    
    # Add some test data
    cycle = CognitiveCycle(
        cycle_id=uuid4(),
        user_id=user_id,
        session_id=uuid4(),
        timestamp=datetime.utcnow(),
        user_input="test input",
        final_response="test response",
        score=0.8
    )
    
    stm = ShortTermMemory(user_id=user_id)
    stm.add_cycle(cycle)
    memory_service._stm_cache[user_id] = stm
    
    # Test query
    results = await memory_service.query_memory(query_request)
    
    # Verify results
    assert len(results) > 0
    assert memory_service._access_stats[user_id].stm_hits > 0
    memory_service.summary_manager.get_relevant_summaries.assert_called_once()

@pytest.mark.asyncio
async def test_memory_error_handling(memory_service):
    """Test error handling in memory operations."""
    user_id = uuid4()
    
    # Test missing user_id
    with pytest.raises(APIException) as exc_info:
        await memory_service.query_memory(MemoryQueryRequest(
            user_id=None,
            query_text="test"
        ))
    assert "user_id is required" in str(exc_info.value)
    
    # Test summary manager failure
    memory_service.summary_manager.get_relevant_summaries.side_effect = Exception("Summary error")
    
    # Should continue without summary
    results = await memory_service.query_memory(MemoryQueryRequest(
        user_id=user_id,
        query_text="test"
    ))
    assert isinstance(results, list)  # Should return empty list, not fail

@pytest.mark.asyncio
async def test_memory_stats_tracking(memory_service):
    """Test proper tracking of memory access statistics."""
    user_id = uuid4()
    query_request = MemoryQueryRequest(
        user_id=user_id,
        query_text="test query"
    )
    
    # First query - should initialize stats
    await memory_service.query_memory(query_request)
    assert user_id in memory_service._access_stats
    
    # Second query - should update stats
    await memory_service.query_memory(query_request)
    stats = memory_service._access_stats[user_id]
    assert stats.stm_misses > 0  # Should have misses since STM is empty
    assert isinstance(stats.last_updated, datetime)

@pytest.mark.asyncio
async def test_memory_cleanup(memory_service):
    """Test memory cleanup mechanisms."""
    # Add some old data
    old_user_id = uuid4()
    old_stm = ShortTermMemory(user_id=old_user_id)
    old_stm.last_accessed = datetime.utcnow() - timedelta(hours=2)
    memory_service._stm_cache[old_user_id] = old_stm
    
    # Add some recent data
    recent_user_id = uuid4()
    recent_stm = ShortTermMemory(user_id=recent_user_id)
    memory_service._stm_cache[recent_user_id] = recent_stm
    
    # Trigger cleanup (if implemented)
    # TODO: Implement cleanup mechanism
    
    # Verify recent data remains
    assert recent_user_id in memory_service._stm_cache