"""Tests for memory service token-aware STM and summary integration."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from uuid import uuid4
from datetime import datetime
import chromadb

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.models.core_models import CognitiveCycle, MemoryQueryRequest
from src.models.memory_models import ShortTermMemory, ConversationSummary
from src.utils.token_counter import TokenCounter

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_embedding.return_value = [0.1] * 10
    service.generate_completion.return_value = "Test summary text"
    return service

@pytest.fixture
def mock_chromadb():
    """Create a mock ChromaDB client."""
    client = Mock(spec=chromadb.Client)
    collection = Mock(spec=chromadb.Collection)
    collection.add = Mock()
    collection.query.return_value = {
        'metadatas': [[{'json_data': '{}', 'user_id': '123'}]],
        'distances': [[0.5]]
    }
    client.get_or_create_collection.return_value = collection
    return client

@pytest.fixture
def memory_service(mock_llm_service):
    """Create a memory service with mocked dependencies."""
    service = MemoryService(mock_llm_service)
    service.client = mock_chromadb
    service.cycles_collection = mock_chromadb.get_or_create_collection()
    service.summary_manager.client = mock_chromadb
    service.summary_manager.summaries_collection = mock_chromadb.get_or_create_collection()
    return service

@pytest.fixture
def sample_cycle():
    """Create a sample cognitive cycle."""
    return CognitiveCycle(
        id=uuid4(),
        user_id=uuid4(),
        input_text="This is a test input",
        output_text="This is a test response",
        context="Testing context",
        timestamp=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_add_cycle_under_budget(memory_service, sample_cycle):
    """Test adding a cycle when under token budget."""
    # Mock token counting to return small number
    memory_service.token_counter.count_tokens = Mock(return_value=100)
    
    should_flush, cycles = await memory_service.add_cycle(sample_cycle)
    
    assert not should_flush
    assert cycles is None
    assert sample_cycle.user_id in memory_service._stm_cache

@pytest.mark.asyncio
async def test_add_cycle_over_budget(memory_service, sample_cycle):
    """Test adding a cycle that exceeds token budget."""
    # Mock token counting to force flush
    memory_service.token_counter.count_tokens = Mock(return_value=30000)
    
    should_flush, cycles = await memory_service.add_cycle(sample_cycle)
    
    assert should_flush
    assert cycles is not None
    assert len(cycles) > 0

@pytest.mark.asyncio
async def test_flush_to_ltm(memory_service, sample_cycle):
    """Test flushing cycles from STM to LTM."""
    cycles = [sample_cycle]
    
    await memory_service.flush_to_ltm(sample_cycle.user_id, cycles)
    
    # Verify summary was generated
    memory_service.summary_manager.summarize_stm.assert_called_once()
    
    # Verify cycles were stored in LTM
    memory_service.cycles_collection.add.assert_called()

@pytest.mark.asyncio
async def test_query_memory_combined(memory_service, sample_cycle):
    """Test querying both STM and LTM."""
    # Add cycle to STM
    await memory_service.add_cycle(sample_cycle)
    
    # Set up query request
    query = MemoryQueryRequest(
        user_id=sample_cycle.user_id,
        query_text="test query",
        limit=10
    )
    
    # Execute query
    results = await memory_service.query_memory(query)
    
    # Should get results from both STM and LTM
    assert len(results) > 0
    
    # Verify stats were updated
    stats = await memory_service.get_access_stats(sample_cycle.user_id)
    assert stats.total_queries > 0

@pytest.mark.asyncio
async def test_token_budget_enforcement(memory_service):
    """Test that token budget is properly enforced."""
    user_id = uuid4()
    cycles = []
    
    # Create cycles that will exceed budget
    for i in range(5):
        cycle = CognitiveCycle(
            id=uuid4(),
            user_id=user_id,
            input_text=f"Test input {i}",
            output_text=f"Test output {i}",
            context="Testing",
            timestamp=datetime.utcnow()
        )
        cycles.append(cycle)
    
    # Mock token counting to return large number
    memory_service.token_counter.count_tokens = Mock(return_value=10000)
    
    # Add cycles until we trigger a flush
    flush_triggered = False
    for cycle in cycles:
        should_flush, _ = await memory_service.add_cycle(cycle)
        if should_flush:
            flush_triggered = True
            break
    
    assert flush_triggered, "Token budget enforcement should have triggered a flush"

@pytest.mark.asyncio
async def test_concurrent_stm_access(memory_service, sample_cycle):
    """Test concurrent access to STM is properly synchronized."""
    import asyncio
    
    async def add_cycle():
        return await memory_service.add_cycle(sample_cycle)
    
    # Run multiple adds concurrently
    results = await asyncio.gather(add_cycle(), add_cycle(), add_cycle())
    
    # Verify all operations completed
    assert all(isinstance(r, tuple) for r in results)
    
    # Verify STM state is consistent
    stm = memory_service._stm_cache[sample_cycle.user_id]
    assert stm.get_token_count() > 0

@pytest.mark.asyncio
async def test_memory_stats_tracking(memory_service, sample_cycle):
    """Test that memory access statistics are properly tracked."""
    # Add some cycles
    await memory_service.add_cycle(sample_cycle)
    
    # Perform some queries
    query = MemoryQueryRequest(
        user_id=sample_cycle.user_id,
        query_text="test query",
        limit=10
    )
    
    for _ in range(3):
        await memory_service.query_memory(query)
    
    # Check stats
    stats = await memory_service.get_access_stats(sample_cycle.user_id)
    assert stats.total_queries == 3
    assert stats.stm_hits > 0