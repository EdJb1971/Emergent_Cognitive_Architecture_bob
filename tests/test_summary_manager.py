import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from src.services.summary_manager import SummaryManager
from src.services.llm_integration_service import LLMIntegrationService
from src.models.core_models import CognitiveCycle
from src.models.memory_models import ConversationSummary
from src.core.exceptions import ConfigurationError, APIException

@pytest.fixture
def mock_llm_service():
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_embedding.return_value = [0.1] * 384
    service.generate_completion.return_value = """
    Topic: Testing Discussion
    Entity: Test Suite
    Context: Running automated tests
    Preference: testing_style: thorough
    State: engaged
    """
    return service

@pytest.fixture
def summary_manager(mock_llm_service):
    return SummaryManager(llm_service=mock_llm_service)

@pytest.fixture
def sample_cognitive_cycle():
    return CognitiveCycle(
        cycle_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        timestamp=datetime.utcnow(),
        user_input="Let's run some tests",
        final_response="I'll help you with testing"
    )

@pytest.mark.asyncio
async def test_summary_initialization(summary_manager):
    """Test summary creation and retrieval."""
    user_id = uuid4()
    
    # Get new summary
    summary = await summary_manager.get_or_create_summary(user_id)
    
    assert isinstance(summary, ConversationSummary)
    assert summary.user_id == user_id
    assert len(summary.key_topics) == 0
    assert summary.conversation_state == "initial"

    # Get same summary again
    same_summary = await summary_manager.get_or_create_summary(user_id)
    assert summary.summary_id == same_summary.summary_id

@pytest.mark.asyncio
async def test_summary_update(summary_manager, sample_cognitive_cycle, mock_llm_service):
    """Test summary updates with new information."""
    user_id = sample_cognitive_cycle.user_id
    
    # Initial summary
    summary = await summary_manager.get_or_create_summary(user_id)
    initial_update_count = summary.update_count
    
    # Update with new interaction
    updated_summary = await summary_manager.update_summary(user_id, sample_cognitive_cycle)
    
    assert updated_summary.update_count > initial_update_count
    assert len(updated_summary.key_topics) > 0
    assert len(updated_summary.entities) > 0
    assert sample_cognitive_cycle.cycle_id in updated_summary.referenced_memories
    
    # Verify LLM was called
    mock_llm_service.generate_completion.assert_called_once()
    mock_llm_service.generate_embedding.assert_called_once()

@pytest.mark.asyncio
async def test_summary_storage(summary_manager, mock_llm_service):
    """Test summary persistence in ChromaDB."""
    await summary_manager.connect()
    
    user_id = uuid4()
    summary = await summary_manager.get_or_create_summary(user_id)
    
    # Modify summary
    summary.add_topic("Test Topic")
    summary.add_entity("Test Entity")
    
    # Store summary
    await summary_manager._store_summary(summary)
    
    # Try to retrieve it
    retrieved_summaries = await summary_manager.get_relevant_summaries("Test Topic", user_id)
    
    assert len(retrieved_summaries) > 0
    assert retrieved_summaries[0].summary_id == summary.summary_id
    assert "Test Topic" in retrieved_summaries[0].key_topics

@pytest.mark.asyncio
async def test_multiple_summaries(summary_manager, sample_cognitive_cycle):
    """Test handling multiple users' summaries."""
    user_ids = [uuid4(), uuid4(), uuid4()]
    summaries = []
    
    # Create summaries for different users
    for user_id in user_ids:
        cycle = sample_cognitive_cycle.model_copy(update={"user_id": user_id})
        summary = await summary_manager.update_summary(user_id, cycle)
        summaries.append(summary)
    
    # Verify each user has distinct summary
    assert len(set(s.summary_id for s in summaries)) == len(user_ids)
    
    # Verify user isolation in retrieval
    relevant = await summary_manager.get_relevant_summaries("test", user_ids[0])
    assert all(s.user_id == user_ids[0] for s in relevant)

@pytest.mark.asyncio
async def test_error_handling(summary_manager, sample_cognitive_cycle):
    """Test error handling in summary operations."""
    user_id = uuid4()
    
    # Test storage error
    summary_manager.summaries_collection = None
    with pytest.raises(APIException) as exc_info:
        await summary_manager._store_summary(await summary_manager.get_or_create_summary(user_id))
    assert "not initialized" in str(exc_info.value)
    
    # Test LLM error
    summary_manager.llm_service.generate_completion.side_effect = Exception("LLM Error")
    with pytest.raises(APIException) as exc_info:
        await summary_manager.update_summary(user_id, sample_cognitive_cycle)
    assert "failed" in str(exc_info.value)