"""Tests for STM summarization and consolidation."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4
import json

from src.services.summary_manager import SummaryManager
from src.models.core_models import CognitiveCycle
from src.models.memory_models import ConversationSummary
from src.services.llm_integration_service import LLMIntegrationService

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock(spec=LLMIntegrationService)
    service.generate_completion.return_value = (
        "Topic: Testing\n"
        "Entity: Test Suite\n"
        "Context: Running test scenarios\n"
        "Preference: test_mode: automated\n"
        "State: testing"
    )
    service.generate_embedding.return_value = [0.1] * 10
    return service

@pytest.fixture
def summary_manager(mock_llm_service):
    """Create a SummaryManager with mock LLM service."""
    manager = SummaryManager(mock_llm_service)
    # Mock ChromaDB setup
    manager.client = Mock()
    manager.summaries_collection = Mock()
    return manager

@pytest.fixture
def sample_cycles():
    """Create a list of sample cognitive cycles."""
    user_id = uuid4()
    return [
        CognitiveCycle(
            id=uuid4(),
            user_id=user_id,
            input_text="What is the test about?",
            output_text="This is a test of the STM summarization.",
            context="Testing context",
            timestamp=datetime.utcnow()
        ),
        CognitiveCycle(
            id=uuid4(),
            user_id=user_id,
            input_text="How does it work?",
            output_text="It summarizes recent interactions.",
            context="Testing context",
            timestamp=datetime.utcnow()
        )
    ]

@pytest.mark.asyncio
async def test_summarize_stm_basic(summary_manager, sample_cycles):
    """Test basic STM summarization functionality."""
    summary, consolidation = await summary_manager.summarize_stm(
        user_id=sample_cycles[0].user_id,
        cycles=sample_cycles
    )
    
    assert isinstance(summary, ConversationSummary)
    assert consolidation  # Should have summary text
    assert "Testing" in summary.key_topics
    assert summary.conversation_state == "testing"
    assert len(summary.referenced_memories) == len(sample_cycles)

@pytest.mark.asyncio
async def test_stm_consolidated_storage(summary_manager, sample_cycles):
    """Test storage of consolidated STM record."""
    await summary_manager.summarize_stm(
        user_id=sample_cycles[0].user_id,
        cycles=sample_cycles
    )
    
    # Verify ChromaDB upsert was called correctly
    summary_manager.summaries_collection.upsert.assert_called()
    call_args = summary_manager.summaries_collection.upsert.call_args[1]
    
    assert len(call_args['ids']) == 1
    assert call_args['ids'][0].startswith('stm_consolidated:')
    assert len(call_args['embeddings']) == 1
    assert len(call_args['metadatas']) == 1
    assert call_args['metadatas'][0]['cycle_count'] == len(sample_cycles)

@pytest.mark.asyncio
async def test_empty_cycles_error(summary_manager):
    """Test that empty cycles list raises error."""
    with pytest.raises(ValueError):
        await summary_manager.summarize_stm(user_id=uuid4(), cycles=[])

@pytest.mark.asyncio
async def test_llm_error_handling(summary_manager, sample_cycles):
    """Test handling of LLM service errors."""
    summary_manager.llm_service.generate_completion.side_effect = Exception("LLM Error")
    
    with pytest.raises(Exception) as exc_info:
        await summary_manager.summarize_stm(
            user_id=sample_cycles[0].user_id,
            cycles=sample_cycles
        )
    assert "Failed to summarize STM" in str(exc_info.value)

@pytest.mark.asyncio
async def test_storage_error_handling(summary_manager, sample_cycles):
    """Test handling of storage errors."""
    summary_manager.summaries_collection.upsert.side_effect = Exception("Storage Error")
    
    with pytest.raises(Exception) as exc_info:
        await summary_manager.summarize_stm(
            user_id=sample_cycles[0].user_id,
            cycles=sample_cycles
        )
    assert "Failed to store" in str(exc_info.value)

@pytest.mark.asyncio
async def test_summary_text_generation(summary_manager, sample_cycles):
    """Test generation of summary text with proper formatting."""
    summary, consolidation = await summary_manager.summarize_stm(
        user_id=sample_cycles[0].user_id,
        cycles=sample_cycles
    )
    
    stm_text = summary_manager._generate_stm_text(summary, consolidation)
    assert "Recent Conversation Summary:" in stm_text
    assert "Active Topics:" in stm_text
    assert "Current State:" in stm_text
    
@pytest.mark.asyncio
async def test_concurrent_summarization(summary_manager, sample_cycles):
    """Test concurrent summarization requests."""
    import asyncio
    
    async def summarize():
        return await summary_manager.summarize_stm(
            user_id=sample_cycles[0].user_id,
            cycles=sample_cycles
        )
    
    # Run multiple summarizations concurrently
    results = await asyncio.gather(summarize(), summarize())
    
    assert len(results) == 2
    assert all(isinstance(r[0], ConversationSummary) for r in results)
    assert all(r[1] for r in results)  # All have consolidation text