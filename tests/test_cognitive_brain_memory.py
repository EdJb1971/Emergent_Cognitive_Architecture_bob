"""Tests for Cognitive Brain memory integration."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from datetime import datetime

from src.services.cognitive_brain import CognitiveBrain
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.models.core_models import (
    CognitiveCycle, AgentOutput, ResponseMetadata,
    OutcomeSignals, MemoryQueryRequest
)
from src.models.memory_models import ConversationSummary, MemoryAccessStats

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = AsyncMock(spec=LLMIntegrationService)
    service.generate_text.return_value = """
    {
        "final_response": "Test response",
        "response_metadata": {
            "response_type": "informational",
            "tone": "neutral",
            "strategies": ["context_integration"],
            "cognitive_moves": ["provide_context"]
        },
        "outcome_signals": {
            "user_satisfaction_potential": 0.8,
            "engagement_potential": 0.7
        }
    }
    """
    service.moderate_content.return_value = {"is_safe": True}
    return service

@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    service = AsyncMock(spec=MemoryService)
    
    # Mock summary manager
    service.summary_manager.get_or_create_summary.return_value = ConversationSummary(
        user_id=uuid4(),
        key_topics=["test", "context"],
        conversation_state="active",
        context_points=["Testing memory integration"]
    )
    
    # Mock memory queries
    service.query_memory.return_value = [
        CognitiveCycle(
            id=uuid4(),
            user_id=uuid4(),
            input_text="Previous test input",
            output_text="Previous test response",
            timestamp=datetime.utcnow()
        )
    ]
    
    # Mock access stats
    service.get_access_stats.return_value = MemoryAccessStats(
        user_id=uuid4(),
        stm_hits=1,
        ltm_hits=1,
        avg_relevance=0.8
    )
    
    return service

@pytest.fixture
def cognitive_brain(mock_llm_service, mock_memory_service):
    """Create a cognitive brain instance with mocked dependencies."""
    return CognitiveBrain(mock_llm_service, mock_memory_service)

@pytest.fixture
def sample_cycle():
    """Create a sample cognitive cycle."""
    return CognitiveCycle(
        id=uuid4(),
        user_id=uuid4(),
        input_text="Test input",
        output_text=None,
        context="Testing context",
        timestamp=datetime.utcnow(),
        agent_outputs=[
            AgentOutput(
                agent_id="test_agent",
                analysis={"key": "value"},
                confidence=0.8,
                priority=5
            )
        ]
    )

@pytest.mark.asyncio
async def test_response_generation_with_memory(cognitive_brain, sample_cycle):
    """Test that response generation includes memory context."""
    response, metadata, signals = await cognitive_brain.generate_response(sample_cycle)
    
    # Verify memory service calls
    cognitive_brain.memory_service.summary_manager.get_or_create_summary.assert_called_once_with(sample_cycle.user_id)
    cognitive_brain.memory_service.query_memory.assert_called_once()
    cognitive_brain.memory_service.get_access_stats.assert_called_once_with(sample_cycle.user_id)
    
    # Verify response components
    assert response == "Test response"
    assert metadata.response_type == "informational"
    assert "context_integration" in metadata.strategies
    
    # Verify memory impact on outcome signals
    assert signals.user_satisfaction_potential > 0.8  # Should be boosted for good memory performance
    assert signals.engagement_potential > 0.7

@pytest.mark.asyncio
async def test_memory_error_handling(cognitive_brain, sample_cycle):
    """Test graceful handling of memory service errors."""
    # Simulate memory service error
    cognitive_brain.memory_service.summary_manager.get_or_create_summary.side_effect = Exception("Memory error")
    
    # Should still generate response even if memory context fails
    response, metadata, signals = await cognitive_brain.generate_response(sample_cycle)
    
    assert response == "Test response"
    assert metadata.response_type == "informational"
    # Signals should not get memory boost
    assert signals.user_satisfaction_potential == 0.8
    assert signals.engagement_potential == 0.7

@pytest.mark.asyncio
async def test_memory_context_integration(cognitive_brain, sample_cycle):
    """Test that memory context is properly integrated into LLM prompt."""
    await cognitive_brain.generate_response(sample_cycle)
    
    # Get the prompt sent to the LLM
    prompt = cognitive_brain.llm_service.generate_text.call_args[1]['prompt']
    
    # Verify memory context in prompt
    assert "Current Conversation:" in prompt
    assert "Relevant Past Context:" in prompt
    assert "test" in prompt  # From mocked summary topics
    assert "Previous test input" in prompt  # From mocked memory query

@pytest.mark.asyncio
async def test_unsafe_content_handling(cognitive_brain, sample_cycle):
    """Test handling of unsafe content with memory context."""
    # Simulate unsafe content detection
    cognitive_brain.llm_service.moderate_content.return_value = {"is_safe": False, "block_reason": "Test violation"}
    
    response, metadata, signals = await cognitive_brain.generate_response(sample_cycle)
    
    assert "cannot provide a response" in response
    assert metadata.response_type == "safety_override"
    assert metadata.strategies == ["safety_protocol"]
    assert signals.user_satisfaction_potential == 0.2
    
    # Memory services should still be called
    cognitive_brain.memory_service.get_access_stats.assert_called_once()