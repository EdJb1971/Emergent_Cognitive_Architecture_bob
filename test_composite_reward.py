#!/usr/bin/env python3
"""
Simple test script to validate composite reward computation logic.
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.orchestration_service import OrchestrationService
from src.models.core_models import CognitiveCycle, OutcomeSignals, UserRequest
from src.models.agent_models import EmotionalProfile
from uuid import uuid4
from unittest.mock import AsyncMock

async def test_composite_reward():
    """Test the composite reward computation logic."""

    # Create mock OrchestrationService with minimal setup
    orchestration = OrchestrationService(
        perception_agent=None,
        emotional_agent=None,
        memory_agent=None,
        planning_agent=None,
        creative_agent=None,
        critic_agent=None,
        discovery_agent=None,
        web_browsing_service=None,
        cognitive_brain=None,
        memory_service=None,
        background_task_queue=None,
        self_reflection_discovery_engine=None,
        emotional_memory_service=AsyncMock()
    )

    # Mock emotional memory service
    mock_profile = EmotionalProfile(
        user_id="test-user",
        trust_level=0.6,
        last_emotion_detected="positive"
    )
    orchestration.emotional_memory_service.get_or_create_profile.return_value = mock_profile

    # Create test cognitive cycle
    user_request = UserRequest(
        user_id=uuid4(),
        input_text="Thank you for the helpful response!",
        session_id=uuid4()
    )

    cognitive_cycle = CognitiveCycle(
        user_id=user_request.user_id,
        session_id=user_request.session_id,
        user_input=user_request.input_text,
        agent_outputs=[]
    )

    cognitive_cycle.outcome_signals = OutcomeSignals(
        user_satisfaction_potential=0.8,
        engagement_potential=0.7
    )

    cognitive_cycle.metadata = {
        "pre_interaction_trust": 0.5,
        "pre_interaction_sentiment": "neutral"
    }

    # Test composite reward computation
    reward = await orchestration._compute_composite_reward(
        user_request.user_id,
        cognitive_cycle,
        None  # pre_interaction_profile
    )

    print(f"Composite reward: {reward:.3f}")
    print(f"Reward breakdown: {cognitive_cycle.metadata.get('reward_breakdown', {})}")

    # Validate reward is in valid range
    assert 0.0 <= reward <= 1.0, f"Reward {reward} is out of valid range [0,1]"

    print("✓ Composite reward computation test passed!")

if __name__ == "__main__":
    asyncio.run(test_composite_reward())