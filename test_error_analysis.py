#!/usr/bin/env python3
"""
Test script for structured error analysis system
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import uuid4
from src.models.core_models import ErrorAnalysis
from src.services.conflict_monitor import ConflictMonitor
from src.services.meta_cognitive_monitor import MetaCognitiveMonitor, ActionRecommendation, GapType

async def test_error_analysis():
    """Test the error analysis generation system"""

    print("Testing Error Analysis Generation System")
    print("=" * 50)

    # Test ConflictMonitor error analysis
    print("\n1. Testing ConflictMonitor Error Analysis")
    conflict_monitor = ConflictMonitor()

    # Mock conflict data
    from src.models.core_models import Conflict, ConflictType
    mock_conflicts = [
        Conflict(
            conflict_type=ConflictType.SENTIMENT_COHERENCE_MISMATCH,
            severity=0.8,
            description="Emotional agent positive, critic agent negative",
            resolution_strategy="weighted_synthesis",
            involved_agents=["emotional", "critic"]
        )
    ]

    cycle_id = uuid4()
    error_analysis = await conflict_monitor.generate_error_analysis(
        cycle_id=cycle_id,
        coherence_score=0.3,
        conflicts=mock_conflicts,
        agents_activated=["perception", "emotional", "critic"],
        user_input_summary="User is asking for help with a problem",
        response_summary="System provided conflicting advice",
        cycle_metadata={"test": True}
    )

    if error_analysis:
        print(f"✓ Generated conflict error analysis: {error_analysis.failure_type}")
        print(f"  Severity: {error_analysis.severity_score}")
        print(f"  Category: {error_analysis.primary_error_category}")
        print(f"  Recommended sequence: {error_analysis.recommended_agent_sequence}")
    else:
        print("✗ Failed to generate conflict error analysis")

    # Test MetaCognitiveMonitor error analysis
    print("\n2. Testing MetaCognitiveMonitor Error Analysis")
    meta_monitor = MetaCognitiveMonitor()

    error_analysis_meta = await meta_monitor.generate_error_analysis(
        cycle_id=cycle_id,
        recommendation=ActionRecommendation.DECLINE_POLITELY,
        gap_type=GapType.TOPIC_UNKNOWN,
        confidence_score=0.2,
        query="What is quantum entanglement?",
        agents_activated=["perception", "discovery"],
        user_input_summary="Complex physics question",
        response_summary="System declined to answer",
        cycle_metadata={"test": True}
    )

    if error_analysis_meta:
        print(f"✓ Generated meta-cognitive error analysis: {error_analysis_meta.failure_type}")
        print(f"  Severity: {error_analysis_meta.severity_score}")
        print(f"  Category: {error_analysis_meta.primary_error_category}")
        print(f"  Skill improvements: {error_analysis_meta.skill_improvement_areas}")
    else:
        print("✗ Failed to generate meta-cognitive error analysis")

    print("\n3. Testing ErrorAnalysis Model Structure")
    if error_analysis:
        print(f"✓ ErrorAnalysis model has {len(error_analysis.model_fields)} fields")
        required_fields = ['cycle_id', 'failure_type', 'severity_score', 'agents_activated', 'primary_error_category']
        for field in required_fields:
            if hasattr(error_analysis, field):
                print(f"  ✓ Has {field}")
            else:
                print(f"  ✗ Missing {field}")

    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_error_analysis())