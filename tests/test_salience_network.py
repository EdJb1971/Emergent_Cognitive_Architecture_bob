from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.agents.memory_agent import MemoryAgent
from src.models.core_models import CognitiveCycle
from src.models.memory_models import ConversationSummary, MemoryAccessStats
from src.services.memory_consolidation_service import MemoryConsolidationService
from src.services.salience_network import SalienceNetwork
from src.services.working_memory_buffer import WorkingMemoryBuffer


NOW = datetime(2026, 8, 2, 12, 0, 0)


def _cycle(
    text: str,
    *,
    score: float | None = None,
    age_days: float = 0.0,
    emotional: float = 0.5,
    novelty: float = 0.5,
    priority: float = 0.7,
    must_keep: bool = False,
) -> CognitiveCycle:
    return CognitiveCycle(
        user_id=uuid4(),
        timestamp=NOW - timedelta(days=age_days),
        user_input=text,
        final_response=f"Response about {text}",
        score=score,
        metadata={
            "must_keep": must_keep,
            "emotional_salience": {
                "salience_score": emotional,
                "novelty_score": novelty,
            },
            "consolidation_metadata": {"consolidation_priority": priority},
        },
    )


def test_salience_ranking_preserves_baseline_and_explains_every_candidate():
    baseline_first = _cycle("routine gardening note", score=0.92, age_days=3)
    protected = _cycle(
        "the user's enduring safety preference",
        score=0.51,
        age_days=200,
        must_keep=True,
    )
    network = SalienceNetwork(enabled=True, shadow_mode=True, top_k=1)

    assessment = network.assess_memories(
        [baseline_first, protected],
        query_text="garden safety preference",
        now=NOW,
    )

    assert assessment.baseline_order == [
        str(baseline_first.cycle_id),
        str(protected.cycle_id),
    ]
    assert assessment.recommended_order[0] == str(protected.cycle_id)
    assert assessment.candidates[0].salience_score >= 0.9
    assert assessment.candidates[0].reasons[0] == "must_keep"
    assert assessment.pruning_applied is False
    assert assessment.top_k == 1
    assert all(0.0 <= item.salience_score <= 1.0 for item in assessment.candidates)
    assert all(
        set(item.weighted_contributions) == set(network.DEFAULT_WEIGHTS)
        for item in assessment.candidates
    )


def test_salience_recency_handles_aware_timestamps_and_ties_are_stable():
    first = _cycle("same signal", score=0.7)
    first.timestamp = first.timestamp.replace(tzinfo=timezone.utc)
    second = _cycle("same signal", score=0.7)
    second.timestamp = second.timestamp.replace(tzinfo=timezone.utc)
    network = SalienceNetwork(enabled=True)

    assessment = network.assess_memories([first, second], now=NOW)

    assert assessment.recommended_order == assessment.baseline_order
    assert assessment.candidates[0].factors.recency == 1.0


def test_working_memory_exposes_only_active_advisory_hints():
    memory_id = str(uuid4())
    advisory = {
        "enabled": True,
        "shadow_mode": True,
        "top_k": 1,
        "candidates": [
            {
                "memory_id": memory_id,
                "baseline_rank": 2,
                "salience_score": 0.91,
                "reasons": ["emotionally_salient"],
            }
        ],
    }
    buffer = WorkingMemoryBuffer()
    buffer.context.recalled_memories = [{"cycle_id": memory_id}]
    buffer.context.salience_advisory = advisory

    assert "Memory Priority Advisory" not in buffer.get_enhanced_prompt_context()

    advisory["shadow_mode"] = False
    assert "baseline item #2" in buffer.get_enhanced_prompt_context()
    assert "consider every recalled memory" in buffer.get_enhanced_prompt_context()


@pytest.mark.asyncio
async def test_memory_agent_attaches_advisory_without_reordering_baseline():
    first = _cycle("older architecture", score=0.9, age_days=30)
    second = _cycle("current emotional priority", score=0.7, emotional=1.0)
    llm_service = MagicMock()
    llm_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    memory_service = MagicMock()
    memory_service.query_memory = AsyncMock(return_value=[first, second])
    memory_service._access_stats = {first.user_id: MemoryAccessStats(stm_hits=0)}
    memory_service.summary_manager = MagicMock()
    memory_service.summary_manager.get_or_create_summary = AsyncMock(
        return_value=ConversationSummary(user_id=first.user_id)
    )
    agent = MemoryAgent(
        llm_service=llm_service,
        memory_service=memory_service,
        salience_network=SalienceNetwork(enabled=True, shadow_mode=True),
    )

    result = await agent.process_input("architecture priority", first.user_id)

    analysis = result.analysis
    assert analysis["source_memory_ids"] == [str(first.cycle_id), str(second.cycle_id)]
    assert [str(item["cycle_id"]) for item in analysis["retrieved_context"]] == [
        str(first.cycle_id),
        str(second.cycle_id),
    ]
    assert analysis["salience_advisory"]["candidate_count"] == 2
    assert analysis["salience_advisory"]["pruning_applied"] is False


@pytest.mark.asyncio
async def test_consolidation_keeps_baseline_selection_and_records_replay_advisory():
    selected = _cycle("established important memory", priority=0.8, age_days=10)
    below_threshold = _cycle(
        "new but not yet consolidation eligible",
        priority=0.4,
        emotional=1.0,
    )
    memory_service = MagicMock()
    memory_service.get_user_cycles = AsyncMock(return_value=[selected, below_threshold])
    service = MemoryConsolidationService(
        memory_service=memory_service,
        autobiographical_system=MagicMock(),
        llm_service=MagicMock(),
        salience_network=SalienceNetwork(enabled=True, shadow_mode=True),
    )

    job = await service.create_consolidation_job(str(selected.user_id))

    assert job.cycle_ids_to_process == [str(selected.cycle_id)]
    assert job.salience_advisory is not None
    assert job.salience_advisory["candidate_count"] == 2
    assert job.salience_advisory["baseline_selected_ids"] == [str(selected.cycle_id)]
    assert job.salience_advisory["pruning_applied"] is False
