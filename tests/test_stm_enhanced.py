"""Tests for the enhanced ShortTermMemory with token awareness and persistence."""
import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
import tempfile
from pathlib import Path

from src.models.memory_models import ShortTermMemory, STMSnapshot, MemoryAccessStats
from src.models.core_models import CognitiveCycle
from src.utils.token_counter import TokenCounter

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for STM snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_cycle():
    """Create a sample cognitive cycle for testing."""
    return CognitiveCycle(
        id=uuid4(),
        user_id=uuid4(),
        input_text="What is the meaning of life?",
        output_text="The meaning of life is to grow and help others grow.",
        context="Philosophy discussion",
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def stm_instance():
    """Create a ShortTermMemory instance for testing."""
    return ShortTermMemory(user_id=uuid4())

@pytest.mark.asyncio
async def test_add_cycle_token_counting(stm_instance, sample_cycle):
    """Test that adding a cycle properly counts tokens."""
    needs_summary, cycles = await stm_instance.add_cycle(sample_cycle)
    
    assert stm_instance.token_count > 0
    assert not needs_summary  # First cycle shouldn't trigger summary
    assert cycles is None
    assert len(stm_instance.recent_cycles) == 1

@pytest.mark.asyncio
async def test_token_budget_enforcement(stm_instance):
    """Test that token budget is enforced with summarization trigger."""
    # Create a cycle that will consume significant tokens
    large_cycle = CognitiveCycle(
        id=uuid4(),
        user_id=stm_instance.user_id,
        input_text="A" * 5000,  # Large input
        output_text="B" * 5000,  # Large output
        context="Test context",
        timestamp=datetime.utcnow()
    )
    
    # Add cycles until we exceed budget
    cycles_added = 0
    cycles_to_summarize = None
    
    while cycles_to_summarize is None and cycles_added < 10:
        needs_summary, cycles_to_summarize = await stm_instance.add_cycle(large_cycle)
        cycles_added += 1
        
    assert needs_summary
    assert cycles_to_summarize is not None
    assert len(cycles_to_summarize) > 0

@pytest.mark.asyncio
async def test_flush_cycles(stm_instance, sample_cycle):
    """Test that flushing cycles properly updates token count."""
    await stm_instance.add_cycle(sample_cycle)
    initial_tokens = stm_instance.token_count
    
    await stm_instance.flush_cycles([sample_cycle])
    
    assert stm_instance.token_count < initial_tokens
    assert len(stm_instance.recent_cycles) == 0

@pytest.mark.asyncio
async def test_concurrent_access(stm_instance, sample_cycle):
    """Test that concurrent access is properly handled."""
    async def add_cycles():
        for _ in range(5):
            await stm_instance.add_cycle(sample_cycle)
            await asyncio.sleep(0.01)  # Simulate work
    
    # Run multiple add operations concurrently
    await asyncio.gather(add_cycles(), add_cycles())
    
    # Verify final state is consistent
    assert len(stm_instance.recent_cycles) == 10
    assert stm_instance.token_count > 0

def test_snapshot_persistence(stm_instance, sample_cycle, temp_data_dir):
    """Test saving and loading STM snapshots."""
    # Add a cycle and save
    asyncio.run(stm_instance.add_cycle(sample_cycle))
    snapshot_path = stm_instance.save_snapshot(temp_data_dir)
    
    assert snapshot_path.exists()
    assert snapshot_path.suffix == '.pkl'
    
    # Load into new instance
    loaded_stm = ShortTermMemory.load_snapshot(
        stm_instance.user_id,
        temp_data_dir
    )
    
    assert loaded_stm is not None
    assert loaded_stm.user_id == stm_instance.user_id
    assert len(loaded_stm.recent_cycles) == 1
    assert loaded_stm.token_count == stm_instance.token_count

def test_snapshot_validation(stm_instance, sample_cycle, temp_data_dir):
    """Test snapshot validation and age checking."""
    asyncio.run(stm_instance.add_cycle(sample_cycle))
    stm_instance.save_snapshot(temp_data_dir)
    
    # Try loading with strict age limit
    loaded_stm = ShortTermMemory.load_snapshot(
        stm_instance.user_id,
        temp_data_dir,
        max_age=timedelta(seconds=1)
    )
    assert loaded_stm is not None
    
    # Wait and try again
    asyncio.sleep(2)
    loaded_stm = ShortTermMemory.load_snapshot(
        stm_instance.user_id,
        temp_data_dir,
        max_age=timedelta(seconds=1)
    )
    assert loaded_stm is None  # Too old

def test_memory_stats_tracking(stm_instance, sample_cycle):
    """Test that memory access statistics are properly tracked."""
    stats = MemoryAccessStats()
    
    # Update stats with token usage
    stats.update_token_stats(100)
    assert stats.total_cycles_processed == 1
    assert stats.avg_token_usage == 100.0
    
    # Add more stats
    stats.update_token_stats(200)
    assert stats.total_cycles_processed == 2
    assert 100.0 < stats.avg_token_usage < 200.0