"""
Models for memory management including short-term memory, conversation summaries, and stats.
"""
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import UUID, uuid4
from pathlib import Path
import pickle
from pydantic import BaseModel, Field, ValidationError

from src.core.config import settings
from src.core.exceptions import APIException
from src.utils.token_counter import TokenCounter
from .core_models import CognitiveCycle

logger = logging.getLogger(__name__)

class ConversationSummary(BaseModel):
    """
    Represents a dynamic summary of the conversation context.
    Updated incrementally as the conversation progresses.
    """
    summary_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this summary")
    user_id: UUID = Field(..., description="The user this summary belongs to")
    key_topics: List[str] = Field(default_factory=list, description="Main topics discussed in conversation")
    entities: Set[str] = Field(default_factory=set, description="Important entities mentioned (names, concepts, etc)")
    context_points: List[str] = Field(default_factory=list, description="Key context points to remember")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="Observed user preferences and patterns")
    latest_topic: Optional[str] = Field(None, description="Most recent topic being discussed")
    conversation_state: str = Field(default="initial", description="Current state of conversation")
    referenced_memories: List[UUID] = Field(default_factory=list, description="IDs of important referenced memories")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    update_count: int = Field(default=0, description="Number of times summary has been updated")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the summary content")

    def add_topic(self, topic: str) -> None:
        """Add a new topic if not already present."""
        if topic not in self.key_topics:
            self.key_topics.append(topic)
            self.latest_topic = topic

    def add_entity(self, entity: str) -> None:
        """Add a new entity to the set."""
        self.entities.add(entity)

    def add_context(self, point: str) -> None:
        """Add a new context point if not redundant. Limits total context points to prevent unbounded growth."""
        MAX_CONTEXT_POINTS = 50  # Limit to prevent summary text from becoming too large
        
        if point not in self.context_points:
            self.context_points.append(point)
            
            # If we've exceeded the limit, remove oldest context points
            if len(self.context_points) > MAX_CONTEXT_POINTS:
                # Keep the most recent points
                self.context_points = self.context_points[-MAX_CONTEXT_POINTS:]

    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference."""
        self.user_preferences[key] = value

    def mark_updated(self) -> None:
        """Mark the summary as updated."""
        self.last_updated = datetime.utcnow()
        self.update_count += 1

class STMSnapshot(BaseModel):
    """
    Represents a persisted snapshot of STM state for recovery.
    """
    user_id: UUID
    cycles: List[CognitiveCycle]
    token_count: int
    last_summary_timestamp: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"

class ShortTermMemory(BaseModel):
    """
    Represents the short-term memory cache for a specific user.
    Token-aware with persistence and async-safe operations.
    """
    user_id: UUID = Field(..., description="The ID of the user this STM belongs to")
    recent_cycles: List[CognitiveCycle] = Field(default_factory=list, 
                                              description="List of recent cognitive cycles, newest first")
    token_count: int = Field(default=0, description="Current token count in STM")
    token_budget: int = Field(default=25000, description="Maximum tokens to keep in STM")
    token_reserve: int = Field(default=5000, description="Token reserve for system prompts")
    last_accessed: datetime = Field(default_factory=datetime.utcnow,
                                  description="Timestamp of last memory access")
    last_summary: datetime = Field(default_factory=datetime.utcnow,
                                 description="Timestamp of last summary generation")
    
    # Runtime state (not persisted)
    _token_counter: Optional[TokenCounter] = None
    _lock: Optional[asyncio.Lock] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self._token_counter = TokenCounter()
        self._lock = asyncio.Lock()
        # Set initial budgets based on model capabilities
        if not data.get('token_budget'):
            self.token_budget, self.token_reserve = self._token_counter.get_token_budget()

    async def add_cycle(self, cycle: CognitiveCycle) -> Tuple[bool, Optional[List[CognitiveCycle]]]:
        """
        Add a new cycle to short-term memory, maintaining token budget.
        Returns (needs_summary, cycles_to_summarize) if token budget exceeded.
        
        Args:
            cycle: The new CognitiveCycle to add
            
        Returns:
            Tuple[bool, Optional[List[CognitiveCycle]]]: 
                - needs_summary: True if summarization needed
                - cycles_to_summarize: List of cycles to summarize if needed
        """
        async with self._lock:
            # Count tokens in new cycle
            cycle_tokens = await self._count_cycle_tokens(cycle)
            
            # If this cycle alone exceeds budget, log warning and add anyway
            if cycle_tokens > self.token_budget:
                logger.warning(
                    f"Single cycle exceeds token budget for user {self.user_id}: "
                    f"{cycle_tokens} > {self.token_budget}"
                )
            
            # Add cycle and update counts
            self.recent_cycles.insert(0, cycle)
            self.token_count += cycle_tokens
            self.last_accessed = datetime.utcnow()
            
            # Check if we need to summarize
            if self.token_count > self.token_budget:
                # Find cycles to summarize (oldest first, up to budget/2)
                target = self.token_budget // 2  # Aim to reduce to 50% of budget
                cycles_to_summarize = []
                tokens_to_remove = 0
                
                for old_cycle in reversed(self.recent_cycles[1:]):  # Skip newest
                    old_tokens = await self._count_cycle_tokens(old_cycle)
                    if tokens_to_remove + old_tokens <= (self.token_count - target):
                        cycles_to_summarize.append(old_cycle)
                        tokens_to_remove += old_tokens
                    else:
                        break
                
                return True, cycles_to_summarize
            
            return False, None

    async def flush_cycles(self, cycles_to_remove: List[CognitiveCycle]) -> None:
        """
        Remove specific cycles from STM after summarization.
        
        Args:
            cycles_to_remove: List of cycles that were summarized
        """
        async with self._lock:
            # Remove cycles and update token count
            for cycle in cycles_to_remove:
                if cycle in self.recent_cycles:
                    self.recent_cycles.remove(cycle)
                    self.token_count -= await self._count_cycle_tokens(cycle)
            
            self.last_summary = datetime.utcnow()

    def get_recent_cycles(self, limit: Optional[int] = None) -> List[CognitiveCycle]:
        """Get the most recent cycles, up to specified limit."""
        self.last_accessed = datetime.utcnow()
        return self.recent_cycles[:limit] if limit else self.recent_cycles

    async def _count_cycle_tokens(self, cycle: CognitiveCycle) -> int:
        """Count tokens in a cognitive cycle using all relevant fields."""
        # Combine all text fields that will be used in prompts
        texts = [
            getattr(cycle, 'user_input', '') or "",
            getattr(cycle, 'final_response', '') or "",
            json.dumps(getattr(cycle, 'metadata', {}) or {}),
        ]
        token_counts = await self._token_counter.count_tokens_batch(texts)
        return sum(token_counts)

    def save_snapshot(self, base_path: Optional[str] = None) -> Path:
        """
        Save STM state to disk for recovery.
        
        Args:
            base_path: Optional base directory for snapshots
            
        Returns:
            Path: Path to the saved snapshot file
        """
        base_path = base_path or settings.CHROMA_DB_PATH
        snapshot_dir = Path(base_path) / "stm_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot = STMSnapshot(
            user_id=self.user_id,
            cycles=self.recent_cycles,
            token_count=self.token_count,
            last_summary_timestamp=self.last_summary
        )
        
        # Save both JSON (for inspection) and pickle (for reliable restoration)
        file_base = snapshot_dir / f"{self.user_id}_stm"
        json_path = file_base.with_suffix('.json')
        pickle_path = file_base.with_suffix('.pkl')
        
        # Save JSON for inspection
        with json_path.open('w') as f:
            json.dump(snapshot.dict(), f, default=str, indent=2)
            
        # Save pickle for restoration
        with pickle_path.open('wb') as f:
            pickle.dump(snapshot, f)
            
        logger.info(f"Saved STM snapshot for user {self.user_id} to {pickle_path}")
        return pickle_path

    @classmethod
    def load_snapshot(cls, user_id: UUID, base_path: Optional[str] = None, 
                     max_age: Optional[timedelta] = None) -> Optional['ShortTermMemory']:
        """
        Load STM state from disk if available and valid.
        
        Args:
            user_id: The user's UUID
            base_path: Optional base directory for snapshots
            max_age: Maximum age for valid snapshot
            
        Returns:
            Optional[ShortTermMemory]: Restored STM or None if no valid snapshot
        """
        try:
            base_path = base_path or settings.CHROMA_DB_PATH
            snapshot_dir = Path(base_path) / "stm_snapshots"
            pickle_path = snapshot_dir / f"{user_id}_stm.pkl"
            
            if not pickle_path.exists():
                return None
                
            # Check age if specified
            if max_age:
                age = datetime.utcnow() - datetime.fromtimestamp(pickle_path.stat().st_mtime)
                if age > max_age:
                    logger.warning(f"Snapshot for user {user_id} too old (age={age})")
                    return None
            
            # Load and validate snapshot
            with pickle_path.open('rb') as f:
                snapshot = pickle.load(f)
                
            if not isinstance(snapshot, STMSnapshot):
                raise ValueError("Invalid snapshot format")
                
            if snapshot.user_id != user_id:
                raise ValueError("User ID mismatch in snapshot")
            
            # Create new STM with restored state
            stm = cls(
                user_id=user_id,
                recent_cycles=snapshot.cycles,
                token_count=snapshot.token_count,
                last_summary=snapshot.last_summary_timestamp
            )
            
            logger.info(f"Restored STM from snapshot for user {user_id}")
            return stm
            
        except Exception as e:
            logger.error(f"Failed to load STM snapshot for user {user_id}: {e}")
            return None

class MemoryAccessStats(BaseModel):
    """
    Tracks memory access patterns for optimization and monitoring.
    """
    stm_hits: int = Field(default=0, description="Number of successful STM retrievals")
    ltm_hits: int = Field(default=0, description="Number of LTM retrievals")
    stm_misses: int = Field(default=0, description="Number of STM misses requiring LTM query")
    ltm_queries: int = Field(default=0, description="Number of LTM queries performed")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    summary_generations: int = Field(default=0, description="Number of summaries generated")
    avg_token_usage: float = Field(default=0.0, description="Average token usage in STM")
    total_cycles_processed: int = Field(default=0, description="Total cognitive cycles processed")
    avg_relevance: float = Field(default=0.0, description="Average relevance score of last query")
    
    def update_token_stats(self, token_count: int) -> None:
        """Update running average of token usage."""
        if self.total_cycles_processed == 0:
            self.avg_token_usage = float(token_count)
        else:
            self.avg_token_usage = (
                (self.avg_token_usage * self.total_cycles_processed + token_count) /
                (self.total_cycles_processed + 1)
            )
        self.total_cycles_processed += 1
        self.last_updated = datetime.utcnow()

    def record_query(self, *, stm_hits: int, ltm_hits: int, avg_relevance: float) -> None:
        """Record stats for a memory query."""
        self.stm_hits += stm_hits
        self.ltm_hits += ltm_hits
        self.ltm_queries += 1
        self.avg_relevance = avg_relevance
        self.last_updated = datetime.utcnow()