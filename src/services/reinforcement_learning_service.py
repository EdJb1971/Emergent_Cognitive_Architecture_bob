"""
Reinforcement Learning Service - Basal Ganglia-Inspired Strategy Learning

Implements Q-learning for action selection and habit formation.
Enables Bob to learn which strategies work through experience, not just memory.

Key Functions:
- Strategy selection: Choose actions with highest expected reward (epsilon-greedy)
- Habit formation: After N successes, solidify strategies as defaults
- Temporal difference learning: Update Q-values from outcomes
- Per-user preferences: Learn individual user patterns

Neuroscience Inspiration:
- Basal ganglia: Action selection, reinforcement learning, habit formation
- Dopamine signals: Reward prediction errors drive learning
- Striatum: Stores action-outcome associations (Q-values)
"""

import logging
import random
import json
from typing import Dict, Tuple, List, Optional, Any
from uuid import UUID
from datetime import datetime
from collections import defaultdict
import chromadb

from src.core.config import settings

logger = logging.getLogger(__name__)


class StrategyPerformance:
    """Track performance metrics for a specific strategy."""
    
    def __init__(self):
        self.total_uses: int = 0
        self.successes: int = 0
        self.total_reward: float = 0.0
        self.last_used: Optional[datetime] = None
        self.is_habit: bool = False
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_uses == 0:
            return 0.0
        return self.successes / self.total_uses
    
    @property
    def average_reward(self) -> float:
        """Calculate average reward received."""
        if self.total_uses == 0:
            return 0.0
        return self.total_reward / self.total_uses


class ReinforcementLearningService:
    """
    Basal ganglia-inspired reinforcement learning for strategy selection.
    
    Uses Q-learning to learn which strategies work in which contexts:
    - Q(context, strategy) → expected reward
    - Epsilon-greedy exploration vs. exploitation
    - Habit formation after repeated successes
    - Per-user and global learning
    
    Example:
        # Conflict between agents
        strategy = await rl_service.select_strategy(
            context="emotional_vs_technical",
            available_strategies=["prioritize_emotional", "prioritize_technical", "blend_both"],
            user_id=user_id
        )
        
        # After user responds
        await rl_service.update_from_outcome(
            context="emotional_vs_technical",
            strategy_used="prioritize_emotional",
            reward=0.8,  # High user satisfaction
            user_id=user_id
        )
    """
    
    # Learning parameters
    ALPHA = 0.1              # Learning rate: how much to update Q-values
    EPSILON_START = 0.2      # Initial exploration rate (20% random)
    EPSILON_MIN = 0.05       # Minimum exploration (always explore 5%)
    EPSILON_DECAY = 0.995    # Decay rate per update
    HABIT_THRESHOLD = 20     # Successes needed to form habit
    DISCOUNT_FACTOR = 0.9    # Future reward discounting (not used in simple Q-learning)
    
    def __init__(self):
        """Initialize the reinforcement learning service."""
        
        # Q-values: {(context, strategy): expected_reward}
        self.q_values: Dict[Tuple[str, str], float] = defaultdict(float)
        
        # Strategy performance tracking: {(context, strategy): StrategyPerformance}
        self.strategy_stats: Dict[Tuple[str, str], StrategyPerformance] = defaultdict(StrategyPerformance)
        
        # Per-user learned preferences: {user_id: {context: preferred_strategy}}
        self.user_habits: Dict[UUID, Dict[str, str]] = defaultdict(dict)
        
        # Global learned habits: {context: preferred_strategy}
        self.global_habits: Dict[str, str] = {}
        
        # Exploration rate (decreases over time)
        self.epsilon = self.EPSILON_START
        
        # Update counter for decay
        self.update_count = 0
        
        # ChromaDB client and collection (initialized via connect())
        self.client: Optional[chromadb.Client] = None
        self.q_values_collection = None
        self.habits_collection = None
        
        logger.info(
            f"ReinforcementLearningService initialized. "
            f"Alpha={self.ALPHA}, Epsilon={self.epsilon:.3f}, "
            f"Habit threshold={self.HABIT_THRESHOLD}"
        )
    
    async def connect(self, client: Optional[chromadb.Client] = None):
        """
        Connect to ChromaDB for Q-value and habit persistence.
        
        Args:
            client: Optional ChromaDB client. If None, creates a new persistent client.
        """
        try:
            if client:
                self.client = client
            else:
                self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Collection for Q-values
            self.q_values_collection = self.client.get_or_create_collection(
                name="rl_q_values",
                metadata={"description": "Reinforcement learning Q-values for strategy selection"}
            )
            
            # Collection for habits (global and per-user)
            self.habits_collection = self.client.get_or_create_collection(
                name="rl_habits",
                metadata={"description": "Learned habits from reinforcement learning"}
            )
            
            # Load existing Q-values from persistence
            await self._load_from_persistence()
            
            logger.info(
                f"ReinforcementLearningService connected to ChromaDB. "
                f"Loaded {len(self.q_values)} Q-values, "
                f"{len(self.global_habits)} global habits."
            )
            
        except Exception as e:
            logger.error(f"Failed to connect ReinforcementLearningService to ChromaDB: {e}", exc_info=True)
            raise
    
    async def _load_from_persistence(self):
        """Load Q-values and habits from ChromaDB."""
        try:
            # Load Q-values
            q_results = self.q_values_collection.get()
            if q_results and q_results['ids']:
                for doc_id, metadata in zip(q_results['ids'], q_results['metadatas']):
                    try:
                        context = metadata.get('context')
                        strategy = metadata.get('strategy')
                        q_value = metadata.get('q_value', 0.0)
                        
                        if context and strategy:
                            self.q_values[(context, strategy)] = q_value
                            
                            # Restore stats if available
                            stats = self.strategy_stats[(context, strategy)]
                            stats.total_uses = metadata.get('total_uses', 0)
                            stats.successes = metadata.get('successes', 0)
                            stats.total_reward = metadata.get('total_reward', 0.0)
                            stats.is_habit = metadata.get('is_habit', False)
                            
                    except Exception as e:
                        logger.warning(f"Failed to load Q-value entry {doc_id}: {e}")
            
            # Load habits
            habit_results = self.habits_collection.get()
            if habit_results and habit_results['ids']:
                for doc_id, metadata in zip(habit_results['ids'], habit_results['metadatas']):
                    try:
                        habit_type = metadata.get('type')
                        context = metadata.get('context')
                        strategy = metadata.get('strategy')
                        
                        if context and strategy:
                            if habit_type == 'global':
                                self.global_habits[context] = strategy
                            elif habit_type == 'user':
                                user_id_str = metadata.get('user_id')
                                if user_id_str:
                                    user_id = UUID(user_id_str)
                                    self.user_habits[user_id][context] = strategy
                    except Exception as e:
                        logger.warning(f"Failed to load habit entry {doc_id}: {e}")
            
            logger.info(f"Loaded {len(self.q_values)} Q-values from persistence")
            
        except Exception as e:
            logger.error(f"Error loading RL data from persistence: {e}", exc_info=True)
    
    async def _persist_q_value(self, context: str, strategy: str):
        """Persist a single Q-value update to ChromaDB."""
        if not self.q_values_collection:
            return
        
        try:
            doc_id = f"{context}||{strategy}"
            q_value = self.q_values.get((context, strategy), 0.0)
            stats = self.strategy_stats.get((context, strategy))
            
            metadata = {
                'context': context,
                'strategy': strategy,
                'q_value': q_value,
                'total_uses': stats.total_uses if stats else 0,
                'successes': stats.successes if stats else 0,
                'total_reward': stats.total_reward if stats else 0.0,
                'is_habit': stats.is_habit if stats else False,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Upsert to ChromaDB
            self.q_values_collection.upsert(
                ids=[doc_id],
                documents=[f"Q-value for {context} with strategy {strategy}"],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Failed to persist Q-value for {context}||{strategy}: {e}")
    
    async def _persist_habit(self, context: str, strategy: str, user_id: Optional[UUID] = None):
        """Persist a learned habit to ChromaDB."""
        if not self.habits_collection:
            return
        
        try:
            habit_type = 'user' if user_id else 'global'
            doc_id = f"{habit_type}||{context}"
            if user_id:
                doc_id += f"||{str(user_id)}"
            
            metadata = {
                'type': habit_type,
                'context': context,
                'strategy': strategy,
                'formed_at': datetime.utcnow().isoformat()
            }
            
            if user_id:
                metadata['user_id'] = str(user_id)
            
            self.habits_collection.upsert(
                ids=[doc_id],
                documents=[f"{habit_type.capitalize()} habit: {strategy} for {context}"],
                metadatas=[metadata]
            )
            
            logger.debug(f"Persisted {habit_type} habit: {context} → {strategy}")
            
        except Exception as e:
            logger.error(f"Failed to persist habit for {context}: {e}")
    
    async def select_strategy(
        self,
        context: str,
        available_strategies: List[str],
        user_id: Optional[UUID] = None,
        force_exploit: bool = False
    ) -> str:
        """
        Select a strategy using epsilon-greedy policy.
        
        Args:
            context: The situation context (e.g., "emotional_vs_technical")
            available_strategies: List of possible strategies to choose from
            user_id: Optional user ID for personalized selection
            force_exploit: If True, always exploit (no exploration)
            
        Returns:
            Selected strategy name
        """
        if not available_strategies:
            raise ValueError("No strategies available for selection")
        
        # Check for user-specific habit first
        if user_id and context in self.user_habits.get(user_id, {}):
            habit_strategy = self.user_habits[user_id][context]
            if habit_strategy in available_strategies:
                logger.info(
                    f"Using user habit for context '{context}': {habit_strategy} "
                    f"(user {user_id})"
                )
                return habit_strategy
        
        # Check for global habit
        if context in self.global_habits:
            habit_strategy = self.global_habits[context]
            if habit_strategy in available_strategies:
                logger.info(f"Using global habit for context '{context}': {habit_strategy}")
                return habit_strategy
        
        # Epsilon-greedy selection
        if not force_exploit and random.random() < self.epsilon:
            # Explore: choose random strategy
            selected = random.choice(available_strategies)
            logger.debug(
                f"Exploring: random strategy '{selected}' for context '{context}' "
                f"(ε={self.epsilon:.3f})"
            )
            return selected
        
        # Exploit: choose strategy with highest Q-value
        q_values = {
            strategy: self.q_values.get((context, strategy), 0.0)
            for strategy in available_strategies
        }
        
        max_q = max(q_values.values())
        
        # If all Q-values are 0 (untried), choose randomly
        if max_q == 0.0 and all(q == 0.0 for q in q_values.values()):
            selected = random.choice(available_strategies)
            logger.debug(f"Untried context '{context}', random selection: {selected}")
            return selected
        
        # Choose best strategy (break ties randomly)
        best_strategies = [s for s, q in q_values.items() if q == max_q]
        selected = random.choice(best_strategies)
        
        logger.info(
            f"Exploiting: strategy '{selected}' for context '{context}' "
            f"(Q={max_q:.3f}, ε={self.epsilon:.3f})"
        )
        
        return selected
    
    async def update_from_outcome(
        self,
        context: str,
        strategy_used: str,
        reward: float,
        user_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update Q-values based on observed reward (temporal difference learning).
        
        Args:
            context: The situation context
            strategy_used: The strategy that was executed
            reward: Reward signal (-1.0 to 1.0, typically from user satisfaction)
            user_id: Optional user ID for personalized learning
            metadata: Optional additional info (e.g., trust_delta, sentiment)
        """
        key = (context, strategy_used)
        
        # Current Q-value
        current_q = self.q_values[key]
        
        # Temporal difference update: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        # Note: In simple Q-learning without future states, we just update toward reward
        new_q = current_q + self.ALPHA * (reward - current_q)
        self.q_values[key] = new_q
        
        # Update strategy statistics
        stats = self.strategy_stats[key]
        stats.total_uses += 1
        stats.total_reward += reward
        stats.last_used = datetime.utcnow()
        
        if reward > 0.5:  # Count as success if reward > 0.5
            stats.successes += 1
        
        # Check for habit formation
        if stats.successes >= self.HABIT_THRESHOLD and not stats.is_habit:
            stats.is_habit = True
            
            # Form global habit
            self.global_habits[context] = strategy_used
            logger.info(
                f"🎯 HABIT FORMED: '{strategy_used}' for context '{context}' "
                f"(success rate: {stats.success_rate:.2%}, uses: {stats.total_uses})"
            )
            
            # Persist global habit
            await self._persist_habit(context, strategy_used)
            
            # Form user-specific habit
            if user_id:
                self.user_habits[user_id][context] = strategy_used
                await self._persist_habit(context, strategy_used, user_id)
                logger.info(f"User-specific habit formed for user {user_id}")
        
        # Persist Q-value update to ChromaDB
        await self._persist_q_value(context, strategy_used)
        
        # Decay exploration rate
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
        self.update_count += 1
        
        logger.info(
            f"Q-value updated: {context} + {strategy_used} = {new_q:.3f} "
            f"(reward: {reward:.2f}, uses: {stats.total_uses}, "
            f"success_rate: {stats.success_rate:.2%}, ε={self.epsilon:.3f})"
        )
        
        if metadata:
            logger.debug(f"Update metadata: {metadata}")
    
    def get_strategy_performance(self, context: str, strategy: str) -> Optional[StrategyPerformance]:
        """Get performance stats for a specific context-strategy pair."""
        key = (context, strategy)
        return self.strategy_stats.get(key)
    
    def get_context_performance(self, context: str) -> Dict[str, StrategyPerformance]:
        """Get performance stats for all strategies in a context."""
        return {
            strategy: stats
            for (ctx, strategy), stats in self.strategy_stats.items()
            if ctx == context
        }
    
    def get_user_habits(self, user_id: UUID) -> Dict[str, str]:
        """Get all learned habits for a specific user."""
        return dict(self.user_habits.get(user_id, {}))
    
    def get_global_habits(self) -> Dict[str, str]:
        """Get all learned global habits."""
        return dict(self.global_habits)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get overall learning statistics."""
        total_contexts = len(set(ctx for ctx, _ in self.q_values.keys()))
        total_strategies = len(self.q_values)
        total_habits = len(self.global_habits)
        
        avg_q = sum(self.q_values.values()) / len(self.q_values) if self.q_values else 0.0
        
        return {
            "total_contexts": total_contexts,
            "total_strategy_entries": total_strategies,
            "total_habits_formed": total_habits,
            "average_q_value": avg_q,
            "current_epsilon": self.epsilon,
            "total_updates": self.update_count,
            "total_users_with_habits": len(self.user_habits)
        }
    
    def reset_exploration(self):
        """Reset exploration rate (useful for testing or major context shifts)."""
        self.epsilon = self.EPSILON_START
        logger.info(f"Exploration rate reset to {self.epsilon}")
    
    def export_q_table(self) -> Dict[str, float]:
        """Export Q-values for persistence."""
        return {
            f"{context}||{strategy}": q_value
            for (context, strategy), q_value in self.q_values.items()
        }
    
    def import_q_table(self, q_table: Dict[str, float]) -> None:
        """Import Q-values from persistence."""
        for key, q_value in q_table.items():
            try:
                context, strategy = key.split("||")
                self.q_values[(context, strategy)] = q_value
            except ValueError:
                logger.warning(f"Invalid Q-table key format: {key}")
        
        logger.info(f"Imported {len(q_table)} Q-values from persistence")


# Common context types for the ECA system
class ContextTypes:
    """Standard context identifiers for RL strategy selection."""
    
    # Agent conflicts
    EMOTIONAL_VS_TECHNICAL = "emotional_vs_technical"
    CREATIVE_VS_FACTUAL = "creative_vs_factual"
    EMPATHY_VS_ACCURACY = "empathy_vs_accuracy"
    DEPTH_VS_BREVITY = "depth_vs_brevity"
    
    # Conversation types
    TECHNICAL_QUESTION = "technical_question"
    EMOTIONAL_SUPPORT = "emotional_support"
    CREATIVE_BRAINSTORM = "creative_brainstorm"
    FACTUAL_LOOKUP = "factual_lookup"
    PHILOSOPHICAL_DISCUSSION = "philosophical_discussion"
    
    # Response strategies
    ANALOGY_FIRST = "analogy_first"
    DIRECT_ANSWER = "direct_answer"
    CLARIFYING_QUESTION = "clarifying_question"
    EMPATHETIC_ACKNOWLEDGMENT = "empathetic_acknowledgment"
    
    # Knowledge gaps
    SEARCH_VS_ADMIT = "search_vs_admit_uncertainty"
    DETAILED_VS_OVERVIEW = "detailed_vs_overview"


class StrategyTypes:
    """Standard strategy identifiers for RL action selection."""
    
    # Conflict resolution
    PRIORITIZE_EMOTIONAL = "prioritize_emotional"
    PRIORITIZE_TECHNICAL = "prioritize_technical"
    BLEND_BOTH = "blend_both"
    WEIGHTED_SYNTHESIS = "weighted_synthesis"
    
    # Response ordering
    EMPATHY_THEN_FACTS = "empathy_then_facts"
    FACTS_THEN_EMPATHY = "facts_then_empathy"
    ANALOGY_THEN_DETAILS = "analogy_then_details"
    DETAILS_THEN_ANALOGY = "details_then_analogy"
    
    # Uncertainty handling
    SEARCH_FIRST = "search_first"
    ADMIT_UNCERTAINTY = "admit_uncertainty"
    ATTEMPT_ANSWER = "attempt_with_caveats"
    ASK_CLARIFICATION = "ask_for_clarification"
