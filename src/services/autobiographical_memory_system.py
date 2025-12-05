"""
Autobiographical Memory System - Full Episodic and Semantic Memory

Inspired by human autobiographical memory, this system:
- Stores rich episodic memories (mental time travel)
- Extracts semantic knowledge from episodes
- Constructs narrative timelines
- Enables "what happened when" queries
- Supports identity and continuity over time

This is an enhancement to the simpler AutobiographicalMemory in SelfModel.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
import chromadb
from chromadb.config import Settings

from src.models.agent_models import EpisodicMemory, SemanticMemory
from src.models.core_models import CognitiveCycle

logger = logging.getLogger(__name__)


class AutobiographicalMemorySystem:
    """
    Full autobiographical memory system with episodic and semantic memory.
    Enables rich narrative construction and knowledge extraction.
    """
    
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.episodic_collection = None
        self.semantic_collection = None
        logger.info("AutobiographicalMemorySystem initialized.")
    
    async def connect(self, client: chromadb.Client):
        """Connect to shared ChromaDB client and get/create collections."""
        self.client = client
        
        # Episodic memories collection
        self.episodic_collection = self.client.get_or_create_collection(
            name="episodic_memories",
            metadata={"description": "Rich episodic memories for mental time travel"}
        )
        logger.info("Connected to episodic_memories collection.")
        
        # Semantic memories collection
        self.semantic_collection = self.client.get_or_create_collection(
            name="semantic_memories",
            metadata={"description": "Extracted semantic knowledge and concepts"}
        )
        logger.info("Connected to semantic_memories collection.")
    
    async def create_episodic_memory(
        self, 
        cycle: CognitiveCycle,
        narrative: str,
        significance: float,
        emotional_tone: str = "neutral",
        key_insights: Optional[List[str]] = None
    ) -> EpisodicMemory:
        """
        Create a rich episodic memory from a cognitive cycle.
        
        Args:
            cycle: The cognitive cycle to memorialize
            narrative: Rich narrative description
            significance: How significant this episode is (0.0-1.0)
            emotional_tone: Overall emotional tone
            key_insights: Key learnings from this episode
            
        Returns:
            EpisodicMemory object
        """
        episode_id = str(uuid4())
        
        # Extract participants
        participants = ["user", "system"]
        # Could extract mentioned names from cycle content
        
        # Extract sensory details from multimodal inputs
        sensory_details = {}
        if cycle.metadata.get("image_present"):
            sensory_details["visual"] = "Image provided"
        if cycle.metadata.get("audio_present"):
            sensory_details["auditory"] = "Audio provided"
        
        episode = EpisodicMemory(
            episode_id=episode_id,
            timestamp=cycle.timestamp,
            narrative=narrative,
            participants=participants,
            emotional_tone=emotional_tone,
            significance=significance,
            key_insights=key_insights or [],
            sensory_details=sensory_details,
            cycle_id=str(cycle.cycle_id)
        )
        
        # Store in ChromaDB
        await self._store_episodic_memory(episode, str(cycle.user_id))
        
        logger.debug(f"Created episodic memory {episode_id} with significance {significance:.2f}")
        return episode
    
    async def _store_episodic_memory(self, episode: EpisodicMemory, user_id: str):
        """Store episodic memory in ChromaDB."""
        if not self.episodic_collection:
            logger.warning("Episodic collection not initialized, skipping storage")
            return
        
        # Use narrative as the document for embedding
        document = episode.narrative
        
        # Metadata for filtering
        metadata = {
            "user_id": user_id,
            "timestamp": episode.timestamp.isoformat(),
            "emotional_tone": episode.emotional_tone,
            "significance": episode.significance,
            "cycle_id": episode.cycle_id or "",
            "participants": json.dumps(episode.participants),
            "key_insights": json.dumps(episode.key_insights)
        }
        
        self.episodic_collection.add(
            ids=[episode.episode_id],
            documents=[document],
            metadatas=[metadata]
        )
        
        logger.debug(f"Stored episodic memory {episode.episode_id} in ChromaDB")
    
    async def query_episodic_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_significance: float = 0.0,
        limit: int = 10
    ) -> List[EpisodicMemory]:
        """
        Query episodic memories by content, time, or significance.
        
        Args:
            user_id: User whose memories to query
            query: Optional semantic query
            time_range: Optional (start, end) datetime tuple
            min_significance: Minimum significance threshold
            limit: Maximum number of memories to return
            
        Returns:
            List of EpisodicMemory objects
        """
        if not self.episodic_collection:
            logger.warning("Episodic collection not initialized")
            return []
        
        # Build where clause
        where = {"user_id": user_id}
        if min_significance > 0.0:
            where["significance"] = {"$gte": min_significance}
        
        try:
            if query:
                # Semantic search
                results = self.episodic_collection.query(
                    query_texts=[query],
                    where=where,
                    n_results=limit
                )
            else:
                # Get all matching
                results = self.episodic_collection.get(
                    where=where,
                    limit=limit
                )
            
            # Reconstruct EpisodicMemory objects
            episodes = []
            ids = results.get("ids", [[]])[0] if query else results.get("ids", [])
            metadatas = results.get("metadatas", [[]])[0] if query else results.get("metadatas", [])
            documents = results.get("documents", [[]])[0] if query else results.get("documents", [])
            
            for episode_id, metadata, narrative in zip(ids, metadatas, documents):
                # Filter by time range if specified
                if time_range:
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                    if not (time_range[0] <= timestamp <= time_range[1]):
                        continue
                
                episode = EpisodicMemory(
                    episode_id=episode_id,
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    narrative=narrative,
                    participants=json.loads(metadata.get("participants", "[]")),
                    emotional_tone=metadata.get("emotional_tone", "neutral"),
                    significance=metadata.get("significance", 0.5),
                    key_insights=json.loads(metadata.get("key_insights", "[]")),
                    cycle_id=metadata.get("cycle_id")
                )
                episodes.append(episode)
            
            logger.debug(f"Retrieved {len(episodes)} episodic memories for user {user_id}")
            return episodes
            
        except Exception as e:
            logger.error(f"Error querying episodic memories: {e}")
            return []
    
    async def construct_narrative_timeline(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Construct a narrative timeline of significant episodes.
        
        Args:
            user_id: User whose timeline to construct
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Narrative string describing the timeline
        """
        time_range = None
        if start_time and end_time:
            time_range = (start_time, end_time)
        
        episodes = await self.query_episodic_memories(
            user_id=user_id,
            time_range=time_range,
            min_significance=0.5,  # Only significant episodes
            limit=20
        )
        
        if not episodes:
            return "No significant episodes found in this time period."
        
        # Sort by timestamp
        episodes.sort(key=lambda e: e.timestamp)
        
        # Construct narrative
        narrative_parts = [f"Timeline of {len(episodes)} significant moments:\n"]
        
        for i, episode in enumerate(episodes, 1):
            time_str = episode.timestamp.strftime("%B %d, %Y at %I:%M %p")
            narrative_parts.append(
                f"{i}. [{time_str}] ({episode.emotional_tone}): {episode.narrative}"
            )
            if episode.key_insights:
                narrative_parts.append(f"   Insights: {', '.join(episode.key_insights)}")
        
        return "\n".join(narrative_parts)
    
    async def extract_semantic_memory(
        self,
        user_id: str,
        concept_name: str,
        description: str,
        category: str,
        source_episodes: List[str],
        confidence: float = 0.7
    ) -> SemanticMemory:
        """
        Extract a semantic memory (fact/concept) from episodic experiences.
        
        Args:
            user_id: User identifier
            concept_name: Name of the concept
            description: Description of the concept
            category: Category (user_preference, user_fact, etc.)
            source_episodes: Episode IDs that support this concept
            confidence: Confidence in this knowledge
            
        Returns:
            SemanticMemory object
        """
        concept_id = str(uuid4())
        now = datetime.utcnow()
        
        semantic_memory = SemanticMemory(
            concept_id=concept_id,
            concept_name=concept_name,
            description=description,
            confidence=confidence,
            source_episode_ids=source_episodes,
            first_learned=now,
            last_reinforced=now,
            reinforcement_count=1,
            category=category
        )
        
        # Store in ChromaDB
        await self._store_semantic_memory(semantic_memory, user_id)
        
        logger.debug(f"Extracted semantic memory: {concept_name} ({category})")
        return semantic_memory
    
    async def _store_semantic_memory(self, concept: SemanticMemory, user_id: str):
        """Store semantic memory in ChromaDB."""
        if not self.semantic_collection:
            logger.warning("Semantic collection not initialized, skipping storage")
            return
        
        # Use description as document for embedding
        document = f"{concept.concept_name}: {concept.description}"
        
        metadata = {
            "user_id": user_id,
            "concept_name": concept.concept_name,
            "category": concept.category,
            "confidence": concept.confidence,
            "first_learned": concept.first_learned.isoformat(),
            "last_reinforced": concept.last_reinforced.isoformat(),
            "reinforcement_count": concept.reinforcement_count,
            "source_episode_ids": json.dumps(concept.source_episode_ids)
        }
        
        self.semantic_collection.add(
            ids=[concept.concept_id],
            documents=[document],
            metadatas=[metadata]
        )
        
        logger.debug(f"Stored semantic memory {concept.concept_id} in ChromaDB")
    
    async def query_semantic_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[SemanticMemory]:
        """
        Query semantic memories (facts/concepts).
        
        Args:
            user_id: User whose knowledge to query
            query: Optional semantic query
            category: Optional category filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results
            
        Returns:
            List of SemanticMemory objects
        """
        if not self.semantic_collection:
            logger.warning("Semantic collection not initialized")
            return []
        
        where = {"user_id": user_id}
        if category:
            where["category"] = category
        if min_confidence > 0.0:
            where["confidence"] = {"$gte": min_confidence}
        
        try:
            if query:
                results = self.semantic_collection.query(
                    query_texts=[query],
                    where=where,
                    n_results=limit
                )
            else:
                results = self.semantic_collection.get(
                    where=where,
                    limit=limit
                )
            
            concepts = []
            ids = results.get("ids", [[]])[0] if query else results.get("ids", [])
            metadatas = results.get("metadatas", [[]])[0] if query else results.get("metadatas", [])
            documents = results.get("documents", [[]])[0] if query else results.get("documents", [])
            
            for concept_id, metadata, document in zip(ids, metadatas, documents):
                # Parse concept_name and description from document
                parts = document.split(": ", 1)
                concept_name = parts[0] if len(parts) > 0 else ""
                description = parts[1] if len(parts) > 1 else document
                
                concept = SemanticMemory(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    description=description,
                    confidence=metadata.get("confidence", 0.5),
                    source_episode_ids=json.loads(metadata.get("source_episode_ids", "[]")),
                    first_learned=datetime.fromisoformat(metadata["first_learned"]),
                    last_reinforced=datetime.fromisoformat(metadata["last_reinforced"]),
                    reinforcement_count=metadata.get("reinforcement_count", 1),
                    category=metadata.get("category", "unknown")
                )
                concepts.append(concept)
            
            logger.debug(f"Retrieved {len(concepts)} semantic memories for user {user_id}")
            return concepts
            
        except Exception as e:
            logger.error(f"Error querying semantic memories: {e}")
            return []
    
    async def reinforce_semantic_memory(self, concept_id: str):
        """
        Reinforce a semantic memory (increases confidence and reinforcement count).
        Called when the concept is encountered again.
        """
        if not self.semantic_collection:
            return
        
        try:
            # Get existing concept
            result = self.semantic_collection.get(ids=[concept_id])
            if not result["ids"]:
                logger.warning(f"Concept {concept_id} not found for reinforcement")
                return
            
            metadata = result["metadatas"][0]
            
            # Update reinforcement
            reinforcement_count = metadata.get("reinforcement_count", 1) + 1
            confidence = min(1.0, metadata.get("confidence", 0.5) + 0.05)  # Boost confidence slightly
            
            metadata["reinforcement_count"] = reinforcement_count
            metadata["confidence"] = confidence
            metadata["last_reinforced"] = datetime.utcnow().isoformat()
            
            # Update in ChromaDB
            self.semantic_collection.update(
                ids=[concept_id],
                metadatas=[metadata]
            )
            
            logger.debug(f"Reinforced semantic memory {concept_id}: count={reinforcement_count}, confidence={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error reinforcing semantic memory: {e}")
