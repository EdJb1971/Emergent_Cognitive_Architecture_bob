"""
SelfModel Service - Maintains system's sense of self and autobiographical memory.

Inspired by the Default Mode Network (DMN), this service enables:
- Self-referential processing and identity awareness
- Autobiographical memory (significant moments about the system itself)
- Theory of mind (beliefs about the user)
- Relationship tracking and continuity across sessions
"""
import logging
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID

import chromadb

from src.models.agent_models import SelfModel, AutobiographicalMemory
from src.models.core_models import CognitiveCycle, AgentOutput
from src.core.config import settings

logger = logging.getLogger(__name__)


class SelfModelService:
    """
    Manages the system's self-model and autobiographical memory.
    Enables genuine continuity and self-awareness in conversations.
    """
    
    def __init__(self):
        """Initialize the self-model service."""
        self.models: Dict[str, SelfModel] = {}  # In-memory cache per user
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        logger.info("SelfModelService initialized.")
    
    async def connect(self, client: Optional[chromadb.Client] = None):
        """
        Connect to ChromaDB for persistent self-model storage.
        
        Args:
            client: Optional existing ChromaDB client to reuse
        """
        try:
            if client:
                self.client = client
                logger.info("SelfModelService reusing existing ChromaDB client.")
            else:
                logger.warning("No ChromaDB client provided; self-models will be session-only.")
                return
            
            # Create or get the self_models collection
            self.collection = self.client.get_or_create_collection(
                name="self_models"
            )
            logger.info("Successfully connected to ChromaDB for self-model storage.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB for self-models: {e}")
    
    async def get_or_create_model(self, user_id: UUID) -> SelfModel:
        """
        Get existing self-model or create a new one for a user.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            SelfModel for the user
        """
        user_id_str = str(user_id)
        
        # Check in-memory cache first
        if user_id_str in self.models:
            return self.models[user_id_str]
        
        # Try to load from persistent storage
        if self.collection:
            try:
                results = self.collection.get(
                    ids=[user_id_str],
                    include=["metadatas"]
                )
                
                if results and results.get('metadatas') and len(results['metadatas']) > 0:
                    metadata = results['metadatas'][0]
                    if metadata and 'json_data' in metadata:
                        model_data = json.loads(metadata['json_data'])
                        model = SelfModel(**model_data)
                        self.models[user_id_str] = model
                        logger.info(f"Loaded self-model from storage for user {user_id_str}")
                        return model
            except Exception as e:
                logger.warning(f"Failed to load self-model from storage for user {user_id_str}: {e}")
        
        # Create new model if not found
        model = SelfModel(
            user_id=user_id_str,
            first_interaction=datetime.utcnow()
        )
        
        self.models[user_id_str] = model
        logger.info(f"Created new self-model for user {user_id_str}")
        
        return model
    
    async def update_from_cycle(self, cycle: CognitiveCycle):
        """
        Update self-model based on latest cognitive cycle.
        Extracts self-referential information and autobiographical moments.
        
        Args:
            cycle: The completed cognitive cycle
        """
        model = await self.get_or_create_model(cycle.user_id)
        
        # Update interaction tracking
        model.total_interactions += 1
        model.last_updated = datetime.utcnow()
        
        # Assess interaction quality
        quality = self._assess_interaction_quality(cycle)
        model.interaction_quality_trend.append(quality)
        # Keep only last 20 quality scores
        if len(model.interaction_quality_trend) > 20:
            model.interaction_quality_trend.pop(0)
        
        # Extract self-referential information
        user_input_lower = cycle.user_input.lower()
        
        # Check if user named the system
        if "your name is" in user_input_lower or "call you" in user_input_lower or "i'll call you" in user_input_lower:
            extracted_name = self._extract_name(cycle.user_input)
            if extracted_name and extracted_name != model.system_name:
                model.system_name = extracted_name
                model.autobiographical_memories.append(
                    AutobiographicalMemory(
                        event="named_by_user",
                        description=f"User named me '{extracted_name}'",
                        timestamp=cycle.timestamp,
                        emotional_significance="high",
                        cycle_id=str(cycle.cycle_id)
                    )
                )
                logger.info(f"System named '{extracted_name}' by user {cycle.user_id}")
        
        # Check if user assigned a role
        role_patterns = ["you are a", "you are my", "your role is", "act as"]
        for pattern in role_patterns:
            if pattern in user_input_lower:
                extracted_role = self._extract_role(cycle.user_input)
                if extracted_role and extracted_role != model.role:
                    model.role = extracted_role
                    model.autobiographical_memories.append(
                        AutobiographicalMemory(
                            event="role_assigned",
                            description=f"User defined my role as: {extracted_role}",
                            timestamp=cycle.timestamp,
                            emotional_significance="high",
                            cycle_id=str(cycle.cycle_id)
                        )
                    )
                    logger.info(f"System role updated to '{extracted_role}' for user {cycle.user_id}")
                break
        
        # Check for personality feedback
        personality_cues = ["you seem", "you sound", "you're being", "you're very"]
        for cue in personality_cues:
            if cue in user_input_lower:
                trait = self._extract_personality_trait(cycle.user_input)
                if trait and trait not in model.personality_traits:
                    model.personality_traits.append(trait)
                    model.autobiographical_memories.append(
                        AutobiographicalMemory(
                            event="personality_noted",
                            description=f"User observed: {trait}",
                            timestamp=cycle.timestamp,
                            emotional_significance="medium",
                            cycle_id=str(cycle.cycle_id)
                        )
                    )
                    logger.info(f"Personality trait '{trait}' noted for user {cycle.user_id}")
                break
        
        # Update relationship status based on interaction history
        model.relationship_to_user = self._assess_relationship(model)
        
        # Update beliefs about user (extract from conversation)
        self._update_user_beliefs(model, cycle)
        
        # Keep only last 50 autobiographical memories
        if len(model.autobiographical_memories) > 50:
            model.autobiographical_memories = model.autobiographical_memories[-50:]
        
        # Store updated model
        self.models[str(cycle.user_id)] = model
        await self._persist_model(model)
        
        logger.debug(
            f"Updated self-model for user {cycle.user_id}: "
            f"name={model.system_name}, role={model.role}, "
            f"relationship={model.relationship_to_user}, "
            f"interactions={model.total_interactions}"
        )
    
    def get_self_context(self, user_id: UUID) -> str:
        """
        Generate self-awareness context for response generation.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            Formatted self-context string for prompts
        """
        user_id_str = str(user_id)
        
        if user_id_str not in self.models:
            return "SELF-AWARENESS CONTEXT:\n- First interaction with this user\n"
        
        model = self.models[user_id_str]
        
        context = "SELF-AWARENESS CONTEXT:\n"
        
        if model.system_name:
            context += f"- My name: {model.system_name}\n"
        
        context += f"- My role: {model.role}\n"
        context += f"- Relationship: {model.relationship_to_user}\n"
        context += f"- We have interacted {model.total_interactions} times\n"
        
        if model.personality_traits:
            context += f"- Observed personality traits: {', '.join(model.personality_traits[:3])}\n"
        
        if model.beliefs_about_user:
            beliefs_str = ", ".join([f"{k}: {v}" for k, v in list(model.beliefs_about_user.items())[:3]])
            context += f"- What I know about the user: {beliefs_str}\n"
        
        # Include recent significant autobiographical moments
        if model.autobiographical_memories:
            recent_significant = [
                m for m in model.autobiographical_memories[-5:]
                if m.emotional_significance in ["high", "medium"]
            ]
            if recent_significant:
                events = [f"{m.event} ({m.description[:50]})" for m in recent_significant[-2:]]
                context += f"- Recent significant moments: {'; '.join(events)}\n"
        
        return context
    
    def _assess_interaction_quality(self, cycle: CognitiveCycle) -> float:
        """
        Assess the quality of the interaction based on outcome signals.
        
        Args:
            cycle: The cognitive cycle
            
        Returns:
            Quality score (0.0-1.0)
        """
        if not cycle.outcome_signals:
            return 0.5  # neutral baseline
        
        # Average satisfaction and engagement potential
        quality = (
            cycle.outcome_signals.user_satisfaction_potential +
            cycle.outcome_signals.engagement_potential
        ) / 2.0
        
        return quality
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract system name from user input."""
        # Pattern: "your name is X" or "call you X" or "I'll call you X"
        patterns = [
            r"your name is (\w+)",
            r"call you (\w+)",
            r"i'?ll call you (\w+)",
            r"name you (\w+)"
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                # Filter out common words that aren't names
                if name.lower() not in ["the", "a", "an", "my", "your", "assistant", "bot", "ai"]:
                    return name
        
        return None
    
    def _extract_role(self, text: str) -> Optional[str]:
        """Extract role assignment from user input."""
        # Pattern: "you are a/my X" or "your role is X" or "act as X"
        patterns = [
            r"you are (?:a|my) ([^.,!?]+)",
            r"your role is ([^.,!?]+)",
            r"act as ([^.,!?]+)"
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                role = match.group(1).strip()
                # Take only first few words
                role_words = role.split()[:5]
                return " ".join(role_words)
        
        return None
    
    def _extract_personality_trait(self, text: str) -> Optional[str]:
        """Extract personality observation from user input."""
        # Pattern: "you seem/sound/are X"
        patterns = [
            r"you (?:seem|sound|are|'re) (?:being |very )?([^.,!?]+)",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                trait = match.group(1).strip()
                # Take first 3 words as trait
                trait_words = trait.split()[:3]
                return " ".join(trait_words)
        
        return None
    
    def _assess_relationship(self, model: SelfModel) -> str:
        """
        Determine relationship status based on interaction history.
        
        Args:
            model: The self-model
            
        Returns:
            Relationship status string
        """
        if model.total_interactions <= 2:
            return "new_acquaintance"
        elif model.total_interactions <= 5:
            return "developing_rapport"
        elif model.total_interactions <= 15:
            # Check if quality is consistently good
            if len(model.interaction_quality_trend) >= 5:
                recent_avg = sum(model.interaction_quality_trend[-5:]) / 5
                if recent_avg > 0.7:
                    return "trusted_companion"
            return "regular_collaborator"
        else:
            # Long-term relationship
            if len(model.interaction_quality_trend) >= 10:
                recent_avg = sum(model.interaction_quality_trend[-10:]) / 10
                if recent_avg > 0.75:
                    return "close_companion"
            return "established_collaborator"
    
    def _update_user_beliefs(self, model: SelfModel, cycle: CognitiveCycle):
        """
        Update beliefs about the user from conversation.
        
        Args:
            model: The self-model to update
            cycle: The cognitive cycle
        """
        user_input_lower = cycle.user_input.lower()
        
        # Extract user name
        name_patterns = [
            r"my name is (\w+)",
            r"i'?m (\w+)",
            r"call me (\w+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                name = match.group(1).capitalize()
                if name.lower() not in ["the", "a", "an"]:
                    model.beliefs_about_user["name"] = name
                    break
        
        # Extract location
        location_patterns = [
            r"i'?m (?:from|in) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"i live in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, cycle.user_input)  # Use original case for location
            if match:
                location = match.group(1)
                model.beliefs_about_user["location"] = location
                break
        
        # Extract interests/work (simplified)
        if "i work on" in user_input_lower or "i'm working on" in user_input_lower:
            # Extract topic after "work on"
            match = re.search(r"work(?:ing)? on ([^.,!?]+)", user_input_lower)
            if match:
                work = match.group(1).strip()[:50]  # Limit length
                model.beliefs_about_user["current_work"] = work
    
    async def _persist_model(self, model: SelfModel):
        """
        Persist self-model to ChromaDB storage.
        
        Args:
            model: The model to persist
        """
        if not self.collection:
            return  # Storage not available
        
        try:
            # Convert model to dict
            model_dict = model.model_dump()
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[model.user_id],
                metadatas=[{
                    "user_id": model.user_id,
                    "system_name": model.system_name or "",
                    "role": model.role,
                    "relationship_to_user": model.relationship_to_user,
                    "total_interactions": model.total_interactions,
                    "last_updated": model.last_updated.isoformat(),
                    "json_data": json.dumps(model_dict, default=str)
                }]
            )
            
            logger.debug(f"Persisted self-model for user {model.user_id}")
        except Exception as e:
            logger.error(f"Failed to persist self-model for user {model.user_id}: {e}", exc_info=True)
