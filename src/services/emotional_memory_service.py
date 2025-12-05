"""
Emotional Memory Service - Manages persistent emotional profiles and relational memory.

This service enables human-like relational awareness by tracking:
- User identity and relationship progression
- Emotional history and trends
- Trust levels and comfort
- Shared experiences and positive moments
"""
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID

import chromadb

from src.models.agent_models import EmotionalProfile
from src.models.memory_models import ConversationSummary
from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.services.metrics_service import MetricsService, MetricType
from src.core.config import settings
from src.core.exceptions import APIException

logger = logging.getLogger(__name__)


class EmotionalMemoryService:
    """
    Manages persistent emotional profiles and relational memory.
    Enables the emotional agent to maintain human-like relational awareness.
    """
    
    def __init__(self, memory_service: MemoryService, llm_service: LLMIntegrationService, metrics_service: Optional[MetricsService] = None):
        """
        Initialize the emotional memory service.
        
        Args:
            memory_service: Main memory service for accessing conversation summaries
            llm_service: LLM service for generating insights
            metrics_service: For recording emotional learning metrics
        """
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.metrics_service = metrics_service
        self.profiles: Dict[str, EmotionalProfile] = {}  # In-memory cache
        self.client: Optional[chromadb.Client] = None
        self.profiles_collection: Optional[chromadb.Collection] = None
        logger.info("EmotionalMemoryService initialized.")
    
    async def connect(self, client: Optional[chromadb.Client] = None):
        """
        Initialize ChromaDB connection for persistent emotional profile storage.
        
        Args:
            client: Optional existing ChromaDB client to reuse.
        """
        try:
            if client:
                self.client = client
                logger.info("EmotionalMemoryService reusing existing ChromaDB client.")
            else:
                # Use the same client from memory_service if available
                if hasattr(self.memory_service, 'client') and self.memory_service.client:
                    self.client = self.memory_service.client
                    logger.info("EmotionalMemoryService using MemoryService's ChromaDB client.")
                else:
                    logger.warning("No ChromaDB client available; emotional profiles will be session-only.")
                    return
            
            # Create or get the emotional_profiles collection
            self.profiles_collection = self.client.get_or_create_collection(
                name="emotional_profiles"
            )
            logger.info("Successfully connected to ChromaDB for emotional profile storage.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB for emotional profiles: {e}")
            # Don't raise - allow service to work in-memory only
    
    async def get_or_create_profile(self, user_id: UUID) -> EmotionalProfile:
        """
        Get existing emotional profile or create a new one for a user.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            EmotionalProfile for the user
        """
        user_id_str = str(user_id)
        
        # Check in-memory cache first
        if user_id_str in self.profiles:
            return self.profiles[user_id_str]
        
        # Try to load from persistent storage
        if self.profiles_collection:
            try:
                results = self.profiles_collection.get(
                    ids=[user_id_str],
                    include=["metadatas"]
                )
                
                if results and results.get('metadatas') and len(results['metadatas']) > 0:
                    metadata = results['metadatas'][0]
                    if metadata and 'json_data' in metadata:
                        profile_data = json.loads(metadata['json_data'])
                        
                        # Convert stored lists back to sets where needed
                        if 'shared_topics' in profile_data and isinstance(profile_data['shared_topics'], list):
                            profile_data['shared_topics'] = set(profile_data['shared_topics'])
                        
                        profile = EmotionalProfile(**profile_data)
                        self.profiles[user_id_str] = profile
                        logger.info(f"Loaded emotional profile from storage for user {user_id_str}")
                        return profile
            except Exception as e:
                logger.warning(f"Failed to load emotional profile from storage for user {user_id_str}: {e}")
        
        # Create new profile if not found
        profile = EmotionalProfile(
            user_id=user_id_str,
            first_interaction_at=datetime.utcnow(),
            last_interaction_at=datetime.utcnow()
        )
        
        self.profiles[user_id_str] = profile
        logger.info(f"Created new emotional profile for user {user_id_str}")
        
        return profile
    
    async def update_profile(
        self,
        user_id: UUID,
        current_sentiment: str,
        user_input: str,
        conversation_summary: Optional[ConversationSummary] = None
    ) -> EmotionalProfile:
        """
        Update emotional profile based on current interaction.
        
        Args:
            user_id: UUID of the user
            current_sentiment: Current detected sentiment (positive/negative/neutral/mixed)
            user_input: The user's input text
            conversation_summary: Optional conversation summary for context
            
        Returns:
            Updated EmotionalProfile
        """
        profile = await self.get_or_create_profile(user_id)
        
        # Update basic stats
        profile.interaction_count += 1
        profile.last_interaction_at = datetime.utcnow()
        profile.last_emotion_detected = current_sentiment
        
        # Track sentiment history (keep last 5)
        profile.recent_sentiments.append(current_sentiment)
        if len(profile.recent_sentiments) > 5:
            profile.recent_sentiments.pop(0)
        
        # Update emotional trend
        profile.emotional_trend = self._calculate_trend(profile.recent_sentiments)
        
        # Extract user name from conversation summary if available
        if conversation_summary:
            # Look for user name in entities
            for entity in conversation_summary.entities:
                entity_lower = entity.lower()
                # Skip common words and focus on proper names
                if entity_lower not in ["i", "me", "my", "you", "and", "the", "a", "an"]:
                    if len(entity) >= 2 and entity[0].isupper():
                        profile.user_name = entity
                        logger.info(f"Extracted user name from summary: {entity}")
                        break
            
            # Extract topics
            for topic in conversation_summary.key_topics:
                profile.shared_topics.add(topic)
        
        # Update relationship type based on interaction count and sentiment
        profile.relationship_type = self._assess_relationship(profile)
        
        # Update trust level (grows with positive interactions)
        if current_sentiment == "positive":
            profile.trust_level = min(1.0, profile.trust_level + 0.05)
        elif current_sentiment == "negative":
            profile.trust_level = max(0.0, profile.trust_level - 0.02)
        
        # Update comfort level (based on message length, openness cues)
        profile.comfort_level = self._assess_comfort(user_input, profile)
        
        # Detect positive moments
        if current_sentiment == "positive" and profile.trust_level > 0.6:
            # Check for memorable positive cues
            positive_cues = ["thank", "appreciate", "love", "amazing", "wonderful", "great", "awesome"]
            if any(cue in user_input.lower() for cue in positive_cues):
                moment = f"{datetime.utcnow().strftime('%Y-%m-%d')}: {user_input[:100]}"
                profile.positive_moments.append(moment)
                # Keep only last 10 positive moments
                if len(profile.positive_moments) > 10:
                    profile.positive_moments.pop(0)
        
        # Detect concerns/frustrations
        if current_sentiment == "negative":
            concern_cues = ["frustrat", "confus", "don't understand", "doesn't work", "problem", "issue"]
            if any(cue in user_input.lower() for cue in concern_cues):
                concern = f"{datetime.utcnow().strftime('%Y-%m-%d')}: {user_input[:100]}"
                profile.concerns_raised.append(concern)
                # Keep only last 10 concerns
                if len(profile.concerns_raised) > 10:
                    profile.concerns_raised.pop(0)
        
        # Store updated profile in cache and persist
        self.profiles[str(user_id)] = profile
        await self._persist_profile(profile)
        
        # Record emotional learning metrics
        if self.metrics_service:
            await self.metrics_service.record_metric(
                MetricType.LEARNING_EVENT,
                {
                    "learning_type": "emotional_memory",
                    "sentiment": current_sentiment,
                    "trust_level": profile.trust_level,
                    "comfort_level": profile.comfort_level,
                    "relationship_type": profile.relationship_type,
                    "emotional_trend": profile.emotional_trend,
                    "interaction_count": profile.interaction_count,
                    "positive_moments_count": len(profile.positive_moments),
                    "concerns_count": len(profile.concerns_raised)
                },
                user_id=str(user_id)
            )
        
        logger.info(
            f"Updated emotional profile for user {user_id}: "
            f"relationship={profile.relationship_type}, trust={profile.trust_level:.2f}, "
            f"trend={profile.emotional_trend}"
        )
        
        return profile
    
    def _calculate_trend(self, recent_sentiments: List[str]) -> str:
        """
        Analyze emotional trajectory from recent sentiments.
        
        Args:
            recent_sentiments: List of recent sentiment strings
            
        Returns:
            Trend description: improving, declining, stable, or volatile
        """
        if len(recent_sentiments) < 2:
            return "stable"
        
        positive_count = recent_sentiments.count("positive")
        negative_count = recent_sentiments.count("negative")
        
        # Look at first half vs second half to detect trends
        mid = len(recent_sentiments) // 2
        first_half = recent_sentiments[:mid]
        second_half = recent_sentiments[mid:]
        
        first_positive = first_half.count("positive")
        second_positive = second_half.count("positive")
        
        if second_positive > first_positive + 1:
            return "improving"
        elif second_positive < first_positive - 1:
            return "declining"
        elif negative_count > positive_count + 1:
            return "volatile"
        else:
            return "stable"
    
    def _assess_relationship(self, profile: EmotionalProfile) -> str:
        """
        Determine relationship type based on interaction history.
        
        Args:
            profile: The emotional profile
            
        Returns:
            Relationship type: new_user, acquaintance, friend, or collaborator
        """
        if profile.interaction_count <= 2:
            return "new_user"
        elif profile.interaction_count <= 5:
            return "acquaintance"
        elif profile.trust_level >= 0.7 and profile.interaction_count > 5:
            return "friend"
        elif profile.interaction_count > 10:
            return "collaborator"
        return "acquaintance"
    
    def _assess_comfort(self, user_input: str, profile: EmotionalProfile) -> float:
        """
        Estimate user's comfort level based on interaction patterns.
        
        Args:
            user_input: Current user input
            profile: The emotional profile
            
        Returns:
            Comfort level (0.0-1.0)
        """
        comfort = profile.comfort_level
        
        # Longer messages suggest comfort
        if len(user_input) > 100:
            comfort += 0.05
        
        # Personal sharing cues
        user_input_lower = user_input.lower()
        personal_cues = ["my name", "i feel", "i'm", "i think", "honestly", "to be honest"]
        if any(cue in user_input_lower for cue in personal_cues):
            comfort += 0.1
        
        return min(1.0, comfort)
    
    def get_recognition_status(self, profile: EmotionalProfile) -> str:
        """
        Determine how well we recognize/know this user.
        
        Args:
            profile: The emotional profile
            
        Returns:
            Recognition status: new_user, returning_user, or recognized_friend
        """
        if profile.interaction_count == 1:
            return "new_user"
        elif profile.user_name and profile.interaction_count > 3:
            return "recognized_friend"
        else:
            return "returning_user"
    
    def summarize_emotional_history(self, profile: EmotionalProfile) -> str:
        """
        Generate a brief summary of emotional history for context.
        
        Args:
            profile: The emotional profile
            
        Returns:
            Brief summary string
        """
        if profile.interaction_count == 1:
            return "First interaction"
        
        summary = f"{profile.interaction_count} interactions, trend: {profile.emotional_trend}"
        
        if profile.user_name:
            summary += f", user: {profile.user_name}"
        
        if profile.relationship_type != "new_user":
            summary += f", relationship: {profile.relationship_type}"
        
        return summary
    
    async def _persist_profile(self, profile: EmotionalProfile):
        """
        Persist emotional profile to ChromaDB storage.
        
        Args:
            profile: The profile to persist
        """
        if not self.profiles_collection:
            return  # Storage not available, profile stays in-memory only
        
        try:
            # Convert profile to dict and handle sets for JSON serialization
            profile_dict = profile.model_dump()
            
            # Convert sets to lists for JSON
            if 'shared_topics' in profile_dict and isinstance(profile_dict['shared_topics'], set):
                profile_dict['shared_topics'] = list(profile_dict['shared_topics'])
            
            # Store in ChromaDB (provide a minimal document to satisfy API requirements)
            document_text = f"Emotional profile for {profile.user_name or profile.user_id}: {profile.relationship_type} relationship"
            self.profiles_collection.upsert(
                ids=[profile.user_id],
                documents=[document_text],
                metadatas=[{
                    "user_id": profile.user_id,
                    "user_name": profile.user_name or "",
                    "relationship_type": profile.relationship_type,
                    "interaction_count": profile.interaction_count,
                    "trust_level": profile.trust_level,
                    "last_updated": profile.last_interaction_at.isoformat(),
                    "json_data": json.dumps(profile_dict, default=str)
                }]
            )
            
            logger.debug(f"Persisted emotional profile for user {profile.user_id}")
        except Exception as e:
            logger.error(f"Failed to persist emotional profile for user {profile.user_id}: {e}", exc_info=True)
            # Don't raise - allow conversation to continue even if persistence fails
