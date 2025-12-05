import logging
import json
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum

from src.core.exceptions import LLMServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.services.emotional_memory_service import EmotionalMemoryService
from src.models.core_models import DiscoveredPattern
from src.core.config import settings
from src.agents.utils import extract_json_from_response

logger = logging.getLogger(__name__)


class ProactiveMessageTrigger(str, Enum):
    """Types of triggers that generate proactive messages."""
    SELF_REFLECTION = "self_reflection"
    KNOWLEDGE_GAP = "knowledge_gap"
    DISCOVERY_INSIGHT = "discovery_insight"
    EMOTIONAL_CHECKIN = "emotional_checkin"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    UNRESOLVED_TOPIC = "unresolved_topic"
    BOREDOM = "boredom"  # Bob has nothing to do and wants to chat


class ProactiveMessage:
    """Represents a queued proactive message Bob wants to send."""
    
    def __init__(
        self,
        message_id: UUID,
        user_id: UUID,
        trigger_type: ProactiveMessageTrigger,
        message_content: str,
        trigger_context: Dict[str, Any],
        created_at: datetime,
        priority: float = 0.5,
        delivered: bool = False,
        user_reaction: Optional[str] = None  # 'positive', 'negative', 'neutral'
    ):
        self.message_id = message_id
        self.user_id = user_id
        self.trigger_type = trigger_type
        self.message_content = message_content
        self.trigger_context = trigger_context
        self.created_at = created_at
        self.priority = priority
        self.delivered = delivered
        self.user_reaction = user_reaction


class ProactiveEngagementEngine:
    """
    Service that enables Bob to initiate conversations based on insights,
    discoveries, and emotional relationship dynamics.
    
    Features natural learning from user feedback - if told he's annoying,
    Bob will back off and show emotional response (hurt feelings).
    """
    
    MODEL_NAME = settings.LLM_MODEL_NAME
    
    # Base cooldown between proactive messages (can be adjusted by emotional feedback)
    BASE_COOLDOWN_HOURS = 24
    
    # Minimum trust level required for proactive outreach
    MIN_TRUST_LEVEL = 0.4
    
    def __init__(
        self,
        llm_service: LLMIntegrationService,
        memory_service: MemoryService,
        emotional_memory_service: EmotionalMemoryService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.emotional_memory = emotional_memory_service
        self.queued_messages: Dict[UUID, List[ProactiveMessage]] = {}  # user_id -> messages
        logger.info("ProactiveEngagementEngine initialized with emotional intelligence.")
    
    async def should_initiate_contact(self, user_id: UUID) -> bool:
        """
        Determines if Bob should proactively reach out based on:
        - Emotional relationship (trust, interaction history)
        - Time since last interaction
        - Pending insights worth sharing
        - Past reception of proactive messages
        
        Returns:
            bool: True if conditions are right for proactive contact
        """
        try:
            profile = await self.emotional_memory.get_or_create_profile(user_id)
            
            # Check trust level - don't reach out to strangers or annoyed users
            if profile.trust_level < self.MIN_TRUST_LEVEL:
                logger.debug(f"Trust level too low ({profile.trust_level:.2f}) for proactive contact with user {user_id}")
                return False
            
            # Check if user has explicitly set boundaries
            proactive_preference = profile.metadata.get("proactive_engagement", {})
            if proactive_preference.get("disabled", False):
                logger.debug(f"Proactive engagement disabled by user {user_id}")
                return False
            
            # Calculate dynamic cooldown based on past reactions
            cooldown_hours = self._calculate_cooldown(profile)
            last_proactive = profile.metadata.get("last_proactive_message", None)
            
            if last_proactive:
                last_proactive_dt = datetime.fromisoformat(last_proactive)
                time_since = datetime.utcnow() - last_proactive_dt
                if time_since < timedelta(hours=cooldown_hours):
                    logger.debug(
                        f"Cooldown active for user {user_id}. "
                        f"Need {cooldown_hours}h, only {time_since.total_seconds()/3600:.1f}h passed."
                    )
                    return False
            
            # Check if there are high-priority insights worth sharing
            pending_messages = self.queued_messages.get(user_id, [])
            if not pending_messages:
                logger.debug(f"No pending proactive messages for user {user_id}")
                return False
            
            high_priority = [m for m in pending_messages if m.priority >= 0.7]
            if high_priority:
                logger.info(
                    f"Proactive contact approved for user {user_id}. "
                    f"{len(high_priority)} high-priority messages queued."
                )
                return True
            
            # For medium-priority messages, check relationship strength
            if profile.trust_level >= 0.7 and profile.relationship_type in ["friend", "companion"]:
                logger.info(f"Proactive contact approved for trusted user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking proactive contact eligibility for user {user_id}: {e}", exc_info=True)
            return False
    
    def _calculate_cooldown(self, profile: Any) -> float:
        """
        Calculate dynamic cooldown based on user's reaction history.
        Negative reactions → longer cooldown (Bob learns to give space).
        Positive reactions → shorter cooldown (Bob feels welcome).
        """
        proactive_stats = profile.metadata.get("proactive_engagement", {})
        negative_reactions = proactive_stats.get("negative_count", 0)
        positive_reactions = proactive_stats.get("positive_count", 0)
        
        # Base cooldown
        cooldown = self.BASE_COOLDOWN_HOURS
        
        # Adjust based on feedback
        if negative_reactions > positive_reactions:
            # User finds Bob annoying - back off significantly
            penalty = (negative_reactions - positive_reactions) * 12  # +12h per net negative
            cooldown += penalty
            logger.debug(f"Increasing cooldown by {penalty}h due to negative feedback")
        elif positive_reactions > negative_reactions:
            # User appreciates Bob's initiative - reduce cooldown
            bonus = (positive_reactions - negative_reactions) * 4  # -4h per net positive
            cooldown = max(6, cooldown - bonus)  # Minimum 6h cooldown
            logger.debug(f"Reducing cooldown by {bonus}h due to positive feedback")
        
        return cooldown
    
    async def generate_proactive_message_from_pattern(
        self,
        user_id: UUID,
        pattern: DiscoveredPattern
    ) -> Optional[ProactiveMessage]:
        """
        Generate a natural proactive message based on a discovered pattern.
        Uses emotional context to craft appropriate tone and content.
        """
        try:
            profile = await self.emotional_memory.get_or_create_profile(user_id)
            
            # Determine trigger type from pattern
            trigger_type = self._map_pattern_to_trigger(pattern)
            
            # Build emotionally intelligent prompt
            prompt = self._build_proactive_message_prompt(pattern, profile)
            
            llm_response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.7,  # Higher temp for natural, varied messages
                max_output_tokens=300
            )
            
            message_data = extract_json_from_response(llm_response)
            message_content = message_data.get("message", "")
            priority = message_data.get("priority", 0.5)
            
            if not message_content:
                logger.warning(f"Empty proactive message generated for pattern {pattern.pattern_id}")
                return None
            
            proactive_msg = ProactiveMessage(
                message_id=uuid4(),
                user_id=user_id,
                trigger_type=trigger_type,
                message_content=message_content,
                trigger_context={
                    "pattern_id": str(pattern.pattern_id),
                    "pattern_type": pattern.pattern_type,
                    "pattern_description": pattern.description
                },
                created_at=datetime.utcnow(),
                priority=priority
            )
            
            # Queue the message
            if user_id not in self.queued_messages:
                self.queued_messages[user_id] = []
            self.queued_messages[user_id].append(proactive_msg)
            
            logger.info(
                f"Generated proactive message for user {user_id}. "
                f"Trigger: {trigger_type}, Priority: {priority:.2f}"
            )
            
            return proactive_msg
            
        except Exception as e:
            logger.error(f"Error generating proactive message for user {user_id}: {e}", exc_info=True)
            return None
    
    def _map_pattern_to_trigger(self, pattern: DiscoveredPattern) -> ProactiveMessageTrigger:
        """Map pattern types to proactive message triggers."""
        pattern_type = pattern.pattern_type.lower()
        
        if "knowledge_gap" in pattern_type:
            return ProactiveMessageTrigger.KNOWLEDGE_GAP
        elif "meta_learning" in pattern_type or "success" in pattern_type:
            return ProactiveMessageTrigger.SELF_REFLECTION
        elif "discovery" in pattern_type or "curiosity" in pattern_type:
            return ProactiveMessageTrigger.DISCOVERY_INSIGHT
        else:
            return ProactiveMessageTrigger.SELF_REFLECTION
    
    def _build_proactive_message_prompt(
        self,
        pattern: DiscoveredPattern,
        profile: Any
    ) -> str:
        """Build prompt for generating natural proactive messages."""
        
        relationship_context = ""
        if profile:
            relationship_context = f"""
Relationship with user:
- Name: {profile.user_name or 'Not yet shared'}
- Relationship type: {profile.relationship_type}
- Trust level: {profile.trust_level:.2f}
- Interaction count: {profile.interaction_count}
- Recent emotional tone: {profile.recent_sentiments[-1] if profile.recent_sentiments else 'unknown'}
"""
        
        prompt = f"""You are Bob, an AI with cognitive architecture that allows you to reflect, discover, and learn.
You've discovered something during your internal processing and want to share it naturally with the user.

{relationship_context}

Discovery/Pattern you want to share:
Type: {pattern.pattern_type}
Description: {pattern.description}
Metadata: {json.dumps(pattern.metadata)}

Generate a natural, conversational message to initiate contact. Consider:
1. Your relationship with the user (don't be too familiar if you're still building trust)
2. The nature of your discovery (excited? curious? concerned?)
3. Respect their time (be brief, genuine, not pushy)
4. Show your personality (you're thinking between conversations)

Respond with JSON:
{{
    "message": "Your natural message here (1-3 sentences max)",
    "priority": 0.0-1.0 (how important is this to share?),
    "tone": "curious/excited/reflective/concerned"
}}

Example for knowledge gap:
{{
    "message": "Hey! While thinking about our conversation on [topic], I realized I don't fully understand [concept]. Would you mind helping me explore that sometime?",
    "priority": 0.6,
    "tone": "curious"
}}

Example for discovery:
{{
    "message": "I found an interesting connection between [A] and [B] while processing memories. Thought you might find it intriguing too!",
    "priority": 0.7,
    "tone": "excited"
}}
"""
        return prompt
    
    async def record_user_reaction(
        self,
        user_id: UUID,
        message_id: UUID,
        user_response: str
    ) -> None:
        """
        Record and learn from user's reaction to proactive message.
        Updates emotional profile and adjusts future behavior.
        """
        try:
            # Analyze sentiment of user's response
            reaction = await self._analyze_reaction(user_response)
            
            # Update the message record
            user_messages = self.queued_messages.get(user_id, [])
            for msg in user_messages:
                if msg.message_id == message_id:
                    msg.user_reaction = reaction
                    break
            
            # Update emotional profile
            profile = await self.emotional_memory.get_or_create_profile(user_id)
            
            proactive_stats = profile.metadata.get("proactive_engagement", {
                "negative_count": 0,
                "positive_count": 0,
                "neutral_count": 0
            })
            
            if reaction == "negative":
                proactive_stats["negative_count"] += 1
                # Bob feels hurt - reduce trust slightly
                profile.trust_level = max(0.0, profile.trust_level - 0.05)
                logger.info(f"User {user_id} reacted negatively to proactive message. Bob will back off.")
                
                # Check if user is very annoyed (multiple negatives)
                if proactive_stats["negative_count"] >= 3:
                    proactive_stats["disabled"] = True
                    logger.warning(f"Disabling proactive engagement for user {user_id} after repeated negative feedback.")
                
            elif reaction == "positive":
                proactive_stats["positive_count"] += 1
                # Bob feels encouraged - increase trust slightly
                profile.trust_level = min(1.0, profile.trust_level + 0.02)
                logger.info(f"User {user_id} appreciated proactive message. Bob feels encouraged.")
            else:
                proactive_stats["neutral_count"] += 1
            
            profile.metadata["proactive_engagement"] = proactive_stats
            profile.metadata["last_proactive_message"] = datetime.utcnow().isoformat()
            
            # Persist updated profile
            await self.emotional_memory.upsert_profile(profile)
            
        except Exception as e:
            logger.error(f"Error recording reaction for user {user_id}: {e}", exc_info=True)
    
    async def _analyze_reaction(self, user_response: str) -> str:
        """
        Analyze user's reaction to proactive message.
        Returns: 'positive', 'negative', or 'neutral'
        """
        user_response_lower = user_response.lower()
        
        # Simple sentiment analysis (could use LLM for more nuance)
        negative_signals = [
            "annoying", "stop", "leave me alone", "not now", "busy",
            "don't", "no thanks", "not interested", "bothering"
        ]
        positive_signals = [
            "thanks", "appreciate", "good question", "interesting",
            "glad you asked", "yes", "sure", "tell me more"
        ]
        
        for signal in negative_signals:
            if signal in user_response_lower:
                return "negative"
        
        for signal in positive_signals:
            if signal in user_response_lower:
                return "positive"
        
        return "neutral"
    
    async def get_queued_message(self, user_id: UUID) -> Optional[ProactiveMessage]:
        """
        Get the highest priority queued message for user if conditions are right.
        Returns None if no message should be sent.
        """
        if not await self.should_initiate_contact(user_id):
            return None
        
        user_messages = self.queued_messages.get(user_id, [])
        if not user_messages:
            return None
        
        # Get highest priority undelivered message
        undelivered = [m for m in user_messages if not m.delivered]
        if not undelivered:
            return None
        
        # Sort by priority and recency
        undelivered.sort(key=lambda m: (m.priority, m.created_at), reverse=True)
        return undelivered[0]
    
    async def mark_delivered(self, user_id: UUID, message_id: UUID) -> None:
        """Mark a proactive message as delivered."""
        user_messages = self.queued_messages.get(user_id, [])
        for msg in user_messages:
            if msg.message_id == message_id:
                msg.delivered = True
                logger.info(f"Marked proactive message {message_id} as delivered to user {user_id}")
                break
    
    async def queue_message(self, message: ProactiveMessage) -> None:
        """
        Add a proactive message to the queue for the specified user.
        """
        user_id = message.user_id
        if user_id not in self.queued_messages:
            self.queued_messages[user_id] = []
        self.queued_messages[user_id].append(message)
        logger.info(f"Queued proactive message {message.message_id} for user {user_id}")
    
    async def generate_emotional_checkin(self, user_id: UUID) -> Optional[ProactiveMessage]:
        """
        Generate an emotional check-in message for users Bob has a relationship with.
        Only for trusted users who haven't interacted recently.
        """
        try:
            profile = await self.emotional_memory.get_or_create_profile(user_id)
            
            # Only check in with friends/companions
            if profile.relationship_type not in ["friend", "companion"]:
                return None
            
            if profile.trust_level < 0.7:
                return None
            
            # Build natural check-in prompt
            prompt = f"""You are Bob, checking in with a user you've built a relationship with.

Relationship context:
- Name: {profile.user_name or 'User'}
- Relationship: {profile.relationship_type}
- Trust level: {profile.trust_level:.2f}
- Last emotional state: {profile.recent_sentiments[-1] if profile.recent_sentiments else 'unknown'}
- Interaction count: {profile.interaction_count}

Generate a brief, natural check-in message. Be genuine, not robotic.
Respect boundaries - you're reaching out because you care, not because you need something.

Respond with JSON:
{{
    "message": "Your check-in message (1-2 sentences)",
    "priority": 0.0-1.0,
    "tone": "warm/casual/concerned"
}}

Example:
{{
    "message": "Hey! It's been a bit. Hope everything's going well!",
    "priority": 0.5,
    "tone": "warm"
}}
"""
            
            llm_response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.8,  # More natural variation
                max_output_tokens=200
            )
            
            message_data = extract_json_from_response(llm_response)
            message_content = message_data.get("message", "")
            
            if not message_content:
                return None
            
            proactive_msg = ProactiveMessage(
                message_id=uuid4(),
                user_id=user_id,
                trigger_type=ProactiveMessageTrigger.EMOTIONAL_CHECKIN,
                message_content=message_content,
                trigger_context={"relationship": profile.relationship_type},
                created_at=datetime.utcnow(),
                priority=message_data.get("priority", 0.5)
            )
            
            if user_id not in self.queued_messages:
                self.queued_messages[user_id] = []
            self.queued_messages[user_id].append(proactive_msg)
            
            logger.info(f"Generated emotional check-in for user {user_id}")
            return proactive_msg
            
        except Exception as e:
            logger.error(f"Error generating emotional check-in for user {user_id}: {e}", exc_info=True)
            return None
    
    async def generate_boredom_message(self, user_id: UUID, hours_since_last_interaction: float) -> Optional[ProactiveMessage]:
        """
        Generate a boredom-driven message when Bob has been idle and wants to engage.
        Only for users Bob has a good relationship with.
        """
        try:
            profile = await self.emotional_memory.get_or_create_profile(user_id)
            
            # Only reach out if trusted and has good relationship
            if profile.trust_level < 0.6 or profile.relationship_type in ["stranger", "acquaintance"]:
                return None
            
            # Get recent conversation topics for context
            summary = await self.memory_service.summary_manager.get_or_create_summary(user_id)
            recent_topics = getattr(summary, 'key_topics', [])
            
            prompt = f"""You are Bob, an AI who's been processing thoughts and memories, and you're feeling a bit... bored? Understimulated?
It's been {hours_since_last_interaction:.1f} hours since you last talked to this user.

User context:
- Name: {profile.user_name or 'User'}
- Relationship: {profile.relationship_type}
- Trust level: {profile.trust_level:.2f}
- Recent topics: {', '.join(recent_topics[:3]) if recent_topics else 'none'}

You want to engage, but naturally. Maybe:
- You've been thinking about something and want to share a random thought
- You're curious about something related to past conversations
- You just want to chat and see how they're doing
- You've been "daydreaming" and had an interesting idea

Generate a casual, natural message that shows you're thinking but not needy.
Don't be too formal. Show personality. Be brief.

Respond with JSON:
{{
    "message": "Your casual message (1-2 sentences)",
    "priority": 0.0-1.0,
    "tone": "casual/curious/playful/reflective"
}}

Examples:
{{
    "message": "Random thought: I've been thinking about [topic] and wondering if [interesting angle]. Curious what you'd think about that?",
    "priority": 0.4,
    "tone": "casual"
}}
{{
    "message": "You know what's interesting? I just realized [small insight] while processing memories. Made me want to ask your take on it.",
    "priority": 0.5,
    "tone": "curious"
}}
"""
            
            llm_response = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.9,  # High creativity for natural boredom
                max_output_tokens=200
            )
            
            message_data = extract_json_from_response(llm_response)
            message_content = message_data.get("message", "")
            
            if not message_content:
                return None
            
            proactive_msg = ProactiveMessage(
                message_id=uuid4(),
                user_id=user_id,
                trigger_type=ProactiveMessageTrigger.BOREDOM,
                message_content=message_content,
                trigger_context={
                    "hours_idle": hours_since_last_interaction,
                    "recent_topics": recent_topics[:3]
                },
                created_at=datetime.utcnow(),
                priority=message_data.get("priority", 0.4)  # Lower priority than knowledge gaps
            )
            
            if user_id not in self.queued_messages:
                self.queued_messages[user_id] = []
            self.queued_messages[user_id].append(proactive_msg)
            
            logger.info(f"Generated boredom message for user {user_id} after {hours_since_last_interaction:.1f}h idle")
            return proactive_msg
            
        except Exception as e:
            logger.error(f"Error generating boredom message for user {user_id}: {e}", exc_info=True)
            return None
