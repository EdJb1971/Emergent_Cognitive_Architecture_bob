from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

class PerceptionAnalysis(BaseModel):
    topics: List[str] = Field(..., description="Main topics identified in the user input.")
    patterns: List[str] = Field(..., description="Recurring patterns or themes detected.")
    context_type: str = Field(..., description="Categorization of the input's context (e.g., 'question', 'statement', 'command', 'narrative').")
    keywords: List[str] = Field(..., description="Key terms extracted from the input.")
    image_analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of the image provided by the user.")
    audio_analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of the audio provided by the user.")

class EmotionScore(BaseModel):
    emotion: str = Field(..., description="Name of the emotion (e.g., 'joy', 'sadness', 'anger').")
    score: float = Field(..., ge=0.0, le=1.0, description="Intensity score of the emotion.")

class EmotionalAnalysis(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment of the input (e.g., 'positive', 'negative', 'neutral', 'mixed').")
    emotions: List[EmotionScore] = Field(..., description="List of detected emotions and their intensity scores.")
    intensity: str = Field(..., description="Overall emotional intensity (e.g., 'high', 'medium', 'low').")
    interpersonal_dynamics: Optional[str] = Field(None, description="Analysis of implied interpersonal dynamics, if any.")

class MemoryAnalysis(BaseModel):
    retrieved_context: List[Dict[str, Any]] = Field(..., description="List of relevant past interactions or knowledge retrieved from memory.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Overall relevance score of the retrieved context.")
    source_memory_ids: List[str] = Field(..., description="List of IDs of the memory entries from which context was retrieved.")

class CriticAnalysis(BaseModel):
    logical_coherence: str = Field(..., description="Assessment of the input's logical coherence (e.g., 'high', 'medium', 'low').")
    contradictions_found: List[str] = Field(..., description="Descriptions of any contradictions or inconsistencies found.")
    biases_identified: List[str] = Field(..., description="Descriptions of any identifiable biases in the input.")
    critical_feedback: str = Field(..., description="Overall critical assessment or feedback on the input.")

class PlanningAnalysis(BaseModel):
    feasible_options: List[str] = Field(..., description="List of potential response options or actions.")
    strategic_considerations: List[str] = Field(..., description="Strategic factors to consider for each option.")
    risks: List[str] = Field(..., description="Potential risks associated with the proposed options.")
    recommended_action: Optional[str] = Field(None, description="The agent's recommended course of action, if any.")

class CreativeAnalysis(BaseModel):
    novel_perspectives: List[str] = Field(..., description="New or unusual ways of looking at the input.")
    analogies: List[str] = Field(..., description="Relevant analogies or metaphors generated.")
    reframings: List[str] = Field(..., description="Alternative ways to frame the problem or situation.")
    creative_score: float = Field(..., ge=0.0, le=1.0, description="A score indicating the novelty and utility of the creative output.")

class DiscoveryAnalysis(BaseModel):
    knowledge_gaps: List[str] = Field(..., description="Identified areas where more information or understanding is needed.")
    curiosities_generated: List[str] = Field(..., description="Questions or areas of interest for further exploration.")
    proposed_explorations: List[str] = Field(..., description="Suggestions for how to explore the identified gaps or curiosities.")
    discovery_priority: int = Field(..., ge=1, description="Priority level for pursuing the identified discoveries.")
    web_search_results: Optional[List[Dict[str, Any]]] = Field(None, description="Summarized results from autonomous web searches, if performed.")


# ============================================================================
# EMOTIONAL INTELLIGENCE MODELS
# ============================================================================

class EmotionalProfile(BaseModel):
    """
    Persistent emotional profile for a user, tracking relationship and emotional history.
    Enables human-like relational awareness and emotional continuity.
    """
    user_id: str = Field(..., description="UUID of the user this profile belongs to")
    user_name: Optional[str] = Field(None, description="User's name if shared")
    relationship_type: str = Field(default="new_user", description="Relationship status: new_user, acquaintance, friend, collaborator")
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Trust level built through interactions")
    interaction_count: int = Field(default=0, description="Total number of interactions")
    
    # Emotional history
    recent_sentiments: List[str] = Field(default_factory=list, description="Last 5 sentiments detected")
    emotional_trend: str = Field(default="stable", description="Emotional trajectory: improving, declining, stable, volatile")
    comfort_level: float = Field(default=0.5, ge=0.0, le=1.0, description="User's comfort/openness level")
    
    # Relational memory
    shared_topics: Set[str] = Field(default_factory=set, description="Topics discussed together")
    positive_moments: List[str] = Field(default_factory=list, description="Memorable positive exchanges")
    concerns_raised: List[str] = Field(default_factory=list, description="Topics that caused frustration")
    
    # Extensible metadata for additional features
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata storage for features like proactive engagement preferences")
    
    # Meta
    first_interaction_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of first interaction")
    last_interaction_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last interaction")
    last_emotion_detected: Optional[str] = Field(None, description="Most recent emotion detected")


class EmotionalAgentOutput(BaseModel):
    """
    Enhanced output from emotional agent with relational and emotional intelligence.
    """
    # Current sentiment (existing functionality)
    current_sentiment: str = Field(..., description="Current sentiment: positive, negative, neutral, mixed")
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in sentiment analysis")
    
    # Relational context (NEW)
    user_recognition: str = Field(default="new_user", description="Recognition status: new_user, returning_user, recognized_friend")
    relationship_status: str = Field(default="new_user", description="Relationship type: new_user, acquaintance, friend, collaborator")
    trust_signal: float = Field(default=0.5, ge=0.0, le=1.0, description="Current trust level")
    
    # Emotional intelligence (NEW)
    emotional_shift: Optional[str] = Field(None, description="Emotional trajectory: improving, declining, stable, volatile")
    empathy_cues: List[str] = Field(default_factory=list, description="Detected emotional cues: excited, concerned, testing, etc.")
    response_tone_recommendation: str = Field(default="neutral", description="Recommended tone: warm, supportive, celebratory, concerned, neutral, playful")
    
    # Context from memory
    relevant_emotional_history: Optional[str] = Field(None, description="Brief summary of emotional history")
    
    # Agent metadata
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Overall confidence")
    priority: int = Field(default=2, ge=1, le=5, description="Priority level")


# ============================================================================
# BRAIN ARCHITECTURE MODELS (Phase 1: Self-Model, Working Memory, Salience)
# ============================================================================

class AutobiographicalMemory(BaseModel):
    """
    A single autobiographical memory event about the system itself.
    Inspired by episodic memory in the Default Mode Network.
    """
    event: str = Field(..., description="Type of event: named_by_user, role_assigned, personality_noted, etc.")
    description: str = Field(..., description="Detailed description of the event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the event occurred")
    emotional_significance: str = Field(default="medium", description="Significance: low, medium, high")
    cycle_id: Optional[str] = Field(None, description="Associated cognitive cycle ID")


class SelfModel(BaseModel):
    """
    System's sense of self and personal history, inspired by Default Mode Network.
    Maintains identity, autobiographical memories, and beliefs about the user.
    """
    user_id: str = Field(..., description="UUID of the user this self-model is associated with")
    
    # Identity
    system_name: Optional[str] = Field(None, description="Name given to the system by the user")
    role: str = Field(default="cognitive_assistant", description="System's understanding of its role")
    personality_traits: List[str] = Field(default_factory=list, description="Emergent personality traits")
    relationship_to_user: str = Field(default="new_acquaintance", description="Relationship status with user")
    
    # Autobiographical memory
    autobiographical_memories: List[AutobiographicalMemory] = Field(default_factory=list, description="Episodic memories about self")
    
    # Theory of mind (beliefs about user)
    beliefs_about_user: Dict[str, Any] = Field(default_factory=dict, description="What the system knows about the user")
    
    # Interaction history
    first_interaction: Optional[datetime] = Field(None, description="Timestamp of first interaction")
    total_interactions: int = Field(default=0, description="Total number of interactions")
    interaction_quality_trend: List[float] = Field(default_factory=list, description="Quality scores over time (0.0-1.0)")
    
    # Meta
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last time this model was updated")


class WorkingMemoryContext(BaseModel):
    """
    Current task context maintained in working memory, inspired by Prefrontal Cortex.
    Synthesizes insights from Stage 1 agents to guide Stage 2 processing.
    """
    # Extracted from Stage 1
    topics: List[str] = Field(default_factory=list, description="Active topics from perception")
    sentiment: Optional[str] = Field(None, description="Current sentiment from emotional agent")
    emotional_priority: bool = Field(default=False, description="Whether emotional response is prioritized")
    recalled_memories: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant memories from memory agent")
    memory_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in memory recall")
    
    # Inferred context
    attention_focus: List[str] = Field(default_factory=list, description="Entities/concepts requiring attention")
    attention_motifs: List[str] = Field(default_factory=list, description="AttentionController-provided motifs for Stage 2 emphasis")
    inferred_goals: List[str] = Field(default_factory=list, description="User's inferred goals for this interaction")
    multimodal: bool = Field(default=False, description="Whether multimodal input is present")
    
    # For Stage 2 guidance
    user_input: Optional[str] = Field(None, description="Original user input")


class EmotionalSalience(BaseModel):
    """
    Emotional salience scoring for memory consolidation, inspired by Amygdala.
    Tags memories with emotional significance for prioritized recall.
    """
    salience_score: float = Field(..., ge=0.0, le=1.0, description="Overall emotional salience (0.0-1.0)")
    emotional_valence: str = Field(default="neutral", description="Emotional tone: positive, negative, neutral, mixed")
    emotional_arousal: str = Field(default="low", description="Arousal level: low, medium, high")
    contains_personal_disclosure: bool = Field(default=False, description="Whether input contains personal information")
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How novel/surprising the interaction was")
    consolidation_priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for long-term consolidation")


# ===========================
# PHASE 2: Brain Architecture Models
# ===========================

class QuickAnalysis(BaseModel):
    """
    Quick pre-processing analysis from Thalamus Gateway.
    Determines input characteristics for selective attention.
    """
    modality: str = Field(..., description="Input modality: text, image, audio, multimodal")
    urgency: str = Field(..., description="Urgency level: high, normal, low")
    complexity: str = Field(..., description="Cognitive complexity: simple, moderate, complex")
    context_need: str = Field(..., description="Memory context needed: minimal, recent, deep")


class InputRouting(BaseModel):
    """
    Routing configuration from Thalamus Gateway.
    Specifies which agents to activate and with what parameters.
    """
    quick_analysis: QuickAnalysis = Field(..., description="Quick analysis results")
    agent_activation: Dict[str, bool] = Field(..., description="Which agents should be activated")
    memory_config: Dict[str, Any] = Field(..., description="Memory retrieval configuration")
    agent_token_budget: Dict[str, int] = Field(default_factory=dict, description="Per-agent token budgets for downstream prompting")
    attention_motifs: List[str] = Field(default_factory=list, description="Priority motifs/topics that Stage 2 agents should emphasize")


class AttentionDirective(BaseModel):
    """
    Directive produced by the AttentionController describing how to bias agent routing.
    """
    shadow_mode: bool = Field(True, description="If True, directive is logged but not enforced")
    agent_bias: Dict[str, float] = Field(default_factory=dict, description="Bias per agent (-1.0 suppression, +1.0 amplification)")
    suppress_agents: List[str] = Field(default_factory=list, description="Agents to forcibly deactivate")
    amplify_agents: List[str] = Field(default_factory=list, description="Agents to force-activate")
    memory_override: Optional[Dict[str, Any]] = Field(None, description="Override for memory retrieval configuration")
    notes: List[str] = Field(default_factory=list, description="Human-readable rationale snippets")
    drift_score: float = Field(0.0, ge=0.0, le=1.0, description="Normalized drift score between previous and current context")
    drift_reasons: List[str] = Field(default_factory=list, description="Reasons explaining detected drift")
    stage: str = Field("pre_stage1", description="Lifecycle stage when directive was generated")
    user_id: Optional[str] = Field(None, description="User identifier for per-user attention state")


class Conflict(BaseModel):
    """
    Individual conflict detected between agent outputs.
    """
    conflict_type: str = Field(..., description="Type of conflict detected")
    agents: List[str] = Field(..., description="Agents involved in the conflict")
    severity: str = Field(..., description="Severity: low, medium, high")
    resolution_strategy: str = Field(..., description="Suggested resolution approach")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional conflict details")


class ConflictReport(BaseModel):
    """
    Report of all conflicts detected in agent outputs.
    """
    conflicts: List[Conflict] = Field(default_factory=list, description="List of detected conflicts")
    requires_adjustment: bool = Field(default=False, description="Whether re-processing is needed")
    coherence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall coherence score")


# ===========================
# PHASE 3: Advanced Brain Architecture Models
# ===========================

class EpisodicMemory(BaseModel):
    """
    Specific episodic memory - a 'mental time travel' moment.
    Richer than AutobiographicalMemory, includes full sensory/emotional reconstruction.
    """
    episode_id: str = Field(..., description="Unique identifier for this episode")
    timestamp: datetime = Field(..., description="When this episode occurred")
    narrative: str = Field(..., description="Rich narrative description of the episode")
    participants: List[str] = Field(default_factory=list, description="Who was involved (user, system, others mentioned)")
    location_context: Optional[str] = Field(None, description="Where this occurred (if mentioned)")
    emotional_tone: str = Field(..., description="Overall emotional tone of the episode")
    significance: float = Field(..., ge=0.0, le=1.0, description="How significant this episode is")
    related_episode_ids: List[str] = Field(default_factory=list, description="IDs of related episodes")
    key_insights: List[str] = Field(default_factory=list, description="Key learnings or insights from this episode")
    sensory_details: Dict[str, Any] = Field(default_factory=dict, description="Sensory details (visual, auditory, etc.)")
    cycle_id: Optional[str] = Field(None, description="Associated cognitive cycle ID")


class SemanticMemory(BaseModel):
    """
    Extracted semantic knowledge - facts, concepts, patterns learned over time.
    Distilled from episodic memories through consolidation.
    """
    concept_id: str = Field(..., description="Unique identifier for this concept")
    concept_name: str = Field(..., description="Name of the concept/fact/pattern")
    description: str = Field(..., description="Description of what this concept means")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this knowledge")
    source_episode_ids: List[str] = Field(default_factory=list, description="Episodes that contributed to this knowledge")
    first_learned: datetime = Field(..., description="When this was first learned")
    last_reinforced: datetime = Field(..., description="When this was last reinforced")
    reinforcement_count: int = Field(default=1, description="How many times this has been reinforced")
    category: str = Field(..., description="Category: user_preference, user_fact, world_knowledge, system_capability, etc.")


class UserMentalState(BaseModel):
    """
    Current theory of mind - what we believe about the user's mental state.
    """
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this state was inferred")
    
    # Beliefs about user's current state
    current_goal: Optional[str] = Field(None, description="What the user is trying to accomplish")
    current_emotion: str = Field(default="neutral", description="User's current emotional state")
    current_needs: List[str] = Field(default_factory=list, description="What the user needs right now")
    
    # Beliefs about user's knowledge
    knows_about: List[str] = Field(default_factory=list, description="Topics/concepts the user understands")
    confused_about: List[str] = Field(default_factory=list, description="Topics/concepts the user is confused about")
    interested_in: List[str] = Field(default_factory=list, description="Topics the user is interested in")
    
    # Beliefs about user's intentions
    likely_next_action: Optional[str] = Field(None, description="What the user will likely do/ask next")
    conversation_intent: str = Field(..., description="Overall intent: seeking_info, problem_solving, venting, exploring, etc.")
    
    # Meta-beliefs
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this mental state model")
    uncertainty_factors: List[str] = Field(default_factory=list, description="What makes us uncertain about this model")


class MemoryConsolidationJob(BaseModel):
    """
    A memory consolidation job to be processed in the background.
    Mimics sleep-like memory processing.
    """
    job_id: str = Field(..., description="Unique identifier for this consolidation job")
    user_id: str = Field(..., description="User whose memories are being consolidated")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this job was created")
    status: str = Field(default="pending", description="pending, processing, completed, failed")
    
    # Job configuration
    cycle_ids_to_process: List[str] = Field(default_factory=list, description="Specific cycles to consolidate")
    consolidation_type: str = Field(..., description="episodic_to_semantic, memory_replay, pattern_extraction, etc.")
    priority: float = Field(..., ge=0.0, le=1.0, description="Priority for processing (based on salience)")
    
    # Job results
    episodes_created: int = Field(default=0, description="Number of episodic memories created")
    semantic_concepts_extracted: int = Field(default=0, description="Number of semantic concepts extracted")
    patterns_discovered: List[str] = Field(default_factory=list, description="Patterns discovered during consolidation")
    completed_at: Optional[datetime] = Field(None, description="When this job completed")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class TheoryOfMindPrediction(BaseModel):
    """
    Prediction about user's mental state and intentions.
    """
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this prediction was made")
    
    # Prediction content
    predicted_intention: str = Field(..., description="What we predict the user intends to do")
    predicted_needs: List[str] = Field(default_factory=list, description="What we predict the user needs")
    predicted_confusion_points: List[str] = Field(default_factory=list, description="Where we predict the user might be confused")
    
    # Prediction confidence
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this prediction")
    reasoning: str = Field(..., description="Why we made this prediction")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this prediction")
    
    # Validation
    was_validated: bool = Field(default=False, description="Whether this prediction was validated")
    validation_result: Optional[bool] = Field(None, description="True if prediction was correct, False if wrong")
    validation_feedback: Optional[str] = Field(None, description="Feedback on prediction accuracy")


