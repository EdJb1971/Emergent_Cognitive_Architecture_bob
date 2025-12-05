from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import model_validator as _model_validator
except Exception:
    _model_validator = None
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from uuid import UUID, uuid4

class AgentOutput(BaseModel):
    """
    Base model for the structured output of any specialized AI agent.
    """
    agent_id: str = Field(..., description="Unique identifier for the agent (e.g., 'perception_agent').")
    analysis: Dict[str, Any] = Field(..., description="Structured analysis or data produced by the agent.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the agent's output (0.0 to 1.0).")
    priority: int = Field(..., ge=1, description="Priority level of the agent's output (higher means more critical).")
    raw_output: Optional[str] = Field(None, description="Optional raw text output from the agent, if applicable.")

class UserRequest(BaseModel):
    """
    Model for an incoming user request to the API Gateway.
    """
    user_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the user.")
    input_text: str = Field("", min_length=0, description="The user's input text (may be empty when image/audio provided).")
    session_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the current session.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the request.")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded image data.")
    audio_base64: Optional[str] = Field(None, description="Optional base64 encoded audio data.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., responding_to_proactive_message)")

    # Ensure at least one modality is provided (text, image, or audio)
    if _model_validator is not None:
        @_model_validator(mode="after")
        def _check_any_input(cls, values):
            if not values.input_text and not values.image_base64 and not values.audio_base64:
                raise ValueError("At least one of input_text, image_base64, or audio_base64 must be provided.")
            return values

class ResponseMetadata(BaseModel):
    """
    Structured metadata about the Cognitive Brain's final response.
    """
    response_type: str = Field(..., description="Categorization of the response (e.g., 'informational', 'empathetic', 'actionable', 'creative').")
    tone: str = Field(..., description="Emotional tone of the response (e.g., 'neutral', 'supportive', 'directive', 'curious').")
    strategies: List[str] = Field(default_factory=list, description="List of cognitive strategies used (e.g., 'clarification', 'reframing', 'problem_solving').")
    cognitive_moves: List[str] = Field(default_factory=list, description="Specific cognitive moves made (e.g., 'ask_follow_up', 'provide_analogy', 'summarize').")

class OutcomeSignals(BaseModel):
    """
    Signals indicating the potential outcome or impact of the response.
    """
    user_satisfaction_potential: float = Field(..., ge=0.0, le=1.0, description="Predicted likelihood of user satisfaction.")
    engagement_potential: float = Field(..., ge=0.0, le=1.0, description="Predicted likelihood of further user engagement.")

class CognitiveCycle(BaseModel):
    """
    Model representing a complete cognitive cycle, from user input to final response.
    This will be stored in the Persistent Memory System.
    """
    cycle_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this cognitive cycle.")
    user_id: UUID = Field(..., description="Unique identifier for the user who initiated this cycle.")
    session_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the session this cycle belongs to.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when the cycle was initiated.")
    user_input: str = Field(..., description="The original user input for this cycle.")
    user_input_embedding: Optional[List[float]] = Field(None, description="Vector embedding of the user input.")
    agent_outputs: List[AgentOutput] = Field(default_factory=list, description="List of structured outputs from all specialized agents.")
    final_response: Optional[str] = Field(None, description="The final natural language response generated for the user.")
    final_response_embedding: Optional[List[float]] = Field(None, description="Vector embedding of the final response.")
    response_metadata: Optional[ResponseMetadata] = Field(None, description="Structured metadata about the final response.")
    outcome_signals: Optional[OutcomeSignals] = Field(None, description="Signals indicating the potential outcome of the response.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional unstructured metadata about the cycle.")
    reflection_status: str = Field("pending", description="Status of self-reflection for this cycle (pending, completed, failed).")
    discovery_status: str = Field("pending", description="Status of autonomous discovery for this cycle (pending, completed, failed).")
    score: Optional[float] = Field(None, description="The relevance score of the cycle from a vector search.")

class MemoryQueryRequest(BaseModel):
    user_id: UUID = Field(..., description="The ID of the user whose memory is being queried.")
    query_text: str = Field(..., min_length=1, description="The natural language query to search memory.")
    query_embedding: Optional[List[float]] = Field(None, description="The vector embedding of the query.")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of relevant memories to retrieve.")
    min_relevance_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score for retrieved memories.")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Key-value pairs to filter memory by metadata (e.g., {'response_metadata.response_type': 'informational'}).")

class MemoryQueryResponse(BaseModel):
    user_id: UUID
    query_text: str
    retrieved_cycles: List[CognitiveCycle] = Field(default_factory=list, description="List of cognitive cycles retrieved from memory.")
    message: str

class ReflectionTriggerRequest(BaseModel):
    user_id: UUID = Field(..., description="The ID of the user for whom reflection is triggered.")
    num_cycles: int = Field(1, ge=1, description="Number of past cognitive cycles to reflect on.")
    trigger_type: Literal["manual", "scheduled", "event_driven"] = Field("manual", description="Type of reflection trigger.")

class DiscoveryTriggerRequest(BaseModel):
    user_id: UUID = Field(..., description="The ID of the user for whom discovery is triggered.")
    discovery_type: Literal["memory_analysis", "curiosity_exploration", "self_assessment"] = Field(..., description="Type of autonomous discovery to initiate.")
    context: Optional[str] = Field(None, description="Optional context for the discovery process.")

class DiscoveredPattern(BaseModel):
    pattern_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the discovered pattern.")
    user_id: UUID = Field(..., description="User ID associated with this pattern.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when the pattern was discovered.")
    pattern_type: str = Field(..., description="Type of pattern (e.g., 'meta_learning', 'knowledge_gap', 'curiosity').")
    description: str = Field(..., description="Detailed description of the discovered pattern or insight.")
    source_cycle_ids: List[UUID] = Field(default_factory=list, description="List of cognitive cycle IDs that contributed to this discovery.")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the pattern description.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the pattern.")

class PatternsResponse(BaseModel):
    user_id: UUID
    patterns: List[DiscoveredPattern] = Field(default_factory=list, description="List of discovered meta-learnings and patterns.")
    message: str

class CycleUpdateRequest(BaseModel):
    """
    Model for updating metadata of a specific cognitive cycle.
    """
    metadata: Dict[str, Any] = Field(..., description="Dictionary of metadata fields to update.")

class CycleListRequest(BaseModel):
    """
    Model for querying a list of cognitive cycles.
    """
    user_id: UUID = Field(..., description="The ID of the user whose cycles are being queried.")
    skip: int = Field(0, ge=0, description="Number of cycles to skip for pagination.")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of cycles to retrieve.")
    session_id: Optional[UUID] = Field(None, description="Optional session ID to filter cycles.")
    start_date: Optional[datetime] = Field(None, description="Optional start date to filter cycles.")
    end_date: Optional[datetime] = Field(None, description="Optional end date to filter cycles.")
    response_type: Optional[str] = Field(None, description="Optional response type to filter cycles (e.g., 'informational').")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence score of any agent output in the cycle.")

class CycleListResponse(BaseModel):
    """
    Model for the response containing a list of cognitive cycles.
    """
    user_id: UUID
    total_cycles: int
    cycles: List[CognitiveCycle] = Field(default_factory=list, description="List of cognitive cycles.")
    message: str


# ===========================
# PHASE 2: Contextual Memory Models
# ===========================

class ContextualBindings(BaseModel):
    """
    Rich contextual bindings for memory encoding, inspired by Hippocampus.
    Captures when, where, how, and why of an interaction.
    """
    # Temporal context
    time_of_day: str = Field(..., description="Time category: morning, afternoon, evening, night")
    session_duration_minutes: float = Field(..., description="Duration of current session in minutes")
    
    # Emotional context
    emotional_valence: str = Field(..., description="Emotional tone: positive, negative, neutral, mixed")
    emotional_arousal: str = Field(..., description="Arousal level: low, medium, high")
    
    # Semantic context
    topics: List[str] = Field(default_factory=list, description="Main topics discussed")
    entities: List[str] = Field(default_factory=list, description="Named entities mentioned")
    intent: str = Field(..., description="User's intent: question, statement, command, exploration, etc.")
    
    # Relational context
    conversation_depth: str = Field(..., description="Depth: superficial, moderate, deep, intimate")
    rapport_level: str = Field(..., description="Rapport: initial, developing, established, strong")
    
    # Cognitive context
    complexity: str = Field(..., description="Cognitive load: simple, moderate, complex")
    novelty: float = Field(..., ge=0.0, le=1.0, description="How novel/surprising this interaction was")


class ConsolidationMetadata(BaseModel):
    """
    Metadata for memory consolidation prioritization.
    """
    consolidation_priority: float = Field(..., ge=0.0, le=1.0, description="Priority for long-term consolidation (0.0-1.0)")
    replay_count: int = Field(default=0, description="Number of times this memory has been replayed/consolidated")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last time this memory was retrieved")
    access_count: int = Field(default=0, description="Number of times this memory has been accessed")


class ErrorAnalysis(BaseModel):
    """
    Structured analysis of cognitive cycle failures for learning systems.
    Generated when cycles fail due to low coherence or meta-cognitive triggers.
    """
    cycle_id: UUID = Field(..., description="ID of the cognitive cycle that failed")
    failure_type: Literal["coherence_failure", "meta_cognitive_decline", "meta_cognitive_uncertainty", "response_error"] = Field(
        ..., description="Type of failure that triggered this analysis"
    )
    severity_score: float = Field(..., ge=0.0, le=1.0, description="Severity of the failure (0.0-1.0)")
    
    # Agent sequence analysis
    agents_activated: List[str] = Field(..., description="List of agent names that were activated in this cycle")
    agent_conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="Details of any agent conflicts detected")
    
    # Failure context
    coherence_score: Optional[float] = Field(None, description="Final coherence score if available")
    meta_cognitive_assessment: Optional[Dict[str, Any]] = Field(None, description="Meta-cognitive assessment details")
    expected_outcome: float = Field(..., description="Expected satisfaction score (typically 0.7+)")
    actual_outcome: float = Field(..., description="Actual satisfaction score achieved")
    
    # Error categorization
    primary_error_category: Literal[
        "agent_conflict_unresolved", 
        "knowledge_gap_uncovered", 
        "response_inappropriate", 
        "emotional_mismatch",
        "logical_inconsistency",
        "context_misinterpretation",
        "skill_deficiency"
    ] = Field(..., description="Primary category of error")
    
    secondary_error_categories: List[str] = Field(default_factory=list, description="Additional error categories")
    
    # Learning signals
    recommended_agent_sequence: Optional[List[str]] = Field(None, description="Suggested better agent activation sequence")
    skill_improvement_areas: List[str] = Field(default_factory=list, description="Skills that need improvement")
    strategy_adjustments: List[str] = Field(default_factory=list, description="Strategic changes recommended")
    
    # Context for learning
    user_input_summary: str = Field(..., description="Brief summary of user input")
    response_summary: str = Field(..., description="Brief summary of system response")
    cycle_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional cycle metadata")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this analysis was generated")

