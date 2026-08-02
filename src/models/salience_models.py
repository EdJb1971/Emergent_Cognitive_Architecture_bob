"""Structured contracts for explainable, advisory memory salience."""

from datetime import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class SalienceFactorScores(BaseModel):
    """Normalized signals used to score one retrieved memory."""

    query_relevance: float = Field(..., ge=0.0, le=1.0)
    recency: float = Field(..., ge=0.0, le=1.0)
    emotional_salience: float = Field(..., ge=0.0, le=1.0)
    novelty: float = Field(..., ge=0.0, le=1.0)
    goal_alignment: float = Field(..., ge=0.0, le=1.0)
    must_keep: float = Field(..., ge=0.0, le=1.0)


class SalienceCandidate(BaseModel):
    """An auditable recommendation for one baseline retrieval candidate."""

    memory_id: str
    baseline_rank: int = Field(..., ge=1)
    recommended_rank: int = Field(..., ge=1)
    salience_score: float = Field(..., ge=0.0, le=1.0)
    factors: SalienceFactorScores
    weighted_contributions: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    must_keep: bool = False


class SalienceAssessment(BaseModel):
    """Full control-group and recommendation record for one ranking decision."""

    version: Literal["salience-v1"] = "salience-v1"
    mode: Literal["advisory"] = "advisory"
    enabled: bool
    shadow_mode: bool
    pruning_applied: Literal[False] = False
    generated_at: datetime
    candidate_count: int = Field(..., ge=0)
    top_k: int = Field(..., ge=0)
    baseline_order: List[str] = Field(default_factory=list)
    recommended_order: List[str] = Field(default_factory=list)
    candidates: List[SalienceCandidate] = Field(default_factory=list)

