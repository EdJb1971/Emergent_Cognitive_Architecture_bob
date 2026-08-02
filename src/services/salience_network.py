"""Explainable, advisory-only memory salience ranking.

This combines retrieval, temporal, affective, novelty, goal and preservation
signals after baseline retrieval. It never drops a candidate or mutates the
retrieval score, which keeps a clean control group for later calibration.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.models.core_models import CognitiveCycle
from src.models.salience_models import (
    SalienceAssessment,
    SalienceCandidate,
    SalienceFactorScores,
)


class SalienceNetwork:
    """Rank retrieved memories with bounded and human-inspectable signals."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "query_relevance": 0.38,
        "recency": 0.15,
        "emotional_salience": 0.20,
        "novelty": 0.10,
        "goal_alignment": 0.12,
        "must_keep": 0.05,
    }
    _TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
    _STOP_WORDS = {
        "about", "after", "again", "also", "and", "are", "but", "can",
        "for", "from", "have", "how", "into", "its", "our", "that", "the",
        "their", "then", "this", "was", "what", "when", "where", "which",
        "who", "why", "will", "with", "would", "you", "your",
    }

    def __init__(
        self,
        *,
        enabled: bool = False,
        shadow_mode: bool = True,
        top_k: int = 3,
        recency_half_life_days: float = 30.0,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        selected_weights = dict(weights or self.DEFAULT_WEIGHTS)
        if set(selected_weights) != set(self.DEFAULT_WEIGHTS):
            raise ValueError("salience weights must define every supported factor")
        if any(value < 0 for value in selected_weights.values()):
            raise ValueError("salience weights cannot be negative")
        weight_total = sum(selected_weights.values())
        if weight_total <= 0:
            raise ValueError("salience weights must have a positive total")

        self.enabled = bool(enabled)
        self.shadow_mode = bool(shadow_mode)
        self.top_k = top_k
        self.recency_half_life_days = recency_half_life_days
        self.weights = {
            name: value / weight_total for name, value in selected_weights.items()
        }

    @property
    def exposes_advisory(self) -> bool:
        """Whether compact recommendations may influence waking prompts."""
        return self.enabled and not self.shadow_mode

    def assess_memories(
        self,
        memories: Sequence[CognitiveCycle],
        *,
        query_text: str = "",
        goal_terms: Optional[Iterable[str]] = None,
        now: Optional[datetime] = None,
        top_k: Optional[int] = None,
    ) -> SalienceAssessment:
        """Return a complete alternative ranking while preserving baseline order."""
        assessment_time = now or datetime.utcnow()
        requested_top_k = self.top_k if top_k is None else max(1, top_k)
        query_tokens = self._tokens(query_text)
        goal_tokens = set(query_tokens)
        for term in goal_terms or ():
            goal_tokens.update(self._tokens(str(term)))

        scored: List[Dict[str, Any]] = []
        baseline_order: List[str] = []
        for baseline_index, memory in enumerate(memories):
            memory_id = str(memory.cycle_id)
            baseline_order.append(memory_id)
            factors = self._factor_scores(
                memory,
                goal_tokens=goal_tokens,
                now=assessment_time,
            )
            factor_dump = factors.model_dump()
            contributions = {
                name: round(factor_dump[name] * weight, 6)
                for name, weight in self.weights.items()
            }
            score = sum(contributions.values())
            must_keep = factors.must_keep == 1.0
            if must_keep:
                score = max(score, 0.90)
            scored.append(
                {
                    "memory_id": memory_id,
                    "baseline_rank": baseline_index + 1,
                    "salience_score": self._round_score(score),
                    "factors": factors,
                    "weighted_contributions": contributions,
                    "reasons": self._reasons(factors),
                    "must_keep": must_keep,
                }
            )

        ranked = sorted(
            scored,
            key=lambda item: (-item["salience_score"], item["baseline_rank"]),
        )
        candidates: List[SalienceCandidate] = []
        for recommended_index, item in enumerate(ranked, start=1):
            candidates.append(
                SalienceCandidate(recommended_rank=recommended_index, **item)
            )

        effective_top_k = min(requested_top_k, len(candidates))
        return SalienceAssessment(
            enabled=self.enabled,
            shadow_mode=self.shadow_mode,
            generated_at=assessment_time,
            candidate_count=len(candidates),
            top_k=effective_top_k,
            baseline_order=baseline_order,
            recommended_order=[item.memory_id for item in candidates],
            candidates=candidates,
        )

    def _factor_scores(
        self,
        memory: CognitiveCycle,
        *,
        goal_tokens: set[str],
        now: datetime,
    ) -> SalienceFactorScores:
        metadata = memory.metadata if isinstance(memory.metadata, dict) else {}
        contextual = self._mapping(metadata.get("contextual_bindings"))
        emotional = self._mapping(metadata.get("emotional_salience"))
        consolidation = self._mapping(metadata.get("consolidation_metadata"))

        relevance_fallback = self._bounded(consolidation.get("consolidation_priority"), 0.5)
        relevance = self._bounded(memory.score, relevance_fallback)
        recency = self._recency(memory.timestamp, now)
        affect = self._bounded(
            emotional.get("salience_score"),
            self._arousal_score(contextual.get("emotional_arousal")),
        )
        novelty = self._bounded(
            emotional.get("novelty_score"),
            self._bounded(contextual.get("novelty"), 0.5),
        )
        memory_tokens = self._memory_tokens(memory, contextual)
        alignment = self._overlap(goal_tokens, memory_tokens) if goal_tokens else 0.5
        must_keep = self._must_keep(metadata)

        return SalienceFactorScores(
            query_relevance=relevance,
            recency=recency,
            emotional_salience=affect,
            novelty=novelty,
            goal_alignment=alignment,
            must_keep=1.0 if must_keep else 0.0,
        )

    def _recency(self, timestamp: datetime, now: datetime) -> float:
        try:
            normalized_now = now
            normalized_timestamp = timestamp
            if normalized_timestamp.tzinfo is not None and normalized_now.tzinfo is None:
                normalized_now = normalized_now.replace(tzinfo=timezone.utc)
            elif normalized_timestamp.tzinfo is None and normalized_now.tzinfo is not None:
                normalized_timestamp = normalized_timestamp.replace(tzinfo=timezone.utc)
            age_seconds = max(
                0.0,
                (normalized_now - normalized_timestamp).total_seconds(),
            )
        except (TypeError, ValueError):
            return 0.5
        age_days = age_seconds / 86400.0
        return self._round_score(math.pow(0.5, age_days / self.recency_half_life_days))

    def _memory_tokens(
        self,
        memory: CognitiveCycle,
        contextual: Mapping[str, Any],
    ) -> set[str]:
        material: List[str] = [memory.user_input or "", memory.final_response or ""]
        for field in ("topics", "entities"):
            value = contextual.get(field, [])
            if isinstance(value, list):
                material.extend(str(item) for item in value[:20])
        return self._tokens(" ".join(material))

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token for token in cls._TOKEN_PATTERN.findall((text or "").lower())
            if token not in cls._STOP_WORDS
        }

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return SalienceNetwork._round_score(len(left & right) / len(left))

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _bounded(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        if not math.isfinite(numeric):
            numeric = default
        return SalienceNetwork._round_score(max(0.0, min(1.0, numeric)))

    @staticmethod
    def _round_score(value: float) -> float:
        return round(max(0.0, min(1.0, value)), 6)

    @staticmethod
    def _arousal_score(value: Any) -> float:
        return {"high": 0.9, "medium": 0.65, "low": 0.35}.get(
            str(value or "").lower(), 0.5
        )

    @staticmethod
    def _must_keep(metadata: Mapping[str, Any]) -> bool:
        if any(metadata.get(flag) is True for flag in ("must_keep", "pinned", "protected")):
            return True
        preservation = metadata.get("preservation")
        return isinstance(preservation, dict) and preservation.get("must_keep") is True

    @staticmethod
    def _reasons(factors: SalienceFactorScores) -> List[str]:
        reasons: List[str] = []
        if factors.must_keep == 1.0:
            reasons.append("must_keep")
        if factors.query_relevance >= 0.75:
            reasons.append("high_query_relevance")
        if factors.recency >= 0.75:
            reasons.append("recent_memory")
        if factors.emotional_salience >= 0.70:
            reasons.append("emotionally_salient")
        if factors.novelty >= 0.70:
            reasons.append("novel_context")
        if factors.goal_alignment >= 0.50:
            reasons.append("goal_aligned")
        return reasons or ["balanced_signal"]
