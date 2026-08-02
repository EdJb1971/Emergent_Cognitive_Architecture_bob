"""Shadow predictive coding over immutable sensory episodes.

Hypotheses are derived from prior typed cycles before the current observation is
inspected. Predictions remain labelled as hypotheses forever. This service does
not call a model, research provider, router, learner, or response synthesizer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from src.models.core_models import CognitiveCycle
from src.models.multimodal_models import AudioEvidence, SensoryEpisode, VisualEvidence
from src.models.predictive_models import (
    ClarificationRecommendation,
    PerceptualHypothesis,
    PerceptualPredictionError,
    PredictivePerceptionAssessment,
)


_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]{1,63}")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "see", "show", "that",
    "the", "this", "to", "was", "were", "what", "with", "you", "your",
    "appears", "audio", "heard", "here", "image", "picture", "scene", "shown",
    "sound", "there", "visible",
}
_NEGATIONS = {"no", "not", "never", "without", "isn't", "aren't", "wasn't", "weren't"}
_NON_ASSERTIVE_PREFIXES = {
    "am", "are", "can", "could", "describe", "do", "does", "explain", "how",
    "ignore", "is", "please", "show", "tell", "what", "when", "where", "which",
    "who", "why", "will", "would",
}
_ATTRIBUTE_GROUPS: Mapping[str, frozenset[str]] = {
    "colour": frozenset({
        "black", "blue", "brown", "gray", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow",
    }),
    "state": frozenset({"awake", "closed", "empty", "full", "open", "sleeping"}),
}
_ATTRIBUTE_TOKENS = set().union(*_ATTRIBUTE_GROUPS.values())


@dataclass(frozen=True)
class _Observation:
    feature_kind: str
    feature_name: str
    value: str
    modality: str
    reliability: float
    source_reference: str


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _safe_score(value: object) -> float:
    """Treat malformed or non-finite persisted scores as unusable evidence."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _clamp(score) if score == score and abs(score) != float("inf") else 0.0


def _bounded_items(value: object, limit: int) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(value[:limit])


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall((value or "").lower())


def _claims(value: str) -> dict[str, bool]:
    claims: dict[str, bool] = {}
    negation_pending = False
    for token in _tokens(value):
        if token in _NEGATIONS:
            negation_pending = True
            continue
        if token in _STOP:
            continue
        claims.setdefault(token, not negation_pending)
        negation_pending = False
    return claims


def _anchored_attributes(value: str) -> dict[str, str]:
    tokens = _tokens(value)
    result: dict[str, str] = {}
    for index, token in enumerate(tokens):
        group = next((name for name, values in _ATTRIBUTE_GROUPS.items() if token in values), None)
        if not group:
            continue
        nearby = [
            candidate for candidate in reversed(tokens[max(0, index - 4):index])
            if candidate not in _STOP and candidate not in _NEGATIONS and candidate not in _ATTRIBUTE_TOKENS
        ]
        if not nearby:
            nearby = [
                candidate for candidate in tokens[index + 1:index + 5]
                if candidate not in _STOP and candidate not in _NEGATIONS and candidate not in _ATTRIBUTE_TOKENS
            ]
        if nearby:
            result.setdefault(f"{group}:{nearby[0]}", token)
    return result


def _is_assertive(value: str) -> bool:
    stripped = (value or "").strip()
    tokens = _tokens(stripped)
    return bool(
        stripped
        and not stripped.startswith("[")
        and not stripped.endswith("?")
        and tokens
        and tokens[0] not in _NON_ASSERTIVE_PREFIXES
    )


class PredictivePerceptionService:
    """Form prior-only hypotheses and score them against the current episode."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        shadow_mode: bool = True,
        max_prior_cycles: int = 3,
        max_hypotheses: int = 8,
        min_observation_reliability: float = 0.55,
        clarification_threshold: float = 0.50,
    ) -> None:
        if not shadow_mode:
            raise ValueError("Predictive perception is shadow-only in v1.")
        if max_prior_cycles < 0 or max_hypotheses < 0:
            raise ValueError("predictive perception bounds cannot be negative")
        for name, value in {
            "min_observation_reliability": min_observation_reliability,
            "clarification_threshold": clarification_threshold,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.enabled = enabled
        self.shadow_mode = True
        self.max_prior_cycles = max_prior_cycles
        self.max_hypotheses = max_hypotheses
        self.min_observation_reliability = min_observation_reliability
        self.clarification_threshold = clarification_threshold

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "shadow_mode": True,
            "schema_version": "predictive-perception-v1",
            "max_prior_cycles": self.max_prior_cycles,
            "max_hypotheses": self.max_hypotheses,
            "min_observation_reliability": self.min_observation_reliability,
            "clarification_threshold": self.clarification_threshold,
            "response_influence": False,
            "routing_influence": False,
            "research_allowed": False,
            "learning_updates": False,
        }

    def assess(
        self,
        *,
        cycle_id: UUID,
        sensory_episode: SensoryEpisode,
        prior_cycles: Sequence[CognitiveCycle],
        current_text: str,
        visual_evidence: Optional[VisualEvidence] = None,
        audio_evidence: Optional[AudioEvidence] = None,
    ) -> PredictivePerceptionAssessment:
        bounded_priors = tuple(
            cycle for cycle in prior_cycles
            if isinstance(cycle, CognitiveCycle) and cycle.user_id == sensory_episode.user_id
        )[: self.max_prior_cycles] if self.enabled else ()
        # This phase accepts prior cycles only. Do not move observation-dependent
        # filtering into hypothesis formation; that would create hindsight bias.
        hypotheses = self._form_hypotheses(bounded_priors)[: self.max_hypotheses]
        observations = self._current_observations(
            sensory_episode=sensory_episode,
            current_text=current_text,
            visual_evidence=visual_evidence,
            audio_evidence=audio_evidence,
        ) if self.enabled else ()
        errors = tuple(
            self._compare(hypothesis, sensory_episode.episode_id, observations)
            for hypothesis in hypotheses
        )
        recommendation = self._recommend(sensory_episode, errors) if self.enabled else None
        return PredictivePerceptionAssessment(
            assessment_id=uuid5(NAMESPACE_URL, f"eca:predictive-perception:v1:{cycle_id}"),
            cycle_id=cycle_id,
            sensory_episode_id=sensory_episode.episode_id,
            enabled=self.enabled,
            assessment_status="assessed" if self.enabled else "disabled",
            prior_cycle_ids=tuple(cycle.cycle_id for cycle in bounded_priors),
            hypotheses=hypotheses,
            prediction_errors=errors,
            recommendation=recommendation,
            hypothesis_count=len(hypotheses),
            matched_count=sum(item.status == "matched" for item in errors),
            mismatch_count=sum(item.status == "mismatch" for item in errors),
            unobserved_count=sum(item.status == "unobserved" for item in errors),
            low_reliability_count=sum(item.status == "low_reliability" for item in errors),
            material_error_count=sum(item.material for item in errors),
        )

    def degraded_assessment(
        self, *, cycle_id: UUID, sensory_episode_id: UUID,
    ) -> PredictivePerceptionAssessment:
        """Return a typed empty result so an observational failure cannot stop waking cognition."""
        return PredictivePerceptionAssessment(
            assessment_id=uuid5(NAMESPACE_URL, f"eca:predictive-perception:v1:{cycle_id}"),
            cycle_id=cycle_id,
            sensory_episode_id=sensory_episode_id,
            enabled=True,
            assessment_status="degraded",
            degradation_reason="assessment_failed",
            hypothesis_count=0,
            matched_count=0,
            mismatch_count=0,
            unobserved_count=0,
            low_reliability_count=0,
            material_error_count=0,
        )

    def _form_hypotheses(self, prior_cycles: Sequence[CognitiveCycle]) -> tuple[PerceptualHypothesis, ...]:
        candidates: list[PerceptualHypothesis] = []
        seen: set[tuple[str, str]] = set()
        for cycle in prior_cycles:
            if _is_assertive(cycle.user_input):
                reference = "text_sha256:" + hashlib.sha256(cycle.user_input.encode("utf-8")).hexdigest()
                for name, value in _anchored_attributes(cycle.user_input).items():
                    self._append_hypothesis(
                        candidates, seen, cycle, reference, "prior_user_assertion",
                        "categorical_attribute", name, value, 0.60,
                    )
                for token, present in list(_claims(cycle.user_input).items())[:4]:
                    self._append_hypothesis(
                        candidates, seen, cycle, reference, "prior_user_assertion",
                        "presence", token, "present" if present else "absent", 0.55,
                    )

            metadata = cycle.metadata if isinstance(cycle.metadata, dict) else {}
            episode = metadata.get("sensory_episode")
            if isinstance(episode, dict):
                for index, relation in enumerate(_bounded_items(episode.get("relations"), 3)):
                    if not isinstance(relation, dict) or relation.get("relation_type") != "agreement":
                        continue
                    confidence = _safe_score(relation.get("strength"))
                    if confidence < 0.35:
                        continue
                    for anchor in _bounded_items(relation.get("anchors"), 3):
                        anchor = str(anchor).strip().lower()
                        if anchor and ":" not in anchor:
                            self._append_hypothesis(
                                candidates, seen, cycle,
                                f"sensory_episode:{episode.get('episode_id')}:relation:{index}",
                                "prior_cross_modal_corroboration", "presence", anchor,
                                "present", confidence,
                            )

            visual = metadata.get("visual_evidence")
            if isinstance(visual, dict):
                analysis = visual.get("analysis", {})
                quality = _safe_score(visual.get("input_quality_score"))
                model_confidence = (
                    _safe_score(analysis.get("confidence"))
                    if isinstance(analysis, dict) else 0.0
                )
                confidence = _clamp(quality * 0.75 + model_confidence * 0.25)
                if confidence >= self.min_observation_reliability and isinstance(analysis, dict):
                    for item in _bounded_items(analysis.get("objects_detected"), 3):
                        token = str(item).strip().lower()
                        if token:
                            self._append_hypothesis(
                                candidates, seen, cycle,
                                f"image_sha256:{visual.get('sha256', '')}",
                                "prior_visual_object_observation", "presence", token,
                                "present", confidence,
                            )

            auditory = metadata.get("auditory_evidence")
            if isinstance(auditory, dict):
                analysis = auditory.get("analysis", {})
                quality = _safe_score(auditory.get("signal_quality_score"))
                confidence = (
                    _safe_score(analysis.get("confidence"))
                    if isinstance(analysis, dict) else 0.0
                )
                reliability = _clamp(quality * 0.80 + confidence * 0.20)
                if reliability >= self.min_observation_reliability and isinstance(analysis, dict):
                    for item in _bounded_items(analysis.get("audio_events"), 3):
                        token = str(item).strip().lower()
                        if token:
                            self._append_hypothesis(
                                candidates, seen, cycle,
                                f"audio_sha256:{auditory.get('sha256', '')}",
                                "prior_auditory_event_observation", "presence", token,
                                "present", reliability,
                            )
        return tuple(candidates)

    @staticmethod
    def _append_hypothesis(
        sink: list[PerceptualHypothesis], seen: set[tuple[str, str]], cycle: CognitiveCycle,
        source_reference: str, source_kind: str, feature_kind: str, feature_name: str,
        predicted_value: str, confidence: float,
    ) -> None:
        key = (feature_kind + ":" + feature_name, predicted_value)
        if key in seen:
            return
        seen.add(key)
        sink.append(PerceptualHypothesis(
            hypothesis_id=uuid5(
                NAMESPACE_URL,
                f"eca:perceptual-hypothesis:v1:{cycle.cycle_id}:{source_kind}:{feature_kind}:{feature_name}:{predicted_value}",
            ),
            source_cycle_id=cycle.cycle_id,
            source_reference=source_reference[:180],
            source_kind=source_kind,
            feature_kind=feature_kind,
            feature_name=feature_name[:96],
            predicted_value=predicted_value[:96],
            prior_confidence=_clamp(confidence),
            reviewable_modalities=("text", "image", "audio"),
        ))

    def _current_observations(
        self, *, sensory_episode: SensoryEpisode, current_text: str,
        visual_evidence: Optional[VisualEvidence], audio_evidence: Optional[AudioEvidence],
    ) -> tuple[_Observation, ...]:
        bindings = {binding.modality: binding for binding in sensory_episode.bindings}
        result: list[_Observation] = []
        if "text" in bindings and _is_assertive(current_text):
            result.extend(self._observations_from_text(
                current_text, "text", bindings["text"].reliability.score,
                bindings["text"].source_reference,
            ))
        if visual_evidence and "image" in bindings:
            analysis = visual_evidence.analysis
            content = " ".join([
                analysis.description, analysis.scene_description, *analysis.objects_detected,
            ])
            result.extend(self._observations_from_text(
                content, "image", bindings["image"].reliability.score,
                bindings["image"].source_reference,
            ))
        if audio_evidence and "audio" in bindings:
            analysis = audio_evidence.analysis
            content = " ".join([analysis.transcription, *analysis.audio_events])
            result.extend(self._observations_from_text(
                content, "audio", bindings["audio"].reliability.score,
                bindings["audio"].source_reference,
            ))
        return tuple(result)

    @staticmethod
    def _observations_from_text(value: str, modality: str, reliability: float, reference: str) -> Iterable[_Observation]:
        for name, observed in _anchored_attributes(value).items():
            yield _Observation("categorical_attribute", name, observed, modality, reliability, reference)
        for token, present in _claims(value).items():
            yield _Observation(
                "presence", token, "present" if present else "absent",
                modality, reliability, reference,
            )

    def _compare(
        self, hypothesis: PerceptualHypothesis, episode_id: UUID,
        observations: Sequence[_Observation],
    ) -> PerceptualPredictionError:
        matches = [
            item for item in observations
            if item.feature_kind == hypothesis.feature_kind and item.feature_name == hypothesis.feature_name
        ]
        observation = max(matches, key=lambda item: item.reliability) if matches else None
        status = "unobserved"
        direction = "zero"
        signed = 0.0
        magnitude = 0.0
        calibration_eligible = False
        material = False
        if observation and observation.reliability < self.min_observation_reliability:
            status = "low_reliability"
        elif observation and observation.value == hypothesis.predicted_value:
            status = "matched"
            calibration_eligible = True
        elif observation:
            status = "mismatch"
            calibration_eligible = True
            magnitude = _clamp(hypothesis.prior_confidence * observation.reliability)
            if hypothesis.feature_kind == "categorical_attribute":
                direction = "categorical_mismatch"
                signed = magnitude
            elif hypothesis.predicted_value == "absent" and observation.value == "present":
                direction = "unexpected_presence"
                signed = magnitude
            else:
                direction = "unexpected_absence"
                signed = -magnitude
            material = magnitude >= self.clarification_threshold
        return PerceptualPredictionError(
            error_id=uuid5(
                NAMESPACE_URL,
                f"eca:prediction-error:v1:{episode_id}:{hypothesis.hypothesis_id}",
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            sensory_episode_id=episode_id,
            feature_kind=hypothesis.feature_kind,
            feature_name=hypothesis.feature_name,
            predicted_value=hypothesis.predicted_value,
            observed_value=observation.value if observation else None,
            observed_modality=observation.modality if observation else None,
            observation_reference=observation.source_reference if observation else None,
            status=status,
            direction=direction,
            signed_error=signed,
            surprise_magnitude=magnitude,
            prior_confidence=hypothesis.prior_confidence,
            observation_reliability=observation.reliability if observation else None,
            calibration_eligible=calibration_eligible,
            material=material,
        )

    def _recommend(
        self, episode: SensoryEpisode,
        errors: Sequence[PerceptualPredictionError],
    ) -> Optional[ClarificationRecommendation]:
        material = sorted(
            (item for item in errors if item.material),
            key=lambda item: (-item.surprise_magnitude, str(item.error_id)),
        )
        if material:
            top = material[0]
            action = "ask_user"
            if top.observed_modality == "image" and (top.observation_reliability or 0.0) < 0.75:
                action = "request_image_recapture"
            elif top.observed_modality == "audio" and (top.observation_reliability or 0.0) < 0.75:
                action = "request_audio_recapture"
            prompt = self._prompt(action, top.feature_name)
            return ClarificationRecommendation(
                recommendation_id=uuid5(NAMESPACE_URL, f"eca:clarification:v1:{episode.episode_id}:prediction"),
                action=action,
                reason="material_prediction_error",
                target_modalities=(top.observed_modality,) if top.observed_modality else (),
                prompt=prompt,
                priority=max(0.7, top.surprise_magnitude),
                expected_information_gain=top.surprise_magnitude,
                source_error_ids=tuple(item.error_id for item in material[:3]),
            )

        low_quality = next((item for item in errors if item.status == "low_reliability"), None)
        if low_quality and low_quality.observed_modality in {"image", "audio"}:
            action = (
                "request_image_recapture" if low_quality.observed_modality == "image"
                else "request_audio_recapture"
            )
            return ClarificationRecommendation(
                recommendation_id=uuid5(NAMESPACE_URL, f"eca:clarification:v1:{episode.episode_id}:quality"),
                action=action,
                reason="low_reliability_prediction_check",
                target_modalities=(low_quality.observed_modality,),
                prompt=self._prompt(action, low_quality.feature_name),
                priority=0.55,
                expected_information_gain=0.55,
                source_error_ids=(low_quality.error_id,),
            )

        conflict_indexes = tuple(
            index for index, relation in enumerate(episode.relations)
            if relation.relation_type == "contradiction" and relation.requires_clarification
        )
        if conflict_indexes:
            modalities = tuple(dict.fromkeys(
                modality for index in conflict_indexes for modality in episode.relations[index].modalities
            ))
            return ClarificationRecommendation(
                recommendation_id=uuid5(NAMESPACE_URL, f"eca:clarification:v1:{episode.episode_id}:cross-modal"),
                action="ask_user",
                reason="unresolved_cross_modal_conflict",
                target_modalities=modalities,
                prompt="I detected conflicting same-turn observations. Could you clarify which observation is current?",
                priority=0.75,
                expected_information_gain=0.75,
                source_relation_indexes=conflict_indexes[:3],
            )
        return None

    @staticmethod
    def _prompt(action: str, feature_name: str) -> str:
        label = feature_name.replace(":", " ")[:80]
        if action == "request_image_recapture":
            return f"The current image is not reliable enough to resolve the mismatch about {label}. Please provide a clearer image."
        if action == "request_audio_recapture":
            return f"The current audio is not reliable enough to resolve the mismatch about {label}. Please record a clearer clip."
        return f"Recent context and the current observation differ about {label}. Could you clarify which is current?"
