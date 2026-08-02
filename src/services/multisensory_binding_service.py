"""Deterministic same-turn multisensory temporal binding.

The service derives relationships and advisory attention from bounded evidence. It
never receives raw media, invokes a model, changes routing, or mutates a primary
evidence object.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Tuple
from uuid import NAMESPACE_URL, UUID, uuid5

from src.models.multimodal_models import (
    AudioEvidence,
    CrossModalRelation,
    ModalityReliability,
    SensoryAttentionAdvisory,
    SensoryAttentionCue,
    SensoryBinding,
    SensoryEpisode,
    VisualEvidence,
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
_ATTRIBUTE_GROUPS: Mapping[str, frozenset[str]] = {
    "colour": frozenset({
        "black", "blue", "brown", "gray", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow",
    }),
    "state": frozenset({"awake", "closed", "empty", "full", "open", "sleeping"}),
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _round(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


class MultisensoryBindingService:
    """Build immutable, conservative, advisory-only sensory episodes."""

    def __init__(self, *, max_alignment_skew_seconds: float = 120.0):
        if max_alignment_skew_seconds < 0:
            raise ValueError("max_alignment_skew_seconds cannot be negative")
        self.max_alignment_skew_ms = int(max_alignment_skew_seconds * 1000)

    def bind_turn(
        self,
        *,
        cycle_id: UUID,
        user_id: UUID,
        session_id: UUID,
        request_timestamp: datetime,
        text: str,
        visual_evidence: Optional[VisualEvidence] = None,
        audio_evidence: Optional[AudioEvidence] = None,
    ) -> SensoryEpisode:
        request_time = _utc(request_timestamp)
        records: list[tuple[SensoryBinding, Dict[str, bool], Dict[str, str]]] = []

        if text.strip() and not text.lstrip().startswith("["):
            reference = "text_sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append((
                self._binding(
                    modality="text",
                    reference=reference,
                    provenance="direct_user_text",
                    trust="user_authored_primary_input",
                    start=request_time,
                    end=request_time,
                    request_time=request_time,
                    reliability=ModalityReliability(
                        modality="text", score=1.0, measured_quality=1.0,
                        model_confidence=None, quality_weight=1.0, confidence_weight=0.0,
                        factors=("direct_user_text_transport",),
                        limitations=("semantic_truth_not_verified",),
                    ),
                    uncertainty=(),
                ),
                self._claims(text),
                self._attributes(text),
            ))

        if visual_evidence:
            analysis = visual_evidence.analysis
            quality = visual_evidence.input_quality_score
            confidence = analysis.confidence
            score = _round((quality * 0.75) + (confidence * 0.25))
            text_parts = [analysis.description, analysis.scene_description, analysis.ocr_text or ""]
            text_parts.extend(analysis.objects_detected)
            observed = _utc(visual_evidence.observed_at)
            uncertainty = tuple(visual_evidence.quality_warnings)
            records.append((
                self._binding(
                    modality="image",
                    reference=f"image_sha256:{visual_evidence.sha256}",
                    provenance=visual_evidence.provenance,
                    trust=visual_evidence.trust_classification,
                    start=observed,
                    end=observed,
                    request_time=request_time,
                    reliability=ModalityReliability(
                        modality="image", score=score, measured_quality=quality,
                        model_confidence=confidence, quality_weight=0.75,
                        confidence_weight=0.25,
                        factors=("decoded_image_quality", "bounded_local_observation"),
                        limitations=tuple(visual_evidence.quality_warnings) +
                            ("model_description_not_ground_truth",),
                    ),
                    uncertainty=uncertainty,
                ),
                self._claims(" ".join(text_parts)),
                self._attributes(" ".join(text_parts)),
            ))

        if audio_evidence:
            analysis = audio_evidence.analysis
            quality = audio_evidence.signal_quality_score
            confidence = analysis.confidence if audio_evidence.inference_performed else None
            score = _round(
                (quality * 0.80) + ((confidence or 0.0) * 0.20)
                if audio_evidence.inference_performed else quality
            )
            end = _utc(audio_evidence.observed_at)
            start = end - timedelta(seconds=audio_evidence.duration_seconds)
            text_parts = [analysis.transcription, *analysis.audio_events]
            uncertainty = tuple(audio_evidence.quality_warnings) + tuple(analysis.uncertainties)
            records.append((
                self._binding(
                    modality="audio",
                    reference=f"audio_sha256:{audio_evidence.sha256}",
                    provenance=audio_evidence.provenance,
                    trust=audio_evidence.trust_classification,
                    start=start,
                    end=end,
                    request_time=request_time,
                    reliability=ModalityReliability(
                        modality="audio", score=score, measured_quality=quality,
                        model_confidence=confidence, quality_weight=(0.80 if confidence is not None else 1.0),
                        confidence_weight=(0.20 if confidence is not None else 0.0),
                        factors=("decoded_signal_quality", "bounded_local_observation"),
                        limitations=uncertainty + ("transcript_and_event_labels_not_ground_truth",),
                    ),
                    uncertainty=uncertainty,
                ),
                self._claims(" ".join(text_parts)),
                self._attributes(" ".join(text_parts)),
            ))

        bindings = tuple(item[0] for item in records)
        relations = tuple(self._relations(records))
        cues = self._attention_cues(bindings, relations)
        focus = tuple(dict.fromkeys(modality for cue in cues for modality in cue.modalities))
        priority = max((cue.priority for cue in cues), default=0.0)
        starts = [binding.observed_start for binding in bindings] or [request_time]
        ends = [binding.observed_end for binding in bindings] or [request_time]

        return SensoryEpisode(
            episode_id=uuid5(NAMESPACE_URL, f"eca:sensory-episode:v1:{cycle_id}"),
            cycle_id=cycle_id,
            user_id=user_id,
            session_id=session_id,
            captured_at=datetime.now(timezone.utc),
            window_start=min(starts),
            window_end=max(ends),
            max_alignment_skew_ms=self.max_alignment_skew_ms,
            modalities=tuple(binding.modality for binding in bindings),
            bindings=bindings,
            relations=relations,
            attention=SensoryAttentionAdvisory(
                overall_priority=priority,
                focus_modalities=focus,
                cues=cues,
                agreement_detected=any(r.relation_type == "agreement" for r in relations),
                contradiction_detected=any(r.relation_type == "contradiction" for r in relations),
            ),
            primary_evidence_references=tuple(binding.source_reference for binding in bindings),
        )

    def _binding(self, *, modality: str, reference: str, provenance: str, trust: str,
                 start: datetime, end: datetime, request_time: datetime,
                 reliability: ModalityReliability, uncertainty: Tuple[str, ...]) -> SensoryBinding:
        start_offset = int((start - request_time).total_seconds() * 1000)
        end_offset = int((end - request_time).total_seconds() * 1000)
        aligned = min(abs(start_offset), abs(end_offset)) <= self.max_alignment_skew_ms
        return SensoryBinding(
            modality=modality, source_reference=reference, provenance=provenance,
            trust_classification=trust, observed_start=start, observed_end=end,
            offset_start_ms=start_offset, offset_end_ms=end_offset,
            temporally_aligned=aligned, reliability=reliability,
            uncertainty_markers=tuple(dict.fromkeys(uncertainty))[:16],
        )

    def _relations(self, records: list[tuple[SensoryBinding, Dict[str, bool], Dict[str, str]]]) -> Iterable[CrossModalRelation]:
        for left, right in combinations(records, 2):
            a, a_claims, a_attrs = left
            b, b_claims, b_attrs = right
            modalities = (a.modality, b.modality)
            ceiling = _round(min(a.reliability.score, b.reliability.score))
            if not a.temporally_aligned or not b.temporally_aligned:
                yield CrossModalRelation(
                    relation_type="insufficient_evidence", modalities=modalities,
                    basis="outside_temporal_window", strength=0.0,
                    reliability_ceiling=ceiling,
                    explanation="Observations fall outside the configured same-turn alignment window.",
                )
                continue

            shared = sorted(set(a_claims) & set(b_claims))
            opposed = [token for token in shared if a_claims[token] != b_claims[token]]
            attribute_tokens = set().union(*_ATTRIBUTE_GROUPS.values())
            stable_shared = [token for token in shared if token not in attribute_tokens]
            attr_conflicts = sorted(
                f"{group}:{a_attrs[group]}|{b_attrs[group]}"
                for group in set(a_attrs) & set(b_attrs)
                if a_attrs[group] != b_attrs[group] and stable_shared
            )
            if opposed or attr_conflicts:
                anchors = tuple((opposed + attr_conflicts)[:8])
                basis = "opposed_claim_polarity" if opposed else "conflicting_categorical_attribute"
                yield CrossModalRelation(
                    relation_type="contradiction", modalities=modalities, anchors=anchors,
                    basis=basis, strength=_round(ceiling * min(1.0, 0.65 + 0.1 * len(anchors))),
                    reliability_ceiling=ceiling, requires_clarification=True,
                    explanation="Bounded observations make incompatible claims about shared anchors; preserve both and clarify.",
                )
            elif shared:
                anchors = tuple(shared[:8])
                yield CrossModalRelation(
                    relation_type="agreement", modalities=modalities, anchors=anchors,
                    basis="shared_claim_polarity",
                    strength=_round(ceiling * min(1.0, 0.45 + 0.1 * len(anchors))),
                    reliability_ceiling=ceiling,
                    explanation="Bounded observations independently share claim anchors with matching polarity.",
                )
            else:
                yield CrossModalRelation(
                    relation_type="insufficient_evidence", modalities=modalities,
                    basis="no_stable_shared_anchor", strength=0.0,
                    reliability_ceiling=ceiling,
                    explanation="No stable shared claim anchor was found; no agreement or contradiction is inferred.",
                )

    def _attention_cues(self, bindings: Tuple[SensoryBinding, ...], relations: Tuple[CrossModalRelation, ...]) -> Tuple[SensoryAttentionCue, ...]:
        cues: list[SensoryAttentionCue] = []
        for relation in relations:
            if relation.relation_type == "contradiction":
                cues.append(SensoryAttentionCue(
                    cue_type="cross_modal_conflict", priority=_round(max(0.75, relation.strength)),
                    modalities=relation.modalities,
                    reasons=(relation.basis, *relation.anchors),
                    recommended_action="ask_for_clarification",
                ))
            elif relation.relation_type == "agreement" and relation.strength >= 0.35:
                cues.append(SensoryAttentionCue(
                    cue_type="cross_modal_corroboration", priority=_round(0.35 + relation.strength * 0.35),
                    modalities=relation.modalities,
                    reasons=(relation.basis, *relation.anchors),
                    recommended_action="mention_corroboration_cautiously",
                ))
        for binding in bindings:
            if not binding.temporally_aligned:
                cues.append(SensoryAttentionCue(
                    cue_type="temporal_misalignment", priority=0.8,
                    modalities=(binding.modality,), reasons=("outside_same_turn_window",),
                    recommended_action="treat_as_separate_observations",
                ))
            if binding.reliability.score < 0.55:
                cues.append(SensoryAttentionCue(
                    cue_type="low_reliability", priority=_round(1.0 - binding.reliability.score),
                    modalities=(binding.modality,),
                    reasons=("measured_quality_below_reliability_threshold", *binding.uncertainty_markers[:4]),
                    recommended_action="avoid_relying_on_low_quality_detail",
                ))
            elif binding.uncertainty_markers:
                cues.append(SensoryAttentionCue(
                    cue_type="preserve_uncertainty", priority=0.45,
                    modalities=(binding.modality,), reasons=binding.uncertainty_markers[:5],
                    recommended_action="state_uncertainty",
                ))
        return tuple(sorted(cues, key=lambda cue: (-cue.priority, cue.cue_type, cue.modalities)))

    @staticmethod
    def _claims(value: str) -> Dict[str, bool]:
        tokens = _TOKEN.findall(value.lower())
        claims: Dict[str, bool] = {}
        negation_pending = False
        for token in tokens:
            if token in _NEGATIONS:
                negation_pending = True
                continue
            if token in _STOP:
                continue
            claims.setdefault(token, not negation_pending)
            negation_pending = False
        return claims

    @staticmethod
    def _attributes(value: str) -> Dict[str, str]:
        tokens = set(_TOKEN.findall(value.lower()))
        result: Dict[str, str] = {}
        for group, vocabulary in _ATTRIBUTE_GROUPS.items():
            found = sorted(tokens & vocabulary)
            if len(found) == 1:
                result[group] = found[0]
        return result
