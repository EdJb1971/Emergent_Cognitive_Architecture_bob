from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models.multimodal_models import (
    AudioAnalysis,
    AudioEvidence,
    VisualAnalysis,
    VisualEvidence,
)
from src.services.multisensory_binding_service import MultisensoryBindingService


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _visual(description: str, *, quality: float = 0.8, confidence: float = 0.9,
            observed_at: datetime = NOW) -> VisualEvidence:
    return VisualEvidence(
        provider="ollama", model="gemma4:e4b", mime_type="image/png",
        byte_count=100, width=640, height=480, input_quality_score=quality,
        sha256="a" * 64, observed_at=observed_at,
        analysis=VisualAnalysis(
            description=description, objects_detected=[],
            scene_description=description, confidence=confidence,
        ),
    )


def _audio(transcript: str, *, quality: float = 0.9, confidence: float = 0.8,
           observed_at: datetime = NOW) -> AudioEvidence:
    return AudioEvidence(
        provenance="live_microphone_capture", provider="ollama", model="gemma4:e4b",
        byte_count=32044, duration_seconds=1.0, sample_rate_hz=16000,
        channels=1, bits_per_sample=16, signal_quality_score=quality,
        sha256="b" * 64, observed_at=observed_at,
        analysis=AudioAnalysis(
            speech_detected=bool(transcript), transcription=transcript,
            audio_events=["speech"] if transcript else ["silence"], confidence=confidence,
        ),
    )


def _bind(*, text: str, visual=None, audio=None, service=None):
    return (service or MultisensoryBindingService()).bind_turn(
        cycle_id=uuid4(), user_id=uuid4(), session_id=uuid4(),
        request_timestamp=NOW, text=text,
        visual_evidence=visual, audio_evidence=audio,
    )


def test_episode_is_immutable_typed_and_references_primary_evidence_without_copying_media():
    visual_evidence = _visual("A red car is parked")
    audio_evidence = _audio("The red car is parked")
    visual_before = visual_evidence.model_dump_json()
    audio_before = audio_evidence.model_dump_json()
    episode = _bind(
        text="The red car is parked",
        visual=visual_evidence,
        audio=audio_evidence,
    )

    assert episode.schema_version == "sensory-episode-v1"
    assert episode.modalities == ("text", "image", "audio")
    assert episode.raw_media_retained is False
    assert episode.generative_fusion_performed is False
    assert episode.attention.routing_changes_applied is False
    assert episode.attention.primary_evidence_rewritten is False
    assert episode.primary_evidence_references[1] == "image_sha256:" + "a" * 64
    assert episode.primary_evidence_references[2] == "audio_sha256:" + "b" * 64
    with pytest.raises(ValidationError):
        episode.attention.overall_priority = 1.0
    with pytest.raises(ValidationError):
        episode.bindings[0].reliability.score = 0.0
    invalid = episode.model_dump()
    invalid["primary_evidence_references"] = ("text_sha256:" + "0" * 64,)
    with pytest.raises(ValidationError):
        type(episode).model_validate(invalid)
    assert visual_evidence.model_dump_json() == visual_before
    assert audio_evidence.model_dump_json() == audio_before


def test_matching_claims_create_bounded_agreement_and_corroboration_cue():
    episode = _bind(
        text="The red car is parked",
        visual=_visual("A red car is parked beside the road"),
    )

    relation = episode.relations[0]
    assert relation.relation_type == "agreement"
    assert {"red", "car", "parked"}.issubset(relation.anchors)
    assert relation.strength <= relation.reliability_ceiling
    assert episode.attention.agreement_detected is True
    assert episode.attention.contradiction_detected is False
    assert any(cue.cue_type == "cross_modal_corroboration" for cue in episode.attention.cues)


def test_opposed_claim_polarity_requests_clarification_without_selecting_a_winner():
    episode = _bind(
        text="There is no dog in the room",
        visual=_visual("A dog is in the room"),
    )

    relation = episode.relations[0]
    assert relation.relation_type == "contradiction"
    assert relation.basis == "opposed_claim_polarity"
    assert "dog" in relation.anchors
    assert relation.requires_clarification is True
    cue = next(cue for cue in episode.attention.cues if cue.cue_type == "cross_modal_conflict")
    assert cue.recommended_action == "ask_for_clarification"
    assert cue.advisory_only is True


def test_conflicting_attribute_requires_a_shared_non_attribute_anchor():
    conflict = _bind(
        text="The car is red",
        visual=_visual("The car is blue"),
    )
    unrelated = _bind(
        text="A red warning appears",
        visual=_visual("A blue lake shimmers"),
    )

    assert conflict.relations[0].basis == "conflicting_categorical_attribute"
    assert conflict.relations[0].relation_type == "contradiction"
    assert unrelated.relations[0].relation_type != "contradiction"


def test_measured_quality_dominates_model_confidence_and_drives_low_reliability_attention():
    episode = _bind(
        text="What is in the image?",
        visual=_visual("A bicycle", quality=0.1, confidence=1.0),
    )
    visual = next(binding for binding in episode.bindings if binding.modality == "image")

    assert visual.reliability.quality_weight == 0.75
    assert visual.reliability.confidence_weight == 0.25
    assert visual.reliability.score == 0.325
    assert any(cue.cue_type == "low_reliability" for cue in episode.attention.cues)


def test_out_of_window_observation_is_not_semantically_fused():
    episode = _bind(
        text="A dog is here",
        visual=_visual("A dog is here", observed_at=NOW + timedelta(seconds=20)),
        service=MultisensoryBindingService(max_alignment_skew_seconds=5),
    )

    assert episode.relations[0].relation_type == "insufficient_evidence"
    assert episode.relations[0].basis == "outside_temporal_window"
    assert any(cue.cue_type == "temporal_misalignment" for cue in episode.attention.cues)


def test_unrelated_modalities_remain_insufficient_instead_of_hallucinating_a_relation():
    episode = _bind(
        text="Please summarize the meeting",
        audio=_audio("birds chirping outside"),
    )

    assert episode.relations[0].relation_type == "insufficient_evidence"
    assert episode.relations[0].basis == "no_stable_shared_anchor"
    assert episode.attention.agreement_detected is False
    assert episode.attention.contradiction_detected is False
