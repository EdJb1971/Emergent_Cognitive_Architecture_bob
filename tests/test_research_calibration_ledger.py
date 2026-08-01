import sqlite3
from uuid import uuid4

import pytest

from src.models.research_models import (
    CalibrationLabelRequest,
    CognitiveEffortAction,
    CognitiveResearchSignals,
    ResearchClaim,
    ResearchLedgerEventType,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchSource,
    SourceQualityFeedbackRequest,
    SourceQualityVerdict,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.research_calibration_ledger import ResearchCalibrationLedger


@pytest.mark.asyncio
async def test_ledger_is_hash_chained_append_only_and_calibrates_shadow_observations(tmp_path):
    path = tmp_path / "inquiries.sqlite3"
    ledger = ResearchCalibrationLedger(path)
    await ledger.connect()
    user_id = uuid4()
    cycle_id = uuid4()
    assessment = CognitiveResearchDrive(enabled=False, shadow_mode=True).assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.95,
            temporal_volatility=0.9,
            task_stakes=0.9,
            expected_information_gain=0.95,
            metacognitive_gap=True,
        ),
        source="real_cycle",
    )

    observation = await ledger.record_assessment(
        assessment,
        user_id=user_id,
        cycle_id=cycle_id,
        event_type=ResearchLedgerEventType.SHADOW_ASSESSMENT,
    )
    label = await ledger.append_calibration_label(
        user_id=user_id,
        assessment_id=assessment.assessment_id,
        label=CalibrationLabelRequest(
            appropriate_action=CognitiveEffortAction.AUTHORIZE_RESEARCH,
            should_external_research=True,
            local_answer_sufficient=False,
            rationale="The fact was current and externally verifiable.",
        ),
    )
    summary = await ledger.calibration_summary(user_id)

    assert label.previous_hash == observation.event_hash
    assert summary["observations"] == 1
    assert summary["labeled_observations"] == 1
    assert summary["external_research_precision"] == 1.0
    assert summary["external_research_recall"] == 1.0
    assert summary["automatic_non_explicit_research_eligible"] is False
    assert summary["ledger_integrity_verified"] is True

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE research_calibration_ledger SET payload = '{}' WHERE sequence = 1"
            )


@pytest.mark.asyncio
async def test_source_feedback_requires_a_source_from_recorded_packet(tmp_path):
    ledger = ResearchCalibrationLedger(tmp_path / "inquiries.sqlite3")
    await ledger.connect()
    user_id = uuid4()
    inquiry_id = uuid4()
    packet = ResearchPacket(
        request_id=uuid4(),
        decision_id=uuid4(),
        query="What changed?",
        status=ResearchPacketStatus.COMPLETED,
        provider="grounded-test",
        grounding_verified=True,
        sources=[ResearchSource(source_id="s1", title="Primary", url="https://example.test")],
        claims=[ResearchClaim(text="It changed.", source_ids=["s1"], confidence=0.9)],
    )
    await ledger.record_packet(packet, user_id=user_id, inquiry_id=inquiry_id)
    feedback = SourceQualityFeedbackRequest(
        request_id=packet.request_id,
        source_id="s1",
        verdict=SourceQualityVerdict.TRUSTWORTHY,
        relevance=5,
        authority=4,
        freshness=5,
        citation_support=5,
        claim_supported=True,
        research_changed_answer=True,
        worth_cost=True,
    )

    event = await ledger.append_source_feedback(
        user_id=user_id,
        inquiry_id=inquiry_id,
        feedback=feedback,
    )
    summary = await ledger.calibration_summary(user_id)

    assert event.event_type == ResearchLedgerEventType.SOURCE_FEEDBACK
    assert summary["source_feedback_count"] == 1
    assert summary["source_quality_averages"]["authority"] == 4.0
    assert summary["source_claim_support_rate"] == 1.0
    assert summary["research_changed_answer_rate"] == 1.0
    assert summary["research_worth_cost_rate"] == 1.0

    with pytest.raises(KeyError, match="not part"):
        await ledger.append_source_feedback(
            user_id=user_id,
            inquiry_id=inquiry_id,
            feedback=feedback.model_copy(update={"source_id": "unknown"}),
        )
