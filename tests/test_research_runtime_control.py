from uuid import uuid4

import pytest

from src.models.research_models import ResearchLedgerEventType, ResearchRuntimeUpdateRequest
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.escalation_policy import EscalationPolicy
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.research_runtime_control import ResearchRuntimeControl
from src.services.research_service import DisabledResearchProvider, ResearchService
from src.services.waking_inquiry_service import WakingInquiryService


class _AvailableResearchProvider:
    provider_name = "gemini-grounded-search"
    model_name = "gemini-test"

    def is_available(self):
        return True

    async def close(self):
        return None


async def _control(path):
    store = InquiryCandidateStore(path)
    ledger = ResearchCalibrationLedger(path)
    await store.connect()
    await ledger.connect()
    drive = CognitiveResearchDrive(enabled=False, shadow_mode=True)
    research = ResearchService(EscalationPolicy(research_enabled=False), DisabledResearchProvider())
    waking = WakingInquiryService(store, drive, research, ledger=ledger)
    return ResearchRuntimeControl(
        research_service=research,
        drive=drive,
        waking_service=waking,
        ledger=ledger,
        api_key="test-key",
        model_name="gemini-3.5-flash-lite",
        timeout_seconds=5,
        local_only=False,
        provider_factory=lambda **_: _AvailableResearchProvider(),
    ), ledger


@pytest.mark.asyncio
async def test_runtime_controls_are_interlocked_audited_and_emergency_stoppable(tmp_path):
    control, ledger = await _control(tmp_path / "inquiries.sqlite3")
    user_id = uuid4()

    with pytest.raises(ValueError, match="grounded provider"):
        await control.update(
            user_id,
            ResearchRuntimeUpdateRequest(controller_active=True, reason="activate"),
        )
    provider = await control.update(
        user_id,
        ResearchRuntimeUpdateRequest(provider_enabled=True, reason="enable provider"),
    )
    controller = await control.update(
        user_id,
        ResearchRuntimeUpdateRequest(controller_active=True, reason="leave shadow"),
    )
    with pytest.raises(ValueError, match="requires confirmation"):
        await control.update(
            user_id,
            ResearchRuntimeUpdateRequest(
                automatic_non_explicit_enabled=True,
                reason="enable automation",
            ),
        )
    automatic = await control.update(
        user_id,
        ResearchRuntimeUpdateRequest(
            automatic_non_explicit_enabled=True,
            reason="calibration reviewed",
            confirmation="ENABLE AUTOMATIC RESEARCH",
        ),
    )
    stopped = await control.update(
        user_id,
        ResearchRuntimeUpdateRequest(emergency_stop=True, reason="operator stop"),
    )

    assert provider.provider_enabled is True
    assert controller.controller_active is True
    assert automatic.automatic_non_explicit_enabled is True
    assert automatic.explicit_approval_required is False
    assert stopped.emergency_stop is True
    assert stopped.provider_enabled is False
    assert stopped.controller_active is False
    assert stopped.automatic_non_explicit_enabled is False
    events = await ledger.list_events(
        user_id, event_types=[ResearchLedgerEventType.RUNTIME_CONTROL_CHANGED]
    )
    assert len(events) == 4


@pytest.mark.asyncio
async def test_runtime_controls_restore_last_ledger_state(tmp_path):
    path = tmp_path / "inquiries.sqlite3"
    user_id = uuid4()
    first, _ = await _control(path)
    await first.update(
        user_id,
        ResearchRuntimeUpdateRequest(provider_enabled=True, reason="enable provider"),
    )
    await first.update(
        user_id,
        ResearchRuntimeUpdateRequest(controller_active=True, reason="activate controller"),
    )
    await first.research_service.close()

    restored, _ = await _control(path)
    state = await restored.restore(user_id)

    assert state.provider_enabled is True
    assert state.controller_active is True
    assert state.automatic_non_explicit_enabled is False
    await restored.research_service.close()
