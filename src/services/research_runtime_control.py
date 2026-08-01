"""Authenticated runtime controls for the guarded research capability."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional
from uuid import UUID

from src.models.research_models import (
    ResearchLedgerEventType,
    ResearchRuntimeState,
    ResearchRuntimeUpdateRequest,
    utc_now,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.gemini_grounded_research_provider import GeminiGroundedResearchProvider
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.research_service import DisabledResearchProvider, ResearchProvider, ResearchService
from src.services.waking_inquiry_service import WakingInquiryService


logger = logging.getLogger(__name__)
_AUTOMATION_CONFIRMATION = "ENABLE AUTOMATIC RESEARCH"


class ResearchRuntimeControl:
    """Apply interlocked runtime settings and persist every change in the ledger."""

    def __init__(
        self,
        *,
        research_service: ResearchService,
        drive: CognitiveResearchDrive,
        waking_service: WakingInquiryService,
        ledger: ResearchCalibrationLedger,
        api_key: Optional[str],
        model_name: str,
        timeout_seconds: float,
        local_only: bool,
        provider_factory: Optional[Callable[..., ResearchProvider]] = None,
    ) -> None:
        self.research_service = research_service
        self.drive = drive
        self.waking_service = waking_service
        self.ledger = ledger
        self.api_key = api_key
        self.model_name = model_name.strip()
        self.timeout_seconds = timeout_seconds
        self.local_only = local_only
        self._provider_factory = provider_factory or GeminiGroundedResearchProvider
        self._lock = asyncio.Lock()
        self._state = self._current_state()

    @property
    def automation_confirmation(self) -> str:
        return _AUTOMATION_CONFIRMATION

    def get_state(self) -> ResearchRuntimeState:
        provider_configured = bool(self.api_key and self.model_name) and not self.local_only
        return self._state.model_copy(
            update={
                "provider_configured": provider_configured,
                "provider_available": (
                    self.research_service.provider.is_available()
                    if self._state.provider_enabled
                    else provider_configured
                ),
                "provider": (
                    self.research_service.provider.provider_name
                    if self._state.provider_enabled
                    else ("gemini-grounded-search" if provider_configured else "disabled")
                ),
                "model": self.model_name or None,
                "local_only": self.local_only,
            }
        )

    async def restore(self, user_id: UUID) -> ResearchRuntimeState:
        event = await self.ledger.latest_event(
            user_id, ResearchLedgerEventType.RUNTIME_CONTROL_CHANGED
        )
        if event is None:
            return self.get_state()
        try:
            stored = ResearchRuntimeState.model_validate(event.payload["new_state"])
            await self._apply_target(stored, record=False, user_id=user_id, reason="restart restore")
        except Exception as error:
            logger.warning(
                "Research runtime controls stayed fail-safe during restore (%s).",
                type(error).__name__,
            )
        return self.get_state()

    async def update(
        self,
        user_id: UUID,
        request: ResearchRuntimeUpdateRequest,
    ) -> ResearchRuntimeState:
        current = self.get_state()
        if request.emergency_stop:
            target = current.model_copy(
                update={
                    "provider_enabled": False,
                    "controller_active": False,
                    "automatic_non_explicit_enabled": False,
                    "emergency_stop": True,
                    "explicit_approval_required": True,
                }
            )
        else:
            target = current.model_copy(
                update={
                    "provider_enabled": (
                        current.provider_enabled
                        if request.provider_enabled is None
                        else request.provider_enabled
                    ),
                    "controller_active": (
                        current.controller_active
                        if request.controller_active is None
                        else request.controller_active
                    ),
                    "automatic_non_explicit_enabled": (
                        current.automatic_non_explicit_enabled
                        if request.automatic_non_explicit_enabled is None
                        else request.automatic_non_explicit_enabled
                    ),
                    "emergency_stop": False,
                }
            )
            target = target.model_copy(
                update={
                    "explicit_approval_required": not target.automatic_non_explicit_enabled
                }
            )
        if (
            target.automatic_non_explicit_enabled
            and not current.automatic_non_explicit_enabled
            and request.confirmation != _AUTOMATION_CONFIRMATION
        ):
            raise ValueError(
                f"Automatic non-explicit research requires confirmation: {_AUTOMATION_CONFIRMATION}"
            )
        return await self._apply_target(
            target,
            record=True,
            user_id=user_id,
            reason=request.reason,
        )

    async def _apply_target(
        self,
        target: ResearchRuntimeState,
        *,
        record: bool,
        user_id: UUID,
        reason: str,
    ) -> ResearchRuntimeState:
        async with self._lock:
            if target.provider_enabled and self.local_only:
                raise ValueError("Research cannot be enabled while local-only mode is active.")
            if target.provider_enabled and not (self.api_key and self.model_name):
                raise ValueError("Gemini research credentials and model are not configured.")
            if target.controller_active and not target.provider_enabled:
                raise ValueError("Enable the grounded provider before activating the controller.")
            if target.automatic_non_explicit_enabled and not target.controller_active:
                raise ValueError("Activate the controller before enabling non-explicit research.")

            old_provider = self.research_service.provider
            target_provider: ResearchProvider = old_provider
            if target.provider_enabled and isinstance(old_provider, DisabledResearchProvider):
                target_provider = self._provider_factory(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    timeout_seconds=self.timeout_seconds,
                )
                if not target_provider.is_available():
                    await target_provider.close()
                    raise ValueError("The configured Gemini research provider is unavailable.")
            elif not target.provider_enabled and not isinstance(
                old_provider, DisabledResearchProvider
            ):
                target_provider = DisabledResearchProvider()

            applied = target.model_copy(
                update={
                    "provider_configured": bool(self.api_key and self.model_name)
                    and not self.local_only,
                    "provider_available": target_provider.is_available(),
                    "provider": target_provider.provider_name,
                    "model": target_provider.model_name or (self.model_name or None),
                    "local_only": self.local_only,
                    "explicit_approval_required": not target.automatic_non_explicit_enabled,
                    "changed_at": utc_now(),
                }
            )
            try:
                if record:
                    await self.ledger.append(
                        ResearchLedgerEventType.RUNTIME_CONTROL_CHANGED,
                        user_id=user_id,
                        payload={
                            "reason": reason,
                            "previous_state": self.get_state().model_dump(mode="json"),
                            "new_state": applied.model_dump(mode="json"),
                        },
                    )
            except Exception:
                if target_provider is not old_provider:
                    await target_provider.close()
                raise

            self.research_service.provider = target_provider
            self.research_service.policy.research_enabled = target.provider_enabled
            self.drive.enabled = target.controller_active
            self.drive.shadow_mode = not target.controller_active
            self.waking_service.require_user_approval = not target.automatic_non_explicit_enabled
            self._state = applied
            if old_provider is not target_provider and not isinstance(
                old_provider, DisabledResearchProvider
            ):
                await old_provider.close()
            return self.get_state()

    def _current_state(self) -> ResearchRuntimeState:
        provider_enabled = bool(
            self.research_service.policy.research_enabled
            and self.research_service.provider.is_available()
        )
        controller_active = bool(self.drive.enabled and not self.drive.shadow_mode)
        automatic = not self.waking_service.require_user_approval
        return ResearchRuntimeState(
            provider_enabled=provider_enabled,
            controller_active=controller_active,
            automatic_non_explicit_enabled=automatic,
            provider_configured=bool(self.api_key and self.model_name) and not self.local_only,
            provider_available=self.research_service.provider.is_available(),
            provider=self.research_service.provider.provider_name,
            model=self.research_service.provider.model_name or (self.model_name or None),
            local_only=self.local_only,
            explicit_approval_required=not automatic,
        )
