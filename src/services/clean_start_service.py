"""Restart-boundary deletion for disposable local cognitive state."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.models.identity_models import CleanStartStatus


CLEAN_START_CONFIRMATION = "RESET COGNITIVE MEMORY"


class CleanStartService:
    def __init__(self, marker_path: str | Path, cognitive_data_path: str | Path) -> None:
        self.marker_path = Path(marker_path)
        self.cognitive_data_path = Path(cognitive_data_path)

    def status(self) -> CleanStartStatus:
        if not self.marker_path.exists():
            return CleanStartStatus(
                pending_restart=False,
                message="No clean start is pending.",
            )
        payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        return CleanStartStatus(
            pending_restart=True,
            preserve_identity=bool(payload.get("preserve_identity", True)),
            requested_at=payload.get("requested_at"),
            message="Cognitive memory will be cleared on the next backend restart.",
        )

    def arm(self, *, confirmation: str, preserve_identity: bool) -> CleanStartStatus:
        if confirmation != CLEAN_START_CONFIRMATION:
            raise ValueError(f'Type "{CLEAN_START_CONFIRMATION}" exactly to continue.')
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "preserve_identity": preserve_identity,
        }
        temporary = self.marker_path.with_suffix(self.marker_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.marker_path)
        return self.status()

    def cancel(self) -> CleanStartStatus:
        self.marker_path.unlink(missing_ok=True)
        return self.status()

    def consume_before_startup(self, *, identity_path: str | Path) -> bool:
        """Clear the exact configured data root before any database opens it."""
        if not self.marker_path.exists():
            return False
        payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        target = self.cognitive_data_path.resolve()
        cwd = Path.cwd().resolve()
        home = Path.home().resolve()
        filesystem_root = Path(target.anchor).resolve()
        if target in {cwd, home, filesystem_root} or len(target.parts) < 3:
            raise RuntimeError(f"Refusing unsafe cognitive data reset target: {target}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        if not bool(payload.get("preserve_identity", True)):
            Path(identity_path).unlink(missing_ok=True)
        self.marker_path.unlink(missing_ok=True)
        return True
