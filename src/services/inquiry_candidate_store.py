"""Durable local queue for waking and dream-generated inquiry candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence
from uuid import UUID

from src.models.research_models import (
    CognitiveResearchAssessment,
    InquiryCandidate,
    InquiryStatus,
)


_OPEN_STATUSES = (
    InquiryStatus.QUEUED,
    InquiryStatus.UNDER_REVIEW,
    InquiryStatus.APPROVED,
    InquiryStatus.RESEARCH_FAILED,
)

_ALLOWED_TRANSITIONS = {
    InquiryStatus.QUEUED: {
        InquiryStatus.UNDER_REVIEW,
        InquiryStatus.RESOLVED_LOCALLY,
        InquiryStatus.APPROVED,
        InquiryStatus.DISMISSED,
        InquiryStatus.EXPIRED,
    },
    InquiryStatus.UNDER_REVIEW: {
        InquiryStatus.QUEUED,
        InquiryStatus.RESOLVED_LOCALLY,
        InquiryStatus.APPROVED,
        InquiryStatus.DISMISSED,
        InquiryStatus.EXPIRED,
    },
    InquiryStatus.APPROVED: {
        InquiryStatus.QUEUED,
        InquiryStatus.RESEARCHED,
        InquiryStatus.RESEARCH_FAILED,
        InquiryStatus.RESOLVED_LOCALLY,
        InquiryStatus.DISMISSED,
        InquiryStatus.EXPIRED,
    },
    InquiryStatus.RESEARCH_FAILED: {
        InquiryStatus.QUEUED,
        InquiryStatus.APPROVED,
        InquiryStatus.DISMISSED,
        InquiryStatus.EXPIRED,
    },
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InquiryCandidateStore:
    """SQLite-backed queue with atomic state transitions and active-item deduplication."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def enqueue(self, candidate: InquiryCandidate) -> tuple[InquiryCandidate, bool]:
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(self._enqueue_sync, candidate)

    async def get(self, inquiry_id: UUID, user_id: UUID) -> Optional[InquiryCandidate]:
        self._require_connected()
        return await asyncio.to_thread(self._get_sync, inquiry_id, user_id)

    async def list_candidates(
        self,
        user_id: UUID,
        *,
        statuses: Optional[Sequence[InquiryStatus]] = None,
        limit: int = 50,
    ) -> list[InquiryCandidate]:
        self._require_connected()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")
        return await asyncio.to_thread(self._list_sync, user_id, statuses, limit)

    async def transition(
        self,
        inquiry_id: UUID,
        user_id: UUID,
        new_status: InquiryStatus,
        *,
        resolution: Optional[str] = None,
    ) -> InquiryCandidate:
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._transition_sync,
                inquiry_id,
                user_id,
                new_status,
                resolution,
            )

    async def record_review(
        self,
        inquiry_id: UUID,
        user_id: UUID,
        assessment: CognitiveResearchAssessment,
        new_status: InquiryStatus,
        *,
        resolution: Optional[str] = None,
    ) -> InquiryCandidate:
        """Persist a fresh waking assessment and its state transition atomically."""
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._record_review_sync,
                inquiry_id,
                user_id,
                assessment,
                new_status,
                resolution,
            )

    async def claim_next(
        self,
        user_id: UUID,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[InquiryCandidate]:
        """Atomically reserve the highest-priority live candidate for waking review."""
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(self._claim_next_sync, user_id, now or _utc_now())

    async def expire_due(self, *, now: Optional[datetime] = None) -> int:
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(self._expire_due_sync, now or _utc_now())

    @staticmethod
    def fingerprint(question: str) -> str:
        normalized = re.sub(r"\s+", " ", question.strip().casefold())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inquiry_candidates (
                    inquiry_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inquiry_user_status_priority "
                "ON inquiry_candidates(user_id, status, priority DESC, created_at ASC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inquiry_fingerprint "
                "ON inquiry_candidates(user_id, fingerprint)"
            )

    def _enqueue_sync(self, candidate: InquiryCandidate) -> tuple[InquiryCandidate, bool]:
        fingerprint = candidate.fingerprint or self.fingerprint(candidate.question)
        candidate = candidate.model_copy(update={"fingerprint": fingerprint, "updated_at": _utc_now()})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in _OPEN_STATUSES)
            row = connection.execute(
                f"SELECT payload FROM inquiry_candidates WHERE user_id = ? AND fingerprint = ? "
                f"AND status IN ({placeholders}) ORDER BY created_at ASC LIMIT 1",
                (str(candidate.user_id), fingerprint, *(status.value for status in _OPEN_STATUSES)),
            ).fetchone()
            if row:
                existing = InquiryCandidate.model_validate_json(row[0])
                merged = self._merge(existing, candidate)
                self._update_row(connection, merged)
                connection.commit()
                return merged, False

            connection.execute(
                "INSERT INTO inquiry_candidates "
                "(inquiry_id, user_id, fingerprint, status, priority, created_at, updated_at, expires_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._row_values(candidate),
            )
            connection.commit()
            return candidate, True

    def _get_sync(self, inquiry_id: UUID, user_id: UUID) -> Optional[InquiryCandidate]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM inquiry_candidates WHERE inquiry_id = ? AND user_id = ?",
                (str(inquiry_id), str(user_id)),
            ).fetchone()
        return InquiryCandidate.model_validate_json(row[0]) if row else None

    def _list_sync(
        self,
        user_id: UUID,
        statuses: Optional[Sequence[InquiryStatus]],
        limit: int,
    ) -> list[InquiryCandidate]:
        parameters: list[object] = [str(user_id)]
        where = "user_id = ?"
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            parameters.extend(status.value for status in statuses)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM inquiry_candidates WHERE {where} "
                "ORDER BY priority DESC, created_at ASC LIMIT ?",
                parameters,
            ).fetchall()
        return [InquiryCandidate.model_validate_json(row[0]) for row in rows]

    def _transition_sync(
        self,
        inquiry_id: UUID,
        user_id: UUID,
        new_status: InquiryStatus,
        resolution: Optional[str],
    ) -> InquiryCandidate:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM inquiry_candidates WHERE inquiry_id = ? AND user_id = ?",
                (str(inquiry_id), str(user_id)),
            ).fetchone()
            if not row:
                raise KeyError(f"Inquiry candidate {inquiry_id} was not found.")
            candidate = InquiryCandidate.model_validate_json(row[0])
            allowed = _ALLOWED_TRANSITIONS.get(candidate.status, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Inquiry status cannot transition from {candidate.status.value} to {new_status.value}."
                )
            candidate = candidate.model_copy(
                update={
                    "status": new_status,
                    "resolution": resolution,
                    "updated_at": _utc_now(),
                }
            )
            self._update_row(connection, candidate)
            connection.commit()
            return candidate

    def _claim_next_sync(self, user_id: UUID, now: datetime) -> Optional[InquiryCandidate]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM inquiry_candidates "
                "WHERE user_id = ? AND status = ? AND expires_at > ? "
                "ORDER BY priority DESC, created_at ASC LIMIT 1",
                (str(user_id), InquiryStatus.QUEUED.value, now.isoformat()),
            ).fetchone()
            if not row:
                connection.commit()
                return None
            candidate = InquiryCandidate.model_validate_json(row[0]).model_copy(
                update={
                    "status": InquiryStatus.UNDER_REVIEW,
                    "updated_at": now,
                }
            )
            self._update_row(connection, candidate)
            connection.commit()
            return candidate

    def _record_review_sync(
        self,
        inquiry_id: UUID,
        user_id: UUID,
        assessment: CognitiveResearchAssessment,
        new_status: InquiryStatus,
        resolution: Optional[str],
    ) -> InquiryCandidate:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM inquiry_candidates WHERE inquiry_id = ? AND user_id = ?",
                (str(inquiry_id), str(user_id)),
            ).fetchone()
            if not row:
                raise KeyError(f"Inquiry candidate {inquiry_id} was not found.")
            candidate = InquiryCandidate.model_validate_json(row[0])
            allowed = _ALLOWED_TRANSITIONS.get(candidate.status, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Inquiry status cannot transition from {candidate.status.value} "
                    f"to {new_status.value}."
                )
            candidate = candidate.model_copy(
                update={
                    "assessment": assessment,
                    "priority": assessment.drive_score,
                    "expected_information_gain": assessment.signals.expected_information_gain,
                    "shadow_mode": assessment.shadow_mode,
                    "status": new_status,
                    "resolution": resolution,
                    "updated_at": _utc_now(),
                }
            )
            self._update_row(connection, candidate)
            connection.commit()
            return candidate

    def _expire_due_sync(self, now: datetime) -> int:
        expired = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in _OPEN_STATUSES)
            rows = connection.execute(
                f"SELECT payload FROM inquiry_candidates WHERE status IN ({placeholders}) "
                "AND expires_at <= ?",
                (*(status.value for status in _OPEN_STATUSES), now.isoformat()),
            ).fetchall()
            for row in rows:
                candidate = InquiryCandidate.model_validate_json(row[0]).model_copy(
                    update={
                        "status": InquiryStatus.EXPIRED,
                        "updated_at": now,
                        "resolution": "Expired before waking research authorization.",
                    }
                )
                self._update_row(connection, candidate)
                expired += 1
            connection.commit()
        return expired

    @staticmethod
    def _merge(existing: InquiryCandidate, incoming: InquiryCandidate) -> InquiryCandidate:
        cycle_ids = list(dict.fromkeys([*existing.source_cycle_ids, *incoming.source_cycle_ids]))
        pattern_ids = list(dict.fromkeys([*existing.source_pattern_ids, *incoming.source_pattern_ids]))
        assessment = (
            incoming.assessment
            if incoming.assessment.drive_score > existing.assessment.drive_score
            else existing.assessment
        )
        return existing.model_copy(
            update={
                "source_cycle_ids": cycle_ids,
                "source_pattern_ids": pattern_ids,
                "assessment": assessment,
                "status": (
                    InquiryStatus.QUEUED
                    if existing.status == InquiryStatus.RESEARCH_FAILED
                    else existing.status
                ),
                "priority": max(existing.priority, incoming.priority),
                "expected_information_gain": max(
                    existing.expected_information_gain,
                    incoming.expected_information_gain,
                ),
                "shadow_mode": existing.shadow_mode and incoming.shadow_mode,
                "updated_at": _utc_now(),
                "expires_at": max(existing.expires_at, incoming.expires_at),
                "metadata": {**existing.metadata, **incoming.metadata},
            }
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @staticmethod
    def _row_values(candidate: InquiryCandidate) -> tuple[object, ...]:
        return (
            str(candidate.inquiry_id),
            str(candidate.user_id),
            candidate.fingerprint,
            candidate.status.value,
            candidate.priority,
            candidate.created_at.isoformat(),
            candidate.updated_at.isoformat(),
            candidate.expires_at.isoformat(),
            candidate.model_dump_json(),
        )

    @staticmethod
    def _update_row(connection: sqlite3.Connection, candidate: InquiryCandidate) -> None:
        connection.execute(
            "UPDATE inquiry_candidates SET status = ?, priority = ?, updated_at = ?, "
            "expires_at = ?, payload = ? WHERE inquiry_id = ? AND user_id = ?",
            (
                candidate.status.value,
                candidate.priority,
                candidate.updated_at.isoformat(),
                candidate.expires_at.isoformat(),
                candidate.model_dump_json(),
                str(candidate.inquiry_id),
                str(candidate.user_id),
            ),
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("InquiryCandidateStore.connect() must be awaited before use.")
