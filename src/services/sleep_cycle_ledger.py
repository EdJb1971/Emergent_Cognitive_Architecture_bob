"""Durable append-only audit ledger for sleep and consolidation work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from src.models.sleep_models import SleepLedgerEvent, SleepLedgerEventType, utc_now


_GENESIS_HASH = "0" * 64


class SleepCycleLedger:
    """Hash-chained sleep events protected from SQL update or deletion."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def append(
        self,
        event_type: SleepLedgerEventType,
        *,
        user_id: UUID,
        payload: dict[str, Any],
        run_id: Optional[UUID] = None,
        job_id: Optional[UUID] = None,
    ) -> SleepLedgerEvent:
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._append_sync,
                event_type,
                user_id,
                payload,
                run_id,
                job_id,
            )

    async def list_events(
        self,
        user_id: UUID,
        *,
        event_types: Optional[Sequence[SleepLedgerEventType]] = None,
        run_id: Optional[UUID] = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[SleepLedgerEvent]:
        self._require_connected()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        return await asyncio.to_thread(
            self._list_sync,
            user_id,
            event_types,
            run_id,
            after_sequence,
            limit,
        )

    async def verify_integrity(self) -> bool:
        self._require_connected()
        return await asyncio.to_thread(self._verify_integrity_sync)

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sleep_cycle_ledger (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    run_id TEXT,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sleep_ledger_user_sequence "
                "ON sleep_cycle_ledger(user_id, sequence)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sleep_ledger_run "
                "ON sleep_cycle_ledger(user_id, run_id, sequence)"
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS sleep_ledger_no_update
                BEFORE UPDATE ON sleep_cycle_ledger
                BEGIN SELECT RAISE(ABORT, 'sleep cycle ledger is append-only'); END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS sleep_ledger_no_delete
                BEFORE DELETE ON sleep_cycle_ledger
                BEGIN SELECT RAISE(ABORT, 'sleep cycle ledger is append-only'); END
                """
            )

    def _append_sync(
        self,
        event_type: SleepLedgerEventType,
        user_id: UUID,
        payload: dict[str, Any],
        run_id: Optional[UUID],
        job_id: Optional[UUID],
    ) -> SleepLedgerEvent:
        event_id = uuid4()
        created_at = utc_now()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT event_hash FROM sleep_cycle_ledger ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous[0] if previous else _GENESIS_HASH
            digest_input = "|".join(
                (
                    previous_hash,
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(run_id or ""),
                    str(job_id or ""),
                    created_at.isoformat(),
                    payload_json,
                )
            )
            event_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT INTO sleep_cycle_ledger "
                "(event_id, event_type, user_id, run_id, job_id, created_at, payload, "
                "previous_hash, event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(run_id) if run_id else None,
                    str(job_id) if job_id else None,
                    created_at.isoformat(),
                    payload_json,
                    previous_hash,
                    event_hash,
                ),
            )
            connection.commit()
        return SleepLedgerEvent(
            sequence=cursor.lastrowid,
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            run_id=run_id,
            job_id=job_id,
            created_at=created_at,
            payload=json.loads(payload_json),
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    def _list_sync(
        self,
        user_id: UUID,
        event_types: Optional[Sequence[SleepLedgerEventType]],
        run_id: Optional[UUID],
        after_sequence: int,
        limit: int,
    ) -> list[SleepLedgerEvent]:
        where = ["user_id = ?", "sequence > ?"]
        parameters: list[Any] = [str(user_id), after_sequence]
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            where.append(f"event_type IN ({placeholders})")
            parameters.extend(item.value for item in event_types)
        if run_id:
            where.append("run_id = ?")
            parameters.append(str(run_id))
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, run_id, job_id, "
                "created_at, payload, previous_hash, event_hash FROM sleep_cycle_ledger "
                f"WHERE {' AND '.join(where)} ORDER BY sequence ASC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _verify_integrity_sync(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, run_id, job_id, "
                "created_at, payload, previous_hash, event_hash "
                "FROM sleep_cycle_ledger ORDER BY sequence ASC"
            ).fetchall()
        expected_previous = _GENESIS_HASH
        for row in rows:
            if row[8] != expected_previous:
                return False
            digest_input = "|".join(
                (expected_previous, *[str(value or "") for value in row[1:8]])
            )
            if hashlib.sha256(digest_input.encode("utf-8")).hexdigest() != row[9]:
                return False
            expected_previous = row[9]
        return True

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> SleepLedgerEvent:
        return SleepLedgerEvent(
            sequence=row[0],
            event_id=UUID(row[1]),
            event_type=SleepLedgerEventType(row[2]),
            user_id=UUID(row[3]),
            run_id=UUID(row[4]) if row[4] else None,
            job_id=UUID(row[5]) if row[5] else None,
            created_at=datetime.fromisoformat(row[6]),
            payload=json.loads(row[7]),
            previous_hash=row[8],
            event_hash=row[9],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("SleepCycleLedger.connect() must be awaited before use")
