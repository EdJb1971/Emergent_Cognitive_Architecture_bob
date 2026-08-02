"""Durable task state and append-only audit history for autonomous work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from src.models.autonomous_work_models import (
    AutonomousEventType,
    AutonomousLedgerEvent,
    AutonomousTaskRecord,
    AutonomousTaskRequest,
    AutonomousTaskStatus,
    AutonomousTaskType,
    utc_now,
)


_GENESIS_HASH = "0" * 64


class AutonomousWorkStore:
    """SQLite operational state plus a database-protected immutable event chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        self._connected = True
        await self.recover_interrupted()

    async def close(self) -> None:
        self._connected = False

    async def save_task(self, record: AutonomousTaskRecord) -> None:
        self._require_connected()
        async with self._lock:
            await asyncio.to_thread(self._save_task_sync, record)

    async def get_task(self, task_id: UUID) -> Optional[AutonomousTaskRecord]:
        self._require_connected()
        return await asyncio.to_thread(self._get_task_sync, task_id)

    async def list_tasks(
        self,
        user_id: UUID,
        *,
        statuses: Optional[Iterable[AutonomousTaskStatus]] = None,
        task_types: Optional[Iterable[AutonomousTaskType]] = None,
        limit: int = 100,
    ) -> list[AutonomousTaskRecord]:
        self._require_connected()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return await asyncio.to_thread(
            self._list_tasks_sync, user_id, tuple(statuses or ()), tuple(task_types or ()), limit
        )

    async def find_active_duplicate(
        self, user_id: UUID, task_type: AutonomousTaskType, key: str
    ) -> Optional[AutonomousTaskRecord]:
        self._require_connected()
        return await asyncio.to_thread(self._find_active_duplicate_sync, user_id, task_type, key)

    async def count_recent(
        self, user_id: UUID, task_type: AutonomousTaskType, *, hours: float = 1.0
    ) -> int:
        since = utc_now() - timedelta(hours=hours)
        return await asyncio.to_thread(self._count_recent_sync, user_id, task_type, since)

    async def last_completed_at(
        self, user_id: UUID, task_type: AutonomousTaskType
    ) -> Optional[datetime]:
        return await asyncio.to_thread(self._last_completed_sync, user_id, task_type)

    async def append_event(
        self,
        event_type: AutonomousEventType,
        *,
        user_id: UUID,
        payload: dict[str, Any],
        task_id: Optional[UUID] = None,
        task_type: Optional[AutonomousTaskType] = None,
    ) -> AutonomousLedgerEvent:
        self._require_connected()
        async with self._lock:
            return await asyncio.to_thread(
                self._append_event_sync, event_type, user_id, payload, task_id, task_type
            )

    async def list_events(
        self, user_id: UUID, *, after_sequence: int = 0, limit: int = 100
    ) -> list[AutonomousLedgerEvent]:
        self._require_connected()
        if after_sequence < 0 or not 1 <= limit <= 500:
            raise ValueError("invalid event pagination")
        return await asyncio.to_thread(self._list_events_sync, user_id, after_sequence, limit)

    async def verify_integrity(self) -> bool:
        self._require_connected()
        return await asyncio.to_thread(self._verify_integrity_sync)

    async def load_runtime(self) -> dict[str, Any]:
        self._require_connected()
        return await asyncio.to_thread(self._load_runtime_sync)

    async def save_runtime(self, state: dict[str, Any]) -> None:
        self._require_connected()
        async with self._lock:
            await asyncio.to_thread(self._save_runtime_sync, state)

    async def recover_interrupted(self) -> None:
        """A process restart cannot resume an in-memory coroutine; record it as cancelled."""
        interrupted = await asyncio.to_thread(self._recover_interrupted_sync)
        for record in interrupted:
            await self.append_event(
                AutonomousEventType.TASK_CANCELLED,
                user_id=record.request.user_id,
                task_id=record.request.task_id,
                task_type=record.request.task_type,
                payload={"reason": "process_restart", "recoverable": True},
            )

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_tasks (
                task_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, task_type TEXT NOT NULL,
                deduplication_key TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                completed_at TEXT, record_json TEXT NOT NULL)"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_user_created "
                "ON autonomous_tasks(user_id, created_at DESC)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_dedup "
                "ON autonomous_tasks(user_id, task_type, deduplication_key, status)"
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_runtime (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1), state_json TEXT NOT NULL)"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_work_ledger (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL, user_id TEXT NOT NULL, task_id TEXT, task_type TEXT,
                created_at TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE)"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomous_ledger_user_sequence "
                "ON autonomous_work_ledger(user_id, sequence)"
            )
            db.execute(
                """CREATE TRIGGER IF NOT EXISTS autonomous_ledger_no_update
                BEFORE UPDATE ON autonomous_work_ledger
                BEGIN SELECT RAISE(ABORT, 'autonomous work ledger is append-only'); END"""
            )
            db.execute(
                """CREATE TRIGGER IF NOT EXISTS autonomous_ledger_no_delete
                BEFORE DELETE ON autonomous_work_ledger
                BEGIN SELECT RAISE(ABORT, 'autonomous work ledger is append-only'); END"""
            )

    def _save_task_sync(self, record: AutonomousTaskRecord) -> None:
        data = record.model_dump_json()
        with self._connect() as db:
            db.execute(
                """INSERT INTO autonomous_tasks
                (task_id,user_id,task_type,deduplication_key,status,created_at,completed_at,record_json)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status, completed_at=excluded.completed_at,
                record_json=excluded.record_json""",
                (
                    str(record.request.task_id), str(record.request.user_id),
                    record.request.task_type.value, record.request.deduplication_key,
                    record.status.value, record.request.created_at.isoformat(),
                    record.completed_at.isoformat() if record.completed_at else None, data,
                ),
            )

    def _get_task_sync(self, task_id: UUID) -> Optional[AutonomousTaskRecord]:
        with self._connect() as db:
            row = db.execute(
                "SELECT record_json FROM autonomous_tasks WHERE task_id=?", (str(task_id),)
            ).fetchone()
        return AutonomousTaskRecord.model_validate_json(row[0]) if row else None

    def _list_tasks_sync(self, user_id, statuses, task_types, limit):
        where, args = ["user_id=?"], [str(user_id)]
        if statuses:
            where.append("status IN (%s)" % ",".join("?" for _ in statuses))
            args.extend(item.value for item in statuses)
        if task_types:
            where.append("task_type IN (%s)" % ",".join("?" for _ in task_types))
            args.extend(item.value for item in task_types)
        args.append(limit)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT record_json FROM autonomous_tasks WHERE {' AND '.join(where)} "
                "ORDER BY created_at DESC LIMIT ?", args,
            ).fetchall()
        return [AutonomousTaskRecord.model_validate_json(row[0]) for row in rows]

    def _find_active_duplicate_sync(self, user_id, task_type, key):
        with self._connect() as db:
            row = db.execute(
                """SELECT record_json FROM autonomous_tasks
                WHERE user_id=? AND task_type=? AND deduplication_key=?
                AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1""",
                (str(user_id), task_type.value, key),
            ).fetchone()
        return AutonomousTaskRecord.model_validate_json(row[0]) if row else None

    def _count_recent_sync(self, user_id, task_type, since):
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM autonomous_tasks WHERE user_id=? AND task_type=? "
                "AND created_at>=? AND status NOT IN ('rejected','duplicate')",
                (str(user_id), task_type.value, since.isoformat()),
            ).fetchone()
        return int(row[0])

    def _last_completed_sync(self, user_id, task_type):
        with self._connect() as db:
            rows = db.execute(
                "SELECT completed_at,record_json FROM autonomous_tasks WHERE user_id=? AND task_type=? "
                "AND status='completed' ORDER BY completed_at DESC LIMIT 100",
                (str(user_id), task_type.value),
            ).fetchall()
        for completed_at, record_json in rows:
            record = AutonomousTaskRecord.model_validate_json(record_json)
            if record.result.get("status") != "skipped":
                return datetime.fromisoformat(completed_at) if completed_at else None
        return None

    def _append_event_sync(self, event_type, user_id, payload, task_id, task_type):
        event_id, created_at = uuid4(), utc_now()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT event_hash FROM autonomous_work_ledger ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = row[0] if row else _GENESIS_HASH
            values = (
                previous_hash, str(event_id), event_type.value, str(user_id),
                str(task_id or ""), task_type.value if task_type else "",
                created_at.isoformat(), payload_json,
            )
            event_hash = hashlib.sha256("|".join(values).encode()).hexdigest()
            cursor = db.execute(
                """INSERT INTO autonomous_work_ledger
                (event_id,event_type,user_id,task_id,task_type,created_at,payload,previous_hash,event_hash)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (str(event_id), event_type.value, str(user_id), str(task_id) if task_id else None,
                 task_type.value if task_type else None, created_at.isoformat(), payload_json,
                 previous_hash, event_hash),
            )
        return AutonomousLedgerEvent(
            sequence=cursor.lastrowid, event_id=event_id, event_type=event_type,
            user_id=user_id, task_id=task_id, task_type=task_type, created_at=created_at,
            payload=json.loads(payload_json), previous_hash=previous_hash, event_hash=event_hash,
        )

    def _list_events_sync(self, user_id, after_sequence, limit):
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence,event_id,event_type,user_id,task_id,task_type,created_at,
                payload,previous_hash,event_hash FROM autonomous_work_ledger
                WHERE user_id=? AND sequence>? ORDER BY sequence ASC LIMIT ?""",
                (str(user_id), after_sequence, limit),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _verify_integrity_sync(self):
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence,event_id,event_type,user_id,task_id,task_type,created_at,
                payload,previous_hash,event_hash FROM autonomous_work_ledger ORDER BY sequence"""
            ).fetchall()
        previous = _GENESIS_HASH
        for row in rows:
            if row[8] != previous:
                return False
            values = (previous, *[str(value or "") for value in row[1:8]])
            if hashlib.sha256("|".join(values).encode()).hexdigest() != row[9]:
                return False
            previous = row[9]
        return True

    @staticmethod
    def _row_to_event(row):
        return AutonomousLedgerEvent(
            sequence=row[0], event_id=UUID(row[1]), event_type=AutonomousEventType(row[2]),
            user_id=UUID(row[3]), task_id=UUID(row[4]) if row[4] else None,
            task_type=AutonomousTaskType(row[5]) if row[5] else None,
            created_at=datetime.fromisoformat(row[6]), payload=json.loads(row[7]),
            previous_hash=row[8], event_hash=row[9],
        )

    def _load_runtime_sync(self):
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM autonomous_runtime WHERE singleton=1").fetchone()
        return json.loads(row[0]) if row else {}

    def _save_runtime_sync(self, state):
        with self._connect() as db:
            db.execute(
                "INSERT INTO autonomous_runtime(singleton,state_json) VALUES(1,?) "
                "ON CONFLICT(singleton) DO UPDATE SET state_json=excluded.state_json",
                (json.dumps(state, sort_keys=True, default=str),),
            )

    def _recover_interrupted_sync(self):
        with self._connect() as db:
            rows = db.execute(
                "SELECT record_json FROM autonomous_tasks WHERE status IN ('queued','running')"
            ).fetchall()
            records = [AutonomousTaskRecord.model_validate_json(row[0]) for row in rows]
            for record in records:
                record.status = AutonomousTaskStatus.CANCELLED
                record.completed_at = utc_now()
                record.error = "Interrupted by process restart"
                self._save_task_sync(record)
        return records

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5.0)
        db.execute("PRAGMA busy_timeout=5000")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError("AutonomousWorkStore.connect() must be awaited before use")
