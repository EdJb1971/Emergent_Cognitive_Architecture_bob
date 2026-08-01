"""Append-only, hash-chained ledger for research governance and calibration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from src.models.research_models import (
    CalibrationLabelRequest,
    CognitiveEffortAction,
    CognitiveResearchAssessment,
    ResearchLedgerEvent,
    ResearchLedgerEventType,
    ResearchPacket,
    SourceQualityFeedbackRequest,
)


_GENESIS_HASH = "0" * 64


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchCalibrationLedger:
    """Durable events that SQLite itself refuses to update or delete."""

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
        event_type: ResearchLedgerEventType,
        *,
        user_id: UUID,
        payload: dict[str, Any],
        inquiry_id: Optional[UUID] = None,
        cycle_id: Optional[UUID] = None,
        assessment_id: Optional[UUID] = None,
        decision_id: Optional[UUID] = None,
        request_id: Optional[UUID] = None,
    ) -> ResearchLedgerEvent:
        self._require_connected()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._append_sync,
                event_type,
                user_id,
                payload,
                inquiry_id,
                cycle_id,
                assessment_id,
                decision_id,
                request_id,
            )

    async def record_assessment(
        self,
        assessment: CognitiveResearchAssessment,
        *,
        user_id: UUID,
        event_type: ResearchLedgerEventType,
        inquiry_id: Optional[UUID] = None,
        cycle_id: Optional[UUID] = None,
    ) -> ResearchLedgerEvent:
        if event_type not in {
            ResearchLedgerEventType.SHADOW_ASSESSMENT,
            ResearchLedgerEventType.WAKING_REVALIDATION,
        }:
            raise ValueError("Assessment events must be shadow_assessment or waking_revalidation.")
        return await self.append(
            event_type,
            user_id=user_id,
            inquiry_id=inquiry_id,
            cycle_id=cycle_id,
            assessment_id=assessment.assessment_id,
            payload={"assessment": assessment.model_dump(mode="json")},
        )

    async def record_packet(
        self,
        packet: ResearchPacket,
        *,
        user_id: UUID,
        inquiry_id: UUID,
    ) -> ResearchLedgerEvent:
        return await self.append(
            ResearchLedgerEventType.RESEARCH_PACKET,
            user_id=user_id,
            inquiry_id=inquiry_id,
            decision_id=packet.decision_id,
            request_id=packet.request_id,
            payload={"packet": packet.model_dump(mode="json")},
        )

    async def list_events(
        self,
        user_id: UUID,
        *,
        event_types: Optional[Sequence[ResearchLedgerEventType]] = None,
        inquiry_id: Optional[UUID] = None,
        assessment_id: Optional[UUID] = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[ResearchLedgerEvent]:
        self._require_connected()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative.")
        return await asyncio.to_thread(
            self._list_sync,
            user_id,
            event_types,
            inquiry_id,
            assessment_id,
            after_sequence,
            limit,
        )

    async def has_assessment(self, user_id: UUID, assessment_id: UUID) -> bool:
        events = await self.list_events(user_id, assessment_id=assessment_id, limit=1)
        return any(
            event.event_type
            in {
                ResearchLedgerEventType.SHADOW_ASSESSMENT,
                ResearchLedgerEventType.WAKING_REVALIDATION,
            }
            for event in events
        )

    async def validate_source_reference(
        self,
        *,
        user_id: UUID,
        inquiry_id: UUID,
        request_id: UUID,
        source_id: str,
    ) -> bool:
        events = await self._all_events(
            user_id,
            event_types=[ResearchLedgerEventType.RESEARCH_PACKET],
            inquiry_id=inquiry_id,
        )
        for event in events:
            if event.request_id != request_id:
                continue
            packet = event.payload.get("packet", {})
            if (
                packet.get("status") == "completed"
                and packet.get("grounding_verified") is True
                and any(
                    source.get("source_id") == source_id
                    for source in packet.get("sources", [])
                )
            ):
                return True
        return False

    async def append_source_feedback(
        self,
        *,
        user_id: UUID,
        inquiry_id: UUID,
        feedback: SourceQualityFeedbackRequest,
    ) -> ResearchLedgerEvent:
        if not await self.validate_source_reference(
            user_id=user_id,
            inquiry_id=inquiry_id,
            request_id=feedback.request_id,
            source_id=feedback.source_id,
        ):
            raise KeyError("The source is not part of a recorded grounded packet for this inquiry.")
        return await self.append(
            ResearchLedgerEventType.SOURCE_FEEDBACK,
            user_id=user_id,
            inquiry_id=inquiry_id,
            request_id=feedback.request_id,
            payload={"feedback": feedback.model_dump(mode="json")},
        )

    async def append_calibration_label(
        self,
        *,
        user_id: UUID,
        assessment_id: UUID,
        label: CalibrationLabelRequest,
    ) -> ResearchLedgerEvent:
        if not await self.has_assessment(user_id, assessment_id):
            raise KeyError("The assessment is not present in this user's calibration ledger.")
        return await self.append(
            ResearchLedgerEventType.CALIBRATION_LABEL,
            user_id=user_id,
            assessment_id=assessment_id,
            payload={"label": label.model_dump(mode="json")},
        )

    async def calibration_summary(self, user_id: UUID) -> dict[str, Any]:
        events = await self._all_events(user_id)
        observations: dict[str, ResearchLedgerEvent] = {}
        latest_labels: dict[str, ResearchLedgerEvent] = {}
        source_feedback: list[dict[str, Any]] = []
        review_outcomes: list[dict[str, Any]] = []
        research_decisions: list[dict[str, Any]] = []
        research_packets: list[dict[str, Any]] = []
        for event in events:
            key = str(event.assessment_id) if event.assessment_id else ""
            if event.event_type in {
                ResearchLedgerEventType.SHADOW_ASSESSMENT,
                ResearchLedgerEventType.WAKING_REVALIDATION,
            } and key:
                observations[key] = event
            elif event.event_type == ResearchLedgerEventType.CALIBRATION_LABEL and key:
                latest_labels[key] = event
            elif event.event_type == ResearchLedgerEventType.SOURCE_FEEDBACK:
                source_feedback.append(event.payload["feedback"])
            elif event.event_type == ResearchLedgerEventType.REVIEW_RESOLVED:
                review_outcomes.append(event.payload)
            elif event.event_type == ResearchLedgerEventType.RESEARCH_DECISION:
                research_decisions.append(event.payload["decision"])
            elif event.event_type == ResearchLedgerEventType.RESEARCH_PACKET:
                research_packets.append(event.payload["packet"])

        action_counts = {action.value: 0 for action in CognitiveEffortAction}
        shadow_count = 0
        tp = fp = fn = tn = 0
        exact_action_matches = 0
        strata = {
            "explicit_request": self._empty_stratum(),
            "non_explicit": self._empty_stratum(),
            "high_stakes": self._empty_stratum(),
            "high_volatility": self._empty_stratum(),
            "privacy_inhibited": self._empty_stratum(),
        }
        for key, event in observations.items():
            assessment = event.payload["assessment"]
            recommended = assessment["recommended_action"]
            action_counts[recommended] = action_counts.get(recommended, 0) + 1
            shadow_count += int(bool(assessment["shadow_mode"]))
            label_event = latest_labels.get(key)
            signals = assessment["signals"]
            active_strata = [
                "explicit_request" if signals["explicit_user_request"] else "non_explicit"
            ]
            if float(signals["task_stakes"]) >= 0.7:
                active_strata.append("high_stakes")
            if float(signals["temporal_volatility"]) >= 0.7:
                active_strata.append("high_volatility")
            if float(signals["privacy_risk"]) >= 0.5:
                active_strata.append("privacy_inhibited")
            for stratum_name in active_strata:
                strata[stratum_name]["observations"] += 1
                strata[stratum_name]["recommended_external"] += int(
                    recommended == CognitiveEffortAction.AUTHORIZE_RESEARCH.value
                )
            if label_event is None:
                continue
            label = label_event.payload["label"]
            predicted = recommended == CognitiveEffortAction.AUTHORIZE_RESEARCH.value
            actual = bool(label["should_external_research"])
            tp += int(predicted and actual)
            fp += int(predicted and not actual)
            fn += int(not predicted and actual)
            tn += int(not predicted and not actual)
            exact_action_matches += int(recommended == label["appropriate_action"])
            for stratum_name in active_strata:
                strata[stratum_name]["labeled"] += 1
                strata[stratum_name]["should_external"] += int(actual)
                strata[stratum_name]["false_positive"] += int(predicted and not actual)
                strata[stratum_name]["false_negative"] += int(not predicted and actual)

        ratings = ("relevance", "authority", "freshness", "citation_support")
        averages = {
            rating: (
                sum(float(item[rating]) for item in source_feedback) / len(source_feedback)
                if source_feedback
                else None
            )
            for rating in ratings
        }
        labeled = len(set(observations).intersection(latest_labels))
        verdict_counts = {
            verdict: sum(1 for item in source_feedback if item["verdict"] == verdict)
            for verdict in ("trustworthy", "useful_with_caveats", "poor", "incorrect")
        }
        review_counts: dict[str, int] = {}
        for outcome in review_outcomes:
            key = outcome.get("disposition") or outcome.get("action") or "unknown"
            review_counts[key] = review_counts.get(key, 0) + 1
        decision_counts: dict[str, int] = {}
        for decision in research_decisions:
            key = decision["disposition"]
            decision_counts[key] = decision_counts.get(key, 0) + 1
        return {
            "observations": len(observations),
            "shadow_observations": shadow_count,
            "labeled_observations": labeled,
            "label_coverage": labeled / len(observations) if observations else 0.0,
            "recommended_action_counts": action_counts,
            "external_research_confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            "external_research_precision": tp / (tp + fp) if tp + fp else None,
            "external_research_recall": tp / (tp + fn) if tp + fn else None,
            "recommended_action_accuracy": exact_action_matches / labeled if labeled else None,
            "calibration_strata": strata,
            "review_outcome_counts": review_counts,
            "research_decision_counts": decision_counts,
            "research_packet_counts": {
                "total": len(research_packets),
                "completed": sum(
                    1 for packet in research_packets if packet["status"] == "completed"
                ),
                "failed": sum(1 for packet in research_packets if packet["status"] == "failed"),
                "grounding_verified": sum(
                    1 for packet in research_packets if packet["grounding_verified"] is True
                ),
            },
            "source_feedback_count": len(source_feedback),
            "source_verdict_counts": verdict_counts,
            "source_quality_averages": averages,
            "source_claim_support_rate": self._optional_boolean_rate(
                source_feedback, "claim_supported"
            ),
            "research_changed_answer_rate": self._optional_boolean_rate(
                source_feedback, "research_changed_answer"
            ),
            "research_resolved_inquiry_rate": self._optional_boolean_rate(
                source_feedback, "research_resolved_inquiry"
            ),
            "research_worth_cost_rate": self._optional_boolean_rate(
                source_feedback, "worth_cost"
            ),
            "automatic_non_explicit_research_eligible": False,
            "eligibility_reason": (
                "Calibration is observational; activation requires a separate reviewed decision."
            ),
            "ledger_integrity_verified": await self.verify_integrity(),
        }

    @staticmethod
    def _empty_stratum() -> dict[str, int]:
        return {
            "observations": 0,
            "labeled": 0,
            "recommended_external": 0,
            "should_external": 0,
            "false_positive": 0,
            "false_negative": 0,
        }

    @staticmethod
    def _optional_boolean_rate(items: list[dict[str, Any]], field: str) -> Optional[float]:
        values = [item[field] for item in items if item.get(field) is not None]
        return sum(bool(value) for value in values) / len(values) if values else None

    async def _all_events(
        self,
        user_id: UUID,
        *,
        event_types: Optional[Sequence[ResearchLedgerEventType]] = None,
        inquiry_id: Optional[UUID] = None,
    ) -> list[ResearchLedgerEvent]:
        events: list[ResearchLedgerEvent] = []
        after_sequence = 0
        while True:
            batch = await self.list_events(
                user_id,
                event_types=event_types,
                inquiry_id=inquiry_id,
                after_sequence=after_sequence,
                limit=500,
            )
            events.extend(batch)
            if len(batch) < 500:
                return events
            after_sequence = batch[-1].sequence

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
                CREATE TABLE IF NOT EXISTS research_calibration_ledger (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    inquiry_id TEXT,
                    cycle_id TEXT,
                    assessment_id TEXT,
                    decision_id TEXT,
                    request_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_ledger_user_sequence "
                "ON research_calibration_ledger(user_id, sequence)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_ledger_inquiry "
                "ON research_calibration_ledger(user_id, inquiry_id, sequence)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_ledger_assessment "
                "ON research_calibration_ledger(user_id, assessment_id, sequence)"
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS research_ledger_no_update
                BEFORE UPDATE ON research_calibration_ledger
                BEGIN SELECT RAISE(ABORT, 'research calibration ledger is append-only'); END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS research_ledger_no_delete
                BEFORE DELETE ON research_calibration_ledger
                BEGIN SELECT RAISE(ABORT, 'research calibration ledger is append-only'); END
                """
            )

    def _append_sync(
        self,
        event_type: ResearchLedgerEventType,
        user_id: UUID,
        payload: dict[str, Any],
        inquiry_id: Optional[UUID],
        cycle_id: Optional[UUID],
        assessment_id: Optional[UUID],
        decision_id: Optional[UUID],
        request_id: Optional[UUID],
    ) -> ResearchLedgerEvent:
        event_id = uuid4()
        created_at = _utc_now()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT event_hash FROM research_calibration_ledger ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous[0] if previous else _GENESIS_HASH
            digest_input = "|".join(
                (
                    previous_hash,
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(inquiry_id or ""),
                    str(cycle_id or ""),
                    str(assessment_id or ""),
                    str(decision_id or ""),
                    str(request_id or ""),
                    created_at.isoformat(),
                    payload_json,
                )
            )
            event_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT INTO research_calibration_ledger "
                "(event_id, event_type, user_id, inquiry_id, cycle_id, assessment_id, decision_id, "
                "request_id, created_at, payload, previous_hash, event_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(inquiry_id) if inquiry_id else None,
                    str(cycle_id) if cycle_id else None,
                    str(assessment_id) if assessment_id else None,
                    str(decision_id) if decision_id else None,
                    str(request_id) if request_id else None,
                    created_at.isoformat(),
                    payload_json,
                    previous_hash,
                    event_hash,
                ),
            )
            connection.commit()
        return ResearchLedgerEvent(
            sequence=cursor.lastrowid,
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            inquiry_id=inquiry_id,
            cycle_id=cycle_id,
            assessment_id=assessment_id,
            decision_id=decision_id,
            request_id=request_id,
            created_at=created_at,
            payload=json.loads(payload_json),
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    def _list_sync(
        self,
        user_id: UUID,
        event_types: Optional[Sequence[ResearchLedgerEventType]],
        inquiry_id: Optional[UUID],
        assessment_id: Optional[UUID],
        after_sequence: int,
        limit: int,
    ) -> list[ResearchLedgerEvent]:
        where = ["user_id = ?", "sequence > ?"]
        parameters: list[Any] = [str(user_id), after_sequence]
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            where.append(f"event_type IN ({placeholders})")
            parameters.extend(event_type.value for event_type in event_types)
        if inquiry_id:
            where.append("inquiry_id = ?")
            parameters.append(str(inquiry_id))
        if assessment_id:
            where.append("assessment_id = ?")
            parameters.append(str(assessment_id))
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, inquiry_id, cycle_id, "
                "assessment_id, decision_id, request_id, created_at, payload, previous_hash, event_hash "
                f"FROM research_calibration_ledger WHERE {' AND '.join(where)} "
                "ORDER BY sequence ASC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _verify_integrity_sync(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, inquiry_id, cycle_id, "
                "assessment_id, decision_id, request_id, created_at, payload, previous_hash, event_hash "
                "FROM research_calibration_ledger ORDER BY sequence ASC"
            ).fetchall()
        expected_previous = _GENESIS_HASH
        for row in rows:
            if row[11] != expected_previous:
                return False
            digest_input = "|".join(
                (expected_previous, *[str(value or "") for value in row[1:10]], row[10])
            )
            if hashlib.sha256(digest_input.encode("utf-8")).hexdigest() != row[12]:
                return False
            expected_previous = row[12]
        return True

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> ResearchLedgerEvent:
        return ResearchLedgerEvent(
            sequence=row[0],
            event_id=UUID(row[1]),
            event_type=ResearchLedgerEventType(row[2]),
            user_id=UUID(row[3]),
            inquiry_id=UUID(row[4]) if row[4] else None,
            cycle_id=UUID(row[5]) if row[5] else None,
            assessment_id=UUID(row[6]) if row[6] else None,
            decision_id=UUID(row[7]) if row[7] else None,
            request_id=UUID(row[8]) if row[8] else None,
            created_at=datetime.fromisoformat(row[9]),
            payload=json.loads(row[10]),
            previous_hash=row[11],
            event_hash=row[12],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("ResearchCalibrationLedger.connect() must be awaited before use.")
