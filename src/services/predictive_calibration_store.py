"""Durable append-only review and calibration plane for predictive perception."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence
from uuid import UUID, uuid4

from src.models.predictive_models import (
    HypothesisVerdict,
    ObservationQualityVerdict,
    PredictionOutcomeVerdict,
    PredictiveAssessmentReview,
    PredictiveCalibrationDay,
    PredictiveCalibrationLabelRequest,
    PredictiveCalibrationStratum,
    PredictiveCalibrationSummary,
    PredictiveConfidenceBin,
    PredictiveLedgerEvent,
    PredictiveLedgerEventType,
    PredictivePerceptionAssessment,
    PredictivePreferredAction,
    PredictiveReviewStatus,
    RecommendationVerdict,
)


logger = logging.getLogger(__name__)
_GENESIS_HASH = "0" * 64
_ASSESSMENT_TARGET = "__assessment__"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


class PredictiveCalibrationStore:
    """SQLite-enforced immutable assessments and append-only human judgements."""

    def __init__(
        self,
        path: str | Path,
        event_sink: Optional[Callable[[PredictiveLedgerEvent], Awaitable[None]]] = None,
    ) -> None:
        self.path = Path(path)
        self._event_sink = event_sink
        self._write_lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def record_assessment(
        self,
        assessment: PredictivePerceptionAssessment,
        *,
        user_id: UUID,
    ) -> PredictiveLedgerEvent:
        self._require_connected()
        async with self._write_lock:
            event, created = await asyncio.to_thread(
                self._record_assessment_sync, assessment, user_id
            )
        if created:
            await self._project(event)
        return event

    async def append_label(
        self,
        *,
        user_id: UUID,
        assessment_id: UUID,
        label: PredictiveCalibrationLabelRequest,
    ) -> PredictiveLedgerEvent:
        self._require_connected()
        review = await self.get_assessment(user_id, assessment_id)
        self._validate_label_target(review.assessment, label)
        async with self._write_lock:
            event = await asyncio.to_thread(
                self._append_label_sync, user_id, review.assessment, label
            )
        await self._project(event)
        return event

    async def get_assessment(
        self, user_id: UUID, assessment_id: UUID,
    ) -> PredictiveAssessmentReview:
        self._require_connected()
        result = await asyncio.to_thread(
            self._get_assessment_sync, user_id, assessment_id
        )
        if result is None:
            raise KeyError("Predictive assessment was not found for this user.")
        return result

    async def list_assessments(
        self,
        user_id: UUID,
        *,
        review_status: Optional[PredictiveReviewStatus] = None,
        material_only: bool = False,
        limit: int = 50,
    ) -> list[PredictiveAssessmentReview]:
        self._require_connected()
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        return await asyncio.to_thread(
            self._list_assessments_sync,
            user_id,
            review_status,
            material_only,
            limit,
        )

    async def list_events(
        self,
        user_id: UUID,
        *,
        event_types: Optional[Sequence[PredictiveLedgerEventType]] = None,
        assessment_id: Optional[UUID] = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[PredictiveLedgerEvent]:
        self._require_connected()
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return await asyncio.to_thread(
            self._list_events_sync,
            user_id,
            event_types,
            assessment_id,
            after_sequence,
            limit,
        )

    async def calibration_summary(self, user_id: UUID) -> PredictiveCalibrationSummary:
        self._require_connected()
        events = await asyncio.to_thread(self._all_events_sync, user_id)
        summary = self._calculate_summary(events)
        return summary.model_copy(
            update={"ledger_integrity_verified": await self.verify_integrity()}
        )

    async def verify_integrity(self) -> bool:
        self._require_connected()
        return await asyncio.to_thread(self._verify_integrity_sync)

    async def status(self) -> dict[str, object]:
        self._require_connected()
        return {
            "connected": True,
            "persistence": "sqlite_wal",
            "append_only": True,
            "hash_chain": "sha256",
            "ledger_integrity_verified": await self.verify_integrity(),
            "predictive_influence_eligible": False,
        }

    async def _project(self, event: PredictiveLedgerEvent) -> None:
        if not self._event_sink:
            return
        try:
            await self._event_sink(event)
        except Exception:
            logger.warning("Predictive telemetry projection failed", exc_info=True)

    @staticmethod
    def _validate_label_target(
        assessment: PredictivePerceptionAssessment,
        label: PredictiveCalibrationLabelRequest,
    ) -> None:
        errors = {item.error_id: item for item in assessment.prediction_errors}
        recommendation = assessment.recommendation
        if label.error_id is not None and label.error_id not in errors:
            raise KeyError("Prediction error is not part of the recorded assessment.")
        if label.error_id is None:
            if recommendation is None or recommendation.source_error_ids:
                raise ValueError(
                    "Assessment-level labels are reserved for recommendations without prediction errors."
                )
        related_to_recommendation = bool(
            recommendation
            and (
                label.error_id in recommendation.source_error_ids
                if label.error_id is not None
                else not recommendation.source_error_ids
            )
        )
        recommendation_reviewed = label.recommendation_verdict not in {
            RecommendationVerdict.NOT_APPLICABLE,
            RecommendationVerdict.UNCERTAIN,
        } or label.preferred_action != PredictivePreferredAction.NONE
        if recommendation_reviewed and not related_to_recommendation:
            raise ValueError("Recommendation judgement does not target this prediction error.")

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictive_calibration_ledger (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    assessment_id TEXT NOT NULL,
                    error_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictive_ledger_user_sequence "
                "ON predictive_calibration_ledger(user_id, sequence)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictive_ledger_assessment "
                "ON predictive_calibration_ledger(user_id, assessment_id, sequence)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictive_assessment_once "
                "ON predictive_calibration_ledger(user_id, assessment_id) "
                "WHERE event_type = 'assessment_recorded'"
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS predictive_ledger_no_update
                BEFORE UPDATE ON predictive_calibration_ledger
                BEGIN SELECT RAISE(ABORT, 'predictive calibration ledger is append-only'); END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS predictive_ledger_no_delete
                BEFORE DELETE ON predictive_calibration_ledger
                BEGIN SELECT RAISE(ABORT, 'predictive calibration ledger is append-only'); END
                """
            )

    def _record_assessment_sync(
        self,
        assessment: PredictivePerceptionAssessment,
        user_id: UUID,
    ) -> tuple[PredictiveLedgerEvent, bool]:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, cycle_id, assessment_id, "
                "error_id, created_at, payload, previous_hash, event_hash "
                "FROM predictive_calibration_ledger "
                "WHERE user_id = ? AND assessment_id = ? AND event_type = ?",
                (
                    str(user_id),
                    str(assessment.assessment_id),
                    PredictiveLedgerEventType.ASSESSMENT_RECORDED.value,
                ),
            ).fetchone()
        if existing:
            return self._row_to_event(existing), False
        event = self._append_sync(
            PredictiveLedgerEventType.ASSESSMENT_RECORDED,
            user_id=user_id,
            cycle_id=assessment.cycle_id,
            assessment_id=assessment.assessment_id,
            error_id=None,
            payload={"assessment": assessment.model_dump(mode="json")},
        )
        return event, True

    def _append_label_sync(
        self,
        user_id: UUID,
        assessment: PredictivePerceptionAssessment,
        label: PredictiveCalibrationLabelRequest,
    ) -> PredictiveLedgerEvent:
        with self._connect() as connection:
            if label.error_id is None:
                latest = connection.execute(
                    "SELECT event_id FROM predictive_calibration_ledger "
                    "WHERE user_id = ? AND assessment_id = ? AND event_type = ? "
                    "AND error_id IS NULL ORDER BY sequence DESC LIMIT 1",
                    (
                        str(user_id),
                        str(assessment.assessment_id),
                        PredictiveLedgerEventType.CALIBRATION_LABEL.value,
                    ),
                ).fetchone()
            else:
                latest = connection.execute(
                    "SELECT event_id FROM predictive_calibration_ledger "
                    "WHERE user_id = ? AND assessment_id = ? AND event_type = ? "
                    "AND error_id = ? ORDER BY sequence DESC LIMIT 1",
                    (
                        str(user_id),
                        str(assessment.assessment_id),
                        PredictiveLedgerEventType.CALIBRATION_LABEL.value,
                        str(label.error_id),
                    ),
                ).fetchone()
        return self._append_sync(
            PredictiveLedgerEventType.CALIBRATION_LABEL,
            user_id=user_id,
            cycle_id=assessment.cycle_id,
            assessment_id=assessment.assessment_id,
            error_id=label.error_id,
            payload={
                "label": label.model_dump(mode="json"),
                "supersedes_event_id": latest[0] if latest else None,
            },
        )

    def _append_sync(
        self,
        event_type: PredictiveLedgerEventType,
        *,
        user_id: UUID,
        cycle_id: UUID,
        assessment_id: UUID,
        error_id: Optional[UUID],
        payload: dict[str, Any],
    ) -> PredictiveLedgerEvent:
        event_id = uuid4()
        created_at = _utc_now()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT event_hash FROM predictive_calibration_ledger "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous[0] if previous else _GENESIS_HASH
            digest_input = "|".join(
                (
                    previous_hash,
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(cycle_id),
                    str(assessment_id),
                    str(error_id or ""),
                    created_at.isoformat(),
                    payload_json,
                )
            )
            event_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT INTO predictive_calibration_ledger "
                "(event_id, event_type, user_id, cycle_id, assessment_id, error_id, "
                "created_at, payload, previous_hash, event_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(event_id),
                    event_type.value,
                    str(user_id),
                    str(cycle_id),
                    str(assessment_id),
                    str(error_id) if error_id else None,
                    created_at.isoformat(),
                    payload_json,
                    previous_hash,
                    event_hash,
                ),
            )
            connection.commit()
        return PredictiveLedgerEvent(
            sequence=cursor.lastrowid,
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            cycle_id=cycle_id,
            assessment_id=assessment_id,
            error_id=error_id,
            created_at=created_at,
            payload=json.loads(payload_json),
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    def _get_assessment_sync(
        self, user_id: UUID, assessment_id: UUID,
    ) -> Optional[PredictiveAssessmentReview]:
        events = self._all_events_sync(user_id)
        assessment_event = next(
            (
                event for event in events
                if event.event_type == PredictiveLedgerEventType.ASSESSMENT_RECORDED
                and event.assessment_id == assessment_id
            ),
            None,
        )
        if not assessment_event:
            return None
        return self._review_from_events(assessment_event, events)

    def _list_assessments_sync(
        self,
        user_id: UUID,
        review_status: Optional[PredictiveReviewStatus],
        material_only: bool,
        limit: int,
    ) -> list[PredictiveAssessmentReview]:
        events = self._all_events_sync(user_id)
        assessments = [
            self._review_from_events(event, events)
            for event in reversed(events)
            if event.event_type == PredictiveLedgerEventType.ASSESSMENT_RECORDED
        ]
        return [
            item for item in assessments
            if (review_status is None or item.review_status == review_status)
            and (not material_only or item.assessment.material_error_count > 0)
        ][:limit]

    @staticmethod
    def _review_from_events(
        assessment_event: PredictiveLedgerEvent,
        events: Sequence[PredictiveLedgerEvent],
    ) -> PredictiveAssessmentReview:
        assessment = PredictivePerceptionAssessment.model_validate(
            assessment_event.payload["assessment"]
        )
        latest: dict[str, PredictiveLedgerEvent] = {}
        for event in events:
            if (
                event.event_type == PredictiveLedgerEventType.CALIBRATION_LABEL
                and event.assessment_id == assessment.assessment_id
            ):
                latest[str(event.error_id) if event.error_id else _ASSESSMENT_TARGET] = event
        targets = {str(item.error_id) for item in assessment.prediction_errors}
        if assessment.recommendation and not assessment.recommendation.source_error_ids:
            targets.add(_ASSESSMENT_TARGET)
        reviewed = len(targets.intersection(latest))
        if not targets:
            status = PredictiveReviewStatus.NOT_APPLICABLE
        elif reviewed == 0:
            status = PredictiveReviewStatus.UNREVIEWED
        elif reviewed < len(targets):
            status = PredictiveReviewStatus.PARTIALLY_REVIEWED
        else:
            status = PredictiveReviewStatus.REVIEWED
        return PredictiveAssessmentReview(
            assessment=assessment,
            recorded_at=assessment_event.created_at,
            ledger_sequence=assessment_event.sequence,
            review_status=status,
            review_target_count=len(targets),
            reviewed_target_count=reviewed,
            latest_labels=tuple(sorted(latest.values(), key=lambda item: item.sequence)),
        )

    def _list_events_sync(
        self,
        user_id: UUID,
        event_types: Optional[Sequence[PredictiveLedgerEventType]],
        assessment_id: Optional[UUID],
        after_sequence: int,
        limit: int,
    ) -> list[PredictiveLedgerEvent]:
        where = ["user_id = ?", "sequence > ?"]
        parameters: list[Any] = [str(user_id), after_sequence]
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            where.append(f"event_type IN ({placeholders})")
            parameters.extend(item.value for item in event_types)
        if assessment_id:
            where.append("assessment_id = ?")
            parameters.append(str(assessment_id))
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, cycle_id, assessment_id, "
                "error_id, created_at, payload, previous_hash, event_hash "
                f"FROM predictive_calibration_ledger WHERE {' AND '.join(where)} "
                "ORDER BY sequence ASC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _all_events_sync(self, user_id: UUID) -> list[PredictiveLedgerEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, cycle_id, assessment_id, "
                "error_id, created_at, payload, previous_hash, event_hash "
                "FROM predictive_calibration_ledger WHERE user_id = ? ORDER BY sequence ASC",
                (str(user_id),),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _calculate_summary(
        events: Sequence[PredictiveLedgerEvent],
    ) -> PredictiveCalibrationSummary:
        assessment_events = {
            str(event.assessment_id): event
            for event in events
            if event.event_type == PredictiveLedgerEventType.ASSESSMENT_RECORDED
        }
        latest_labels: dict[tuple[str, str], PredictiveLedgerEvent] = {}
        for event in events:
            if event.event_type == PredictiveLedgerEventType.CALIBRATION_LABEL:
                target = str(event.error_id) if event.error_id else _ASSESSMENT_TARGET
                latest_labels[(str(event.assessment_id), target)] = event

        assessment_status_counts: dict[str, int] = defaultdict(int)
        recommendation_counts: dict[str, int] = defaultdict(int)
        strata: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "observations": 0,
                "labeled": 0,
                "predicted_mismatch": 0,
                "confirmed_mismatch": 0,
                "false_conflict": 0,
                "missed_mismatch": 0,
                "hypothesis_correct": 0,
                "hypothesis_incorrect": 0,
                "recommendation_reviewed": 0,
                "recommendation_useful": 0,
            }
        )
        confidence_samples: list[tuple[float, bool]] = []
        daily: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "assessments": 0,
                "errors": 0,
                "labeled_errors": 0,
                "material_errors": 0,
                "confirmed_mismatches": 0,
                "false_conflicts": 0,
            }
        )
        errors_total = labeled_errors = material_errors = 0
        actionable_assessments = labeled_assessments = 0
        tp = fp = fn = tn = 0
        hypothesis_correct = hypothesis_incorrect = 0
        observation_reliable = observation_reviewed = 0
        recommendation_useful = recommendation_reviewed = 0
        preferred_agreement = preferred_reviewed = 0

        for assessment_id, event in assessment_events.items():
            assessment = PredictivePerceptionAssessment.model_validate(
                event.payload["assessment"]
            )
            day = event.created_at.date().isoformat()
            daily[day]["assessments"] += 1
            assessment_status_counts[assessment.assessment_status] += 1
            if assessment.recommendation:
                recommendation_counts[assessment.recommendation.action] += 1
            else:
                recommendation_counts[PredictivePreferredAction.NONE.value] += 1

            targets = {str(item.error_id) for item in assessment.prediction_errors}
            if assessment.recommendation and not assessment.recommendation.source_error_ids:
                targets.add(_ASSESSMENT_TARGET)
            if targets:
                actionable_assessments += 1
                if any((assessment_id, target) in latest_labels for target in targets):
                    labeled_assessments += 1

            hypotheses = {str(item.hypothesis_id): item for item in assessment.hypotheses}
            recommendation_error_ids = {
                str(item) for item in (
                    assessment.recommendation.source_error_ids
                    if assessment.recommendation else ()
                )
            }
            for error in assessment.prediction_errors:
                errors_total += 1
                daily[day]["errors"] += 1
                material_errors += int(error.material)
                daily[day]["material_errors"] += int(error.material)
                hypothesis = hypotheses[str(error.hypothesis_id)]
                reliability_key = (
                    "unobserved" if error.observation_reliability is None
                    else "low" if error.observation_reliability < 0.55
                    else "moderate" if error.observation_reliability < 0.75
                    else "high"
                )
                stratum_keys = (
                    f"modality:{error.observed_modality or 'unobserved'}",
                    f"prior_source:{hypothesis.source_kind}",
                    f"reliability:{reliability_key}",
                    f"feature:{error.feature_kind}",
                    f"error_status:{error.status}",
                )
                if str(error.error_id) in recommendation_error_ids and assessment.recommendation:
                    stratum_keys += (f"recommendation:{assessment.recommendation.action}",)
                for key in stratum_keys:
                    strata[key]["observations"] += 1
                    strata[key]["predicted_mismatch"] += int(error.status == "mismatch")

                label_event = latest_labels.get((assessment_id, str(error.error_id)))
                if not label_event:
                    continue
                labeled_errors += 1
                daily[day]["labeled_errors"] += 1
                label = label_event.payload["label"]
                observation_verdict = label["observation_quality"]
                if observation_verdict in {
                    ObservationQualityVerdict.RELIABLE.value,
                    ObservationQualityVerdict.UNRELIABLE.value,
                    ObservationQualityVerdict.INSUFFICIENT.value,
                }:
                    observation_reviewed += 1
                    observation_reliable += int(
                        observation_verdict == ObservationQualityVerdict.RELIABLE.value
                    )
                for key in stratum_keys:
                    strata[key]["labeled"] += 1
                if not label["outcome_known"]:
                    continue
                outcome = label["prediction_outcome"]
                actual_mismatch: Optional[bool] = None
                if outcome in {
                    PredictionOutcomeVerdict.CONFIRMED_MISMATCH.value,
                    PredictionOutcomeVerdict.MISSED_MISMATCH.value,
                }:
                    actual_mismatch = True
                elif outcome in {
                    PredictionOutcomeVerdict.CONFIRMED_MATCH.value,
                    PredictionOutcomeVerdict.FALSE_CONFLICT.value,
                }:
                    actual_mismatch = False
                predicted_mismatch = error.status == "mismatch"
                if actual_mismatch is not None:
                    tp += int(predicted_mismatch and actual_mismatch)
                    fp += int(predicted_mismatch and not actual_mismatch)
                    fn += int(not predicted_mismatch and actual_mismatch)
                    tn += int(not predicted_mismatch and not actual_mismatch)
                if outcome == PredictionOutcomeVerdict.CONFIRMED_MISMATCH.value:
                    daily[day]["confirmed_mismatches"] += 1
                if outcome == PredictionOutcomeVerdict.FALSE_CONFLICT.value:
                    daily[day]["false_conflicts"] += 1

                hypothesis_verdict = label["hypothesis_verdict"]
                if hypothesis_verdict == HypothesisVerdict.CORRECT.value:
                    hypothesis_correct += 1
                    confidence_samples.append((error.prior_confidence, True))
                elif hypothesis_verdict == HypothesisVerdict.INCORRECT.value:
                    hypothesis_incorrect += 1
                    confidence_samples.append((error.prior_confidence, False))
                rec_verdict = label["recommendation_verdict"]
                if rec_verdict in {
                    RecommendationVerdict.USEFUL.value,
                    RecommendationVerdict.UNNECESSARY.value,
                    RecommendationVerdict.WRONG_ACTION.value,
                }:
                    recommendation_reviewed += 1
                    recommendation_useful += int(
                        rec_verdict == RecommendationVerdict.USEFUL.value
                    )
                preferred = label["preferred_action"]
                if preferred != PredictivePreferredAction.NONE.value:
                    preferred_reviewed += 1
                    preferred_agreement += int(
                        bool(assessment.recommendation)
                        and assessment.recommendation.action == preferred
                    )

                for key in stratum_keys:
                    values = strata[key]
                    values["confirmed_mismatch"] += int(
                        outcome == PredictionOutcomeVerdict.CONFIRMED_MISMATCH.value
                    )
                    values["false_conflict"] += int(
                        outcome == PredictionOutcomeVerdict.FALSE_CONFLICT.value
                    )
                    values["missed_mismatch"] += int(
                        outcome == PredictionOutcomeVerdict.MISSED_MISMATCH.value
                    )
                    values["hypothesis_correct"] += int(
                        hypothesis_verdict == HypothesisVerdict.CORRECT.value
                    )
                    values["hypothesis_incorrect"] += int(
                        hypothesis_verdict == HypothesisVerdict.INCORRECT.value
                    )
                    values["recommendation_reviewed"] += int(
                        rec_verdict in {
                            RecommendationVerdict.USEFUL.value,
                            RecommendationVerdict.UNNECESSARY.value,
                            RecommendationVerdict.WRONG_ACTION.value,
                        }
                    )
                    values["recommendation_useful"] += int(
                        rec_verdict == RecommendationVerdict.USEFUL.value
                    )

            if assessment.recommendation and not assessment.recommendation.source_error_ids:
                label_event = latest_labels.get((assessment_id, _ASSESSMENT_TARGET))
                if label_event:
                    label = label_event.payload["label"]
                    if not label["outcome_known"]:
                        continue
                    rec_verdict = label["recommendation_verdict"]
                    if rec_verdict in {
                        RecommendationVerdict.USEFUL.value,
                        RecommendationVerdict.UNNECESSARY.value,
                        RecommendationVerdict.WRONG_ACTION.value,
                    }:
                        recommendation_reviewed += 1
                        recommendation_useful += int(
                            rec_verdict == RecommendationVerdict.USEFUL.value
                        )
                    preferred = label["preferred_action"]
                    if preferred != PredictivePreferredAction.NONE.value:
                        preferred_reviewed += 1
                        preferred_agreement += int(
                            assessment.recommendation.action == preferred
                        )

        confidence_bins, expected_calibration_error = (
            PredictiveCalibrationStore._confidence_calibration(confidence_samples)
        )
        daily_models = []
        for day, values in sorted(daily.items())[-30:]:
            evaluated = values["confirmed_mismatches"] + values["false_conflicts"]
            daily_models.append(PredictiveCalibrationDay(
                date=day,
                **values,
                label_coverage=(
                    values["labeled_errors"] / values["errors"]
                    if values["errors"] else 0.0
                ),
                false_conflict_rate=_ratio(values["false_conflicts"], evaluated),
            ))
        return PredictiveCalibrationSummary(
            assessments=len(assessment_events),
            actionable_assessments=actionable_assessments,
            labeled_assessments=labeled_assessments,
            assessment_label_coverage=(
                labeled_assessments / actionable_assessments
                if actionable_assessments else 0.0
            ),
            errors=errors_total,
            labeled_errors=labeled_errors,
            error_label_coverage=labeled_errors / errors_total if errors_total else 0.0,
            material_errors=material_errors,
            mismatch_confusion_matrix={
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            mismatch_precision=_ratio(tp, tp + fp),
            mismatch_recall=_ratio(tp, tp + fn),
            false_conflict_rate=_ratio(fp, tp + fp),
            hypothesis_accuracy=_ratio(
                hypothesis_correct, hypothesis_correct + hypothesis_incorrect
            ),
            observation_reliable_rate=_ratio(observation_reliable, observation_reviewed),
            recommendation_usefulness_rate=_ratio(
                recommendation_useful, recommendation_reviewed
            ),
            preferred_action_agreement=_ratio(preferred_agreement, preferred_reviewed),
            expected_calibration_error=expected_calibration_error,
            assessment_status_counts=dict(assessment_status_counts),
            recommendation_counts=dict(recommendation_counts),
            strata={
                key: PredictiveCalibrationStratum(**values)
                for key, values in sorted(strata.items())
            },
            confidence_bins=confidence_bins,
            daily=daily_models,
            ledger_integrity_verified=False,
            eligibility_reason=(
                "Predictive influence remains structurally disabled; representative "
                "human-labelled calibration must be reviewed separately before activation."
            ),
        )

    @staticmethod
    def _confidence_calibration(
        samples: Sequence[tuple[float, bool]],
    ) -> tuple[list[PredictiveConfidenceBin], Optional[float]]:
        specifications = (
            ("0-49%", 0.0, 0.5),
            ("50-69%", 0.5, 0.7),
            ("70-84%", 0.7, 0.85),
            ("85-100%", 0.85, 1.0),
        )
        bins: list[PredictiveConfidenceBin] = []
        weighted_gap = 0.0
        for index, (label, lower, upper) in enumerate(specifications):
            selected = [
                item for item in samples
                if lower <= item[0] <= upper
                and (index == len(specifications) - 1 or item[0] < upper)
            ]
            average = sum(item[0] for item in selected) / len(selected) if selected else None
            accuracy = sum(item[1] for item in selected) / len(selected) if selected else None
            gap = abs(average - accuracy) if average is not None and accuracy is not None else None
            if gap is not None:
                weighted_gap += gap * len(selected)
            bins.append(PredictiveConfidenceBin(
                label=label,
                lower_bound=lower,
                upper_bound=upper,
                count=len(selected),
                average_confidence=average,
                empirical_accuracy=accuracy,
                absolute_gap=gap,
            ))
        return bins, weighted_gap / len(samples) if samples else None

    def _verify_integrity_sync(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_id, event_type, user_id, cycle_id, assessment_id, "
                "error_id, created_at, payload, previous_hash, event_hash "
                "FROM predictive_calibration_ledger ORDER BY sequence ASC"
            ).fetchall()
        expected_previous = _GENESIS_HASH
        for row in rows:
            if row[9] != expected_previous:
                return False
            digest_input = "|".join(
                (expected_previous, *[str(value or "") for value in row[1:9]])
            )
            if hashlib.sha256(digest_input.encode("utf-8")).hexdigest() != row[10]:
                return False
            expected_previous = row[10]
        return True

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> PredictiveLedgerEvent:
        return PredictiveLedgerEvent(
            sequence=row[0],
            event_id=UUID(row[1]),
            event_type=PredictiveLedgerEventType(row[2]),
            user_id=UUID(row[3]),
            cycle_id=UUID(row[4]),
            assessment_id=UUID(row[5]),
            error_id=UUID(row[6]) if row[6] else None,
            created_at=datetime.fromisoformat(row[7]),
            payload=json.loads(row[8]),
            previous_hash=row[9],
            event_hash=row[10],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PredictiveCalibrationStore.connect() must be awaited before use.")
