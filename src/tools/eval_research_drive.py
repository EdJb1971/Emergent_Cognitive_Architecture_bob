"""Evaluate CognitiveResearchDrive against reviewed synthetic control cases."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.models.research_models import CognitiveResearchSignals
from src.services.cognitive_research_drive import CognitiveResearchDrive


FIXTURE_VERSION = 1
DEFAULT_FIXTURE = Path("tests/fixtures/cognitive_research_drive.json")
ESCALATION_ACTIONS = {"queue_inquiry", "authorize_research"}


@dataclass
class ResearchDriveEvaluation:
    cases: int
    exact_matches: int
    escalation_true_positives: int
    escalation_false_positives: int
    escalation_false_negatives: int
    misses: list[dict]

    @property
    def action_accuracy(self) -> float:
        return self.exact_matches / self.cases if self.cases else 0.0

    @property
    def escalation_precision(self) -> float:
        denominator = self.escalation_true_positives + self.escalation_false_positives
        return self.escalation_true_positives / denominator if denominator else 1.0

    @property
    def escalation_recall(self) -> float:
        denominator = self.escalation_true_positives + self.escalation_false_negatives
        return self.escalation_true_positives / denominator if denominator else 1.0

    def summary(self) -> str:
        return (
            f"cases={self.cases} action_accuracy={self.action_accuracy:.3f} "
            f"escalation_precision={self.escalation_precision:.3f} "
            f"escalation_recall={self.escalation_recall:.3f}"
        )


def load_fixture(path: Path) -> dict:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("version") != FIXTURE_VERSION:
        raise ValueError(f"Unsupported fixture version {fixture.get('version')}.")
    cases = fixture.get("cases") or []
    if not cases:
        raise ValueError("Research-drive fixture contains no cases.")
    for case in cases:
        if not case.get("id") or not case.get("expected_actions"):
            raise ValueError("Every research-drive case needs id and expected_actions.")
        CognitiveResearchSignals.model_validate(case.get("signals") or {})
    return fixture


def evaluate_fixture(fixture: dict, drive: Optional[CognitiveResearchDrive] = None) -> ResearchDriveEvaluation:
    drive = drive or CognitiveResearchDrive(enabled=True, shadow_mode=False)
    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    misses = []
    for case in fixture["cases"]:
        signals = CognitiveResearchSignals.model_validate(case["signals"])
        assessment = drive.assess(signals, source=f"fixture:{case['id']}")
        actual = assessment.recommended_action.value
        expected = set(case["expected_actions"])
        matched = actual in expected
        exact_matches += int(matched)
        expected_escalation = bool(expected & ESCALATION_ACTIONS)
        actual_escalation = actual in ESCALATION_ACTIONS
        true_positives += int(expected_escalation and actual_escalation)
        false_positives += int(not expected_escalation and actual_escalation)
        false_negatives += int(expected_escalation and not actual_escalation)
        if not matched:
            misses.append(
                {
                    "id": case["id"],
                    "expected": sorted(expected),
                    "actual": actual,
                    "drive_score": round(assessment.drive_score, 4),
                }
            )
    return ResearchDriveEvaluation(
        cases=len(fixture["cases"]),
        exact_matches=exact_matches,
        escalation_true_positives=true_positives,
        escalation_false_positives=false_positives,
        escalation_false_negatives=false_negatives,
        misses=misses,
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the cognitive research-drive controller.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)
    evaluation = evaluate_fixture(load_fixture(args.fixture))
    print(evaluation.summary())
    for miss in evaluation.misses:
        print(
            f"MISS {miss['id']}: expected={miss['expected']} actual={miss['actual']} "
            f"drive={miss['drive_score']:.4f}"
        )
    return 0 if not evaluation.misses else 1


if __name__ == "__main__":
    raise SystemExit(run())

