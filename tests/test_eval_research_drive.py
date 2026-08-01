from pathlib import Path

from src.tools.eval_research_drive import evaluate_fixture, load_fixture


FIXTURE = Path("tests/fixtures/cognitive_research_drive.json")


def test_reviewed_research_drive_fixture_has_diverse_cases():
    fixture = load_fixture(FIXTURE)

    assert len(fixture["cases"]) >= 20
    tags = {tag for case in fixture["cases"] for tag in case.get("tags", [])}
    assert {"routine", "local_first", "clarification", "dream", "privacy", "high_value"} <= tags


def test_research_drive_matches_reviewed_shadow_baseline():
    evaluation = evaluate_fixture(load_fixture(FIXTURE))

    assert evaluation.action_accuracy >= 0.90, evaluation.misses
    assert evaluation.escalation_precision == 1.0
    assert evaluation.escalation_recall == 1.0
