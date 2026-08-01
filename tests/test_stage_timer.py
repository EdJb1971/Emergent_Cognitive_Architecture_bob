import time

from src.utils.stage_timer import StageTimer


def test_stage_records_and_ranks_durations():
    timer = StageTimer()
    with timer.stage("slow"):
        time.sleep(0.02)
    with timer.stage("fast"):
        time.sleep(0.001)

    ranked = timer.ranked()
    assert [name for name, _ in ranked] == ["slow", "fast"]
    assert ranked[0][1] >= 15.0


def test_repeated_stage_names_accumulate():
    timer = StageTimer()
    timer.record("agents", 100.0)
    timer.record("agents", 50.0)

    assert dict(timer.ranked())["agents"] == 150.0


def test_as_dict_reports_total_and_unattributed():
    timer = StageTimer()
    time.sleep(0.02)
    timer.record("measured", 5.0)

    result = timer.as_dict()
    assert result["measured"] == 5.0
    assert result["_total"] >= 15.0
    assert result["_unattributed"] == round(result["_total"] - 5.0, 1)


def test_stage_records_duration_even_when_body_raises():
    timer = StageTimer()
    try:
        with timer.stage("boom"):
            raise ValueError("failure inside stage")
    except ValueError:
        pass

    assert "boom" in dict(timer.ranked())


def test_summary_includes_total_and_top_stages():
    timer = StageTimer()
    timer.record("alpha", 2000.0)
    timer.record("beta", 1000.0)

    summary = timer.summary()
    assert summary.startswith("total=")
    assert "alpha=2.0s" in summary
    assert "beta=1.0s" in summary
    assert "unattributed=" in summary
