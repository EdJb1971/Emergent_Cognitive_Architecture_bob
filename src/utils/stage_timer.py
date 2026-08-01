"""Per-stage wall-clock timing for a single cognitive cycle."""

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple


class StageTimer:
    """Accumulates named stage durations so a cycle's latency can be attributed."""

    def __init__(self) -> None:
        self._cycle_start = time.perf_counter()
        self._stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name: str, duration_ms: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + duration_ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._cycle_start) * 1000.0

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self._stages.items(), key=lambda kv: kv[1], reverse=True)

    def as_dict(self) -> Dict[str, float]:
        stages = {name: round(ms, 1) for name, ms in self.ranked()}
        total = self.total_ms
        measured = sum(self._stages.values())
        stages["_total"] = round(total, 1)
        stages["_unattributed"] = round(max(total - measured, 0.0), 1)
        return stages

    def summary(self, top_n: int = 8) -> str:
        total = self.total_ms
        measured = sum(self._stages.values())
        parts = [f"total={total / 1000:.1f}s"]
        for name, ms in self.ranked()[:top_n]:
            share = (ms / total * 100.0) if total > 0 else 0.0
            parts.append(f"{name}={ms / 1000:.1f}s({share:.0f}%)")
        parts.append(f"unattributed={max(total - measured, 0.0) / 1000:.1f}s")
        return " | ".join(parts)
