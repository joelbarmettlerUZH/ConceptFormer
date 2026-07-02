"""`log_artifact_resilient` — a flaky W&B upload must retry, then WARN-not-crash (a completed run is
never lost to a transient service-process timeout; the checkpoint is already saved locally)."""

from __future__ import annotations

from conceptformer.cli import log_artifact_resilient


class _Flaky:
    """Fails its first ``fail`` log_artifact calls, then succeeds."""

    def __init__(self, fail: int) -> None:
        self.fail = fail
        self.calls = 0

    def log_artifact(self, art: object) -> None:
        self.calls += 1
        if self.calls <= self.fail:
            raise RuntimeError("the service process is busy")


def test_retries_then_succeeds() -> None:
    wb = _Flaky(fail=2)
    assert log_artifact_resilient(wb, art=None, label="x", retries=3, base_delay=0.0) is True
    assert wb.calls == 3  # two failures + one success


def test_first_try_success_no_retry() -> None:
    wb = _Flaky(fail=0)
    assert log_artifact_resilient(wb, art=None, label="x", retries=3, base_delay=0.0) is True
    assert wb.calls == 1


def test_gives_up_without_raising() -> None:
    wb = _Flaky(fail=99)  # always fails
    # Must NOT raise — returns False so the run finishes cleanly (checkpoint is local).
    assert log_artifact_resilient(wb, art=None, label="x", retries=2, base_delay=0.0) is False
    assert wb.calls == 2
