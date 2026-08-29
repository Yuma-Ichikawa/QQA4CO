"""Stateful, inspectable solve sessions over the stable QQA API."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from threading import Lock
from typing import Any

from qqa.config import SolverConfig


class SessionState(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SolveSession:
    """Own one model, explainable plan, result, and event stream.

    Cancellation is guaranteed only before execution starts. Once running, the
    solver's explicit wall-clock budget is the portable interruption boundary.
    """

    def __init__(
        self,
        model: Any,
        *,
        config: SolverConfig | None = None,
        **options: Any,
    ) -> None:
        self.model = model
        self.config = config
        self.options = dict(options)
        self.state = SessionState.CREATED
        self.result = None
        self.error_type: str | None = None
        self._plan = None
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None

    def plan(self):
        from qqa.api import plan

        with self._lock:
            if self.state is SessionState.CANCELLED:
                raise RuntimeError("A cancelled session cannot be planned.")
            if self._plan is None:
                self._plan = plan(self.model, config=self.config, **self.options)
                self.state = SessionState.PLANNED
            return self._plan

    def run(self):
        from qqa.api import solve

        with self._lock:
            if self.state is SessionState.CANCELLED:
                raise RuntimeError("A cancelled session cannot run.")
            if self.state is SessionState.RUNNING:
                raise RuntimeError("The session is already running.")
            if self.state is SessionState.COMPLETED:
                return self.result
            self.state = SessionState.RUNNING
        try:
            result = solve(self.model, config=self.config, **self.options)
        except Exception as exc:
            with self._lock:
                self.error_type = type(exc).__name__
                self.state = SessionState.FAILED
            raise
        with self._lock:
            self.result = result
            self.state = SessionState.COMPLETED
        return result

    def run_async(self) -> Future:
        """Run in one owned background thread for notebook/UI integration."""
        with self._lock:
            if self._executor is not None:
                raise RuntimeError("The session already owns an asynchronous run.")
            self._executor = ThreadPoolExecutor(max_workers=1)
            return self._executor.submit(self.run)

    def cancel(self) -> bool:
        with self._lock:
            if self.state not in {SessionState.CREATED, SessionState.PLANNED}:
                return False
            self.state = SessionState.CANCELLED
            return True

    def close(self, *, wait: bool = True) -> None:
        """Release the optional background executor."""
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> SolveSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def events(self) -> tuple[Any, ...]:
        return () if self.result is None else tuple(getattr(self.result, "events", ()))

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "planned": self._plan is not None,
            "has_result": self.result is not None,
            "error_type": self.error_type,
            "event_count": len(self.events),
        }


__all__ = ["SessionState", "SolveSession"]
