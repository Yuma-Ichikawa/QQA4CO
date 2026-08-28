"""Process-isolated, schema-only remote job service.

The service deliberately accepts portable ModelIR dictionaries, never Python
source, pickle payloads, server-side file paths, or arbitrary import names.
FastAPI is optional and imported only by :func:`create_app`.
"""

from __future__ import annotations

import hmac
import math
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any
from uuid import uuid4


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ServicePolicy:
    maximum_variables: int = 1_000_000
    maximum_budget_seconds: float = 3600.0
    maximum_jobs: int = 1_000
    allowed_devices: tuple[str, ...] = ("auto", "cpu", "cuda")

    def __post_init__(self) -> None:
        if self.maximum_variables < 1 or self.maximum_jobs < 1:
            raise ValueError("Service limits must be positive.")
        if not math.isfinite(self.maximum_budget_seconds) or self.maximum_budget_seconds <= 0:
            raise ValueError("maximum_budget_seconds must be finite and positive.")


@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: JobStatus
    future: Future | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


def _solve_portable_job(
    model_payload: dict[str, Any], solve_options: dict[str, Any]
) -> dict[str, Any]:
    from qqa.api import solve
    from qqa.io.formats import model_ir_from_dict

    model = model_ir_from_dict(model_payload, default_name="remote-model")
    result = solve(model, **solve_options)
    return result.to_dict(include_solutions=True)


class JobManager:
    """Bounded process pool with portable inputs and redacted error output."""

    def __init__(self, *, workers: int = 1, policy: ServicePolicy | None = None) -> None:
        if workers < 1:
            raise ValueError("workers must be positive.")
        self.policy = policy or ServicePolicy()
        self._executor = ProcessPoolExecutor(max_workers=workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, model_payload: dict[str, Any], solve_options: dict[str, Any]) -> str:
        from qqa.io.formats import model_ir_from_dict

        model = model_ir_from_dict(model_payload, default_name="remote-model")
        if model.num_variables > self.policy.maximum_variables:
            raise ValueError("Model exceeds the configured variable limit.")
        options = dict(solve_options)
        allowed = {"goal", "profile", "budget", "device", "seed"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("Remote solve option is not allowed: " + ", ".join(unknown))
        budget = options.get("budget")
        if isinstance(budget, str):
            from qqa.api import _parse_budget

            budget = _parse_budget(budget)
        if budget is not None and float(budget) > self.policy.maximum_budget_seconds:
            raise ValueError("Requested budget exceeds the service policy.")
        device = str(options.get("device", "auto"))
        if device.split(":", 1)[0] not in self.policy.allowed_devices:
            raise ValueError("Requested device is not allowed by the service policy.")
        with self._lock:
            if len(self._jobs) >= self.policy.maximum_jobs:
                completed = [
                    key
                    for key, record in self._jobs.items()
                    if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
                ]
                if not completed:
                    raise RuntimeError("The service job registry is full.")
                del self._jobs[completed[0]]
            job_id = uuid4().hex
            record = JobRecord(job_id, JobStatus.QUEUED)
            self._jobs[job_id] = record
            future = self._executor.submit(_solve_portable_job, model_payload, options)
            record.future = future
            record.status = JobStatus.RUNNING
        future.add_done_callback(partial(self._complete, job_id))
        return job_id

    def _complete(self, job_id: str, future: Future) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if future.cancelled():
                record.status = JobStatus.CANCELLED
                return
            try:
                record.result = future.result()
                record.status = JobStatus.SUCCEEDED
            except Exception as exc:  # process boundary; return type only, no traceback/path
                record.status = JobStatus.FAILED
                record.error = type(exc).__name__

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].public()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            cancelled = bool(record.future and record.future.cancel())
            if cancelled:
                record.status = JobStatus.CANCELLED
            return cancelled

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


def create_app(
    *,
    workers: int = 1,
    policy: ServicePolicy | None = None,
    api_token: str | None = None,
):
    """Create an authenticated FastAPI app around :class:`JobManager`."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise ImportError("Install `qqa[service]` to create the remote job API.") from exc

    manager = JobManager(workers=workers, policy=policy)

    class SubmitRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model: dict[str, Any]
        options: dict[str, Any] = Field(default_factory=dict)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if api_token is None:
            return
        expected = f"Bearer {api_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token.")

    app = FastAPI(title="QQA job service", version="1")

    @app.post("/v1/jobs", dependencies=[Depends(authenticate)], status_code=202)
    def submit(request: SubmitRequest):
        try:
            return {"job_id": manager.submit(request.model, request.options)}
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def status(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job.") from exc

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def cancel(job_id: str):
        try:
            return {"cancelled": manager.cancel(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job.") from exc

    app.state.job_manager = manager
    return app


__all__ = ["JobManager", "JobRecord", "JobStatus", "ServicePolicy", "create_app"]
