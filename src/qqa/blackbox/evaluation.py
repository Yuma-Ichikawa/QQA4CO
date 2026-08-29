"""Persistent black-box evaluation cache and asynchronous state machine."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from time import perf_counter, time
from typing import Any

import torch

from qqa.blackbox.problem import BlackBoxProblem


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    point_hash: str
    point: tuple[float, ...]
    status: EvaluationStatus
    objective: float | None = None
    violations: tuple[float, ...] = ()
    runtime: float = 0.0
    exception_category: str | None = None
    timestamp: float = 0.0
    seed: int = 0
    worker: int | None = None
    fidelity: str = "default"
    replicate: int = 0
    evaluator_version: str = "1"


def point_hash(point: torch.Tensor) -> str:
    values = point.detach().reshape(-1).cpu().to(torch.float64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class EvaluationDatabase:
    """SQLite observations keyed by full experimental identity.

    Point, problem, seed, fidelity, replicate, and evaluator version are all
    part of the identity. Repeated noisy observations therefore coexist rather
    than overwriting one another.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    problem TEXT NOT NULL,
                    point_hash TEXT NOT NULL,
                    point TEXT NOT NULL,
                    status TEXT NOT NULL,
                    objective REAL,
                    violations TEXT NOT NULL,
                    runtime REAL NOT NULL,
                    exception_category TEXT,
                    timestamp REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    worker INTEGER,
                    PRIMARY KEY(problem, point_hash)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_observations (
                    problem TEXT NOT NULL,
                    point_hash TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    fidelity TEXT NOT NULL,
                    replicate INTEGER NOT NULL,
                    evaluator_version TEXT NOT NULL,
                    point TEXT NOT NULL,
                    status TEXT NOT NULL,
                    objective REAL,
                    violations TEXT NOT NULL,
                    runtime REAL NOT NULL,
                    exception_category TEXT,
                    timestamp REAL NOT NULL,
                    worker INTEGER,
                    PRIMARY KEY(
                        problem, point_hash, seed, fidelity, replicate, evaluator_version
                    )
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    def get(
        self,
        problem: str,
        point: torch.Tensor,
        *,
        seed: int = 0,
        fidelity: str = "default",
        replicate: int = 0,
        evaluator_version: str = "1",
    ) -> EvaluationRecord | None:
        digest = point_hash(point)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT point_hash, point, status, objective, violations, runtime,
                          exception_category, timestamp, seed, worker, fidelity,
                          replicate, evaluator_version
                   FROM evaluation_observations
                   WHERE problem=? AND point_hash=? AND seed=? AND fidelity=?
                         AND replicate=? AND evaluator_version=?""",
                (problem, digest, seed, fidelity, replicate, evaluator_version),
            ).fetchone()
        if row is None:
            return None
        return EvaluationRecord(
            row[0],
            tuple(json.loads(row[1])),
            EvaluationStatus(row[2]),
            row[3],
            tuple(json.loads(row[4])),
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[12],
        )

    def put(self, problem: str, record: EvaluationRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO evaluation_observations
                   (problem, point_hash, seed, fidelity, replicate, evaluator_version,
                    point, status, objective, violations, runtime, exception_category,
                    timestamp, worker)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    problem,
                    record.point_hash,
                    record.seed,
                    record.fidelity,
                    record.replicate,
                    record.evaluator_version,
                    json.dumps(record.point),
                    record.status.value,
                    record.objective,
                    json.dumps(record.violations),
                    record.runtime,
                    record.exception_category,
                    record.timestamp,
                    record.worker,
                ),
            )


def _isolated_evaluate(problem: BlackBoxProblem, point: torch.Tensor, output) -> None:
    try:
        objective, violations, _ = problem.evaluate_one(point)
        output.put(("ok", objective, tuple(violations)))
    except Exception as exc:  # noqa: BLE001 - explicit worker boundary
        output.put(("error", type(exc).__name__))


class AsynchronousEvaluationScheduler:
    """Submit independent points and expose non-blocking completion events."""

    def __init__(
        self,
        problem: BlackBoxProblem,
        *,
        workers: int = 1,
        database: EvaluationDatabase | None = None,
        problem_fingerprint: str | None = None,
        seed: int = 0,
        fidelity: str = "default",
        evaluator_version: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        self.problem = problem
        self.database = database
        self.problem_fingerprint = problem_fingerprint or problem.name
        self.seed = seed
        if not isinstance(fidelity, str) or not fidelity:
            raise ValueError("fidelity must be a non-empty string.")
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive or None.")
        self.fidelity = fidelity
        self.evaluator_version = evaluator_version or problem.evaluator_version
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._futures: dict[Future, EvaluationRecord] = {}

    def _evaluate(self, point: torch.Tensor, pending: EvaluationRecord) -> EvaluationRecord:
        running = replace(pending, status=EvaluationStatus.RUNNING)
        if self.database is not None:
            self.database.put(self.problem_fingerprint, running)
        started = perf_counter()
        try:
            if self.timeout is None:
                objective, violations, _ = self.problem.evaluate_one(point)
            else:
                methods = mp.get_all_start_methods()
                context = mp.get_context("fork" if "fork" in methods else "spawn")
                output = context.Queue(maxsize=1)
                process = context.Process(
                    target=_isolated_evaluate,
                    args=(self.problem, point, output),
                    daemon=True,
                )
                process.start()
                process.join(timeout=self.timeout)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    output.close()
                    record = replace(
                        pending,
                        status=EvaluationStatus.TIMED_OUT,
                        runtime=perf_counter() - started,
                        exception_category="TimeoutError",
                        timestamp=time(),
                    )
                    if self.database is not None:
                        self.database.put(self.problem_fingerprint, record)
                    return record
                envelope = output.get(timeout=0.25)
                output.close()
                if envelope[0] != "ok":
                    raise RuntimeError(f"Isolated evaluator failed ({envelope[1]}).")
                objective, violations = envelope[1], envelope[2]
            record = EvaluationRecord(
                pending.point_hash,
                pending.point,
                EvaluationStatus.COMPLETED,
                objective,
                tuple(violations),
                perf_counter() - started,
                timestamp=time(),
                seed=pending.seed,
                worker=pending.worker,
                fidelity=pending.fidelity,
                replicate=pending.replicate,
                evaluator_version=pending.evaluator_version,
            )
        except Exception as exc:
            record = EvaluationRecord(
                pending.point_hash,
                pending.point,
                EvaluationStatus.FAILED,
                runtime=perf_counter() - started,
                exception_category=type(exc).__name__,
                timestamp=time(),
                seed=pending.seed,
                worker=pending.worker,
                fidelity=pending.fidelity,
                replicate=pending.replicate,
                evaluator_version=pending.evaluator_version,
            )
        if self.database is not None:
            self.database.put(self.problem_fingerprint, record)
        return record

    def submit(
        self,
        point: torch.Tensor,
        *,
        worker: int | None = None,
        replicate: int = 0,
    ) -> Future:
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("replicate must be a non-negative integer.")
        cpu = point.detach().reshape(-1).cpu().to(torch.float64)
        cached = (
            self.database.get(
                self.problem_fingerprint,
                cpu,
                seed=self.seed,
                fidelity=self.fidelity,
                replicate=replicate,
                evaluator_version=self.evaluator_version,
            )
            if self.database
            else None
        )
        if cached is not None and cached.status is EvaluationStatus.COMPLETED:
            future: Future = Future()
            future.set_result(cached)
            return future
        pending = EvaluationRecord(
            point_hash(cpu),
            tuple(cpu.tolist()),
            EvaluationStatus.PENDING,
            timestamp=time(),
            seed=self.seed,
            worker=worker,
            fidelity=self.fidelity,
            replicate=replicate,
            evaluator_version=self.evaluator_version,
        )
        if self.database is not None:
            self.database.put(self.problem_fingerprint, pending)
        future = self._executor.submit(self._evaluate, cpu, pending)
        self._futures[future] = pending
        return future

    def completed(self) -> list[EvaluationRecord]:
        records = []
        for future in tuple(self._futures):
            if future.done():
                records.append(future.result())
                self._futures.pop(future, None)
        return records

    def close(self, *, cancel_pending: bool = False) -> None:
        if cancel_pending:
            for future, pending in tuple(self._futures.items()):
                if future.cancel() and self.database is not None:
                    self.database.put(
                        self.problem_fingerprint,
                        EvaluationRecord(
                            pending.point_hash,
                            pending.point,
                            EvaluationStatus.CANCELLED,
                            timestamp=time(),
                            seed=pending.seed,
                            worker=pending.worker,
                            fidelity=pending.fidelity,
                            replicate=pending.replicate,
                            evaluator_version=pending.evaluator_version,
                        ),
                    )
        self._executor.shutdown(wait=not cancel_pending, cancel_futures=cancel_pending)

    def __enter__(self) -> AsynchronousEvaluationScheduler:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "AsynchronousEvaluationScheduler",
    "EvaluationDatabase",
    "EvaluationRecord",
    "EvaluationStatus",
    "point_hash",
]
