"""In-process background jobs with cooperative cancellation.

The MCP transport is synchronous: one request is handled at a time. Long
quality scans therefore run in daemon worker threads and expose a small
submit/status/cancel surface. Results stay in memory for a configurable TTL;
the jobs are deliberately not persisted because the dataset fingerprint and
cache already make a restarted analysis cheap.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ProgressReporter = Callable[[float, str], None]
Runner = Callable[["Job", ProgressReporter], Any]
ProgressEmitter = Callable[["Job", float, str], None]
ErrorFormatter = Callable[[BaseException], dict[str, Any]]

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


class JobCancelled(Exception):
    """Raised by a progress callback when the caller requested cancellation."""


class JobNotFound(KeyError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"unknown job {job_id!r}")


@dataclass
class Job:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict, repr=False)
    progress_token: Any = None
    status: str = "queued"
    progress: float = 0.0
    message: str | None = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATES

    def to_dict(self, *, include_result: bool = False, include_arguments: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "tool": self.tool,
            "status": self.status,
            "progress": round(float(self.progress), 1),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (
                round((self.finished_at or time.time()) - self.started_at, 3) if self.started_at is not None else None
            ),
            "result_ready": self.status == "succeeded",
            "error": self.error,
        }
        if include_arguments:
            body["arguments"] = self.arguments
        if include_result and self.status == "succeeded":
            body["result"] = self.result
        return body


class JobManager:
    """Run a bounded number of daemon jobs and retain their state."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        ttl_s: float = 3600.0,
        max_jobs: int = 200,
        error_formatter: ErrorFormatter | None = None,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.ttl_s = max(1.0, float(ttl_s))
        self.max_jobs = max(1, int(max_jobs))
        self.error_formatter = error_formatter
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(self.max_workers)
        self._closed = False

    def submit(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        runner: Runner,
        *,
        progress_token: Any = None,
        emit_progress: ProgressEmitter | None = None,
    ) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._prune_locked(time.time())
            job = Job(
                id=f"job_{uuid.uuid4().hex[:12]}",
                tool=tool,
                arguments=dict(arguments),
                progress_token=progress_token,
            )
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job, runner, emit_progress),
            name=f"embodied-job-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: Job, runner: Runner, emit_progress: ProgressEmitter | None) -> None:
        acquired = False
        try:
            if job.cancel_requested:
                self._finish_cancelled(job, emit_progress)
                return
            while not self._slots.acquire(timeout=0.05):
                if job.cancel_requested:
                    self._finish_cancelled(job, emit_progress)
                    return
                if self._closed:
                    self._finish_cancelled(job, emit_progress, message="server shutting down")
                    return
            acquired = True
            if job.cancel_requested:
                self._finish_cancelled(job, emit_progress)
                return
            with self._lock:
                job.status = "running"
                job.started_at = time.time()
                job.message = "running"
            self._emit(emit_progress, job, job.progress, job.message)

            def report(progress: float, message: str) -> None:
                with self._lock:
                    if job.cancel_requested:
                        raise JobCancelled()
                    job.progress = min(100.0, max(0.0, float(progress)))
                    job.message = str(message)
                    current = job.progress
                    text = job.message
                self._emit(emit_progress, job, current, text)

            result = runner(job, report)
            with self._lock:
                if job.cancel_requested:
                    cancelled = True
                else:
                    cancelled = False
                    job.result = result
                    job.status = "succeeded"
                    job.progress = 100.0
                    job.message = "completed"
                    job.finished_at = time.time()
                    job._done.set()
            if cancelled:
                self._finish_cancelled(job, emit_progress)
            else:
                self._emit(emit_progress, job, 100.0, "completed")
        except JobCancelled:
            self._finish_cancelled(job, emit_progress)
        except BaseException as exc:  # keep worker errors in the job, never lose them
            if job.cancel_requested:
                self._finish_cancelled(job, emit_progress)
                return
            with self._lock:
                job.status = "failed"
                job.error = (
                    self.error_formatter(exc)
                    if self.error_formatter is not None
                    else {"type": type(exc).__name__, "message": str(exc)}
                )
                job.message = f"failed: {exc}"
                job.finished_at = time.time()
                job._done.set()
            self._emit(emit_progress, job, job.progress, job.message or "failed")
        finally:
            if acquired:
                self._slots.release()

    def _finish_cancelled(self, job: Job, emit_progress: ProgressEmitter | None, *, message: str = "cancelled") -> None:
        with self._lock:
            if job.terminal:
                return
            job.status = "cancelled"
            job.message = message
            job.finished_at = time.time()
            job._done.set()
            progress = job.progress
        self._emit(emit_progress, job, progress, message)

    @staticmethod
    def _emit(emit_progress: ProgressEmitter | None, job: Job, progress: float, message: str) -> None:
        if emit_progress is None:
            return
        try:
            emit_progress(job, progress, message)
        except Exception:
            # A broken transport must not turn a successful analysis into a
            # failed job. Status polling remains the source of truth.
            pass

    def get(self, job_id: str, *, include_result: bool = False) -> dict[str, Any]:
        with self._lock:
            self._prune_locked(time.time())
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            return job.to_dict(include_result=include_result)

    def list(self, *, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked(time.time())
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            if status:
                jobs = [job for job in jobs if job.status == status]
            return [job.to_dict() for job in jobs[: max(1, int(limit))]]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            if job.terminal:
                return job.to_dict()
            job._cancel.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "cancelled before start"
                job.finished_at = time.time()
                job._done.set()
            else:
                job.status = "cancelling"
                job.message = "cancellation requested"
            return job.to_dict()

    def wait(self, job_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
        job._done.wait(timeout=max(0.0, float(timeout_s)))
        return self.get(job_id)

    def shutdown(self) -> int:
        """Stop accepting work and request cancellation for every active job."""

        with self._lock:
            self._closed = True
            active = [job for job in self._jobs.values() if not job.terminal]
            for job in active:
                job._cancel.set()
                if job.status == "queued":
                    job.status = "cancelled"
                    job.message = "server shutting down"
                    job.finished_at = time.time()
                    job._done.set()
                else:
                    job.status = "cancelling"
                    job.message = "server shutting down"
            return len(active)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.ttl_s
        expired = [key for key, job in self._jobs.items() if job.terminal and (job.finished_at or now) < cutoff]
        for key in expired:
            del self._jobs[key]
        if len(self._jobs) <= self.max_jobs:
            return
        candidates = sorted(
            (job for job in self._jobs.values() if job.terminal),
            key=lambda job: job.finished_at or job.created_at,
        )
        for job in candidates[: len(self._jobs) - self.max_jobs]:
            del self._jobs[job.id]
