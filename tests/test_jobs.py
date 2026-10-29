"""Background job state, progress and cooperative cancellation."""
from __future__ import annotations

import threading
import time
import unittest

from embodied_core.jobs import JobManager, JobNotFound


class JobManagerTestCase(unittest.TestCase):
    def test_success_reports_progress_and_result(self) -> None:
        manager = JobManager(max_workers=1)
        seen: list[tuple[float, str]] = []

        def runner(_job, report):
            report(10, "one")
            report(55, "two")
            return {"answer": 42}

        job = manager.submit("demo", {"x": 1}, runner, emit_progress=lambda _j, p, m: seen.append((p, m)))
        body = manager.wait(job.id, timeout_s=5)
        self.assertEqual(body["status"], "succeeded")
        self.assertEqual(body["progress"], 100.0)
        result = manager.get(job.id, include_result=True)
        self.assertEqual(result["result"], {"answer": 42})
        self.assertIn((55.0, "two"), seen)
        self.assertEqual(seen[-1], (100.0, "completed"))

    def test_failure_is_structured(self) -> None:
        manager = JobManager(max_workers=1)

        def runner(_job, _report):
            raise ValueError("broken dataset")

        job = manager.submit("demo", {}, runner)
        body = manager.wait(job.id, timeout_s=5)
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error"]["type"], "ValueError")
        self.assertIn("broken dataset", body["error"]["message"])

    def test_running_job_cancels_at_next_checkpoint(self) -> None:
        manager = JobManager(max_workers=1)
        started = threading.Event()

        def runner(_job, report):
            started.set()
            for step in range(200):
                time.sleep(0.01)
                report(step / 2.0, "working")
            return "late"

        job = manager.submit("demo", {}, runner)
        self.assertTrue(started.wait(2), "runner never started")
        cancelled = manager.cancel(job.id)
        self.assertIn(cancelled["status"], {"cancelling", "cancelled"})
        body = manager.wait(job.id, timeout_s=5)
        self.assertEqual(body["status"], "cancelled")

    def test_queued_job_cancels_immediately(self) -> None:
        manager = JobManager(max_workers=1)
        gate = threading.Event()
        started = threading.Event()

        def blocker(_job, _report):
            started.set()
            gate.wait(5)
            return "first"

        first = manager.submit("demo", {}, blocker)
        self.assertTrue(started.wait(2))
        second = manager.submit("demo", {}, lambda _job, report: report(100, "done"))
        body = manager.cancel(second.id)
        self.assertEqual(body["status"], "cancelled")
        gate.set()
        self.assertEqual(manager.wait(first.id, timeout_s=5)["status"], "succeeded")

    def test_unknown_job_and_shutdown(self) -> None:
        manager = JobManager(max_workers=1)
        with self.assertRaises(JobNotFound):
            manager.get("job_missing")
        manager.shutdown()
        with self.assertRaises(RuntimeError):
            manager.submit("demo", {}, lambda _job, _report: None)

    def test_list_filters_by_status(self) -> None:
        manager = JobManager(max_workers=1)
        job = manager.submit("demo", {}, lambda _job, report: report(100, "done"))
        manager.wait(job.id, timeout_s=5)
        rows = manager.list(status="succeeded")
        self.assertEqual([row["job_id"] for row in rows], [job.id])


if __name__ == "__main__":
    unittest.main()
