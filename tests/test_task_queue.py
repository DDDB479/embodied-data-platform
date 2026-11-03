"""Human task queue sampling, prelabels and assignment."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core.task_queue import TaskQueueStore, sample_tasks


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "lm_test",
        "dataset": {"path": "/data/demo", "format": "embodied-jsonl", "episodes": 3},
        "tasks": [
            {"task_id": "task_a", "kind": "verify_window", "episode": 0, "start_frame": 1, "end_frame": 2, "question": "a"},
            {"task_id": "task_b", "kind": "verify_window", "episode": 1, "start_frame": 2, "end_frame": 3, "question": "b"},
            {"task_id": "task_c", "kind": "verify_window", "episode": 2, "start_frame": 3, "end_frame": 4, "question": "c"},
        ],
    }


class TaskQueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TaskQueueStore(Path(self._tmp.name) / "tasks.sqlite3")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sampling_strategies(self) -> None:
        tasks = _manifest()["tasks"]
        prelabels = [
            {"task_id": "task_a", "label": "valid", "confidence": 0.99},
            {"task_id": "task_b", "label": "valid", "confidence": 0.2},
            {"task_id": "task_c", "label": "invalid", "confidence": 0.5},
        ]
        uncertain = sample_tasks(tasks, strategy="uncertainty", limit=2, prelabels=prelabels)
        self.assertEqual([task["task_id"] for task in uncertain], ["task_b", "task_c"])
        stratified = sample_tasks(tasks, strategy="stratified", limit=3)
        self.assertEqual([task["episode"] for task in stratified], [0, 1, 2])

    def test_ingest_assign_complete_and_stats(self) -> None:
        result = self.store.ingest(
            _manifest(),
            prelabels=[{"task_id": "task_b", "label": "valid", "confidence": 0.3}],
            strategy="uncertainty",
            limit=2,
        )
        self.assertEqual(result["inserted"], 2)
        self.assertEqual(result["by_status"].get("prefilled"), 1)
        assigned = self.store.assign("alice", limit=2)
        self.assertEqual(assigned["assigned"], 2)
        completed = self.store.complete(
            [
                {"task_id": assigned["tasks"][0], "answer": "valid", "reason": "ok"},
                {"task_id": assigned["tasks"][1], "answer": "needs_fix", "reason": "bad"},
            ],
            reviewer="alice",
        )
        self.assertEqual(completed["updated"], 2)
        stats = self.store.stats()
        self.assertEqual(stats["by_status"].get("reviewed"), 2)
        self.assertEqual(stats["completion_ratio"], 1.0)
        self.assertEqual(len(self.store.export_labels()), 2)

    def test_ingest_is_idempotent(self) -> None:
        first = self.store.ingest(_manifest(), limit=2)
        second = self.store.ingest(_manifest(), limit=2)
        self.assertEqual(first["inserted"], 2)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["existing"], 2)


if __name__ == "__main__":
    unittest.main()
