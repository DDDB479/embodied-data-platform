"""Labeling manifest IDs and annotation writeback reconciliation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_core import apply_annotation_feedback, open_dataset, review_annotations
from embodied_core.quality import build_labeling_tasks, detect_anomalies

from tests import support


class AnnotationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        path = support.write_jsonl_dataset(self.root / "canonical", spikes=True)
        dataset = open_dataset(path)
        try:
            ids, columns = dataset.frames()
            windows = detect_anomalies(ids, columns, fps=dataset.summary.fps)
            self.manifest = build_labeling_tasks(dataset.summary, windows, max_tasks=4)
        finally:
            dataset.close()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manifest_ids_are_stable(self) -> None:
        dataset = open_dataset(self.root / "canonical")
        try:
            ids, columns = dataset.frames()
            again = build_labeling_tasks(dataset.summary, detect_anomalies(ids, columns, fps=dataset.summary.fps), max_tasks=4)
        finally:
            dataset.close()
        self.assertEqual(self.manifest["manifest_id"], again["manifest_id"])
        self.assertEqual(
            [task["task_id"] for task in self.manifest["tasks"]],
            [task["task_id"] for task in again["tasks"]],
        )
        self.assertTrue(all(task["status"] == "pending" for task in self.manifest["tasks"]))

    def test_review_counts_and_flags_episode(self) -> None:
        first = self.manifest["tasks"][0]
        reviewed = review_annotations(
            self.manifest,
            [
                {"task_id": first["task_id"], "answer": "invalid", "reason": "teleport"},
                {"id": 1, "answer": "pass"},
            ],
        )
        self.assertEqual(reviewed["reviewed_tasks"], 2)
        self.assertEqual(reviewed["counts"]["invalid"], 1)
        self.assertEqual(reviewed["counts"]["valid"], 1)
        self.assertIn(first["episode"], reviewed["flagged_episodes"])

    def test_duplicate_labels_are_last_wins_with_conflict_log(self) -> None:
        task_id = self.manifest["tasks"][0]["task_id"]
        reviewed = review_annotations(
            self.manifest,
            [
                {"task_id": task_id, "answer": "invalid"},
                {"task_id": task_id, "answer": "valid"},
            ],
        )
        self.assertEqual(reviewed["counts"]["valid"], 1)
        self.assertEqual(len(reviewed["conflicts"]), 1)

    def test_unknown_label_and_incomplete_coverage(self) -> None:
        reviewed = review_annotations(self.manifest, [{"task_id": "task_missing", "answer": "valid"}], require_complete=True)
        self.assertFalse(reviewed["ok"])
        self.assertFalse(reviewed["complete"])
        self.assertEqual(len(reviewed["unknown_labels"]), 1)

    def test_writeback_is_atomic_and_idempotent(self) -> None:
        out = self.root / "review.json"
        labels = self.root / "labels.jsonl"
        task_id = self.manifest["tasks"][0]["task_id"]
        labels.write_text(json.dumps({"task_id": task_id, "answer": "needs_fix"}) + "\n", encoding="utf-8")
        first = apply_annotation_feedback(self.manifest, labels, out=out)
        self.assertTrue(out.is_file())
        self.assertTrue(first["written_to"].endswith("review.json"))
        second = apply_annotation_feedback(self.manifest, labels, out=out, previous=labels)
        self.assertEqual(second["reviewed_tasks"], 1)
        self.assertEqual(second["counts"]["needs_fix"], 1)


if __name__ == "__main__":
    unittest.main()
