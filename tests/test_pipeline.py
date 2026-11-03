"""Persistent pipeline DAGs: templates, human waits, retries and policy."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core.pipeline import PipelineStore, resume_pipeline, run_pipeline
from embodied_core.processors import register_processor

from tests import support


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical", spikes=True)
        self.store = self.root / "pipeline-runs.sqlite3"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_template_human_wait_and_resume(self) -> None:
        spec = {
            "schema_version": 1,
            "name": "review-flow",
            "nodes": [
                {"id": "detect", "type": "tool", "tool": "detect_dataset", "arguments": {"path": "${input}"}},
                {"id": "summary", "type": "tool", "tool": "summarize_dataset", "arguments": {"path": "${input}"}, "depends_on": ["detect"]},
                {"id": "noop", "type": "processor", "processor": "noop", "arguments": {"episodes": "${nodes.summary.episodes}"}, "depends_on": ["summary"]},
                {"id": "review", "type": "human", "arguments": {"path": "${input}", "max_tasks": 2}, "depends_on": ["noop"]},
                {"id": "gate", "type": "tool", "tool": "quality_gate", "arguments": {"path": "${input}"}, "depends_on": ["review"]},
            ],
        }
        first = run_pipeline(spec, store=self.store, variables={"input": str(self.dataset)})
        self.assertEqual(first["status"], "waiting_human")
        self.assertEqual(first["current_node"], "review")
        review_node = next(node for node in first["nodes"] if node["node_id"] == "review")
        self.assertEqual(review_node["status"], "waiting_human")
        self.assertGreaterEqual(review_node["output"]["task_count"], 1)

        resumed = resume_pipeline(
            first["run_id"],
            store=self.store,
            human_results={"review": {"reviewed": True, "labels": [{"task_id": "x", "answer": "valid"}]}},
        )
        self.assertEqual(resumed["status"], "succeeded", resumed)
        self.assertEqual(resumed["current_node"], None)
        gate = next(node for node in resumed["nodes"] if node["node_id"] == "gate")
        self.assertEqual(gate["status"], "succeeded")

    def test_write_tool_requires_policy(self) -> None:
        out = self.root / "written.json"
        spec = {
            "name": "write-flow",
            "nodes": [
                {"id": "write", "type": "tool", "tool": "write_json", "arguments": {"path": str(out), "payload": {"ok": True}}}
            ],
        }
        denied = run_pipeline(spec, store=self.store)
        self.assertEqual(denied["status"], "failed")
        self.assertFalse(out.exists())

        allowed = run_pipeline(spec, store=self.store, allow_write=True)
        self.assertEqual(allowed["status"], "succeeded", allowed)
        self.assertTrue(out.is_file())

    def test_retry_runs_until_success(self) -> None:
        state = {"count": 0}

        def flaky(arguments, context):  # noqa: ANN001
            state["count"] += 1
            if state["count"] < 2:
                raise RuntimeError("transient")
            return {"attempt": state["count"]}

        register_processor("test_flaky", flaky, replace=True)
        spec = {
            "name": "retry-flow",
            "nodes": [{"id": "flaky", "type": "processor", "processor": "test_flaky", "retries": 1}],
        }
        result = run_pipeline(spec, store=self.store)
        self.assertEqual(result["status"], "succeeded", result)
        node = result["nodes"][0]
        self.assertEqual(node["attempts"], 2)
        self.assertEqual(node["output"]["attempt"], 2)

    def test_stats_and_list(self) -> None:
        spec = {"name": "noop", "nodes": [{"id": "n", "type": "noop"}]}
        result = run_pipeline(spec, store=self.store)
        stats = PipelineStore(self.store).stats()
        self.assertGreaterEqual(stats["total_runs"], 1)
        self.assertEqual(stats["by_status"].get("succeeded"), 1)
        listed = PipelineStore(self.store).list_runs(limit=1)
        self.assertEqual(listed[0]["run_id"], result["run_id"])


if __name__ == "__main__":
    unittest.main()
