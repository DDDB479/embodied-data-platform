"""The MCP server: protocol framing, tool surface and error codes."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from embodied_core import mcp_server

from tests import support


def _rpc(method: str, params: dict | None = None, request_id: int | None = 1) -> dict | None:
    message: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return mcp_server.handle_message(message)


def _call(name: str, arguments: dict) -> tuple[bool, dict]:
    response = _rpc("tools/call", {"name": name, "arguments": arguments})
    assert response is not None
    result = response["result"]
    return bool(result.get("isError")), json.loads(result["content"][0]["text"])


class McpProtocolTestCase(unittest.TestCase):
    def test_initialize_reports_server_info(self) -> None:
        response = _rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}})
        self.assertIsNotNone(response)
        result = response["result"]
        self.assertEqual(result["serverInfo"]["name"], mcp_server.SERVER_NAME)
        self.assertEqual(result["protocolVersion"], mcp_server.PROTOCOL_VERSION)
        self.assertIn("tools", result["capabilities"])

    def test_ping(self) -> None:
        self.assertEqual(_rpc("ping")["result"], {})

    def test_notification_returns_nothing(self) -> None:
        self.assertIsNone(_rpc("notifications/initialized", None, request_id=None))

    def test_tools_list_shape(self) -> None:
        result = _rpc("tools/list")["result"]
        self.assertEqual(len(result["tools"]), 23)
        for tool in result["tools"]:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertEqual(
                set(tool["annotations"]),
                {"readOnlyHint", "destructiveHint", "openWorldHint"},
            )
            self.assertIs(tool["annotations"]["openWorldHint"], tool["name"] == "storage_pull")
        by_name = {tool["name"]: tool for tool in result["tools"]}
        self.assertTrue(by_name["list_formats"]["annotations"]["readOnlyHint"])
        self.assertFalse(by_name["convert_dataset"]["annotations"]["readOnlyHint"])
        self.assertIn("submit_job", by_name)
        self.assertIn("job_status", by_name)
        self.assertIn("quality_gate", by_name)
        self.assertIn("catalog_history", by_name)
        self.assertIn("inspect_media", by_name)
        self.assertIn("decode_samples", by_name)
        self.assertIn("apply_annotations", by_name)
        self.assertIn("storage_pull", by_name)

    def test_unknown_method_is_an_error(self) -> None:
        response = _rpc("does/not/exist")
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)

    def test_batch_request(self) -> None:
        payload = mcp_server.handle_message(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ]
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)


class McpToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")
        self._cache_env = mock.patch.dict(
            os.environ, {"EMBODIED_QUALITY_CACHE": str(self.root / "default-quality-cache.sqlite3")}
        )
        self._cache_env.start()

    def tearDown(self) -> None:
        self._cache_env.stop()
        self._tmp.cleanup()

    def test_detect_and_summarize(self) -> None:
        failed, payload = _call("detect_dataset", {"path": str(self.dataset)})
        self.assertFalse(failed)
        self.assertEqual(payload["format"], "embodied-jsonl")

        failed, summary = _call("summarize_dataset", {"path": str(self.dataset), "include_episodes": True})
        self.assertFalse(failed)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(len(summary["episode_list"]), 2)

    def test_validate_and_quality(self) -> None:
        failed, report = _call("validate_dataset", {"path": str(self.dataset)})
        self.assertFalse(failed)
        self.assertTrue(report["ok"])
        self.assertIn("score", report)

        failed, quality = _call("quality_report", {"path": str(self.dataset)})
        self.assertFalse(failed)
        self.assertIn("observation.state", quality["stats"])
        self.assertNotIn("episode_index", quality["stats"])

    def test_segment_returns_per_episode_map(self) -> None:
        failed, payload = _call("segment_episodes", {"path": str(self.dataset)})
        self.assertFalse(failed)
        self.assertIsInstance(payload["segments"], dict)
        self.assertIn("0", payload["segments"])

    def test_labeling_tasks_with_anomaly(self) -> None:
        support.write_jsonl_dataset(self.root / "spiky", spikes=True)
        failed, payload = _call("labeling_tasks", {"path": str(self.root / "spiky"), "max_tasks": 5})
        self.assertFalse(failed)
        self.assertGreater(payload["task_count"], 0)
        task = payload["tasks"][0]
        self.assertIn("question", task)
        self.assertIn("episode", task)
        self.assertIn("start_frame", task)

    def test_convert_and_plan_windows(self) -> None:
        failed, result = _call(
            "convert_dataset",
            {"source": str(self.dataset), "destination": str(self.root / "out"), "target": "jsonl"},
        )
        self.assertFalse(failed)
        self.assertEqual(result["frames"], 60)

        failed, plan = _call("plan_windows", {"path": str(self.dataset), "window": 8, "stride": 8})
        self.assertFalse(failed)
        self.assertGreater(plan["count"], 0)

    def test_convert_to_lerobot_v3(self) -> None:
        if not support.has_pyarrow():
            self.skipTest("pyarrow not installed")
        failed, result = _call(
            "convert_dataset",
            {
                "source": str(self.dataset),
                "destination": str(self.root / "v3"),
                "target": "lerobot-v3",
                "train_ratio": 0.5,
                "validation_ratio": 0.5,
            },
        )
        self.assertFalse(failed)
        self.assertEqual(result["target"], "lerobot(v3.0)")
        self.assertEqual(result["splits"], {"train": "0:1", "validation": "1:2"})

    def test_inspect_media_and_decode_cdr_samples(self) -> None:
        with mock.patch("embodied_core.media.find_ffprobe", return_value=None):
            failed, media = _call("inspect_media", {"path": str(self.dataset)})
        self.assertFalse(failed)
        self.assertIn("counts", media)
        self.assertFalse(media["video_probe"]["available"])

        path = support.write_rosbag2_cdr(self.root / "bag", messages=3)
        failed, samples = _call(
            "decode_samples",
            {"path": str(path), "topic": "/instruction", "limit": 1, "include_arrays": False},
        )
        self.assertFalse(failed)
        self.assertEqual(samples["samples"][0]["text"]["data"], "pick up the red cube")

    def test_apply_annotations_and_storage_pull(self) -> None:
        spiky = support.write_jsonl_dataset(self.root / "spiky", spikes=True)
        failed, manifest = _call("labeling_tasks", {"path": str(spiky), "max_tasks": 2})
        self.assertFalse(failed)
        task_id = manifest["tasks"][0]["task_id"]
        out = self.root / "review.json"
        failed, reviewed = _call(
            "apply_annotations",
            {"manifest": manifest, "labels": [{"task_id": task_id, "answer": "needs_fix"}], "out": str(out)},
        )
        self.assertFalse(failed)
        self.assertEqual(reviewed["counts"]["needs_fix"], 1)
        self.assertTrue(out.is_file())

        source = self.root / "object-store"
        source.mkdir()
        (source / "payload.bin").write_bytes(b"abc")
        failed, pulled = _call("storage_pull", {"uri": str(source), "destination": str(self.root / "pulled")})
        self.assertFalse(failed)
        self.assertEqual(pulled["files"], 1)

    def test_catalog_round_trip(self) -> None:
        db = self.root / "catalog.db"
        failed, indexed = _call("catalog_index", {"root": str(self.root), "db": str(db)})
        self.assertFalse(failed)
        self.assertGreaterEqual(indexed["added"], 1)

        failed, found = _call("catalog_search", {"db": str(db), "text": "canonical"})
        self.assertFalse(failed)
        self.assertGreaterEqual(found["count"], 1)

        failed, stats = _call("catalog_stats", {"db": str(db)})
        self.assertFalse(failed)
        self.assertGreaterEqual(stats["total_datasets"], 1)

        failed, history = _call("catalog_history", {"db": str(db)})
        self.assertFalse(failed)
        self.assertEqual(history["schema_version"], 2)
        self.assertTrue(history["versions"])
        self.assertTrue(history["runs"])

    def test_list_formats(self) -> None:
        failed, payload = _call("list_formats", {})
        self.assertFalse(failed)
        self.assertEqual(len(payload["formats"]), 7)
        self.assertIn("checks", payload)

    def test_scaffold(self) -> None:
        failed, result = _call(
            "scaffold_project", {"destination": str(self.root / "proj"), "kind": "adapter", "name": "myfmt"}
        )
        self.assertFalse(failed)
        self.assertTrue(result["files"])

    def test_quality_progress_notifications(self) -> None:
        cache = self.root / "quality-cache.sqlite3"
        events: list[dict] = []
        with mock.patch.dict(os.environ, {"EMBODIED_QUALITY_CACHE": str(cache)}):
            response = mcp_server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {
                        "name": "quality_report",
                        "arguments": {"path": str(self.dataset)},
                        "_meta": {"progressToken": "quality-1"},
                    },
                },
                emit=events.append,
            )
        self.assertFalse(response["result"]["isError"])
        progress = [event for event in events if event.get("method") == "notifications/progress"]
        self.assertTrue(progress)
        self.assertTrue(all(item["params"]["progressToken"] == "quality-1" for item in progress))
        self.assertEqual(progress[-1]["params"]["progress"], 100.0)

    def test_async_job_round_trip(self) -> None:
        cache = self.root / "quality-cache.sqlite3"
        events: list[dict] = []
        with mock.patch.dict(os.environ, {"EMBODIED_QUALITY_CACHE": str(cache)}):
            submitted = mcp_server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": "submit_job",
                        "arguments": {"tool": "quality_report", "arguments": {"path": str(self.dataset)}},
                        "_meta": {"progressToken": "async-1"},
                    },
                },
                emit=events.append,
            )
            payload = json.loads(submitted["result"]["content"][0]["text"])
            job_id = payload["job"]["job_id"]
            deadline = time.time() + 10
            status = {}
            while time.time() < deadline:
                failed, status = _call("job_status", {"job_id": job_id})
                self.assertFalse(failed)
                if status["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
        self.assertEqual(status.get("status"), "succeeded", status)
        failed, complete = _call("job_status", {"job_id": job_id, "include_result": True})
        self.assertFalse(failed)
        self.assertIn("observation.state", complete["result"]["stats"])
        progress = [event for event in events if event.get("method") == "notifications/progress"]
        self.assertTrue(progress)
        self.assertTrue(all(item["params"]["progressToken"] == "async-1" for item in progress))

    def test_async_job_rejects_write_tools(self) -> None:
        failed, payload = _call(
            "submit_job",
            {"tool": "convert_dataset", "arguments": {"source": str(self.dataset), "destination": str(self.root / "out")}},
        )
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "bad_arguments")

    def test_quality_gate_is_available_as_an_async_job(self) -> None:
        submitted = mcp_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "submit_job",
                    "arguments": {
                        "tool": "quality_gate",
                        "arguments": {"path": str(self.dataset), "rules": {"expected_fps": 30, "use_cache": False}},
                    },
                },
            }
        )
        payload = json.loads(submitted["result"]["content"][0]["text"])
        job_id = payload["job"]["job_id"]
        deadline = time.time() + 10
        status = {}
        while time.time() < deadline:
            failed, status = _call("job_status", {"job_id": job_id})
            self.assertFalse(failed)
            if status["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        self.assertEqual(status.get("status"), "succeeded", status)
        failed, complete = _call("job_status", {"job_id": job_id, "include_result": True})
        self.assertFalse(failed)
        self.assertTrue(complete["result"]["gate"]["passed"])

    def test_unknown_job_is_structured(self) -> None:
        failed, payload = _call("job_status", {"job_id": "job_missing"})
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "job_not_found")

    # ---- error codes -------------------------------------------------
    def test_missing_path_is_not_found(self) -> None:
        failed, payload = _call("detect_dataset", {"path": str(self.root / "nope")})
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_unknown_tool_is_bad_arguments(self) -> None:
        failed, payload = _call("nope", {})
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "bad_arguments")

    def test_unknown_check_lists_alternatives(self) -> None:
        failed, payload = _call("validate_dataset", {"path": str(self.dataset), "checks": ["nope"]})
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "bad_arguments")
        self.assertIn("available", payload["error"]["message"])

    def test_unsupported_path_is_structured(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        failed, payload = _call("summarize_dataset", {"path": str(empty)})
        self.assertTrue(failed)
        self.assertEqual(payload["error"]["code"], "unsupported_format")


class McpServeTestCase(unittest.TestCase):
    def test_main_list_tools(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(mcp_server.main(["--list-tools"]), 0)
        self.assertIn("detect_dataset", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
