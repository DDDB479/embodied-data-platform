"""The MCP server: protocol framing, tool surface and error codes."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
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
        self.assertEqual(len(result["tools"]), 13)
        for tool in result["tools"]:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")

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

    def tearDown(self) -> None:
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
