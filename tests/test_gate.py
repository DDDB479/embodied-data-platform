"""Configurable quality-gate rules and their MCP surface."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_core import evaluate_quality_gate, load_quality_rules
from embodied_core import mcp_server

from tests import support


def _call(name: str, arguments: dict) -> tuple[bool, dict]:
    response = mcp_server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    )
    result = response["result"]
    return bool(result.get("isError")), json.loads(result["content"][0]["text"])


class QualityGateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_gate_passes_clean_dataset(self) -> None:
        report = evaluate_quality_gate(self.dataset)
        self.assertTrue(report["ok"], report["gate"])
        self.assertEqual(report["gate"]["failure_count"], 0)

    def test_expected_fps_and_features_fail_with_stable_codes(self) -> None:
        report = evaluate_quality_gate(
            self.dataset,
            rules={
                "expected_fps": 60,
                "required_features": ["observation.state", "not-a-feature"],
                "min_score": 95,
            },
        )
        codes = {failure["code"] for failure in report["gate"]["failures"]}
        self.assertFalse(report["ok"])
        self.assertIn("gate.fps_mismatch", codes)
        self.assertIn("gate.required_features", codes)

    def test_anomaly_threshold_runs_streaming_quality(self) -> None:
        spiky = support.write_jsonl_dataset(self.root / "spiky", spikes=True)
        report = evaluate_quality_gate(
            spiky,
            rules={"max_anomaly_windows": 0, "use_cache": False, "chunk_frames": 11},
        )
        self.assertIsNotNone(report["quality"])
        self.assertGreater(report["quality"]["scan"]["chunks"], 1)
        self.assertIn("gate.anomaly_windows", {failure["code"] for failure in report["gate"]["failures"]})

    def test_json_and_toml_rule_files(self) -> None:
        rules = {"expected_fps": 30, "required_features": ["action"], "min_score": 90}
        json_path = self.root / "rules.json"
        json_path.write_text(json.dumps({"gate": rules}), encoding="utf-8")
        self.assertEqual(load_quality_rules(json_path)["expected_fps"], 30.0)
        try:
            import tomllib  # noqa: F401
        except ImportError:
            return
        toml_path = self.root / "rules.toml"
        toml_path.write_text(
            "[gate]\nexpected_fps = 30\nrequired_features = [\"action\"]\nmin_score = 90\n",
            encoding="utf-8",
        )
        self.assertEqual(load_quality_rules(toml_path)["required_features"], ["action"])

    def test_unknown_rule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_quality_rules({"expected_fsp": 30})

    def test_mcp_gate_tool(self) -> None:
        failed, payload = _call(
            "quality_gate",
            {"path": str(self.dataset), "rules": {"expected_fps": 30, "max_nan_ratio": 0.01, "use_cache": False}},
        )
        self.assertFalse(failed)
        self.assertTrue(payload["quality"]["scan"]["streamed"])
        self.assertIn("gate", payload)


if __name__ == "__main__":
    unittest.main()
