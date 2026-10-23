"""The command line: exit codes, JSON output and human output."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from embodied_core.cli import main

from tests import support


def _run(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = main(list(argv))
    return code, buffer.getvalue()


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_formats_lists_every_layout(self) -> None:
        code, output = _run("formats", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIsInstance(payload, list)
        names = {item["format"] for item in payload}
        self.assertTrue({"lerobot", "hdf5", "rosbag2", "embodied-jsonl"} <= names, names)

    def test_detect_json(self) -> None:
        code, output = _run("detect", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["format"], "embodied-jsonl")
        self.assertGreater(payload["confidence"], 0.5)

    def test_info_json(self) -> None:
        code, output = _run("info", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["episodes"], 2)
        self.assertEqual(payload["total_frames"], 60)
        self.assertEqual(len(payload["episode_list"]), 2)

    def test_validate_exit_codes(self) -> None:
        code, output = _run("validate", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output)["ok"])
        # A directory that is not a dataset is a failed validation, not a crash.
        empty = self.root / "empty"
        empty.mkdir()
        code, output = _run("validate", str(empty), "--json")
        self.assertEqual(code, 1)
        payload = json.loads(output)
        self.assertFalse(payload["ok"])
        self.assertIsNone(payload["format"])
        self.assertIn("format.unsupported", {issue["code"] for issue in payload["issues"]})

    def test_quality_reports_anomalies(self) -> None:
        code, output = _run("quality", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn("stats", payload)
        self.assertIn("anomalies", payload)
        # Bookkeeping columns must not be reported as robot signals.
        self.assertNotIn("episode_index", payload["stats"])
        self.assertTrue(all(window["feature"] not in {"episode_index", "frame_index"} for window in payload["anomalies"]))

    def test_segment_json(self) -> None:
        code, output = _run("segment", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        self.assertIn("segments", json.loads(output))

    def test_label_writes_manifest(self) -> None:
        out = self.root / "tasks.json"
        code, output = _run("label", str(self.dataset), "--out", str(out))
        self.assertEqual(code, 0)
        self.assertIn("wrote", output)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("tasks", payload)

    def test_label_prints_without_out(self) -> None:
        code, output = _run("label", str(self.dataset), "--json")
        self.assertEqual(code, 0)
        self.assertIn("instructions", json.loads(output))

    def test_windows_plan(self) -> None:
        code, output = _run("windows", str(self.dataset), "--window", "10", "--stride", "10", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["window"], 10)
        self.assertGreater(payload["count"], 0)
        self.assertEqual(payload["windows"][0]["start_frame"], 0)

    def test_convert_command(self) -> None:
        target = self.root / "converted"
        code, output = _run("convert", str(self.dataset), str(target), "--target", "lerobot", "--json")
        if code == 3:
            self.skipTest("pyarrow not installed")
        self.assertEqual(code, 0)
        self.assertTrue((target / "meta" / "info.json").is_file())

    def test_scaffold_command(self) -> None:
        code, _ = _run("scaffold", str(self.root / "proj"), "--kind", "pipeline", "--name", "daily")
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "proj" / "README.md").is_file())

    def test_catalog_round_trip(self) -> None:
        db = self.root / "catalog.db"
        code, _ = _run("catalog", "index", str(self.root), "--db", str(db), "--json")
        self.assertEqual(code, 0)
        code, output = _run("catalog", "search", "--db", str(db), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertGreaterEqual(payload["count"], 1)
        self.assertIn("results", payload)
        code, output = _run("catalog", "stats", "--db", str(db), "--json")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(json.loads(output)["total_datasets"], 1)

    def test_checks_command(self) -> None:
        code, output = _run("checks", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn("available", payload)
        self.assertIn("lerobot.spec", payload["available"])
        self.assertIn("default", payload)
        self.assertNotIn("parquet", payload["default"])

    def test_unknown_path_is_a_clean_error(self) -> None:
        code, output = _run("detect", str(self.root / "nope"))
        self.assertEqual(code, 2)
        self.assertIn("error", output)

    def test_human_output_is_readable(self) -> None:
        code, output = _run("info", str(self.dataset))
        self.assertEqual(code, 0)
        self.assertIn("episodes", output)


if __name__ == "__main__":
    unittest.main()
