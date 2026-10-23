"""Structural validation and the publish gate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core import UnsupportedFormat, detect_dataset, validate
from embodied_core.validate import CHECKS, DEFAULT_CHECKS

from tests import support


def _detects_as_lerobot(path: Path) -> bool:
    try:
        return detect_dataset(path).format == "lerobot"
    except UnsupportedFormat:
        return False


class ValidateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_canonical_dataset_has_no_errors(self) -> None:
        path = support.write_jsonl_dataset(self.root / "canonical")
        report = validate(path)
        self.assertTrue(report["ok"], report["issues"])
        errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
        self.assertEqual(errors, [])
        self.assertGreaterEqual(report["score"], 80)
        self.assertEqual(report["format"], "embodied-jsonl")

    def test_good_lerobot_dataset_passes_spec_check(self) -> None:
        path = support.write_lerobot_dataset(self.root / "lerobot-parent")
        if path is None:
            self.skipTest("pyarrow not installed")
        report = validate(path)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertNotIn("lerobot.info_missing", codes)
        self.assertNotIn("lerobot.codebase_version", codes)
        self.assertTrue(report["ok"], report["issues"])

    def test_lerobot_without_meta_info_is_flagged(self) -> None:
        path = support.write_lerobot_dataset(self.root / "lerobot-parent")
        if path is None:
            self.skipTest("pyarrow not installed")
        # Without meta/info.json the layout is no longer detectable, so the caller
        # has to name the format; the spec check must then report the gap.
        self.assertTrue(_detects_as_lerobot(path))
        (path / "meta" / "info.json").unlink()
        self.assertFalse(_detects_as_lerobot(path))
        report = validate(path, format="lerobot", checks=["lerobot.spec"])
        self.assertEqual(report["format"], "lerobot")
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("lerobot.meta_missing", codes)
        self.assertFalse(report["ok"])

    def test_unsupported_path_is_reported_not_raised(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(UnsupportedFormat):
            detect_dataset(empty)
        report = validate(empty)
        self.assertFalse(report["ok"])
        self.assertIsNone(report["format"])
        self.assertIn("format.unsupported", {issue["code"] for issue in report["issues"]})

    def test_unknown_check_name_raises(self) -> None:
        path = support.write_jsonl_dataset(self.root / "canonical")
        with self.assertRaises(KeyError):
            validate(path, checks=["does_not_exist"])

    def test_check_names_are_documented(self) -> None:
        self.assertEqual(
            sorted(CHECKS),
            ["episodes", "features", "fps", "hdf5.demos", "lerobot.spec", "metadata", "parquet", "structure"],
        )
        self.assertNotIn("parquet", DEFAULT_CHECKS)

    def test_selected_checks_are_the_only_ones_run(self) -> None:
        path = support.write_jsonl_dataset(self.root / "canonical")
        report = validate(path, checks=["features"])
        self.assertEqual(list(report["checks"]), ["features"])

    def test_issue_shape_is_stable(self) -> None:
        path = support.write_jsonl_dataset(self.root / "canonical")
        report = validate(path, checks=["metadata", "features"])
        for issue in report["issues"]:
            self.assertTrue({"code", "severity", "message", "scope"} <= set(issue))


if __name__ == "__main__":
    unittest.main()
