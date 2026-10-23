"""Round-trip conversion between the canonical and LeRobot layouts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_core import open_dataset, validate
from embodied_core.convert import ConversionError, convert

from tests import support


class ConvertTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_jsonl_to_lerobot_round_trip(self) -> None:
        source = support.write_jsonl_dataset(self.root / "canonical")
        target = self.root / "converted"
        result = convert(source, target, target="lerobot")
        if result.get("skipped_reason"):
            self.skipTest(result["skipped_reason"])
        self.assertTrue((target / "meta" / "info.json").is_file())
        self.assertEqual(result["episodes"], 2)
        rebuilt = open_dataset(target)
        self.assertEqual(rebuilt.summary.format, "lerobot")
        self.assertEqual(rebuilt.summary.episode_count, 2)
        self.assertEqual(rebuilt.summary.total_frames, 60)
        self.assertEqual(rebuilt.summary.fps, support.FPS)

    def test_lerobot_to_jsonl_round_trip(self) -> None:
        lerobot = support.write_lerobot_dataset(self.root / "lerobot-parent")
        if lerobot is None:
            self.skipTest("pyarrow not installed")
        target = self.root / "canonical"
        result = convert(lerobot, target, target="jsonl")
        self.assertEqual(result["episodes"], 2)
        self.assertTrue((target / "dataset.json").is_file())
        self.assertTrue((target / "episodes.jsonl").is_file())
        rebuilt = open_dataset(target)
        self.assertEqual(rebuilt.summary.format, "embodied-jsonl")
        self.assertEqual(rebuilt.summary.total_frames, 60)

    def test_converted_dataset_passes_validation(self) -> None:
        source = support.write_jsonl_dataset(self.root / "canonical")
        target = self.root / "converted"
        result = convert(source, target, target="lerobot")
        if result.get("skipped_reason"):
            self.skipTest(result["skipped_reason"])
        report = validate(target)
        self.assertTrue(report["ok"], report["issues"])

    def test_canonical_copy_keeps_episode_subset(self) -> None:
        source = support.write_jsonl_dataset(self.root / "canonical", episodes=3)
        target = self.root / "subset"
        convert(source, target, target="jsonl", episodes=[1])
        self.assertEqual(open_dataset(target).summary.episode_count, 1)

    def test_existing_destination_requires_overwrite(self) -> None:
        source = support.write_jsonl_dataset(self.root / "canonical")
        target = self.root / "taken"
        target.mkdir()
        with self.assertRaises(ConversionError):
            convert(source, target, target="jsonl")
        result = convert(source, target, target="jsonl", overwrite=True)
        self.assertEqual(result["episodes"], 2)

    def test_unknown_target_is_rejected(self) -> None:
        source = support.write_jsonl_dataset(self.root / "canonical")
        with self.assertRaises((ConversionError, ValueError)):
            convert(source, self.root / "out", target="parquet-csv-thing")

    def test_manifest_mentions_dropped_features(self) -> None:
        lerobot = support.write_lerobot_dataset(self.root / "lerobot-parent")
        if lerobot is None:
            self.skipTest("pyarrow not installed")
        result = convert(lerobot, self.root / "canonical", target="jsonl")
        self.assertIn("dropped_features", result)


if __name__ == "__main__":
    unittest.main()
