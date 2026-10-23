"""Format detection, summarisation and frame access."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core import UnsupportedFormat, detect_dataset, describe_formats, open_dataset
from embodied_core.formats.base import FramesUnavailable

from tests import support


class FormatTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.jsonl = support.write_jsonl_dataset(cls.root / "canonical")
        cls.hdf5 = support.write_hdf5_dataset(cls.root)
        cls.lerobot = support.write_lerobot_dataset(cls.root / "lerobot-parent")
        cls.rosbag = support.write_rosbag2(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_detects_canonical_jsonl(self) -> None:
        detection = detect_dataset(self.jsonl)
        self.assertEqual(detection.format, "embodied-jsonl")
        self.assertGreater(detection.confidence, 0.5)

    def test_detects_hdf5(self) -> None:
        if self.hdf5 is None:
            self.skipTest("h5py not installed")
        self.assertEqual(detect_dataset(self.hdf5).format, "hdf5")

    def test_detects_lerobot(self) -> None:
        if self.lerobot is None:
            self.skipTest("pyarrow not installed")
        detection = detect_dataset(self.lerobot)
        self.assertEqual(detection.format, "lerobot")
        self.assertIn(detection.version, {"v2.1", "v3.0"})

    def test_detects_rosbag2(self) -> None:
        self.assertEqual(detect_dataset(self.rosbag).format, "rosbag2")

    def test_unknown_format_raises(self) -> None:
        empty = self.root / "empty"
        empty.mkdir(exist_ok=True)
        with self.assertRaises(UnsupportedFormat):
            detect_dataset(empty)

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            detect_dataset(self.root / "nope")

    def test_summary_fields(self) -> None:
        summary = open_dataset(self.jsonl).summary
        self.assertEqual(summary.episode_count, 2)
        self.assertEqual(summary.total_frames, 60)
        self.assertAlmostEqual(summary.fps or 0, support.FPS)
        self.assertEqual(summary.robot_type, "so100")
        self.assertIn("observation.state", summary.feature_keys())
        self.assertIn("action", summary.feature_keys())

    def test_frames_are_numeric_and_aligned(self) -> None:
        ids, columns = open_dataset(self.jsonl).frames()
        self.assertEqual(len(ids), 60)
        self.assertIn("observation.state", columns)
        self.assertEqual(columns["observation.state"].shape, (60, 6))

    def test_hdf5_frames_follow_demo_count(self) -> None:
        if self.hdf5 is None:
            self.skipTest("h5py not installed")
        dataset = open_dataset(self.hdf5)
        ids, columns = dataset.frames()
        self.assertEqual(len(ids), 50)
        self.assertIn("actions", columns)

    def test_stream_formats_report_why_frames_are_unavailable(self) -> None:
        dataset = open_dataset(self.rosbag)
        self.assertIn("streams", dataset.capabilities)
        self.assertNotIn("frames", dataset.capabilities)
        with self.assertRaises(FramesUnavailable) as caught:
            dataset.frames()
        self.assertTrue(caught.exception.reason)

    def test_describe_formats_covers_registry(self) -> None:
        formats = {item["format"] for item in describe_formats()}
        self.assertTrue({"lerobot", "hdf5", "rlds", "mcap", "rosbag2", "embodied-jsonl"} <= formats)

    def test_json_alias_matches_module(self) -> None:
        self.assertEqual(open_dataset(self.jsonl, format="canonical").summary.format, "embodied-jsonl")


if __name__ == "__main__":
    unittest.main()
