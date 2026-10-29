"""Bounded frame iteration and streaming quality reports."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core import FrameLimitExceeded, open_dataset
from embodied_core.quality import compute_dataset_stats, detect_anomalies, merge_feature_stats, shift_anomaly_windows
from embodied_core.reports import quality_report

from tests import support


class StreamingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.jsonl = support.write_jsonl_dataset(self.root / "canonical")
        self.lerobot = support.write_lerobot_dataset(self.root / "lerobot-parent")
        self.hdf5 = support.write_hdf5_dataset(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_chunks(self, path: Path, chunk_frames: int, expected: int) -> None:
        dataset = open_dataset(path)
        try:
            chunks = list(dataset.iter_frames(chunk_frames=chunk_frames))
        finally:
            dataset.close()
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk.ids) <= chunk_frames for chunk in chunks))
        self.assertEqual(sum(len(chunk.ids) for chunk in chunks), expected)
        self.assertEqual(len({episode for chunk in chunks for episode in chunk.ids}), 2)

    def test_jsonl_chunks_are_bounded(self) -> None:
        self._assert_chunks(self.jsonl, 7, 60)

    def test_lerobot_chunks_are_bounded(self) -> None:
        if self.lerobot is None:
            self.skipTest("pyarrow not installed")
        self._assert_chunks(self.lerobot, 11, 60)

    def test_hdf5_chunks_are_bounded(self) -> None:
        if self.hdf5 is None:
            self.skipTest("h5py not installed")
        dataset = open_dataset(self.hdf5)
        try:
            chunks = list(dataset.iter_frames(chunk_frames=7))
        finally:
            dataset.close()
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.ids) <= 7 for chunk in chunks))
        self.assertEqual(sum(len(chunk.ids) for chunk in chunks), 50)

    def test_frame_limit_is_enforced(self) -> None:
        dataset = open_dataset(self.jsonl)
        try:
            with self.assertRaises(FrameLimitExceeded):
                list(dataset.iter_frames(chunk_frames=7, max_frames=10))
            with self.assertRaises(FrameLimitExceeded):
                dataset.frames(max_frames=10)
            with self.assertRaises(ValueError):
                list(dataset.iter_frames(max_frames=-1))
        finally:
            dataset.close()

    def test_quality_report_scans_in_chunks(self) -> None:
        payload = quality_report(
            self.jsonl,
            use_cache=False,
            chunk_frames=13,
            max_frames=0,
        )
        self.assertTrue(payload["scan"]["streamed"])
        self.assertGreater(payload["scan"]["chunks"], 1)
        self.assertEqual(payload["scan"]["frames_scanned"], 60)
        self.assertEqual(payload["stats"]["observation.state"]["frames"], 60)

    def test_streamed_stats_and_windows_match_full_read(self) -> None:
        dataset = open_dataset(self.jsonl)
        try:
            ids, columns = dataset.frames()
            full = compute_dataset_stats(columns)
            windows = detect_anomalies(ids, columns, fps=30.0)
            streamed_windows = []
            merged = merge_feature_stats(
                [compute_dataset_stats(chunk.columns) for chunk in dataset.iter_frames(chunk_frames=7)]
            )
            for chunk in dataset.iter_frames(chunk_frames=7):
                streamed_windows.extend(
                    shift_anomaly_windows(detect_anomalies(chunk.ids, chunk.columns, fps=30.0), chunk.offsets, fps=30.0)
                )
        finally:
            dataset.close()
        for key in full:
            self.assertAlmostEqual(full[key]["mean"], merged[key]["mean"], places=9)
            self.assertAlmostEqual(full[key]["std"], merged[key]["std"], places=9)
            self.assertEqual(full[key]["min"], merged[key]["min"])
            self.assertEqual(full[key]["max"], merged[key]["max"])
        self.assertEqual(
            [(item["episode"], item["start_frame"], item["code"]) for item in windows],
            [(item["episode"], item["start_frame"], item["code"]) for item in streamed_windows],
        )


if __name__ == "__main__":
    unittest.main()
