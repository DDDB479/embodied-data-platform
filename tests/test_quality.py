"""Statistics, anomaly detection, segmentation and labeling tasks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from embodied_core import open_dataset
from embodied_core.quality import (
    basic_issue_scan,
    build_labeling_tasks,
    compute_feature_stats,
    detect_anomalies,
    make_windows,
    segment_by_motion,
)

from tests import support


def _ramp(length: int = 200, dims: int = 3) -> np.ndarray:
    """Half idle at zero, then a constant-velocity ramp."""
    values = np.zeros((length, dims), dtype=np.float64)
    values[length // 2 :] = np.linspace(0.0, 4.0, length - length // 2)[:, None]
    return values


def _sine(length: int = 200) -> np.ndarray:
    """Always moving, smooth: the negative control for anomaly detection."""
    phase = np.linspace(0.0, 6.0 * np.pi, length)
    return np.stack([np.sin(phase), 2.0 * np.sin(phase)], axis=1)


class StatsTestCase(unittest.TestCase):
    def test_feature_stats_capture_nan_and_range(self) -> None:
        column = np.asarray([[0.0, 1.0], [np.nan, 2.0], [4.0, 3.0]])
        stats = compute_feature_stats({"action": column})["action"]
        self.assertEqual(stats["frames"], 3)
        self.assertEqual(stats["dims"], 2)
        self.assertEqual(stats["nan_count"], 1)
        self.assertAlmostEqual(stats["nan_ratio"], 1 / 6, places=3)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 4.0)

    def test_constant_feature_is_flagged(self) -> None:
        codes = {issue.code for issue in basic_issue_scan({"hold": np.ones((50, 2))})}
        self.assertIn("feature.constant", codes)

    def test_non_finite_ratio_triggers_issue(self) -> None:
        column = np.zeros((100, 2))
        column[:20] = np.nan
        codes = {issue.code for issue in basic_issue_scan({"pose": column})}
        self.assertIn("feature.non_finite", codes)

    def test_clean_columns_produce_no_issues(self) -> None:
        self.assertEqual(basic_issue_scan({"action": _ramp()}), [])


class AnomalyTestCase(unittest.TestCase):
    def test_injected_jump_is_detected(self) -> None:
        column = _ramp(200, 2)
        column[120] += 500.0
        windows = detect_anomalies([0] * 200, {"action": column}, fps=30.0)
        self.assertIn("jump", {window["code"] for window in windows})
        for window in windows:
            self.assertLessEqual(window["start_frame"], window["end_frame"])
            self.assertIn("start_s", window)

    def test_freeze_inside_motion_is_a_stall(self) -> None:
        column = _ramp(200, 2)
        column[140:160] = column[139]
        windows = detect_anomalies([0] * 200, {"action": column})
        stalls = [window for window in windows if window["code"] == "stall"]
        self.assertTrue(stalls, windows)
        self.assertTrue(any(window["start_frame"] >= 100 for window in stalls), stalls)

    def test_smooth_motion_has_no_windows(self) -> None:
        self.assertEqual(detect_anomalies([0] * 200, {"action": _sine()}), [])

    def test_non_finite_span_is_an_error(self) -> None:
        column = _sine()
        column[30:35] = np.nan
        windows = detect_anomalies([0] * 200, {"action": column})
        non_finite = [window for window in windows if window["code"] == "non_finite"]
        self.assertTrue(non_finite)
        self.assertEqual(non_finite[0]["severity"], "error")


class SegmentTestCase(unittest.TestCase):
    def test_ramp_splits_into_idle_and_moving(self) -> None:
        segments = segment_by_motion([0] * 200, {"action": _ramp()}, fps=30.0)
        self.assertIn(0, segments)
        labels = [segment["label"] for segment in segments[0]]
        self.assertIn("moving", labels)
        self.assertGreaterEqual(len(segments[0]), 2)

    def test_min_segment_filters_short_runs(self) -> None:
        self.assertEqual(segment_by_motion([0] * 20, {"action": _ramp(20, 2)}, min_segment=50).get(0, []), [])


class WindowsTestCase(unittest.TestCase):
    def test_window_count_follows_stride(self) -> None:
        windows = make_windows([100], window=10, stride=10)
        self.assertEqual(len(windows), 10)
        self.assertEqual(windows[0]["start_frame"], 0)
        self.assertEqual(windows[1]["start_frame"], 10)
        self.assertEqual(windows[0]["end_frame"], 10)

    def test_tail_is_not_emitted_partial(self) -> None:
        self.assertEqual(len(make_windows([25], window=10, stride=10)), 2)


class LabelingTestCase(unittest.TestCase):
    def test_slicing_plan_can_be_annotated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = support.write_jsonl_dataset(Path(tmp) / "canonical")
            dataset = open_dataset(path)
            lengths = [record.length or 0 for record in dataset.summary.episodes]
            windows = make_windows(lengths, window=10, stride=10)
            bundle = build_labeling_tasks(dataset.summary, windows, max_tasks=5)
        self.assertLessEqual(len(bundle["tasks"]), 5)
        self.assertEqual(bundle["task_count"], len(bundle["tasks"]))
        self.assertIn("instructions", bundle)
        first = bundle["tasks"][0]
        self.assertEqual(first["kind"], "verify_window")
        self.assertLess(first["start_frame"], first["end_frame"])

    def test_anomaly_windows_keep_their_cues(self) -> None:
        column = _ramp(200, 2)
        column[120] += 500.0
        windows = detect_anomalies([0] * 200, {"action": column})
        with tempfile.TemporaryDirectory() as tmp:
            path = support.write_jsonl_dataset(Path(tmp) / "canonical")
            dataset = open_dataset(path)
            bundle = build_labeling_tasks(dataset.summary, windows)
        cues = [cue for task in bundle["tasks"] for cue in task["cues"]]
        self.assertIn("jump", cues)


if __name__ == "__main__":
    unittest.main()
