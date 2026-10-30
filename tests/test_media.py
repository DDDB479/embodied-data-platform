"""Optional ffprobe media metadata and modality inventory."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from embodied_core.formats.base import guess_kind
from embodied_core.media import modality_inventory, probe_video
from embodied_core.model import DatasetSummary, FeatureSpec
from embodied_core.platform import capability_report


class MediaTestCase(unittest.TestCase):
    def test_feature_kinds_cover_depth_cloud_and_language(self) -> None:
        self.assertEqual(guess_kind("observation.depth"), "depth")
        self.assertEqual(guess_kind("lidar/point_cloud"), "point_cloud")
        self.assertEqual(guess_kind("language_instruction"), "language")
        self.assertEqual(guess_kind("observation.images.top"), "image")

    def test_modality_inventory(self) -> None:
        summary = DatasetSummary(
            path="x",
            format="test",
            features=[
                FeatureSpec("observation.depth", kind="depth"),
                FeatureSpec("points", kind="point_cloud"),
                FeatureSpec("language_instruction", kind="language"),
                FeatureSpec("observation.images.top", kind="image"),
            ],
            tasks=["pick"],
        )
        inventory = modality_inventory(summary)
        self.assertEqual(inventory["counts"]["depth"], 1)
        self.assertEqual(inventory["counts"]["point_cloud"], 1)
        self.assertEqual(inventory["tasks"], ["pick"])

    def test_capability_report_covers_decoders_and_limits(self) -> None:
        report = capability_report()
        self.assertEqual(report["version"], "0.5.0")
        self.assertIn("sensor_msgs/JointState", report["ros2_cdr"]["message_types"])
        self.assertGreater(report["ros2_cdr"]["max_message_bytes"], 0)
        self.assertTrue(report["rlds"]["light_decoder"])
        self.assertIn("s3", report["storage"]["schemes"])

    def test_ffprobe_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "episode.mp4"
            video.write_bytes(b"not-real")
            payload = {
                "format": {"duration": "2.5", "size": "100", "format_name": "mov,mp4", "bit_rate": "320000"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480, "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "nb_frames": "75"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
            }

            def runner(*args, **kwargs):  # noqa: ANN002, ANN003
                return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

            result = probe_video(video, ffprobe="ffprobe", runner=runner)
        self.assertTrue(result["available"])
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        self.assertEqual(result["duration_s"], 2.5)
        self.assertTrue(result["has_audio"])

    def test_ffprobe_absence_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "episode.mp4"
            video.write_bytes(b"x")
            with mock.patch("embodied_core.media.find_ffprobe", return_value=None):
                result = probe_video(video)
        self.assertFalse(result["available"])
        self.assertIn("install", result)

    def test_ffprobe_timeout_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "episode.mp4"
            video.write_bytes(b"x")

            def runner(*args, **kwargs):  # noqa: ANN002, ANN003
                raise TimeoutError("slow")

            result = probe_video(video, ffprobe="ffprobe", runner=runner)
        self.assertFalse(result["available"])
        self.assertIn("TimeoutError", result["reason"])


if __name__ == "__main__":
    unittest.main()
