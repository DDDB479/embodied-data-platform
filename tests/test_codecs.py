"""CDR, rosbag2/MCAP and RLDS lightweight decoding."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from embodied_core import open_dataset
from embodied_core.ros2_codec import CDRError, decode_message

from tests import support


class CDRTestCase(unittest.TestCase):
    def test_joint_state_and_instruction(self) -> None:
        decoded = decode_message(
            "sensor_msgs/msg/JointState",
            support.encode_cdr_joint_state(positions=[0.25, 0.5, 0.75]),
        )
        self.assertEqual(decoded.text["name"], "joint_0,joint_1,joint_2")
        self.assertEqual(decoded.numeric["position"].tolist(), [0.25, 0.5, 0.75])

        text = decode_message("std_msgs/msg/String", support.encode_cdr_string("pick up the cube"))
        self.assertEqual(text.text["data"], "pick up the cube")

    def test_depth_and_pointcloud(self) -> None:
        depth = decode_message("sensor_msgs/msg/Image", support.encode_cdr_depth_image())
        self.assertEqual(depth.meta["kind"], "depth")
        self.assertEqual(depth.arrays["depth"].shape, (3, 4))
        self.assertGreater(depth.numeric["depth_max"], 0)

        cloud = decode_message("sensor_msgs/msg/PointCloud2", support.encode_cdr_pointcloud2(6))
        self.assertEqual(cloud.numeric["point_count"], 6)
        self.assertEqual(cloud.arrays["xyz"].shape, (6, 3))
        self.assertEqual(cloud.numeric["x_max"], 5.0)

    def test_message_size_limit_is_enforced(self) -> None:
        payload = support.encode_cdr_joint_state(positions=[1.0, 2.0])
        with mock.patch.dict("os.environ", {"EMBODIED_MAX_MESSAGE_BYTES": "8"}):
            with self.assertRaises(CDRError):
                decode_message("sensor_msgs/msg/JointState", payload)


class StreamDecodeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rosbag2_cdr_frames_and_samples(self) -> None:
        path = support.write_rosbag2_cdr(self.root, messages=12)
        dataset = open_dataset(path)
        try:
            self.assertIn("frames", dataset.capabilities)
            self.assertIn("cdr", dataset.capabilities)
            ids, columns = dataset.frames(features=["/joint_states"])
            self.assertEqual(len(ids), 12)
            self.assertIn("/joint_states.position", columns)
            self.assertEqual(columns["/joint_states.position"].shape, (12, 3))
            self.assertEqual(columns["/joint_states.position"][-1, 0], 11.0)

            language = dataset.topic_samples("/instruction")
            self.assertEqual(language["samples"][0]["text"]["data"], "pick up the red cube")
            depth = dataset.topic_samples("/depth")
            self.assertEqual(depth["samples"][0]["meta"]["kind"], "depth")
            points = dataset.topic_samples("/points", include_arrays=True)
            self.assertEqual(points["samples"][0]["numeric"]["point_count"], 4)
        finally:
            dataset.close()

    def test_mcap_cdr_frames(self) -> None:
        path = support.write_mcap_cdr(self.root, messages=8)
        if path is None:
            self.skipTest("mcap not installed")
        dataset = open_dataset(path)
        try:
            self.assertIn("frames", dataset.capabilities)
            ids, columns = dataset.frames(features=["/joint_states"])
            self.assertEqual(len(ids), 8)
            self.assertEqual(columns["/joint_states.position"].shape, (8, 2))
            self.assertEqual(columns["/joint_states.position"][-1, 0], 7.0)
        finally:
            dataset.close()

    def test_rlds_sequence_frames_and_language(self) -> None:
        path = support.write_rlds_sequence(self.root, episodes=2, steps=5)
        dataset = open_dataset(path)
        try:
            self.assertIn("frames", dataset.capabilities)
            self.assertAlmostEqual(dataset.summary.fps or 0.0, 10.0, places=6)
            ids, columns = dataset.frames()
            self.assertEqual(len(ids), 10)
            self.assertEqual(sorted(set(ids)), [0, 1])
            self.assertEqual(columns["steps/observation/state"].shape, (10, 3))
            self.assertEqual(columns["steps/action"][0, 0], 0.0)
            sample = dataset.episode_samples(episode=0)
            self.assertEqual(sample["samples"][0]["texts"]["steps/language_instruction"], "pick up the cube | pick up the cube | pick up the cube | pick up the cube | pick up the cube")
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()
