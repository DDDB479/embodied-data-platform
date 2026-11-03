"""Delivery packs, manifests, checksums and dependency-free Zarr export."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_core import open_dataset
from embodied_core.convert import to_zarr
from embodied_core.delivery import build_delivery, check_delivery

from tests import support


class DeliveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_zarr_export_is_valid_v2_layout(self) -> None:
        target = self.root / "zarr"
        dataset = open_dataset(self.dataset)
        try:
            result = to_zarr(dataset, target, chunk_rows=20)
        finally:
            dataset.close()
        self.assertEqual(result["frames"], 60)
        self.assertTrue((target / ".zgroup").is_file())
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["format"], "zarr-v2")
        feature = meta["features"]["observation.state"]
        self.assertEqual(feature["shape"], [60, 6])
        array_meta = json.loads((target / feature["path"] / ".zarray").read_text(encoding="utf-8"))
        self.assertEqual(array_meta["chunks"], [20, 6])
        self.assertTrue((target / feature["path"] / "0.0").is_file())

    def test_build_and_check_delivery_pack(self) -> None:
        target = self.root / "delivery"
        result = build_delivery(
            self.dataset,
            target,
            formats=["jsonl", "zarr", "mcap-inventory"],
            name="demo-pack",
            version="2026.11",
            license="CC-BY-4.0",
            quality_rules={"expected_fps": 30},
        )
        self.assertEqual(result["formats"], ["jsonl", "zarr", "mcap-inventory"])
        self.assertGreaterEqual(result["artifacts"], 3)
        manifest = json.loads((target / "delivery-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "demo-pack")
        self.assertEqual(manifest["license"], "CC-BY-4.0")
        self.assertTrue(manifest["quality_gate"]["ok"])
        checked = check_delivery(target)
        self.assertTrue(checked["ok"], checked)
        self.assertGreater(checked["files_checked"], 0)

    def test_tampered_file_fails_check(self) -> None:
        target = self.root / "delivery"
        build_delivery(self.dataset, target, formats=["jsonl"])
        artifact = target / "artifacts" / "jsonl" / "dataset.json"
        artifact.write_text(artifact.read_text(encoding="utf-8") + " ", encoding="utf-8")
        checked = check_delivery(target)
        self.assertFalse(checked["ok"])
        self.assertIn("size_mismatch", {item["code"] for item in checked["issues"]})


if __name__ == "__main__":
    unittest.main()
