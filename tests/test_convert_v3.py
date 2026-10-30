"""LeRobot v3 export: stats, splits and a re-openable dataset."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_core import open_dataset, validate
from embodied_core.convert import convert

from tests import support


class LerobotV3ExportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = support.write_jsonl_dataset(self.root / "canonical")
        self.target = self.root / "v3"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_export_contains_stats_splits_and_reopens(self) -> None:
        if not support.has_pyarrow():
            self.skipTest("pyarrow not installed")
        result = convert(
            self.source,
            self.target,
            target="lerobot-v3",
            train_ratio=0.5,
            validation_ratio=0.5,
        )
        self.assertEqual(result["target"], "lerobot(v3.0)")
        self.assertTrue((self.target / "meta" / "stats.json").is_file())
        self.assertTrue((self.target / "meta" / "episodes" / "chunk-000" / "file-000.parquet").is_file())
        self.assertTrue((self.target / "data" / "chunk-000" / "file-000.parquet").is_file())
        info = json.loads((self.target / "meta" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["codebase_version"], "v3.0")
        self.assertEqual(info["splits"], {"train": "0:1", "validation": "1:2"})
        stats = json.loads((self.target / "meta" / "stats.json").read_text(encoding="utf-8"))
        self.assertIn("observation.state", stats)
        self.assertEqual(len(stats["action"]["mean"]), 6)
        self.assertTrue(result["validation"]["ok"], result["validation"])

        dataset = open_dataset(self.target)
        try:
            ids, columns = dataset.frames()
        finally:
            dataset.close()
        self.assertEqual(len(ids), 60)
        self.assertIn("observation.state", columns)

    def test_overwrite_is_atomic_and_replaces_previous_files(self) -> None:
        if not support.has_pyarrow():
            self.skipTest("pyarrow not installed")
        convert(self.source, self.target, target="lerobot-v3")
        stale = self.target / "stale.txt"
        stale.write_text("old", encoding="utf-8")
        convert(self.source, self.target, target="lerobot-v3", overwrite=True)
        self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
