"""Fingerprints, cache keys and the SQLite quality cache."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core.cache import (
    QualityCache,
    dataset_fingerprint,
    normalize_quality_params,
    quality_cache_key,
)

from tests import support


class FingerprintTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fingerprint_is_stable_and_changes_with_content(self) -> None:
        first = dataset_fingerprint(self.dataset)
        second = dataset_fingerprint(self.dataset)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertGreater(first["files"], 0)
        self.assertGreater(first["bytes"], 0)

        target = self.dataset / "episodes.jsonl"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = dataset_fingerprint(self.dataset)
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_cache_files_can_be_excluded(self) -> None:
        before = dataset_fingerprint(self.dataset)
        cache = self.dataset / "quality.sqlite3"
        cache.write_bytes(b"cache")
        after = dataset_fingerprint(self.dataset, exclude=[cache])
        self.assertEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["files"], after["files"])

    def test_key_normalizes_order_and_tracks_parameters(self) -> None:
        fingerprint = dataset_fingerprint(self.dataset)["sha256"]
        first = quality_cache_key(
            self.dataset,
            fingerprint,
            {"episodes": [1, 0], "features": ["b", "a"], "z": 8, "include_segments": False},
        )
        second = quality_cache_key(
            self.dataset,
            fingerprint,
            {"episodes": [0, 1], "features": ["a", "b"], "z": 8.0, "include_segments": False},
        )
        self.assertEqual(first, second)
        different_z = quality_cache_key(self.dataset, fingerprint, {"z": 7.0})
        self.assertNotEqual(first, different_z)

    def test_normalize_quality_params(self) -> None:
        params = normalize_quality_params({"episodes": [2, 2, 1], "features": ["a", "a"], "z": 8})
        self.assertEqual(params["episodes"], [1, 2])
        self.assertEqual(params["features"], ["a"])
        self.assertEqual(params["z"], 8.0)


class QualityCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "cache.sqlite3"
        self.cache = QualityCache(self.path)
        self.dataset = self.root / "dataset"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip_counts_hits(self) -> None:
        self.cache.put(
            "key-1",
            path=self.dataset,
            fingerprint="abc",
            params={"z": 8.0},
            payload={"stats": {"action": {"frames": 3}}},
            compute_ms=12.5,
        )
        first = self.cache.get("key-1")
        second = self.cache.get("key-1")
        self.assertIsNotNone(first)
        self.assertEqual(first["payload"]["stats"]["action"]["frames"], 3)
        self.assertEqual(first["compute_ms"], 12.5)
        self.assertEqual(second["hits"], first["hits"] + 1)
        stats = self.cache.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 2)
        self.assertGreater(stats["bytes"], 0)

    def test_clear_removes_entries(self) -> None:
        self.cache.put("key-1", path=self.dataset, fingerprint="abc", params={}, payload={"ok": True})
        self.assertEqual(self.cache.clear(), 1)
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertIsNone(self.cache.get("key-1"))

    def test_max_entries_prunes_oldest(self) -> None:
        small = QualityCache(self.path, max_entries=1)
        small.put("old", path=self.dataset, fingerprint="a", params={}, payload={"n": 1})
        small.put("new", path=self.dataset, fingerprint="b", params={}, payload={"n": 2})
        self.assertEqual(small.stats()["entries"], 1)
        self.assertIsNone(small.get("old"))
        self.assertIsNotNone(small.get("new"))


if __name__ == "__main__":
    unittest.main()
