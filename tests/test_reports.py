"""Incremental quality reports and their progress callbacks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core.cache import QualityCache
from embodied_core.reports import quality_report

from tests import support


class QualityReportCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = support.write_jsonl_dataset(self.root / "canonical")
        self.cache_path = self.root / "quality-cache.sqlite3"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_second_report_is_a_cache_hit(self) -> None:
        first = quality_report(self.dataset, cache_path=self.cache_path)
        second = quality_report(self.dataset, cache_path=self.cache_path)
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["cache"]["hits"], 1)
        self.assertEqual(first["stats"], second["stats"])
        self.assertGreaterEqual(second["cache"]["age_s"], 0)
        self.assertEqual(QualityCache(self.cache_path).stats()["entries"], 1)

    def test_parameter_change_misses(self) -> None:
        quality_report(self.dataset, cache_path=self.cache_path, z=8.0)
        changed = quality_report(self.dataset, cache_path=self.cache_path, z=7.0)
        self.assertFalse(changed["cache"]["hit"])

    def test_dataset_change_misses(self) -> None:
        quality_report(self.dataset, cache_path=self.cache_path)
        target = self.dataset / "episodes.jsonl"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = quality_report(self.dataset, cache_path=self.cache_path)
        self.assertFalse(changed["cache"]["hit"])

    def test_refresh_and_no_cache(self) -> None:
        quality_report(self.dataset, cache_path=self.cache_path)
        refreshed = quality_report(self.dataset, cache_path=self.cache_path, refresh_cache=True)
        self.assertFalse(refreshed["cache"]["hit"])
        uncached = quality_report(self.dataset, cache_path=self.cache_path, use_cache=False)
        self.assertFalse(uncached["cache"]["enabled"])
        self.assertFalse(uncached["cache"]["hit"])

    def test_progress_is_monotonic_and_completes(self) -> None:
        events: list[tuple[float, str]] = []
        quality_report(
            self.dataset,
            use_cache=False,
            include_segments=True,
            progress=lambda value, message: events.append((value, message)),
        )
        values = [value for value, _ in events]
        self.assertGreater(values[0], 0)
        self.assertEqual(values[-1], 100.0)
        self.assertEqual(values, sorted(values))
        self.assertIn("detecting anomalies", [message for _, message in events])

    def test_cache_hit_still_reports_progress(self) -> None:
        quality_report(self.dataset, cache_path=self.cache_path)
        events: list[tuple[float, str]] = []
        quality_report(
            self.dataset,
            cache_path=self.cache_path,
            progress=lambda value, message: events.append((value, message)),
        )
        self.assertEqual(events[-1], (100.0, "cache hit"))


if __name__ == "__main__":
    unittest.main()
