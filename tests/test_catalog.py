"""The dataset catalog: incremental indexing and querying."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from embodied_core import catalog

from tests import support


def _names(hits: list[dict]) -> list[str]:
    return sorted(Path(hit["path"]).name for hit in hits)


class CatalogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tree = self.root / "datasets"
        self.db = self.root / "catalog.db"
        support.write_jsonl_dataset(self.tree / "pick-cube")
        support.write_jsonl_dataset(self.tree / "place-cube", episodes=3)
        support.write_rosbag2(self.tree / "drive-bag")
        support.write_lerobot_dataset(self.tree / "push-block")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _index(self, **kwargs):
        return catalog.index(self.tree, self.db, **kwargs)

    def test_index_discovers_nested_datasets(self) -> None:
        report = self._index()
        self.assertEqual(report["failed"], 0, report["failures"])
        self.assertGreaterEqual(report["total"], 3, report)

    def test_index_is_incremental(self) -> None:
        self._index()
        second = self._index()
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["updated"], 0)
        self.assertGreaterEqual(second["skipped"], 3)

        forced = self._index(force=True)
        self.assertEqual(forced["skipped"], 0)

    def test_appended_episodes_are_reindexed(self) -> None:
        self._index()
        target = self.tree / "pick-cube" / "episodes.jsonl"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        report = self._index()
        self.assertGreaterEqual(report["updated"], 1, report)

    def test_search_filters(self) -> None:
        self._index()
        self.assertGreaterEqual(len(catalog.search(self.db)), 3)
        self.assertEqual(_names(catalog.search(self.db, text="place")), ["place-cube"])
        many = catalog.search(self.db, min_episodes=3)
        self.assertTrue(many)
        self.assertTrue(all(hit["episodes"] >= 3 for hit in many), many)
        jsonl_only = catalog.search(self.db, fmt="embodied-jsonl")
        self.assertTrue(jsonl_only)
        self.assertTrue(all(hit["format"] == "embodied-jsonl" for hit in jsonl_only))
        ordered = catalog.search(self.db, order_by="episodes", descending=True)
        self.assertEqual(ordered[0]["episodes"], max(hit["episodes"] for hit in ordered))
        limited = catalog.search(self.db, limit=1)
        self.assertEqual(len(limited), 1)

    def test_stats(self) -> None:
        self._index()
        report = catalog.stats(self.db)
        self.assertGreaterEqual(report["total_datasets"], 3)
        self.assertIn("embodied-jsonl", report["by_format"])
        self.assertGreater(report["total_frames"], 0)
        self.assertGreater(report["total_episodes"], 0)

    def test_remove_drops_only_the_row(self) -> None:
        self._index()
        path = str((self.tree / "pick-cube").resolve())
        self.assertTrue(catalog.remove(self.db, path))
        self.assertNotIn("pick-cube", _names(catalog.search(self.db)))
        self.assertTrue((self.tree / "pick-cube" / "episodes.jsonl").is_file())

    def test_manifest_is_serialisable(self) -> None:
        self._index()
        manifest = catalog.export_manifest(self.db)
        json.dumps(manifest)
        self.assertIn("datasets", manifest)

    def test_validate_on_index_records_scores(self) -> None:
        self._index(validate=True)
        scored = [hit for hit in catalog.search(self.db) if hit.get("score") is not None]
        self.assertTrue(scored)

    def test_stats_without_database_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            catalog.stats(self.root / "missing.db")

    def test_schema_v2_records_versions_runs_and_lineage(self) -> None:
        first = self._index()
        self.assertEqual(first["schema_version"], 2)
        self.assertGreaterEqual(first["versions_added"], 3)
        stats = catalog.stats(self.db)
        self.assertEqual(stats["schema_version"], 2)
        self.assertGreaterEqual(stats["total_versions"], stats["total_datasets"])
        self.assertGreaterEqual(stats["total_runs"], 1)

        target = self.tree / "pick-cube" / "episodes.jsonl"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = self._index()
        self.assertGreaterEqual(changed["versions_added"], 1)
        self.assertGreaterEqual(changed["lineage_added"], 1)

        rows = catalog.versions(self.db, path=self.tree / "pick-cube")
        self.assertGreaterEqual(len(rows), 2)
        edges = catalog.lineage(self.db, path=self.tree / "pick-cube")
        self.assertTrue(edges)
        self.assertEqual(edges[0]["relation"], "supersedes")
        self.assertGreaterEqual(len(catalog.runs(self.db)), 2)

    def test_history_combines_tables(self) -> None:
        self._index()
        body = catalog.history(self.db, limit=10)
        self.assertEqual(body["schema_version"], 2)
        self.assertTrue(body["versions"])
        self.assertTrue(body["runs"])

    def test_v1_database_is_migrated_in_place(self) -> None:
        db = self.root / "legacy.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(
                """
                CREATE TABLE datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL, format TEXT NOT NULL,
                    version TEXT, episodes INTEGER, frames INTEGER, duration_s REAL, fps REAL,
                    robot_type TEXT, tasks TEXT, features TEXT, notes TEXT, meta TEXT, score INTEGER,
                    marker TEXT, mtime REAL, size INTEGER, indexed_at REAL
                );
                """
            )
            conn.execute(
                "INSERT INTO datasets(path, format, version, episodes, frames, duration_s, fps, robot_type, "
                "tasks, features, notes, meta, score, marker, mtime, size, indexed_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str((self.tree / "pick-cube").resolve()), "embodied-jsonl", "1.0", 2, 60, 2.0, 30.0,
                    "so100", "[]", "[]", "[]", "{}", 90, "marker", time.time(), 100, time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        migrated = catalog.connect(db)
        try:
            version = migrated.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)
        finally:
            migrated.close()
        stats = catalog.stats(db)
        self.assertEqual(stats["schema_version"], 2)
        self.assertEqual(stats["total_datasets"], 1)
        self.assertGreaterEqual(stats["total_versions"], 1)
        self.assertGreaterEqual(stats["total_runs"], 1)
        found = catalog.search(db)
        self.assertEqual(found[0]["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
