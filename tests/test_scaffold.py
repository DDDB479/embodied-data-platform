"""Project scaffolding: the skeleton a platform team starts from."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embodied_core.scaffold import scaffold


class ScaffoldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_platform_layout(self) -> None:
        result = scaffold(self.root / "platform", kind="platform", name="acme-platform")
        self.assertEqual(result["kind"], "platform")
        self.assertEqual(result["package"], "acme_platform")
        for relative in (
            "README.md",
            "docs/data-contract.md",
            "src/acme_platform/__init__.py",
            "src/acme_platform/api.py",
            "src/acme_platform/ingest.py",
            "src/acme_platform/quality.py",
            "tests/test_contract.py",
            "scaffold.json",
        ):
            self.assertTrue((self.root / "platform" / relative).is_file(), relative)

    def test_generated_python_files_compile(self) -> None:
        scaffold(self.root / "platform", kind="platform", name="acme-platform")
        for path in (self.root / "platform").rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_templates_keep_single_braces(self) -> None:
        scaffold(self.root / "platform", kind="platform", name="acme-platform")
        ingest = (self.root / "platform" / "src" / "acme_platform" / "ingest.py").read_text(encoding="utf-8")
        self.assertNotIn("{{", ingest)
        self.assertIn("argparse", ingest)

    def test_adapter_and_pipeline_kinds(self) -> None:
        adapter = scaffold(self.root / "adapter", kind="adapter", name="my_robot")
        self.assertIn("my_adapter.py", {Path(p).name for p in adapter["files"]})
        pipeline = scaffold(self.root / "pipeline", kind="pipeline", name="daily")
        self.assertIn("tests/test_ingest.py", {Path(p).as_posix() for p in pipeline["files"]})
        self.assertNotEqual(
            {Path(p).name for p in adapter["files"]},
            {Path(p).name for p in pipeline["files"]},
        )

    def test_existing_directory_needs_overwrite(self) -> None:
        target = self.root / "platform"
        target.mkdir()
        keep = target / "keep.txt"
        keep.write_text("user data", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            scaffold(target, kind="platform")
        scaffold(target, kind="platform", overwrite=True)
        self.assertTrue((target / "README.md").is_file())
        self.assertTrue(keep.is_file(), "overwrite=True must not wipe user files")

    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scaffold(self.root / "x", kind="not-a-kind")

    def test_next_steps_are_actionable(self) -> None:
        result = scaffold(self.root / "platform", kind="platform", name="acme-platform")
        steps = result["next_steps"]
        self.assertTrue(steps)
        self.assertTrue(any("acme_platform" in step for step in steps), steps)


if __name__ == "__main__":
    unittest.main()
