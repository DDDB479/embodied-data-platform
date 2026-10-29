"""Packaging invariants: one version, and the npm tarball must contain Python."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import embodied_core

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "dsh-plugin" / "package.json"


class PackagingTestCase(unittest.TestCase):
    def test_python_and_npm_versions_match(self) -> None:
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', metadata, re.MULTILINE)
        self.assertIsNotNone(match)
        package = json.loads(PACKAGE.read_text(encoding="utf-8"))
        self.assertEqual(match.group(1), embodied_core.__version__)
        self.assertEqual(match.group(1), package["version"])

    def test_npm_package_bundles_the_python_tree(self) -> None:
        package = json.loads(PACKAGE.read_text(encoding="utf-8"))
        self.assertIn("python", package["files"])
        scripts = package["scripts"]
        self.assertIn("prepack", scripts)
        self.assertIn("postpack", scripts)
        self.assertTrue((ROOT / "packages" / "dsh-plugin" / "scripts" / "prepack.mjs").is_file())
        self.assertTrue((ROOT / "packages" / "dsh-plugin" / "scripts" / "postpack.mjs").is_file())

    def test_plugin_exports_its_runtime_locator(self) -> None:
        source = (ROOT / "packages" / "dsh-plugin" / "index.js").read_text(encoding="utf-8")
        self.assertIn("isPythonRoot", source)
        self.assertIn("locatePythonRoot", source)


if __name__ == "__main__":
    unittest.main()
