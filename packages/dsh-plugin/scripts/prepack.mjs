#!/usr/bin/env node
/**
 * Copy the Python package into the npm tarball as `python/embodied_core`.
 *
 * The repository keeps one Python source of truth under `src/embodied_core`.
 * npm consumers should not need a git checkout (or a separate pip install), so
 * `prepack` stages that tree next to the JS bridge. `postpack` removes it.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PLUGIN_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const REPO_ROOT = path.resolve(PLUGIN_ROOT, '..', '..');
const SOURCE = path.join(REPO_ROOT, 'src', 'embodied_core');
const BUNDLE_ROOT = path.join(PLUGIN_ROOT, 'python');
const DESTINATION = path.join(BUNDLE_ROOT, 'embodied_core');

function excluded(source) {
  return /(^|[\\/])(__pycache__|\.pytest_cache|\.mypy_cache)([\\/]|$)/.test(source) || /\.(pyc|pyo|pyd)$/i.test(source);
}

if (!fs.existsSync(path.join(SOURCE, '__init__.py'))) {
  console.error(`[dsh-embodied-data] Python source not found: ${SOURCE}`);
  process.exit(1);
}

fs.rmSync(BUNDLE_ROOT, { recursive: true, force: true });
fs.mkdirSync(DESTINATION, { recursive: true });
fs.cpSync(SOURCE, DESTINATION, { recursive: true, filter: (source) => !excluded(source) });

// Keep the package metadata with the bundle for provenance and version checks.
fs.copyFileSync(path.join(REPO_ROOT, 'pyproject.toml'), path.join(BUNDLE_ROOT, 'pyproject.toml'));
fs.copyFileSync(path.join(REPO_ROOT, 'LICENSE'), path.join(BUNDLE_ROOT, 'LICENSE'));

const files = fs.readdirSync(DESTINATION, { recursive: true }).filter((name) => String(name).endsWith('.py')).length;
console.log(`[dsh-embodied-data] bundled ${files} Python module(s) into python/embodied_core`);
