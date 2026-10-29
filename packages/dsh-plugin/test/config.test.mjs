import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { findPythonRoot, isPythonRoot, locatePythonRoot, resolveConfig } from '../src/config.js';

function makeBundle() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-config-'));
  const pkg = path.join(root, 'python', 'embodied_core');
  fs.mkdirSync(pkg, { recursive: true });
  fs.writeFileSync(path.join(pkg, '__init__.py'), '__version__ = "test"\n');
  return root;
}

test('findPythonRoot discovers the bundled python/ layout', () => {
  const root = makeBundle();
  try {
    assert.equal(findPythonRoot(root), path.join(root, 'python'));
    assert.equal(isPythonRoot(path.join(root, 'python')), true);
    assert.deepEqual(locatePythonRoot(root), { root: path.join(root, 'python'), source: 'bundled' });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('resolveConfig prefers an explicit root and reports its source', () => {
  const root = makeBundle();
  try {
    const explicit = path.join(root, 'python');
    const config = resolveConfig({ pythonRoot: explicit }, root, { env: {}, platform: 'linux' });
    assert.equal(config.pythonRoot, explicit);
    assert.equal(config.pythonRootSource, 'config');
    assert.deepEqual(config.pythonPath, [explicit]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('resolveConfig can discover an installed package through pythonPath', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-installed-'));
  try {
    const sitePackages = path.join(root, 'site-packages');
    fs.mkdirSync(path.join(sitePackages, 'embodied_core'), { recursive: true });
    fs.writeFileSync(path.join(sitePackages, 'embodied_core', '__init__.py'), '__version__ = "test"\n');
    const config = resolveConfig({ pythonPath: [sitePackages] }, root, { env: {}, platform: 'linux' });
    assert.equal(config.pythonRoot, sitePackages);
    assert.equal(config.pythonRootSource, 'pythonPath');
    assert.deepEqual(config.pythonPath, [sitePackages]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('workspace safety defaults are conservative', () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const config = resolveConfig({}, path.resolve(here, '..'), { env: {}, platform: 'linux' });
  assert.equal(config.confirmOutsideWorkspace, true);
});
