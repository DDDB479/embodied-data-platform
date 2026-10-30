#!/usr/bin/env node
/**
 * Package a clean install and prove it works outside the source checkout.
 *
 * Steps: npm/pnpm pack -> install the tarball into a temporary app -> resolve
 * the bundled Python root -> run the installed probe against a real fixture.
 *
 * Usage:
 *   node tools/check_clean_install.mjs --python /path/to/python \
 *     --fixture .devtools/fixtures/lerobot_v21 [--deps .devtools/pydeps]
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(HERE);
const PLUGIN_ROOT = path.join(REPO_ROOT, 'packages', 'dsh-plugin');

function parseArgs(argv) {
  const out = { python: '', fixture: '', deps: '', keep: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--python') out.python = argv[++i] ?? '';
    else if (arg === '--fixture') out.fixture = argv[++i] ?? '';
    else if (arg === '--deps') out.deps = argv[++i] ?? '';
    else if (arg === '--keep') out.keep = true;
  }
  return out;
}

function quoteCmd(value) {
  const text = String(value);
  if (text === '') return '""';
  if (!/[ \t"&|<>^()%!]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function run(command, args, cwd) {
  const options = { cwd, encoding: 'utf8', env: { ...process.env, CI: '1' } };
  if (process.platform === 'win32') {
    const line = [command, ...args].map(quoteCmd).join(' ');
    return spawnSync(process.env.ComSpec || 'cmd.exe', ['/d', '/s', '/c', line], options);
  }
  return spawnSync(command, args, options);
}

function fail(message, detail = '') {
  console.error(`clean-install: ${message}`);
  if (detail) console.error(detail.trim());
  process.exit(1);
}

function pickPackager() {
  for (const command of ['npm', 'pnpm']) {
    const result = run(command, ['--version'], PLUGIN_ROOT);
    if (result.status === 0) return command;
  }
  fail('neither npm nor pnpm is available on PATH');
}

function packedTarball(stdout, packDir) {
  // pnpm writes lifecycle logs before the JSON document; try every object/array
  // boundary from the end so a leading `[dsh-embodied-data]` banner is harmless.
  const starts = [];
  for (let index = 0; index < stdout.length; index += 1) {
    if (stdout[index] === '[' || stdout[index] === '{') starts.push(index);
  }
  let parsed = null;
  for (const start of starts.reverse()) {
    try {
      parsed = JSON.parse(stdout.slice(start));
      break;
    } catch {
      // Keep looking for the document start.
    }
  }
  if (!parsed) throw new Error(`unexpected pack output: ${stdout.slice(0, 400)}`);
  const entry = Array.isArray(parsed) ? parsed[0] : parsed;
  const filename = entry?.filename ?? entry?.tarball ?? entry?.path;
  if (!filename) throw new Error(`pack output has no filename: ${stdout.slice(0, 400)}`);
  return path.resolve(packDir, filename);
}

const args = parseArgs(process.argv.slice(2));
const fixture = path.resolve(REPO_ROOT, args.fixture || path.join('.devtools', 'fixtures', 'lerobot_v21'));
if (!fs.existsSync(fixture)) fail(`fixture not found: ${fixture} (run python tools/make_fixtures.py)`);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-clean-install-'));
const app = path.join(tmp, 'app');
let tarball = '';

try {
  const packager = pickPackager();
  const pack = run(packager, ['pack', '--json', '--pack-destination', tmp], PLUGIN_ROOT);
  if (pack.status !== 0) fail(`${packager} pack failed`, `${pack.stdout}\n${pack.stderr}`);
  tarball = packedTarball(pack.stdout, tmp);
  if (!fs.existsSync(tarball)) fail(`packed tarball missing: ${tarball}`);

  fs.mkdirSync(app, { recursive: true });
  fs.writeFileSync(path.join(app, 'package.json'), JSON.stringify({ name: 'embodied-clean-install-check', private: true, type: 'module' }, null, 2));

  const installArgs = packager === 'npm'
    ? ['install', '--no-audit', '--no-fund', '--ignore-scripts', tarball]
    : ['add', '--ignore-scripts', tarball];
  const install = run(packager, installArgs, app);
  if (install.status !== 0) fail(`${packager} install failed`, `${install.stdout}\n${install.stderr}`);

  const installedRoot = path.join(app, 'node_modules', 'dsh-embodied-data');
  const initFile = path.join(installedRoot, 'python', 'embodied_core', '__init__.py');
  if (!fs.existsSync(initFile)) fail(`tarball did not include the Python bundle: ${initFile}`);

  const { resolveConfig, isPythonRoot } = await import(pathToFileURL(path.join(installedRoot, 'index.js')).href);
  const config = resolveConfig({ python: args.python }, installedRoot, { env: {}, platform: process.platform });
  if (!config.pythonRoot || config.pythonRootSource !== 'bundled') {
    fail(`installed package did not resolve its bundled Python root: ${JSON.stringify({ root: config.pythonRoot, source: config.pythonRootSource })}`);
  }
  if (!isPythonRoot(config.pythonRoot)) fail(`resolved Python root is invalid: ${config.pythonRoot}`);

  const probeArgs = [path.join(installedRoot, 'scripts', 'probe.mjs'), '--json', '--python', args.python || (process.platform === 'win32' ? 'python' : 'python3'), '--fixture', fixture];
  if (args.deps) probeArgs.push('--deps', path.resolve(REPO_ROOT, args.deps));
  const probe = run(process.execPath, probeArgs, installedRoot);
  if (probe.status !== 0) fail('installed package probe failed', `${probe.stdout}\n${probe.stderr}`);
  const report = JSON.parse(probe.stdout);
  if (!report.ok || report.toolCount < 24) fail(`installed probe returned an unhealthy report: ${probe.stdout}`);

  console.log('clean-install: ok');
  console.log(`  packager     ${packager}`);
  console.log(`  tarball      ${path.basename(tarball)}`);
  console.log(`  python root  ${config.pythonRoot}`);
  console.log(`  tools        ${report.toolCount}`);
} finally {
  if (!args.keep) fs.rmSync(tmp, { recursive: true, force: true });
  else console.log(`clean-install: kept ${tmp}`);
}
