/**
 * Runtime configuration for the embodied data plugin.
 *
 * The plugin is a thin adapter: every capability lives in the Python
 * `embodied-core` package, which is exposed twice — as the `embodied` CLI and
 * as an MCP stdio server (`embodied-mcp`). This module resolves *how* to reach
 * that server: which interpreter, which import path, which timeouts.
 *
 * Everything here is deliberately dependency-free and side-effect-free so it
 * can be unit tested without cordis, DSH, or a live Python process.
 */
import fs from 'node:fs';
import path from 'node:path';

/** Default module entry point of the bundled MCP server. */
const DEFAULT_ARGS = ['-m', 'embodied_core.mcp_server'];

/** How many parent directories to inspect when hunting for the Python package. */
const SEARCH_DEPTH = 6;

function asStringArray(value) {
  if (!Array.isArray(value)) return [];
  return value.filter((item) => typeof item === 'string' && item.length > 0);
}

function positiveNumber(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * A Python source root is any directory containing `src/embodied_core/__init__.py`
 * or `embodied_core/__init__.py` (installed layout).
 */
export function isPythonRoot(candidate) {
  const roots = [path.join(candidate, 'src', 'embodied_core'), path.join(candidate, 'embodied_core')];
  return roots.some((dir) => fs.existsSync(path.join(dir, '__init__.py')));
}

/**
 * Walk up from `start` looking for the Python package. This is what lets the
 * plugin work both from a repo checkout (sibling `src/`) and from an installed
 * npm package that ships the Python tree under `python/`.
 */
export function findPythonRoot(start) {
  const candidates = [start];
  for (let i = 0; i < SEARCH_DEPTH; i += 1) {
    const parent = path.dirname(candidates[candidates.length - 1]);
    if (parent === candidates[candidates.length - 1]) break;
    candidates.push(parent);
  }
  const nested = [];
  for (const base of candidates) nested.push(path.join(base, 'python'), path.join(base, 'python', 'core'));
  for (const candidate of [...candidates, ...nested]) {
    if (isPythonRoot(candidate)) return candidate;
  }
  return null;
}

/**
 * Import path entries for a Python root: the source tree plus an optional
 * vendored dependency directory (used by offline / sandboxed installs where
 * `pip install` is unavailable).
 */
export function importPathFor(pythonRoot) {
  if (!pythonRoot) return [];
  const entries = [];
  for (const relative of ['src', 'core', '.deps', 'vendor']) {
    const dir = path.join(pythonRoot, relative);
    if (fs.existsSync(dir)) entries.push(dir);
  }
  if (entries.length === 0) entries.push(pythonRoot);
  return entries;
}

/**
 * Normalise raw cordis config into the shape the rest of the plugin consumes.
 * Unknown keys are ignored rather than rejected: a stale config must not brick
 * the agent session.
 */
export function resolveConfig(raw, pluginRoot, options = {}) {
  const config = raw && typeof raw === 'object' ? raw : {};
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;

  const explicitRoot = typeof config.pythonRoot === 'string' && config.pythonRoot.trim() ? path.resolve(config.pythonRoot) : '';
  const pythonRoot = explicitRoot && isPythonRoot(explicitRoot) ? explicitRoot : findPythonRoot(pluginRoot);

  const interpreter = typeof config.python === 'string' && config.python.trim() ? config.python.trim() : '';
  const candidates = [];
  if (interpreter) candidates.push(interpreter);
  if (env.EMBODIED_PYTHON) candidates.push(env.EMBODIED_PYTHON);
  if (env.DSH_PYTHON) candidates.push(env.DSH_PYTHON);
  if (env.PYTHON) candidates.push(env.PYTHON);
  candidates.push(platform === 'win32' ? 'python' : 'python3', 'python');

  const args = asStringArray(config.args);

  return {
    /** Candidate interpreters, tried in order until one answers `--version`. */
    pythonCandidates: [...new Set(candidates)],
    /** MCP server entry point arguments. */
    args: args.length > 0 ? args : DEFAULT_ARGS,
    /** Working directory for the child process (the Python root when known). */
    cwd: typeof config.cwd === 'string' && config.cwd ? config.cwd : pythonRoot ?? pluginRoot,
    /** Import path entries prepended to PYTHONPATH. */
    pythonPath: [...new Set([...importPathFor(pythonRoot), ...asStringArray(config.pythonPath).map((p) => path.resolve(pluginRoot, p))])],
    /** Extra environment variables for the child process. */
    env: config.env && typeof config.env === 'object' ? Object.fromEntries(Object.entries(config.env).map(([k, v]) => [k, String(v)])) : {},
    /** Tool name prefix; keeps the plugin's tools grouped and collision-free. */
    prefix: typeof config.prefix === 'string' ? config.prefix : 'embodied_',
    /** Per-call budget for a tool invocation. */
    timeoutMs: positiveNumber(config.timeoutMs, 120_000),
    /** Budget for starting the server at session boot. */
    startupTimeoutMs: positiveNumber(config.startupTimeoutMs, 20_000),
    /** Restart the child after an unexpected exit (default: yes). */
    restart: config.restart !== false,
    /** Allow-list / deny-list of raw MCP tool names. */
    includeTools: asStringArray(config.includeTools),
    excludeTools: asStringArray(config.excludeTools),
    /** Truncate rendered tool output to keep transcripts bounded. */
    maxOutputChars: positiveNumber(config.maxOutputChars, 24_000),
    /** Inject a system-prompt section describing the toolbox and workflow. */
    promptPolicy: config.promptPolicy !== false,
    /** Directory holding the Python package, or null when it cannot be found. */
    pythonRoot,
  };
}

/** Human-readable diagnostics for the `embodied_self_check` tool. */
export function describeConfig(config, extra = {}) {
  return {
    pythonRoot: config.pythonRoot ?? null,
    pythonCandidates: config.pythonCandidates,
    args: config.args,
    cwd: config.cwd,
    pythonPath: config.pythonPath,
    prefix: config.prefix,
    timeoutMs: config.timeoutMs,
    restart: config.restart,
    ...extra,
  };
}
