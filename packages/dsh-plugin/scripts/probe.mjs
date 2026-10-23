#!/usr/bin/env node
/**
 * Standalone probe for the dsh-embodied-data plugin.
 *
 * It exercises the same code path DSH uses — `apply()` with a mock cordis
 * context — so a broken registration, a bad interpreter path or a stale Python
 * import is visible without restarting the GUI:
 *
 *   node scripts/probe.mjs [--json] [--python <path>] [--root <dir>] [--deps <path>]
 *
 * Exits 0 when the server starts, tools register and a live tool call returns
 * data; exits 1 with the reason otherwise.
 */
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { apply } from '../index.js';
import { resolveConfig } from '../src/config.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = path.dirname(HERE);

function parseArgs(argv) {
  const out = { json: false, python: '', root: '', deps: '', fixture: '' };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--json') out.json = true;
    else if (arg === '--python') out.python = argv[++i] ?? '';
    else if (arg === '--root') out.root = argv[++i] ?? '';
    else if (arg === '--deps') out.deps = argv[++i] ?? '';
    else if (arg === '--fixture') out.fixture = argv[++i] ?? '';
  }
  return out;
}

/** Minimal cordis stand-ins: just enough surface for `apply()`. */
function makeContext() {
  const tools = new Map();
  const sections = [];
  const disposers = [];
  const ctx = {
    logger: { info() {}, warn() {}, error() {} },
    inject(_deps, callback) {
      callback(ctx);
    },
    effect(fn) {
      const disposer = fn();
      if (typeof disposer === 'function') disposers.push(disposer);
      return disposer;
    },
    tools: {
      register(tool) {
        tools.set(tool.name, tool);
        return () => tools.delete(tool.name);
      },
    },
    systemPrompt: {
      section(options) {
        sections.push(options);
        return () => {};
      },
    },
  };
  return { ctx, tools, sections, dispose: () => disposers.splice(0).forEach((fn) => fn()) };
}

async function waitFor(predicate, { timeoutMs = 30_000, intervalMs = 100 } = {}) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const value = await predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return null;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const config = resolveConfig(
    {
      python: args.python,
      pythonRoot: args.root,
      pythonPath: args.deps ? [args.deps] : [],
    },
    PLUGIN_ROOT,
  );

  const { ctx, tools, sections, dispose } = makeContext();
  apply(ctx, { python: args.python, pythonRoot: args.root, pythonPath: args.deps ? [args.deps] : [] });

  // Discovery is asynchronous by design (the server may need a second to boot).
  const loaded = await waitFor(() => tools.size > 1 || null);
  const names = [...tools.keys()];
  const report = {
    pythonRoot: config.pythonRoot,
    pythonPath: config.pythonPath,
    tools: names,
    toolCount: names.length,
    promptSection: sections[0]?.name ?? null,
  };

  if (!loaded) {
    const check = tools.get('embodied_self_check');
    const detail = check ? await check.execute({}, {}) : null;
    report.ok = false;
    report.error = 'MCP server did not come up';
    report.selfCheck = detail?.data ?? null;
    emit(report, args.json);
    dispose();
    process.exit(1);
  }

  const fixture = args.fixture || path.join(PLUGIN_ROOT, '..', '..', '.devtools', 'fixtures', 'lerobot_v21');
  const detect = tools.get('embodied_detect_dataset');
  const summarize = tools.get('embodied_summarize_dataset');
  let call = null;
  try {
    if (detect) call = await detect.execute({ path: path.resolve(fixture) }, {});
  } catch (error) {
    report.ok = false;
    report.error = `tool call failed: ${error.message}`;
    emit(report, args.json);
    dispose();
    process.exit(1);
  }
  report.detect = call?.data ?? null;
  if (summarize) {
    try {
      const summary = await summarize.execute({ path: path.resolve(fixture) }, {});
      report.summary = summary?.data ?? null;
    } catch (error) {
      report.summaryError = error.message;
    }
  }
  report.ok = Boolean(report.detect);
  emit(report, args.json);
  dispose();
  process.exit(report.ok ? 0 : 1);
}

function emit(report, asJson) {
  if (asJson) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    return;
  }
  const line = (label, value) => process.stdout.write(`${label.padEnd(14)}${value}\n`);
  line('ok', report.ok ? 'yes' : 'no');
  if (report.error) line('error', report.error);
  line('python root', report.pythonRoot ?? '(not found)');
  line('import path', (report.pythonPath ?? []).join(path.delimiter) || '(empty)');
  line('tools', String(report.toolCount));
  if (report.tools.length) line('', report.tools.join(', '));
  if (report.promptSection) line('prompt', report.promptSection);
  if (report.detect) line('detect', JSON.stringify(report.detect));
  if (report.summary) line('summary', JSON.stringify(report.summary));
  if (report.selfCheck) line('self-check', JSON.stringify(report.selfCheck).slice(0, 600));
}

await main();
