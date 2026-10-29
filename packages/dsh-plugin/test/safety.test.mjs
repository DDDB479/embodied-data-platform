import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  canonicalPath,
  isInsideWorkspace,
  normalizePathArguments,
  prepareToolCall,
  writeTargets,
} from '../src/safety.js';
import { buildToolDefinitions } from '../src/tools.js';

function config(overrides = {}) {
  return {
    prefix: 'embodied_',
    maxOutputChars: 24_000,
    confirmOutsideWorkspace: true,
    workspaceRoot: null,
    ...overrides,
  };
}

function ctxFor({ workspaceRoot, mode = 'workspace-write', approval = null } = {}) {
  const calls = [];
  return {
    calls,
    get(name) {
      if (name === 'sandboxPolicy') return { resolve: () => ({ workspaceRoot, mode }) };
      if (name === 'approval') return approval ? { request: (request) => { calls.push(request); return approval(request); } } : undefined;
      return undefined;
    },
  };
}

function tool(name, readOnly) {
  return { name, annotations: { readOnlyHint: readOnly, destructiveHint: !readOnly, openWorldHint: false } };
}

test('workspace containment does not confuse sibling prefixes', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-scope-'));
  try {
    const inside = path.join(root, 'datasets');
    const sibling = `${root}-other`;
    fs.mkdirSync(inside, { recursive: true });
    fs.mkdirSync(sibling, { recursive: true });
    assert.equal(isInsideWorkspace(root, inside), true);
    assert.equal(isInsideWorkspace(root, root), true);
    assert.equal(isInsideWorkspace(root, path.join(sibling, 'x')), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(`${root}-other`, { recursive: true, force: true });
  }
});

test('relative path arguments resolve against the session workspace', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-paths-'));
  try {
    const input = normalizePathArguments('convert_dataset', { source: 'in', destination: 'out' }, root);
    assert.equal(input.source, path.join(root, 'in'));
    assert.equal(input.destination, path.join(root, 'out'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('async job paths resolve against the session workspace too', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-job-paths-'));
  try {
    const input = normalizePathArguments('submit_job', { tool: 'quality_report', arguments: { path: 'datasets/pick' } }, root);
    assert.equal(input.arguments.path, path.join(root, 'datasets', 'pick'));
    assert.equal(input.tool, 'quality_report');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('gate and catalog history paths resolve against the session workspace', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-new-paths-'));
  try {
    const gate = normalizePathArguments('quality_gate', { path: 'datasets/pick' }, root);
    assert.equal(gate.path, path.join(root, 'datasets', 'pick'));
    const history = normalizePathArguments('catalog_history', { db: 'catalog.db', path: 'datasets/pick' }, root);
    assert.equal(history.db, path.join(root, 'catalog.db'));
    assert.equal(history.path, path.join(root, 'datasets', 'pick'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('write targets are explicit and ignore an absent optional out', () => {
  assert.deepEqual(writeTargets('convert_dataset', { destination: 'out' }).length, 1);
  assert.deepEqual(writeTargets('labeling_tasks', {}), []);
});

test('in-workspace writes do not request approval', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-inside-'));
  try {
    const ctx = ctxFor({ workspaceRoot: root, approval: () => 'allowed-once' });
    const args = await prepareToolCall({ ctx, tool: tool('convert_dataset', false), args: { source: 'in', destination: 'out' }, exec: { agent: { session: {} } }, config: config() });
    assert.equal(args.destination, canonicalPath(path.join(root, 'out')));
    assert.equal(ctx.calls.length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('outside writes require a one-shot approval', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-outside-'));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-outside-target-'));
  try {
    const ctx = ctxFor({ workspaceRoot: root, approval: (request) => { assert.match(request.reason, /outside the session workspace/); return 'allowed-once'; } });
    const args = await prepareToolCall({ ctx, tool: tool('scaffold_project', false), args: { destination: outside }, exec: { agent: { session: {} }, signal: { throwIfAborted() {} } }, config: config() });
    assert.equal(args.destination, outside);
    assert.equal(ctx.calls.length, 1);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test('outside writes fail closed without approval support', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-noapproval-'));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-noapproval-target-'));
  try {
    const ctx = ctxFor({ workspaceRoot: root });
    await assert.rejects(
      prepareToolCall({ ctx, tool: tool('catalog_index', false), args: { root, db: path.join(outside, 'catalog.db') }, exec: { agent: { session: {} } }, config: config() }),
      { code: 'APPROVAL_UNAVAILABLE' },
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test('read-only mode blocks writes even inside the workspace', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-readonly-'));
  try {
    const ctx = ctxFor({ workspaceRoot: root, mode: 'read-only', approval: () => 'allowed-once' });
    await assert.rejects(
      prepareToolCall({ ctx, tool: tool('convert_dataset', false), args: { source: 'in', destination: 'out' }, exec: { agent: { session: {} } }, config: config() }),
      { code: 'POLICY_DENIED' },
    );
    assert.equal(ctx.calls.length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a denied approval never reaches the write', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-denied-'));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-denied-target-'));
  try {
    const ctx = ctxFor({ workspaceRoot: root, approval: () => 'denied' });
    await assert.rejects(
      prepareToolCall({ ctx, tool: tool('convert_dataset', false), args: { source: root, destination: outside }, exec: { agent: { session: {} } }, config: config() }),
      { code: 'APPROVAL_DENIED' },
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test('read-only tools stay concurrent and write tools stay serial', () => {
  const client = { callTool: async () => ({}) };
  const definitions = buildToolDefinitions(
    [tool('detect_dataset', true), tool('convert_dataset', false)],
    client,
    config(),
    console,
    null,
  );
  const byName = Object.fromEntries(definitions.map((item) => [item.name, item]));
  assert.equal(byName.embodied_detect_dataset.isConcurrencySafe(), true);
  assert.equal(byName.embodied_convert_dataset.isConcurrencySafe, undefined);
});
