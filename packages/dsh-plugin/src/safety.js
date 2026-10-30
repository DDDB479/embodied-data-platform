/**
 * Safety policy for tools forwarded from the MCP server.
 *
 * MCP annotations describe side effects, but DSH's tool contract only exposes
 * `isConcurrencySafe` and an approval service. This module turns the annotation
 * into those two concrete behaviours:
 *
 *   - read-only tools may run concurrently;
 *   - write tools always run serially;
 *   - a write destination outside the session workspace needs an explicit
 *     one-shot approval. If the host has no approval service, the call fails
 *     closed instead of silently writing outside the workspace.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** Arguments that contain filesystem paths, grouped by MCP tool name. */
export const PATH_ARGUMENTS = Object.freeze({
  detect_dataset: ['path'],
  summarize_dataset: ['path'],
  validate_dataset: ['path'],
  quality_report: ['path'],
  quality_gate: ['path'],
  segment_episodes: ['path'],
  labeling_tasks: ['path', 'out'],
  convert_dataset: ['source', 'destination'],
  scaffold_project: ['destination'],
  catalog_index: ['root', 'db'],
  catalog_search: ['db'],
  catalog_stats: ['db'],
  catalog_history: ['db', 'path'],
  inspect_media: ['path'],
  decode_samples: ['path'],
  apply_annotations: ['manifest', 'previous', 'out'],
  storage_pull: ['destination'],
  plan_windows: ['path'],
});

/** Path arguments that the tool may create or modify. */
export const WRITE_TARGETS = Object.freeze({
  labeling_tasks: ['out'],
  convert_dataset: ['destination'],
  scaffold_project: ['destination'],
  catalog_index: ['db'],
  apply_annotations: ['out'],
  storage_pull: ['destination'],
});

export const WRITE_TOOLS = new Set(Object.keys(WRITE_TARGETS));

function expandHome(value) {
  if (value === '~') return os.homedir();
  if (value.startsWith(`~${path.sep}`) || value.startsWith('~/') || value.startsWith('~\\')) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

/** Resolve symlinks for existing path components while tolerating new files. */
export function canonicalPath(value) {
  let current = path.resolve(expandHome(String(value)));
  const suffix = [];
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) break;
    suffix.unshift(path.basename(current));
    current = parent;
  }
  try {
    current = fs.realpathSync.native(current);
  } catch {
    // Keep the lexical path when the filesystem cannot resolve it.
  }
  return path.join(current, ...suffix);
}

function comparable(value) {
  return process.platform === 'win32' ? value.toLowerCase() : value;
}

/** True when `target` is the workspace root or a descendant of it. */
export function isInsideWorkspace(workspaceRoot, target) {
  if (!workspaceRoot || !target) return false;
  const root = comparable(canonicalPath(workspaceRoot));
  const candidate = comparable(canonicalPath(target));
  const relative = path.relative(root, candidate);
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

export function isReadOnlyTool(tool) {
  return tool?.annotations?.readOnlyHint === true;
}

export function isWriteTool(tool) {
  if (tool?.annotations?.readOnlyHint === false) return true;
  // Older servers may omit annotations; keep the known write surface guarded.
  return tool?.annotations?.readOnlyHint === undefined && WRITE_TOOLS.has(tool?.name);
}

/** Return absolute write destinations from the raw tool arguments. */
export function writeTargets(toolName, args) {
  const fields = WRITE_TARGETS[toolName] ?? [];
  return fields
    .map((field) => args?.[field])
    .filter((value) => typeof value === 'string' && value.trim() !== '')
    .map((value) => canonicalPath(value));
}

/**
 * Resolve the workspace root from DSH's sandbox policy, then from the plugin
 * config, then from the session cwd. The policy is the authoritative source
 * when the host exposes it.
 */
export function resolveWorkspace(ctx, exec, config) {
  const session = exec?.agent?.session;
  let policy = null;
  try {
    policy = ctx?.get?.('sandboxPolicy')?.resolve?.({ session }) ?? null;
  } catch {
    policy = null;
  }
  if (policy?.workspaceRoot) {
    return { root: canonicalPath(policy.workspaceRoot), mode: policy.mode ?? null };
  }
  if (config?.workspaceRoot) return { root: canonicalPath(config.workspaceRoot), mode: null };
  const cwd = session?.cwd ?? exec?.cwd ?? process.cwd();
  return { root: canonicalPath(cwd), mode: null };
}

/**
 * Make relative path arguments deterministic: they are relative to the session
 * workspace, not to the Python server process's install directory.
 */
export function normalizePathArguments(toolName, args, workspaceRoot) {
  const fields = PATH_ARGUMENTS[toolName] ?? [];
  if (toolName === 'submit_job' && args?.arguments && typeof args.arguments === 'object' && !Array.isArray(args.arguments)) {
    const nested = { ...args.arguments };
    if (typeof nested.path === 'string' && nested.path.trim() !== '') {
      nested.path = path.resolve(workspaceRoot, expandHome(nested.path));
    }
    return { ...args, arguments: nested };
  }
  if (fields.length === 0 || !args || typeof args !== 'object') return args ?? {};
  const normalized = { ...args };
  for (const field of fields) {
    const value = normalized[field];
    if (typeof value === 'string' && value.trim() !== '') normalized[field] = path.resolve(workspaceRoot, expandHome(value));
  }
  return normalized;
}

function safetyError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

/**
 * Normalize paths and enforce the write policy before forwarding a call to the
 * MCP server. Returns the arguments to send.
 */
export async function prepareToolCall({ ctx, tool, args, exec, config, logger = null }) {
  const workspace = resolveWorkspace(ctx, exec, config);
  const input = normalizePathArguments(tool.name, args ?? {}, workspace.root);
  if (!isWriteTool(tool)) return input;

  const targets = writeTargets(tool.name, input);
  if (targets.length === 0) return input;

  if (workspace.mode === 'read-only') {
    throw safetyError('POLICY_DENIED', `${tool.name} cannot write while DSH is in read-only mode.`);
  }

  const outside = targets.filter((target) => !isInsideWorkspace(workspace.root, target));
  if (outside.length === 0) return input;

  const destination = outside[0];
  if (!config?.confirmOutsideWorkspace) return input;

  const approval = ctx?.get?.('approval');
  if (!approval?.request || !exec?.agent) {
    throw safetyError(
      'APPROVAL_UNAVAILABLE',
      `${tool.name} would write outside the workspace (${destination}). ` +
        `Move the destination under ${workspace.root}, or configure an approval-capable host.`,
    );
  }

  logger?.info?.(`embodied-data: requesting approval for ${tool.name} -> ${destination}`);
  const outcome = await approval.request({
    agent: exec.agent,
    callId: exec.callId,
    toolName: `${config?.prefix ?? 'embodied_'}${tool.name}`,
    reason: `${tool.name} writes outside the session workspace: ${destination}. Allow this one write?`,
    signal: exec.signal,
  });
  if (outcome !== 'allowed-once') {
    throw safetyError('APPROVAL_DENIED', `no write was performed for ${destination}.`);
  }
  exec.signal?.throwIfAborted?.();
  return input;
}
