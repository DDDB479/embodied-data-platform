/**
 * DSH plugin entry point: `dsh-embodied-data`.
 *
 * Cordis contract:
 *   export const name = '...'
 *   export function apply(ctx, config) { ... }
 *
 * The plugin owns a single long-lived `embodied_core.mcp_server` child process
 * and republishes its MCP tools as DSH tools. Every registration is paired with
 * a disposer inside `ctx.effect`, so disabling the row leaves no trace.
 */
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { resolveConfig } from './src/config.js';
import { McpClient } from './src/mcp.js';
import { buildToolDefinitions, selfCheckTool } from './src/tools.js';
import { buildPromptSection } from './src/prompt.js';

export const name = 'dsh-embodied-data';

const PLUGIN_ROOT = path.dirname(fileURLToPath(import.meta.url));

function log(ctx, level, message) {
  const logger = ctx?.logger;
  const sink = logger ? Reflect.get(logger, level) : undefined;
  if (typeof sink === 'function') {
    sink.call(logger, message);
    return;
  }
  if (level === 'warn' || level === 'error') console.warn(`[dsh-embodied-data] ${message}`);
}

function errorText(error) {
  return error && typeof error === 'object' && 'message' in error ? String(error.message) : String(error);
}

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {object} rawConfig cordis config row (`config:` in cordis.patch.yml)
 */
export function apply(ctx, rawConfig) {
  const config = resolveConfig(rawConfig, PLUGIN_ROOT);
  const client = new McpClient(config, ctx?.logger ?? console);
  /** Tool names actually published this session; read by the prompt section. */
  const state = { tools: [] };

  // The child must die with the plugin so a profile reload cannot leak Python.
  if (typeof ctx.effect === 'function') {
    ctx.effect(() => () => client.dispose(), 'dsh-embodied-data: mcp server');
  } else {
    process.once('exit', () => client.dispose());
  }

  ctx.inject(['tools'], (toolsCtx) => {
    const disposers = [];
    let disposed = false;

    if (typeof toolsCtx.effect === 'function') {
      toolsCtx.effect(() => () => {
        disposed = true;
        for (const dispose of disposers.splice(0)) {
          try {
            dispose();
          } catch (error) {
            log(ctx, 'warn', `unregister failed: ${errorText(error)}`);
          }
        }
        state.tools = [];
      }, 'dsh-embodied-data: tools');
    }

    const add = (definition) => {
      if (disposed) return;
      const dispose = toolsCtx.tools.register(definition);
      if (typeof dispose === 'function') disposers.push(dispose);
      state.tools = [...state.tools, definition.name];
    };

    // Register the diagnostic tool synchronously: even a broken Python install
    // must leave the agent a way to explain what is wrong.
    add(selfCheckTool(config, client));

    void (async () => {
      try {
        const tools = await client.listTools();
        for (const definition of buildToolDefinitions(tools, client, config, ctx?.logger ?? console, ctx)) add(definition);
        log(ctx, 'info', `dsh-embodied-data: ${state.tools.length} tool(s) ready (python root: ${config.pythonRoot ?? 'auto'})`);
      } catch (error) {
        log(ctx, 'warn', `dsh-embodied-data: MCP server unavailable — ${errorText(error)}. Only self-check is registered.`);
      }
    })();
  });

  if (config.promptPolicy) {
    ctx.inject(['systemPrompt'], (promptCtx) => {
      if (typeof promptCtx.systemPrompt?.section !== 'function') return;
      promptCtx.effect(
        () =>
          promptCtx.systemPrompt.section({
            name: 'embodied-data:toolbox',
            order: 3320,
            interpolate: false,
            text: () => buildPromptSection(config, state.tools),
          }),
        'dsh-embodied-data: prompt',
      );
    });
  }
}

export { isPythonRoot, locatePythonRoot, resolveConfig } from './src/config.js';
export { McpClient } from './src/mcp.js';
